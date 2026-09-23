//! Deterministic, ignore-aware source discovery.
//!
//! This module deliberately performs a sequential walk and sorts candidate
//! paths before opening files. That makes source IDs, total-byte-limit
//! behavior, and later ranking tie-breaks reproducible across runs.

use std::{
    ffi::OsStr,
    fs,
    path::{Component, Path, PathBuf},
};

use globset::{Glob, GlobSet, GlobSetBuilder};
use ignore::WalkBuilder;
use thiserror::Error;

use crate::{
    source::{Source, SourceId, SourceReadError},
    SupergrepError,
};

pub const DEFAULT_MAX_FILE_SIZE: u64 = 2 * 1024 * 1024;
pub const DEFAULT_MAX_TOTAL_BYTES: u64 = 64 * 1024 * 1024;
pub const DEFAULT_MAX_CHUNKS: usize = 50_000;

/// User-selected discovery and resource-limit policy.
#[derive(Debug, Clone)]
pub struct DiscoveryOptions {
    pub root: PathBuf,
    /// Include dot-prefixed filesystem entries. This does not include `.git`.
    pub include_hidden: bool,
    /// Respect `.ignore`, Git ignore, global Git ignore, and Git exclude files.
    pub respect_ignore: bool,
    /// Optional path filters, matched against a path relative to `root`.
    pub globs: Vec<String>,
    pub max_file_size: u64,
    pub max_total_bytes: u64,
    /// Shared engine limit carried here so the CLI has one resource-policy
    /// source. Chunk construction applies it and reports truncation itself.
    pub max_chunks: usize,
}

impl Default for DiscoveryOptions {
    fn default() -> Self {
        Self {
            root: PathBuf::from("."),
            include_hidden: false,
            respect_ignore: true,
            globs: Vec::new(),
            max_file_size: DEFAULT_MAX_FILE_SIZE,
            max_total_bytes: DEFAULT_MAX_TOTAL_BYTES,
            max_chunks: DEFAULT_MAX_CHUNKS,
        }
    }
}

impl DiscoveryOptions {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<(), DiscoveryError> {
        if self.max_file_size == 0 {
            return Err(DiscoveryError::InvalidOption(
                "max_file_size must be at least one byte".into(),
            ));
        }
        if self.max_total_bytes == 0 {
            return Err(DiscoveryError::InvalidOption(
                "max_total_bytes must be at least one byte".into(),
            ));
        }
        if self.max_chunks == 0 {
            return Err(DiscoveryError::InvalidOption(
                "max_chunks must be at least one".into(),
            ));
        }
        Ok(())
    }
}

/// Why a path was not usable, or why a scan is incomplete. Policy exclusions
/// never set `partial`; read/traversal/limit failures always do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DiscoveryDiagnosticKind {
    GitMetadata,
    Symlink,
    NotRegularFile,
    GlobFiltered,
    FileTooLarge,
    TotalByteLimit,
    NulByte,
    InvalidUtf8,
    ChangedDuringRead,
    ReadFailure,
    TraversalFailure,
    IgnoreRuleFailure,
}

impl DiscoveryDiagnosticKind {
    pub const fn is_policy_exclusion(self) -> bool {
        matches!(
            self,
            Self::GitMetadata
                | Self::Symlink
                | Self::NotRegularFile
                | Self::GlobFiltered
                | Self::NulByte
                | Self::InvalidUtf8
        )
    }
}

/// A path-specific, structured diagnostic. Paths intentionally remain `PathBuf`
/// values instead of lossy strings; output formatting decides how to represent
/// non-UTF-8 Unix names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveryDiagnostic {
    pub kind: DiscoveryDiagnosticKind,
    pub path: Option<PathBuf>,
    pub message: String,
}

impl DiscoveryDiagnostic {
    fn path(kind: DiscoveryDiagnosticKind, path: &Path, message: impl Into<String>) -> Self {
        Self {
            kind,
            path: Some(path.to_path_buf()),
            message: message.into(),
        }
    }

    fn no_path(kind: DiscoveryDiagnosticKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            path: None,
            message: message.into(),
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct DiscoveryStats {
    /// Entries emitted by the walker. This deliberately excludes paths pruned
    /// by ignore rules, so it is not presented as a count of all descendants.
    pub entries_seen: u64,
    pub files_considered: u64,
    pub files_read: u64,
    pub files_accepted: u64,
    /// Physical bytes successfully read from the filesystem, including bytes
    /// from files later rejected for NUL/UTF-8/mutation validation.
    pub bytes_read: u64,
    pub policy_exclusions_observed: u64,
    pub failures: u64,
}

/// Sources and diagnostics from one bounded scan.
#[derive(Debug)]
pub struct DiscoveryReport {
    pub root: PathBuf,
    pub sources: Vec<Source>,
    pub diagnostics: Vec<DiscoveryDiagnostic>,
    pub stats: DiscoveryStats,
    /// True only when the walker and every attempted source read completed
    /// without an operational failure or resource limit.
    pub scan_complete: bool,
    /// A convenient output-level status. Policy filtering alone is not partial.
    pub partial: bool,
}

impl DiscoveryReport {
    fn empty(root: PathBuf) -> Self {
        Self {
            root,
            sources: Vec::new(),
            diagnostics: Vec::new(),
            stats: DiscoveryStats::default(),
            scan_complete: true,
            partial: false,
        }
    }

    fn policy(&mut self, diagnostic: DiscoveryDiagnostic) {
        debug_assert!(diagnostic.kind.is_policy_exclusion());
        self.stats.policy_exclusions_observed += 1;
        self.diagnostics.push(diagnostic);
    }

    fn failure(&mut self, diagnostic: DiscoveryDiagnostic) {
        debug_assert!(!diagnostic.kind.is_policy_exclusion());
        self.stats.failures += 1;
        self.scan_complete = false;
        self.partial = true;
        self.diagnostics.push(diagnostic);
    }
}

#[derive(Debug, Error)]
pub enum DiscoveryError {
    #[error("search root does not exist: {}", .0.display())]
    MissingRoot(PathBuf),

    #[error("search root is a symbolic link, which is unsupported: {}", .0.display())]
    SymlinkRoot(PathBuf),

    #[error("search root is neither a directory nor a regular file: {}", .0.display())]
    UnsupportedRoot(PathBuf),

    #[error("could not inspect search root {}: {source}", path.display())]
    RootMetadata {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("invalid discovery option: {0}")]
    InvalidOption(String),

    #[error("invalid --glob pattern {pattern:?}: {message}")]
    InvalidGlob { pattern: String, message: String },
}

impl From<DiscoveryError> for SupergrepError {
    fn from(value: DiscoveryError) -> Self {
        Self::Input(value.to_string())
    }
}

/// Discovers and snapshots regular UTF-8 files under one root.
///
/// A missing or unsupported root is an input error. Problems encountered after
/// a valid directory walk starts are returned as diagnostics with any valid
/// sibling sources preserved.
pub fn discover(options: &DiscoveryOptions) -> Result<DiscoveryReport, DiscoveryError> {
    options.validate()?;
    let globs = compile_globs(&options.globs)?;
    let root_metadata = fs::symlink_metadata(&options.root).map_err(|source| {
        if source.kind() == std::io::ErrorKind::NotFound {
            DiscoveryError::MissingRoot(options.root.clone())
        } else {
            DiscoveryError::RootMetadata {
                path: options.root.clone(),
                source,
            }
        }
    })?;
    if root_metadata.file_type().is_symlink() {
        return Err(DiscoveryError::SymlinkRoot(options.root.clone()));
    }
    // `symlink_metadata(root)` only inspects the final component. Reject a
    // root reached through a symlinked ancestor too, otherwise `link/subdir`
    // would silently traverse outside the user-visible tree despite the
    // no-follow-links contract.
    if root_has_symlink_component(&options.root).map_err(|source| DiscoveryError::RootMetadata {
        path: options.root.clone(),
        source,
    })? {
        return Err(DiscoveryError::SymlinkRoot(options.root.clone()));
    }
    if !root_metadata.file_type().is_dir() && !root_metadata.file_type().is_file() {
        return Err(DiscoveryError::UnsupportedRoot(options.root.clone()));
    }

    let working_directory =
        std::env::current_dir().map_err(|source| DiscoveryError::RootMetadata {
            path: options.root.clone(),
            source,
        })?;
    let absolute_root = if options.root.is_absolute() {
        options.root.clone()
    } else {
        working_directory.join(&options.root)
    };
    // Symlink components were rejected above, so lexical normalization now
    // gives the actual non-link ancestry for the unconditional `.git` policy.
    // It avoids treating `project/.git/../visible.txt` as Git metadata merely
    // because of how the caller spelled a root outside `.git`.
    let normalized_absolute_root = normalize_absolute_path(&absolute_root);

    let mut report = DiscoveryReport::empty(options.root.clone());
    if path_has_git_component(&normalized_absolute_root) {
        report.policy(DiscoveryDiagnostic::path(
            DiscoveryDiagnosticKind::GitMetadata,
            &options.root,
            "the .git directory is always excluded",
        ));
        return Ok(report);
    }

    if root_metadata.file_type().is_file() {
        report.stats.entries_seen += 1;
        if glob_matches(globs.as_ref(), &options.root, &options.root) {
            read_candidate(&mut report, options, &options.root);
        } else {
            report.policy(DiscoveryDiagnostic::path(
                DiscoveryDiagnosticKind::GlobFiltered,
                &options.root,
                "path did not match the requested glob filters",
            ));
        }
        return Ok(report);
    }

    let mut candidates = Vec::new();
    let mut builder = WalkBuilder::new(&options.root);
    let walk_working_directory = working_directory.clone();
    builder
        .hidden(!options.include_hidden)
        .parents(options.respect_ignore)
        .ignore(options.respect_ignore)
        .git_ignore(options.respect_ignore)
        .git_global(options.respect_ignore)
        .git_exclude(options.respect_ignore)
        // Keep ignore's normal Git-boundary behavior. `.ignore` remains useful
        // outside a Git worktree, while Git-specific rules follow Git bounds.
        .require_git(true)
        .follow_links(false)
        .skip_stdout(true)
        .sort_by_file_path(|left, right| left.cmp(right))
        .filter_entry(move |entry| {
            let path = entry.path();
            let absolute = if path.is_absolute() {
                path.to_path_buf()
            } else {
                walk_working_directory.join(path)
            };
            !path_has_git_component(&normalize_absolute_path(&absolute))
        });

    for item in builder.build() {
        match item {
            Err(error) => report.failure(DiscoveryDiagnostic::no_path(
                DiscoveryDiagnosticKind::TraversalFailure,
                format!("could not traverse part of the search tree: {error}"),
            )),
            Ok(entry) => {
                report.stats.entries_seen += 1;
                if let Some(error) = entry.error() {
                    report.failure(DiscoveryDiagnostic::path(
                        DiscoveryDiagnosticKind::IgnoreRuleFailure,
                        entry.path(),
                        format!("could not fully apply ignore rules: {error}"),
                    ));
                }
                if entry.path_is_symlink() {
                    report.policy(DiscoveryDiagnostic::path(
                        DiscoveryDiagnosticKind::Symlink,
                        entry.path(),
                        "symbolic links are not followed or read",
                    ));
                    continue;
                }
                let Some(file_type) = entry.file_type() else {
                    report.policy(DiscoveryDiagnostic::path(
                        DiscoveryDiagnosticKind::NotRegularFile,
                        entry.path(),
                        "path has no regular-file type",
                    ));
                    continue;
                };
                if !file_type.is_file() {
                    continue;
                }
                if glob_matches(globs.as_ref(), &options.root, entry.path()) {
                    candidates.push(entry.into_path());
                } else {
                    report.policy(DiscoveryDiagnostic::path(
                        DiscoveryDiagnosticKind::GlobFiltered,
                        entry.path(),
                        "path did not match the requested glob filters",
                    ));
                }
            }
        }
    }

    // `WalkBuilder` sorts siblings, while a final sort makes the public
    // contract global and independent of traversal implementation details.
    candidates.sort();
    candidates.dedup();
    let mut candidates = candidates.into_iter().peekable();
    while let Some(path) = candidates.next() {
        match read_candidate(&mut report, options, &path) {
            CandidateControl::Continue => {}
            CandidateControl::StopAtBudget => {
                if let Some(next) = candidates.peek() {
                    report.failure(DiscoveryDiagnostic::path(
                        DiscoveryDiagnosticKind::TotalByteLimit,
                        next,
                        "the total read budget was exhausted before this path could be read",
                    ));
                }
                break;
            }
            CandidateControl::StopAfterDiagnostic => break,
        }
    }
    Ok(report)
}

enum CandidateControl {
    Continue,
    /// A valid source consumed the final byte of the budget. The caller checks
    /// whether there is a later candidate before declaring the scan partial.
    StopAtBudget,
    /// A diagnostic already explains why no later candidates can be read.
    StopAfterDiagnostic,
}

fn read_candidate(
    report: &mut DiscoveryReport,
    options: &DiscoveryOptions,
    path: &Path,
) -> CandidateControl {
    report.stats.files_considered += 1;
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) => {
            report.failure(DiscoveryDiagnostic::path(
                DiscoveryDiagnosticKind::ReadFailure,
                path,
                format!("could not inspect candidate before reading: {error}"),
            ));
            return CandidateControl::Continue;
        }
    };
    if metadata.file_type().is_symlink() {
        report.policy(DiscoveryDiagnostic::path(
            DiscoveryDiagnosticKind::Symlink,
            path,
            "symbolic links are not read",
        ));
        return CandidateControl::Continue;
    }
    if !metadata.file_type().is_file() {
        report.policy(DiscoveryDiagnostic::path(
            DiscoveryDiagnosticKind::NotRegularFile,
            path,
            "path is no longer a regular file",
        ));
        return CandidateControl::Continue;
    }
    if metadata.len() > options.max_file_size {
        report.failure(DiscoveryDiagnostic::path(
            DiscoveryDiagnosticKind::FileTooLarge,
            path,
            format!(
                "file metadata reports {} bytes, above the {}-byte per-file limit",
                metadata.len(),
                options.max_file_size
            ),
        ));
        return CandidateControl::Continue;
    }
    let remaining = options
        .max_total_bytes
        .saturating_sub(report.stats.bytes_read);
    if metadata.len() > remaining {
        report.failure(DiscoveryDiagnostic::path(
            DiscoveryDiagnosticKind::TotalByteLimit,
            path,
            format!(
                "{} bytes remain in the total read budget; this file needs {} bytes",
                remaining,
                metadata.len()
            ),
        ));
        // Sorting defines which prefix is scanned. Do not pretend to know how
        // many descendants remain unvisited after a hard budget stop.
        return CandidateControl::StopAfterDiagnostic;
    }

    report.stats.files_read += 1;
    let read_limit = options.max_file_size.min(remaining);
    match Source::read(SourceId::new(report.sources.len()), path, read_limit) {
        Ok(source) => {
            report.stats.bytes_read += source.len_bytes() as u64;
            report.stats.files_accepted += 1;
            report.sources.push(source);
            if report.stats.bytes_read == options.max_total_bytes {
                CandidateControl::StopAtBudget
            } else {
                CandidateControl::Continue
            }
        }
        Err(error) => {
            report.stats.bytes_read += error.bytes_read().min(remaining);
            let diagnostic =
                source_error_diagnostic(path, &error, read_limit, options.max_file_size);
            let policy_exclusion = diagnostic.kind.is_policy_exclusion();
            if policy_exclusion {
                report.policy(diagnostic);
            } else {
                report.failure(diagnostic);
            }
            if report.stats.bytes_read == options.max_total_bytes {
                if policy_exclusion {
                    // The invalid/binary file itself is a normal policy
                    // exclusion. Only a remaining unread candidate turns the
                    // exhausted budget into an incomplete scan.
                    CandidateControl::StopAtBudget
                } else {
                    CandidateControl::StopAfterDiagnostic
                }
            } else {
                CandidateControl::Continue
            }
        }
    }
}

fn source_error_diagnostic(
    path: &Path,
    error: &SourceReadError,
    read_limit: u64,
    per_file_limit: u64,
) -> DiscoveryDiagnostic {
    let kind = match error {
        // `read_candidate` already observed a regular non-symlink path. If it
        // becomes either of these before `Source::read` opens it, this is an
        // operational mutation, not an ordinary policy exclusion.
        SourceReadError::Symlink { .. } | SourceReadError::NotRegular { .. } => {
            DiscoveryDiagnosticKind::ChangedDuringRead
        }
        SourceReadError::TooLarge { .. } if read_limit < per_file_limit => {
            DiscoveryDiagnosticKind::TotalByteLimit
        }
        SourceReadError::TooLarge { .. } => DiscoveryDiagnosticKind::FileTooLarge,
        SourceReadError::Nul { .. } => DiscoveryDiagnosticKind::NulByte,
        SourceReadError::InvalidUtf8 { .. } => DiscoveryDiagnosticKind::InvalidUtf8,
        SourceReadError::Changed { .. } => DiscoveryDiagnosticKind::ChangedDuringRead,
        SourceReadError::Io { .. } => DiscoveryDiagnosticKind::ReadFailure,
    };
    DiscoveryDiagnostic::path(kind, path, error.to_string())
}

fn compile_globs(patterns: &[String]) -> Result<Option<GlobSet>, DiscoveryError> {
    if patterns.is_empty() {
        return Ok(None);
    }
    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        let glob = Glob::new(pattern).map_err(|error| DiscoveryError::InvalidGlob {
            pattern: pattern.clone(),
            message: error.to_string(),
        })?;
        builder.add(glob);
    }
    builder.build().map(Some).map_err(|error| {
        DiscoveryError::InvalidOption(format!("could not compile glob set: {error}"))
    })
}

fn glob_matches(globs: Option<&GlobSet>, root: &Path, path: &Path) -> bool {
    let Some(globs) = globs else {
        return true;
    };
    let relative = path.strip_prefix(root).unwrap_or(path);
    if relative.as_os_str().is_empty() {
        // A file root has no non-empty relative path. Match its basename so
        // `--glob '*.rs' path/to/file.rs` behaves like a normal file filter.
        path.file_name()
            .is_some_and(|name| globs.is_match(Path::new(name)))
    } else {
        globs.is_match(relative)
    }
}

fn path_has_git_component(path: &Path) -> bool {
    path.components().any(|component| match component {
        Component::Normal(name) => name == OsStr::new(".git"),
        _ => false,
    })
}

fn root_has_symlink_component(path: &Path) -> std::io::Result<bool> {
    let mut current = if path.is_absolute() {
        PathBuf::new()
    } else {
        std::env::current_dir()?
    };
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => current.push(prefix.as_os_str()),
            Component::RootDir => current.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                current.pop();
            }
            Component::Normal(name) => {
                current.push(name);
                if fs::symlink_metadata(&current)?.file_type().is_symlink() {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

/// Normalizes `.` and `..` lexically without resolving symlinks. Callers use
/// it only after checking every supplied component for symlink identity.
fn normalize_absolute_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                let _ = normalized.pop();
            }
            Component::Normal(name) => normalized.push(name),
        }
    }
    normalized
}

#[cfg(test)]
mod tests {
    use std::{io::Write, sync::Mutex};

    use tempfile::tempdir;

    use super::*;

    static CWD_LOCK: Mutex<()> = Mutex::new(());

    struct RestoreCwd(PathBuf);

    impl Drop for RestoreCwd {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.0);
        }
    }

    fn write(path: &Path, contents: &[u8]) {
        let mut file = fs::File::create(path).unwrap();
        file.write_all(contents).unwrap();
    }

    #[test]
    fn direct_file_overrides_hidden_and_ignore_filters() {
        let dir = tempdir().unwrap();
        write(&dir.path().join(".ignore"), b".hidden.txt\n");
        write(
            &dir.path().join(".hidden.txt"),
            b"visible by explicit root\n",
        );
        let report = discover(&DiscoveryOptions::new(dir.path().join(".hidden.txt"))).unwrap();
        assert_eq!(report.sources.len(), 1);
        assert_eq!(report.sources[0].text(), "visible by explicit root\n");
    }

    #[test]
    fn nul_and_utf8_are_policy_filters_not_partial_failures() {
        let dir = tempdir().unwrap();
        write(&dir.path().join("nul"), b"ok\0no");
        write(&dir.path().join("invalid"), &[0xff]);
        let report = discover(&DiscoveryOptions::new(dir.path())).unwrap();
        assert!(!report.partial);
        assert!(report.scan_complete);
        assert!(report.sources.is_empty());
        assert!(report
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.kind == DiscoveryDiagnosticKind::NulByte));
        assert!(report
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.kind == DiscoveryDiagnosticKind::InvalidUtf8));
    }

    #[test]
    fn dot_root_inside_git_metadata_is_always_excluded() {
        let _lock = CWD_LOCK.lock().unwrap();
        let directory = tempdir().unwrap();
        let git_directory = directory.path().join(".git");
        fs::create_dir(&git_directory).unwrap();
        write(&git_directory.join("config"), b"not searchable\n");

        let restore = RestoreCwd(std::env::current_dir().unwrap());
        std::env::set_current_dir(&git_directory).unwrap();
        let report = discover(&DiscoveryOptions::new(".")).unwrap();
        drop(restore);

        assert!(report.sources.is_empty());
        assert!(report
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.kind == DiscoveryDiagnosticKind::GitMetadata));
    }
}
