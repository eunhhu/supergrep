//! Presentation of search evidence without rereading the filesystem.
//!
//! The search layer intentionally returns coordinates and raw scores instead
//! of output strings.  This module turns those coordinates back into text from
//! the immutable [`Source`] snapshots that were searched.  In particular, it
//! never opens a result path while rendering, so a changed file cannot make a
//! result claim evidence that the engine did not score.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde::Serialize;

use crate::{
    search::{CandidateSelection, RankedEntry, SearchMode, SearchReport},
    source::{Source, SourceId},
    Result, SupergrepError,
};

/// Version of the machine-readable search-result contract.
pub const OUTPUT_SCHEMA_VERSION: u32 = 1;

/// The meaning of a model score.  It is deliberately not a probability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelScoreKind {
    /// A direct, uncalibrated ONNX cross-encoder output.
    RawLogit,
}

impl ModelScoreKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::RawLogit => "raw_logit",
        }
    }
}

/// Optional reproducibility metadata for a loaded local model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelMetadata {
    pub id: String,
    pub revision: String,
    pub score_kind: ModelScoreKind,
}

/// Timings measured by the caller.  `None` means the phase was not measured,
/// rather than a misleading zero duration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PresentationTiming {
    pub discovery_ms: Option<u64>,
    pub chunking_ms: Option<u64>,
    pub model_load_ms: Option<u64>,
    pub tokenization_ms: Option<u64>,
    pub inference_ms: Option<u64>,
    pub search_ms: Option<u64>,
    pub total_ms: Option<u64>,
}

/// Neutral, serializable accounting data supplied by the discovery/CLI layer.
///
/// The count fields in [`SearchReport`] are copied by
/// [`PresentationStats::from_report`].  The caller fills discovery facts and
/// timings so this module stays independent of filesystem traversal policy.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PresentationStats {
    pub entries_seen: Option<u64>,
    pub files_considered: Option<u64>,
    pub files_read: Option<u64>,
    pub files_accepted: Option<u64>,
    pub bytes_read: Option<u64>,
    pub policy_exclusions_observed: Option<u64>,
    pub failures: Option<u64>,
    pub scan_complete: bool,
    pub partial: bool,
    pub lexical_evidence: bool,
    /// `None` is meaningful for lexical-only search because no model coverage
    /// claim is made there.
    pub scoring_complete: Option<bool>,
    pub chunks_total: usize,
    pub chunks_selected: usize,
    pub chunks_evaluated: usize,
    pub results_returned: usize,
    pub deduplicated_count: usize,
    pub candidate_selection: Option<CandidateSelection>,
    pub timing: PresentationTiming,
}

impl PresentationStats {
    /// Starts with all search-derived status and count fields populated.
    pub fn from_report(report: &SearchReport) -> Self {
        Self {
            scan_complete: true,
            partial: false,
            lexical_evidence: report.lexical_evidence,
            scoring_complete: report.scoring_complete,
            chunks_total: report.all_chunk_indexes.len(),
            chunks_selected: report.selected_chunk_indexes.len(),
            chunks_evaluated: report.evaluated_chunk_indexes.len(),
            results_returned: report.results.len(),
            deduplicated_count: report.deduplicated_count,
            candidate_selection: report.candidate_selection,
            ..Self::default()
        }
    }
}

/// A caller-owned diagnostic.  The `kind` is intentionally a stable string
/// so discovery, chunking, model setup, and CLI validation can report their
/// own domains without output depending on each domain's Rust enum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PresentationDiagnostic {
    pub kind: String,
    pub path: Option<PathBuf>,
    pub message: String,
}

impl PresentationDiagnostic {
    pub fn new(kind: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            path: None,
            message: message.into(),
        }
    }

    pub fn with_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.path = Some(path.into());
        self
    }
}

/// Everything needed to render one command result.  `sources` must be the
/// exact discovery snapshots passed to the search engine, not paths reopened
/// after scoring.
#[derive(Debug)]
pub struct PresentationInput<'a> {
    pub query: &'a str,
    pub root: &'a Path,
    pub report: &'a SearchReport,
    pub sources: &'a [Source],
    pub model: Option<&'a ModelMetadata>,
    pub stats: PresentationStats,
    pub diagnostics: Vec<PresentationDiagnostic>,
}

impl<'a> PresentationInput<'a> {
    pub fn new(
        query: &'a str,
        root: &'a Path,
        report: &'a SearchReport,
        sources: &'a [Source],
    ) -> Self {
        Self {
            query,
            root,
            report,
            sources,
            model: None,
            stats: PresentationStats::from_report(report),
            diagnostics: Vec::new(),
        }
    }
}

/// Renders exactly one compact JSON object, with no newline or logging around
/// it.  Callers can write this string directly to stdout for `--json`.
pub fn render_json(input: &PresentationInput<'_>) -> Result<String> {
    let sources = source_map(input.sources);
    let results = input
        .report
        .results
        .iter()
        .map(|entry| json_result(entry, input.root, &sources))
        .collect::<Result<Vec<_>>>()?;
    let diagnostics = rendered_diagnostics(input);
    serde_json::to_string(&JsonDocument {
        schema_version: OUTPUT_SCHEMA_VERSION,
        query: input.query,
        root: path_for_output(input.root, input.root),
        mode: mode_name(input.report.mode),
        model: input.model.map(JsonModel::from),
        results,
        stats: JsonStats::from(&input.stats),
        diagnostics,
    })
    .map_err(|error| SupergrepError::Internal(format!("could not serialize JSON output: {error}")))
}

/// Renders a stable, terminal-safe text view.  Every C0/C1 control character
/// in user-controlled paths, query text, snippets, and diagnostics is escaped
/// before it reaches stdout.
pub fn render_human(input: &PresentationInput<'_>) -> String {
    let sources = source_map(input.sources);
    let mut output = String::new();
    output.push_str("query: ");
    output.push_str(&escape_controls(input.query));
    output.push('\n');
    output.push_str("root: ");
    output.push_str(&escape_controls(
        &path_for_output(input.root, input.root).path_display,
    ));
    output.push('\n');
    output.push_str("mode: ");
    output.push_str(mode_name(input.report.mode));
    output.push('\n');
    match input.model {
        Some(model) => {
            output.push_str("model: ");
            output.push_str(&escape_controls(&model.id));
            output.push_str(" revision=");
            output.push_str(&escape_controls(&model.revision));
            output.push_str(" score_kind=");
            output.push_str(model.score_kind.as_str());
            output.push('\n');
        }
        None => {
            output.push_str("model: none (lexical scores are BM25, not relevance probabilities)\n")
        }
    }
    output.push_str("status: scan_complete=");
    output.push_str(&input.stats.scan_complete.to_string());
    output.push_str(" scoring_complete=");
    output.push_str(&optional_bool(input.stats.scoring_complete));
    output.push_str(" partial=");
    output.push_str(&input.stats.partial.to_string());
    output.push_str(" lexical_evidence=");
    output.push_str(&input.stats.lexical_evidence.to_string());
    output.push('\n');
    output.push_str(&format!(
        "stats: chunks_total={} selected={} evaluated={} returned={} deduplicated={}\n",
        input.stats.chunks_total,
        input.stats.chunks_selected,
        input.stats.chunks_evaluated,
        input.stats.results_returned,
        input.stats.deduplicated_count,
    ));

    let diagnostics = rendered_diagnostics(input);
    if !diagnostics.is_empty() {
        output.push_str("warnings:\n");
        for diagnostic in diagnostics {
            output.push_str("  [");
            output.push_str(&escape_controls(&diagnostic.kind));
            output.push_str("] ");
            if !diagnostic.path.path_display.is_empty() {
                output.push_str(&escape_controls(&diagnostic.path.path_display));
                output.push_str(": ");
            }
            output.push_str(&escape_controls(&diagnostic.message));
            output.push('\n');
        }
    }

    output.push_str("results:\n");
    for (rank, entry) in input.report.results.iter().enumerate() {
        match human_result(entry, input.root, &sources) {
            Ok(result) => {
                output.push_str(&format!(
                    "  {}. {} bytes={}..{} lines={}..{} kind={} score_kind={} score={}\n",
                    rank + 1,
                    escape_controls(&result.path.path_display),
                    result.start_byte,
                    result.end_byte,
                    result.start_line,
                    result.end_line,
                    result.kind,
                    result.score_kind,
                    result.score,
                ));
                output.push_str("     ");
                output.push_str(&escape_controls(&result.snippet));
                output.push('\n');
            }
            Err(error) => {
                output.push_str(&format!(
                    "  {}. [unrenderable snapshot evidence: {}]\n",
                    rank + 1,
                    escape_controls(&error.to_string())
                ));
            }
        }
    }
    output
}

#[derive(Debug, Serialize)]
struct JsonDocument<'a> {
    schema_version: u32,
    query: &'a str,
    root: JsonPath,
    mode: &'static str,
    model: Option<JsonModel>,
    results: Vec<JsonResult>,
    stats: JsonStats,
    diagnostics: Vec<JsonDiagnostic>,
}

#[derive(Debug, Serialize)]
struct JsonModel {
    id: String,
    revision: String,
    score_kind: &'static str,
}

impl From<&ModelMetadata> for JsonModel {
    fn from(value: &ModelMetadata) -> Self {
        Self {
            id: value.id.clone(),
            revision: value.revision.clone(),
            score_kind: value.score_kind.as_str(),
        }
    }
}

#[derive(Debug, Serialize)]
struct JsonPath {
    /// Relative UTF-8 path when representable.  Non-UTF-8 names use null and
    /// retain a readable lossy display plus lossless Base64 bytes instead.
    path: Option<String>,
    path_display: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    path_bytes_base64: Option<String>,
}

#[derive(Debug, Serialize)]
struct JsonResult {
    #[serde(flatten)]
    path: JsonPath,
    start_byte: usize,
    end_byte: usize,
    start_line: usize,
    end_line: usize,
    kind: &'static str,
    score: f64,
    score_kind: &'static str,
    snippet: String,
}

#[derive(Debug, Serialize)]
struct JsonDiagnostic {
    kind: String,
    #[serde(flatten)]
    path: JsonPath,
    message: String,
}

#[derive(Debug, Serialize)]
struct JsonStats {
    #[serde(skip_serializing_if = "Option::is_none")]
    entries_seen: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    files_considered: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    files_read: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    files_accepted: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bytes_read: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    policy_exclusions_observed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    failures: Option<u64>,
    scan_complete: bool,
    partial: bool,
    lexical_evidence: bool,
    scoring_complete: Option<bool>,
    chunks_total: usize,
    chunks_selected: usize,
    chunks_evaluated: usize,
    results_returned: usize,
    deduplicated_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    candidate_selection: Option<JsonCandidateSelection>,
    timing: JsonTiming,
}

impl From<&PresentationStats> for JsonStats {
    fn from(value: &PresentationStats) -> Self {
        Self {
            entries_seen: value.entries_seen,
            files_considered: value.files_considered,
            files_read: value.files_read,
            files_accepted: value.files_accepted,
            bytes_read: value.bytes_read,
            policy_exclusions_observed: value.policy_exclusions_observed,
            failures: value.failures,
            scan_complete: value.scan_complete,
            partial: value.partial,
            lexical_evidence: value.lexical_evidence,
            scoring_complete: value.scoring_complete,
            chunks_total: value.chunks_total,
            chunks_selected: value.chunks_selected,
            chunks_evaluated: value.chunks_evaluated,
            results_returned: value.results_returned,
            deduplicated_count: value.deduplicated_count,
            candidate_selection: value.candidate_selection.map(JsonCandidateSelection::from),
            timing: JsonTiming::from(value.timing),
        }
    }
}

#[derive(Debug, Serialize)]
struct JsonCandidateSelection {
    candidate_limit: usize,
    quota: JsonCandidateQuota,
    body_selected: usize,
    path_selected: usize,
    neighbours_selected: usize,
    diverse_selected: usize,
}

impl From<CandidateSelection> for JsonCandidateSelection {
    fn from(value: CandidateSelection) -> Self {
        Self {
            candidate_limit: value.candidate_limit,
            quota: JsonCandidateQuota {
                body: value.quota.body,
                path: value.quota.path,
                neighbours: value.quota.neighbours,
                diverse: value.quota.diverse,
            },
            body_selected: value.body_selected,
            path_selected: value.path_selected,
            neighbours_selected: value.neighbours_selected,
            diverse_selected: value.diverse_selected,
        }
    }
}

#[derive(Debug, Serialize)]
struct JsonCandidateQuota {
    body: usize,
    path: usize,
    neighbours: usize,
    diverse: usize,
}

#[derive(Debug, Serialize)]
struct JsonTiming {
    #[serde(skip_serializing_if = "Option::is_none")]
    discovery_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chunking_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_load_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokenization_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inference_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    search_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_ms: Option<u64>,
}

impl From<PresentationTiming> for JsonTiming {
    fn from(value: PresentationTiming) -> Self {
        Self {
            discovery_ms: value.discovery_ms,
            chunking_ms: value.chunking_ms,
            model_load_ms: value.model_load_ms,
            tokenization_ms: value.tokenization_ms,
            inference_ms: value.inference_ms,
            search_ms: value.search_ms,
            total_ms: value.total_ms,
        }
    }
}

#[derive(Debug)]
struct HumanResult {
    path: JsonPath,
    start_byte: usize,
    end_byte: usize,
    start_line: usize,
    end_line: usize,
    kind: &'static str,
    score: f64,
    score_kind: &'static str,
    snippet: String,
}

fn json_result(
    entry: &RankedEntry,
    root: &Path,
    sources: &HashMap<SourceId, &Source>,
) -> Result<JsonResult> {
    let result = human_result(entry, root, sources)?;
    Ok(JsonResult {
        path: result.path,
        start_byte: result.start_byte,
        end_byte: result.end_byte,
        start_line: result.start_line,
        end_line: result.end_line,
        kind: result.kind,
        score: result.score,
        score_kind: result.score_kind,
        snippet: result.snippet,
    })
}

fn human_result(
    entry: &RankedEntry,
    root: &Path,
    sources: &HashMap<SourceId, &Source>,
) -> Result<HumanResult> {
    let source = sources.get(&entry.source_id).ok_or_else(|| {
        SupergrepError::Internal(format!(
            "cannot render result for source id {}: exact search snapshot is unavailable",
            entry.source_id.get()
        ))
    })?;
    let snippet = source
        .slice(entry.start_byte..entry.end_byte)
        .ok_or_else(|| {
            SupergrepError::Internal(format!(
                "cannot render result for {} bytes {}..{}: range is not valid in the search snapshot",
                source.path().display(),
                entry.start_byte,
                entry.end_byte
            ))
        })?;
    let (score, score_kind, kind) = if let Some(logit) = entry.relevance_logit {
        (f64::from(logit), "raw_logit", "model_reranked_chunk")
    } else {
        (
            entry.lexical_score.ok_or_else(|| {
                SupergrepError::Internal(format!(
                    "cannot render lexical result for chunk {} without a BM25 score",
                    entry.chunk_index
                ))
            })?,
            "bm25",
            "lexical_chunk",
        )
    };
    Ok(HumanResult {
        path: path_for_output(root, &entry.path),
        start_byte: entry.start_byte,
        end_byte: entry.end_byte,
        start_line: entry.start_line,
        end_line: entry.end_line,
        kind,
        score,
        score_kind,
        snippet: truncate_snippet(snippet, 500),
    })
}

fn source_map(sources: &[Source]) -> HashMap<SourceId, &Source> {
    sources.iter().map(|source| (source.id(), source)).collect()
}

fn mode_name(mode: SearchMode) -> &'static str {
    match mode {
        SearchMode::Lexical => "lexical",
        SearchMode::Fast => "fast",
        SearchMode::Deep => "deep",
    }
}

fn optional_bool(value: Option<bool>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "not_applicable".into())
}

fn path_for_output(root: &Path, path: &Path) -> JsonPath {
    let relative = if path == root {
        path
    } else {
        path.strip_prefix(root).unwrap_or(path)
    };
    match relative.to_str() {
        Some(path) => JsonPath {
            path: Some(path.to_owned()),
            path_display: path.to_owned(),
            path_bytes_base64: None,
        },
        None => JsonPath {
            path: None,
            path_display: relative.to_string_lossy().into_owned(),
            path_bytes_base64: path_bytes(relative).map(|bytes| BASE64.encode(bytes)),
        },
    }
}

#[cfg(unix)]
fn path_bytes(path: &Path) -> Option<Vec<u8>> {
    use std::os::unix::ffi::OsStrExt;

    Some(path.as_os_str().as_bytes().to_vec())
}

#[cfg(not(unix))]
fn path_bytes(_path: &Path) -> Option<Vec<u8>> {
    // Non-Unix `OsString` values have no portable lossless byte encoding.
    None
}

fn rendered_diagnostics(input: &PresentationInput<'_>) -> Vec<JsonDiagnostic> {
    let mut diagnostics = input
        .diagnostics
        .iter()
        .map(|diagnostic| JsonDiagnostic {
            kind: diagnostic.kind.clone(),
            path: diagnostic
                .path
                .as_deref()
                .map(|path| path_for_output(input.root, path))
                .unwrap_or_else(empty_path),
            message: diagnostic.message.clone(),
        })
        .collect::<Vec<_>>();
    let mut status = |kind: &str, message: String| {
        diagnostics.push(JsonDiagnostic {
            kind: kind.to_owned(),
            path: empty_path(),
            message,
        });
    };
    if !input.stats.scan_complete {
        status(
            "scan_incomplete",
            "not every source traversal/read operation completed".into(),
        );
    }
    if input.stats.partial {
        status(
            "partial_results",
            "one or more resource or operational limits left coverage partial".into(),
        );
    }
    if input.stats.scoring_complete == Some(false) {
        status(
            "scoring_incomplete",
            format!(
                "fast mode evaluated {} of {} chunks; use --deep to evaluate every chunk",
                input.stats.chunks_evaluated, input.stats.chunks_total
            ),
        );
    }
    if !input.stats.lexical_evidence {
        status(
            "no_lexical_evidence",
            "the query had no positive lexical evidence; fast candidate coverage used its broad sample".into(),
        );
    }
    diagnostics
}

fn empty_path() -> JsonPath {
    JsonPath {
        path: None,
        path_display: String::new(),
        path_bytes_base64: None,
    }
}

fn truncate_snippet(snippet: &str, limit: usize) -> String {
    if snippet.chars().count() <= limit {
        return snippet.to_owned();
    }
    let end = snippet
        .char_indices()
        .nth(limit)
        .map(|(index, _)| index)
        .unwrap_or(snippet.len());
    format!("{}…", &snippet[..end])
}

fn escape_controls(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            '\0' => escaped.push_str("\\0"),
            character if character.is_control() => {
                use std::fmt::Write as _;
                let _ = write!(escaped, "\\u{{{:04X}}}", character as u32);
            }
            character => escaped.push(character),
        }
    }
    escaped
}
