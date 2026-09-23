use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

use supergrep::discovery::{discover, DiscoveryDiagnosticKind, DiscoveryError, DiscoveryOptions};
use tempfile::tempdir;

fn write(path: &Path, bytes: &[u8]) {
    let mut file = fs::File::create(path).expect("create fixture file");
    file.write_all(bytes).expect("write fixture file");
}

fn source_names(report: &supergrep::discovery::DiscoveryReport) -> Vec<String> {
    report
        .sources
        .iter()
        .map(|source| {
            source
                .path()
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned()
        })
        .collect()
}

fn has_diagnostic(
    report: &supergrep::discovery::DiscoveryReport,
    kind: DiscoveryDiagnosticKind,
) -> bool {
    report
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.kind == kind)
}

#[test]
fn ignore_hidden_and_no_ignore_are_independent() {
    let directory = tempdir().unwrap();
    write(
        &directory.path().join(".ignore"),
        b"ignored.txt\n!unignored.txt\n",
    );
    write(&directory.path().join("visible.txt"), b"visible\n");
    write(&directory.path().join("ignored.txt"), b"ignored\n");
    write(&directory.path().join("unignored.txt"), b"unignored\n");
    write(&directory.path().join(".hidden.txt"), b"hidden\n");

    let normal = discover(&DiscoveryOptions::new(directory.path())).unwrap();
    assert_eq!(source_names(&normal), vec!["unignored.txt", "visible.txt"]);
    assert!(!normal.partial);

    let mut no_ignore = DiscoveryOptions::new(directory.path());
    no_ignore.respect_ignore = false;
    let no_ignore = discover(&no_ignore).unwrap();
    assert_eq!(
        source_names(&no_ignore),
        vec!["ignored.txt", "unignored.txt", "visible.txt"]
    );

    let mut hidden = DiscoveryOptions::new(directory.path());
    hidden.include_hidden = true;
    let hidden = discover(&hidden).unwrap();
    assert!(source_names(&hidden).contains(&".hidden.txt".to_owned()));
    assert!(!source_names(&hidden).contains(&"ignored.txt".to_owned()));
}

#[test]
fn git_metadata_is_excluded_even_when_hidden_and_ignore_filters_are_disabled() {
    let directory = tempdir().unwrap();
    fs::create_dir(directory.path().join(".git")).unwrap();
    write(
        &directory.path().join(".git").join("config"),
        b"secret git configuration\n",
    );
    write(&directory.path().join("outside.txt"), b"search me\n");

    let mut options = DiscoveryOptions::new(directory.path());
    options.include_hidden = true;
    options.respect_ignore = false;
    let report = discover(&options).unwrap();

    assert_eq!(source_names(&report), vec!["outside.txt"]);
    assert!(!report
        .sources
        .iter()
        .any(|source| source.path().ends_with(".git/config")));
}

#[test]
fn git_component_canceled_by_parent_is_not_mistaken_for_git_metadata() {
    let directory = tempdir().unwrap();
    let git = directory.path().join(".git");
    fs::create_dir(&git).unwrap();
    let visible = directory.path().join("visible.txt");
    write(&visible, b"visible\n");

    let spelled_through_git = git.join("..").join("visible.txt");
    let report = discover(&DiscoveryOptions::new(spelled_through_git)).unwrap();
    assert_eq!(report.sources.len(), 1);
    assert_eq!(report.sources[0].text(), "visible\n");
    assert!(!has_diagnostic(
        &report,
        DiscoveryDiagnosticKind::GitMetadata
    ));
}

#[test]
fn explicit_file_root_overrides_hidden_and_ignore_but_not_content_policy() {
    let directory = tempdir().unwrap();
    write(&directory.path().join(".ignore"), b".chosen.txt\n");
    let chosen = directory.path().join(".chosen.txt");
    write(&chosen, b"direct root\n");
    let report = discover(&DiscoveryOptions::new(&chosen)).unwrap();
    assert_eq!(report.sources.len(), 1);
    assert_eq!(report.sources[0].text(), "direct root\n");

    let binary = directory.path().join("binary.txt");
    write(&binary, b"direct\0but binary");
    let binary_report = discover(&DiscoveryOptions::new(&binary)).unwrap();
    assert!(binary_report.sources.is_empty());
    assert!(!binary_report.partial);
    assert!(binary_report.scan_complete);
    assert!(has_diagnostic(
        &binary_report,
        DiscoveryDiagnosticKind::NulByte
    ));
}

#[test]
fn deterministic_order_and_limits_surface_partial_status() {
    let directory = tempdir().unwrap();
    write(&directory.path().join("c.txt"), b"ccc");
    write(&directory.path().join("a.txt"), b"aaa");
    write(&directory.path().join("b.txt"), b"bbb");
    let mut options = DiscoveryOptions::new(directory.path());
    options.max_file_size = 10;
    options.max_total_bytes = 4;
    let report = discover(&options).unwrap();

    assert_eq!(source_names(&report), vec!["a.txt"]);
    assert_eq!(report.stats.bytes_read, 3);
    assert!(report.partial);
    assert!(!report.scan_complete);
    assert!(has_diagnostic(
        &report,
        DiscoveryDiagnosticKind::TotalByteLimit
    ));

    let mut per_file = DiscoveryOptions::new(directory.path());
    per_file.max_file_size = 2;
    let report = discover(&per_file).unwrap();
    assert!(report.sources.is_empty());
    assert!(has_diagnostic(
        &report,
        DiscoveryDiagnosticKind::FileTooLarge
    ));
}

#[test]
fn exact_budget_exhaustion_is_partial_only_when_later_candidates_exist() {
    let directory = tempdir().unwrap();
    write(&directory.path().join("a.txt"), b"aaa");
    write(&directory.path().join("b.txt"), b"b");
    write(&directory.path().join("c.txt"), b"c");
    let mut options = DiscoveryOptions::new(directory.path());
    options.max_file_size = 10;
    options.max_total_bytes = 4;
    let report = discover(&options).unwrap();
    assert_eq!(source_names(&report), vec!["a.txt", "b.txt"]);
    assert!(report.partial);
    assert!(has_diagnostic(
        &report,
        DiscoveryDiagnosticKind::TotalByteLimit
    ));

    fs::remove_file(directory.path().join("c.txt")).unwrap();
    let complete = discover(&options).unwrap();
    assert_eq!(source_names(&complete), vec!["a.txt", "b.txt"]);
    assert!(!complete.partial);
    assert!(complete.scan_complete);
}

#[test]
fn binary_policy_exclusion_that_uses_the_budget_still_reports_unread_later_files() {
    let directory = tempdir().unwrap();
    write(&directory.path().join("a.bin"), b"x\0y");
    write(&directory.path().join("b.txt"), b"b");
    let mut options = DiscoveryOptions::new(directory.path());
    options.max_file_size = 10;
    options.max_total_bytes = 3;
    let report = discover(&options).unwrap();
    assert!(report.sources.is_empty());
    assert!(report.partial);
    assert!(has_diagnostic(&report, DiscoveryDiagnosticKind::NulByte));
    assert!(has_diagnostic(
        &report,
        DiscoveryDiagnosticKind::TotalByteLimit
    ));
}

#[test]
fn invalid_text_has_distinct_diagnostics_and_source_order_is_stable() {
    let directory = tempdir().unwrap();
    write(&directory.path().join("z.txt"), b"z\n");
    write(&directory.path().join("nul.bin"), b"x\0y");
    write(&directory.path().join("invalid.bin"), &[0xff, b'\n']);
    write(&directory.path().join("a.txt"), b"a\n");
    let report = discover(&DiscoveryOptions::new(directory.path())).unwrap();

    assert_eq!(source_names(&report), vec!["a.txt", "z.txt"]);
    assert!(!report.partial);
    assert!(report.scan_complete);
    assert!(has_diagnostic(&report, DiscoveryDiagnosticKind::NulByte));
    assert!(has_diagnostic(
        &report,
        DiscoveryDiagnosticKind::InvalidUtf8
    ));
}

#[cfg(unix)]
#[test]
fn unreadable_regular_file_is_a_read_failure_and_partial_scan() {
    use std::os::unix::fs::PermissionsExt;

    let directory = tempdir().unwrap();
    let unreadable = directory.path().join("unreadable.txt");
    write(&unreadable, b"not readable\n");
    fs::set_permissions(&unreadable, fs::Permissions::from_mode(0o000)).unwrap();

    let report = discover(&DiscoveryOptions::new(directory.path())).unwrap();
    assert!(report.sources.is_empty());
    assert!(report.partial);
    assert!(!report.scan_complete);
    assert!(has_diagnostic(
        &report,
        DiscoveryDiagnosticKind::ReadFailure
    ));
}

#[test]
fn globs_are_applied_to_relative_paths() {
    let directory = tempdir().unwrap();
    fs::create_dir(directory.path().join("nested")).unwrap();
    write(
        &directory.path().join("nested").join("kept.rs"),
        b"fn kept() {}\n",
    );
    write(&directory.path().join("nested").join("other.md"), b"nope\n");
    let mut options = DiscoveryOptions::new(directory.path());
    options.globs.push("**/*.rs".into());
    let report = discover(&options).unwrap();
    assert_eq!(report.sources.len(), 1);
    assert!(report.sources[0].path().ends_with("nested/kept.rs"));
    assert!(!report.partial);
}

#[test]
fn glob_matches_an_explicit_file_root_by_its_basename() {
    let directory = tempdir().unwrap();
    let file = directory.path().join("chosen.rs");
    write(&file, b"fn chosen() {}\n");
    let mut options = DiscoveryOptions::new(&file);
    options.globs.push("*.rs".into());
    assert_eq!(discover(&options).unwrap().sources.len(), 1);

    options.globs = vec!["*.md".into()];
    let report = discover(&options).unwrap();
    assert!(report.sources.is_empty());
    assert!(!report.partial);
    assert!(has_diagnostic(
        &report,
        DiscoveryDiagnosticKind::GlobFiltered
    ));
}

#[test]
fn missing_root_is_an_input_error() {
    let missing = PathBuf::from("/definitely/not/a/supergrep-fixture-root");
    assert!(matches!(
        discover(&DiscoveryOptions::new(missing)),
        Err(DiscoveryError::MissingRoot(_))
    ));
}

#[cfg(unix)]
#[test]
fn symbolic_links_are_never_read_or_followed() {
    use std::os::unix::fs::symlink;

    let directory = tempdir().unwrap();
    let target = directory.path().join("target.txt");
    write(&target, b"target\n");
    let root_link = directory.path().join("root-link");
    symlink(&target, &root_link).unwrap();
    assert!(matches!(
        discover(&DiscoveryOptions::new(&root_link)),
        Err(DiscoveryError::SymlinkRoot(_))
    ));

    let real_directory = directory.path().join("real-directory");
    fs::create_dir(&real_directory).unwrap();
    fs::create_dir(real_directory.join("nested")).unwrap();
    write(
        &real_directory.join("nested").join("inside.txt"),
        b"inside\n",
    );
    let directory_link = directory.path().join("directory-link");
    symlink(&real_directory, &directory_link).unwrap();
    assert!(matches!(
        discover(&DiscoveryOptions::new(directory_link.join("nested"))),
        Err(DiscoveryError::SymlinkRoot(_))
    ));

    let child_link = directory.path().join("child-link.txt");
    symlink(&target, &child_link).unwrap();
    let report = discover(&DiscoveryOptions::new(directory.path())).unwrap();
    assert_eq!(source_names(&report), vec!["inside.txt", "target.txt"]);
    assert!(has_diagnostic(&report, DiscoveryDiagnosticKind::Symlink));
}
