use std::{
    fs,
    process::{Command, Stdio},
};

use serde_json::Value;
use tempfile::tempdir;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_supergrep")
}

#[test]
fn help_and_model_list_do_not_require_an_onnx_runtime() {
    let help = Command::new(binary()).arg("--help").output().unwrap();
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("Search code, documentation"));

    let cache = tempdir().unwrap();
    let list = Command::new(binary())
        .args(["model", "list", "--json", "--cache-dir"])
        .arg(cache.path())
        .env("SUPERGREP_ORT_LIB", "/not/a/runtime/library.so")
        .output()
        .unwrap();
    assert!(
        list.status.success(),
        "{}",
        String::from_utf8_lossy(&list.stderr)
    );
    let document: Value = serde_json::from_slice(&list.stdout).unwrap();
    assert_eq!(document["schema_version"], 1);
    assert_eq!(document["profiles"].as_array().unwrap().len(), 2);
}

#[test]
fn lexical_cli_is_model_and_runtime_independent_and_emits_one_json_object() {
    let directory = tempdir().unwrap();
    fs::write(
        directory.path().join("settings.toml"),
        "# retry policy\nretry_delay_seconds = 30\n",
    )
    .unwrap();
    let output = Command::new(binary())
        .args(["retry delay"])
        .arg(directory.path())
        .args(["--lexical", "--json"])
        .env("SUPERGREP_ORT_LIB", "/not/a/runtime/library.so")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.lines().count(), 1);
    let document: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(document["mode"], "lexical");
    assert!(document["model"].is_null());
    assert_eq!(document["results"][0]["path"], "settings.toml");
    assert_eq!(document["results"][0]["score_kind"], "bm25");
    assert!(document["stats"]["timing"].get("inference_ms").is_none());
}

#[test]
fn lexical_cli_reloads_file_edits_renames_and_deletions_on_each_run() {
    let directory = tempdir().unwrap();
    let original = directory.path().join("settings.toml");
    fs::write(&original, "saffron_delta = true\n").unwrap();

    let search = |query: &str| {
        Command::new(binary())
            .arg(query)
            .arg(directory.path())
            .args(["--lexical", "--json"])
            .output()
            .unwrap()
    };
    let first = search("saffron_delta");
    assert!(first.status.success());
    let first_json: Value = serde_json::from_slice(&first.stdout).unwrap();
    assert_eq!(first_json["results"][0]["path"], "settings.toml");

    fs::write(&original, "quartz_orbit = true\n").unwrap();
    let stale = search("saffron_delta");
    assert_eq!(stale.status.code(), Some(1));
    let stale_json: Value = serde_json::from_slice(&stale.stdout).unwrap();
    assert!(stale_json["results"].as_array().unwrap().is_empty());

    let edited = search("quartz_orbit");
    assert!(edited.status.success());
    let edited_json: Value = serde_json::from_slice(&edited.stdout).unwrap();
    assert_eq!(edited_json["results"][0]["path"], "settings.toml");

    let renamed = directory.path().join("renamed.toml");
    fs::rename(&original, &renamed).unwrap();
    let renamed_result = search("quartz_orbit");
    assert!(renamed_result.status.success());
    let renamed_json: Value = serde_json::from_slice(&renamed_result.stdout).unwrap();
    assert_eq!(renamed_json["results"][0]["path"], "renamed.toml");

    fs::remove_file(renamed).unwrap();
    let deleted = search("quartz_orbit");
    assert_eq!(deleted.status.code(), Some(1));
    let deleted_json: Value = serde_json::from_slice(&deleted.stdout).unwrap();
    assert!(deleted_json["results"].as_array().unwrap().is_empty());
}

#[test]
fn read_limit_returns_explicit_partial_json_and_exit_code() {
    let directory = tempdir().unwrap();
    fs::write(directory.path().join("a.txt"), "retry first\n").unwrap();
    fs::write(directory.path().join("b.txt"), "retry second\n").unwrap();

    let output = Command::new(binary())
        .arg("retry")
        .arg(directory.path())
        .args(["--lexical", "--json", "--max-total-bytes", "20"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(3));
    let document: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(document["stats"]["partial"], true);
    assert_eq!(document["stats"]["scan_complete"], false);
    assert_eq!(document["stats"]["scoring_complete"], Value::Null);
    assert_eq!(document["results"][0]["path"], "a.txt");
    assert!(document["diagnostics"]
        .as_array()
        .unwrap()
        .iter()
        .any(|diagnostic| diagnostic["kind"] == "total_byte_limit"));
}

#[test]
fn missing_model_is_a_clear_fatal_error_not_a_lexical_fallback() {
    let directory = tempdir().unwrap();
    let corpus = tempdir().unwrap();
    fs::write(
        corpus.path().join("readme.md"),
        "retry after a failed operation\n",
    )
    .unwrap();
    let output = Command::new(binary())
        .args(["failed operation retry"])
        .arg(corpus.path())
        .args(["--json", "--cache-dir"])
        .arg(directory.path())
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("not prepared"), "{stderr}");
    assert!(
        stderr.contains("model download compact-multilingual"),
        "{stderr}"
    );
}

#[test]
fn rejects_search_options_that_have_no_meaning_in_the_selected_mode() {
    let deep = Command::new(binary())
        .args(["anything", ".", "--deep", "--candidates", "4"])
        .output()
        .unwrap();
    assert_eq!(deep.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&deep.stderr).contains("does not apply with --deep"));

    let lexical = Command::new(binary())
        .args(["anything", ".", "--lexical", "--candidates", "4"])
        .output()
        .unwrap();
    assert_eq!(lexical.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&lexical.stderr).contains("does not apply with --lexical"));
}

#[test]
fn option_like_query_can_be_passed_after_separator() {
    let directory = tempdir().unwrap();
    fs::write(
        directory.path().join("flags.txt"),
        "--dangerous is ordinary text\n",
    )
    .unwrap();
    let output = Command::new(binary())
        .args(["--lexical", "--json", "--", "--dangerous"])
        .arg(directory.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let document: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(document["query"], "--dangerous");
}

#[test]
fn closed_stdout_pipe_does_not_panic() {
    let cache = tempdir().unwrap();
    let mut child = Command::new(binary())
        .args(["model", "list", "--cache-dir"])
        .arg(cache.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    drop(child.stdout.take());
    let output = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert!(!output.status.success());
}

#[cfg(unix)]
#[test]
fn interrupt_signal_terminates_search_without_a_panic() {
    use std::os::unix::process::ExitStatusExt;

    let directory = tempdir().unwrap();
    fs::write(
        directory.path().join("large.txt"),
        vec![b'x'; 2 * 1024 * 1024],
    )
    .unwrap();
    let child = Command::new(binary())
        .args(["a query that will not match"])
        .arg(directory.path())
        .arg("--lexical")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let signal = Command::new("sh")
        .args(["-c", &format!("kill -INT {}", child.id())])
        .status()
        .unwrap();
    assert!(signal.success(), "could not signal child {}", child.id());
    let output = child.wait_with_output().unwrap();
    assert_eq!(output.status.signal(), Some(2));
    assert!(!String::from_utf8_lossy(&output.stderr).contains("panicked"));
}
