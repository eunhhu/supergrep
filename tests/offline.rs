use std::{fs, path::PathBuf, process::Command};

use serde_json::Value;
use tempfile::tempdir;

/// This check is deliberately opt-in: it requires the real pinned model,
/// official ARM64 ONNX Runtime, and the external syscall tracer.
#[test]
#[ignore = "requires SUPERGREP_TEST_MODEL_CACHE, SUPERGREP_TEST_ORT_LIB, and strace"]
fn prepared_model_search_succeeds_without_any_network_syscalls() {
    let cache = PathBuf::from(std::env::var_os("SUPERGREP_TEST_MODEL_CACHE").unwrap());
    let runtime = PathBuf::from(std::env::var_os("SUPERGREP_TEST_ORT_LIB").unwrap());
    assert!(cache.is_dir(), "missing model cache: {}", cache.display());
    assert!(
        runtime.is_file(),
        "missing runtime library: {}",
        runtime.display()
    );

    let corpus = tempdir().unwrap();
    let output_directory = tempdir().unwrap();
    let trace = output_directory.path().join("strace.txt");
    let retries = b"pub fn backoff(failures: u32) -> u32 {\n    1 << failures.min(8)\n}\n";
    fs::write(corpus.path().join("retries.rs"), retries).unwrap();
    fs::write(
        corpus.path().join("other.rs"),
        "pub fn display_name(first: &str, last: &str) -> String {\n    format!(\"{first} {last}\")\n}\n",
    )
    .unwrap();

    let result = Command::new("strace")
        .args(["-f", "-e", "trace=network", "-o"])
        .arg(&trace)
        .arg(env!("CARGO_BIN_EXE_supergrep"))
        .args([
            "exponential retry delay",
            "--deep",
            "--json",
            "--top-k",
            "2",
        ])
        .arg(corpus.path())
        .args(["--cache-dir"])
        .arg(cache)
        .env("SUPERGREP_ORT_LIB", runtime)
        .output()
        .unwrap();

    assert!(
        result.status.success(),
        "status={:?}\nstdout={}\nstderr={}",
        result.status.code(),
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let document: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(document["model"]["id"], "compact-multilingual");
    assert_eq!(document["mode"], "deep");
    assert_eq!(document["stats"]["scoring_complete"], true);
    assert_eq!(document["stats"]["chunks_total"], 2);
    assert_eq!(document["stats"]["chunks_evaluated"], 2);
    let retries_result = document["results"]
        .as_array()
        .unwrap()
        .iter()
        .find(|result| result["path"] == "retries.rs")
        .expect("deep search should return the retry source");
    assert_eq!(retries_result["start_byte"], 0);
    assert_eq!(retries_result["end_byte"], retries.len());
    assert_eq!(retries_result["start_line"], 1);
    assert_eq!(retries_result["end_line"], 3);
    assert_eq!(retries_result["score_kind"], "raw_logit");
    assert!(
        document["stats"]["timing"]["model_load_ms"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(
        document["stats"]["timing"]["tokenization_ms"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(
        document["stats"]["timing"]["inference_ms"]
            .as_u64()
            .unwrap()
            > 0
    );

    let trace_text = fs::read_to_string(trace).unwrap();
    for syscall in ["socket(", "connect(", "sendto(", "recvfrom("] {
        assert!(
            !trace_text.contains(syscall),
            "offline search attempted network syscall {syscall}:\n{trace_text}"
        );
    }
}
