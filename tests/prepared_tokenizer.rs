//! Opt-in equivalence check against the pinned, locally prepared tokenizer.

use std::{
    fs,
    path::{Path, PathBuf},
};

use supergrep::model::{built_in_registry, PairTokenizer, TokenizerContract};

fn text_files(root: &Path, output: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(root).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            text_files(&path, output);
        } else {
            output.push(path);
        }
    }
}

#[test]
#[ignore = "requires SUPERGREP_TEST_MODEL_CACHE with the pinned tokenizer"]
fn prepared_pair_lengths_match_direct_encoding_on_every_fixed_evaluation_query_and_file() {
    let cache = PathBuf::from(
        std::env::var_os("SUPERGREP_TEST_MODEL_CACHE")
            .expect("set SUPERGREP_TEST_MODEL_CACHE to the prepared model cache"),
    );
    let registry = built_in_registry().unwrap();
    let profile = registry.profile("compact-multilingual").unwrap();
    let metadata = profile.tokenizer();
    let tokenizer_path = cache
        .join(profile.id())
        .join(profile.revision())
        .join(&metadata.file);
    let tokenizer = PairTokenizer::from_file(
        &tokenizer_path,
        TokenizerContract {
            max_pair_tokens: metadata.max_pair_tokens,
            max_query_tokens: metadata.max_query_tokens,
            pad_id: metadata.pad_token_id,
        },
    )
    .unwrap();

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("eval");
    let mut paths = Vec::new();
    text_files(&root.join("corpus"), &mut paths);
    paths.sort();
    let mut passages = paths
        .iter()
        .map(|path| fs::read_to_string(path).unwrap())
        .collect::<Vec<_>>();
    passages.extend([
        "한글 😀\r\nwith a final line without newline".to_owned(),
        "x".repeat(2_000),
        "".to_owned(),
    ]);
    let definitions = fs::read_to_string(root.join("queries.jsonl")).unwrap();
    let mut checked = 0usize;
    for line in definitions.lines().filter(|line| !line.is_empty()) {
        let record: serde_json::Value = serde_json::from_str(line).unwrap();
        let query = record["query"].as_str().unwrap();
        let prepared = tokenizer.prepare_pair_query(query).unwrap();
        for passage in &passages {
            assert_eq!(
                prepared.token_count(passage).unwrap(),
                tokenizer.pair_token_count(query, passage).unwrap(),
                "query {}",
                record["id"]
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 90 * passages.len());
}
