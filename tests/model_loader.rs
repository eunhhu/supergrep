use std::{fs, path::Path};

use sha2::{Digest, Sha256};
use supergrep::{
    model::{
        resolve_model_spec, validate_graph_contract, ModelCache, ModelRegistry, OnnxSessionContract,
    },
    SupergrepError,
};

fn fixture_registry(body: &[u8], tokenizer: &[u8]) -> ModelRegistry {
    let body_digest = format!("{:x}", Sha256::digest(body));
    let tokenizer_digest = format!("{:x}", Sha256::digest(tokenizer));
    ModelRegistry::from_toml_str(&format!(
        r#"
format_version = 1
[[profile]]
id = "fixture"
repository = "example/fixture"
revision = "0123456789abcdef0123456789abcdef01234567"
license = "Apache-2.0"
tokenizer_file = "tokenizer.json"
tokenizer_family = "bert-wordpiece"
tokenizer_pad_token_id = 0
max_pair_tokens = 256
max_query_tokens = 64
tokenizer_model_max_tokens = 512
model_type = "bert"
model_architecture = "BertForSequenceClassification"
type_vocab_size = 2
runtime_family = "onnxruntime-cpu-dynamic"
runtime_wrapper_version = "2.0.0-rc.9"
dynamic_loading = true
required_inputs = ["input_ids", "attention_mask"]
optional_inputs = ["token_type_ids"]
expected_output_count = 1
score_kind = "relevance_logit"
[[profile.artifact]]
kind = "onnx_model"
path = "onnx/model.onnx"
sha256 = "{body_digest}"
size_bytes = {}
[[profile.artifact]]
kind = "tokenizer"
path = "tokenizer.json"
sha256 = "{tokenizer_digest}"
size_bytes = {}
"#,
        body.len(),
        tokenizer.len()
    ))
    .unwrap()
}

fn write_fixture(directory: &Path, body: &[u8], tokenizer: &[u8]) {
    fs::create_dir_all(directory.join("onnx")).unwrap();
    fs::write(directory.join("onnx/model.onnx"), body).unwrap();
    fs::write(directory.join("tokenizer.json"), tokenizer).unwrap();
}

#[test]
fn local_directory_must_verify_all_profile_artifacts() {
    let body = b"model";
    let tokenizer = b"tokenizer";
    let registry = fixture_registry(body, tokenizer);
    let temp = tempfile::tempdir().unwrap();
    let directory = temp.path().join("local");
    write_fixture(&directory, body, tokenizer);
    let cache = ModelCache::new(temp.path().join("cache")).unwrap();

    let resolved = resolve_model_spec(&registry, &cache, directory.to_str().unwrap()).unwrap();
    assert_eq!(resolved.profile.id(), "fixture");
    assert_eq!(resolved.model_path, directory.join("onnx/model.onnx"));
    assert_eq!(resolved.tokenizer_path, directory.join("tokenizer.json"));

    fs::write(directory.join("tokenizer.json"), b"tampered").unwrap();
    let error = resolve_model_spec(&registry, &cache, directory.to_str().unwrap()).unwrap_err();
    assert!(error.to_string().contains("did not verify"));
}

#[test]
fn named_profile_reports_download_guidance_without_network() {
    let registry = fixture_registry(b"model", b"tokenizer");
    let temp = tempfile::tempdir().unwrap();
    let cache = ModelCache::new(temp.path().join("cache")).unwrap();
    let error = resolve_model_spec(&registry, &cache, "fixture").unwrap_err();
    assert!(error.to_string().contains("model download fixture"));
}

#[test]
fn graph_contract_rejects_missing_or_extra_inputs() {
    let registry = fixture_registry(b"model", b"tokenizer");
    let profile = registry.profile("fixture").unwrap();
    validate_graph_contract(
        profile,
        &OnnxSessionContract {
            inputs: vec![
                "token_type_ids".into(),
                "input_ids".into(),
                "attention_mask".into(),
            ],
            outputs: vec!["logits".into()],
        },
    )
    .unwrap();
    let error = validate_graph_contract(
        profile,
        &OnnxSessionContract {
            inputs: vec!["input_ids".into(), "attention_mask".into(), "wrong".into()],
            outputs: vec!["logits".into()],
        },
    )
    .unwrap_err();
    assert!(matches!(error, SupergrepError::Model(_)));
}
