//! Network-free tests for revision-pinned model lifecycle behavior.

#![allow(dead_code)]

#[path = "../src/error.rs"]
mod error;

pub use error::{Result, SupergrepError};

#[path = "../src/model/download.rs"]
mod download;
#[path = "../src/model/registry.rs"]
mod registry;

use std::{
    cell::RefCell,
    collections::BTreeMap,
    fs,
    io::{Cursor, Read},
    path::Path,
};

use download::{ArtifactFetcher, ModelCache, ModelDownloader};
use registry::ModelRegistry;
use sha2::{Digest, Sha256};

#[derive(Default)]
struct FakeFetcher {
    bodies: BTreeMap<String, Vec<u8>>,
    calls: RefCell<Vec<String>>,
}

impl FakeFetcher {
    fn with_bodies(bodies: BTreeMap<String, Vec<u8>>) -> Self {
        Self {
            bodies,
            calls: RefCell::new(Vec::new()),
        }
    }

    fn for_profile(profile: &registry::ModelProfile, model_body: &[u8]) -> Self {
        const TOKENIZER_BODY: &[u8] = b"verified fixture tokenizer bytes";
        let mut bodies = BTreeMap::new();
        bodies.insert(
            profile.artifact_url(profile.model_artifact()),
            model_body.to_vec(),
        );
        let tokenizer = profile
            .tokenizer_artifact()
            .expect("fixture profiles include a tokenizer artifact");
        bodies.insert(profile.artifact_url(tokenizer), TOKENIZER_BODY.to_vec());
        Self::with_bodies(bodies)
    }

    fn call_count(&self) -> usize {
        self.calls.borrow().len()
    }
}

impl ArtifactFetcher for FakeFetcher {
    fn open(&self, url: &str) -> Result<Box<dyn Read + Send + Sync>> {
        self.calls.borrow_mut().push(url.to_owned());
        let body = self
            .bodies
            .get(url)
            .cloned()
            .ok_or_else(|| SupergrepError::model(format!("fake has no body for {url}")))?;
        Ok(Box::new(Cursor::new(body)))
    }
}

fn fixture_registry(body: &[u8]) -> ModelRegistry {
    ModelRegistry::from_toml_str(&fixture_registry_text(body))
        .expect("fixture registry should parse")
}

fn fixture_registry_text(body: &[u8]) -> String {
    let digest = format!("{:x}", Sha256::digest(body));
    const TOKENIZER_BODY: &[u8] = b"verified fixture tokenizer bytes";
    let tokenizer_digest = format!("{:x}", Sha256::digest(TOKENIZER_BODY));
    format!(
        r#"
format_version = 1

[[profile]]
id = "test-model"
repository = "example-org/test-model"
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
sha256 = "{digest}"
size_bytes = {}

[[profile.artifact]]
kind = "tokenizer"
path = "tokenizer.json"
sha256 = "{tokenizer_digest}"
size_bytes = {}
"#,
        body.len(),
        TOKENIZER_BODY.len()
    )
}

#[test]
fn checked_in_registry_has_exact_known_model_pins_and_contracts() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/registry.toml");
    let registry = ModelRegistry::from_file(&path).expect("checked-in registry should parse");

    let tiny = registry.profile("tiny-en").expect("tiny profile");
    let tiny_model = tiny.model_artifact();
    assert_eq!(tiny.revision(), "233902d25c440f23af6f7d6e94d2946bac0bee0a");
    assert_eq!(tiny_model.size_bytes, 23_200_716);
    assert_eq!(
        tiny_model.sha256,
        "3573b6b9593cb2f75987a31815d409ca3dd8808629118fd20451bb1a5d90cec7"
    );
    assert_eq!(tiny.tokenizer().pad_token_id, 0);
    assert_eq!(tiny.tokenizer().model_max_tokens, 512);
    let tiny_tokenizer = tiny
        .tokenizer_artifact()
        .expect("tiny tokenizer must be content-addressed");
    assert_eq!(tiny_tokenizer.path, Path::new("tokenizer.json"));
    assert_eq!(tiny_tokenizer.size_bytes, 711_396);
    assert_eq!(
        tiny_tokenizer.sha256,
        "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"
    );
    assert_eq!(
        tiny.runtime().required_inputs,
        ["input_ids", "attention_mask", "token_type_ids"]
    );
    assert!(tiny.runtime().optional_inputs.is_empty());
    assert!(tiny
        .artifact_url(tiny_model)
        .contains("/resolve/233902d25c440f23af6f7d6e94d2946bac0bee0a/"));

    let compact = registry
        .profile("compact-multilingual")
        .expect("compact profile");
    let compact_model = compact.model_artifact();
    assert_eq!(
        compact.revision(),
        "1427fd652930e4ba29e8149678df786c240d8825"
    );
    assert_eq!(compact_model.size_bytes, 118_620_017);
    assert_eq!(
        compact_model.sha256,
        "1825907d6c1a9001ff78124780bbde20a614a8c3df3b63409cf3c72c6fe5c8b4"
    );
    assert_eq!(compact.tokenizer().pad_token_id, 1);
    assert_eq!(compact.tokenizer().model_max_tokens, 514);
    assert_eq!(
        compact.runtime().required_inputs,
        ["input_ids", "attention_mask"]
    );
    assert!(compact.runtime().optional_inputs.is_empty());
    assert_eq!(compact.runtime().score_kind, "relevance_logit");
    let compact_tokenizer = compact
        .tokenizer_artifact()
        .expect("compact tokenizer must be content-addressed");
    assert_eq!(compact_tokenizer.path, Path::new("tokenizer.json"));
    assert_eq!(compact_tokenizer.size_bytes, 17_082_660);
    assert_eq!(
        compact_tokenizer.sha256,
        "62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626"
    );
}

#[test]
fn registry_rejects_moving_revisions_and_unsafe_artifact_paths() {
    let body = b"test bytes";
    let valid = fixture_registry(body);
    assert!(valid.profile("test-model").is_ok());

    let digest = format!("{:x}", Sha256::digest(body));
    let invalid = format!(
        r#"
format_version = 1
[[profile]]
id = "test-model"
repository = "example-org/test-model"
revision = "main"
license = "Apache-2.0"
tokenizer_file = "../tokenizer.json"
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
expected_output_count = 1
score_kind = "relevance_logit"
[[profile.artifact]]
kind = "onnx_model"
path = "../escape.onnx"
sha256 = "{digest}"
size_bytes = {}
"#,
        body.len()
    );
    let moving_error = ModelRegistry::from_toml_str(&invalid).expect_err("main must be rejected");
    assert!(moving_error.to_string().contains("revision"));

    let unsafe_path = invalid.replace(
        "revision = \"main\"",
        "revision = \"0123456789abcdef0123456789abcdef01234567\"",
    );
    let path_error =
        ModelRegistry::from_toml_str(&unsafe_path).expect_err("parent traversal must be rejected");
    assert!(path_error.to_string().contains("tokenizer_file"));
}

#[test]
fn registry_rejects_a_tokenizer_artifact_that_does_not_match_the_contract() {
    let body = b"test bytes";
    let digest = format!("{:x}", Sha256::digest(body));
    let mut invalid = fixture_registry_text(body);
    invalid.push_str(&format!(
        r#"

[[profile.artifact]]
kind = "tokenizer"
path = "unexpected-tokenizer.json"
sha256 = "{digest}"
size_bytes = {}
"#,
        body.len()
    ));

    let error = ModelRegistry::from_toml_str(&invalid)
        .expect_err("tokenizer artifact must match tokenizer_file");
    assert!(error.to_string().contains("tokenizer artifact path"));
}

#[test]
fn download_uses_staging_then_verifies_and_reuses_a_valid_cache() {
    let body = b"verified local model bytes";
    let registry = fixture_registry(body);
    let profile = registry.profile("test-model").expect("profile");
    let artifact = profile.model_artifact();
    let fetcher = FakeFetcher::for_profile(profile, body);
    let temporary = tempfile::tempdir().expect("temporary root");
    let cache = ModelCache::new(temporary.path().join("override-cache")).expect("cache");
    let downloader = ModelDownloader::with_fetcher(cache.clone(), fetcher);

    let installed = downloader
        .download(profile)
        .expect("download should verify");
    assert_eq!(installed.directory, cache.profile_dir(profile));
    assert_eq!(
        fs::read(installed.artifact_path(artifact)).expect("model bytes"),
        body
    );
    assert_eq!(downloader.cache().root(), cache.root());
    assert_eq!(downloader.fetcher().call_count(), 2);

    let parent = cache.profile_parent(profile);
    let entries = fs::read_dir(&parent)
        .expect("profile parent")
        .map(|entry| {
            entry
                .expect("directory entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect::<Vec<_>>();
    assert!(
        !entries
            .iter()
            .any(|entry| entry.contains(".staging-") || entry.contains(".part-")),
        "temporary paths must not survive a successful install: {entries:?}"
    );

    let cached = downloader
        .download(profile)
        .expect("valid cache should verify");
    assert_eq!(cached, installed);
    assert_eq!(
        downloader.fetcher().call_count(),
        2,
        "cache hit must not fetch"
    );
}

#[test]
fn bad_download_digest_leaves_no_completed_revision() {
    let expected = b"good";
    let received = b"evil";
    let registry = fixture_registry(expected);
    let profile = registry.profile("test-model").expect("profile");
    let fetcher = FakeFetcher::for_profile(profile, received);
    let temporary = tempfile::tempdir().expect("temporary root");
    let cache = ModelCache::new(temporary.path().join("override-cache")).expect("cache");
    let downloader = ModelDownloader::with_fetcher(cache.clone(), fetcher);

    let error = downloader
        .download(profile)
        .expect_err("digest mismatch must fail");
    assert!(error.to_string().contains("SHA-256 mismatch"));
    assert!(!cache.profile_dir(profile).exists());
    assert_eq!(downloader.fetcher().call_count(), 1);
    let entries = fs::read_dir(cache.profile_parent(profile))
        .expect("profile parent")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect::<Vec<_>>();
    assert!(
        !entries
            .iter()
            .any(|entry| entry.contains(".staging-") || entry.contains(".part-")),
        "failed download must not leave a completion-looking path: {entries:?}"
    );
}

#[test]
fn local_verification_detects_corruption_without_fetching() {
    let body = b"good";
    let registry = fixture_registry(body);
    let profile = registry.profile("test-model").expect("profile");
    let artifact = profile.model_artifact();
    let fetcher = FakeFetcher::for_profile(profile, body);
    let temporary = tempfile::tempdir().expect("temporary root");
    let cache = ModelCache::new(temporary.path().join("override-cache")).expect("cache");
    let downloader = ModelDownloader::with_fetcher(cache.clone(), fetcher);

    downloader.download(profile).expect("initial install");
    fs::write(cache.artifact_path(profile, artifact), b"evil").expect("corrupt artifact");
    let error = download::verify_cached(profile, &cache).expect_err("corruption must fail");
    assert!(error.to_string().contains("digest mismatch"));
    assert_eq!(
        downloader.fetcher().call_count(),
        2,
        "verify must not fetch"
    );
}

#[test]
fn explicit_cache_override_does_not_need_process_environment_mutation() {
    let temporary = tempfile::tempdir().expect("temporary root");
    let override_root = temporary.path().join("chosen-cache");
    let cache = ModelCache::from_override_or_platform_default(Some(override_root.clone()))
        .expect("explicit override");
    assert_eq!(cache.root(), override_root);
}
