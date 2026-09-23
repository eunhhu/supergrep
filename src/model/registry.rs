//! Immutable, revision-pinned metadata for supported local model profiles.
//!
//! Loading a registry is deliberately local-only. Network access belongs to
//! the explicit downloader, never registry lookup or artifact verification.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Component, Path, PathBuf},
};

use serde::Deserialize;

use crate::{Result, SupergrepError};

pub const REGISTRY_FORMAT_VERSION: u32 = 1;

/// The collection of model profiles declared by models/registry.toml.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRegistry {
    profiles: BTreeMap<String, ModelProfile>,
}

/// A single immutable source revision and its locally verifiable artifacts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelProfile {
    id: String,
    repository: String,
    revision: String,
    license: String,
    tokenizer: TokenizerMetadata,
    runtime: RuntimeContract,
    artifacts: Vec<ModelArtifact>,
}

/// Metadata needed to configure the local tokenizer.
///
/// file names the companion tokenizer asset at the same pinned repository
/// revision. A tokenizer becomes downloadable only when a matching tokenizer
/// artifact carries a file-level SHA-256 in the registry; metadata alone never
/// authorizes fetching an unhashed companion file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerMetadata {
    pub file: PathBuf,
    pub family: TokenizerFamily,
    pub pad_token_id: u32,
    pub max_pair_tokens: usize,
    pub max_query_tokens: usize,
    pub model_max_tokens: usize,
    pub config_model_type: String,
    pub config_architecture: String,
    pub token_type_vocab_size: usize,
}

/// The tokenizer family affects special-token and pair encoding conventions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerFamily {
    BertWordPiece,
    XlmRobertaSentencePiece,
}

/// The runtime/graph invariants shared by the model profile and ONNX loader.
///
/// Output names are intentionally not pinned before the real S0 graph probe;
/// the registry instead declares the required input set and one-logit output
/// contract for each profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeContract {
    pub runtime_family: String,
    pub wrapper_version: String,
    pub dynamic_loading: bool,
    pub required_inputs: Vec<String>,
    pub optional_inputs: Vec<String>,
    pub expected_output_count: usize,
    pub score_kind: String,
}

/// A content-addressed file that the downloader may install and verifier may
/// trust.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelArtifact {
    pub kind: ArtifactKind,
    pub path: PathBuf,
    pub sha256: String,
    pub size_bytes: u64,
}

/// The artifact role guards against a profile accidentally pointing inference
/// at a tokenizer or configuration file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactKind {
    OnnxModel,
    Tokenizer,
    Config,
}

impl ModelRegistry {
    /// Parse and validate registry text. A successful return means all model
    /// IDs, revisions, relative paths, sizes, and digests are safe to use.
    pub fn from_toml_str(contents: &str) -> Result<Self> {
        let raw: RawRegistry = toml::from_str(contents).map_err(|error| {
            SupergrepError::model(format!("invalid model registry TOML: {error}"))
        })?;
        if raw.format_version != REGISTRY_FORMAT_VERSION {
            return Err(SupergrepError::model(format!(
                "unsupported model registry format {}; expected {}",
                raw.format_version, REGISTRY_FORMAT_VERSION
            )));
        }
        if raw.profile.is_empty() {
            return Err(SupergrepError::model("model registry contains no profiles"));
        }

        let mut profiles = BTreeMap::new();
        for raw_profile in raw.profile {
            let profile = ModelProfile::try_from(raw_profile)?;
            if profiles.insert(profile.id.clone(), profile).is_some() {
                return Err(SupergrepError::model(
                    "model registry contains duplicate profile IDs",
                ));
            }
        }
        Ok(Self { profiles })
    }

    /// Load a local registry file without making a network request.
    pub fn from_file(path: &Path) -> Result<Self> {
        let contents = fs::read_to_string(path)?;
        Self::from_toml_str(&contents)
    }

    pub fn profile(&self, id: &str) -> Result<&ModelProfile> {
        self.profiles.get(id).ok_or_else(|| {
            SupergrepError::Input(format!(
                "unknown model profile {id:?}; available profiles: {}",
                self.profiles.keys().cloned().collect::<Vec<_>>().join(", ")
            ))
        })
    }

    pub fn profiles(&self) -> impl Iterator<Item = &ModelProfile> {
        self.profiles.values()
    }
}

impl ModelProfile {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn repository(&self) -> &str {
        &self.repository
    }

    pub fn revision(&self) -> &str {
        &self.revision
    }

    pub fn license(&self) -> &str {
        &self.license
    }

    pub fn tokenizer(&self) -> &TokenizerMetadata {
        &self.tokenizer
    }

    pub fn runtime(&self) -> &RuntimeContract {
        &self.runtime
    }

    pub fn artifacts(&self) -> &[ModelArtifact] {
        &self.artifacts
    }

    pub fn model_artifact(&self) -> &ModelArtifact {
        // Validation enforces exactly one model artifact.
        self.artifacts
            .iter()
            .find(|artifact| artifact.kind == ArtifactKind::OnnxModel)
            .expect("validated profiles always include an ONNX model artifact")
    }

    pub fn tokenizer_artifact(&self) -> Option<&ModelArtifact> {
        self.artifacts
            .iter()
            .find(|artifact| artifact.kind == ArtifactKind::Tokenizer)
    }

    /// Build a deterministic, revision-pinned Hugging Face resolve URL.
    ///
    /// An artifact returned by this profile was syntax-checked during registry
    /// parsing, so its interpolation cannot add a query, fragment, parent
    /// traversal, or a different repository/revision.
    pub fn artifact_url(&self, artifact: &ModelArtifact) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}/{}?download=true",
            self.repository,
            self.revision,
            artifact.path.display()
        )
    }

    pub fn source_url(&self) -> String {
        format!("https://huggingface.co/{}", self.repository)
    }
}

#[derive(Debug, Deserialize)]
struct RawRegistry {
    format_version: u32,
    #[serde(default)]
    profile: Vec<RawProfile>,
}

#[derive(Debug, Deserialize)]
struct RawProfile {
    id: String,
    repository: String,
    revision: String,
    license: String,
    tokenizer_file: String,
    tokenizer_family: String,
    tokenizer_pad_token_id: u32,
    max_pair_tokens: usize,
    max_query_tokens: usize,
    tokenizer_model_max_tokens: usize,
    model_type: String,
    model_architecture: String,
    type_vocab_size: usize,
    runtime_family: String,
    runtime_wrapper_version: String,
    dynamic_loading: bool,
    required_inputs: Vec<String>,
    #[serde(default)]
    optional_inputs: Vec<String>,
    expected_output_count: usize,
    score_kind: String,
    #[serde(default)]
    artifact: Vec<RawArtifact>,
}

#[derive(Debug, Deserialize)]
struct RawArtifact {
    kind: String,
    path: String,
    sha256: String,
    size_bytes: u64,
}

impl TryFrom<RawProfile> for ModelProfile {
    type Error = SupergrepError;

    fn try_from(raw: RawProfile) -> Result<Self> {
        validate_profile_id(&raw.id)?;
        validate_repository(&raw.repository)?;
        validate_lower_hex(&raw.revision, 40, "revision")?;
        if raw.license.trim().is_empty() {
            return Err(SupergrepError::model(format!(
                "profile {} has an empty license",
                raw.id
            )));
        }

        let tokenizer = TokenizerMetadata {
            file: parse_relative_path(&raw.tokenizer_file, "tokenizer_file")?,
            family: parse_tokenizer_family(&raw.tokenizer_family)?,
            pad_token_id: raw.tokenizer_pad_token_id,
            max_pair_tokens: positive_usize(raw.max_pair_tokens, "max_pair_tokens")?,
            max_query_tokens: positive_usize(raw.max_query_tokens, "max_query_tokens")?,
            model_max_tokens: positive_usize(
                raw.tokenizer_model_max_tokens,
                "tokenizer_model_max_tokens",
            )?,
            config_model_type: nonempty(&raw.model_type, "model_type")?,
            config_architecture: nonempty(&raw.model_architecture, "model_architecture")?,
            token_type_vocab_size: positive_usize(raw.type_vocab_size, "type_vocab_size")?,
        };
        if tokenizer.max_query_tokens > tokenizer.max_pair_tokens {
            return Err(SupergrepError::model(format!(
                "profile {} has max_query_tokens greater than max_pair_tokens",
                raw.id
            )));
        }
        if tokenizer.max_pair_tokens > tokenizer.model_max_tokens {
            return Err(SupergrepError::model(format!(
                "profile {} has max_pair_tokens greater than tokenizer model capacity",
                raw.id
            )));
        }
        match tokenizer.family {
            TokenizerFamily::BertWordPiece if tokenizer.config_model_type != "bert" => {
                return Err(SupergrepError::model(format!(
                    "profile {} uses bert-wordpiece but config model_type is {}",
                    raw.id, tokenizer.config_model_type
                )));
            }
            TokenizerFamily::XlmRobertaSentencePiece
                if tokenizer.config_model_type != "xlm-roberta" =>
            {
                return Err(SupergrepError::model(format!(
                    "profile {} uses xlm-roberta-sentencepiece but config model_type is {}",
                    raw.id, tokenizer.config_model_type
                )));
            }
            _ => {}
        }

        let runtime = RuntimeContract {
            runtime_family: nonempty(&raw.runtime_family, "runtime_family")?,
            wrapper_version: nonempty(&raw.runtime_wrapper_version, "runtime_wrapper_version")?,
            dynamic_loading: raw.dynamic_loading,
            required_inputs: validate_input_names(&raw.required_inputs, "required_inputs", true)?,
            optional_inputs: validate_input_names(&raw.optional_inputs, "optional_inputs", false)?,
            expected_output_count: positive_usize(
                raw.expected_output_count,
                "expected_output_count",
            )?,
            score_kind: nonempty(&raw.score_kind, "score_kind")?,
        };
        for required in ["input_ids", "attention_mask"] {
            if !runtime
                .required_inputs
                .iter()
                .any(|input| input == required)
            {
                return Err(SupergrepError::model(format!(
                    "profile {} must require ONNX input {required}",
                    raw.id
                )));
            }
        }
        if runtime.expected_output_count != 1 {
            return Err(SupergrepError::model(format!(
                "profile {} must declare exactly one relevance-logit output",
                raw.id
            )));
        }
        if runtime.score_kind != "relevance_logit" {
            return Err(SupergrepError::model(format!(
                "profile {} has unsupported score_kind {}; expected relevance_logit",
                raw.id, runtime.score_kind
            )));
        }
        let required: BTreeSet<_> = runtime.required_inputs.iter().collect();
        if runtime
            .optional_inputs
            .iter()
            .any(|input| required.contains(input))
        {
            return Err(SupergrepError::model(format!(
                "profile {} repeats an ONNX input as required and optional",
                raw.id
            )));
        }

        let mut artifact_paths = BTreeSet::new();
        let mut model_artifact_count = 0usize;
        let mut tokenizer_artifact_count = 0usize;
        let mut artifacts = Vec::with_capacity(raw.artifact.len());
        for raw_artifact in raw.artifact {
            let kind = parse_artifact_kind(&raw_artifact.kind)?;
            if kind == ArtifactKind::OnnxModel {
                model_artifact_count += 1;
            }
            let path = parse_relative_path(&raw_artifact.path, "artifact.path")?;
            if kind == ArtifactKind::Tokenizer {
                tokenizer_artifact_count += 1;
                if path.as_path() != tokenizer.file.as_path() {
                    return Err(SupergrepError::model(format!(
                        "profile {} tokenizer artifact path {} does not match tokenizer_file {}",
                        raw.id,
                        path.display(),
                        tokenizer.file.display()
                    )));
                }
            }
            if !artifact_paths.insert(path.clone()) {
                return Err(SupergrepError::model(format!(
                    "profile {} declares the same artifact path twice: {}",
                    raw.id,
                    path.display()
                )));
            }
            validate_lower_hex(&raw_artifact.sha256, 64, "artifact.sha256")?;
            if raw_artifact.size_bytes == 0 {
                return Err(SupergrepError::model(format!(
                    "profile {} has a zero-byte artifact {}",
                    raw.id,
                    path.display()
                )));
            }
            artifacts.push(ModelArtifact {
                kind,
                path,
                sha256: raw_artifact.sha256,
                size_bytes: raw_artifact.size_bytes,
            });
        }
        if model_artifact_count != 1 {
            return Err(SupergrepError::model(format!(
                "profile {} must declare exactly one ONNX model artifact",
                raw.id
            )));
        }
        if tokenizer_artifact_count != 1 {
            return Err(SupergrepError::model(format!(
                "profile {} must declare exactly one tokenizer artifact",
                raw.id
            )));
        }

        Ok(Self {
            id: raw.id,
            repository: raw.repository,
            revision: raw.revision,
            license: raw.license,
            tokenizer,
            runtime,
            artifacts,
        })
    }
}

fn validate_profile_id(id: &str) -> Result<()> {
    if id.is_empty()
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        return Err(SupergrepError::model(format!(
            "invalid model profile ID {id:?}; use lowercase ASCII letters, digits, and hyphens"
        )));
    }
    Ok(())
}

fn validate_repository(repository: &str) -> Result<()> {
    let parts = repository.split('/').collect::<Vec<_>>();
    if parts.len() != 2
        || parts.iter().any(|part| {
            part.is_empty()
                || matches!(*part, "." | "..")
                || !part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        })
    {
        return Err(SupergrepError::model(format!(
            "invalid Hugging Face repository {repository:?}"
        )));
    }
    Ok(())
}

fn validate_lower_hex(value: &str, expected_len: usize, field: &str) -> Result<()> {
    if value.len() != expected_len
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(SupergrepError::model(format!(
            "{field} must be {expected_len} lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn parse_relative_path(value: &str, field: &str) -> Result<PathBuf> {
    if value.is_empty()
        || value.contains('\\')
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'/'))
    {
        return Err(SupergrepError::model(format!(
            "{field} must be a non-empty URL-safe forward-slash relative path"
        )));
    }
    let path = Path::new(value);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::Prefix(_)
                    | Component::RootDir
                    | Component::CurDir
                    | Component::ParentDir
            )
        })
    {
        return Err(SupergrepError::model(format!(
            "{field} must not contain an absolute, current, or parent path component"
        )));
    }
    Ok(path.to_path_buf())
}

fn parse_tokenizer_family(value: &str) -> Result<TokenizerFamily> {
    match value {
        "bert-wordpiece" => Ok(TokenizerFamily::BertWordPiece),
        "xlm-roberta-sentencepiece" => Ok(TokenizerFamily::XlmRobertaSentencePiece),
        _ => Err(SupergrepError::model(format!(
            "unsupported tokenizer_family {value:?}"
        ))),
    }
}

fn parse_artifact_kind(value: &str) -> Result<ArtifactKind> {
    match value {
        "onnx_model" => Ok(ArtifactKind::OnnxModel),
        "tokenizer" => Ok(ArtifactKind::Tokenizer),
        "config" => Ok(ArtifactKind::Config),
        _ => Err(SupergrepError::model(format!(
            "unsupported artifact kind {value:?}"
        ))),
    }
}

fn positive_usize(value: usize, field: &str) -> Result<usize> {
    if value == 0 {
        return Err(SupergrepError::model(format!("{field} must be at least 1")));
    }
    Ok(value)
}

fn nonempty(value: &str, field: &str) -> Result<String> {
    if value.trim().is_empty() {
        return Err(SupergrepError::model(format!("{field} must not be empty")));
    }
    Ok(value.to_owned())
}

fn validate_input_names(values: &[String], field: &str, required: bool) -> Result<Vec<String>> {
    if required && values.is_empty() {
        return Err(SupergrepError::model(format!("{field} must not be empty")));
    }
    let mut seen = BTreeSet::new();
    for value in values {
        if value.is_empty()
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            return Err(SupergrepError::model(format!(
                "{field} contains an invalid ONNX input name {value:?}"
            )));
        }
        if !seen.insert(value) {
            return Err(SupergrepError::model(format!(
                "{field} contains a duplicate ONNX input name {value:?}"
            )));
        }
    }
    Ok(values.to_vec())
}
