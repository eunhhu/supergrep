//! Resolution of a verified model profile into a local ONNX scorer.
//!
//! This module deliberately has no downloader fallback.  A search either
//! resolves a complete, content-verified local model or returns an actionable
//! error that names the explicit `model download` command.

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use crate::{Result, SupergrepError};

use super::{
    verify_cached, verify_directory, ModelCache, ModelProfile, ModelRegistry,
    OnnxOptimizationLevel, OnnxScorer, OnnxScorerConfig, OnnxSessionContract, TokenizerContract,
    VerifiedModel,
};

/// The built-in, revision-pinned model manifest embedded in every binary.
///
/// The package also ships a readable copy for users, but search never depends
/// on finding that copy in the current directory.
pub fn built_in_registry() -> Result<ModelRegistry> {
    ModelRegistry::from_toml_str(include_str!("../../models/registry.toml"))
}

/// A profile whose artifacts have been verified locally and whose paths are
/// ready for a runtime/ONNX contract check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModel {
    pub profile: ModelProfile,
    pub directory: PathBuf,
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
}

impl ResolvedModel {
    /// Loads the local ONNX scorer after resolving a runtime path.  Loading is
    /// separate from resolution so `model verify`, `model list`, and lexical
    /// search can work without a runtime library.
    pub fn load_scorer(&self, runtime_path: &Path, threads: usize) -> Result<OnnxScorer> {
        let tokenizer = self.profile.tokenizer();
        let scorer = OnnxScorer::load(
            &self.model_path,
            &self.tokenizer_path,
            runtime_path,
            OnnxScorerConfig {
                tokenizer_contract: TokenizerContract {
                    max_pair_tokens: tokenizer.max_pair_tokens,
                    max_query_tokens: tokenizer.max_query_tokens,
                    pad_id: tokenizer.pad_token_id,
                },
                intra_threads: threads,
                optimization_level: OnnxOptimizationLevel::Level3,
            },
        )?;
        validate_graph_contract(&self.profile, scorer.graph_contract())?;
        Ok(scorer)
    }
}

/// Resolves either a registered profile ID from `cache` or a local directory
/// that exactly verifies against one profile in `registry`.  A directory is
/// not trusted merely because it contains an `.onnx` file: its checked-in
/// manifest contract must match every artifact hash/size.
pub fn resolve_model_spec(
    registry: &ModelRegistry,
    cache: &ModelCache,
    specification: &str,
) -> Result<ResolvedModel> {
    if let Ok(profile) = registry.profile(specification) {
        let verified = verify_cached(profile, cache).map_err(|error| match error {
            SupergrepError::MissingArtifact(_) => SupergrepError::model(format!(
                "model profile `{}` is not prepared in {}; run `supergrep model download {}` before model search",
                profile.id(),
                cache.profile_dir(profile).display(),
                profile.id(),
            )),
            other => other,
        })?;
        return resolved_from_verified(profile, verified);
    }

    let directory = Path::new(specification);
    let metadata = fs::symlink_metadata(directory).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            SupergrepError::Input(format!(
                "unknown model profile `{specification}` and local model directory does not exist; available profiles: {}",
                registry.profiles().map(ModelProfile::id).collect::<Vec<_>>().join(", ")
            ))
        } else {
            SupergrepError::Io(error)
        }
    })?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
        return Err(SupergrepError::Input(format!(
            "local model specification `{}` must be a non-symlink directory or registered profile ID",
            directory.display()
        )));
    }

    let mut matches = Vec::new();
    let mut failures = Vec::new();
    for profile in registry.profiles() {
        match verify_directory(profile, directory) {
            Ok(verified) => matches.push((profile, verified)),
            Err(error) => failures.push(format!("{}: {error}", profile.id())),
        }
    }
    match matches.len() {
        1 => {
            let (profile, verified) = matches.pop().expect("length checked");
            resolved_from_verified(profile, verified)
        }
        0 => Err(SupergrepError::model(format!(
            "local model directory `{}` did not verify against any built-in immutable profile; expected a complete revision-pinned model/tokenizer directory. Verification summaries: {}",
            directory.display(),
            failures.join("; ")
        ))),
        _ => Err(SupergrepError::model(format!(
            "local model directory `{}` ambiguously verifies as multiple profiles; use a profile ID",
            directory.display()
        ))),
    }
}

fn resolved_from_verified(
    profile: &ModelProfile,
    verified: VerifiedModel,
) -> Result<ResolvedModel> {
    let tokenizer_artifact = profile.tokenizer_artifact().ok_or_else(|| {
        SupergrepError::model(format!(
            "profile `{}` has no content-addressed tokenizer artifact in the registry and cannot be loaded safely",
            profile.id()
        ))
    })?;
    let model_path = verified.artifact_path(profile.model_artifact());
    let tokenizer_path = verified.artifact_path(tokenizer_artifact);
    // `verify_directory` checked every declared artifact. Retain explicit file
    // checks here so an incomplete registry can never cause a tokenizer to be
    // loaded by accident from an unverified companion path.
    SupergrepError::require_file(&model_path)?;
    SupergrepError::require_file(&tokenizer_path)?;
    Ok(ResolvedModel {
        profile: profile.clone(),
        directory: verified.directory,
        model_path,
        tokenizer_path,
    })
}

/// Validates actual loaded graph names/count against the immutable profile
/// declaration before any result is returned to a user.
pub fn validate_graph_contract(profile: &ModelProfile, graph: &OnnxSessionContract) -> Result<()> {
    let expected_inputs = profile
        .runtime()
        .required_inputs
        .iter()
        .chain(profile.runtime().optional_inputs.iter())
        .cloned()
        .collect::<BTreeSet<_>>();
    let actual_inputs = graph.inputs.iter().cloned().collect::<BTreeSet<_>>();
    if actual_inputs != expected_inputs {
        return Err(SupergrepError::model(format!(
            "ONNX graph inputs for profile `{}` do not match its manifest: expected {:?}, got {:?}",
            profile.id(),
            expected_inputs,
            actual_inputs
        )));
    }
    if graph.outputs.len() != profile.runtime().expected_output_count {
        return Err(SupergrepError::model(format!(
            "ONNX graph output count for profile `{}` is {}; manifest requires {}",
            profile.id(),
            graph.outputs.len(),
            profile.runtime().expected_output_count
        )));
    }
    Ok(())
}
