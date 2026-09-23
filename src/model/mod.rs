//! Local model loading, tokenization, and scoring.

mod download;
mod loader;
mod onnx;
mod registry;
mod runtime;
mod tokenizer;

pub use download::{
    verify_cached, verify_directory, ArtifactFetcher, HttpArtifactFetcher, ModelCache,
    ModelDownloader, VerifiedArtifact, VerifiedModel,
};
pub use loader::{built_in_registry, resolve_model_spec, validate_graph_contract, ResolvedModel};
pub use onnx::{
    OnnxOptimizationLevel, OnnxScorer, OnnxScorerConfig, OnnxSessionContract, ScoringTimings,
};
pub use registry::{
    ArtifactKind, ModelArtifact, ModelProfile, ModelRegistry, RuntimeContract, TokenizerFamily,
    TokenizerMetadata, REGISTRY_FORMAT_VERSION,
};
pub use runtime::{
    resolve_runtime_library, resolve_runtime_library_with_env, validate_ort_runtime,
    validate_ort_version_string, RuntimeLibrary, RuntimeLibrarySource, BUNDLED_ORT_LIBRARY_FILE,
    ORT_LIBRARY_ENV, REQUIRED_ORT_RUNTIME_VERSION,
};
pub use tokenizer::{PairBatch, PairTokenizer, PreparedPairQuery, TokenizerContract};

use crate::Result;

/// Scores a query against passages in the same order supplied.
///
/// Values are raw model relevance logits, never calibrated probabilities.
pub trait Scorer {
    fn score_batch(&self, query: &str, passages: &[String]) -> Result<Vec<f32>>;
    fn score_kind(&self) -> &'static str {
        "relevance_logit"
    }
}
