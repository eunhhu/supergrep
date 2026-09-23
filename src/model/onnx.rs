use std::{
    collections::HashSet,
    panic::AssertUnwindSafe,
    path::{Path, PathBuf},
    sync::Mutex,
    time::{Duration, Instant},
};

use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};

use crate::{Result, SupergrepError};

use super::{validate_ort_runtime, PairBatch, PairTokenizer, Scorer, TokenizerContract};

#[derive(Debug, Clone)]
pub struct OnnxScorerConfig {
    pub tokenizer_contract: TokenizerContract,
    pub intra_threads: usize,
    pub optimization_level: OnnxOptimizationLevel,
}

/// ONNX graph optimization is an explicit, measurable model-profile choice.
/// The default starts conservatively; S0 records any faster level only after
/// checking that batching preserves ranking/score behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxOptimizationLevel {
    Disable,
    Level1,
    Level2,
    Level3,
}

impl OnnxOptimizationLevel {
    fn as_ort(self) -> GraphOptimizationLevel {
        match self {
            Self::Disable => GraphOptimizationLevel::Disable,
            Self::Level1 => GraphOptimizationLevel::Level1,
            Self::Level2 => GraphOptimizationLevel::Level2,
            Self::Level3 => GraphOptimizationLevel::Level3,
        }
    }
}

impl Default for OnnxScorerConfig {
    fn default() -> Self {
        Self {
            tokenizer_contract: TokenizerContract::default(),
            intra_threads: 2,
            optimization_level: OnnxOptimizationLevel::Level1,
        }
    }
}

/// Graph facts observed from an actual loaded ONNX session.  This is emitted by
/// the S0 probe and later validated against the profile registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnnxSessionContract {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

pub struct OnnxScorer {
    tokenizer: PairTokenizer,
    session: Session,
    graph: OnnxSessionContract,
    model_path: PathBuf,
    timings: Mutex<ScoringTimings>,
}

/// Cumulative model work measured by the production scorer. Tokenization and
/// ONNX execution are separate so CLI benchmarks do not mislabel selection or
/// output time as inference.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScoringTimings {
    pub tokenization: Duration,
    pub inference: Duration,
}

impl OnnxScorer {
    pub fn load(
        model_path: &Path,
        tokenizer_path: &Path,
        runtime_path: &Path,
        config: OnnxScorerConfig,
    ) -> Result<Self> {
        SupergrepError::require_file(model_path)?;
        SupergrepError::require_file(tokenizer_path)?;
        SupergrepError::require_file(runtime_path)?;
        validate_ort_runtime(runtime_path)?;
        if config.intra_threads == 0 {
            return Err(SupergrepError::Input("--threads must be at least 1".into()));
        }

        let tokenizer = PairTokenizer::from_file(tokenizer_path, config.tokenizer_contract)?;
        let model_display = model_path.display().to_string();
        let runtime_display = runtime_path.display().to_string();

        // `ort` rc.9 deliberately loads a dynamic library only when first used.
        // Turn a library/session panic into a CLI-quality error; the explicit
        // existence checks above cover the usual missing-runtime case first.
        let session_result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            ort::init_from(runtime_display)
                .with_name("supergrep")
                .with_telemetry(false)
                .commit()
                .map_err(|error| SupergrepError::runtime(error.to_string()))?;
            Session::builder()
                .map_err(|error| SupergrepError::runtime(error.to_string()))?
                .with_optimization_level(config.optimization_level.as_ort())
                .map_err(|error| SupergrepError::runtime(error.to_string()))?
                .with_intra_threads(config.intra_threads)
                .map_err(|error| SupergrepError::runtime(error.to_string()))?
                .commit_from_file(model_path)
                .map_err(|error| {
                    SupergrepError::model(format!(
                        "could not load ONNX model {model_display}: {error}"
                    ))
                })
        }));
        let session = match session_result {
            Ok(result) => result?,
            Err(_) => {
                return Err(SupergrepError::runtime(format!(
                    "ONNX Runtime failed while loading {}; verify runtime/model ABI compatibility",
                    runtime_path.display()
                )))
            }
        };

        let graph = OnnxSessionContract {
            inputs: session
                .inputs
                .iter()
                .map(|input| input.name.clone())
                .collect(),
            outputs: session
                .outputs
                .iter()
                .map(|output| output.name.clone())
                .collect(),
        };
        validate_graph(&graph)?;
        Ok(Self {
            tokenizer,
            session,
            graph,
            model_path: model_path.to_path_buf(),
            timings: Mutex::new(ScoringTimings::default()),
        })
    }

    pub fn reset_timings(&self) {
        if let Ok(mut timings) = self.timings.lock() {
            *timings = ScoringTimings::default();
        }
    }

    pub fn scoring_timings(&self) -> ScoringTimings {
        self.timings
            .lock()
            .map(|timings| *timings)
            .unwrap_or_default()
    }

    pub fn graph_contract(&self) -> &OnnxSessionContract {
        &self.graph
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub fn tokenizer(&self) -> &PairTokenizer {
        &self.tokenizer
    }

    /// Scores already-tokenized rows. This exists so measurement code can
    /// report tokenization and ONNX execution separately while product search
    /// still uses [`Scorer::score_batch`].
    pub fn score_encoded_batch(&self, batch: PairBatch) -> Result<Vec<f32>> {
        if batch.batch_len == 0 {
            return Ok(Vec::new());
        }
        if batch.sequence_len == 0 {
            return Err(SupergrepError::model(
                "non-empty model batch has zero sequence length",
            ));
        }
        let expected_len = batch
            .batch_len
            .checked_mul(batch.sequence_len)
            .ok_or_else(|| {
                SupergrepError::Input(
                    "batch dimensions overflowed while preparing model inputs".into(),
                )
            })?;
        if [
            batch.input_ids.len(),
            batch.attention_mask.len(),
            batch.token_type_ids.len(),
        ]
        .iter()
        .any(|&length| length != expected_len)
        {
            return Err(SupergrepError::model(
                "tokenizer returned inconsistent batch tensor lengths",
            ));
        }
        let started = Instant::now();
        let result = self.score_pair_batch(batch);
        self.add_inference_time(started.elapsed());
        result
    }

    fn add_tokenization_time(&self, elapsed: Duration) {
        if let Ok(mut timings) = self.timings.lock() {
            timings.tokenization = timings.tokenization.saturating_add(elapsed);
        }
    }

    fn add_inference_time(&self, elapsed: Duration) {
        if let Ok(mut timings) = self.timings.lock() {
            timings.inference = timings.inference.saturating_add(elapsed);
        }
    }

    fn score_pair_batch(&self, batch: PairBatch) -> Result<Vec<f32>> {
        let shape = [batch.batch_len, batch.sequence_len];
        let input_ids = Tensor::from_array((shape, batch.input_ids)).map_err(|error| {
            SupergrepError::runtime(format!("could not build input_ids tensor: {error}"))
        })?;
        let attention_mask =
            Tensor::from_array((shape, batch.attention_mask)).map_err(|error| {
                SupergrepError::runtime(format!("could not build attention_mask tensor: {error}"))
            })?;
        let token_type_ids =
            Tensor::from_array((shape, batch.token_type_ids)).map_err(|error| {
                SupergrepError::runtime(format!("could not build token_type_ids tensor: {error}"))
            })?;

        let required: HashSet<&str> = self.graph.inputs.iter().map(String::as_str).collect();
        let output = if required.contains("token_type_ids") {
            let inputs = ort::inputs! {
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
                "token_type_ids" => token_type_ids,
            }
            .map_err(|error| {
                SupergrepError::runtime(format!("could not build ONNX input map: {error}"))
            })?;
            self.session.run(inputs)
        } else {
            let inputs = ort::inputs! {
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
            }
            .map_err(|error| {
                SupergrepError::runtime(format!("could not build ONNX input map: {error}"))
            })?;
            self.session.run(inputs)
        }
        .map_err(|error| SupergrepError::runtime(format!("ONNX inference failed: {error}")))?;

        if output.len() != 1 {
            return Err(SupergrepError::model(format!(
                "model produced {} outputs; v0.1 expects one relevance-logit output",
                output.len()
            )));
        }
        let (output_shape, logits) =
            output[0].try_extract_raw_tensor::<f32>().map_err(|error| {
                SupergrepError::runtime(format!("could not read ONNX output: {error}"))
            })?;
        if !logits.iter().all(|value| value.is_finite()) {
            return Err(SupergrepError::model(
                "model returned a non-finite relevance logit",
            ));
        }
        if output_shape.len() != 2
            || output_shape[0] != batch.batch_len as i64
            || output_shape[1] != 1
        {
            return Err(SupergrepError::model(format!(
                "unexpected relevance-logit shape {:?}; expected [{}, 1]",
                output_shape, batch.batch_len
            )));
        }
        Ok(logits.to_vec())
    }
}

impl Scorer for OnnxScorer {
    fn score_batch(&self, query: &str, passages: &[String]) -> Result<Vec<f32>> {
        let started = Instant::now();
        let batch = self.tokenizer.encode_batch(query, passages);
        self.add_tokenization_time(started.elapsed());
        let batch = batch?;
        self.score_encoded_batch(batch)
    }
}

fn validate_graph(graph: &OnnxSessionContract) -> Result<()> {
    let inputs: HashSet<&str> = graph.inputs.iter().map(String::as_str).collect();
    for name in ["input_ids", "attention_mask"] {
        if !inputs.contains(name) {
            return Err(SupergrepError::model(format!(
                "ONNX graph is missing required {name} input; found {:?}",
                graph.inputs
            )));
        }
    }
    if graph.outputs.len() != 1 {
        return Err(SupergrepError::model(format!(
            "ONNX graph has {} outputs; v0.1 requires one relevance-logit output",
            graph.outputs.len()
        )));
    }
    Ok(())
}
