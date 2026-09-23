//! A deliberately small, real-model S0 experiment runner.  It is not a
//! product command; later stages use the same `OnnxScorer` implementation.

use std::{path::PathBuf, time::Instant};

use clap::{Parser, ValueEnum};
use supergrep::{
    model::{OnnxOptimizationLevel, OnnxScorer, OnnxScorerConfig, TokenizerContract},
    Result,
};

#[derive(Debug, Clone, ValueEnum)]
enum OptimizationArg {
    Disable,
    Level1,
    Level2,
    Level3,
}

impl From<OptimizationArg> for OnnxOptimizationLevel {
    fn from(value: OptimizationArg) -> Self {
        match value {
            OptimizationArg::Disable => Self::Disable,
            OptimizationArg::Level1 => Self::Level1,
            OptimizationArg::Level2 => Self::Level2,
            OptimizationArg::Level3 => Self::Level3,
        }
    }
}

#[derive(Debug, Parser)]
#[command(name = "s0-probe", about = "Run a local ONNX cross-encoder batch")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    #[arg(long)]
    runtime: PathBuf,
    #[arg(long, default_value_t = 2)]
    threads: usize,
    #[arg(long, default_value_t = 256)]
    max_pair_tokens: usize,
    #[arg(long, default_value_t = 64)]
    max_query_tokens: usize,
    #[arg(long, default_value_t = 0)]
    pad_id: u32,
    #[arg(long, value_enum, default_value_t = OptimizationArg::Level1)]
    optimization: OptimizationArg,
    /// Repeat every positional passage without changing the query. This keeps
    /// one loaded session for fixed-length throughput experiments.
    #[arg(long, default_value_t = 1)]
    repeat_each: usize,
    /// Split the supplied passages into model calls of this size. Zero means
    /// one model call containing all supplied passages.
    #[arg(long, default_value_t = 0)]
    score_batch_size: usize,
    #[arg(long)]
    query: String,
    #[arg(required = true)]
    passages: Vec<String>,
}

fn run() -> Result<()> {
    let args = Args::parse();
    if args.repeat_each == 0 {
        return Err(supergrep::SupergrepError::Input(
            "--repeat-each must be at least 1".into(),
        ));
    }
    let passages = args
        .passages
        .iter()
        .flat_map(|passage| std::iter::repeat_with(|| passage.clone()).take(args.repeat_each))
        .collect::<Vec<_>>();
    let score_batch_size = if args.score_batch_size == 0 {
        passages.len()
    } else {
        args.score_batch_size
    };
    let loaded = Instant::now();
    let scorer = OnnxScorer::load(
        &args.model,
        &args.tokenizer,
        &args.runtime,
        OnnxScorerConfig {
            tokenizer_contract: TokenizerContract {
                max_pair_tokens: args.max_pair_tokens,
                max_query_tokens: args.max_query_tokens,
                pad_id: args.pad_id,
            },
            intra_threads: args.threads,
            optimization_level: args.optimization.clone().into(),
        },
    )?;
    let load_elapsed = loaded.elapsed();
    let mut tokenization_elapsed = std::time::Duration::ZERO;
    let mut inference_elapsed = std::time::Duration::ZERO;
    let mut scores = Vec::with_capacity(passages.len());
    let mut max_sequence_len = 0usize;
    let mut model_calls = 0usize;
    for score_batch in passages.chunks(score_batch_size) {
        let tokenized = Instant::now();
        let batch = scorer.tokenizer().encode_batch(&args.query, score_batch)?;
        tokenization_elapsed += tokenized.elapsed();
        max_sequence_len = max_sequence_len.max(batch.sequence_len);
        let inferred = Instant::now();
        scores.extend(scorer.score_encoded_batch(batch)?);
        inference_elapsed += inferred.elapsed();
        model_calls += 1;
    }
    println!("model={}", scorer.model_path().display());
    println!("inputs={:?}", scorer.graph_contract().inputs);
    println!("outputs={:?}", scorer.graph_contract().outputs);
    println!("optimization={:?}", args.optimization);
    println!("load_ms={}", load_elapsed.as_millis());
    println!("tokenization_ms={}", tokenization_elapsed.as_millis());
    println!("inference_ms={}", inference_elapsed.as_millis());
    println!("passage_count={}", passages.len());
    println!("score_batch_size={score_batch_size}");
    println!("model_calls={model_calls}");
    println!("max_sequence_len={max_sequence_len}");
    if let Some(rss_kib) = proc_status_kib("VmRSS:") {
        println!("rss_kib={rss_kib}");
    }
    if let Some(peak_rss_kib) = proc_status_kib("VmHWM:") {
        println!("peak_rss_kib={peak_rss_kib}");
    }
    if scores.len() <= 16 {
        for (index, score) in scores.iter().enumerate() {
            println!("score[{index}]={score:.7}");
        }
    } else {
        println!("scores_omitted={}", scores.len());
    }
    Ok(())
}

/// Linux-only process-memory observation for S0.  The caller records it as an
/// observed host result, not as a portable promise about model memory use.
fn proc_status_kib(field: &str) -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        let rest = line.strip_prefix(field)?;
        rest.split_whitespace().next()?.parse::<u64>().ok()
    })
}

fn main() {
    if let Err(error) = run() {
        eprintln!("s0-probe: {error}");
        std::process::exit(2);
    }
}
