//! S0 evidence-pool comparison for revision-pinned models.
//!
//! This is intentionally an experiment, not the product evaluator.  It reads
//! queries and evidence spans from the fixed development corpus, builds a pool
//! of relevant and judged-near-miss passages, and calls the real `OnnxScorer`.
//! No query text, answer path, or translation is embedded in product code.

use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use supergrep::{
    model::{OnnxOptimizationLevel, OnnxScorer, OnnxScorerConfig, Scorer, TokenizerContract},
    Result, SupergrepError,
};

#[derive(Debug, Parser)]
#[command(about = "Compare a real ONNX cross-encoder on fixed S0 evidence pools")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    #[arg(long)]
    runtime: PathBuf,
    #[arg(long, default_value = "eval/queries.jsonl")]
    queries: PathBuf,
    #[arg(long, default_value = "eval/corpus")]
    corpus: PathBuf,
    #[arg(long, default_value = "development")]
    split: String,
    /// Restrict the experimental pool to one query language. The product does
    /// not use this filter; it lets S0 compare an English-only baseline fairly.
    #[arg(long)]
    language: Option<String>,
    #[arg(long, default_value_t = 4)]
    batch_size: usize,
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
}

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

#[derive(Debug, Deserialize)]
struct QueryRecord {
    schema_version: u32,
    record_type: String,
    id: String,
    intent_id: String,
    split: String,
    language: String,
    query_type: String,
    query: String,
    #[serde(default)]
    tags: Vec<String>,
    labels: Vec<Label>,
}

#[derive(Debug, Clone, Deserialize)]
struct Label {
    evidence_id: String,
    path: String,
    start_byte: usize,
    end_byte: usize,
    start_line: usize,
    end_line: usize,
    relevance: u8,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
struct SpanKey {
    path: String,
    start_byte: usize,
    end_byte: usize,
}

#[derive(Debug, Clone)]
struct Candidate {
    evidence_id: String,
    key: SpanKey,
    start_line: usize,
    end_line: usize,
    text: String,
}

#[derive(Debug, Serialize)]
struct RankedCandidate {
    rank: usize,
    evidence_id: String,
    path: String,
    start_byte: usize,
    end_byte: usize,
    start_line: usize,
    end_line: usize,
    score: f32,
    relevant: bool,
}

#[derive(Debug, Serialize)]
struct QueryResult {
    query_id: String,
    intent_id: String,
    language: String,
    tags: Vec<String>,
    hit_at_5: bool,
    reciprocal_rank_at_10: f64,
    top_results: Vec<RankedCandidate>,
}

#[derive(Debug, Default, Serialize)]
struct GroupMetrics {
    query_count: usize,
    hit_at_5: f64,
    mrr_at_10: f64,
}

#[derive(Debug, Serialize)]
struct Summary {
    schema_version: u8,
    record_type: &'static str,
    split: String,
    score_kind: &'static str,
    model_path: String,
    graph_inputs: Vec<String>,
    graph_outputs: Vec<String>,
    candidate_count: usize,
    model_load_ms: u128,
    total_scoring_ms: u128,
    metrics: BTreeMap<String, GroupMetrics>,
    per_query: Vec<QueryResult>,
    caveat: &'static str,
}

fn parse_queries(path: &Path, split: &str, language: Option<&str>) -> Result<Vec<QueryRecord>> {
    let content = fs::read_to_string(path)?;
    let mut records = Vec::new();
    for (line_number, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let record: QueryRecord = serde_json::from_str(line).map_err(|error| {
            SupergrepError::Input(format!(
                "invalid query JSON at {}:{}: {error}",
                path.display(),
                line_number + 1
            ))
        })?;
        if record.schema_version != 1 || record.record_type != "query" {
            return Err(SupergrepError::Input(format!(
                "unsupported query record at {}:{}",
                path.display(),
                line_number + 1
            )));
        }
        if record.split == split
            && record.query_type == "relevant"
            && language.map_or(true, |language| record.language == language)
        {
            records.push(record);
        }
    }
    if records.is_empty() {
        return Err(SupergrepError::Input(format!(
            "no relevant {split} queries in {}",
            path.display()
        )));
    }
    Ok(records)
}

fn candidate_pool(records: &[QueryRecord], corpus: &Path) -> Result<Vec<Candidate>> {
    let mut labels = BTreeMap::<SpanKey, Label>::new();
    for record in records {
        for label in &record.labels {
            let key = SpanKey {
                path: label.path.clone(),
                start_byte: label.start_byte,
                end_byte: label.end_byte,
            };
            labels.entry(key).or_insert_with(|| label.clone());
        }
    }

    let mut source_cache = HashMap::<String, String>::new();
    let mut candidates = Vec::with_capacity(labels.len());
    for (key, label) in labels {
        let text = match source_cache.entry(key.path.clone()) {
            std::collections::hash_map::Entry::Occupied(value) => value.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let path = corpus.join(&key.path);
                let bytes = fs::read(&path)?;
                let text = String::from_utf8(bytes).map_err(|error| {
                    SupergrepError::Input(format!(
                        "evaluation corpus {} is not UTF-8: {error}",
                        path.display()
                    ))
                })?;
                entry.insert(text)
            }
        };
        let span = text
            .as_bytes()
            .get(key.start_byte..key.end_byte)
            .ok_or_else(|| {
                SupergrepError::Input(format!(
                    "evaluation span {}:{}..{} is outside corpus source",
                    key.path, key.start_byte, key.end_byte
                ))
            })?;
        let passage = std::str::from_utf8(span).map_err(|error| {
            SupergrepError::Input(format!(
                "evaluation span {} is not UTF-8 aligned: {error}",
                key.path
            ))
        })?;
        candidates.push(Candidate {
            evidence_id: label.evidence_id,
            key,
            start_line: label.start_line,
            end_line: label.end_line,
            text: passage.to_owned(),
        });
    }
    Ok(candidates)
}

fn add_metric(group: &mut GroupMetrics, hit: bool, reciprocal_rank: f64) {
    group.query_count += 1;
    group.hit_at_5 += f64::from(hit);
    group.mrr_at_10 += reciprocal_rank;
}

fn finish_metrics(metrics: &mut BTreeMap<String, GroupMetrics>) {
    for metric in metrics.values_mut() {
        if metric.query_count > 0 {
            let count = metric.query_count as f64;
            metric.hit_at_5 /= count;
            metric.mrr_at_10 /= count;
        }
    }
}

fn run() -> Result<()> {
    let args = Args::parse();
    if args.batch_size == 0 {
        return Err(SupergrepError::Input(
            "--batch-size must be at least 1".into(),
        ));
    }
    let records = parse_queries(&args.queries, &args.split, args.language.as_deref())?;
    let candidates = candidate_pool(&records, &args.corpus)?;
    let passages = candidates
        .iter()
        .map(|candidate| candidate.text.clone())
        .collect::<Vec<_>>();

    let load_started = Instant::now();
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
            optimization_level: args.optimization.into(),
        },
    )?;
    let model_load_ms = load_started.elapsed().as_millis();
    let scoring_started = Instant::now();
    let mut metrics = BTreeMap::<String, GroupMetrics>::new();
    let mut per_query = Vec::with_capacity(records.len());

    for record in records {
        let mut scores = Vec::with_capacity(passages.len());
        for batch in passages.chunks(args.batch_size) {
            scores.extend(scorer.score_batch(&record.query, batch)?);
        }
        if scores.len() != candidates.len() {
            return Err(SupergrepError::Internal(
                "model score count did not match candidate count".into(),
            ));
        }
        let relevant = record
            .labels
            .iter()
            .filter(|label| label.relevance > 0)
            .map(|label| SpanKey {
                path: label.path.clone(),
                start_byte: label.start_byte,
                end_byte: label.end_byte,
            })
            .collect::<std::collections::HashSet<_>>();
        let mut order = (0..candidates.len()).collect::<Vec<_>>();
        order.sort_by(|&left, &right| {
            scores[right]
                .total_cmp(&scores[left])
                .then_with(|| candidates[left].key.cmp(&candidates[right].key))
        });
        let hit_at_5 = order
            .iter()
            .take(5)
            .any(|&index| relevant.contains(&candidates[index].key));
        let reciprocal_rank_at_10 = order
            .iter()
            .take(10)
            .position(|&index| relevant.contains(&candidates[index].key))
            .map_or(0.0, |index| 1.0 / (index + 1) as f64);
        let top_results = order
            .iter()
            .take(10)
            .enumerate()
            .map(|(rank, &index)| RankedCandidate {
                rank: rank + 1,
                evidence_id: candidates[index].evidence_id.clone(),
                path: candidates[index].key.path.clone(),
                start_byte: candidates[index].key.start_byte,
                end_byte: candidates[index].key.end_byte,
                start_line: candidates[index].start_line,
                end_line: candidates[index].end_line,
                score: scores[index],
                relevant: relevant.contains(&candidates[index].key),
            })
            .collect();
        add_metric(
            metrics.entry("all".into()).or_default(),
            hit_at_5,
            reciprocal_rank_at_10,
        );
        add_metric(
            metrics
                .entry(format!("language:{}", record.language))
                .or_default(),
            hit_at_5,
            reciprocal_rank_at_10,
        );
        if record.tags.iter().any(|tag| tag == "korean_to_english") {
            add_metric(
                metrics.entry("tag:korean_to_english".into()).or_default(),
                hit_at_5,
                reciprocal_rank_at_10,
            );
        }
        per_query.push(QueryResult {
            query_id: record.id,
            intent_id: record.intent_id,
            language: record.language,
            tags: record.tags,
            hit_at_5,
            reciprocal_rank_at_10,
            top_results,
        });
    }
    let total_scoring_ms = scoring_started.elapsed().as_millis();
    finish_metrics(&mut metrics);
    let output = Summary {
        schema_version: 1,
        record_type: "s0_evidence_pool_comparison",
        split: args.split,
        score_kind: scorer.score_kind(),
        model_path: scorer.model_path().display().to_string(),
        graph_inputs: scorer.graph_contract().inputs.clone(),
        graph_outputs: scorer.graph_contract().outputs.clone(),
        candidate_count: candidates.len(),
        model_load_ms,
        total_scoring_ms,
        metrics,
        per_query,
        caveat: "This S0 evidence-pool ranking checks tokenizer/runtime/model behavior on fixed, judged spans. It is not the final chunk-level fast/deep product evaluation.",
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&output)
            .map_err(|error| SupergrepError::Internal(error.to_string()))?
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("s0-compare: {error}");
        std::process::exit(2);
    }
}
