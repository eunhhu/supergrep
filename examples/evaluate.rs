//! Run the production discovery, chunking, lexical, candidate, and ONNX paths
//! over the fixed evaluation corpus and emit span-auditable JSONL.

use std::{
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
};

use clap::{Parser, ValueEnum};
use serde::Deserialize;
use serde_json::{json, Value};
use supergrep::{
    chunk::{chunk_sources, ChunkConfig},
    discovery::{discover, DiscoveryOptions},
    fitting::fit_chunks_parallel,
    model::{
        built_in_registry, resolve_model_spec, resolve_runtime_library, ModelCache,
        OnnxOptimizationLevel, OnnxScorer, OnnxScorerConfig, TokenizerContract,
    },
    search::{SearchConfig, SearchEngine, SearchMode, SearchReport},
    source::Source,
};

#[derive(Debug, Parser)]
#[command(about = "Emit production-engine evaluation results as span-auditable JSONL")]
struct Args {
    #[arg(long, value_enum)]
    split: Split,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "compact-multilingual")]
    model: String,
    /// Optional direct ONNX path for development-only model comparisons.
    #[arg(long, requires = "tokenizer_path")]
    onnx_model: Option<PathBuf>,
    #[arg(long, requires = "onnx_model")]
    tokenizer_path: Option<PathBuf>,
    #[arg(long, requires = "onnx_model")]
    model_id: Option<String>,
    #[arg(long, requires = "onnx_model")]
    model_revision: Option<String>,
    #[arg(long, default_value_t = 256)]
    max_pair_tokens: usize,
    #[arg(long, default_value_t = 64)]
    max_query_tokens: usize,
    #[arg(long, default_value_t = 1)]
    pad_id: u32,
    #[arg(long, default_value_t = 4)]
    threads: usize,
    #[arg(long)]
    cache_dir: Option<PathBuf>,
    #[arg(long)]
    runtime: Option<PathBuf>,
    #[arg(long, default_value = "eval/queries.jsonl")]
    queries: PathBuf,
    #[arg(long, default_value = "eval/corpus")]
    corpus: PathBuf,
    #[arg(long, default_value_t = 128)]
    candidates: usize,
    #[arg(long, default_value_t = 10)]
    top_k: usize,
    #[arg(long, default_value_t = 1)]
    batch_size: usize,
    /// Optional development-only query-language subset.
    #[arg(long, value_parser = ["en", "ko"])]
    language: Option<String>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Split {
    Development,
    Holdout,
    Irrelevant,
}

impl Split {
    fn as_str(self) -> &'static str {
        match self {
            Self::Development => "development",
            Self::Holdout => "holdout",
            Self::Irrelevant => "irrelevant",
        }
    }
}

#[derive(Debug, Deserialize)]
struct QueryRecord {
    schema_version: u32,
    record_type: String,
    id: String,
    split: String,
    language: String,
    query: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.candidates == 0 || args.top_k == 0 || args.batch_size == 0 {
        return Err("--candidates, --top-k, and --batch-size must be at least one".into());
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let corpus_root = absolute_from(&manifest_dir, &args.corpus)?;
    let query_file = absolute_from(&manifest_dir, &args.queries)?;
    let corpus = discover(&DiscoveryOptions {
        root: corpus_root.clone(),
        include_hidden: false,
        respect_ignore: false,
        globs: Vec::new(),
        max_file_size: 2 * 1024 * 1024,
        max_total_bytes: 64 * 1024 * 1024,
        max_chunks: 50_000,
    })?;
    if corpus.partial || !corpus.scan_complete {
        return Err(format!(
            "evaluation corpus scan was incomplete: {} failures, {} bytes read",
            corpus.stats.failures, corpus.stats.bytes_read
        )
        .into());
    }
    let coarse_chunks = chunk_sources(&corpus.sources, ChunkConfig::default(), 50_000)?;
    if coarse_chunks.limit_reached {
        return Err("evaluation corpus exceeded the 50,000 chunk limit".into());
    }

    let runtime = match args.runtime.clone() {
        Some(path) => path,
        None => resolve_runtime_library(&std::env::current_exe()?)?.path,
    };
    if args.threads == 0 || args.max_pair_tokens == 0 || args.max_query_tokens == 0 {
        return Err("--threads and tokenizer token limits must be at least one".into());
    }
    let (scorer, model_id, model_revision) = match (&args.onnx_model, &args.tokenizer_path) {
        (Some(model_path), Some(tokenizer_path)) => {
            let model_id = args
                .model_id
                .as_deref()
                .ok_or("--model-id is required with --onnx-model")?;
            let model_revision = args
                .model_revision
                .as_deref()
                .ok_or("--model-revision is required with --onnx-model")?;
            let scorer = OnnxScorer::load(
                model_path,
                tokenizer_path,
                &runtime,
                OnnxScorerConfig {
                    tokenizer_contract: TokenizerContract {
                        max_pair_tokens: args.max_pair_tokens,
                        max_query_tokens: args.max_query_tokens,
                        pad_id: args.pad_id,
                    },
                    intra_threads: args.threads,
                    optimization_level: OnnxOptimizationLevel::Level3,
                },
            )?;
            (scorer, model_id.to_owned(), model_revision.to_owned())
        }
        (None, None) => {
            let registry = built_in_registry()?;
            let cache = ModelCache::from_override_or_platform_default(args.cache_dir.clone())?;
            let resolved = resolve_model_spec(&registry, &cache, &args.model)?;
            let id = resolved.profile.id().to_owned();
            let revision = resolved.profile.revision().to_owned();
            (resolved.load_scorer(&runtime, args.threads)?, id, revision)
        }
        _ => return Err("--onnx-model and --tokenizer-path must be supplied together".into()),
    };

    let queries = BufReader::new(File::open(&query_file)?)
        .lines()
        .enumerate()
        .filter_map(|(index, line)| match line {
            Ok(line) if line.trim().is_empty() => None,
            Ok(line) => Some(Ok((index + 1, line))),
            Err(error) => Some(Err(error)),
        })
        .map(|line| {
            let (line_number, line) = line?;
            let record: QueryRecord = serde_json::from_str(&line).map_err(|error| {
                format!(
                    "could not parse {}:{line_number}: {error}",
                    query_file.display()
                )
            })?;
            if record.schema_version != 1 || record.record_type != "query" {
                return Err(format!(
                    "{}:{line_number}: unsupported query record",
                    query_file.display()
                )
                .into());
            }
            Ok(record)
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let selected = queries
        .into_iter()
        .filter(|record| {
            record.split == args.split.as_str()
                && args
                    .language
                    .as_deref()
                    .map_or(true, |language| record.language == language)
        })
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err(format!("no query records found for split {}", args.split.as_str()).into());
    }

    let output_path = if args.output.is_absolute() {
        args.output.clone()
    } else {
        std::env::current_dir()?.join(&args.output)
    };
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let output_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output_path)
        .map_err(|error| {
            format!(
                "could not create new output file {} without overwriting an existing artifact: {error}",
                output_path.display()
            )
        })?;
    let mut writer = BufWriter::new(output_file);

    for query_record in selected {
        scorer
            .tokenizer()
            .validate_query(&query_record.query)
            .map_err(|error| {
                format!(
                    "query {} exceeds the model tokenizer contract: {error}",
                    query_record.id
                )
            })?;
        let prepared_query = scorer.tokenizer().prepare_pair_query(&query_record.query)?;
        let fit = fit_chunks_parallel(
            &corpus.sources,
            &coarse_chunks.chunks,
            &query_record.query,
            scorer.tokenizer().contract().max_pair_tokens,
            50_000,
            |_, passage| prepared_query.token_count(passage),
        )?;
        if fit.limit_reached {
            return Err(format!(
                "query {} exceeded the 50,000 fitted chunk limit",
                query_record.id
            )
            .into());
        }
        let engine = SearchEngine::new(&corpus.sources, &fit.chunks)?;
        let lexical_report = engine.search(
            &query_record.query,
            SearchConfig {
                mode: SearchMode::Lexical,
                candidate_limit: args.candidates,
                top_k: args.top_k,
                batch_size: args.batch_size,
            },
            Some(&scorer),
        )?;
        let fast_report = engine.search(
            &query_record.query,
            SearchConfig {
                mode: SearchMode::Fast,
                candidate_limit: args.candidates,
                top_k: args.top_k,
                batch_size: args.batch_size,
            },
            Some(&scorer),
        )?;
        let mut deep_report = if fast_report.scoring_complete == Some(true) {
            // Fast has scored the whole fixed corpus; Deep is equivalent and
            // can reuse its identical full-set ranking without doubling the
            // benchmark inference workload.
            fast_report.clone()
        } else {
            engine.search(
                &query_record.query,
                SearchConfig {
                    mode: SearchMode::Deep,
                    candidate_limit: args.candidates,
                    top_k: args.top_k,
                    batch_size: args.batch_size,
                },
                Some(&scorer),
            )?
        };
        deep_report.mode = SearchMode::Deep;
        deep_report.scoring_complete = Some(true);
        let context = EvalContext {
            args: &args,
            model_id: &model_id,
            model_revision: &model_revision,
            corpus_root: &corpus_root,
            sources: &corpus.sources,
            chunks: &fit.chunks,
        };
        for report in [&lexical_report, &fast_report, &deep_report] {
            let mode = report.mode;
            let event = make_event(&query_record, &context, report)?;
            serde_json::to_writer(&mut writer, &event)?;
            writer.write_all(b"\n")?;
            eprintln!(
                "evaluated {} ({mode_name})",
                query_record.id,
                mode_name = mode_name(mode)
            );
        }
    }
    writer.flush()?;
    eprintln!("wrote evaluation results to {}", output_path.display());
    Ok(())
}

struct EvalContext<'a> {
    args: &'a Args,
    model_id: &'a str,
    model_revision: &'a str,
    corpus_root: &'a Path,
    sources: &'a [Source],
    chunks: &'a [supergrep::chunk::Chunk],
}

fn make_event(
    query: &QueryRecord,
    context: &EvalContext<'_>,
    report: &SearchReport,
) -> Result<Value, Box<dyn std::error::Error>> {
    let all_chunk_spans = (0..context.chunks.len())
        .map(|index| chunk_span(index, context.chunks, context.sources, context.corpus_root))
        .collect::<Result<Vec<_>, _>>()?;
    let selected_spans = report
        .selected_chunk_indexes
        .iter()
        .map(|&index| chunk_span(index, context.chunks, context.sources, context.corpus_root))
        .collect::<Result<Vec<_>, _>>()?;
    let evaluated_spans = report
        .evaluated_chunk_indexes
        .iter()
        .map(|&index| chunk_span(index, context.chunks, context.sources, context.corpus_root))
        .collect::<Result<Vec<_>, _>>()?;
    let results = report
        .results
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            Ok(json!({
                "rank": index + 1,
                "path": relative_path(&entry.path, context.corpus_root)?,
                "start_byte": entry.start_byte,
                "end_byte": entry.end_byte,
                "score": entry.relevance_logit.map(f64::from).or(entry.lexical_score),
                "score_kind": if entry.relevance_logit.is_some() { "raw_logit" } else { "bm25" },
            }))
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;

    let mode = mode_name(report.mode);
    let mut event = json!({
        "schema_version": 1,
        "record_type": "evaluation_result",
        "query_id": query.id,
        "mode": mode,
        "chunk_contract_id": "model-chunks-v1",
        "model_profile": context.model_id,
        "model_revision": context.model_revision,
        "candidate_limit": context.args.candidates,
        "all_chunk_spans": all_chunk_spans,
        "candidate_spans": selected_spans,
        "evaluated_spans": evaluated_spans,
        "scoring_complete": report.scoring_complete,
        "results": results,
    });
    if report.mode == SearchMode::Lexical {
        event
            .as_object_mut()
            .expect("JSON event object")
            .remove("candidate_limit");
    }
    Ok(event)
}

fn chunk_span(
    index: usize,
    chunks: &[supergrep::chunk::Chunk],
    sources: &[Source],
    corpus_root: &Path,
) -> Result<Value, Box<dyn std::error::Error>> {
    let chunk = chunks
        .get(index)
        .ok_or_else(|| format!("chunk index {index} is out of bounds"))?;
    let source = sources
        .iter()
        .find(|source| source.id() == chunk.source_id)
        .ok_or_else(|| format!("chunk index {index} references a missing source"))?;
    Ok(json!({
        "path": relative_path(source.path(), corpus_root)?,
        "start_byte": chunk.start_byte,
        "end_byte": chunk.end_byte,
    }))
}

fn relative_path(path: &Path, root: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let relative = path.strip_prefix(root).map_err(|_| {
        format!(
            "evaluated path {} is outside corpus {}",
            path.display(),
            root.display()
        )
    })?;
    Ok(relative
        .to_str()
        .ok_or_else(|| format!("evaluation path is not UTF-8: {}", relative.display()))?
        .replace('\\', "/"))
}

fn absolute_from(base: &Path, path: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    };
    Ok(resolved.canonicalize()?)
}

fn mode_name(mode: SearchMode) -> &'static str {
    match mode {
        SearchMode::Lexical => "lexical",
        SearchMode::Fast => "fast",
        SearchMode::Deep => "deep",
    }
}
