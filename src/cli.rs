//! User-facing command-line workflow.

use std::{io::Write, path::PathBuf, time::Instant};

use clap::{Args, Parser, Subcommand};
use serde_json::json;

use crate::{
    chunk::{chunk_sources, ChunkConfig},
    discovery::{discover, DiscoveryDiagnosticKind, DiscoveryOptions},
    fitting::fit_chunks,
    model::{
        built_in_registry, resolve_model_spec, resolve_runtime_library, validate_ort_runtime,
        verify_cached, ModelCache, ModelDownloader,
    },
    output::{
        render_human, render_json, ModelMetadata, ModelScoreKind, PresentationDiagnostic,
        PresentationInput, PresentationStats, PresentationTiming,
    },
    search::{SearchConfig, SearchEngine, SearchMode},
    Result, SupergrepError,
};

const DEFAULT_MODEL: &str = "compact-multilingual";
const DEFAULT_CANDIDATES: usize = 128;
const DEFAULT_TOP_K: usize = 10;
const DEFAULT_BATCH_SIZE: usize = 1;
const DEFAULT_THREADS: usize = 4;

#[derive(Debug, Parser)]
#[command(
    name = "supergrep",
    version,
    about = "Search UTF-8 project text with a local natural-language model",
    long_about = "Search code, documentation, and configuration using a local ONNX model. Search never downloads models or calls a network service. Run `supergrep model download compact-multilingual` once before semantic search; use --lexical for model-free search."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Natural-language query. Use `--` before it when it begins with `-` or
    /// matches a subcommand name.
    #[arg(value_name = "QUERY")]
    query: Option<String>,

    /// Search root (file or directory; defaults to the current directory).
    #[arg(value_name = "ROOT")]
    root: Option<PathBuf>,

    #[arg(long, group = "mode", conflicts_with = "lexical")]
    deep: bool,

    #[arg(long, group = "mode", conflicts_with = "deep")]
    lexical: bool,

    #[arg(long, value_name = "PROFILE_OR_DIRECTORY")]
    model: Option<String>,

    #[arg(long, default_value_t = DEFAULT_TOP_K)]
    top_k: usize,

    #[arg(long, value_name = "N")]
    candidates: Option<usize>,

    #[arg(long, default_value_t = DEFAULT_BATCH_SIZE)]
    batch_size: usize,

    #[arg(long)]
    json: bool,

    #[arg(long)]
    hidden: bool,

    #[arg(long = "no-ignore")]
    no_ignore: bool,

    #[arg(long = "glob", value_name = "PATTERN", action = clap::ArgAction::Append)]
    globs: Vec<String>,

    #[arg(long, default_value_t = crate::discovery::DEFAULT_MAX_FILE_SIZE)]
    max_file_size: u64,

    #[arg(long, default_value_t = crate::discovery::DEFAULT_MAX_TOTAL_BYTES)]
    max_total_bytes: u64,

    #[arg(long, default_value_t = crate::discovery::DEFAULT_MAX_CHUNKS)]
    max_chunks: usize,

    #[arg(long, default_value_t = DEFAULT_THREADS)]
    threads: usize,

    #[arg(long, value_name = "DIRECTORY")]
    cache_dir: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Inspect, prepare, or verify pinned local model artifacts.
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    /// Report local model, cache, runtime, and platform readiness.
    Doctor(DoctorArgs),
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
    /// List built-in profiles and report whether each local cache verifies.
    List(ModelCacheArgs),
    /// Download one pinned profile. This is the only command that uses the network.
    Download(ModelProfileArgs),
    /// Verify a profile already present in the local cache. Never uses the network.
    Verify(ModelProfileArgs),
}

#[derive(Debug, Args)]
struct ModelCacheArgs {
    #[arg(long, value_name = "DIRECTORY")]
    cache_dir: Option<PathBuf>,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct ModelProfileArgs {
    profile: String,
    #[arg(long, value_name = "DIRECTORY")]
    cache_dir: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct DoctorArgs {
    #[arg(long, value_name = "DIRECTORY")]
    cache_dir: Option<PathBuf>,
    #[arg(long)]
    json: bool,
}

pub fn entry() -> u8 {
    let cli = Cli::parse();
    match run(cli) {
        Ok(status) => status,
        Err(error) => {
            eprintln!("supergrep: {error}");
            2
        }
    }
}

fn run(mut cli: Cli) -> Result<u8> {
    match cli.command.take() {
        Some(Command::Model { command }) => run_model_command(command),
        Some(Command::Doctor(args)) => run_doctor(args),
        None => match cli.query.take() {
            Some(query) => run_search(cli, query),
            None => {
                use clap::CommandFactory;
                let mut command = Cli::command();
                command.print_help().map_err(SupergrepError::Io)?;
                print_stdout("")?;
                Ok(0)
            }
        },
    }
}

fn run_search(cli: Cli, query: String) -> Result<u8> {
    if cli.top_k == 0 {
        return Err(SupergrepError::Input("--top-k must be at least 1".into()));
    }
    if cli.batch_size == 0 {
        return Err(SupergrepError::Input(
            "--batch-size must be at least 1".into(),
        ));
    }
    if cli.deep && cli.candidates.is_some() {
        return Err(SupergrepError::Input(
            "--candidates does not apply with --deep".into(),
        ));
    }
    if cli.threads == 0 {
        return Err(SupergrepError::Input("--threads must be at least 1".into()));
    }
    if cli.lexical
        && (cli.model.is_some()
            || cli.cache_dir.is_some()
            || cli.threads != DEFAULT_THREADS
            || cli.batch_size != DEFAULT_BATCH_SIZE)
    {
        return Err(SupergrepError::Input(
            "--model, --cache-dir, --threads, and --batch-size do not apply with --lexical".into(),
        ));
    }
    if cli.lexical && cli.candidates.is_some() {
        return Err(SupergrepError::Input(
            "--candidates does not apply with --lexical".into(),
        ));
    }

    let total_started = Instant::now();
    let mode = if cli.lexical {
        SearchMode::Lexical
    } else if cli.deep {
        SearchMode::Deep
    } else {
        SearchMode::Fast
    };
    let candidate_limit = cli.candidates.unwrap_or(DEFAULT_CANDIDATES);
    if mode == SearchMode::Fast && candidate_limit == 0 {
        return Err(SupergrepError::Input(
            "--candidates must be at least 1".into(),
        ));
    }

    let mut model_load_ms = None;
    let mut model_metadata = None;
    let mut scorer = None;
    let mut tokenizer = None;

    if mode != SearchMode::Lexical {
        let cache = resolve_cache(cli.cache_dir.clone())?;
        let registry = crate::model::built_in_registry()?;
        let model_specification = cli.model.as_deref().unwrap_or(DEFAULT_MODEL);
        let model_started = Instant::now();
        let resolved = resolve_model_spec(&registry, &cache, model_specification)?;
        let runtime = resolve_runtime_library(&std::env::current_exe()?)?;
        validate_ort_runtime(&runtime.path)?;
        let onnx = resolved.load_scorer(&runtime.path, cli.threads)?;
        onnx.tokenizer().validate_query(&query)?;
        model_load_ms = Some(duration_ms(model_started.elapsed()));
        model_metadata = Some(ModelMetadata {
            id: resolved.profile.id().to_owned(),
            revision: resolved.profile.revision().to_owned(),
            score_kind: ModelScoreKind::RawLogit,
        });
        tokenizer = Some(onnx.tokenizer().clone());
        scorer = Some(onnx);
    }

    let root = cli.root.unwrap_or_else(|| PathBuf::from("."));
    let discovery_started = Instant::now();
    let mut discovery = discover(&DiscoveryOptions {
        root: root.clone(),
        include_hidden: cli.hidden,
        respect_ignore: !cli.no_ignore,
        globs: cli.globs,
        max_file_size: cli.max_file_size,
        max_total_bytes: cli.max_total_bytes,
        max_chunks: cli.max_chunks,
    })?;
    let discovery_ms = duration_ms(discovery_started.elapsed());

    let chunk_started = Instant::now();
    let chunked = chunk_sources(&discovery.sources, ChunkConfig::default(), cli.max_chunks)
        .map_err(|error| SupergrepError::Input(error.to_string()))?;
    let mut chunks = chunked.chunks;
    let mut fitting_split_count = 0usize;
    let mut fitting_limited = false;
    if let Some(tokenizer) = tokenizer.as_ref() {
        let prepared_query = tokenizer.prepare_pair_query(&query)?;
        let fit = fit_chunks(
            &discovery.sources,
            &chunks,
            &query,
            tokenizer.contract().max_pair_tokens,
            cli.max_chunks,
            |_, passage| prepared_query.token_count(passage),
        )?;
        chunks = fit.chunks;
        fitting_split_count = fit.split_count;
        fitting_limited = fit.limit_reached;
    }
    let chunking_ms = duration_ms(chunk_started.elapsed());
    let chunk_limit_reached = chunked.limit_reached || fitting_limited;
    if chunk_limit_reached {
        discovery.partial = true;
        discovery.scan_complete = false;
        discovery.stats.failures += 1;
    }

    let search_started = Instant::now();
    if let Some(scorer) = scorer.as_ref() {
        scorer.reset_timings();
    }
    let engine = SearchEngine::new(&discovery.sources, &chunks)
        .map_err(|error| SupergrepError::Input(error.to_string()))?;
    let report = engine.search(
        &query,
        SearchConfig {
            mode,
            candidate_limit,
            top_k: cli.top_k,
            batch_size: cli.batch_size,
        },
        scorer
            .as_ref()
            .map(|scorer| scorer as &dyn crate::model::Scorer),
    )?;
    let search_ms = duration_ms(search_started.elapsed());
    let scoring_timings = scorer.as_ref().map(|scorer| scorer.scoring_timings());

    let mut stats = PresentationStats::from_report(&report);
    stats.entries_seen = Some(discovery.stats.entries_seen);
    stats.files_considered = Some(discovery.stats.files_considered);
    stats.files_read = Some(discovery.stats.files_read);
    stats.files_accepted = Some(discovery.stats.files_accepted);
    stats.bytes_read = Some(discovery.stats.bytes_read);
    stats.policy_exclusions_observed = Some(discovery.stats.policy_exclusions_observed);
    stats.failures = Some(discovery.stats.failures);
    stats.scan_complete = discovery.scan_complete;
    stats.partial = discovery.partial;
    stats.lexical_evidence = report.lexical_evidence;
    stats.scoring_complete = report.scoring_complete;
    stats.chunks_total = report.all_chunk_indexes.len();
    stats.chunks_selected = report.selected_chunk_indexes.len();
    stats.chunks_evaluated = report.evaluated_chunk_indexes.len();
    stats.results_returned = report.results.len();
    stats.deduplicated_count = report.deduplicated_count;
    stats.candidate_selection = report.candidate_selection;
    stats.timing = PresentationTiming {
        discovery_ms: Some(discovery_ms),
        chunking_ms: Some(chunking_ms),
        model_load_ms,
        tokenization_ms: scoring_timings.map(|timings| duration_ms(timings.tokenization)),
        inference_ms: scoring_timings.map(|timings| duration_ms(timings.inference)),
        search_ms: Some(search_ms),
        total_ms: Some(duration_ms(total_started.elapsed())),
    };

    let mut input = PresentationInput::new(&query, &discovery.root, &report, &discovery.sources);
    input.model = model_metadata.as_ref();
    input.stats = stats;
    input.diagnostics = discovery
        .diagnostics
        .iter()
        .map(|diagnostic| {
            let mut result = PresentationDiagnostic::new(
                diagnostic_name(diagnostic.kind),
                diagnostic.message.clone(),
            );
            if let Some(path) = diagnostic.path.as_ref() {
                result = result.with_path(path);
            }
            result
        })
        .collect();
    if fitting_split_count > 0 {
        input.diagnostics.push(PresentationDiagnostic::new(
            "model_pair_split",
            format!("split {fitting_split_count} chunks further to satisfy the exact tokenizer pair limit"),
        ));
    }
    if chunk_limit_reached {
        input.diagnostics.push(PresentationDiagnostic::new(
            "chunk_limit_reached",
            format!(
                "processing stopped at the configured {}-chunk limit; results cover only retained chunks",
                cli.max_chunks
            ),
        ));
    }

    let output = if cli.json {
        render_json(&input)?
    } else {
        render_human(&input)
    };
    print_stdout(&output)?;
    if discovery.partial {
        Ok(3)
    } else if report.results.is_empty() {
        Ok(1)
    } else {
        Ok(0)
    }
}

fn run_model_command(command: ModelCommand) -> Result<u8> {
    let registry = built_in_registry()?;
    match command {
        ModelCommand::List(args) => {
            let cache = resolve_cache(args.cache_dir)?;
            let statuses = registry
                .profiles()
                .map(|profile| {
                    let verified = verify_cached(profile, &cache).ok();
                    (
                        profile.id().to_owned(),
                        verified.is_some(),
                        json!({
                            "id": profile.id(),
                            "repository": profile.repository(),
                            "revision": profile.revision(),
                            "license": profile.license(),
                            "installed": verified.is_some(),
                            "directory": cache.profile_dir(profile),
                        }),
                    )
                })
                .collect::<Vec<_>>();
            if args.json {
                let profiles = statuses
                    .iter()
                    .map(|(_, _, profile)| profile.clone())
                    .collect::<Vec<_>>();
                print_stdout(
                    &serde_json::to_string(&json!({
                        "schema_version": 1,
                        "profiles": profiles
                    }))
                    .map_err(json_error)?,
                )?;
            } else {
                for (id, installed, profile) in statuses {
                    let revision = profile["revision"].as_str().unwrap_or_default();
                    print_stdout(&format!(
                        "{}\t{}\t{}\t{}",
                        id,
                        if installed {
                            "verified"
                        } else {
                            "not prepared"
                        },
                        revision,
                        profile["directory"].as_str().unwrap_or_default()
                    ))?;
                }
            }
            Ok(0)
        }
        ModelCommand::Download(args) => {
            let cache = resolve_cache(args.cache_dir)?;
            let profile = registry.profile(&args.profile)?;
            let downloaded = ModelDownloader::with_http(cache).download(profile)?;
            print_stdout(&format!(
                "verified model {} @ {} in {}",
                downloaded.profile_id,
                downloaded.revision,
                downloaded.directory.display()
            ))?;
            Ok(0)
        }
        ModelCommand::Verify(args) => {
            let cache = resolve_cache(args.cache_dir)?;
            let profile = registry.profile(&args.profile)?;
            let verified = verify_cached(profile, &cache)?;
            print_stdout(&format!(
                "verified model {} @ {} in {} ({} artifacts)",
                verified.profile_id,
                verified.revision,
                verified.directory.display(),
                verified.artifacts.len()
            ))?;
            Ok(0)
        }
    }
}

fn run_doctor(args: DoctorArgs) -> Result<u8> {
    let registry = built_in_registry()?;
    let cache = resolve_cache(args.cache_dir)?;
    let profile = registry.profile(DEFAULT_MODEL)?;
    let model_ready = verify_cached(profile, &cache).is_ok();
    let runtime_result = std::env::current_exe()
        .map_err(SupergrepError::Io)
        .and_then(|executable| resolve_runtime_library(&executable))
        .and_then(|runtime| validate_ort_runtime(&runtime.path).map(|version| (runtime, version)));
    let runtime_ready = runtime_result.is_ok();
    let document = json!({
        "schema_version": 1,
        "platform": std::env::consts::OS,
        "architecture": std::env::consts::ARCH,
        "model_profile": DEFAULT_MODEL,
        "model_ready": model_ready,
        "model_cache": cache.root(),
        "runtime_ready": runtime_ready,
        "runtime": runtime_result.as_ref().ok().map(|(runtime, version)| json!({"version":version,"path":runtime.path})),
        "runtime_error": runtime_result.as_ref().err().map(ToString::to_string),
    });
    if args.json {
        print_stdout(&serde_json::to_string(&document).map_err(json_error)?)?;
    } else {
        print_stdout(&format!(
            "platform: {} {}",
            std::env::consts::OS,
            std::env::consts::ARCH
        ))?;
        print_stdout(&format!(
            "default model: {DEFAULT_MODEL} ({})",
            if model_ready {
                "verified"
            } else {
                "not prepared; run `supergrep model download compact-multilingual`"
            }
        ))?;
        print_stdout(&format!(
            "runtime: {}",
            runtime_result
                .map(|(runtime, version)| format!("{version} at {}", runtime.path.display()))
                .unwrap_or_else(|error| format!("unavailable: {error}"))
        ))?;
        print_stdout(&format!("cache: {}", cache.root().display()))?;
    }
    Ok(if model_ready && runtime_ready { 0 } else { 1 })
}

fn resolve_cache(cli_override: Option<PathBuf>) -> Result<ModelCache> {
    let override_path =
        cli_override.or_else(|| std::env::var_os("SUPERGREP_MODEL_CACHE").map(PathBuf::from));
    ModelCache::from_override_or_platform_default(override_path)
}

fn duration_ms(duration: std::time::Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

fn json_error(error: serde_json::Error) -> SupergrepError {
    SupergrepError::Internal(format!("could not encode JSON: {error}"))
}

fn print_stdout(contents: &str) -> Result<()> {
    let mut stdout = std::io::stdout().lock();
    stdout
        .write_all(contents.as_bytes())
        .map_err(SupergrepError::Io)?;
    stdout.write_all(b"\n").map_err(SupergrepError::Io)?;
    stdout.flush().map_err(SupergrepError::Io)?;
    Ok(())
}

fn diagnostic_name(kind: DiscoveryDiagnosticKind) -> &'static str {
    match kind {
        DiscoveryDiagnosticKind::GitMetadata => "git_metadata",
        DiscoveryDiagnosticKind::Symlink => "symlink",
        DiscoveryDiagnosticKind::NotRegularFile => "not_regular_file",
        DiscoveryDiagnosticKind::GlobFiltered => "glob_filtered",
        DiscoveryDiagnosticKind::FileTooLarge => "file_too_large",
        DiscoveryDiagnosticKind::TotalByteLimit => "total_byte_limit",
        DiscoveryDiagnosticKind::NulByte => "nul_byte",
        DiscoveryDiagnosticKind::InvalidUtf8 => "invalid_utf8",
        DiscoveryDiagnosticKind::ChangedDuringRead => "changed_during_read",
        DiscoveryDiagnosticKind::ReadFailure => "read_failure",
        DiscoveryDiagnosticKind::TraversalFailure => "traversal_failure",
        DiscoveryDiagnosticKind::IgnoreRuleFailure => "ignore_rule_failure",
    }
}
