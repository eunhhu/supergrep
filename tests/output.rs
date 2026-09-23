use std::path::{Path, PathBuf};

use supergrep::{
    output::{
        render_human, render_json, ModelMetadata, ModelScoreKind, PresentationDiagnostic,
        PresentationInput, PresentationStats,
    },
    search::{RankedEntry, SearchMode, SearchReport},
    source::{Source, SourceId},
};

fn report(mode: SearchMode, entry: RankedEntry) -> SearchReport {
    SearchReport {
        mode,
        all_chunk_indexes: vec![0, 1, 2],
        selected_chunk_indexes: vec![0],
        evaluated_chunk_indexes: (mode != SearchMode::Lexical)
            .then_some(0)
            .into_iter()
            .collect(),
        lexical_evidence: false,
        scoring_complete: (mode != SearchMode::Lexical).then_some(false),
        deduplicated_count: 1,
        candidate_selection: None,
        results: vec![entry],
    }
}

fn entry(path: PathBuf, logit: Option<f32>, bm25: Option<f64>) -> RankedEntry {
    RankedEntry {
        chunk_index: 0,
        source_id: SourceId::new(7),
        path,
        start_byte: 0,
        end_byte: "alpha\nline\ttext".len(),
        start_line: 1,
        end_line: 2,
        lexical_score: bm25,
        lexical_body_score: bm25,
        lexical_path_score: Some(0.0),
        relevance_logit: logit,
    }
}

#[test]
fn json_is_one_parseable_versioned_object_with_fast_coverage_warnings() {
    let path = PathBuf::from("/repo/dir/file name\ncontrol.rs");
    let source = Source::from_text(SourceId::new(7), &path, "alpha\nline\ttext");
    let report = report(SearchMode::Fast, entry(path, Some(-3.5), Some(41.0)));
    let model = ModelMetadata {
        id: "compact-multilingual".into(),
        revision: "abc123".into(),
        score_kind: ModelScoreKind::RawLogit,
    };
    let sources = [source];
    let mut input =
        PresentationInput::new("retry\nsettings", Path::new("/repo"), &report, &sources);
    input.model = Some(&model);
    input.stats.scan_complete = false;
    input.stats.partial = true;
    input.diagnostics.push(PresentationDiagnostic::new(
        "read_failure",
        "message\twith control",
    ));

    let rendered = render_json(&input).unwrap();
    assert!(
        !rendered.contains('\n'),
        "stdout JSON must stay one object/line"
    );
    let value: serde_json::Value = serde_json::from_str(&rendered).unwrap();
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["query"], "retry\nsettings");
    assert_eq!(value["root"]["path"], "/repo");
    assert_eq!(value["mode"], "fast");
    assert_eq!(value["model"]["id"], "compact-multilingual");
    assert_eq!(value["model"]["score_kind"], "raw_logit");
    assert_eq!(value["results"][0]["path"], "dir/file name\ncontrol.rs");
    assert_eq!(value["results"][0]["start_byte"], 0);
    assert_eq!(value["results"][0]["end_byte"], "alpha\nline\ttext".len());
    assert_eq!(value["results"][0]["score"], -3.5);
    assert_eq!(value["results"][0]["score_kind"], "raw_logit");
    assert_eq!(value["results"][0]["snippet"], "alpha\nline\ttext");
    let kinds = value["diagnostics"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["kind"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert!(kinds.contains(&"read_failure"));
    assert!(kinds.contains(&"scan_incomplete"));
    assert!(kinds.contains(&"partial_results"));
    assert!(kinds.contains(&"scoring_incomplete"));
    assert!(kinds.contains(&"no_lexical_evidence"));
}

#[test]
fn lexical_scores_are_bm25_and_model_scores_are_raw_logits() {
    let path = PathBuf::from("/repo/src/search.rs");
    let source = Source::from_text(SourceId::new(7), &path, "alpha\nline\ttext");

    let lexical_report = report(SearchMode::Lexical, entry(path.clone(), None, Some(1.25)));
    let lexical_sources = [source.clone()];
    let lexical = PresentationInput::new(
        "alpha",
        Path::new("/repo"),
        &lexical_report,
        &lexical_sources,
    );
    let lexical_json: serde_json::Value =
        serde_json::from_str(&render_json(&lexical).unwrap()).unwrap();
    assert_eq!(lexical_json["results"][0]["score"], 1.25);
    assert_eq!(lexical_json["results"][0]["score_kind"], "bm25");
    assert!(lexical_json["model"].is_null());

    let model_report = report(SearchMode::Deep, entry(path, Some(2.75), Some(999.0)));
    let model = ModelMetadata {
        id: "test".into(),
        revision: "pinned".into(),
        score_kind: ModelScoreKind::RawLogit,
    };
    let model_sources = [source];
    let mut model_input =
        PresentationInput::new("alpha", Path::new("/repo"), &model_report, &model_sources);
    model_input.model = Some(&model);
    let model_json: serde_json::Value =
        serde_json::from_str(&render_json(&model_input).unwrap()).unwrap();
    assert_eq!(model_json["results"][0]["score"], 2.75);
    assert_eq!(model_json["results"][0]["score_kind"], "raw_logit");
}

#[cfg(unix)]
#[test]
fn non_utf8_path_is_lossless_in_json_and_safe_in_human_output() {
    use std::os::unix::ffi::OsStringExt;

    let path =
        PathBuf::from("/repo").join(std::ffi::OsString::from_vec(b"bad\xffname.rs".to_vec()));
    let source = Source::from_text(SourceId::new(7), &path, "alpha\nline\ttext");
    let report = report(SearchMode::Lexical, entry(path, None, Some(1.0)));
    let sources = [source];
    let input = PresentationInput::new(
        "query\twith\ncontrol",
        Path::new("/repo"),
        &report,
        &sources,
    );

    let json: serde_json::Value = serde_json::from_str(&render_json(&input).unwrap()).unwrap();
    assert!(json["results"][0]["path"].is_null());
    let encoded = json["results"][0]["path_bytes_base64"].as_str().unwrap();
    use base64::Engine as _;
    assert_eq!(
        base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .unwrap(),
        b"bad\xffname.rs"
    );

    let human = render_human(&input);
    assert!(human.contains("query\\twith\\ncontrol"));
    assert!(human.contains("alpha\\nline\\ttext"));
    assert!(!human.contains('\t'));
    assert!(!human.contains('\r'));
    assert!(human.contains("scoring_complete=not_applicable"));
}

#[test]
fn supplied_stats_are_emitted_without_reinterpreting_them() {
    let path = PathBuf::from("/repo/a.rs");
    let source = Source::from_text(SourceId::new(7), &path, "alpha\nline\ttext");
    let report = report(SearchMode::Lexical, entry(path, None, Some(1.0)));
    let sources = [source];
    let mut input = PresentationInput::new("alpha", Path::new("/repo"), &report, &sources);
    input.stats = PresentationStats {
        entries_seen: Some(12),
        files_considered: Some(9),
        files_read: Some(8),
        files_accepted: Some(7),
        bytes_read: Some(1234),
        policy_exclusions_observed: Some(2),
        failures: Some(1),
        timing: supergrep::output::PresentationTiming {
            discovery_ms: Some(3),
            chunking_ms: Some(4),
            model_load_ms: None,
            tokenization_ms: None,
            inference_ms: None,
            search_ms: None,
            total_ms: Some(10),
        },
        ..PresentationStats::from_report(&report)
    };
    let value: serde_json::Value = serde_json::from_str(&render_json(&input).unwrap()).unwrap();
    assert_eq!(value["stats"]["entries_seen"], 12);
    assert_eq!(value["stats"]["bytes_read"], 1234);
    assert_eq!(value["stats"]["timing"]["discovery_ms"], 3);
    assert!(value["stats"]["timing"].get("model_load_ms").is_none());
}
