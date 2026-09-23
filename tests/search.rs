use std::{cell::RefCell, collections::HashMap};

use supergrep::{
    chunk::{Chunk, ChunkKind},
    model::Scorer,
    search::{CandidateQuota, SearchConfig, SearchEngine, SearchMode},
    source::{Source, SourceId},
    Result, SupergrepError,
};

#[derive(Default)]
struct FakeScorer {
    scores: HashMap<String, f32>,
    calls: RefCell<Vec<Vec<String>>>,
    failure: Option<String>,
    wrong_length: bool,
    non_finite: bool,
}

impl FakeScorer {
    fn with_scores(entries: impl IntoIterator<Item = (&'static str, f32)>) -> Self {
        Self {
            scores: entries
                .into_iter()
                .map(|(passage, score)| (passage.to_owned(), score))
                .collect(),
            ..Self::default()
        }
    }

    fn scored_passages(&self) -> Vec<String> {
        self.calls.borrow().iter().flatten().cloned().collect()
    }
}

impl Scorer for FakeScorer {
    fn score_batch(&self, _query: &str, passages: &[String]) -> Result<Vec<f32>> {
        self.calls.borrow_mut().push(passages.to_vec());
        if let Some(message) = &self.failure {
            return Err(SupergrepError::Model(message.clone()));
        }
        if self.wrong_length {
            return Ok(Vec::new());
        }
        if self.non_finite {
            return Ok(vec![f32::NAN; passages.len()]);
        }
        Ok(passages
            .iter()
            .map(|passage| self.scores.get(passage).copied().unwrap_or(0.0))
            .collect())
    }
}

fn source_with_chunks(
    id: usize,
    path: impl Into<std::path::PathBuf>,
    passages: &[&str],
) -> (Source, Vec<Chunk>) {
    let mut text = String::new();
    let mut ranges = Vec::new();
    for (position, passage) in passages.iter().enumerate() {
        if position > 0 {
            text.push('\n');
        }
        let start = text.len();
        text.push_str(passage);
        ranges.push(start..text.len());
    }
    let source = Source::from_text(SourceId::new(id), path, text);
    let chunks = ranges
        .into_iter()
        .map(|range| Chunk::new(&source, range, ChunkKind::Lines).unwrap())
        .collect();
    (source, chunks)
}

fn engine<'a>(sources: &'a [Source], chunks: &'a [Chunk]) -> SearchEngine<'a> {
    SearchEngine::new(sources, chunks).unwrap()
}

fn model_config(mode: SearchMode, candidate_limit: usize) -> SearchConfig {
    SearchConfig {
        mode,
        candidate_limit,
        top_k: 20,
        batch_size: 2,
    }
}

#[test]
fn fast_and_deep_evaluate_the_same_complete_set_when_n_is_at_most_k() {
    let (source, chunks) = source_with_chunks(1, "src/client.rs", &["alpha", "beta"]);
    let sources = [source];
    let scorer = FakeScorer::with_scores([("alpha", 0.25), ("beta", 2.0)]);
    let search = engine(&sources, &chunks);

    let fast = search
        .search(
            "no lexical overlap",
            model_config(SearchMode::Fast, 2),
            Some(&scorer),
        )
        .unwrap();
    let deep = search
        .search(
            "no lexical overlap",
            model_config(SearchMode::Deep, 1),
            Some(&scorer),
        )
        .unwrap();

    assert_eq!(fast.selected_chunk_indexes, [0, 1]);
    assert_eq!(fast.evaluated_chunk_indexes, deep.evaluated_chunk_indexes);
    assert_eq!(fast.results, deep.results);
    assert_eq!(fast.scoring_complete, Some(true));
    assert_eq!(deep.scoring_complete, Some(true));
    assert_eq!(fast.results[0].chunk_index, 1);
    assert_eq!(fast.results[0].relevance_logit, Some(2.0));
}

#[test]
fn fast_caps_candidates_without_hiding_the_full_chunk_count() {
    let passages = (0..10)
        .map(|number| format!("retry delay setting {number}"))
        .collect::<Vec<_>>();
    let references = passages.iter().map(String::as_str).collect::<Vec<_>>();
    let (source, chunks) = source_with_chunks(1, "src/retry.rs", &references);
    let sources = [source];
    let scorer = FakeScorer::default();
    let report = engine(&sources, &chunks)
        .search(
            "retry delay",
            model_config(SearchMode::Fast, 4),
            Some(&scorer),
        )
        .unwrap();

    assert_eq!(report.all_chunk_indexes.len(), 10);
    assert_eq!(report.selected_chunk_indexes.len(), 4);
    assert_eq!(report.evaluated_chunk_indexes.len(), 4);
    assert_eq!(report.scoring_complete, Some(false));
    assert_eq!(scorer.scored_passages().len(), 4);
}

#[test]
fn unmatched_korean_query_uses_broad_fast_sampling_and_exposes_incomplete_coverage() {
    let (first, first_chunks) = source_with_chunks(1, "a/config.rs", &["one", "two", "three"]);
    let (second, second_chunks) = source_with_chunks(2, "b/client.rs", &["four", "five", "six"]);
    let (third, third_chunks) = source_with_chunks(3, "c/docs.md", &["seven", "eight", "nine"]);
    let sources = [first, second, third];
    let chunks = first_chunks
        .into_iter()
        .chain(second_chunks)
        .chain(third_chunks)
        .collect::<Vec<_>>();
    let scorer = FakeScorer::default();
    let report = engine(&sources, &chunks)
        .search(
            "재시도 간격을 정하는 부분",
            model_config(SearchMode::Fast, 3),
            Some(&scorer),
        )
        .unwrap();

    assert!(!report.lexical_evidence);
    assert_eq!(report.scoring_complete, Some(false));
    assert_eq!(report.selected_chunk_indexes.len(), 3);
    let sampled_sources = report
        .selected_chunk_indexes
        .iter()
        .map(|index| chunks[*index].source_id)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(sampled_sources.len(), 3);
}

#[test]
fn unmatched_query_samples_late_paths_when_paths_exceed_candidate_limit() {
    let mut sources = Vec::new();
    let mut chunks = Vec::new();
    for index in 0..145 {
        let path = if index == 144 {
            "z-target.rs".to_owned()
        } else {
            format!("a{index:03}.rs")
        };
        let passage = if index == 144 {
            "pub fn retry_delay() {}"
        } else {
            "pub fn display_name() {}"
        };
        let (source, source_chunks) = source_with_chunks(index, path, &[passage]);
        sources.push(source);
        chunks.extend(source_chunks);
    }
    let scorer = FakeScorer::with_scores([("pub fn retry_delay() {}", 10.0)]);
    let report = engine(&sources, &chunks)
        .search(
            "요청 실패 후 재시도 대기 시간을 늘리는 부분",
            model_config(SearchMode::Fast, 128),
            Some(&scorer),
        )
        .unwrap();

    assert!(!report.lexical_evidence);
    assert_eq!(report.evaluated_chunk_indexes.len(), 128);
    assert!(report.evaluated_chunk_indexes.contains(&144));
    assert_eq!(report.results[0].path, std::path::Path::new("z-target.rs"));
}

#[test]
fn deep_scores_every_chunk_even_when_candidate_limit_is_small() {
    let passages = ["a", "b", "c", "d", "e"];
    let (source, chunks) = source_with_chunks(1, "src/all.rs", &passages);
    let sources = [source];
    let scorer = FakeScorer::default();
    let report = engine(&sources, &chunks)
        .search(
            "unmatched",
            model_config(SearchMode::Deep, 1),
            Some(&scorer),
        )
        .unwrap();

    assert_eq!(report.selected_chunk_indexes, [0, 1, 2, 3, 4]);
    assert_eq!(report.evaluated_chunk_indexes, [0, 1, 2, 3, 4]);
    assert_eq!(report.scoring_complete, Some(true));
    assert_eq!(scorer.scored_passages(), passages);
}

#[test]
fn quota_rounding_and_model_ties_are_deterministic_by_path_then_byte() {
    let (z_source, z_chunks) = source_with_chunks(
        2,
        "z.rs",
        &[
            "cache cache 0",
            "cache cache 1",
            "cache cache 2",
            "cache cache 3",
            "cache cache 8",
        ],
    );
    let (a_source, a_chunks) = source_with_chunks(
        1,
        "a.rs",
        &[
            "cache cache 4",
            "cache cache 5",
            "cache cache 6",
            "cache cache 7",
        ],
    );
    let sources = [z_source, a_source];
    let chunks = z_chunks.into_iter().chain(a_chunks).collect::<Vec<_>>();
    let scorer = FakeScorer::default();
    let search = engine(&sources, &chunks);
    let config = model_config(SearchMode::Fast, 8);
    let first = search.search("cache", config, Some(&scorer)).unwrap();
    let second = search.search("cache", config, Some(&scorer)).unwrap();

    assert_eq!(
        first.candidate_selection.unwrap().quota,
        CandidateQuota {
            body: 5,
            path: 1,
            neighbours: 1,
            diverse: 1,
        }
    );
    assert_eq!(first.selected_chunk_indexes, second.selected_chunk_indexes);
    assert_eq!(
        first
            .results
            .iter()
            .map(|entry| entry.path.to_string_lossy().into_owned())
            .collect::<Vec<_>>(),
        ["a.rs", "a.rs", "a.rs", "a.rs", "z.rs", "z.rs", "z.rs", "z.rs"]
    );
}

#[test]
fn batch_scores_stay_attached_to_their_original_chunks() {
    let passages = ["first", "second", "third", "fourth", "fifth"];
    let (source, chunks) = source_with_chunks(1, "src/batches.rs", &passages);
    let sources = [source];
    let scorer = FakeScorer::with_scores([
        ("first", 1.0),
        ("second", 5.0),
        ("third", 2.0),
        ("fourth", 4.0),
        ("fifth", 3.0),
    ]);
    let report = engine(&sources, &chunks)
        .search(
            "unmatched",
            model_config(SearchMode::Deep, 1),
            Some(&scorer),
        )
        .unwrap();

    assert_eq!(
        scorer
            .calls
            .borrow()
            .iter()
            .map(Vec::len)
            .collect::<Vec<_>>(),
        [2, 2, 1]
    );
    assert_eq!(
        report
            .results
            .iter()
            .map(|entry| (entry.chunk_index, entry.relevance_logit.unwrap()))
            .collect::<Vec<_>>(),
        [(1, 5.0), (3, 4.0), (4, 3.0), (2, 2.0), (0, 1.0)]
    );
}

#[test]
fn high_same_source_byte_overlap_suppresses_the_lower_ranked_result() {
    let source = Source::from_text(
        SourceId::new(1),
        "src/overlap.rs",
        "abcdefghijklmnopqrstuvwxyz",
    );
    let chunks = vec![
        Chunk::new(&source, 0..10, ChunkKind::Lines).unwrap(),
        Chunk::new(&source, 2..12, ChunkKind::Lines).unwrap(),
        Chunk::new(&source, 13..23, ChunkKind::Lines).unwrap(),
    ];
    let sources = [source];
    let scorer = FakeScorer::with_scores([
        ("abcdefghij", 3.0),
        ("cdefghijkl", 2.0),
        ("nopqrstuvw", 1.0),
    ]);
    let report = engine(&sources, &chunks)
        .search(
            "unmatched",
            model_config(SearchMode::Deep, 1),
            Some(&scorer),
        )
        .unwrap();

    assert_eq!(report.deduplicated_count, 1);
    assert_eq!(
        report
            .results
            .iter()
            .map(|entry| entry.chunk_index)
            .collect::<Vec<_>>(),
        [0, 2]
    );
}

#[test]
fn lexical_mode_returns_only_positive_hits_and_never_calls_a_scorer() {
    let (source, chunks) =
        source_with_chunks(1, "src/client.rs", &["retry delay", "unrelated text"]);
    let sources = [source];
    let scorer = FakeScorer {
        failure: Some("must not be called".into()),
        ..FakeScorer::default()
    };
    let report = engine(&sources, &chunks)
        .search("retry", model_config(SearchMode::Lexical, 1), Some(&scorer))
        .unwrap();

    assert_eq!(report.selected_chunk_indexes, [0]);
    assert!(report.evaluated_chunk_indexes.is_empty());
    assert_eq!(report.scoring_complete, None);
    assert_eq!(report.results.len(), 1);
    assert!(report.results[0].lexical_score.unwrap() > 0.0);
    assert_eq!(report.results[0].relevance_logit, None);
    assert!(scorer.calls.borrow().is_empty());
}

#[test]
fn scorer_failures_count_mismatches_and_nonfinite_scores_are_fatal() {
    let (source, chunks) = source_with_chunks(1, "src/fail.rs", &["one", "two"]);
    let sources = [source];
    let search = engine(&sources, &chunks);

    let failure = FakeScorer {
        failure: Some("fake failure".into()),
        ..FakeScorer::default()
    };
    assert!(matches!(
        search.search("query", model_config(SearchMode::Deep, 1), Some(&failure)),
        Err(SupergrepError::Model(message)) if message == "fake failure"
    ));

    let wrong_length = FakeScorer {
        wrong_length: true,
        ..FakeScorer::default()
    };
    assert!(matches!(
        search.search("query", model_config(SearchMode::Deep, 1), Some(&wrong_length)),
        Err(SupergrepError::Model(message)) if message.contains("returned 0 scores")
    ));

    let non_finite = FakeScorer {
        non_finite: true,
        ..FakeScorer::default()
    };
    assert!(matches!(
        search.search("query", model_config(SearchMode::Deep, 1), Some(&non_finite)),
        Err(SupergrepError::Model(message)) if message.contains("non-finite")
    ));
}
