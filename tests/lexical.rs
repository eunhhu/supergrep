use supergrep::{
    chunk::{Chunk, ChunkKind},
    lexical::{normalized_terms, LexicalCorpus},
    source::{Source, SourceId},
};

fn one_chunk(source: &Source) -> Chunk {
    Chunk::new(source, 0..source.len_bytes(), ChunkKind::Lines).unwrap()
}

#[test]
fn unicode_normalization_and_korean_terms_are_searchable() {
    let terms = normalized_terms("CAFÉ cafe\u{301} 재시도 간격");
    assert_eq!(terms, ["café", "café", "재시도", "간격"]);

    let source = Source::from_text(
        SourceId::new(1),
        "src/재시도.rs",
        "재시도 간격을 설정합니다",
    );
    let chunk = one_chunk(&source);
    let ranking = LexicalCorpus::new(&[source], &[chunk])
        .unwrap()
        .score("재시도 간격");
    assert!(ranking.has_evidence());
    assert!(ranking.scores[0].body_score > 0.0);
}

#[test]
fn identifiers_keep_whole_form_and_split_components() {
    let terms = normalized_terms("retryDelay_ms2 HTTPServer");
    assert_eq!(
        terms,
        [
            "retrydelay_ms2",
            "retry",
            "delay",
            "ms",
            "2",
            "httpserver",
            "http",
            "server"
        ]
    );

    let source = Source::from_text(
        SourceId::new(1),
        "src/client.rs",
        "let retryDelay_ms2 = 100;",
    );
    let chunk = one_chunk(&source);
    let ranking = LexicalCorpus::new(&[source], &[chunk])
        .unwrap()
        .score("retry delay");
    assert_eq!(ranking.scores.len(), 1);
    assert!(ranking.scores[0].body_score > 0.0);
}

#[test]
fn path_hits_are_separate_from_body_hits() {
    let path_source =
        Source::from_text(SourceId::new(1), "config/retry-policy.toml", "timeout = 30");
    let body_source = Source::from_text(
        SourceId::new(2),
        "src/main.rs",
        "retry policy is configured here",
    );
    let path_chunk = one_chunk(&path_source);
    let body_chunk = one_chunk(&body_source);
    let sources = [path_source, body_source];
    let chunks = [path_chunk, body_chunk];

    let ranking = LexicalCorpus::new(&sources, &chunks)
        .unwrap()
        .score("retry policy");
    let from_path = ranking
        .scores
        .iter()
        .find(|score| score.chunk_index == 0)
        .unwrap();
    let from_body = ranking
        .scores
        .iter()
        .find(|score| score.chunk_index == 1)
        .unwrap();
    assert!(from_path.path_score > 0.0);
    assert_eq!(from_path.body_score, 0.0);
    assert!(from_body.body_score > 0.0);
    assert_eq!(from_body.path_score, 0.0);
}

#[test]
fn bm25_rewards_term_frequency_and_keeps_exact_ties_in_chunk_order() {
    let first = Source::from_text(SourceId::new(1), "first.txt", "cache cache cache");
    let second = Source::from_text(SourceId::new(2), "second.txt", "cache cache");
    let third = Source::from_text(SourceId::new(3), "third.txt", "cache cache");
    let chunks = [one_chunk(&first), one_chunk(&second), one_chunk(&third)];
    let sources = [first, second, third];
    let ranking = LexicalCorpus::new(&sources, &chunks)
        .unwrap()
        .score("cache");

    assert_eq!(ranking.scores[0].chunk_index, 0);
    assert!(ranking.scores[0].body_score > ranking.scores[1].body_score);
    assert_eq!(ranking.scores[1].chunk_index, 1);
    assert_eq!(ranking.scores[2].chunk_index, 2);
    assert_eq!(ranking.scores[1].score, ranking.scores[2].score);
}

#[test]
fn empty_or_unmatched_query_has_no_positive_lexical_evidence() {
    let source = Source::from_text(SourceId::new(1), "src/main.rs", "fn main() {}");
    let chunk = one_chunk(&source);
    let sources = [source];
    let chunks = [chunk];
    let corpus = LexicalCorpus::new(&sources, &chunks).unwrap();
    assert!(corpus.score("   --- ").scores.is_empty());
    assert!(corpus.score("unrelated query").scores.is_empty());
}
