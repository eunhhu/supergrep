use supergrep::{
    chunk::{Chunk, ChunkKind},
    fitting::fit_chunks,
    source::{Source, SourceId},
    Result, SupergrepError,
};

fn pair_length(query: &str, passage: &str) -> Result<usize> {
    // A deliberately simple fake tokenizer: query/passages plus three pair
    // special tokens.  Tests exercise the exact-pair contract without making
    // an ONNX model a test dependency.
    Ok(query.chars().count() + passage.chars().count() + 3)
}

fn full_chunk(source: &Source) -> Chunk {
    Chunk::new(source, 0..source.len_bytes(), ChunkKind::LongLine).unwrap()
}

#[test]
fn pair_overhead_controls_retention_not_passage_only_length() {
    let source = Source::from_text(SourceId::new(3), "memory", "abcd");
    let report = fit_chunks(
        std::slice::from_ref(&source),
        &[full_chunk(&source)],
        "qq",
        9,
        8,
        pair_length,
    )
    .unwrap();

    // `qq` + `abcd` + three special pair tokens is 9 exactly.  A passage-only
    // test would leave no evidence that query and special-token overhead were
    // accounted for.
    assert_eq!(report.chunks.len(), 1);
    assert_eq!(report.chunks[0].text(&source), Some("abcd"));
    assert_eq!(report.split_count, 0);
    assert!(!report.limit_reached);
}

#[test]
fn splitting_keeps_utf8_crlf_and_exact_line_coordinates() {
    let text = "alpha\r\n한글 😀\r\nomega";
    let source = Source::from_text(SourceId::new(4), "fixture", text);
    let original = full_chunk(&source);
    let report = fit_chunks(
        std::slice::from_ref(&source),
        &[original],
        "q",
        8,
        20,
        pair_length,
    )
    .unwrap();

    assert!(report.chunks.len() > 1);
    assert_eq!(
        report
            .chunks
            .iter()
            .map(|chunk| chunk.text(&source).unwrap())
            .collect::<String>(),
        text
    );
    assert!(report.chunks.iter().all(|chunk| {
        source.slice(chunk.byte_range()).is_some()
            && source.line_span(chunk.byte_range()) == Some((chunk.start_line, chunk.end_line))
            && !chunk.text(&source).unwrap().is_empty()
    }));
    assert!(report
        .chunks
        .windows(2)
        .all(|pair| pair[0].end_byte == pair[1].start_byte));
}

#[test]
fn long_one_line_makes_progress_without_losing_unicode_bytes() {
    let text = "😀한".repeat(80);
    let source = Source::from_text(SourceId::new(0), "long", text.clone());
    let report = fit_chunks(
        std::slice::from_ref(&source),
        &[full_chunk(&source)],
        "q",
        12,
        300,
        pair_length,
    )
    .unwrap();

    assert!(report.chunks.len() > 2);
    assert_eq!(report.split_count, report.chunks.len() - 1);
    assert_eq!(
        report
            .chunks
            .iter()
            .map(|chunk| chunk.text(&source).unwrap())
            .collect::<String>(),
        text
    );
    assert!(report.chunks.iter().all(|chunk| {
        pair_length("q", chunk.text(&source).unwrap()).unwrap() <= 12
            && source.text().is_char_boundary(chunk.start_byte)
            && source.text().is_char_boundary(chunk.end_byte)
    }));
}

#[test]
fn global_cap_is_visible_and_never_silently_discards_fitted_ranges() {
    let first = Source::from_text(SourceId::new(1), "first", "a\nb\nc\nd\n");
    let second = Source::from_text(SourceId::new(2), "second", "e\nf\n");
    let chunks = [full_chunk(&first), full_chunk(&second)];
    let report = fit_chunks(&[first.clone(), second], &chunks, "q", 6, 2, pair_length).unwrap();

    assert_eq!(report.input_count, 2);
    assert_eq!(report.chunks.len(), 2);
    assert!(report.limit_reached);
    assert!(report
        .chunks
        .iter()
        .all(|chunk| chunk.source_id == first.id()));
}

#[test]
fn impossible_single_scalar_is_a_model_error() {
    let source = Source::from_text(SourceId::new(0), "emoji", "😀");
    let error = fit_chunks(
        std::slice::from_ref(&source),
        &[full_chunk(&source)],
        "q",
        4,
        1,
        pair_length,
    )
    .unwrap_err();

    assert!(matches!(error, SupergrepError::Model(_)));
    assert!(error.to_string().contains("one Unicode scalar"));
}

#[test]
fn source_mismatch_and_invalid_coordinates_are_rejected_before_fitting() {
    let source = Source::from_text(SourceId::new(1), "fixture", "한글\r\nnext");
    let missing_source = Chunk {
        source_id: SourceId::new(99),
        ..full_chunk(&source)
    };
    let error = fit_chunks(
        std::slice::from_ref(&source),
        &[missing_source],
        "q",
        20,
        4,
        pair_length,
    )
    .unwrap_err();
    assert!(matches!(error, SupergrepError::Input(_)));
    assert!(error.to_string().contains("absent"));

    let invalid_range = Chunk {
        end_byte: 1,
        ..full_chunk(&source)
    };
    let error = fit_chunks(&[source], &[invalid_range], "q", 20, 4, pair_length).unwrap_err();
    assert!(matches!(error, SupergrepError::Input(_)));
    assert!(error.to_string().contains("invalid UTF-8 byte range"));
}
