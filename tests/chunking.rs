use supergrep::{
    chunk::{
        chunk_source, chunk_source_with_token_count, chunk_sources, chunk_sources_with_token_count,
        estimated_token_count, ChunkConfig, ChunkKind,
    },
    source::{Source, SourceId},
};

fn config() -> ChunkConfig {
    ChunkConfig {
        target_tokens: 100,
        overlap_tokens: 0,
        max_lines: 1,
        max_bytes: 100,
    }
}

#[test]
fn crlf_unicode_and_final_newline_keep_exact_evidence_coordinates() {
    let text = "alpha\r\n한글 😀\r\nlast";
    let source = Source::from_text(SourceId::new(9), "fixture", text);
    let chunks = chunk_source(&source, config()).unwrap();

    assert_eq!(source.line_starts(), &[0, 7, 20]);
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].text(&source), Some("alpha\r\n"));
    assert_eq!(chunks[1].text(&source), Some("한글 😀\r\n"));
    assert_eq!(chunks[2].text(&source), Some("last"));
    assert_eq!((chunks[1].start_byte, chunks[1].end_byte), (7, 20));
    assert_eq!((chunks[1].start_line, chunks[1].end_line), (2, 2));
    assert_eq!((chunks[2].start_line, chunks[2].end_line), (3, 3));
    assert!(chunks
        .iter()
        .all(|chunk| source.slice(chunk.byte_range()).is_some()));
}

#[test]
fn a_long_unicode_line_splits_on_utf8_boundaries_without_losing_bytes() {
    let text = "😀한😀한😀한😀한😀한";
    let source = Source::from_text(SourceId::new(0), "fixture", text);
    let mut config = config();
    config.max_lines = 40;
    config.max_bytes = 8;
    let chunks = chunk_source(&source, config).unwrap();

    assert!(chunks.len() > 1);
    assert!(chunks.iter().all(|chunk| chunk.kind == ChunkKind::LongLine));
    assert!(chunks.iter().all(|chunk| {
        source.text().is_char_boundary(chunk.start_byte)
            && source.text().is_char_boundary(chunk.end_byte)
            && chunk.end_byte - chunk.start_byte <= 8
    }));
    let rebuilt = chunks
        .iter()
        .map(|chunk| chunk.text(&source).unwrap())
        .collect::<String>();
    assert_eq!(rebuilt, text);
}

#[test]
fn whitespace_only_content_does_not_create_a_chunk() {
    let source = Source::from_text(SourceId::new(0), "fixture", " \t\r\n\n");
    assert!(chunk_source(&source, config()).unwrap().is_empty());
}

#[test]
fn overlap_never_exceeds_the_requested_token_budget() {
    let text = "one two three four\nfive six seven eight\nnine ten eleven twelve\n";
    let source = Source::from_text(SourceId::new(0), "fixture", text);
    let config = ChunkConfig {
        target_tokens: 8,
        overlap_tokens: 1,
        max_lines: 2,
        max_bytes: 200,
    };
    let chunks = chunk_source_with_token_count(&source, config, estimated_token_count).unwrap();
    assert!(chunks.len() >= 2);
    for pair in chunks.windows(2) {
        if pair[1].start_byte < pair[0].end_byte {
            let overlap = source.slice(pair[1].start_byte..pair[0].end_byte).unwrap();
            assert!(
                estimated_token_count(overlap) <= 1,
                "overlap was {overlap:?}"
            );
        }
    }
}

#[test]
fn chunk_cap_is_explicit_instead_of_silent_truncation() {
    let first = Source::from_text(SourceId::new(0), "a", "one\ntwo\nthree\n");
    let second = Source::from_text(SourceId::new(1), "b", "four\n");
    let report = chunk_sources(&[first, second], config(), 2).unwrap();
    assert_eq!(report.chunks.len(), 2);
    assert!(report.limit_reached);
}

#[test]
fn line_spans_are_inclusive_for_no_final_newline() {
    let source = Source::from_text(SourceId::new(0), "fixture", "first\nlast");
    let chunks = chunk_source(&source, config()).unwrap();
    assert_eq!(chunks.len(), 2);
    assert_eq!((chunks[1].start_line, chunks[1].end_line), (2, 2));
    assert_eq!(chunks[1].end_byte, source.len_bytes());
}

#[test]
fn many_short_lines_keep_expected_coordinates() {
    let text = "x\n".repeat(4_000);
    let source = Source::from_text(SourceId::new(0), "fixture", text);
    let config = ChunkConfig {
        target_tokens: 100,
        overlap_tokens: 0,
        max_lines: 40,
        max_bytes: 1_000,
    };
    let chunks = chunk_source(&source, config).unwrap();
    assert_eq!(chunks.len(), 100);
    assert_eq!(chunks.first().unwrap().start_line, 1);
    assert_eq!(chunks.last().unwrap().end_line, 4_000);
}

#[test]
fn chunk_cap_stops_planning_before_processing_an_entire_large_source() {
    use std::cell::Cell;

    let source = Source::from_text(SourceId::new(0), "fixture", "x\n".repeat(1_000));
    let config = ChunkConfig {
        target_tokens: 100,
        overlap_tokens: 0,
        max_lines: 1,
        max_bytes: 100,
    };
    let calls = Cell::new(0_usize);
    let report = chunk_sources_with_token_count(&[source], config, 1, |text| {
        calls.set(calls.get() + 1);
        estimated_token_count(text)
    })
    .unwrap();
    assert_eq!(report.chunks.len(), 1);
    assert!(report.limit_reached);
    assert!(
        calls.get() <= 3,
        "planned too far past the cap: {} calls",
        calls.get()
    );
}
