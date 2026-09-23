//! Exact model-pair fitting for already snapshotted chunks.
//!
//! Initial chunking deliberately uses a lightweight passage-only estimate so
//! discovery can remain independent from a selected model.  Before a chunk is
//! sent to a cross-encoder, this module asks the selected tokenizer for the
//! exact special-token-inclusive `(query, passage)` length and, when needed,
//! subdivides the immutable source range.  It never rewrites passage text:
//! every fitted result is another byte range in the original [`Source`].

use std::{collections::HashMap, ops::Range};

use rayon::prelude::*;

use crate::{
    chunk::Chunk,
    source::{Source, SourceId},
    Result, SupergrepError,
};

/// The outcome of fitting initial chunks to a model's exact pair limit.
///
/// `split_count` is the number of binary split operations performed.  Thus a
/// single input chunk that becomes `n` fitted chunks contributes `n - 1`.
/// `limit_reached` says that at least one valid input range was not returned
/// because appending it would exceed the caller's global `max_chunks` cap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FitReport {
    pub chunks: Vec<Chunk>,
    pub input_count: usize,
    pub split_count: usize,
    pub limit_reached: bool,
}

impl FitReport {
    fn new(input_count: usize) -> Self {
        Self {
            chunks: Vec::new(),
            input_count,
            split_count: 0,
            limit_reached: false,
        }
    }
}

/// Fits `chunks` to `max_pair_tokens` using the exact tokenizer count supplied
/// by `pair_token_count`.
///
/// The counter is called with the unchanged query and the exact source slice;
/// its result must include the model's pair special tokens.  A tokenizer or
/// model error is returned unchanged.  Inputs are fully snapshot-validated
/// before any token counts are requested, so a malformed later chunk cannot
/// be hidden by an earlier output-cap stop.
///
/// Returned chunks are ordered by the supplied source order, then ascending
/// original byte range, then input order for equal ranges.  Fragments of one
/// original range are always in increasing byte order.
pub fn fit_chunks<F>(
    sources: &[Source],
    chunks: &[Chunk],
    query: &str,
    max_pair_tokens: usize,
    max_chunks: usize,
    pair_token_count: F,
) -> Result<FitReport>
where
    F: Fn(&str, &str) -> Result<usize>,
{
    if max_pair_tokens == 0 {
        return Err(SupergrepError::Input(
            "max_pair_tokens must be at least one".into(),
        ));
    }
    if max_chunks == 0 {
        return Err(SupergrepError::Input(
            "max_chunks must be at least one".into(),
        ));
    }

    let source_indexes = validate_distinct_source_ids(sources)?;
    let mut inputs = validate_chunks(sources, chunks, &source_indexes)?;
    inputs.sort_by_key(|input| {
        (
            input.source_index,
            input.chunk.start_byte,
            input.chunk.end_byte,
            input.input_index,
        )
    });

    let mut report = FitReport::new(chunks.len());
    for input in inputs {
        if report.chunks.len() == max_chunks {
            // Each validated initial chunk is non-empty, hence it has at
            // least one potential output even if it later needs splitting.
            report.limit_reached = true;
            break;
        }

        let source = &sources[input.source_index];
        fit_one_range(
            source,
            input.chunk,
            query,
            max_pair_tokens,
            max_chunks,
            &pair_token_count,
            &mut report,
        )?;
        if report.limit_reached {
            break;
        }
    }
    Ok(report)
}

/// Fits independent initial chunks with concurrent exact pair counts while
/// preserving the serial function's output order, split behavior, and cap.
/// The counter must be pure: counts may be computed ahead of an output cap.
pub fn fit_chunks_parallel<F>(
    sources: &[Source],
    chunks: &[Chunk],
    query: &str,
    max_pair_tokens: usize,
    max_chunks: usize,
    pair_token_count: F,
) -> Result<FitReport>
where
    F: Fn(&str, &str) -> Result<usize> + Sync,
{
    if max_pair_tokens == 0 || max_chunks == 0 {
        return fit_chunks(
            sources,
            chunks,
            query,
            max_pair_tokens,
            max_chunks,
            pair_token_count,
        );
    }
    let source_indexes = validate_distinct_source_ids(sources)?;
    let mut inputs = validate_chunks(sources, chunks, &source_indexes)?;
    inputs.sort_by_key(|input| {
        (
            input.source_index,
            input.chunk.start_byte,
            input.chunk.end_byte,
            input.input_index,
        )
    });

    // Every non-empty input yields at least one output, so inputs beyond the
    // cap cannot be reached.  Validate all source ranges first, as serial does.
    let count = inputs.len().min(max_chunks);
    let first_counts = inputs[..count]
        .par_iter()
        .map(|input| {
            let source = &sources[input.source_index];
            let passage = source
                .slice(input.chunk.byte_range())
                .expect("validated source range");
            pair_token_count(query, passage)
        })
        .collect::<Vec<_>>();

    let mut report = FitReport::new(chunks.len());
    for (input, pair_length) in inputs.iter().zip(first_counts) {
        if report.chunks.len() == max_chunks {
            report.limit_reached = true;
            break;
        }
        let source = &sources[input.source_index];
        if pair_length? <= max_pair_tokens {
            report
                .chunks
                .push(fitted_chunk(source, input.chunk, input.chunk.byte_range())?);
        } else {
            fit_one_range(
                source,
                input.chunk,
                query,
                max_pair_tokens,
                max_chunks,
                &pair_token_count,
                &mut report,
            )?;
            if report.limit_reached {
                break;
            }
        }
    }
    if inputs.len() > count && report.chunks.len() == max_chunks {
        report.limit_reached = true;
    }
    Ok(report)
}

#[derive(Clone, Copy)]
struct ValidatedInput<'a> {
    source_index: usize,
    input_index: usize,
    chunk: &'a Chunk,
}

fn validate_distinct_source_ids(sources: &[Source]) -> Result<HashMap<SourceId, usize>> {
    let mut indexes = HashMap::with_capacity(sources.len());
    for (index, source) in sources.iter().enumerate() {
        if indexes.insert(source.id(), index).is_some() {
            return Err(SupergrepError::Input(format!(
                "source snapshot list contains duplicate source id {}",
                source.id().get()
            )));
        }
    }
    Ok(indexes)
}

fn validate_chunks<'a>(
    sources: &[Source],
    chunks: &'a [Chunk],
    source_indexes: &HashMap<SourceId, usize>,
) -> Result<Vec<ValidatedInput<'a>>> {
    chunks
        .iter()
        .enumerate()
        .map(|(input_index, chunk)| {
            let source_index = source_indexes.get(&chunk.source_id).copied().ok_or_else(|| {
                SupergrepError::Input(format!(
                    "chunk {input_index} references source id {} that is absent from the current snapshots",
                    chunk.source_id.get()
                ))
            })?;
            let source = &sources[source_index];
            let range = chunk.byte_range();
            let text = source.slice(range.clone()).ok_or_else(|| {
                SupergrepError::Input(format!(
                    "chunk {input_index} has an invalid UTF-8 byte range {}..{} for {}",
                    range.start,
                    range.end,
                    source.path().display()
                ))
            })?;
            if text.is_empty() {
                return Err(SupergrepError::Input(format!(
                    "chunk {input_index} has an empty byte range for {}",
                    source.path().display()
                )));
            }
            let expected_lines = source.line_span(range.clone()).ok_or_else(|| {
                SupergrepError::Input(format!(
                    "chunk {input_index} has invalid line coordinates for {}",
                    source.path().display()
                ))
            })?;
            if (chunk.start_line, chunk.end_line) != expected_lines {
                return Err(SupergrepError::Input(format!(
                    "chunk {input_index} line coordinates {}..{} do not match snapshot coordinates {}..{} for {}",
                    chunk.start_line,
                    chunk.end_line,
                    expected_lines.0,
                    expected_lines.1,
                    source.path().display()
                )));
            }
            Ok(ValidatedInput {
                source_index,
                input_index,
                chunk,
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn fit_one_range<F>(
    source: &Source,
    original: &Chunk,
    query: &str,
    max_pair_tokens: usize,
    max_chunks: usize,
    pair_token_count: &F,
    report: &mut FitReport,
) -> Result<()>
where
    F: Fn(&str, &str) -> Result<usize>,
{
    // A stack with the right half pushed first produces a left-to-right,
    // deterministic depth-first traversal without borrowing the source text
    // across the fallible tokenizer callback.
    let mut pending = vec![original.byte_range()];
    while let Some(range) = pending.pop() {
        if report.chunks.len() == max_chunks {
            report.limit_reached = true;
            return Ok(());
        }

        let passage = source
            .slice(range.clone())
            .expect("validated ranges and internal split points are UTF-8-valid");
        let pair_length = pair_token_count(query, passage)?;
        if pair_length <= max_pair_tokens {
            report.chunks.push(fitted_chunk(source, original, range)?);
            continue;
        }

        let split = split_point(source, range.clone()).ok_or_else(|| {
            // The only unsplittable non-empty UTF-8 range is a single scalar.
            // State both the exact model count and model cap: callers must not
            // treat this as a lexical fallback opportunity.
            SupergrepError::Model(format!(
                "query plus one Unicode scalar at byte range {}..{} in {} encodes to {pair_length} tokens, exceeding the model pair limit of {max_pair_tokens}",
                range.start,
                range.end,
                source.path().display()
            ))
        })?;
        debug_assert!(range.start < split && split < range.end);
        report.split_count += 1;
        pending.push(split..range.end);
        pending.push(range.start..split);
    }
    Ok(())
}

fn fitted_chunk(source: &Source, original: &Chunk, range: Range<usize>) -> Result<Chunk> {
    let mut chunk = Chunk::new(source, range, original.kind).map_err(|error| {
        SupergrepError::Internal(format!(
            "validated fitted chunk could not be constructed: {error}"
        ))
    })?;
    chunk.context = original.context.clone();
    Ok(chunk)
}

/// Selects a valid non-edge split near the byte midpoint.
///
/// Newlines and other whitespace boundaries within the central half of the
/// range are preferred, so ordinary prose preserves readable lines when it
/// can do so without creating an extremely unbalanced recursion tree.  A
/// nearest Unicode-scalar boundary is always available for a range with two
/// or more scalars, guaranteeing long minified lines make progress.
fn split_point(source: &Source, range: Range<usize>) -> Option<usize> {
    let passage = source.slice(range.clone())?;
    let midpoint = range.start + (range.end - range.start) / 2;
    let nearby_limit = (range.end - range.start) / 4;
    let mut nearest_boundary: Option<(usize, usize)> = None;
    let mut nearby_newline: Option<(usize, usize)> = None;
    let mut nearby_whitespace: Option<(usize, usize)> = None;

    for (offset, character) in passage.char_indices() {
        let boundary = range.start + offset + character.len_utf8();
        if boundary == range.end {
            continue;
        }
        let distance = boundary.abs_diff(midpoint);
        update_nearest(&mut nearest_boundary, boundary, distance);
        if distance > nearby_limit {
            continue;
        }
        if character == '\n' {
            update_nearest(&mut nearby_newline, boundary, distance);
        } else if character.is_whitespace()
            // Preserve a CRLF pair as one line separator rather than fitting
            // `\r` and `\n` into independent fragments.
            && !(character == '\r' && passage[offset + character.len_utf8()..].starts_with('\n'))
        {
            update_nearest(&mut nearby_whitespace, boundary, distance);
        }
    }

    nearby_newline
        .or(nearby_whitespace)
        .or(nearest_boundary)
        .map(|(boundary, _)| boundary)
}

fn update_nearest(slot: &mut Option<(usize, usize)>, boundary: usize, distance: usize) {
    match slot {
        Some((current, current_distance))
            if *current_distance < distance
                || (*current_distance == distance && *current <= boundary) => {}
        _ => *slot = Some((boundary, distance)),
    }
}
