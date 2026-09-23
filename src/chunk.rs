//! Position-preserving source chunking.
//!
//! Chunks own only lightweight coordinates. Their passage text is always
//! recovered from the immutable [`Source`] snapshot, which prevents a second
//! filesystem read from changing result evidence.

use std::{ops::Range, path::PathBuf};

use thiserror::Error;

use crate::source::{Source, SourceId};

/// v0.1 never carries more than this many passage tokens into an adjacent
/// chunk. Keeping it public lets CLI validation and output diagnostics share
/// the same contract.
pub const MAX_OVERLAP_TOKENS: usize = 32;

/// Broad shape of the boundary selected for a chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChunkKind {
    /// Ended at a blank-line paragraph boundary.
    Paragraph,
    /// Ended at a normal line boundary.
    Lines,
    /// Had to split within a line (or a token-sized fragment) to guarantee
    /// forward progress for a very long line.
    LongLine,
}

/// A range in one [`Source`]. Byte bounds are 0-based and half-open; line
/// bounds are 1-based and inclusive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub source_id: SourceId,
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: usize,
    pub end_line: usize,
    pub kind: ChunkKind,
    pub context: Option<String>,
}

impl Chunk {
    pub fn new(source: &Source, range: Range<usize>, kind: ChunkKind) -> Result<Self, ChunkError> {
        let Some(text) = source.slice(range.clone()) else {
            return Err(ChunkError::InvalidRange {
                path: source.path().to_path_buf(),
                start: range.start,
                end: range.end,
            });
        };
        if text.is_empty() {
            return Err(ChunkError::EmptyRange {
                path: source.path().to_path_buf(),
            });
        }
        let Some((start_line, end_line)) = source.line_span(range.clone()) else {
            return Err(ChunkError::InvalidRange {
                path: source.path().to_path_buf(),
                start: range.start,
                end: range.end,
            });
        };
        Ok(Self {
            source_id: source.id(),
            start_byte: range.start,
            end_byte: range.end,
            start_line,
            end_line,
            kind,
            context: None,
        })
    }

    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    pub fn byte_range(&self) -> Range<usize> {
        self.start_byte..self.end_byte
    }

    /// Returns the exact snapshot passage only when it belongs to `source`.
    pub fn text<'a>(&self, source: &'a Source) -> Option<&'a str> {
        (self.source_id == source.id())
            .then(|| source.slice(self.byte_range()))
            .flatten()
    }
}

/// Initial chunking limits. Token count comes from a caller-provided function
/// so the same coordinate-preserving algorithm works both before and after a
/// model tokenizer is available.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkConfig {
    pub target_tokens: usize,
    pub overlap_tokens: usize,
    pub max_lines: usize,
    /// A hard byte ceiling that ensures a huge single lexical token (for
    /// example, minified code) still becomes finite chunks. A single Unicode
    /// scalar may exceed a tiny caller-supplied ceiling to avoid corrupting it.
    pub max_bytes: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            target_tokens: 160,
            overlap_tokens: 32,
            max_lines: 40,
            max_bytes: 1024,
        }
    }
}

impl ChunkConfig {
    pub fn validate(&self) -> Result<(), ChunkError> {
        if self.target_tokens == 0 {
            return Err(ChunkError::InvalidConfig(
                "target_tokens must be at least one".into(),
            ));
        }
        if self.overlap_tokens >= self.target_tokens {
            return Err(ChunkError::InvalidConfig(
                "overlap_tokens must be smaller than target_tokens".into(),
            ));
        }
        if self.overlap_tokens > MAX_OVERLAP_TOKENS {
            return Err(ChunkError::InvalidConfig(format!(
                "overlap_tokens must not exceed {MAX_OVERLAP_TOKENS}"
            )));
        }
        if self.max_lines == 0 {
            return Err(ChunkError::InvalidConfig(
                "max_lines must be at least one".into(),
            ));
        }
        if self.max_bytes == 0 {
            return Err(ChunkError::InvalidConfig(
                "max_bytes must be at least one".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum ChunkError {
    #[error("invalid chunk configuration: {0}")]
    InvalidConfig(String),

    #[error("invalid UTF-8 chunk range {start}..{end} for {}", path.display())]
    InvalidRange {
        path: PathBuf,
        start: usize,
        end: usize,
    },

    #[error("empty chunk range for {}", path.display())]
    EmptyRange { path: PathBuf },
}

/// Result of enforcing the engine-wide chunk cap. A reached cap is surfaced to
/// the search/output layer instead of silently claiming full coverage.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ChunkingReport {
    pub chunks: Vec<Chunk>,
    pub limit_reached: bool,
}

/// Chunks a source using a deterministic Unicode-aware estimate. Production
/// model search can use [`chunk_source_with_token_count`] with its exact
/// tokenizer counter before candidate selection.
pub fn chunk_source(source: &Source, config: ChunkConfig) -> Result<Vec<Chunk>, ChunkError> {
    chunk_source_with_token_count(source, config, estimated_token_count)
}

/// Chunks a source with a caller-supplied token counter. The counter must be
/// deterministic for deterministic chunk boundaries.
pub fn chunk_source_with_token_count<F>(
    source: &Source,
    config: ChunkConfig,
    token_count: F,
) -> Result<Vec<Chunk>, ChunkError>
where
    F: Fn(&str) -> usize,
{
    Ok(chunk_source_with_limit(source, config, usize::MAX, &token_count)?.chunks)
}

/// Internal bounded planner used by the corpus-wide cap. It stops as soon as
/// the next non-blank chunk would exceed `max_chunks`; unlike collecting a
/// source first and truncating afterward, it never allocates unneeded chunks.
fn chunk_source_with_limit<F>(
    source: &Source,
    config: ChunkConfig,
    max_chunks: usize,
    token_count: &F,
) -> Result<ChunkingReport, ChunkError>
where
    F: Fn(&str) -> usize,
{
    config.validate()?;
    if source.text().trim().is_empty() {
        return Ok(ChunkingReport::default());
    }

    let mut report = ChunkingReport::default();
    let mut start = 0usize;
    let mut covered_through = 0usize;
    while start < source.len_bytes() {
        let remaining = source
            .slice(start..source.len_bytes())
            .expect("chunk start is a UTF-8 boundary");
        // `trim_start` stops as soon as it sees content. Using `trim` here
        // would repeatedly rescan a long trailing-whitespace suffix for every
        // preceding chunk.
        if remaining.trim_start().is_empty() {
            break;
        }

        if let Some((end, kind)) =
            best_line_boundary(source, start, covered_through, config, token_count)
        {
            if let Some(chunk) = nonblank_chunk(source, start..end, kind)? {
                if report.chunks.len() == max_chunks {
                    report.limit_reached = true;
                    return Ok(report);
                }
                report.chunks.push(chunk);
            }
            covered_through = end;
            start = next_start_with_overlap(source, start, end, config, token_count);
            continue;
        }

        // An overlap can consume every permitted line/token slot. Retry from
        // the first uncovered byte before resorting to an intra-line split.
        if start < covered_through {
            start = covered_through;
            continue;
        }

        let end = best_character_boundary(source, start, config, token_count);
        debug_assert!(end > start);
        if let Some(chunk) = nonblank_chunk(source, start..end, ChunkKind::LongLine)? {
            if report.chunks.len() == max_chunks {
                report.limit_reached = true;
                return Ok(report);
            }
            report.chunks.push(chunk);
        }
        covered_through = end;
        start = next_start_with_overlap(source, start, end, config, token_count);
    }
    Ok(report)
}

/// Applies a shared cap across sources while preserving the deterministic
/// source/path order supplied by discovery.
pub fn chunk_sources(
    sources: &[Source],
    config: ChunkConfig,
    max_chunks: usize,
) -> Result<ChunkingReport, ChunkError> {
    chunk_sources_with_token_count(sources, config, max_chunks, estimated_token_count)
}

pub fn chunk_sources_with_token_count<F>(
    sources: &[Source],
    config: ChunkConfig,
    max_chunks: usize,
    token_count: F,
) -> Result<ChunkingReport, ChunkError>
where
    F: Fn(&str) -> usize,
{
    config.validate()?;
    if max_chunks == 0 {
        return Err(ChunkError::InvalidConfig(
            "max_chunks must be at least one".into(),
        ));
    }
    let mut report = ChunkingReport::default();
    for source in sources {
        let remaining = max_chunks.saturating_sub(report.chunks.len());
        let source_report = chunk_source_with_limit(source, config, remaining, &token_count)?;
        report.chunks.extend(source_report.chunks);
        if source_report.limit_reached {
            report.limit_reached = true;
            break;
        }
    }
    Ok(report)
}

fn nonblank_chunk(
    source: &Source,
    range: Range<usize>,
    kind: ChunkKind,
) -> Result<Option<Chunk>, ChunkError> {
    let text = source
        .slice(range.clone())
        .expect("planner produces valid UTF-8 ranges");
    if !text.trim().is_empty() {
        Ok(Some(Chunk::new(source, range, kind)?))
    } else {
        Ok(None)
    }
}

fn best_line_boundary<F>(
    source: &Source,
    start: usize,
    covered_through: usize,
    config: ChunkConfig,
    token_count: &F,
) -> Option<(usize, ChunkKind)>
where
    F: Fn(&str) -> usize,
{
    let mut latest = None;
    let mut latest_paragraph = None;
    // Starting at the current physical line is important for files with many
    // short lines: restarting at line 1 for every chunk turns a linear scan
    // into quadratic work under the 50,000-chunk safety limit.
    let first_line = source.line_of_byte(start)?;
    for line in first_line..=source.line_count() {
        let end = source.line_end(line)?;
        if end <= start || end <= covered_through {
            continue;
        }
        if !range_fits(source, start..end, config, token_count) {
            // Byte length and line count only grow. Tokenizers normally have
            // monotonic length too; continuing is still correct for a custom
            // counter with unusual behavior.
            if end - start > config.max_bytes
                || line_count_in_range(source, start..end) > config.max_lines
            {
                break;
            }
            continue;
        }
        latest = Some(end);
        if line_is_blank(source, line) {
            latest_paragraph = Some(end);
        }
    }
    // A fragment that began midway through a physical line remains a long-line
    // fragment even when its final piece happens to reach that line's newline.
    let begins_at_line_start = source
        .line_of_byte(start)
        .and_then(|line| source.line_start(line))
        == Some(start);
    if !begins_at_line_start {
        latest.map(|end| (end, ChunkKind::LongLine))
    } else {
        latest_paragraph
            .filter(|&end| {
                !source
                    .slice(start..end)
                    .expect("planner produces valid UTF-8 ranges")
                    .trim()
                    .is_empty()
            })
            .map(|end| (end, ChunkKind::Paragraph))
            .or_else(|| latest.map(|end| (end, ChunkKind::Lines)))
    }
}

fn best_character_boundary<F>(
    source: &Source,
    start: usize,
    config: ChunkConfig,
    token_count: &F,
) -> usize
where
    F: Fn(&str) -> usize,
{
    let text = source.text();
    let mut latest = None;
    let mut latest_whitespace = None;
    for (relative, character) in text[start..].char_indices() {
        let end = start + relative + character.len_utf8();
        let range = start..end;
        if range.end - range.start > config.max_bytes
            || line_count_in_range(source, range.clone()) > config.max_lines
        {
            break;
        }
        if token_count(source.slice(range.clone()).expect("char boundaries"))
            <= config.target_tokens
        {
            latest = Some(end);
            if character.is_whitespace() {
                latest_whitespace = Some(end);
            }
        } else if latest.is_some() {
            // Once an ordinary counter exceeds the target, later endpoints do
            // too. A custom non-monotonic counter can still use line splits.
            break;
        }
    }
    latest_whitespace
        .or(latest)
        .unwrap_or_else(|| first_character_end(text, start))
}

fn first_character_end(text: &str, start: usize) -> usize {
    let character = text[start..]
        .chars()
        .next()
        .expect("planner only asks for a non-empty suffix");
    start + character.len_utf8()
}

fn range_fits<F>(source: &Source, range: Range<usize>, config: ChunkConfig, token_count: &F) -> bool
where
    F: Fn(&str) -> usize,
{
    range.end - range.start <= config.max_bytes
        && line_count_in_range(source, range.clone()) <= config.max_lines
        && token_count(
            source
                .slice(range)
                .expect("planner produces valid UTF-8 ranges"),
        ) <= config.target_tokens
}

fn line_count_in_range(source: &Source, range: Range<usize>) -> usize {
    source
        .line_span(range)
        .map(|(start, end)| end - start + 1)
        .unwrap_or(usize::MAX)
}

fn line_is_blank(source: &Source, line: usize) -> bool {
    let start = source.line_start(line).expect("line is in range");
    let end = source.line_end(line).expect("line is in range");
    source
        .slice(start..end)
        .expect("line bounds are valid")
        .trim()
        .is_empty()
}

fn next_start_with_overlap<F>(
    source: &Source,
    chunk_start: usize,
    end: usize,
    config: ChunkConfig,
    token_count: &F,
) -> usize
where
    F: Fn(&str) -> usize,
{
    if config.overlap_tokens == 0 || end >= source.len_bytes() {
        return end;
    }
    let text = source.text();
    let mut best = end;
    for (relative, _) in text[chunk_start..end].char_indices().rev() {
        let candidate = chunk_start + relative;
        if end - candidate > config.max_bytes {
            break;
        }
        let tail = source.slice(candidate..end).expect("char boundary");
        if token_count(tail) <= config.overlap_tokens {
            best = candidate;
        } else {
            break;
        }
    }
    // Avoid repeating a complete tiny chunk forever.
    if best <= chunk_start {
        end
    } else {
        best
    }
}

/// Small deterministic fallback used by the lexical-only path. It counts
/// CJK characters and punctuation individually, while runs of word characters
/// count as one approximate token. It never changes the source text.
pub fn estimated_token_count(text: &str) -> usize {
    let mut count = 0usize;
    let mut in_word = false;
    for character in text.chars() {
        if character.is_whitespace() {
            in_word = false;
        } else if is_cjk(character)
            || character.is_ascii_punctuation()
            || (!character.is_alphanumeric() && character != '_')
        {
            count += 1;
            in_word = false;
        } else if !in_word {
            count += 1;
            in_word = true;
        }
    }
    count
}

fn is_cjk(character: char) -> bool {
    matches!(
        character as u32,
        0x3040..=0x30ff | 0x3400..=0x4dbf | 0x4e00..=0x9fff | 0xac00..=0xd7af | 0xf900..=0xfaff
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::SourceId;

    #[test]
    fn chunks_keep_crlf_unicode_and_byte_line_coordinates() {
        let text = "alpha\r\n한글 😀\r\n\r\nomega";
        let source = Source::from_text(SourceId::new(3), "memory", text);
        let config = ChunkConfig {
            target_tokens: 100,
            overlap_tokens: 0,
            max_lines: 2,
            max_bytes: 100,
        };
        let chunks = chunk_source(&source, config).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text(&source), Some("alpha\r\n한글 😀\r\n"));
        assert_eq!((chunks[0].start_byte, chunks[0].end_byte), (0, 20));
        assert_eq!((chunks[0].start_line, chunks[0].end_line), (1, 2));
        assert_eq!(chunks[1].text(&source), Some("\r\nomega"));
        assert_eq!((chunks[1].start_line, chunks[1].end_line), (3, 4));
    }

    #[test]
    fn long_unicode_line_splits_only_on_character_boundaries() {
        let source = Source::from_text(SourceId::new(0), "memory", "😀😀😀😀😀");
        let config = ChunkConfig {
            target_tokens: 99,
            overlap_tokens: 0,
            max_lines: 40,
            max_bytes: 8,
        };
        let chunks = chunk_source(&source, config).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(
            chunks
                .iter()
                .map(|chunk| chunk.text(&source).unwrap())
                .collect::<String>(),
            "😀😀😀😀😀"
        );
        assert!(chunks.iter().all(|chunk| chunk.kind == ChunkKind::LongLine));
        assert!(chunks
            .iter()
            .all(|chunk| source.slice(chunk.byte_range()).is_some()));
    }
}
