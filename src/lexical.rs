//! Deterministic, query-local lexical scoring.
//!
//! This module deliberately does not build a persistent inverted index.  A
//! [`LexicalCorpus`] keeps only the source/chunk snapshots; each query derives
//! term frequencies and document frequencies for *that query* while scanning
//! the current chunk set.  This keeps the lexical path honest about its memory
//! use and makes the result correspond exactly to the bytes searched.

use std::{
    collections::{HashMap, HashSet},
    fmt,
};

use unicode_normalization::UnicodeNormalization;

use crate::{
    chunk::Chunk,
    source::{Source, SourceId},
};

/// BM25's term-frequency saturation constant used by v0.1.
pub const BM25_K1: f64 = 1.2;

/// BM25's document-length normalization constant used by v0.1.
pub const BM25_B: f64 = 0.75;

/// A corpus cannot be scored if a chunk cannot be tied to its exact source
/// snapshot.  Treating that as a construction error avoids accidentally
/// scoring a path with text from a different file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LexicalError {
    DuplicateSourceId {
        source_id: SourceId,
    },
    MissingSource {
        chunk_index: usize,
        source_id: SourceId,
    },
    InvalidChunkRange {
        chunk_index: usize,
        source_id: SourceId,
    },
}

impl fmt::Display for LexicalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateSourceId { source_id } => {
                write!(formatter, "duplicate source id {}", source_id.get())
            }
            Self::MissingSource {
                chunk_index,
                source_id,
            } => write!(
                formatter,
                "chunk {chunk_index} references missing source id {}",
                source_id.get()
            ),
            Self::InvalidChunkRange {
                chunk_index,
                source_id,
            } => write!(
                formatter,
                "chunk {chunk_index} has an invalid range for source id {}",
                source_id.get()
            ),
        }
    }
}

impl std::error::Error for LexicalError {}

/// Immutable references needed to score one current scan.  The corpus owns no
/// normalized text or term-frequency index; `score` creates only
/// query-specific statistics and drops them before returning.
#[derive(Debug)]
pub struct LexicalCorpus<'a> {
    chunks: &'a [Chunk],
    sources: HashMap<SourceId, &'a Source>,
}

impl<'a> LexicalCorpus<'a> {
    /// Validates that every chunk points at one of `sources` and has a valid
    /// byte range in that exact snapshot.
    pub fn new(sources: &'a [Source], chunks: &'a [Chunk]) -> Result<Self, LexicalError> {
        let mut source_by_id = HashMap::with_capacity(sources.len());
        for source in sources {
            if source_by_id.insert(source.id(), source).is_some() {
                return Err(LexicalError::DuplicateSourceId {
                    source_id: source.id(),
                });
            }
        }
        for (chunk_index, chunk) in chunks.iter().enumerate() {
            let Some(source) = source_by_id.get(&chunk.source_id) else {
                return Err(LexicalError::MissingSource {
                    chunk_index,
                    source_id: chunk.source_id,
                });
            };
            if chunk.text(source).is_none() {
                return Err(LexicalError::InvalidChunkRange {
                    chunk_index,
                    source_id: chunk.source_id,
                });
            }
        }
        Ok(Self {
            chunks,
            sources: source_by_id,
        })
    }

    /// Number of model/lexical chunks in the current scan.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Calculates separate body and path BM25 scores for a query.  Only
    /// positive lexical hits are returned; callers selecting semantic
    /// candidates can use their original chunk list for zero-overlap sampling.
    ///
    /// Ties are stable by original chunk index.  Scores are lexical evidence,
    /// not probabilities and not comparable with model logits.
    pub fn score(&self, query: &str) -> LexicalRanking {
        let query_terms = distinct_terms(query);
        if query_terms.is_empty() || self.chunks.is_empty() {
            return LexicalRanking {
                query_terms,
                corpus_chunks: self.chunks.len(),
                average_body_length: 0.0,
                average_path_length: 0.0,
                scores: Vec::new(),
            };
        }

        let query_positions: HashMap<&str, usize> = query_terms
            .iter()
            .enumerate()
            .map(|(position, term)| (term.as_str(), position))
            .collect();
        let mut body_document_frequency = vec![0usize; query_terms.len()];
        let mut path_document_frequency = vec![0usize; query_terms.len()];
        let mut total_body_length = 0usize;
        let mut total_path_length = 0usize;
        let mut documents = Vec::with_capacity(self.chunks.len());

        for (chunk_index, chunk) in self.chunks.iter().enumerate() {
            let source = self.sources[&chunk.source_id];
            let body_terms = normalized_terms(
                chunk
                    .text(source)
                    .expect("LexicalCorpus::new validates every chunk range"),
            );
            let path_terms = normalized_terms(&source.path().to_string_lossy());
            let body_counts = query_counts(&body_terms, &query_positions);
            let path_counts = query_counts(&path_terms, &query_positions);
            for (position, count) in body_counts.iter().enumerate() {
                if *count > 0 {
                    body_document_frequency[position] += 1;
                }
            }
            for (position, count) in path_counts.iter().enumerate() {
                if *count > 0 {
                    path_document_frequency[position] += 1;
                }
            }
            total_body_length += body_terms.len();
            total_path_length += path_terms.len();
            documents.push(DocumentStats {
                chunk_index,
                source_id: chunk.source_id,
                body_length: body_terms.len(),
                path_length: path_terms.len(),
                body_counts,
                path_counts,
            });
        }

        let document_count = documents.len();
        let average_body_length = total_body_length as f64 / document_count as f64;
        let average_path_length = total_path_length as f64 / document_count as f64;
        let body_idf = idf_values(document_count, &body_document_frequency);
        let path_idf = idf_values(document_count, &path_document_frequency);
        let mut scores = Vec::new();
        for document in documents {
            let body_score = bm25_score(
                &document.body_counts,
                &body_idf,
                document.body_length,
                average_body_length,
            );
            let path_score = bm25_score(
                &document.path_counts,
                &path_idf,
                document.path_length,
                average_path_length,
            );
            let score = body_score + path_score;
            if score > 0.0 {
                scores.push(LexicalScore {
                    chunk_index: document.chunk_index,
                    source_id: document.source_id,
                    body_score,
                    path_score,
                    score,
                });
            }
        }
        scores.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| right.body_score.total_cmp(&left.body_score))
                .then_with(|| right.path_score.total_cmp(&left.path_score))
                .then_with(|| left.chunk_index.cmp(&right.chunk_index))
        });

        LexicalRanking {
            query_terms,
            corpus_chunks: document_count,
            average_body_length,
            average_path_length,
            scores,
        }
    }
}

#[derive(Debug)]
struct DocumentStats {
    chunk_index: usize,
    source_id: SourceId,
    body_length: usize,
    path_length: usize,
    body_counts: Vec<usize>,
    path_counts: Vec<usize>,
}

/// A ranked lexical result.  `scores` contains only chunks with a positive
/// body or path score, in deterministic order.
#[derive(Debug, Clone, PartialEq)]
pub struct LexicalRanking {
    pub query_terms: Vec<String>,
    pub corpus_chunks: usize,
    pub average_body_length: f64,
    pub average_path_length: f64,
    pub scores: Vec<LexicalScore>,
}

impl LexicalRanking {
    /// True when at least one chunk had lexical evidence for the query.
    pub fn has_evidence(&self) -> bool {
        !self.scores.is_empty()
    }
}

/// One chunk's lexical evidence. `score` is `body_score + path_score` solely
/// for lexical ordering; each component remains available to callers that
/// need to explain why a candidate was selected.
#[derive(Debug, Clone, PartialEq)]
pub struct LexicalScore {
    /// Index in the `chunks` slice passed to [`LexicalCorpus::new`].
    pub chunk_index: usize,
    pub source_id: SourceId,
    pub body_score: f64,
    pub path_score: f64,
    pub score: f64,
}

/// Normalizes text with Unicode NFKC and lowercase mapping, then emits whole
/// identifier tokens as well as their snake_case/camelCase/digit components.
/// For example, `retryDelay_ms2` yields `retrydelay_ms2`, `retry`, `delay`,
/// `ms`, and `2`. Korean and other non-Latin letter runs remain valid terms.
pub fn normalized_terms(text: &str) -> Vec<String> {
    let compatibility = text.nfkc().collect::<String>();
    let mut terms = Vec::new();
    let mut segment_start = None;
    for (offset, character) in compatibility.char_indices() {
        if character.is_alphanumeric() || character == '_' {
            segment_start.get_or_insert(offset);
        } else if let Some(start) = segment_start.take() {
            push_identifier_terms(&compatibility[start..offset], &mut terms);
        }
    }
    if let Some(start) = segment_start {
        push_identifier_terms(&compatibility[start..], &mut terms);
    }
    terms
}

fn distinct_terms(query: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    normalized_terms(query)
        .into_iter()
        .filter(|term| seen.insert(term.clone()))
        .collect()
}

fn push_identifier_terms(identifier: &str, terms: &mut Vec<String>) {
    if identifier.chars().all(|character| character == '_') {
        return;
    }
    let whole = lowercase(identifier);
    if !whole.is_empty() {
        terms.push(whole.clone());
    }

    let mut components = Vec::new();
    let characters = identifier.char_indices().collect::<Vec<_>>();
    let mut part_start = 0usize;
    for position in 0..characters.len() {
        let (offset, character) = characters[position];
        if character == '_' {
            push_component(identifier, part_start, offset, &mut components);
            part_start = offset + character.len_utf8();
            continue;
        }
        if offset > part_start && component_boundary(&characters, position) {
            push_component(identifier, part_start, offset, &mut components);
            part_start = offset;
        }
    }
    push_component(identifier, part_start, identifier.len(), &mut components);
    // Plain words have exactly one component equal to their preserved whole
    // form. Do not turn every ordinary word into an artificial tf=2 hit.
    terms.extend(
        components
            .into_iter()
            .filter(|component| component != &whole),
    );
}

fn lowercase(value: &str) -> String {
    value.nfkc().flat_map(char::to_lowercase).collect()
}

fn push_component(identifier: &str, start: usize, end: usize, terms: &mut Vec<String>) {
    if start < end {
        let component = lowercase(&identifier[start..end]);
        if !component.is_empty() {
            terms.push(component);
        }
    }
}

fn component_boundary(characters: &[(usize, char)], position: usize) -> bool {
    let previous = characters[position - 1].1;
    let current = characters[position].1;
    if previous.is_ascii_digit() != current.is_ascii_digit()
        && (previous.is_alphanumeric() && current.is_alphanumeric())
    {
        return true;
    }
    if previous.is_lowercase() && current.is_uppercase() {
        return true;
    }
    // `HTTPServer` should expose both `http` and `server`, not only the
    // unsplit whole identifier.  Split before the last capital if it starts a
    // normal title-cased word.
    current.is_uppercase()
        && previous.is_uppercase()
        && characters
            .get(position + 1)
            .is_some_and(|(_, next)| next.is_lowercase())
}

fn query_counts(terms: &[String], query_positions: &HashMap<&str, usize>) -> Vec<usize> {
    let mut counts = vec![0usize; query_positions.len()];
    for term in terms {
        if let Some(position) = query_positions.get(term.as_str()) {
            counts[*position] += 1;
        }
    }
    counts
}

fn idf_values(document_count: usize, document_frequency: &[usize]) -> Vec<f64> {
    document_frequency
        .iter()
        .map(|frequency| {
            ((document_count as f64 - *frequency as f64 + 0.5) / (*frequency as f64 + 0.5) + 1.0)
                .ln()
        })
        .collect()
}

fn bm25_score(counts: &[usize], idf: &[f64], document_length: usize, average_length: f64) -> f64 {
    if average_length == 0.0 {
        return 0.0;
    }
    counts
        .iter()
        .zip(idf)
        .filter(|(count, _)| **count > 0)
        .map(|(count, idf)| {
            let frequency = *count as f64;
            let denominator = frequency
                + BM25_K1 * (1.0 - BM25_B + BM25_B * document_length as f64 / average_length);
            idf * frequency * (BM25_K1 + 1.0) / denominator
        })
        .sum()
}
