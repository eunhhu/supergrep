//! Deterministic candidate selection and reranked result assembly.
//!
//! This module deliberately has no filesystem, runtime, or tokenizer policy.
//! It receives the exact [`Source`] snapshots and already-fitted [`Chunk`]s
//! produced by those layers, then keeps every selected chunk index attached to
//! its original passage while calling a [`Scorer`].  That separation lets the
//! product use a local ONNX scorer while tests exercise the same selection and
//! ranking contract with a small fake.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use crate::{
    chunk::Chunk,
    lexical::{LexicalCorpus, LexicalError, LexicalRanking, LexicalScore},
    model::Scorer,
    source::{Source, SourceId},
    Result, SupergrepError,
};

/// How a query is evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// Return only chunks with positive query-local lexical evidence.  A
    /// scorer is neither required nor called in this mode.
    Lexical,
    /// Score a deterministic subset of chunks when the corpus is larger than
    /// the configured candidate budget.
    Fast,
    /// Score every supplied chunk.  Result display may still be capped by
    /// `top_k`, but inference is never silently capped.
    Deep,
}

/// Bounded search parameters.  These limits describe the chunks passed to the
/// scorer, not a pre-built index or a hidden filtering pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchConfig {
    pub mode: SearchMode,
    /// Maximum fast-mode candidate count.  Ignored by lexical and deep mode.
    pub candidate_limit: usize,
    /// Maximum number of de-duplicated entries returned to a caller.
    pub top_k: usize,
    /// Maximum number of passages in one scorer call.
    pub batch_size: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            mode: SearchMode::Fast,
            candidate_limit: 128,
            top_k: 10,
            batch_size: 4,
        }
    }
}

/// The integer split used for a fast-mode candidate budget.
///
/// The intended weights are body 5/8 and path/neighbour/diverse 1/8 each.
/// Integer floors are assigned first; remaining slots use largest remainders,
/// breaking a remainder tie in that same order.  Thus the allocation always
/// sums to `candidate_limit` without random rounding.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CandidateQuota {
    pub body: usize,
    pub path: usize,
    pub neighbours: usize,
    pub diverse: usize,
}

impl CandidateQuota {
    fn for_limit(limit: usize) -> Self {
        const WEIGHTS: [usize; 4] = [5, 1, 1, 1];
        let mut allocations = WEIGHTS.map(|weight| limit.saturating_mul(weight) / 8);
        let allocated = allocations.iter().sum::<usize>();
        let mut remaining = limit.saturating_sub(allocated);
        let mut order = (0..WEIGHTS.len()).collect::<Vec<_>>();
        order.sort_by(|left, right| {
            let left_remainder = limit.saturating_mul(WEIGHTS[*left]) % 8;
            let right_remainder = limit.saturating_mul(WEIGHTS[*right]) % 8;
            right_remainder
                .cmp(&left_remainder)
                .then_with(|| left.cmp(right))
        });
        for index in order {
            if remaining == 0 {
                break;
            }
            allocations[index] += 1;
            remaining -= 1;
        }
        Self {
            body: allocations[0],
            path: allocations[1],
            neighbours: allocations[2],
            diverse: allocations[3],
        }
    }
}

/// Audit information about fast-mode candidate selection.
///
/// A category can select fewer chunks than its quota when its candidates were
/// already selected by an earlier category.  Remaining slots are then filled
/// first from all positive lexical hits, then from the deterministic diverse
/// order; this fact is visible through `selected` and `candidate_limit`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CandidateSelection {
    pub candidate_limit: usize,
    pub quota: CandidateQuota,
    pub body_selected: usize,
    pub path_selected: usize,
    pub neighbours_selected: usize,
    pub diverse_selected: usize,
}

/// A result retains both query-local lexical evidence and a raw model logit.
/// They are intentionally separate kinds of values and must never be
/// displayed or compared as a common probability scale.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedEntry {
    pub chunk_index: usize,
    pub source_id: SourceId,
    pub path: std::path::PathBuf,
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: usize,
    pub end_line: usize,
    pub lexical_score: Option<f64>,
    pub lexical_body_score: Option<f64>,
    pub lexical_path_score: Option<f64>,
    /// Raw relevance logit returned by the selected model.  It is `None` for
    /// lexical mode, and is never converted to a probability.
    pub relevance_logit: Option<f32>,
}

/// Full structured evidence for one search.  `evaluated_chunk_indexes`
/// records model evaluations only: it is empty for lexical mode.  In fast
/// mode, `scoring_complete` becomes true exactly when the candidate set is
/// the full chunk set (`N <= K`); in deep mode it is always true.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchReport {
    pub mode: SearchMode,
    pub all_chunk_indexes: Vec<usize>,
    pub selected_chunk_indexes: Vec<usize>,
    pub evaluated_chunk_indexes: Vec<usize>,
    pub lexical_evidence: bool,
    pub scoring_complete: Option<bool>,
    pub deduplicated_count: usize,
    pub candidate_selection: Option<CandidateSelection>,
    pub results: Vec<RankedEntry>,
}

/// A validated view of the current source/chunk snapshots.
///
/// Construction checks that every chunk references the exact source snapshot
/// from which its passage will be recovered.  It does not build a persistent
/// lexical index: [`search`](Self::search) obtains a fresh query-local BM25
/// ranking each time.
#[derive(Debug)]
pub struct SearchEngine<'a> {
    sources: HashMap<SourceId, &'a Source>,
    chunks: &'a [Chunk],
    lexical: LexicalCorpus<'a>,
}

impl<'a> SearchEngine<'a> {
    pub fn new(
        sources: &'a [Source],
        chunks: &'a [Chunk],
    ) -> std::result::Result<Self, LexicalError> {
        let lexical = LexicalCorpus::new(sources, chunks)?;
        let sources = sources.iter().map(|source| (source.id(), source)).collect();
        Ok(Self {
            sources,
            chunks,
            lexical,
        })
    }

    /// Runs exactly the requested mode.  Model modes require an explicit
    /// scorer; callers must not pass a lexical fallback when a model failed to
    /// load or infer.
    pub fn search(
        &self,
        query: &str,
        config: SearchConfig,
        scorer: Option<&dyn Scorer>,
    ) -> Result<SearchReport> {
        let lexical = self.lexical.score(query);
        let all_chunk_indexes = (0..self.chunks.len()).collect::<Vec<_>>();
        match config.mode {
            SearchMode::Lexical => {
                Ok(self.search_lexical(lexical, all_chunk_indexes, config.top_k))
            }
            SearchMode::Fast | SearchMode::Deep => {
                if config.batch_size == 0 {
                    return Err(SupergrepError::Input(
                        "batch_size must be at least one for model search".into(),
                    ));
                }
                let scorer = scorer.ok_or_else(|| {
                    SupergrepError::Model(
                        "a local scorer is required for fast or deep search; use --lexical for model-free search".into(),
                    )
                })?;
                let (selected_chunk_indexes, candidate_selection, scoring_complete) = match config
                    .mode
                {
                    SearchMode::Fast => self.fast_candidates(&lexical, config.candidate_limit)?,
                    SearchMode::Deep => (all_chunk_indexes.clone(), None, true),
                    SearchMode::Lexical => unreachable!("handled above"),
                };
                let scored =
                    self.score_selected(query, &selected_chunk_indexes, config.batch_size, scorer)?;
                let (mut results, deduplicated_count) = self.model_results(&lexical, scored);
                results.truncate(config.top_k);
                Ok(SearchReport {
                    mode: config.mode,
                    all_chunk_indexes,
                    evaluated_chunk_indexes: selected_chunk_indexes.clone(),
                    selected_chunk_indexes,
                    lexical_evidence: lexical.has_evidence(),
                    scoring_complete: Some(scoring_complete),
                    deduplicated_count,
                    candidate_selection,
                    results,
                })
            }
        }
    }

    fn search_lexical(
        &self,
        lexical: LexicalRanking,
        all_chunk_indexes: Vec<usize>,
        top_k: usize,
    ) -> SearchReport {
        let selected_chunk_indexes = lexical
            .scores
            .iter()
            .map(|score| score.chunk_index)
            .collect::<Vec<_>>();
        let mut entries = lexical
            .scores
            .iter()
            .map(|score| self.entry_from_lexical(score))
            .collect::<Vec<_>>();
        entries.sort_by(lexical_entry_order);
        let (mut results, deduplicated_count) = self.suppress_overlaps(entries, false);
        results.truncate(top_k);
        SearchReport {
            mode: SearchMode::Lexical,
            all_chunk_indexes,
            selected_chunk_indexes,
            evaluated_chunk_indexes: Vec::new(),
            lexical_evidence: lexical.has_evidence(),
            scoring_complete: None,
            deduplicated_count,
            candidate_selection: None,
            results,
        }
    }

    fn fast_candidates(
        &self,
        lexical: &LexicalRanking,
        candidate_limit: usize,
    ) -> Result<(Vec<usize>, Option<CandidateSelection>, bool)> {
        if candidate_limit == 0 {
            return Err(SupergrepError::Input(
                "candidate_limit must be at least one for fast search".into(),
            ));
        }
        if self.chunks.len() <= candidate_limit {
            return Ok(((0..self.chunks.len()).collect(), None, true));
        }

        let quota = CandidateQuota::for_limit(candidate_limit);
        let mut audit = CandidateSelection {
            candidate_limit,
            quota,
            ..CandidateSelection::default()
        };
        let mut selected = Vec::with_capacity(candidate_limit);
        let mut seen = HashSet::with_capacity(candidate_limit);

        let body = self.sorted_lexical(lexical, |score| score.body_score);
        audit.body_selected = add_until(&mut selected, &mut seen, &body, quota.body);

        let path = self.sorted_lexical(lexical, |score| score.path_score);
        audit.path_selected = add_until(&mut selected, &mut seen, &path, quota.path);

        let neighbours = self.neighbour_order(lexical);
        audit.neighbours_selected =
            add_until(&mut selected, &mut seen, &neighbours, quota.neighbours);

        let diverse = self.diverse_order();
        audit.diverse_selected = add_until(&mut selected, &mut seen, &diverse, quota.diverse);

        // A category that could not meet its quota cannot leave a silent hole:
        // first use every remaining positive lexical candidate in ranking
        // order, then cover the rest with the stable broad sample.
        let lexical_order = lexical
            .scores
            .iter()
            .map(|score| score.chunk_index)
            .collect::<Vec<_>>();
        let remaining_after_quotas = candidate_limit.saturating_sub(selected.len());
        add_until(
            &mut selected,
            &mut seen,
            &lexical_order,
            remaining_after_quotas,
        );
        let remaining_after_lexical = candidate_limit.saturating_sub(selected.len());
        add_until(&mut selected, &mut seen, &diverse, remaining_after_lexical);
        debug_assert_eq!(selected.len(), candidate_limit);
        Ok((selected, Some(audit), false))
    }

    fn sorted_lexical<F>(&self, lexical: &LexicalRanking, field: F) -> Vec<usize>
    where
        F: Fn(&LexicalScore) -> f64,
    {
        let mut scores = lexical
            .scores
            .iter()
            .filter(|score| field(score) > 0.0)
            .collect::<Vec<_>>();
        scores.sort_by(|left, right| {
            field(right)
                .total_cmp(&field(left))
                .then_with(|| self.chunk_order(left.chunk_index, right.chunk_index))
        });
        scores.into_iter().map(|score| score.chunk_index).collect()
    }

    /// Direct neighbours are calculated within one source's byte order, never
    /// from adjacent global indexes (which could accidentally cross files).
    fn neighbour_order(&self, lexical: &LexicalRanking) -> Vec<usize> {
        let groups = self.source_chunk_order();
        let mut locations = HashMap::new();
        for indexes in groups.values() {
            for (position, chunk_index) in indexes.iter().copied().enumerate() {
                locations.insert(chunk_index, (indexes, position));
            }
        }
        let mut neighbours = Vec::new();
        for hit in &lexical.scores {
            let Some((indexes, position)) = locations.get(&hit.chunk_index) else {
                continue;
            };
            if *position > 0 {
                neighbours.push(indexes[*position - 1]);
            }
            if let Some(next) = indexes.get(*position + 1) {
                neighbours.push(*next);
            }
        }
        neighbours
    }

    /// Spreads the first pass across the entire sorted path range, then visits
    /// progressively finer path ranges. Within each file, visit the middle
    /// chunk before expanding toward its edges. This matters when there are
    /// more paths than candidate slots: path-order truncation must not always
    /// discard the final paths.
    fn diverse_order(&self) -> Vec<usize> {
        let groups = self.source_chunk_order();
        let per_source = groups.into_values().map(midpoint_order).collect::<Vec<_>>();
        let path_order = spread_order((0..per_source.len()).collect());
        let mut order = Vec::with_capacity(self.chunks.len());
        let mut position = 0usize;
        loop {
            let mut added = false;
            for &path_index in &path_order {
                let indexes = &per_source[path_index];
                if let Some(chunk_index) = indexes.get(position) {
                    order.push(*chunk_index);
                    added = true;
                }
            }
            if !added {
                break;
            }
            position += 1;
        }
        order
    }

    fn source_chunk_order(&self) -> BTreeMap<SourceOrderingKey, Vec<usize>> {
        let mut groups = BTreeMap::<SourceOrderingKey, Vec<usize>>::new();
        for (chunk_index, chunk) in self.chunks.iter().enumerate() {
            let source = self.sources[&chunk.source_id];
            groups
                .entry(SourceOrderingKey {
                    path: source.path().to_path_buf(),
                    source_id: source.id(),
                })
                .or_default()
                .push(chunk_index);
        }
        for indexes in groups.values_mut() {
            indexes.sort_by(|left, right| self.chunk_order(*left, *right));
        }
        groups
    }

    fn score_selected(
        &self,
        query: &str,
        selected: &[usize],
        batch_size: usize,
        scorer: &dyn Scorer,
    ) -> Result<Vec<(usize, f32)>> {
        let mut scored = Vec::with_capacity(selected.len());
        for batch_indexes in selected.chunks(batch_size) {
            let passages = batch_indexes
                .iter()
                .map(|chunk_index| self.passage(*chunk_index).to_owned())
                .collect::<Vec<_>>();
            let scores = scorer.score_batch(query, &passages)?;
            if scores.len() != batch_indexes.len() {
                return Err(SupergrepError::Model(format!(
                    "scorer returned {} scores for a batch of {} passages",
                    scores.len(),
                    batch_indexes.len()
                )));
            }
            for (chunk_index, score) in batch_indexes.iter().copied().zip(scores) {
                if !score.is_finite() {
                    return Err(SupergrepError::Model(format!(
                        "scorer returned a non-finite relevance logit for chunk {chunk_index}"
                    )));
                }
                scored.push((chunk_index, score));
            }
        }
        Ok(scored)
    }

    fn model_results(
        &self,
        lexical: &LexicalRanking,
        scored: Vec<(usize, f32)>,
    ) -> (Vec<RankedEntry>, usize) {
        let lexical_by_chunk = lexical
            .scores
            .iter()
            .map(|score| (score.chunk_index, score))
            .collect::<HashMap<_, _>>();
        let mut entries = scored
            .into_iter()
            .map(|(chunk_index, relevance_logit)| {
                let lexical = lexical_by_chunk.get(&chunk_index).copied();
                self.entry_from_scores(chunk_index, lexical, Some(relevance_logit))
            })
            .collect::<Vec<_>>();
        entries.sort_by(model_entry_order);
        self.suppress_overlaps(entries, true)
    }

    fn entry_from_lexical(&self, score: &LexicalScore) -> RankedEntry {
        self.entry_from_scores(score.chunk_index, Some(score), None)
    }

    fn entry_from_scores(
        &self,
        chunk_index: usize,
        lexical: Option<&LexicalScore>,
        relevance_logit: Option<f32>,
    ) -> RankedEntry {
        let chunk = &self.chunks[chunk_index];
        let source = self.sources[&chunk.source_id];
        RankedEntry {
            chunk_index,
            source_id: chunk.source_id,
            path: source.path().to_path_buf(),
            start_byte: chunk.start_byte,
            end_byte: chunk.end_byte,
            start_line: chunk.start_line,
            end_line: chunk.end_line,
            lexical_score: lexical.map(|score| score.score),
            lexical_body_score: lexical.map(|score| score.body_score),
            lexical_path_score: lexical.map(|score| score.path_score),
            relevance_logit,
        }
    }

    fn suppress_overlaps(
        &self,
        entries: Vec<RankedEntry>,
        _model: bool,
    ) -> (Vec<RankedEntry>, usize) {
        let mut kept = Vec::with_capacity(entries.len());
        let mut deduplicated_count = 0usize;
        for entry in entries {
            let overlaps_kept = kept.iter().any(|existing: &RankedEntry| {
                existing.source_id == entry.source_id && overlap_ratio(existing, &entry) >= 0.6
            });
            if overlaps_kept {
                deduplicated_count += 1;
            } else {
                kept.push(entry);
            }
        }
        (kept, deduplicated_count)
    }

    fn passage(&self, chunk_index: usize) -> &str {
        let chunk = &self.chunks[chunk_index];
        chunk
            .text(self.sources[&chunk.source_id])
            .expect("SearchEngine::new validates every chunk range")
    }

    fn chunk_order(&self, left: usize, right: usize) -> std::cmp::Ordering {
        let left_chunk = &self.chunks[left];
        let right_chunk = &self.chunks[right];
        let left_source = self.sources[&left_chunk.source_id];
        let right_source = self.sources[&right_chunk.source_id];
        left_source
            .path()
            .cmp(right_source.path())
            .then_with(|| left_chunk.start_byte.cmp(&right_chunk.start_byte))
            .then_with(|| left_chunk.end_byte.cmp(&right_chunk.end_byte))
            .then_with(|| left_chunk.source_id.cmp(&right_chunk.source_id))
            .then_with(|| left.cmp(&right))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SourceOrderingKey {
    path: std::path::PathBuf,
    source_id: SourceId,
}

fn add_until(
    selected: &mut Vec<usize>,
    seen: &mut HashSet<usize>,
    candidates: &[usize],
    amount: usize,
) -> usize {
    let mut added = 0usize;
    for chunk_index in candidates {
        if added == amount {
            break;
        }
        if seen.insert(*chunk_index) {
            selected.push(*chunk_index);
            added += 1;
        }
    }
    added
}

fn midpoint_order(indexes: Vec<usize>) -> Vec<usize> {
    let mut output = Vec::with_capacity(indexes.len());
    let mut ranges = VecDeque::from([(0usize, indexes.len())]);
    while let Some((start, end)) = ranges.pop_front() {
        if start >= end {
            continue;
        }
        let middle = start + (end - start) / 2;
        output.push(indexes[middle]);
        ranges.push_back((start, middle));
        ranges.push_back((middle + 1, end));
    }
    output
}

fn spread_order(indexes: Vec<usize>) -> Vec<usize> {
    match indexes.len() {
        0 | 1 => return indexes,
        _ => {}
    }
    let mut output = Vec::with_capacity(indexes.len());
    output.push(indexes[0]);
    output.push(indexes[indexes.len() - 1]);
    let mut ranges = VecDeque::from([(1usize, indexes.len() - 1)]);
    while let Some((start, end)) = ranges.pop_front() {
        if start >= end {
            continue;
        }
        let middle = start + (end - start) / 2;
        output.push(indexes[middle]);
        ranges.push_back((start, middle));
        ranges.push_back((middle + 1, end));
    }
    output
}

fn model_entry_order(left: &RankedEntry, right: &RankedEntry) -> std::cmp::Ordering {
    right
        .relevance_logit
        .expect("model entries always have a score")
        .total_cmp(
            &left
                .relevance_logit
                .expect("model entries always have a score"),
        )
        .then_with(|| path_byte_order(left, right))
}

fn lexical_entry_order(left: &RankedEntry, right: &RankedEntry) -> std::cmp::Ordering {
    right
        .lexical_score
        .expect("lexical entries always have a score")
        .total_cmp(
            &left
                .lexical_score
                .expect("lexical entries always have a score"),
        )
        .then_with(|| path_byte_order(left, right))
}

fn path_byte_order(left: &RankedEntry, right: &RankedEntry) -> std::cmp::Ordering {
    left.path
        .cmp(&right.path)
        .then_with(|| left.start_byte.cmp(&right.start_byte))
        .then_with(|| left.end_byte.cmp(&right.end_byte))
        .then_with(|| left.source_id.cmp(&right.source_id))
        .then_with(|| left.chunk_index.cmp(&right.chunk_index))
}

fn overlap_ratio(left: &RankedEntry, right: &RankedEntry) -> f64 {
    let overlap_start = left.start_byte.max(right.start_byte);
    let overlap_end = left.end_byte.min(right.end_byte);
    let overlap = overlap_end.saturating_sub(overlap_start);
    let shortest = (left.end_byte - left.start_byte).min(right.end_byte - right.start_byte);
    if shortest == 0 {
        0.0
    } else {
        overlap as f64 / shortest as f64
    }
}
