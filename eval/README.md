# Evaluation corpus and aggregation

This directory is a fixed, small quality fixture for supergrep. The original
fixtures are self-authored UTF-8 material. Ten code holdout intents are small,
verbatim UTF-8 excerpts from immutable commits of two permissively licensed
public Rust projects; their source URL, commit, license, original line range,
local path, and SHA-256 are recorded in [manifest.json](manifest.json).

## Contents

- corpus/ — self-authored code, configuration, documentation, log, and data
  fixtures plus ten attributed public-source code excerpts. It includes
  semantic decoys, duplicate/near-miss judgments, Korean/emoji text, and a
  431-byte JSONL line.
- queries.jsonl — one JSON object per query, designed for streaming parsers.
- manifest.json — immutable-fixture inventory, SHA-256 values, split metadata,
  label rules, and public snapshot provenance.
- ../scripts/evaluate.py — validates definitions and aggregates evaluator
  output. It does not search or score text itself.

The definition has 40 relevant intents: 20 code, 8 configuration, 8
documentation, and 4 log/data. Each intent has a paired English and Korean
query in the same split, producing 80 relevant query records. Five unrelated
English/Korean pairs add 10 negative-query records for descriptive score
distribution reporting. Development and holdout each contain 20 relevant
intents. Exactly 10 holdout code intents use five snippets from serde_json and
five from ripgrep, satisfying the plan's minimum of 10 public-source holdout
intents across two projects. At least 12 intent pairs are marked
low_lexical_overlap.

The public excerpts are evidence fixtures, not license substitutes. Preserve
their verbatim text and attribution fields when updating the corpus; acquire
new source only by immutable revision and keep it small and relevant.

## Definition JSONL

Every queries.jsonl record has this shape:

    {
      "schema_version": 1,
      "record_type": "query",
      "id": "dev-code-01-en",
      "intent_id": "dev-code-01",
      "split": "development",
      "category": "code",
      "language": "en",
      "query_type": "relevant",
      "query": "Where is the pause enlarged after repeated temporary delivery refusals?",
      "tags": ["low_lexical_overlap", "english_code"],
      "labels": [
        {
          "evidence_id": "ember-wait-after-refusal",
          "path": "service/ember.rs",
          "start_byte": 93,
          "end_byte": 304,
          "start_line": 4,
          "end_line": 8,
          "relevance": 2,
          "rationale": "The function grows a bounded delay from the number of failed attempts."
        }
      ]
    }

path is relative to eval/corpus; byte spans are 0-based, half-open; and line
spans are 1-based, inclusive. A chunk hits a positive evidence unit only when
it covers at least 50% of that label's byte span. Relevance 2 means a direct
answer, 1 means supporting context, and 0 is a judged near miss. An unrelated
query has no labels, which represents no answer evidence rather than an
implicit relevance threshold.

English/Korean variants must retain the identical label set. Do not add query
terms, translations, path hints, or gold spans to product code.

## Result JSONL contract

examples/evaluate.rs should emit one evaluation_result object for every
query_id and each of lexical, fast, and deep. Run one fixed model/profile per
result file. Relevant fields are:

    {
      "schema_version": 1,
      "record_type": "evaluation_result",
      "query_id": "dev-code-01-en",
      "mode": "fast",
      "chunk_contract_id": "model-chunks-v1",
      "all_chunk_spans": [
        {"path": "service/ember.rs", "start_byte": 93, "end_byte": 304}
      ],
      "candidate_spans": [
        {"path": "service/ember.rs", "start_byte": 93, "end_byte": 304}
      ],
      "evaluated_spans": [
        {"path": "service/ember.rs", "start_byte": 93, "end_byte": 304}
      ],
      "candidate_limit": 128,
      "scoring_complete": false,
      "results": [
        {
          "rank": 1,
          "path": "service/ember.rs",
          "start_byte": 93,
          "end_byte": 304,
          "score": 1.25
        }
      ]
    }

all_chunk_spans must be the same model-chunk set for lexical, fast, and deep
for a query. This prevents a different lexical chunker from inflating the
baseline. In fast mode, candidate_spans and evaluated_spans must be the same
selected subset of all_chunk_spans; their count is the actual candidate budget
used. In deep mode, scoring_complete must be true and evaluated_spans must
equal all_chunk_spans. Ranked result spans must have been evaluated. Paths and
spans are checked against the fixed corpus.

For a larger, deterministic sensitivity evaluation, copy the fixed fixtures
and add neutral distractors without changing any gold labels:

```bash
python3 -B scripts/generate_scale_corpus.py eval/corpus /tmp/supergrep-scale-corpus \
  --manifest /tmp/supergrep-scale-manifest.json --additional 240
python3 -B scripts/evaluate.py --check --corpus /tmp/supergrep-scale-corpus \
  --manifest /tmp/supergrep-scale-manifest.json
SUPERGREP_ORT_LIB=/absolute/path/libonnxruntime.so.1.20.0 \
  cargo run --release --locked --example evaluate -- \
  --split holdout --corpus /tmp/supergrep-scale-corpus \
  --cache-dir /path/to/prepared/model/cache \
  --output artifacts/evaluation/scale-holdout.jsonl
python3 -B scripts/evaluate.py --results artifacts/evaluation/scale-holdout.jsonl \
  --corpus /tmp/supergrep-scale-corpus \
  --manifest /tmp/supergrep-scale-manifest.json \
  --split holdout --output artifacts/evaluation/scale-holdout-summary.json
```

The scale corpus and its inventory stay outside `eval/corpus`, so the original
fixed evaluation remains unchanged. The evaluator reports both gold coverage
against all labels and candidate recall among labels that at least one model
chunk can represent. The former includes chunking losses; the latter isolates
candidate selection. The added corpus uses previously inspected holdout labels,
so its results are a sensitivity check, not a new blind holdout.

The script permits an evaluator to omit rank (array order becomes rank), but
emitting explicit 1-based ranks makes artifacts easier to inspect. Scores are
optional for ranking metrics, but finite numeric top scores are needed to
describe irrelevant-query score distributions.

## Commands

Validate the checked-in data, labels, UTF-8 encoding, pair integrity, SHA-256
inventory, and counts:

    python3 scripts/evaluate.py --check

After the Rust evaluator has emitted one JSONL file for a fixed run:

    python3 scripts/evaluate.py \
      --results artifacts/evaluation/compact-multilingual-development.jsonl \
      --split development \
      --output artifacts/evaluation/compact-multilingual-development-summary.json

The default is strict: all three modes are required for each selected query.
--allow-partial is only for bring-up diagnostics and must not be used for a
reported quality result.

The summary reports:

- Same-chunk-set lexical, fast, and deep Hit@5, MRR@10, and nDCG@10.
- Fast candidate evidence recall (macro) and query hit rate, together with the
  observed candidate-count distribution.
- Fast/deep top-10 Jaccard overlap and the deep-minus-fast Hit@5 delta.
- Breakdowns by split, language, category, and tags, including
  low_lexical_overlap and korean_to_english.
- Descriptive top-score distributions for unrelated queries only. It does not
  turn those values into a calibrated no-result threshold.

## Public snapshot provenance

The ten public-source holdout code intents from serde_json and ripgrep were
frozen before the first product holdout measurement. `manifest.json` records
the source URLs, immutable commits, licenses, original paths and line ranges,
local paths, hashes, and evidence spans. Keep those labels unchanged when
running either the original or larger-corpus evaluation. The original corpus
has only 40 model chunks, so its fast results do not test K=128 reduction;
the scale run above tests that path but reuses inspected labels and is a
sensitivity check rather than a new blind holdout.
