# supergrep v0.1

`supergrep` searches UTF-8 source code, documentation, and configuration with a
local ONNX cross-encoder. It has a model-free lexical mode, a bounded fast mode,
and `--deep`, which scores every constructed model chunk. Results point back to
the exact source snapshot with 0-based half-open byte offsets and 1-based
inclusive line numbers.

The product is a Rust CLI. Python is only used by the development evaluation and
benchmark scripts; it is not required to search.

## Supported bundle

The checked bundle targets Linux ARM64 (`aarch64`) with glibc. The measured
machine had four Cortex-A76 CPUs and 7.9 GiB RAM. The CLI bundle includes the
measured ONNX Runtime 1.20.0 shared libraries and their notices; the model is
downloaded separately into the user's cache. The ONNX Runtime CPU package and
model are third-party components with their own notices/licenses.

After unpacking the local bundle, prepare the pinned multilingual model once:

```bash
./bin/supergrep model list
./bin/supergrep model download compact-multilingual
./bin/supergrep model verify compact-multilingual
./bin/supergrep doctor
```

Only `model download` accesses the network. Search, `model list`, `model
verify`, `doctor`, and lexical mode do not download anything or call an API.
Pass `--cache-dir DIRECTORY` to any model/cache command to select a cache
explicitly. The default is the platform cache under `supergrep/models`.

## Search

```bash
./bin/supergrep "where are failed requests retried?" ./my-project
./bin/supergrep "요청이 반복 실패하면 지연 시간을 늘리는 위치" ./my-project --deep
./bin/supergrep "retry delay" ./my-project --lexical
./bin/supergrep "how are ignore rules normalized?" ./my-project --top-k 5 --json
./bin/supergrep "cache invalidation" ./my-project --glob '*.rs' --candidates 256
```

The default profile is `compact-multilingual` (pinned revision
`1427fd652930e4ba29e8149678df786c240d8825`). `tiny-en` is an English-only
comparison profile and is not a Korean-capable substitute. `--model` accepts a
built-in profile or a complete local directory that verifies byte-for-byte
against one of the built-in immutable profiles.

Default fast search chooses at most 128 chunks. It scans and reads the bounded
text corpus each run, but scores only the selected candidates. `--deep` scores
all constructed chunks and may take substantially longer. `--lexical` skips
model/runtime setup and returns only positive BM25 matches. Fast search reports
when it has no lexical evidence and whether it evaluated the complete chunk
set. Neither a model score nor a lexical score is a calibrated relevance
probability; the JSON `score_kind` is `raw_logit` or `bm25`.

Default scan policy honors ignore rules, excludes hidden files and `.git`, does
not follow symlinks, and accepts non-binary UTF-8 text. Defaults are 2 MiB per
file, 64 MiB read total, and 50,000 chunks. Partial scans and reached limits are
reported in diagnostics and affect the exit code. Use `--hidden`, `--no-ignore`,
repeatable `--glob`, and explicit limit options to change the scan policy.

JSON mode writes one versioned JSON object to stdout. Diagnostics and model
errors go to stderr. A valid result can contain escaped `path_display`; for a
non-UTF-8 Linux path, `path` is null and `path_bytes_base64` preserves the raw
name.

Exit status: 0 means results returned with a complete scan; 1 means a complete
run returned no output rows; 2 means input/model/runtime/other fatal error; 3
means a resource or operational issue left coverage partial. Fast candidate
reduction by itself is not an error and does not set `partial`.

## Build from source

```bash
cargo build --release --locked
cargo test --locked
```

For development outside the packaged bundle, set `SUPERGREP_ORT_LIB` to an
absolute path to the exact ONNX Runtime 1.20.0 library, or place
`libonnxruntime.so.1.20.0` under `runtime/` next to the executable. A
`SUPERGREP_MODEL_CACHE` environment override is also supported. Search never
uses the current working directory to discover ONNX Runtime.

## Evaluation and measured limits

The fixed corpus, query definitions, immutable public-source attributions, and
evaluation contract are in [`eval/`](eval/). Reproduce the summaries after
preparing the model:

```bash
cargo run --release --locked --example evaluate -- \
  --split development --output artifacts/evaluation/development.jsonl
python3 -B scripts/evaluate.py --results artifacts/evaluation/development.jsonl \
  --split development --output artifacts/evaluation/development-summary.json
cargo run --release --locked --example evaluate -- \
  --split holdout --output artifacts/evaluation/holdout.jsonl
python3 -B scripts/evaluate.py --results artifacts/evaluation/holdout.jsonl \
  --split holdout --output artifacts/evaluation/holdout-summary.json
```

On the immutable 40-query holdout, compact multilingual produced deep Hit@5 of
0.85 for English and 0.85 for Korean overall. The Korean-to-English **code**
subset was 0.70 (7 of 10 queries). The recorded fast English evidence recall
was 0.90, while recall among gold spans representable by a model chunk was
1.00; fast Hit@5 was 0.85. The matched
lexical baseline was 0.80. The fixed holdout corpus generated 40 chunks, all
below K=128, so this run measured ranking quality but not large-corpus candidate
recall behavior. Two public code labels span entire 34- and 54-line excerpts;
under the declared requirement that one result chunk cover at least half of a
gold span, those labels make recall conservative because the model input cap
prevents one chunk covering that much of the whole excerpt. Labels were not
changed after evaluation.

A separate scale sensitivity run adds 240 deterministic neutral text files to
the fixed corpus (280 chunks total). With K=128, English fast Hit@5 remains
0.85, but Korean fast Hit@5 falls to 0.50 while deep remains 0.85. Fast selects
only 0.56 of representable Korean gold evidence on average. This exposes a
candidate-retrieval limitation for low-overlap queries; the larger corpus reuses
inspected holdout labels and is not a new blind holdout. Recreate it with
`scripts/generate_scale_corpus.py` and see
`artifacts/evaluation/compact-multilingual-scale-v2-holdout-summary.json`.

ARM timing and RSS use fresh CLI processes, so every measured run includes model
verification/session loading. The original 10 MiB / 1,000-file result used one
repeated query. The current benchmark uses distinct development queries and
records their identities with at least 20 steady-state runs:

```bash
SUPERGREP_ORT_LIB=/absolute/path/libonnxruntime.so.1.20.0 \
  bash scripts/benchmark.sh target/release/supergrep
```

On the measured four-core ARM host, the 20 steady-state fast runs had 25.69 s
median and 26.42 s p95 process wall time, 1.64 s median model load, 10.24 s
median batch inference, and 733,456 KiB p95 sampled peak RSS. Memory was under
the 1.5 GiB goal; the 15 s end-to-end and 10 s 128-passage inference goals were
**not met**. The search is functional, but the default bounded fast path should
not be described as meeting those latency targets for a 10 MiB / 1,000-file
corpus.

The new 21-distinct-query run on the same synthetic corpus measured 26.62 s
steady-state median wall time (31.63 s p95), 10.50 s median inference, and
733,648 KiB p95 sampled peak RSS. It also misses both latency targets. Its
query IDs, binary hash, corpus digest, and per-run timings are in
`artifacts/benchmarks/cli-fast-varied-10MiB-1000files.json`; the generated
corpus is repetitive and does not represent all real repositories.

Deep throughput at 128, 512, and 2,048 constructed chunks was measured
separately. The reproducible stress command deliberately applies
`--max-chunks`, so those reports mark scan coverage partial while asserting
that every retained chunk was scored:

```bash
SUPERGREP_MODEL_CACHE=/path/to/prepared/cache \
  SUPERGREP_ORT_LIB=/absolute/path/libonnxruntime.so.1.20.0 \
  bash scripts/benchmark-deep.sh target/release/supergrep 2 artifacts/benchmarks/deep-rerun
```

Each row below is the two observed fresh processes (not a stable percentile).
All measured exactly the cap and reported `scoring_complete=true`; `partial`
and `scan_complete=false` are expected because the synthetic 10 MiB corpus has
more chunks than the explicit cap.

| Deep chunks scored | Batch inference, runs (s) | Process wall time, runs (s) | Sampled peak RSS (KiB) |
| ---: | ---: | ---: | ---: |
| 128 | 9.73 / 10.03 | 12.10 / 12.42 | 727,248 / 726,928 |
| 512 | 40.39 / 48.17 | 43.09 / 51.02 | 727,744 / 727,456 |
| 2,048 | 321.88 / 171.70 | 326.24 / 175.92 | 727,264 / 728,064 |

The large 2,048-chunk variation shows these two runs are stress observations,
not a latency guarantee; they used batch size 1 and the runtime remains CPU
bound. Raw JSON is in `artifacts/benchmarks/cli-deep-{128,512,2048}-chunks.json`.

Details and raw records are in [`docs/progress.md`](docs/progress.md),
[`docs/model-decision.md`](docs/model-decision.md), and
`artifacts/benchmarks/`.

The product has no translation service or persistent index. PDF, DOCX, OCR,
embeddings, GPU execution, GUI, and MCP are out of scope for v0.1.

## License

The supergrep source is Apache-2.0; see [`LICENSE`](LICENSE). Model and runtime
licenses/notices are recorded separately in
[`docs/model-decision.md`](docs/model-decision.md) and included for ONNX Runtime
in the local bundle.
