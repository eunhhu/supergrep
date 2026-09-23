# Parallel fitting validation — 2026-09-23, ARM64

All paths are relative to `/home/eunhhu/work/supergrep` unless absolute.
The prepared pinned model cache was `/tmp/supergrep-model-cache.bLSH09` and
the local runtime was `.supergrep-dev-runtime/onnxruntime-linux-aarch64-1.20.0/lib/libonnxruntime.so.1.20.0`.

| Command/check | Exit/result |
| --- | --- |
| `cargo fmt --all --check` | 0 |
| `cargo clippy --locked --all-targets -j 1 -- -D warnings` | 0 |
| `cargo test --locked -j 1` | 0; all default tests passed, offline and pinned-tokenizer tests opt-in |
| `SUPERGREP_TEST_MODEL_CACHE=/tmp/supergrep-model-cache.bLSH09 cargo test --locked -j 1 --test prepared_tokenizer -- --ignored --nocapture` | 0; pinned tokenizer parity test passed |
| `python3 -B -m unittest discover -s scripts -p 'test_*.py' -v` | 0; large-stdout/batch-argument benchmark test passed |
| `cargo build --release --locked -j 1 --bin supergrep --example evaluate` | 0 |
| Fixed holdout evaluator with local runtime/model and `--split holdout` | 0; `artifacts/evaluation/compact-multilingual-holdout-parallel.jsonl` |

The locally generated current holdout JSONL and the pre-optimization
`artifacts/evaluation/compact-multilingual-holdout.jsonl` have identical
SHA-256:
`5f773503dc107a9eb512d927efedecbee02624396e6c154ea82008f776514b59`.
This checks the complete evaluator output, including fitted spans, candidate
sets, model scores, and rankings. It does not claim the original 40-chunk
holdout tests candidate reduction.

The 21-query before/after benchmarks are
`artifacts/benchmarks/cli-fast-varied-10MiB-1000files.json` and
`artifacts/benchmarks/cli-fast-parallel-fitting-varied-10MiB-1000files.json`.
Median chunking fell 11.405→5.019 s and median wall time 26.615→21.580 s.
The latter run had wall p95 36.346 s, so neither 15 s wall nor 10 s
inference targets were met. The run is on a repetitive synthetic corpus and
has no controlled background CPU-load record.

The packaged binary and benchmark binary both have SHA-256
`99a87d47e3911c76ffea5f16a643685f18dcde72fb4d09fd8b94e8c8b2323535`.
The package archive is
`artifacts/packages-parallel/supergrep-v0.1.0-linux-aarch64.tar.gz`, SHA-256
`c97d34ac90e3e1251cc7089938408444b8ac772f9633b0017f37c34d9bdc0941`.
The separate-directory bundle smoke passed with the pinned real model,
exact byte/line span, adjacent runtime load, and no network syscalls
observed by `strace -f -e trace=network`.
Local captured smoke output: `artifacts/validation/parallel-bundle-smoke.log`
(exit 0). Generated JSONL, log, and package files are retained locally but
excluded from version control; their hashes and test results are recorded here.
