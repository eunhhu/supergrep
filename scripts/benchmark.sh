#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: bash scripts/benchmark.sh <supergrep-binary> [runs] [output.json]" >&2
  exit 2
fi

sg_binary=$1
sg_runs=${2:-21}
sg_output=${3:-artifacts/benchmarks/cli-fast-varied-10MiB-1000files.json}
sg_temporary=$(mktemp -d "${TMPDIR:-/tmp}/supergrep-benchmark.XXXXXX")
trap 'rm -rf -- "$sg_temporary"' EXIT

python3 -B scripts/generate_benchmark_corpus.py "$sg_temporary/corpus" \
  --files 1000 --total-bytes 10485760

sg_benchmark_args=("$sg_binary" "$sg_temporary/corpus" --runs "$sg_runs" --output "$sg_output" \
  --queries-file eval/queries.jsonl --query-split development)
if [[ -n "${SUPERGREP_MODEL_CACHE:-}" ]]; then
  sg_benchmark_args+=(--cache-dir "$SUPERGREP_MODEL_CACHE")
fi
python3 -B scripts/benchmark.py "${sg_benchmark_args[@]}"
