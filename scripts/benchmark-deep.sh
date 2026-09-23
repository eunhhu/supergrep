#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: bash scripts/benchmark-deep.sh <supergrep-binary> [runs] [output-directory]" >&2
  exit 2
fi

sg_binary=$1
sg_runs=${2:-2}
sg_output=${3:-artifacts/benchmarks}
sg_temporary=$(mktemp -d "${TMPDIR:-/tmp}/supergrep-deep-benchmark.XXXXXX")
trap 'rm -rf -- "$sg_temporary"' EXIT

python3 -B scripts/generate_benchmark_corpus.py "$sg_temporary/corpus" \
  --files 1000 --total-bytes 10485760

for sg_chunks in 128 512 2048; do
  sg_arguments=(
    "$sg_binary" "$sg_temporary/corpus"
    --runs "$sg_runs"
    --deep
    --max-chunks "$sg_chunks"
    --output "$sg_output/cli-deep-${sg_chunks}-chunks.json"
  )
  if [[ -n "${SUPERGREP_MODEL_CACHE:-}" ]]; then
    sg_arguments+=(--cache-dir "$SUPERGREP_MODEL_CACHE")
  fi
  python3 -B scripts/benchmark.py "${sg_arguments[@]}"
done
