#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 2 ]]; then
  echo "usage: bash scripts/test-bundle.sh <bundle.tar.gz> <prepared-model-cache>" >&2
  exit 2
fi

sg_archive=$(realpath -- "$1")
sg_cache=$(realpath -- "$2")
if ! command -v strace >/dev/null 2>&1; then
  echo "strace is required to prove the bundled search path makes no network syscalls" >&2
  exit 1
fi
sg_temporary=$(mktemp -d "${TMPDIR:-/tmp}/supergrep-bundle-smoke.XXXXXX")
trap 'rm -rf -- "$sg_temporary"' EXIT

tar -xzf "$sg_archive" -C "$sg_temporary"
sg_bundle="$sg_temporary/supergrep-v0.1.0-linux-aarch64"
sg_binary="$sg_bundle/bin/supergrep"
mkdir -p "$sg_temporary/work/corpus"
printf 'pub fn bounded_retry(attempt: u32) -> u64 { attempt.min(8) as u64 }\n' \
  > "$sg_temporary/work/corpus/retry.rs"

(
  cd "$sg_temporary/work"
  env -u SUPERGREP_ORT_LIB "$sg_binary" --help > help.txt
  env -u SUPERGREP_ORT_LIB "$sg_binary" doctor --cache-dir "$sg_cache" --json > doctor.json
  strace -f -e trace=network -o network.trace env -u SUPERGREP_ORT_LIB "$sg_binary" \
    "where is retry attempts bounded?" corpus --deep --json \
    --cache-dir "$sg_cache" > search.json
  python3 -B -c 'import json; from pathlib import Path; d=json.load(open("doctor.json")); assert d["runtime_ready"] and d["model_ready"]; s=json.load(open("search.json")); assert s["model"]["id"] == "compact-multilingual" and s["stats"]["scoring_complete"] is True and s["stats"]["chunks_total"] == 1 and s["stats"]["chunks_evaluated"] == 1; r=s["results"][0]; assert r["path"] == "retry.rs" and r["start_byte"] == 0 and r["end_byte"] == len(Path("corpus/retry.rs").read_bytes()) and r["start_line"] == 1 and r["end_line"] == 1 and r["score_kind"] == "raw_logit"; t=Path("network.trace").read_text(); assert not any("(" in line for line in t.splitlines()), t; print("fresh-directory bundled runtime/model smoke: passed; exact byte/line span; no network syscalls")'
)
