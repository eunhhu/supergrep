# S1 foundation code review

- codeQualityStatus: CLEAR
- recommendation: APPROVE
- blockers: none in the reviewed S1 scope
- Reviewed on: 2026-09-22
- Goal: review discovery, immutable source snapshots, and position-preserving lexical chunking against `docs/implementation-plan.md` sections 5.1–5.2.
- Scope: `src/discovery.rs`, `src/source.rs`, `src/chunk.rs`, `tests/discovery.rs`, `tests/chunking.rs` only.

## Review inputs and artifact provenance

All five files are untracked new files, so the reviewed change is their complete contents rather than an existing tracked-file diff. The specification and all five files were inspected directly, including changes made by the executor in response to review findings. No production or test file was edited by this reviewer.

`omo ulw-loop status --json` returned `command not found`; no `.omo` loop state or supplied attempt directory was available. This report therefore uses the required fallback path. No executor evidence paths or notepad path were supplied. Independent verification is recorded below; this approval does not rely on an executor success claim.

Final reviewed SHA-256 values:

```text
935de58e58cd18a2d1769b40cd6ef82865209cabc05e1a248aa19baaab1d3ab7  src/discovery.rs
014bbea4ffc92fc0cc0a53dc3615ba2443a50639c8f0f67692d41e6e9ac2efed  src/source.rs
3a3cfb664ee46a31bfb6c20118c5908ce7aa741d8886e5740017bec99b554489  src/chunk.rs
13b3fc93391e25e9694ed7fcef7b3abf54c70eb8e96e748626b197c8b5657baf  tests/discovery.rs
cee5bb184cc1fb36ab08c3a691a65454f87c7e0e1b553127552d1b409ebab8c7  tests/chunking.rs
```

## Findings remaining by severity

### CRITICAL

None.

### HIGH

None.

### MEDIUM

None.

### LOW

None requiring a change in this scoped review.

## Findings resolved during review

1. Exact exhaustion of the total read budget previously stopped before later candidate paths without setting partial status. The final implementation checks whether a candidate remains and emits `TotalByteLimit` when appropriate, including when the final bytes belonged to a binary/invalid-UTF-8 policy exclusion.
2. NUL and invalid UTF-8 files were previously counted as operational failures. They are now policy exclusions; their consumed bytes still count toward the read budget.
3. The corpus chunk cap previously applied only after constructing every chunk of the next source. An independent bounded probe observed 1,001 token-counter calls for a 1,000-line source with a one-chunk cap. The final bounded planner stops before processing the whole source, and a counter-based regression test exercises this requirement.
4. Line-boundary planning previously restarted at line 1 for each chunk. It now starts at the current line. The repeated suffix emptiness check now uses `trim_start`, avoiding repeated scans of a large trailing whitespace suffix.
5. Git metadata policy originally missed `.` when the current directory itself was inside `.git`. Checking absolute ancestry fixes this. A follow-up probe found that `.git/../visible.txt` was incorrectly excluded; final root and walker checks normalize parent components after symlink validation.
6. A newly added test initially claimed to verify that old lines were not revisited while asserting only coordinates. Its name now describes its actual coordinate coverage. The independent bounded-work regression supplies the meaningful cap verification.

## Independent verification

Final command:

```text
cargo test --offline --lib --test discovery --test chunking
```

Observed exit status: 0.

```text
library unit tests: 7 passed, 0 failed, 0 ignored
tests/chunking.rs: 8 passed, 0 failed, 0 ignored
tests/discovery.rs: 12 passed, 0 failed, 0 ignored
total: 27 passed
```

The tests exercise exact CRLF/Unicode byte and line coordinates, long-line splitting, whitespace-only exclusion, overlap bounds, chunk-cap reporting and bounded planning, ignore/hidden independence, explicit file roots, invalid text policy, deterministic ordering and resource limits, missing/symlink roots, and Git metadata path handling. Code inspection also checked read-length accounting, descriptor/path metadata checks around reads, source snapshot ownership, and shared-cap propagation.

## Required skill perspectives

The `remove-ai-slops` and `programming` skills were not listed as available and their `SKILL.md` files were not found under the configured Codex/agent skill and plugin roots. The review therefore applied the documented criteria supplied in the review task. This fallback perspective check ran before judging test relevance and maintainability.

No remaining violations of either perspective were found in the final scoped files. Tests exercise observable behavior rather than requested code deletion, prompt wording, or mirrored implementation constants. The bounded-counter test covers real resource behavior. UTF-8 validation, path normalization for the explicit Git exclusion policy, and configuration checks serve stated input-boundary contracts. No untyped escape hatches, unrelated extraction/parsing, or unnecessary production abstraction were found.

## Scope limits

This is approval of the five S1 files, not approval of the full application or all of section 5.2's later model integration. Model tokenizer offsets, real query/passage pair fitting, candidate selection, CLI exit-code wiring, runtime inference, and end-to-end output behavior are outside this review. Filesystem mutation handling was inspected but not stress-tested under concurrent changes; the implementation appropriately describes its checks as best effort.
