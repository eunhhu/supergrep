#!/usr/bin/env python3
"""Measure fresh CLI process time, ONNX phases, and sampled peak RSS."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def high_water_rss_kib(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return None
    return None


def corpus_digest(corpus: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(path for path in corpus.rglob("*") if path.is_file()):
        relative = path.relative_to(corpus).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def run_once(
    binary: Path,
    corpus: Path,
    cache: Path | None,
    index: int,
    deep: bool,
    max_chunks: int | None,
    query: str = "retry delay after repeated network refusals",
    query_id: str | None = None,
    timeout_seconds: float = 600,
    batch_size: int = 1,
) -> dict[str, object]:
    command = [
        str(binary),
        query,
        str(corpus),
        "--json",
        "--top-k",
        "10",
    ]
    if cache is not None:
        command.extend(["--cache-dir", str(cache)])
    if deep:
        command.append("--deep")
    if max_chunks is not None:
        command.extend(["--max-chunks", str(max_chunks)])
    if batch_size != 1:
        command.extend(["--batch-size", str(batch_size)])

    started = time.monotonic_ns()
    peak_rss = 0
    # Drain neither pipe in the sampling loop: send both streams to temporary
    # files so a large JSON result or diagnostic cannot block the child.
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stdout_file, tempfile.TemporaryFile(
        mode="w+t", encoding="utf-8"
    ) as stderr_file:
        process = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file)
        deadline = time.monotonic() + timeout_seconds
        try:
            while process.poll() is None:
                current = high_water_rss_kib(process.pid)
                if current is not None:
                    peak_rss = max(peak_rss, current)
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"run {index + 1} exceeded {timeout_seconds:g}s")
                time.sleep(0.01)
            process.wait()
            finished = time.monotonic_ns()
        except BaseException:
            process.kill()
            process.wait()
            raise
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout, stderr = stdout_file.read(), stderr_file.read()
    wall_ms = (finished - started) / 1_000_000
    expected_capped_deep_exit = deep and max_chunks is not None and process.returncode == 3
    if process.returncode != 0 and not expected_capped_deep_exit:
        raise RuntimeError(
            f"run {index + 1} failed with exit={process.returncode}: {stderr.strip()}"
        )
    document = json.loads(stdout)
    if document.get("model", {}).get("id") != "compact-multilingual":
        raise RuntimeError(f"run {index + 1} did not use the pinned default model")
    if document.get("query") != query:
        raise RuntimeError(f"run {index + 1} returned a different query than the requested one")
    if deep and not document["stats"]["scoring_complete"]:
        raise RuntimeError(f"run {index + 1} did not score every constructed deep chunk")
    if expected_capped_deep_exit and (
        not document["stats"]["partial"]
        or document["stats"]["chunks_total"] != max_chunks
        or document["stats"]["chunks_evaluated"] != max_chunks
        or not any(
            item["kind"] == "chunk_limit_reached"
            for item in document["diagnostics"]
        )
    ):
        raise RuntimeError(
            f"run {index + 1} exited 3 without the expected explicit {max_chunks}-chunk cap"
        )
    timing = document["stats"]["timing"]
    return {
        "run": index + 1,
        "query_id": query_id,
        "query": query,
        "phase": "initial" if index == 0 else "steady_state",
        "mode": document["mode"],
        "model_revision": document["model"].get("revision"),
        "wall_ms": wall_ms,
        "sampled_peak_rss_kib": peak_rss,
        "model_load_ms": timing.get("model_load_ms"),
        "tokenization_ms": timing.get("tokenization_ms"),
        "inference_ms": timing.get("inference_ms"),
        "discovery_ms": timing.get("discovery_ms"),
        "chunking_ms": timing.get("chunking_ms"),
        "search_ms": timing.get("search_ms"),
        "total_ms": timing.get("total_ms"),
        "chunks_total": document["stats"]["chunks_total"],
        "chunks_evaluated": document["stats"]["chunks_evaluated"],
        "scan_complete": document["stats"]["scan_complete"],
        "partial": document["stats"]["partial"],
        "scoring_complete": document["stats"]["scoring_complete"],
    }


def summarize(runs: list[dict[str, object]]) -> dict[str, object]:
    steady = runs[1:]
    fields = (
        "wall_ms",
        "sampled_peak_rss_kib",
        "model_load_ms",
        "tokenization_ms",
        "inference_ms",
        "discovery_ms",
        "chunking_ms",
        "search_ms",
        "total_ms",
    )
    aggregate: dict[str, object] = {}
    for field in fields:
        values = [float(run[field]) for run in steady if run[field] is not None]
        aggregate[field] = {
            "median": statistics.median(values),
            "p95": percentile(values, 0.95),
            "min": min(values),
            "max": max(values),
            "sample_count": len(values),
        }
    return {
        "first_run": runs[0],
        "steady_state": aggregate,
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--runs", type=int, default=21)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--queries-file", type=Path, help="JSONL query definitions for distinct-query runs")
    parser.add_argument("--query-split", default="development")
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least two to separate the first run from steady state")
    if args.max_chunks is not None and args.max_chunks < 1:
        parser.error("--max-chunks must be at least one")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be a positive finite number")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least one")

    binary = args.binary.resolve(strict=True)
    corpus = args.corpus.resolve(strict=True)
    cache = args.cache_dir.resolve(strict=False) if args.cache_dir else None
    query_records = []
    if args.queries_file:
        for line in args.queries_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split") == args.query_split and record.get("query_type") == "relevant":
                query_records.append((record["id"], record["query"]))
        if len(query_records) < args.runs:
            parser.error(f"--queries-file has only {len(query_records)} relevant queries in {args.query_split}; need {args.runs}")
        query_records = query_records[: args.runs]
        if len({query for _, query in query_records}) != len(query_records):
            parser.error("--queries-file contains duplicate query text in the selected runs")
    else:
        query_records = [(None, "retry delay after repeated network refusals")] * args.runs
    results = []
    for index, (query_id, query) in enumerate(query_records):
        print(f"CLI benchmark run {index + 1}/{args.runs}", file=sys.stderr, flush=True)
        results.append(
            run_once(
                binary, corpus, cache, index, args.deep, args.max_chunks,
                query=query, query_id=query_id, timeout_seconds=args.timeout_seconds,
                batch_size=args.batch_size,
            )
        )

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "record_type": "cli_benchmark",
        "model_profile": "compact-multilingual",
        "mode": "deep" if args.deep else "fast",
        "batch_size": args.batch_size,
        "max_chunks": args.max_chunks,
        "query_set": {
            "kind": "jsonl" if args.queries_file else "single_query",
            "path": str(args.queries_file.resolve()) if args.queries_file else None,
            "sha256": hashlib.sha256(args.queries_file.read_bytes()).hexdigest() if args.queries_file else None,
            "split": args.query_split if args.queries_file else None,
            "distinct_queries": len({query for _, query in query_records}),
        },
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "host": {"machine": platform.machine(), "platform": platform.platform()},
        "runs_requested": args.runs,
        "fresh_process_per_run": True,
        "first_run_is_strict_cold_disk": False,
        "rss_method": "sampled Linux /proc/<pid>/status VmHWM at 10ms intervals",
        "corpus": {
            "path": str(corpus),
            "sha256": corpus_digest(corpus),
            "files": len(list(corpus.glob("*.rs"))),
            "bytes": sum(path.stat().st_size for path in corpus.glob("*.rs")),
        },
        **summarize(results),
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
    print(f"wrote benchmark results to {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
