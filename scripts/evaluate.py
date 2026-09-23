#!/usr/bin/env python3
"""Aggregate supergrep evaluation JSONL without reimplementing search.

The Rust evaluator owns chunking, lexical ranking, candidate selection, and
model scoring.  This program validates its span-based output against the fixed
labels and aggregates the measures defined in docs/implementation-plan.md §7.
It deliberately contains no query-specific behavior or retrieval logic.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


SCHEMA_VERSION = 1
MODES = ("lexical", "fast", "deep")
RELEVANT_SPLITS = ("development", "holdout")


class ValidationError(Exception):
    """Raised for a corpus or evaluator-contract violation."""


def fail(message: str) -> None:
    raise ValidationError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"missing JSON file: {path}")
    except UnicodeDecodeError as exc:
        fail(f"{path} is not UTF-8: {exc}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{path} must contain one JSON object")
    return value


def read_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        fail(f"missing {description}: {path}")
    except UnicodeDecodeError as exc:
        fail(f"{description} is not UTF-8 ({path}): {exc}")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"invalid JSON in {path}:{line_number}: {exc}")
        if not isinstance(value, dict):
            fail(f"{path}:{line_number} must contain a JSON object")
        value["_line_number"] = line_number
        rows.append(value)
    if not rows:
        fail(f"{description} has no records: {path}")
    return rows


def require_string(value: Any, field: str, context: str) -> str:
    if not isinstance(value, str) or not value:
        fail(f"{context}: {field} must be a non-empty string")
    return value


def require_integer(value: Any, field: str, context: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{context}: {field} must be an integer")
    if minimum is not None and value < minimum:
        fail(f"{context}: {field} must be at least {minimum}")
    return value


def require_relative_path(value: Any, field: str, context: str) -> str:
    path = require_string(value, field, context)
    if "\\" in path:
        fail(f"{context}: {field} must use forward-slash corpus-relative paths")
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts or path == ".":
        fail(f"{context}: {field} must be a safe corpus-relative path")
    return parsed.as_posix()


def span_key(span: dict[str, Any]) -> tuple[str, int, int]:
    return (span["path"], span["start_byte"], span["end_byte"])


def validate_span(
    span: Any,
    context: str,
    *,
    label: bool = False,
    require_rationale: bool = False,
) -> dict[str, Any]:
    if not isinstance(span, dict):
        fail(f"{context}: span must be an object")
    normalized = {
        "path": require_relative_path(span.get("path"), "path", context),
        "start_byte": require_integer(span.get("start_byte"), "start_byte", context, 0),
        "end_byte": require_integer(span.get("end_byte"), "end_byte", context, 1),
    }
    if normalized["end_byte"] <= normalized["start_byte"]:
        fail(f"{context}: end_byte must be greater than start_byte")
    if "start_line" in span:
        normalized["start_line"] = require_integer(
            span["start_line"], "start_line", context, 1
        )
    if "end_line" in span:
        normalized["end_line"] = require_integer(span["end_line"], "end_line", context, 1)
    if "start_line" in normalized and "end_line" in normalized:
        if normalized["end_line"] < normalized["start_line"]:
            fail(f"{context}: end_line must not precede start_line")
    if label:
        normalized["evidence_id"] = require_string(
            span.get("evidence_id"), "evidence_id", context
        )
        relevance = require_integer(span.get("relevance"), "relevance", context, 0)
        if relevance > 2:
            fail(f"{context}: relevance must be 0, 1, or 2")
        normalized["relevance"] = relevance
    if require_rationale:
        normalized["rationale"] = require_string(span.get("rationale"), "rationale", context)
    for optional in ("rank", "score", "score_kind", "chunk_id"):
        if optional in span:
            normalized[optional] = span[optional]
    return normalized


def line_starts(data: bytes) -> list[int]:
    return [0] + [index + 1 for index, byte in enumerate(data) if byte == 10]


def byte_line(starts: list[int], byte_offset: int) -> int:
    # A newline belongs to the line preceding the next line start.
    return bisect.bisect_right(starts, byte_offset)


def validate_label_against_file(
    label: dict[str, Any], corpus_root: Path, cache: dict[str, bytes]
) -> None:
    path = label["path"]
    if path not in cache:
        try:
            data = (corpus_root / path).read_bytes()
        except FileNotFoundError:
            fail(f"label {label['evidence_id']}: fixture path does not exist: {path}")
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as exc:
            fail(f"label {label['evidence_id']}: fixture is not UTF-8 ({path}): {exc}")
        cache[path] = data
    data = cache[path]
    start = label["start_byte"]
    end = label["end_byte"]
    if end > len(data):
        fail(
            f"label {label['evidence_id']}: byte range {start}:{end} exceeds "
            f"{path} ({len(data)} bytes)"
        )
    if not data[start:end].strip():
        fail(f"label {label['evidence_id']}: evidence span is empty or whitespace")
    starts = line_starts(data)
    actual_start_line = byte_line(starts, start)
    actual_end_line = byte_line(starts, end - 1)
    if label.get("start_line") != actual_start_line:
        fail(
            f"label {label['evidence_id']}: start_line={label.get('start_line')} "
            f"does not match byte {start} ({actual_start_line})"
        )
    if label.get("end_line") != actual_end_line:
        fail(
            f"label {label['evidence_id']}: end_line={label.get('end_line')} "
            f"does not match byte {end} ({actual_end_line})"
        )


def label_signature(labels: Iterable[dict[str, Any]]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        sorted(
            (
                label["evidence_id"],
                label["path"],
                label["start_byte"],
                label["end_byte"],
                label["start_line"],
                label["end_line"],
                label["relevance"],
            )
            for label in labels
        )
    )


def validate_manifest(
    manifest: dict[str, Any], corpus_root: Path, queries: list[dict[str, Any]]
) -> dict[str, Any]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        fail("manifest schema_version must be 1")
    inventory = manifest.get("fixture_inventory")
    if not isinstance(inventory, list) or not inventory:
        fail("manifest fixture_inventory must be a non-empty list")

    expected_paths: set[str] = set()
    for item in inventory:
        if not isinstance(item, dict):
            fail("manifest fixture_inventory entries must be objects")
        path = require_relative_path(item.get("path"), "fixture_inventory.path", "manifest")
        digest = require_string(item.get("sha256"), "fixture_inventory.sha256", "manifest")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            fail(f"manifest fixture inventory hash is not lowercase SHA-256: {path}")
        if path in expected_paths:
            fail(f"manifest lists fixture twice: {path}")
        expected_paths.add(path)
        target = corpus_root / path
        try:
            contents = target.read_bytes()
        except FileNotFoundError:
            fail(f"manifest fixture is missing: {path}")
        try:
            contents.decode("utf-8")
        except UnicodeDecodeError as exc:
            fail(f"manifest fixture is not UTF-8 ({path}): {exc}")
        actual = hashlib.sha256(contents).hexdigest()
        if actual != digest:
            fail(f"fixture SHA-256 mismatch for {path}: expected {digest}, got {actual}")

    actual_paths = {
        item.relative_to(corpus_root).as_posix()
        for item in corpus_root.rglob("*")
        if item.is_file()
    }
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        unlisted = sorted(actual_paths - expected_paths)
        fail(f"fixture inventory mismatch; missing={missing}, unlisted={unlisted}")

    counts = manifest.get("intent_counts")
    if not isinstance(counts, dict):
        fail("manifest intent_counts must be an object")
    relevant = [record for record in queries if record["query_type"] == "relevant"]
    irrelevant = [record for record in queries if record["query_type"] == "irrelevant"]
    actual_counts = {
        "relevant_intents": len({record["intent_id"] for record in relevant}),
        "relevant_query_records": len(relevant),
        "irrelevant_intents": len({record["intent_id"] for record in irrelevant}),
        "irrelevant_query_records": len(irrelevant),
    }
    for field, actual in actual_counts.items():
        if counts.get(field) != actual:
            fail(f"manifest intent_counts.{field}={counts.get(field)!r}, expected {actual}")
    return manifest


def validate_queries(
    raw_queries: list[dict[str, Any]], corpus_root: Path
) -> list[dict[str, Any]]:
    query_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    source_cache: dict[str, bytes] = {}

    for record in raw_queries:
        context = f"queries.jsonl:{record['_line_number']}"
        if record.get("schema_version") != SCHEMA_VERSION:
            fail(f"{context}: schema_version must be 1")
        if record.get("record_type") != "query":
            fail(f"{context}: record_type must be query")
        query_id = require_string(record.get("id"), "id", context)
        if query_id in query_ids:
            fail(f"{context}: duplicate query id {query_id}")
        query_ids.add(query_id)
        intent_id = require_string(record.get("intent_id"), "intent_id", context)
        split = require_string(record.get("split"), "split", context)
        if split not in (*RELEVANT_SPLITS, "irrelevant"):
            fail(f"{context}: unsupported split {split!r}")
        category = require_string(record.get("category"), "category", context)
        language = require_string(record.get("language"), "language", context)
        if language not in ("en", "ko"):
            fail(f"{context}: language must be en or ko")
        query_type = require_string(record.get("query_type"), "query_type", context)
        if query_type not in ("relevant", "irrelevant"):
            fail(f"{context}: query_type must be relevant or irrelevant")
        if (query_type == "irrelevant") != (split == "irrelevant"):
            fail(f"{context}: irrelevant query_type must use the irrelevant split")
        require_string(record.get("query"), "query", context)
        tags = record.get("tags")
        if not isinstance(tags, list) or not all(isinstance(tag, str) and tag for tag in tags):
            fail(f"{context}: tags must be a list of non-empty strings")
        labels_raw = record.get("labels")
        if not isinstance(labels_raw, list):
            fail(f"{context}: labels must be a list")
        if query_type == "relevant" and not labels_raw:
            fail(f"{context}: relevant query must have labels")
        if query_type == "irrelevant" and labels_raw:
            fail(f"{context}: irrelevant query must have no labels")

        labels: list[dict[str, Any]] = []
        seen_evidence_ids: set[str] = set()
        for position, label_raw in enumerate(labels_raw):
            label = validate_span(
                label_raw,
                f"{context}.labels[{position}]",
                label=True,
                require_rationale=True,
            )
            if "start_line" not in label or "end_line" not in label:
                fail(f"{context}.labels[{position}]: labels require start_line and end_line")
            if label["evidence_id"] in seen_evidence_ids:
                fail(f"{context}: duplicate evidence_id {label['evidence_id']}")
            seen_evidence_ids.add(label["evidence_id"])
            validate_label_against_file(label, corpus_root, source_cache)
            labels.append(label)
        normalized.append(
            {
                "id": query_id,
                "intent_id": intent_id,
                "split": split,
                "category": category,
                "language": language,
                "query_type": query_type,
                "query": record["query"],
                "tags": tuple(tags),
                "labels": labels,
            }
        )

    by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in normalized:
        by_intent[record["intent_id"]].append(record)
    for intent_id, variants in by_intent.items():
        if len(variants) != 2:
            fail(f"intent {intent_id} must have exactly English and Korean variants")
        languages = {variant["language"] for variant in variants}
        if languages != {"en", "ko"}:
            fail(f"intent {intent_id} must have one en and one ko query")
        if len({variant["split"] for variant in variants}) != 1:
            fail(f"intent {intent_id} crosses splits")
        if len({variant["category"] for variant in variants}) != 1:
            fail(f"intent {intent_id} crosses categories")
        if len({variant["query_type"] for variant in variants}) != 1:
            fail(f"intent {intent_id} mixes relevant and irrelevant records")
        if variants[0]["query_type"] == "relevant":
            if label_signature(variants[0]["labels"]) != label_signature(variants[1]["labels"]):
                fail(f"intent {intent_id} has language variants with different labels")

    return normalized


def validate_low_overlap_intents(manifest: dict[str, Any], queries: list[dict[str, Any]]) -> None:
    listed = manifest.get("low_lexical_overlap_intents")
    if not isinstance(listed, list) or not all(isinstance(item, str) for item in listed):
        fail("manifest low_lexical_overlap_intents must be a string list")
    intents = {record["intent_id"] for record in queries}
    if not set(listed).issubset(intents):
        fail("manifest low_lexical_overlap_intents references an unknown intent")
    if len(set(listed)) < 12:
        fail("at least 12 low-lexical-overlap intents are required")
    for intent_id in listed:
        variants = [record for record in queries if record["intent_id"] == intent_id]
        if not all("low_lexical_overlap" in record["tags"] for record in variants):
            fail(f"low-overlap intent {intent_id} is missing the low_lexical_overlap tag")


def overlap_fraction(result: dict[str, Any], label: dict[str, Any]) -> float:
    if result["path"] != label["path"]:
        return 0.0
    overlap = max(
        0,
        min(result["end_byte"], label["end_byte"])
        - max(result["start_byte"], label["start_byte"]),
    )
    return overlap / (label["end_byte"] - label["start_byte"])


def matching_labels(span: dict[str, Any], labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        label
        for label in labels
        if label["relevance"] > 0 and overlap_fraction(span, label) >= 0.5
    ]


def normalize_ranked_results(
    raw_results: Any, context: str, corpus_root: Path
) -> list[dict[str, Any]]:
    if not isinstance(raw_results, list):
        fail(f"{context}: results must be a list")
    normalized = [
        validate_span(item, f"{context}.results[{index}]")
        for index, item in enumerate(raw_results)
    ]
    ranks: list[int] = []
    for index, span in enumerate(normalized, 1):
        rank = span.get("rank", index)
        rank = require_integer(rank, "rank", f"{context}.results[{index - 1}]", 1)
        span["rank"] = rank
        score = span.get("score")
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                fail(f"{context}.results[{index - 1}]: score must be finite when present")
            span["score"] = float(score)
        ranks.append(rank)
        validate_output_span_exists(span, corpus_root, f"{context}.results[{index - 1}]")
    if len(set(ranks)) != len(ranks):
        fail(f"{context}: results have duplicate ranks")
    return sorted(normalized, key=lambda item: item["rank"])


def normalize_span_list(
    raw_spans: Any, field: str, context: str, corpus_root: Path
) -> list[dict[str, Any]]:
    if not isinstance(raw_spans, list):
        fail(f"{context}: {field} must be a list")
    spans = [
        validate_span(item, f"{context}.{field}[{index}]")
        for index, item in enumerate(raw_spans)
    ]
    keys = [span_key(span) for span in spans]
    if len(set(keys)) != len(keys):
        fail(f"{context}: {field} contains duplicate spans")
    for index, span in enumerate(spans):
        validate_output_span_exists(span, corpus_root, f"{context}.{field}[{index}]")
    return spans


def validate_output_span_exists(span: dict[str, Any], corpus_root: Path, context: str) -> None:
    target = corpus_root / span["path"]
    try:
        byte_count = target.stat().st_size
    except FileNotFoundError:
        fail(f"{context}: path does not exist in evaluation corpus: {span['path']}")
    if span["end_byte"] > byte_count:
        fail(
            f"{context}: span {span['start_byte']}:{span['end_byte']} exceeds "
            f"{span['path']} ({byte_count} bytes)"
        )


def normalize_event(
    raw: dict[str, Any], query_by_id: dict[str, dict[str, Any]], corpus_root: Path
) -> dict[str, Any]:
    context = f"results JSONL:{raw['_line_number']}"
    if raw.get("schema_version") != SCHEMA_VERSION:
        fail(f"{context}: schema_version must be 1")
    if raw.get("record_type") != "evaluation_result":
        fail(f"{context}: record_type must be evaluation_result")
    query_id = require_string(raw.get("query_id"), "query_id", context)
    if query_id not in query_by_id:
        fail(f"{context}: query_id is not defined: {query_id}")
    mode = require_string(raw.get("mode"), "mode", context)
    if mode not in MODES:
        fail(f"{context}: unsupported mode {mode!r}")
    contract = require_string(raw.get("chunk_contract_id"), "chunk_contract_id", context)
    normalized = {
        "query_id": query_id,
        "mode": mode,
        "chunk_contract_id": contract,
        "results": normalize_ranked_results(raw.get("results"), context, corpus_root),
    }
    if "query" in raw and raw["query"] != query_by_id[query_id]["query"]:
        fail(f"{context}: query text does not match its definition")
    for field in ("all_chunk_spans", "candidate_spans", "evaluated_spans"):
        if field in raw:
            normalized[field] = normalize_span_list(raw[field], field, context, corpus_root)
    if "scoring_complete" in raw:
        if raw["scoring_complete"] is not None and not isinstance(raw["scoring_complete"], bool):
            fail(f"{context}: scoring_complete must be boolean or null")
        normalized["scoring_complete"] = raw["scoring_complete"]
    if "candidate_limit" in raw:
        normalized["candidate_limit"] = require_integer(
            raw["candidate_limit"], "candidate_limit", context, 1
        )
    return normalized


def require_contract_fields(event: dict[str, Any], context: str) -> None:
    mode = event["mode"]
    if "all_chunk_spans" not in event:
        fail(f"{context}: {mode} requires all_chunk_spans for chunk-set comparability")
    all_keys = {span_key(span) for span in event["all_chunk_spans"]}
    result_keys = {span_key(span) for span in event["results"]}
    if mode == "lexical":
        return
    if "evaluated_spans" not in event:
        fail(f"{context}: {mode} requires evaluated_spans")
    evaluated_keys = {span_key(span) for span in event["evaluated_spans"]}
    if not evaluated_keys.issubset(all_keys):
        fail(f"{context}: evaluated_spans are not a subset of all_chunk_spans")
    if not result_keys.issubset(evaluated_keys):
        fail(f"{context}: ranked results include a span that was not evaluated")
    if mode == "fast":
        if "candidate_spans" not in event:
            fail(f"{context}: fast requires candidate_spans")
        candidate_keys = {span_key(span) for span in event["candidate_spans"]}
        if candidate_keys != evaluated_keys:
            fail(f"{context}: fast candidate_spans must equal evaluated_spans")
        if not candidate_keys.issubset(all_keys):
            fail(f"{context}: candidate_spans are not a subset of all_chunk_spans")
        limit = event.get("candidate_limit")
        if limit is not None and len(candidate_keys) > limit:
            fail(f"{context}: fast selected more candidates than candidate_limit")
    if mode == "deep":
        if event.get("scoring_complete") is not True:
            fail(f"{context}: deep must set scoring_complete=true")
        if evaluated_keys != all_keys:
            fail(f"{context}: deep must evaluate every generated model chunk")
        if "candidate_spans" in event:
            candidate_keys = {span_key(span) for span in event["candidate_spans"]}
            if candidate_keys != all_keys:
                fail(f"{context}: deep candidate_spans, when present, must cover all chunks")


def validate_event_set(
    events: list[dict[str, Any]],
    selected_queries: list[dict[str, Any]],
    allow_partial: bool,
) -> dict[tuple[str, str], dict[str, Any]]:
    selected_ids = {record["id"] for record in selected_queries}
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        if event["query_id"] not in selected_ids:
            continue
        key = (event["query_id"], event["mode"])
        if key in by_key:
            fail(f"duplicate result event for query={key[0]}, mode={key[1]}")
        require_contract_fields(event, f"query={key[0]}, mode={key[1]}")
        by_key[key] = event

    if not allow_partial:
        missing = [
            f"{record['id']}:{mode}"
            for record in selected_queries
            for mode in MODES
            if (record["id"], mode) not in by_key
        ]
        if missing:
            preview = ", ".join(missing[:12])
            suffix = "" if len(missing) <= 12 else f" (+{len(missing) - 12} more)"
            fail(f"missing required evaluation results: {preview}{suffix}")

    for record in selected_queries:
        present = [
            by_key[(record["id"], mode)]
            for mode in MODES
            if (record["id"], mode) in by_key
        ]
        contracts = {event["chunk_contract_id"] for event in present}
        if len(contracts) > 1:
            fail(f"query {record['id']}: modes use different chunk_contract_id values")
        all_sets = {
            tuple(sorted(span_key(span) for span in event["all_chunk_spans"]))
            for event in present
        }
        if len(all_sets) > 1:
            fail(f"query {record['id']}: modes use different model chunk sets")
    return by_key


def ranked_relevances(
    results: list[dict[str, Any]], labels: list[dict[str, Any]]
) -> tuple[list[int], list[set[str]]]:
    seen: set[str] = set()
    relevances: list[int] = []
    matches_by_rank: list[set[str]] = []
    for result in results:
        hits = matching_labels(result, labels)
        unseen = [label for label in hits if label["evidence_id"] not in seen]
        seen.update(label["evidence_id"] for label in unseen)
        matches_by_rank.append({label["evidence_id"] for label in hits})
        relevances.append(max((label["relevance"] for label in unseen), default=0))
    return relevances, matches_by_rank


def dcg(relevances: list[int], cutoff: int) -> float:
    return sum(
        (2**relevance - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances[:cutoff], 1)
    )


def quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = (len(ordered) - 1) * fraction
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": percentile(0.5),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def metric_for_event(query: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    labels = [label for label in query["labels"] if label["relevance"] > 0]
    results = event["results"]
    coverable_ids = {
        label["evidence_id"]
        for span in event["all_chunk_spans"]
        for label in matching_labels(span, labels)
    }
    relevances, matches = ranked_relevances(results, labels)
    first_rank = next((index for index, value in enumerate(relevances[:10], 1) if value > 0), None)
    ideal = sorted((label["relevance"] for label in labels), reverse=True)
    output: dict[str, Any] = {
        "query_id": query["id"],
        "intent_id": query["intent_id"],
        "split": query["split"],
        "category": query["category"],
        "language": query["language"],
        "tags": list(query["tags"]),
        "mode": event["mode"],
        "result_count": len(results),
        "hit_at_5": any(value > 0 for value in relevances[:5]),
        "mrr_at_10": 0.0 if first_rank is None else 1.0 / first_rank,
        "ndcg_at_10": 0.0 if not ideal else dcg(relevances, 10) / dcg(ideal, 10),
        "top10_chunk_keys": [span_key(result) for result in results[:10]],
        "top_score": results[0].get("score") if results else None,
        "matched_evidence_by_rank": [sorted(match) for match in matches[:10]],
        "gold_coverability": len(coverable_ids) / len(labels) if labels else None,
        "uncoverable_evidence_ids": sorted(
            label["evidence_id"]
            for label in labels
            if label["evidence_id"] not in coverable_ids
        ),
    }
    if event["mode"] == "fast":
        candidates = event["candidate_spans"]
        candidate_hit_ids = {
            label["evidence_id"]
            for candidate in candidates
            for label in matching_labels(candidate, labels)
        }
        output.update(
            {
                "candidate_count": len(candidates),
                "candidate_limit": event.get("candidate_limit"),
                "candidate_evidence_recall": (
                    len(candidate_hit_ids) / len(labels) if labels else None
                ),
                "candidate_recall_given_coverable": (
                    len(candidate_hit_ids) / len(coverable_ids)
                    if coverable_ids
                    else None
                ),
                "candidate_hit": bool(candidate_hit_ids),
                "candidate_matched_evidence": sorted(candidate_hit_ids),
            }
        )
    return output


def summarize_mode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    relevant = [row for row in rows if row["split"] in RELEVANT_SPLITS]
    output: dict[str, Any] = {
        "query_count": len(relevant),
        "hit_at_5": (
            sum(row["hit_at_5"] for row in relevant) / len(relevant) if relevant else None
        ),
        "mrr_at_10": (
            sum(row["mrr_at_10"] for row in relevant) / len(relevant) if relevant else None
        ),
        "ndcg_at_10": (
            sum(row["ndcg_at_10"] for row in relevant) / len(relevant) if relevant else None
        ),
    }
    if rows and rows[0]["mode"] == "fast":
        fast_rows = relevant
        recalls = [row["candidate_evidence_recall"] for row in fast_rows]
        conditional_recalls = [
            row["candidate_recall_given_coverable"]
            for row in fast_rows
            if row["candidate_recall_given_coverable"] is not None
        ]
        output["candidate_coverage"] = {
            "query_count": len(fast_rows),
            "evidence_recall_macro": sum(recalls) / len(recalls) if recalls else None,
            "representable_query_count": len(conditional_recalls),
            "recall_given_coverable_macro": (
                sum(conditional_recalls) / len(conditional_recalls)
                if conditional_recalls
                else None
            ),
            "hit_rate": (
                sum(row["candidate_hit"] for row in fast_rows) / len(fast_rows)
                if fast_rows
                else None
            ),
            "candidate_count": quantiles(
                [float(row["candidate_count"]) for row in fast_rows]
            ),
        }
    return output


def grouped_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row[field]
        if isinstance(value, list):
            for item in value:
                groups[item].append(row)
        else:
            groups[str(value)].append(row)
    output: dict[str, Any] = {}
    for group, group_rows in sorted(groups.items()):
        by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group_rows:
            by_mode[row["mode"]].append(row)
        output[group] = {
            mode: summarize_mode(mode_rows) for mode, mode_rows in sorted(by_mode.items())
        }
    return output


def irrelevant_score_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "irrelevant":
            by_mode[row["mode"]].append(row)
    for mode, mode_rows in sorted(by_mode.items()):
        scores = [float(row["top_score"]) for row in mode_rows if row["top_score"] is not None]
        result[mode] = {
            "query_count": len(mode_rows),
            "top_result_score_distribution": quantiles(scores),
            "note": "Descriptive only; no calibrated no-result threshold is inferred."
        }
    return result


def compare_fast_deep(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_query: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_query[row["query_id"]][row["mode"]] = row
    overlaps: list[float] = []
    intersections: list[int] = []
    hit_deltas: list[float] = []
    for modes in by_query.values():
        fast = modes.get("fast")
        deep = modes.get("deep")
        if not fast or not deep or fast["split"] not in RELEVANT_SPLITS:
            continue
        fast_keys = set(fast["top10_chunk_keys"])
        deep_keys = set(deep["top10_chunk_keys"])
        union = fast_keys | deep_keys
        overlaps.append(1.0 if not union else len(fast_keys & deep_keys) / len(union))
        intersections.append(len(fast_keys & deep_keys))
        hit_deltas.append(float(deep["hit_at_5"]) - float(fast["hit_at_5"]))
    return {
        "query_pair_count": len(overlaps),
        "top10_jaccard_mean": statistics.fmean(overlaps) if overlaps else None,
        "top10_intersection_mean": statistics.fmean(intersections) if intersections else None,
        "deep_minus_fast_hit_at_5": statistics.fmean(hit_deltas) if hit_deltas else None,
    }


def build_report(
    queries: list[dict[str, Any]],
    events: dict[tuple[str, str], dict[str, Any]],
    selected_splits: set[str],
) -> dict[str, Any]:
    query_by_id = {query["id"]: query for query in queries}
    rows: list[dict[str, Any]] = []
    for (query_id, _mode), event in sorted(events.items()):
        query = query_by_id[query_id]
        if query["split"] not in selected_splits:
            continue
        rows.append(metric_for_event(query, event))

    mode_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        mode_rows[row["mode"]].append(row)
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "evaluation_summary",
        "selected_splits": sorted(selected_splits),
        "modes": {
            mode: summarize_mode(mode_rows[mode]) for mode in sorted(mode_rows)
        },
        "groups": {
            "by_split": grouped_summary(rows, "split"),
            "by_language": grouped_summary(rows, "language"),
            "by_category": grouped_summary(rows, "category"),
            "by_tag": grouped_summary(rows, "tags"),
        },
        "fast_deep_comparison": compare_fast_deep(rows),
        "irrelevant_score_distribution": irrelevant_score_distribution(rows),
        "per_query": rows,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=Path("eval/queries.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("eval/manifest.json"))
    parser.add_argument("--corpus", type=Path, default=Path("eval/corpus"))
    parser.add_argument(
        "--results",
        type=Path,
        help="JSONL emitted by examples/evaluate.rs for one fixed model/run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the JSON summary here; otherwise print it to stdout",
    )
    parser.add_argument(
        "--split",
        choices=("all", "development", "holdout", "irrelevant"),
        default="all",
        help="Aggregate only one split (default: all)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow missing query/mode events; malformed events still fail",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate manifest, UTF-8 fixtures, labels, pairs, and counts without results",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        raw_queries = read_jsonl(args.queries, "query definitions")
        manifest = read_json(args.manifest)
        queries = validate_queries(raw_queries, args.corpus)
        validate_manifest(manifest, args.corpus, queries)
        validate_low_overlap_intents(manifest, queries)

        selected_splits = (
            {"development", "holdout", "irrelevant"}
            if args.split == "all"
            else {args.split}
        )
        if args.check:
            summary = {
                "schema_version": SCHEMA_VERSION,
                "record_type": "evaluation_definition_check",
                "valid": True,
                "query_records": len(queries),
                "fixture_files": len(manifest["fixture_inventory"]),
                "selected_splits": sorted(selected_splits),
                "public_snapshot_requirement": manifest["public_snapshot_requirement"],
            }
        else:
            if args.results is None:
                fail("--results is required unless --check is used")
            raw_events = read_jsonl(args.results, "evaluation results")
            query_by_id = {query["id"]: query for query in queries}
            events = [
                normalize_event(event, query_by_id, args.corpus) for event in raw_events
            ]
            selected_queries = [
                query for query in queries if query["split"] in selected_splits
            ]
            event_index = validate_event_set(events, selected_queries, args.allow_partial)
            summary = build_report(queries, event_index, selected_splits)
            summary["definition_status"] = manifest["status"]
            summary["public_snapshot_requirement"] = manifest["public_snapshot_requirement"]

        rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
        return 0
    except ValidationError as exc:
        sys.stderr.write(f"evaluate.py: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
