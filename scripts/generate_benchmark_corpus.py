#!/usr/bin/env python3
"""Generate a deterministic UTF-8 corpus for fresh-process CLI benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--files", type=int, default=1000)
    parser.add_argument("--total-bytes", type=int, default=10 * 1024 * 1024)
    args = parser.parse_args()
    if args.files < 1 or args.total_bytes < args.files:
        parser.error("--files and --total-bytes must be positive; each file needs at least one byte")
    args.root.mkdir(parents=True, exist_ok=False)

    template = (
        b"pub fn retry_delay(attempt: u32) -> u64 {\n"
        b"    let bounded = attempt.min(8);\n"
        b"    250_u64.saturating_mul(1_u64 << bounded)\n"
        b"}\n\n"
        b"pub fn request_timeout() -> u64 { 30 }\n\n"
    )
    base_size, remainder = divmod(args.total_bytes, args.files)
    for index in range(args.files):
        size = base_size + (index < remainder)
        repetitions = (size + len(template) - 1) // len(template)
        contents = (template * repetitions)[:size]
        (args.root / f"module-{index:04}.rs").write_bytes(contents)

    print(f"created {args.files} UTF-8 files, {args.total_bytes} bytes at {args.root}")


if __name__ == "__main__":
    main()
