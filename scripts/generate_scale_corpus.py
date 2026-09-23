#!/usr/bin/env python3
"""Copy the fixed evaluation corpus and add labeled-neutral scale distractors."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--additional", type=int, default=240)
    args = parser.parse_args()
    if args.additional < 129:
        parser.error("--additional must be at least 129 to exercise candidate reduction")
    source = args.source.resolve(strict=True)
    if not source.is_dir() or args.output.exists() or args.manifest.exists():
        parser.error("source must be a directory and output/manifest must not already exist")
    if args.output.resolve().is_relative_to(source) or args.manifest.resolve().is_relative_to(source):
        parser.error("output and manifest must be outside the source corpus")

    manifest = json.loads((source.parent / "manifest.json").read_text(encoding="utf-8"))
    inventory = manifest["fixture_inventory"]
    shutil.copytree(source, args.output)
    distractors = args.output / "distractors"
    distractors.mkdir()
    for index in range(args.additional):
        contents = (
            f"Field note {index:04}: the archive lists color swatches, "
            "room names, and material weights.\n"
        ).encode("utf-8")
        path = distractors / f"extra-{index:04}.txt"
        path.write_bytes(contents)
        inventory.append(
            {
                "path": path.relative_to(args.output).as_posix(),
                "sha256": hashlib.sha256(contents).hexdigest(),
            }
        )
    manifest["fixture_inventory"] = sorted(inventory, key=lambda item: item["path"])
    manifest["scale_evaluation"] = {
        "source": "eval/corpus at the same repository revision",
        "additional_files": args.additional,
        "generation": "deterministic neutral text; no query terms or gold spans selected",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.additional} distractors to {args.output}; manifest {args.manifest}")


if __name__ == "__main__":
    main()
