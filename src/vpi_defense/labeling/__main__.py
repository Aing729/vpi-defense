"""CLI: Label Studio export -> RegionAnnotation JSONL.

Usage:
    python -m vpi_defense.labeling \\
        --in  data/labels/export-2026-07-16.json \\
        --out data/labels/regions.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .adapter import parse_export


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="vpi_defense.labeling")
    p.add_argument("--in", dest="in_path", type=Path, required=True,
                   help="Label Studio export JSON (single object or array)")
    p.add_argument("--out", type=Path, required=True,
                   help="output JSONL of RegionAnnotation records")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_tasks = 0
    n_regions = 0
    with args.out.open("a", encoding="utf-8") as out_f:
        for parent_id, regions in parse_export(args.in_path):
            n_tasks += 1
            for region in regions:
                out_f.write(region.model_dump_json() + "\n")
                n_regions += 1

    print(f"processed {n_tasks} tasks -> {n_regions} regions written to {args.out}")


if __name__ == "__main__":
    main()
