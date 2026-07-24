"""CLI entrypoint for ingesters.

Usage:
    python -m vpi_defense.ingest --dataset docvqa --n 50 --split validation --out data/raw
    python -m vpi_defense.ingest --dataset doclaynet --n 50 --split train --out data/raw
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import INGESTERS, save_items


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="vpi_defense.ingest")
    p.add_argument("--dataset", required=True, choices=sorted(INGESTERS.keys()))
    p.add_argument("--n", type=int, default=50, help="max samples to pull (default: 50)")
    p.add_argument("--split", default=None, help="dataset split; falls back to ingester default")
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output root, e.g. data/raw ; images land under <out>/<dataset>/",
    )
    p.add_argument(
        "--index",
        type=Path,
        default=None,
        help="path to JSONL index; defaults to <out>/index.jsonl",
    )
    p.add_argument("--overwrite", action="store_true", help="overwrite existing image files")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ingester_cls = INGESTERS[args.dataset]
    ingester = ingester_cls()

    # each ingester has a sensible default split
    if args.split is None:
        default_split = {"docvqa": "validation", "doclaynet": "train"}[args.dataset]
    else:
        default_split = args.split

    items = ingester.fetch(n=args.n, split=default_split)
    written = save_items(items, out_root=args.out, index_path=args.index, overwrite=args.overwrite)
    print(f"wrote {len(written)} items to {args.out} (index: {args.index or args.out / 'index.jsonl'})")


if __name__ == "__main__":
    main()
