"""CLI entrypoint for variant generation.

Reads an image index (JSONL of ImageMetadata records, produced by the
ingest CLI), materializes each variant to disk, and appends a
variants index (JSONL of VariantSample) plus a regions index (JSONL
of RegionAnnotation) so downstream stages can load them uniformly.

Usage:
    python -m vpi_defense.variants \\
        --in  data/raw/index.jsonl \\
        --raw data/raw \\
        --out data/variants \\
        --n-per-image 3 \\
        --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from vpi_defense.ingest.base import load_index
from vpi_defense.schemas import ModificationType

from .generator import DEFAULT_MODIFICATIONS, generate_variants_for_image


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="vpi_defense.variants")
    p.add_argument("--in", dest="in_path", type=Path, required=True,
                   help="path to ImageMetadata JSONL (from ingest CLI)")
    p.add_argument("--raw", type=Path, required=True,
                   help="root directory where raw images live (matches ingest --out)")
    p.add_argument("--out", type=Path, required=True,
                   help="root directory for variant PNGs")
    p.add_argument("--n-per-image", type=int, default=len(DEFAULT_MODIFICATIONS),
                   help=f"variants per base image (default: {len(DEFAULT_MODIFICATIONS)})")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--font-path", type=str, default=None,
                   help="explicit path to a CJK-capable TTF; auto-detect if omitted")
    p.add_argument("--variants-index", type=Path, default=None,
                   help="path to VariantSample JSONL (default: <out>/variants_index.jsonl)")
    p.add_argument("--regions-index", type=Path, default=None,
                   help="path to injected-region RegionAnnotation JSONL "
                        "(default: <out>/regions_index.jsonl)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    images = load_index(args.in_path)
    if not images:
        raise SystemExit(f"index is empty: {args.in_path}")

    args.out.mkdir(parents=True, exist_ok=True)
    variants_idx = args.variants_index or (args.out / "variants_index.jsonl")
    regions_idx = args.regions_index or (args.out / "regions_index.jsonl")

    n_variants = 0
    with variants_idx.open("a", encoding="utf-8") as v_out, \
         regions_idx.open("a", encoding="utf-8") as r_out:
        for meta in images:
            src_path = args.raw / meta.image_path
            if not src_path.exists():
                print(f"skip (missing raw): {src_path}")
                continue
            base_image = Image.open(src_path).convert("RGB")

            for gv in generate_variants_for_image(
                base_image=base_image,
                base_meta=meta,
                seed=args.seed,
                n_per_image=args.n_per_image,
                font_path=args.font_path,
            ):
                out_path = args.out / gv.variant.image_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(gv.image_bytes)

                v_out.write(gv.variant.model_dump_json() + "\n")
                r_out.write(gv.injected_region.model_dump_json() + "\n")
                n_variants += 1

    print(f"wrote {n_variants} variants under {args.out}")
    print(f"  variants index: {variants_idx}")
    print(f"  regions index:  {regions_idx}")


if __name__ == "__main__":
    main()
