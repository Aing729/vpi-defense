"""CLI: run the VLM over a set of images and produce region annotations.

Reads an ImageMetadata JSONL (from ingest CLI) or a VariantSample JSONL
(from variants CLI). For each entry, calls the VLM and appends resulting
RegionAnnotation records to an output JSONL.

Usage:
    # Raw images with Claude (default provider)
    python -m vpi_defense.analysis \\
        --in  data/raw/index.jsonl \\
        --root data/raw \\
        --out data/analysis/regions.jsonl \\
        --save-raw-dir data/analysis/raw_responses

    # Variants with a specific provider
    python -m vpi_defense.analysis \\
        --in  data/variants/variants_index.jsonl \\
        --root data/variants \\
        --input-kind variant \\
        --provider claude \\
        --out data/analysis/regions_variants.jsonl

    # Dry-run without hitting any API
    python -m vpi_defense.analysis --in ... --root ... --out ... --provider mock
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from vpi_defense.schemas import ImageMetadata, VariantSample

from .vlm_client import (
    BaseVLMClient,
    ClaudeVLMClient,
    GeminiVLMClient,
    MockVLMClient,
    canned_pharmacy_response,
)


# Default per-provider models. Override with --model.
_DEFAULT_MODELS = {
    "claude": "claude-sonnet-4-6",
    "gemini": "gemini-3.5-flash",
    "mock": "mock-vlm",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="vpi_defense.analysis")
    p.add_argument("--in", dest="in_path", type=Path, required=True,
                   help="input JSONL (ImageMetadata or VariantSample records)")
    p.add_argument("--root", type=Path, required=True,
                   help="root directory where image files live "
                        "(matches --out from ingest / variants CLI)")
    p.add_argument("--out", type=Path, required=True,
                   help="output JSONL of RegionAnnotation records")
    p.add_argument("--input-kind", choices=("image", "variant"), default="image",
                   help="whether --in is ImageMetadata or VariantSample records")
    p.add_argument("--provider", choices=("claude", "gemini", "mock"), default="claude",
                   help="which VLM to call (default: claude)")
    p.add_argument("--model", default=None,
                   help="model name override; defaults per provider "
                        f"({_DEFAULT_MODELS})")
    p.add_argument("--save-raw-dir", type=Path, default=None,
                   help="if set, dump raw model responses to <dir>/<id>.json for replay")
    return p.parse_args()


def _build_client(args: argparse.Namespace) -> BaseVLMClient:
    model = args.model or _DEFAULT_MODELS[args.provider]
    if args.provider == "mock":
        return MockVLMClient(canned_pharmacy_response())
    if args.provider == "claude":
        return ClaudeVLMClient(model=model)
    if args.provider == "gemini":
        return GeminiVLMClient(model=model)
    raise SystemExit(f"unknown provider: {args.provider}")


def _iter_records(path: Path, kind: str):
    """Yield (id, image_path, parent_kind) tuples from either input schema."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if kind == "image":
                m = ImageMetadata.model_validate_json(line)
                yield m.image_id, m.image_path, "image"
            else:
                v = VariantSample.model_validate_json(line)
                yield v.variant_id, v.image_path, "variant"


def main() -> None:
    args = _parse_args()
    client = _build_client(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_processed = 0
    n_failed = 0
    with args.out.open("a", encoding="utf-8") as out_f:
        for parent_id, img_rel_path, parent_kind in _iter_records(args.in_path, args.input_kind):
            src = args.root / img_rel_path
            if not src.exists():
                print(f"skip (missing image): {src}")
                continue

            image = Image.open(src).convert("RGB")

            save_raw_to = None
            if args.save_raw_dir is not None:
                save_raw_to = args.save_raw_dir / f"{parent_id}.json"

            try:
                result = client.extract_regions(
                    image=image,
                    parent_id=parent_id,
                    parent_kind=parent_kind,
                    save_raw_to=save_raw_to,
                )
            except Exception as e:
                # Per-image failure should NOT kill the batch. The raw
                # response (if any) is still on disk under save_raw_dir
                # for post-hoc inspection.
                n_failed += 1
                print(f"  FAILED {parent_id}: {type(e).__name__}: {e}")
                continue

            for region in result.regions:
                out_f.write(region.model_dump_json() + "\n")

            n_processed += 1
            print(f"  {parent_id}: {len(result.regions)} regions, {result.latency_ms:.0f}ms")

    print(f"processed {n_processed} images ({n_failed} failed) → {args.out}")


if __name__ == "__main__":
    main()
