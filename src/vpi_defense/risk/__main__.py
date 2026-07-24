"""CLI: train and evaluate the risk function.

Usage:
    # Full training with real data
    python -m vpi_defense.risk \\
        --regions-gt            data/variants/regions_index.jsonl \\
        --regions-pred-variants data/analysis/regions_variants.jsonl \\
        --regions-pred-raw      data/analysis/regions_raw.jsonl \\
        --images                data/raw/index.jsonl \\
        --variants              data/variants/variants_index.jsonl \\
        --raw-root              data/raw \\
        --variants-root         data/variants \\
        --model-out             models/risk_v1.pkl

    # Without sentence-transformer (falls back to rule-only semantic scoring)
    ... --no-embedder
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import build_dataset, load_images, load_regions, load_variants
from .evaluate import (
    compute_metrics,
    cross_validate,
    format_coefficients,
    format_cv_report,
    format_metrics,
)
from .model import RiskModel


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="vpi_defense.risk")
    p.add_argument("--regions-gt", type=Path, required=True,
                   help="JSONL of ground-truth injection RegionAnnotation records "
                        "(from variants CLI, parent_kind='variant')")
    p.add_argument("--regions-pred-variants", type=Path, required=True,
                   help="JSONL of Gemini predictions on variant images")
    p.add_argument("--regions-pred-raw", type=Path, required=True,
                   help="JSONL of Gemini predictions on raw base images")
    p.add_argument("--images", type=Path, required=True,
                   help="JSONL of ImageMetadata for raw images")
    p.add_argument("--variants", type=Path, required=True,
                   help="JSONL of VariantSample records")
    p.add_argument("--raw-root", type=Path, required=True,
                   help="root directory for raw images (matches ingest --out)")
    p.add_argument("--variants-root", type=Path, required=True,
                   help="root directory for variant images (matches variants --out)")

    p.add_argument("--iou-threshold", type=float, default=0.3,
                   help="minimum IoU to count a prediction as matching a GT injection")
    p.add_argument("--C", type=float, default=1.0,
                   help="inverse of L2 regularization strength (sklearn convention)")
    p.add_argument("--n-splits", type=int, default=5,
                   help="folds for grouped cross-validation")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="probability cutoff for the binary decision")
    p.add_argument("--no-embedder", action="store_true",
                   help="skip loading sentence-transformer (rule-only semantic scoring)")

    p.add_argument("--model-out", type=Path, default=None,
                   help="if set, refit on the full dataset and save the model here")
    return p.parse_args()


def _load_embedder(disabled: bool):
    if disabled:
        return None
    from vpi_defense.features import try_load_default_embedder
    print("loading sentence-transformer (first run downloads ~120MB)...")
    embedder = try_load_default_embedder()
    if embedder is None:
        print("  sentence-transformer unavailable; falling back to rule-only.")
    else:
        print("  ok.")
    return embedder


def main() -> None:
    args = _parse_args()

    print("=== loading inputs ===")
    regions_gt = load_regions(args.regions_gt)
    regions_pred_var = load_regions(args.regions_pred_variants)
    regions_pred_raw = load_regions(args.regions_pred_raw)
    images = load_images(args.images)
    variants = load_variants(args.variants)
    print(f"  gt injections:        {len(regions_gt)}")
    print(f"  predictions (raw):    {len(regions_pred_raw)}")
    print(f"  predictions (var):    {len(regions_pred_var)}")
    print(f"  base images:          {len(images)}")
    print(f"  variants:             {len(variants)}")

    embedder = _load_embedder(args.no_embedder)

    print("\n=== assembling labeled dataset ===")
    dataset = build_dataset(
        regions_gt=regions_gt,
        regions_pred_variants=regions_pred_var,
        regions_pred_raw=regions_pred_raw,
        images=images,
        variants=variants,
        raw_root=args.raw_root,
        variants_root=args.variants_root,
        iou_threshold=args.iou_threshold,
        embedder=embedder,
    )
    X, y, groups = dataset.to_arrays()
    if len(y) == 0:
        raise SystemExit("dataset is empty. Check paths and JSONL contents.")

    print("\n=== grouped cross-validation ===")
    report = cross_validate(
        X, y, groups,
        n_splits=args.n_splits,
        threshold=args.threshold,
        C=args.C,
    )
    print(format_cv_report(report))

    print("\n=== training on the full dataset ===")
    model = RiskModel.train(X, y, C=args.C)
    train_metrics = compute_metrics(y, model.predict_proba(X), threshold=args.threshold)
    print("training-set metrics (upper bound, not for generalization claims):")
    print(format_metrics(train_metrics))
    print()
    print(format_coefficients(model))

    if args.model_out is not None:
        model.save(args.model_out)
        print(f"\nsaved model → {args.model_out}")


if __name__ == "__main__":
    main()
