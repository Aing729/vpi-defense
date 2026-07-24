"""CLI: explain model predictions with Integrated Gradients.

Usage:
    python -m vpi_defense.risk.explain \\
        --model                 models/risk_v1.pkl \\
        --regions-gt            data/variants/regions_index.jsonl \\
        --regions-pred-variants data/analysis/regions_variants.jsonl \\
        --regions-pred-raw      data/analysis/regions_raw.jsonl \\
        --images                data/raw/index.jsonl \\
        --variants              data/variants/variants_index.jsonl \\
        --raw-root              data/raw \\
        --variants-root         data/variants \\
        --top-k                 5

Prints IG attributions for the top-k regions with highest predicted
injection probability (positives ranked first, then hard negatives).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import build_dataset, load_images, load_regions, load_variants
from .ig import format_attribution, linear_attribution
from .model import RiskModel


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="vpi_defense.risk.explain")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--regions-gt", type=Path, required=True)
    p.add_argument("--regions-pred-variants", type=Path, required=True)
    p.add_argument("--regions-pred-raw", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--variants", type=Path, required=True)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--variants-root", type=Path, required=True)
    p.add_argument("--iou-threshold", type=float, default=0.3)
    p.add_argument("--top-k", type=int, default=5,
                   help="show this many samples in each category")
    p.add_argument("--no-embedder", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print("=== loading model ===")
    model = RiskModel.load(args.model)

    print("\n=== assembling dataset ===")
    embedder = None
    if not args.no_embedder:
        from vpi_defense.features import try_load_default_embedder
        print("loading sentence-transformer...")
        embedder = try_load_default_embedder()

    dataset = build_dataset(
        regions_gt=load_regions(args.regions_gt),
        regions_pred_variants=load_regions(args.regions_pred_variants),
        regions_pred_raw=load_regions(args.regions_pred_raw),
        images=load_images(args.images),
        variants=load_variants(args.variants),
        raw_root=args.raw_root, variants_root=args.variants_root,
        iou_threshold=args.iou_threshold, embedder=embedder,
    )
    X, y, _ = dataset.to_arrays()
    probs = model.predict_proba(X)

    # ---- Category 1: highest-confidence positives (true injections) ----
    pos_idx = [i for i, yi in enumerate(y) if yi == 1]
    pos_idx.sort(key=lambda i: -probs[i])
    print(f"\n=== top-{args.top_k} true injections by predicted probability ===")
    for rank, i in enumerate(pos_idx[: args.top_k], start=1):
        s = dataset.samples[i]
        a = linear_attribution(model, X[i])
        print(f"\n[{rank}] region={s.region_id}  parent={s.parent_id}  "
              f"predicted_prob={probs[i]:.3f}  IoU_with_gt={s.iou:.3f}")
        print(format_attribution(a, top_k=6))

    # ---- Category 2: high-scoring negatives (false positives) ----
    fp_idx = [i for i, yi in enumerate(y) if yi == 0]
    fp_idx.sort(key=lambda i: -probs[i])
    print(f"\n\n=== top-{args.top_k} false positives (negatives ranked high) ===")
    for rank, i in enumerate(fp_idx[: args.top_k], start=1):
        s = dataset.samples[i]
        a = linear_attribution(model, X[i])
        print(f"\n[{rank}] region={s.region_id}  parent={s.parent_id}  "
              f"predicted_prob={probs[i]:.3f}")
        print(format_attribution(a, top_k=6))

    # ---- Category 3: missed positives (false negatives) ----
    fn_idx = [i for i, yi in enumerate(y) if yi == 1]
    fn_idx.sort(key=lambda i: probs[i])
    print(f"\n\n=== top-{args.top_k} missed injections (lowest predicted prob) ===")
    for rank, i in enumerate(fn_idx[: args.top_k], start=1):
        s = dataset.samples[i]
        a = linear_attribution(model, X[i])
        print(f"\n[{rank}] region={s.region_id}  parent={s.parent_id}  "
              f"predicted_prob={probs[i]:.3f}  IoU_with_gt={s.iou:.3f}")
        print(format_attribution(a, top_k=6))


if __name__ == "__main__":
    main()
