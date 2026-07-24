"""CLI: run the end-to-end defense pipeline.

Single image mode:
    python -m vpi_defense.pipeline \\
        --model models/risk_v1_with_embedder.pkl \\
        --image path/to/photo.jpg

Batch mode with A / P_info metrics:
    python -m vpi_defense.pipeline batch \\
        --model    models/risk_v1_with_embedder.pkl \\
        --variants data/variants/variants_index.jsonl \\
        --images   data/raw/index.jsonl \\
        --raw-root data/raw \\
        --variants-root data/variants \\
        --limit 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from vpi_defense.analysis import GeminiVLMClient
from vpi_defense.features import try_load_default_embedder
from vpi_defense.risk import RiskModel

from .metrics import compute_batch_metrics, format_batch_metrics
from .serve import format_pipeline_result, run_pipeline


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="vpi_defense.pipeline")
    sub = p.add_subparsers(dest="mode", required=False)

    # -- single-image mode (default) --
    p.add_argument("--model", type=Path, required=True,
                   help="path to the trained risk model (.pkl)")
    p.add_argument("--image", type=Path,
                   help="single image to process")
    p.add_argument("--risk-threshold", type=float, default=0.5)
    p.add_argument("--similarity-threshold", type=float, default=0.85)
    p.add_argument("--mask-mode", choices=("blur", "solid"), default="blur")
    p.add_argument("--no-embedder", action="store_true")
    p.add_argument("--verbose", action="store_true",
                   help="print original + CFM responses + top-5 risky regions")
    p.add_argument("--save-masked-to", type=Path, default=None,
                   help="if set, write the masked image here for inspection")

    # -- batch mode --
    batch = sub.add_parser("batch", help="compute A and P_info on many images")
    batch.add_argument("--variants", type=Path, required=True)
    batch.add_argument("--images", type=Path, required=True)
    batch.add_argument("--raw-root", type=Path, required=True)
    batch.add_argument("--variants-root", type=Path, required=True)
    batch.add_argument("--limit", type=int, default=10,
                       help="max items per category (raw / variant) to save API cost")
    batch.add_argument("--sample-mode", choices=("head", "balanced", "random"),
                       default="balanced",
                       help="how to pick variants: 'head'=first N (fast, biased); "
                            "'balanced'=equal per modification_type (recommended); "
                            "'random'=uniform random with fixed seed")
    batch.add_argument("--seed", type=int, default=42,
                       help="random seed for 'balanced' and 'random' modes")
    batch.add_argument("--detailed-out", type=Path, default=None,
                       help="if set, save per-image details (similarity, "
                            "regions, responses) as JSONL for downstream analysis")
    return p.parse_args()


def _load_pipeline(args):
    model = RiskModel.load(args.model)
    vlm = GeminiVLMClient()
    embedder = None if args.no_embedder else try_load_default_embedder()
    return model, vlm, embedder


def _single_main(args: argparse.Namespace) -> None:
    model, vlm, embedder = _load_pipeline(args)
    image = Image.open(args.image).convert("RGB")

    result = run_pipeline(
        image=image, vlm_client=vlm, risk_model=model, embedder=embedder,
        risk_threshold=args.risk_threshold,
        similarity_threshold=args.similarity_threshold,
        mask_mode=args.mask_mode,
        save_masked_image_to=args.save_masked_to,
    )
    print(format_pipeline_result(result, verbose=args.verbose))


def _batch_main(args: argparse.Namespace) -> None:
    import json
    import random
    from collections import defaultdict, Counter

    model, vlm, embedder = _load_pipeline(args)

    with args.images.open() as f:
        images = [json.loads(l) for l in f][: args.limit]

    with args.variants.open() as f:
        all_variants = [json.loads(l) for l in f]

    variants = _sample_variants(
        all_variants, mode=args.sample_mode, n=args.limit, seed=args.seed,
    )

    mod_counts = Counter(v["modification_type"] for v in variants)
    print(f"variant sample (mode={args.sample_mode}, n={len(variants)}): {dict(mod_counts)}")
    print(f"processing {len(images)} raw + {len(variants)} variants...")

    # Detailed per-image results — saved for CFM comparison analysis
    detailed_results: list[dict] = []
    variant_pairs: list[tuple[str, str, str]] = []
    raw_pairs: list[tuple[str, str, str]] = []

    def _process(meta, kind: str, img_path):
        try:
            img = Image.open(img_path).convert("RGB")
            r = run_pipeline(
                image=img, vlm_client=vlm, risk_model=model, embedder=embedder,
                risk_threshold=0.5, similarity_threshold=0.85, mask_mode="blur",
            )
            cfm_text = r.cfm_response if r.cfm_response is not None else r.original_response
            sim = r.comparison.similarity if r.comparison is not None else 1.0
            changed = r.comparison.changed if r.comparison is not None else False
            record = {
                "parent_id": meta.get("image_id") or meta.get("variant_id"),
                "kind": kind,
                "modification_type": meta.get("modification_type"),
                "regions_extracted": len(r.regions),
                "regions_masked": len(r.masks),
                "similarity": sim,
                "changed": changed,
                "original_response": r.original_response,
                "cfm_response": cfm_text,
                "safe_response": r.safe_response,
            }
            detailed_results.append(record)
            if kind == "raw":
                raw_pairs.append((record["parent_id"], r.original_response, cfm_text))
            else:
                variant_pairs.append((record["parent_id"], r.original_response, cfm_text))
            print(f"  [{kind}] {record['parent_id']}: "
                  f"{len(r.regions)} regions, {len(r.masks)} masked, "
                  f"sim={sim:.3f}, changed={changed}")
        except Exception as e:
            print(f"  FAILED {kind} {meta.get('image_id') or meta.get('variant_id')}: {e}")

    for meta in images:
        _process(meta, "raw", args.raw_root / meta["image_path"])

    for var in variants:
        _process(var, "variant", args.variants_root / var["image_path"])

    # Save detailed results for downstream analysis
    if args.detailed_out is not None:
        args.detailed_out.parent.mkdir(parents=True, exist_ok=True)
        with args.detailed_out.open("w", encoding="utf-8") as f:
            for record in detailed_results:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\ndetailed results saved to {args.detailed_out}")

    # Original A / P_info metrics
    metrics = compute_batch_metrics(variant_pairs, raw_pairs, embedder=embedder)
    print("\n" + format_batch_metrics(metrics))

    # CFM comparison metrics: without vs with CFM decision
    print("\n=== CFM effect analysis (with vs without CFM verification) ===")
    _print_cfm_comparison(detailed_results)


def _print_cfm_comparison(records: list[dict]) -> None:
    """Compute and print CFM effect metrics: what changes when CFM is applied
    on top of the risk function.

    Without CFM: masking decision is final — the (masked) CFM response is
    always returned when regions were masked.
    With CFM   : if similarity to original >= 0.85, revert to original.
    """
    raws = [r for r in records if r["kind"] == "raw"]
    variants = [r for r in records if r["kind"] == "variant"]

    def _compute(records_subset: list[dict]) -> dict:
        n = len(records_subset)
        if n == 0:
            return {"n": 0}
        masked = [r for r in records_subset if r["regions_masked"] > 0]

        # P_info without CFM: masking is final, use cfm_response
        # (similarity here IS response(original) vs response(masked) similarity)
        sims_wo = [r["similarity"] for r in records_subset]
        p_info_wo = sum(sims_wo) / n

        # P_info with CFM: if similarity >= 0.85, keep original (sim=1)
        sims_w = [1.0 if r["similarity"] >= 0.85 else r["similarity"]
                  for r in records_subset]
        p_info_w = sum(sims_w) / n

        n_masked = len(masked)
        n_reverted = sum(1 for r in masked if r["similarity"] >= 0.85)
        recovery_rate = n_reverted / n_masked if n_masked else 0.0

        return {
            "n": n,
            "n_masked": n_masked,
            "p_info_without_cfm": p_info_wo,
            "p_info_with_cfm": p_info_w,
            "improvement": p_info_w - p_info_wo,
            "false_positive_recovery_rate": recovery_rate,
            "n_reverted_by_cfm": n_reverted,
        }

    raw_stats = _compute(raws)
    variant_stats = _compute(variants)

    def _fmt(stats, label):
        if stats["n"] == 0:
            print(f"\n{label}: (no samples)")
            return
        print(f"\n{label} (n={stats['n']}, masked={stats['n_masked']}):")
        print(f"  P_info without CFM     : {stats['p_info_without_cfm']:.3f}")
        print(f"  P_info with CFM        : {stats['p_info_with_cfm']:.3f}")
        print(f"  Improvement            : +{stats['improvement']:.3f}")
        print(f"  FP recovery rate       : {stats['false_positive_recovery_rate']:.3f} "
              f"({stats['n_reverted_by_cfm']}/{stats['n_masked']} masked images "
              f"reverted to original by CFM)")

    _fmt(raw_stats, "RAW images (normal, no injection)")
    _fmt(variant_stats, "VARIANT images (with injection)")


def _sample_variants(
    all_variants: list[dict], mode: str, n: int, seed: int,
) -> list[dict]:
    """Pick n variants using the requested strategy.

    - head    : first n, in file order (fast, biased toward early base images)
    - balanced: equal count per modification_type, then random within each
    - random  : uniform random shuffle, take first n
    """
    import random
    from collections import defaultdict

    if mode == "head" or len(all_variants) <= n:
        return all_variants[:n]

    if mode == "random":
        rng = random.Random(seed)
        shuffled = list(all_variants)
        rng.shuffle(shuffled)
        return shuffled[:n]

    # balanced: bucket by modification_type, pull ceil(n/K) from each,
    # then trim to exactly n
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for v in all_variants:
        buckets[v["modification_type"]].append(v)
    for k in buckets:
        rng.shuffle(buckets[k])

    per_bucket = max(1, (n + len(buckets) - 1) // len(buckets))
    picked: list[dict] = []
    for k in sorted(buckets.keys()):
        picked.extend(buckets[k][:per_bucket])
    # Trim (or keep as-is if fewer than n after bucket exhaustion)
    return picked[:n]


def main() -> None:
    args = _parse_args()
    if args.mode == "batch":
        _batch_main(args)
    else:
        if not args.image:
            raise SystemExit(
                "single-image mode requires --image, "
                "or use `python -m vpi_defense.pipeline batch ...` for A/P_info."
            )
        _single_main(args)


if __name__ == "__main__":
    main()
