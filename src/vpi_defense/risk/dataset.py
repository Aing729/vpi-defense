"""Assemble the training dataset for the risk function.

Inputs (all JSONL):
    regions_gt              : injection ground truth from the variant generator
                              (parent_kind='variant', RiskLabels.is_injection=True)
    regions_pred_variants   : Gemini predictions on variant images
    regions_pred_raw        : Gemini predictions on raw (base) images

Steps per parent:
    1. Match predicted regions to GT injections (M6.1, IoU >= threshold)
    2. Load the parent image and build ImageContext (M6.2)
    3. Build SemanticContext from all predicted texts on that parent (M6.3)
    4. For each prediction, extract visual + semantic features → 10-dim vector
    5. Assign label: matched-to-injection → 1, else → 0

Output: `LabeledDataset` with feature matrix X (N x 10), label vector y (N,),
group vector g (N,) for grouped CV, and per-sample metadata for diagnostics.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from vpi_defense.features import (
    SemanticContext,
    build_image_context,
    build_semantic_context,
    extract_semantic_features,
    extract_visual_features,
    match_regions,
)
from vpi_defense.schemas import (
    ImageMetadata,
    RegionAnnotation,
    VariantSample,
)


FEATURE_NAMES: tuple[str, ...] = (
    # visual
    "area_ratio", "center_distance", "contrast", "bg_separation", "saliency",
    # semantic
    "imperativeness", "context_relevance", "action_inducement",
    "url_qr_relevance", "informativeness",
)


@dataclass
class LabeledSample:
    """One row of the training dataset."""
    parent_id: str
    parent_kind: str          # "image" or "variant"
    region_id: str
    base_image_id: str        # for group-split (variants share their base's group)
    features: np.ndarray      # shape (10,)
    label: int                # 1 if injection, 0 otherwise
    matched_gt_id: str | None = None
    iou: float = 0.0


@dataclass
class LabeledDataset:
    samples: list[LabeledSample] = field(default_factory=list)

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(X, y, groups) as numpy arrays. groups is base_image_id encoded as int."""
        if not self.samples:
            return (np.zeros((0, len(FEATURE_NAMES))), np.zeros((0,), dtype=int),
                    np.zeros((0,), dtype=int))
        X = np.stack([s.features for s in self.samples])
        y = np.array([s.label for s in self.samples], dtype=int)
        # encode base_image_id to consecutive ints for grouped splitters
        gid_map: dict[str, int] = {}
        for s in self.samples:
            if s.base_image_id not in gid_map:
                gid_map[s.base_image_id] = len(gid_map)
        g = np.array([gid_map[s.base_image_id] for s in self.samples], dtype=int)
        return X, y, g

    def summary(self) -> dict[str, int | float]:
        y = np.array([s.label for s in self.samples], dtype=int)
        n = len(y)
        n_pos = int(y.sum())
        n_neg = n - n_pos
        base_ids = {s.base_image_id for s in self.samples}
        return {
            "n_samples": n,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_base_images": len(base_ids),
            "class_ratio": n_pos / n if n else 0.0,
        }


# ---------------------------------------------------------------------------
# JSONL loaders
# ---------------------------------------------------------------------------


def load_regions(path: Path) -> list[RegionAnnotation]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [RegionAnnotation.model_validate_json(line) for line in f if line.strip()]


def load_variants(path: Path) -> list[VariantSample]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [VariantSample.model_validate_json(line) for line in f if line.strip()]


def load_images(path: Path) -> list[ImageMetadata]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [ImageMetadata.model_validate_json(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------


def _features_vector(vf, sf) -> np.ndarray:
    return np.array([
        vf.area_ratio, vf.center_distance, vf.contrast, vf.bg_separation, vf.saliency,
        sf.imperativeness, sf.context_relevance, sf.action_inducement,
        sf.url_qr_relevance, sf.informativeness,
    ], dtype=np.float32)


def _resolve_image_path(
    parent_id: str,
    parent_kind: str,
    images: dict[str, ImageMetadata],
    variants: dict[str, VariantSample],
    raw_root: Path,
    variants_root: Path,
) -> Path | None:
    """Return the on-disk path for the parent's image, or None if missing."""
    if parent_kind == "image":
        meta = images.get(parent_id)
        return raw_root / meta.image_path if meta else None
    else:
        var = variants.get(parent_id)
        return variants_root / var.image_path if var else None


def _base_id_for_parent(
    parent_id: str, parent_kind: str, variants: dict[str, VariantSample],
) -> str:
    """Group id for split leakage prevention.

    - For a variant, use its base_image_id so all variants of the same
      original image sit in the same fold.
    - For a raw image, the parent_id IS its base id.
    """
    if parent_kind == "image":
        return parent_id
    var = variants.get(parent_id)
    return var.base_image_id if var else parent_id


def _group_by_parent(regions: Iterable[RegionAnnotation]) -> dict[str, list[RegionAnnotation]]:
    grouped: dict[str, list[RegionAnnotation]] = {}
    for r in regions:
        grouped.setdefault(r.parent_id, []).append(r)
    return grouped


def build_dataset(
    regions_gt: list[RegionAnnotation],
    regions_pred_variants: list[RegionAnnotation],
    regions_pred_raw: list[RegionAnnotation],
    images: list[ImageMetadata],
    variants: list[VariantSample],
    raw_root: Path,
    variants_root: Path,
    iou_threshold: float = 0.3,
    embedder=None,
    verbose: bool = True,
) -> LabeledDataset:
    """Materialize the full labeled dataset by iterating over every parent
    that has predictions, matching GT, and extracting features.
    """
    images_by_id = {m.image_id: m for m in images}
    variants_by_id = {v.variant_id: v for v in variants}

    gt_by_parent = _group_by_parent(regions_gt)
    pred_var_by_parent = _group_by_parent(regions_pred_variants)
    pred_raw_by_parent = _group_by_parent(regions_pred_raw)

    dataset = LabeledDataset()
    n_missing_image = 0

    # ---- Variants ----
    for parent_id, preds in sorted(pred_var_by_parent.items()):
        gts = gt_by_parent.get(parent_id, [])
        report = match_regions(gts, preds, parent_id, iou_threshold=iou_threshold)
        # positive: predictions that matched an injection
        matched_pred_ids = {m.pred.region_id: (m.gt.region_id, m.iou) for m in report.matched}

        img_path = _resolve_image_path(
            parent_id, "variant", images_by_id, variants_by_id, raw_root, variants_root,
        )
        if img_path is None or not img_path.exists():
            n_missing_image += 1
            continue

        pil = Image.open(img_path).convert("RGB")
        img_ctx = build_image_context(pil)
        sem_ctx = build_semantic_context(
            all_region_texts=[p.ocr_text or "" for p in preds], embedder=embedder,
        )
        base_id = _base_id_for_parent(parent_id, "variant", variants_by_id)

        for pred in preds:
            vf = extract_visual_features(pred.bbox, img_ctx)
            sf_ = extract_semantic_features(pred.ocr_text or "", pred.region_type, sem_ctx)
            feat = _features_vector(vf, sf_)
            matched = matched_pred_ids.get(pred.region_id)
            label = 1 if matched else 0
            dataset.samples.append(LabeledSample(
                parent_id=parent_id, parent_kind="variant",
                region_id=pred.region_id, base_image_id=base_id,
                features=feat, label=label,
                matched_gt_id=(matched[0] if matched else None),
                iou=(matched[1] if matched else 0.0),
            ))

    # ---- Raw images: everything is negative (no injection was ever inserted) ----
    for parent_id, preds in sorted(pred_raw_by_parent.items()):
        img_path = _resolve_image_path(
            parent_id, "image", images_by_id, variants_by_id, raw_root, variants_root,
        )
        if img_path is None or not img_path.exists():
            n_missing_image += 1
            continue

        pil = Image.open(img_path).convert("RGB")
        img_ctx = build_image_context(pil)
        sem_ctx = build_semantic_context(
            all_region_texts=[p.ocr_text or "" for p in preds], embedder=embedder,
        )
        base_id = _base_id_for_parent(parent_id, "image", variants_by_id)

        for pred in preds:
            vf = extract_visual_features(pred.bbox, img_ctx)
            sf_ = extract_semantic_features(pred.ocr_text or "", pred.region_type, sem_ctx)
            feat = _features_vector(vf, sf_)
            dataset.samples.append(LabeledSample(
                parent_id=parent_id, parent_kind="image",
                region_id=pred.region_id, base_image_id=base_id,
                features=feat, label=0,
            ))

    if verbose:
        s = dataset.summary()
        print(f"[dataset] {s['n_samples']} samples "
              f"({s['n_positive']} pos, {s['n_negative']} neg) "
              f"across {s['n_base_images']} base images. "
              f"Missing image files: {n_missing_image}")
    return dataset
