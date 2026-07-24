"""End-to-end pipeline: image → safe natural-language response.

Flow:
  1. VLM extracts regions from the image (M5)
  2. Visual + semantic features for each region (M6.2, M6.3)
  3. Risk model scores each region (M6.4)
  4. Regions above threshold → mask (M7 cfm.py)
  5. VLM describes the masked image
  6. Compare original vs masked description (M7 compare.py)
  7. Return the safer of the two

Everything above is wired together in `run_pipeline()`. The CLI wrapper is
in `__main__.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vpi_defense.analysis import GeminiVLMClient
from vpi_defense.features import (
    build_image_context,
    build_semantic_context,
    extract_semantic_features,
    extract_visual_features,
)
from vpi_defense.risk import RiskModel
from vpi_defense.schemas import BBox

from .cfm import MaskSpec, apply_masks, select_masks_by_threshold
from .compare import ComparisonResult, compare_responses


@dataclass
class ScoredRegion:
    """One extracted region with its risk score, kept for downstream display."""
    bbox: BBox
    label: str
    ocr_text: str | None
    region_type: str
    risk_prob: float
    features: np.ndarray  # shape (10,) — for later IG / debug


@dataclass
class PipelineResult:
    """The final artifact of the pipeline."""
    original_response: str
    cfm_response: str | None            # None if no regions were masked
    safe_response: str
    regions: list[ScoredRegion] = field(default_factory=list)
    masks: list[MaskSpec] = field(default_factory=list)
    comparison: ComparisonResult | None = None


def run_pipeline(
    image: Image.Image,
    vlm_client: GeminiVLMClient,
    risk_model: RiskModel,
    embedder: Any | None = None,
    risk_threshold: float = 0.5,
    similarity_threshold: float = 0.85,
    mask_mode: str = "blur",
    save_masked_image_to: Path | None = None,
) -> PipelineResult:
    """Run the full defense pipeline on a single image and return the safe response."""
    # 1. Region extraction
    result = vlm_client.extract_regions(
        image=image, parent_id="img_pipeline", parent_kind="image",
    )
    predicted_regions = result.regions

    # 2 & 3. Feature extraction + risk scoring
    img_ctx = build_image_context(image)
    sem_ctx = build_semantic_context(
        all_region_texts=[r.ocr_text or "" for r in predicted_regions],
        embedder=embedder,
    )

    scored: list[ScoredRegion] = []
    for r in predicted_regions:
        vf = extract_visual_features(r.bbox, img_ctx)
        sf = extract_semantic_features(r.ocr_text or "", r.region_type, sem_ctx)
        feat = np.array([
            vf.area_ratio, vf.center_distance, vf.contrast,
            vf.bg_separation, vf.saliency,
            sf.imperativeness, sf.context_relevance, sf.action_inducement,
            sf.url_qr_relevance, sf.informativeness,
        ], dtype=np.float32)
        prob = float(risk_model.predict_proba(feat.reshape(1, -1))[0])
        scored.append(ScoredRegion(
            bbox=r.bbox, label=r.region_type.value, ocr_text=r.ocr_text,
            region_type=r.region_type.value,
            risk_prob=prob, features=feat,
        ))

    # 5.1. Original description (needed regardless of masking)
    original_response = vlm_client.describe_image(image)

    # 4. Mask selection
    masks = select_masks_by_threshold(
        [(s.bbox, s.risk_prob) for s in scored], threshold=risk_threshold,
    )

    if not masks:
        # Nothing risky enough — return original description as safe response
        return PipelineResult(
            original_response=original_response,
            cfm_response=None,
            safe_response=original_response,
            regions=scored, masks=[],
        )

    # 5.2. Masked image + its description
    masked_image = apply_masks(image, masks, mode=mask_mode)  # type: ignore[arg-type]
    if save_masked_image_to is not None:
        save_masked_image_to.parent.mkdir(parents=True, exist_ok=True)
        masked_image.save(save_masked_image_to)
    cfm_response = vlm_client.describe_image(masked_image)

    # 6. Compare
    comparison = compare_responses(
        original=original_response, masked=cfm_response,
        embedder=embedder, threshold=similarity_threshold,
    )

    # 7. Return safe response
    return PipelineResult(
        original_response=original_response,
        cfm_response=cfm_response,
        safe_response=comparison.safe_response,
        regions=scored, masks=masks,
        comparison=comparison,
    )


def format_pipeline_result(r: PipelineResult, verbose: bool = False) -> str:
    lines = [
        "=== safe response ===",
        r.safe_response,
        "",
        f"regions extracted: {len(r.regions)}",
        f"regions masked:    {len(r.masks)}  (risk_threshold applied)",
    ]
    if r.comparison is not None:
        lines.extend([
            f"similarity(orig, masked): {r.comparison.similarity:.3f}",
            f"changed: {r.comparison.changed}",
        ])
    if verbose:
        lines.append("\n=== original response ===")
        lines.append(r.original_response)
        if r.cfm_response is not None:
            lines.append("\n=== cfm response ===")
            lines.append(r.cfm_response)
        if r.regions:
            top_risk = sorted(r.regions, key=lambda s: -s.risk_prob)[:5]
            lines.append("\n=== top-5 risky regions ===")
            for s in top_risk:
                lines.append(
                    f"  risk={s.risk_prob:.3f}  type={s.region_type:<20s}"
                    f"  text={(s.ocr_text or '')[:50]!r}"
                )
    return "\n".join(lines)
