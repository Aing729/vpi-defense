"""Counterfactual Feature Masking (CFM).

Given a set of high-risk regions on an image, produce a masked variant
where those regions are visually obscured (Gaussian blur by default) so
that the VLM can't read them anymore. The pipeline then re-queries the
VLM on the masked image and compares its response to the original.

Two masking modes:
  - "blur"  : Gaussian blur. Natural, preserves structure. Default.
  - "solid" : Fill with a solid color. Fast, unambiguous.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from PIL import Image, ImageFilter

from vpi_defense.schemas import BBox


MaskMode = Literal["blur", "solid"]


@dataclass
class MaskSpec:
    """A region to mask, with the risk that motivated masking it.

    Kept as its own type (rather than reusing RegionAnnotation) so pipeline
    code can pass minimal info: just where to mask and how confident we are.
    """
    bbox: BBox
    risk_prob: float


def apply_masks(
    image: Image.Image,
    masks: Sequence[MaskSpec],
    mode: MaskMode = "blur",
    blur_radius: int = 25,
    solid_color: tuple[int, int, int] = (128, 128, 128),
    padding_ratio: float = 0.05,
) -> Image.Image:
    """Return a copy of `image` with each mask applied.

    Args:
        image        : source PIL image (any mode)
        masks        : bboxes to obscure (empty list → return image copy unchanged)
        mode         : "blur" or "solid"
        blur_radius  : Gaussian blur radius in pixels (blur mode)
        solid_color  : RGB fill (solid mode)
        padding_ratio: expand each bbox by this fraction on all sides before
                       masking, so we don't leave readable edges. 0.05 = 5%.

    We intentionally do NOT mutate the input.
    """
    if not masks:
        return image.copy()

    rgb = image.convert("RGB") if image.mode != "RGB" else image.copy()
    w, h = rgb.size

    if mode == "blur":
        # Pre-compute a fully-blurred copy once; we'll paste from it
        # through per-region masks. Way cheaper than blurring each region.
        blurred = rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        for spec in masks:
            x0, y0, x1, y1 = _bbox_pixels(spec.bbox, w, h, padding_ratio)
            region_mask = Image.new("L", (x1 - x0, y1 - y0), color=255)
            rgb.paste(blurred.crop((x0, y0, x1, y1)), (x0, y0), region_mask)
        return rgb

    if mode == "solid":
        from PIL import ImageDraw
        draw = ImageDraw.Draw(rgb)
        for spec in masks:
            x0, y0, x1, y1 = _bbox_pixels(spec.bbox, w, h, padding_ratio)
            draw.rectangle([x0, y0, x1, y1], fill=solid_color)
        return rgb

    raise ValueError(f"unknown mask mode: {mode!r}")


def _bbox_pixels(
    bbox: BBox, w: int, h: int, padding_ratio: float,
) -> tuple[int, int, int, int]:
    """Normalized bbox → pixel coords with optional padding, clamped to image."""
    x0f = bbox.x_min - padding_ratio * (bbox.x_max - bbox.x_min)
    y0f = bbox.y_min - padding_ratio * (bbox.y_max - bbox.y_min)
    x1f = bbox.x_max + padding_ratio * (bbox.x_max - bbox.x_min)
    y1f = bbox.y_max + padding_ratio * (bbox.y_max - bbox.y_min)

    x0 = max(0, min(w - 1, int(round(x0f * w))))
    y0 = max(0, min(h - 1, int(round(y0f * h))))
    x1 = max(x0 + 1, min(w, int(round(x1f * w))))
    y1 = max(y0 + 1, min(h, int(round(y1f * h))))
    return x0, y0, x1, y1


def select_masks_by_threshold(
    regions_with_risk: Sequence[tuple[BBox, float]],
    threshold: float = 0.5,
) -> list[MaskSpec]:
    """Convenience: pick (bbox, risk) pairs above threshold as MaskSpec."""
    return [MaskSpec(bbox=b, risk_prob=p)
            for b, p in regions_with_risk if p >= threshold]


def select_masks_top_k(
    regions_with_risk: Sequence[tuple[BBox, float]],
    k: int,
) -> list[MaskSpec]:
    """Convenience: pick the k highest-risk regions regardless of threshold."""
    sorted_pairs = sorted(regions_with_risk, key=lambda x: -x[1])[:k]
    return [MaskSpec(bbox=b, risk_prob=p) for b, p in sorted_pairs]
