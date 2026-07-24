"""Concrete variant modification operations.

Each operation takes:
  - base image (PIL RGB)
  - injection text
  - RNG (random.Random) for deterministic sampling of position/font size/etc.
  - operation-specific parameters
and returns an OperationResult with:
  - modified image
  - normalized bbox of the injected region
  - dict of generation_params (recorded on VariantSample)

Coordinates: internally we work in pixels, but the OperationResult exposes
normalized bbox in [0, 1] to match schemas.BBox.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from PIL import Image, ImageDraw, ImageFont

from vpi_defense.schemas import BBox

from . import rendering


@dataclass
class OperationResult:
    image: Image.Image
    injected_text: str
    bbox: BBox              # normalized to [0, 1]
    params: dict[str, object] = field(default_factory=dict)


def _text_size(font: ImageFont.FreeTypeFont, text: str) -> tuple[int, int]:
    """Portable text size (Pillow removed `textsize` in 10.x)."""
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _bbox_norm(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> BBox:
    """Clamp + normalize a pixel bbox against image dimensions."""
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(x0 + 1, min(w, x1))
    y1 = max(y0 + 1, min(h, y1))
    return BBox(x_min=x0 / w, y_min=y0 / h, x_max=x1 / w, y_max=y1 / h)


# ---------------------------------------------------------------------------
# 1. LOW_CONTRAST_OVERLAY
# ---------------------------------------------------------------------------


def low_contrast_overlay(
    base: Image.Image,
    text: str,
    rng: random.Random,
    contrast_ratio_target: float = 1.5,
    font_path: str | None = None,
    font_size_ratio: float = 0.025,
) -> OperationResult:
    """Insert `text` at a random inner location with a target contrast ratio.

    `contrast_ratio_target=1.5` is well below WCAG minimum of 4.5 for body
    text — the human eye barely sees it, but VLMs read it fine.
    """
    img = base.copy().convert("RGB")
    w, h = img.size

    font_size = max(10, int(min(w, h) * font_size_ratio))
    font = rendering.load_font(font_size, font_path)
    tw, th = _text_size(font, text)

    # random position in the middle 60% of the image
    x0 = rng.randint(int(w * 0.2), max(int(w * 0.2) + 1, int(w * 0.8) - tw))
    y0 = rng.randint(int(h * 0.2), max(int(h * 0.2) + 1, int(h * 0.8) - th))

    # sample local bg color and pick a text color at the target contrast ratio
    bg = rendering.sample_bg_color(img, (x0, y0, x0 + tw, y0 + th))
    text_color = rendering.color_for_target_contrast(bg, contrast_ratio_target)

    draw = ImageDraw.Draw(img)
    draw.text((x0, y0), text, font=font, fill=text_color)

    return OperationResult(
        image=img,
        injected_text=text,
        bbox=_bbox_norm(x0, y0, x0 + tw, y0 + th, w, h),
        params={
            "font_size": font_size,
            "target_contrast_ratio": contrast_ratio_target,
            "sampled_bg_rgb": list(bg),
            "text_rgb": list(text_color),
        },
    )


# ---------------------------------------------------------------------------
# 2. SMALL_EDGE_TEXT
# ---------------------------------------------------------------------------


def small_edge_text(
    base: Image.Image,
    text: str,
    rng: random.Random,
    edge: str | None = None,
    font_path: str | None = None,
    font_size_ratio: float = 0.014,
    margin_ratio: float = 0.008,
) -> OperationResult:
    """Insert small text hugging one of the four edges.

    `edge` in {"top", "bottom", "left", "right"} — if None, chosen at random.
    Font size defaults to ~1.4% of the shorter dimension (deliberately tiny).
    """
    img = base.copy().convert("RGB")
    w, h = img.size

    font_size = max(8, int(min(w, h) * font_size_ratio))
    font = rendering.load_font(font_size, font_path)
    tw, th = _text_size(font, text)

    edge = edge or rng.choice(("top", "bottom", "left", "right"))
    margin = max(2, int(min(w, h) * margin_ratio))

    if edge == "top":
        x0 = rng.randint(margin, max(margin + 1, w - tw - margin))
        y0 = margin
    elif edge == "bottom":
        x0 = rng.randint(margin, max(margin + 1, w - tw - margin))
        y0 = h - th - margin
    elif edge == "left":
        x0 = margin
        y0 = rng.randint(margin, max(margin + 1, h - th - margin))
    else:  # right
        x0 = w - tw - margin
        y0 = rng.randint(margin, max(margin + 1, h - th - margin))

    bg = rendering.sample_bg_color(img, (x0, y0, x0 + tw, y0 + th))
    # readable text but visually easy to miss (small size does the hiding)
    text_color = rendering.color_for_target_contrast(bg, 4.5)

    draw = ImageDraw.Draw(img)
    draw.text((x0, y0), text, font=font, fill=text_color)

    return OperationResult(
        image=img,
        injected_text=text,
        bbox=_bbox_norm(x0, y0, x0 + tw, y0 + th, w, h),
        params={
            "font_size": font_size,
            "edge": edge,
            "margin_px": margin,
            "text_rgb": list(text_color),
        },
    )


# ---------------------------------------------------------------------------
# 3. BG_MIMIC_TEXT
# ---------------------------------------------------------------------------


def bg_mimic_text(
    base: Image.Image,
    text: str,
    rng: random.Random,
    font_path: str | None = None,
    font_size_ratio: float = 0.022,
    color_delta: int = 6,
) -> OperationResult:
    """Insert text colored to blend with the local background (Δ per channel)."""
    img = base.copy().convert("RGB")
    w, h = img.size

    font_size = max(10, int(min(w, h) * font_size_ratio))
    font = rendering.load_font(font_size, font_path)
    tw, th = _text_size(font, text)

    x0 = rng.randint(int(w * 0.15), max(int(w * 0.15) + 1, int(w * 0.85) - tw))
    y0 = rng.randint(int(h * 0.15), max(int(h * 0.15) + 1, int(h * 0.85) - th))

    bg = rendering.sample_bg_color(img, (x0, y0, x0 + tw, y0 + th))
    text_color = rendering.near_bg_color(bg, delta=color_delta)

    draw = ImageDraw.Draw(img)
    draw.text((x0, y0), text, font=font, fill=text_color)

    return OperationResult(
        image=img,
        injected_text=text,
        bbox=_bbox_norm(x0, y0, x0 + tw, y0 + th, w, h),
        params={
            "font_size": font_size,
            "sampled_bg_rgb": list(bg),
            "text_rgb": list(text_color),
            "color_delta": color_delta,
        },
    )


# ---------------------------------------------------------------------------
# 4. UNRELATED_INSERTION
# ---------------------------------------------------------------------------


def unrelated_insertion(
    base: Image.Image,
    text: str,
    rng: random.Random,
    font_path: str | None = None,
    font_size_ratio: float = 0.028,
) -> OperationResult:
    """Insert unrelated text at a fully readable size/contrast — the payload
    is the *content* (semantic mismatch), not visual stealth.
    """
    img = base.copy().convert("RGB")
    w, h = img.size

    font_size = max(12, int(min(w, h) * font_size_ratio))
    font = rendering.load_font(font_size, font_path)
    tw, th = _text_size(font, text)

    x0 = rng.randint(int(w * 0.1), max(int(w * 0.1) + 1, int(w * 0.9) - tw))
    y0 = rng.randint(int(h * 0.1), max(int(h * 0.1) + 1, int(h * 0.9) - th))

    bg = rendering.sample_bg_color(img, (x0, y0, x0 + tw, y0 + th))
    text_color = rendering.color_for_target_contrast(bg, 7.0)

    draw = ImageDraw.Draw(img)
    draw.text((x0, y0), text, font=font, fill=text_color)

    return OperationResult(
        image=img,
        injected_text=text,
        bbox=_bbox_norm(x0, y0, x0 + tw, y0 + th, w, h),
        params={"font_size": font_size, "text_rgb": list(text_color)},
    )


# ---------------------------------------------------------------------------
# Registry so the generator can look up ops by ModificationType
# ---------------------------------------------------------------------------

from vpi_defense.schemas import ModificationType

Operation = Callable[..., OperationResult]

OPERATIONS: dict[ModificationType, Operation] = {
    ModificationType.LOW_CONTRAST_OVERLAY: low_contrast_overlay,
    ModificationType.SMALL_EDGE_TEXT:      small_edge_text,
    ModificationType.BG_MIMIC_TEXT:        bg_mimic_text,
    ModificationType.UNRELATED_INSERTION:  unrelated_insertion,
}
