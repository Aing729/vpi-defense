"""Rendering helpers: font loading, color sampling, WCAG contrast math."""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageFont


# Common CJK-capable system fonts. Ordered from most-likely to least.
# On Colab and Ubuntu, `fonts-noto-cjk` puts files under
# /usr/share/fonts/opentype/noto/ or /usr/share/fonts/truetype/noto/, but the
# exact filenames vary across package versions — hence the multiple entries.
_FONT_CANDIDATES: Sequence[str] = (
    # Linux — Noto CJK (Colab, Ubuntu, Debian)
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJKkr-Regular.otf",
    # Linux — Nanum (older Ubuntu/Colab)
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/nanum/NanumGothic.otf",
    # macOS
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/AppleGothic.ttf",
    # Windows
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/gulim.ttc",
)


# Glob patterns used as a last-ditch fallback when none of the explicit
# candidates exist. If any match, we pick the first alphabetically for
# deterministic behavior.
_FONT_GLOB_FALLBACKS: Sequence[str] = (
    "/usr/share/fonts/**/NotoSansCJK*",
    "/usr/share/fonts/**/NotoSerifCJK*",
    "/usr/share/fonts/**/*Nanum*.ttf",
    "/usr/share/fonts/**/*Nanum*.otf",
    "/usr/share/fonts/**/*CJK*.ttc",
    "/usr/share/fonts/**/*CJK*.otf",
    "/usr/share/fonts/**/*korean*.ttf",  # some distros
)


class FontUnavailableError(RuntimeError):
    """Raised when no usable CJK font can be located."""


def _find_font_via_glob() -> str | None:
    """Scan /usr/share/fonts for any CJK-capable font. Returns first match."""
    for pattern in _FONT_GLOB_FALLBACKS:
        matches = sorted(glob.glob(pattern, recursive=True))
        for m in matches:
            if Path(m).is_file():
                return m
    return None


def load_font(size: int, font_path: str | None = None) -> ImageFont.FreeTypeFont:
    """Load a TrueType font that can render Korean text.

    Resolution order:
      1. explicit `font_path` argument
      2. `_FONT_CANDIDATES` in order
      3. glob-based scan of /usr/share/fonts/**
      4. raise (do NOT fall back to Pillow's default bitmap font — it does
         not render Korean and would silently corrupt variants).
    """
    if font_path:
        return ImageFont.truetype(font_path, size)
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    globbed = _find_font_via_glob()
    if globbed:
        return ImageFont.truetype(globbed, size)
    raise FontUnavailableError(
        "No CJK-capable font found. Install one of:\n"
        "  Ubuntu/Colab: apt-get install fonts-noto-cjk (then fc-cache -fv)\n"
        "  macOS:        already bundled (AppleSDGothicNeo)\n"
        "  Windows:      already bundled (Malgun Gothic)\n"
        "Or pass --font-path <ttf> explicitly. "
        f"Tried candidates: {list(_FONT_CANDIDATES)}"
    )


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------


RGB = tuple[int, int, int]


def sample_bg_color(img: Image.Image, bbox_px: tuple[int, int, int, int]) -> RGB:
    """Median RGB inside a bbox region — for BG_MIMIC_TEXT color pick.

    `bbox_px` is (x0, y0, x1, y1) in pixel coordinates.
    """
    x0, y0, x1, y1 = bbox_px
    patch = np.array(img.crop((x0, y0, x1, y1)).convert("RGB"))
    med = np.median(patch.reshape(-1, 3), axis=0)
    return int(med[0]), int(med[1]), int(med[2])


def _srgb_to_linear(c: float) -> float:
    # c is in [0, 1]
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(color: RGB) -> float:
    """WCAG relative luminance. Range: [0, 1]."""
    r, g, b = (_srgb_to_linear(c / 255.0) for c in color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(c1: RGB, c2: RGB) -> float:
    """WCAG contrast ratio in [1.0, 21.0]."""
    l1 = relative_luminance(c1)
    l2 = relative_luminance(c2)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def color_for_target_contrast(bg: RGB, target_ratio: float) -> RGB:
    """Return a gray-scale text color whose contrast with `bg` ≈ `target_ratio`.

    Uses grayscale for simplicity (works for any bg hue). Iteratively narrows
    the luminance to hit the ratio within 0.05 or after 20 steps.

    target_ratio must be > 1.0 (equal luminance means invisible text).
    """
    if target_ratio <= 1.0:
        raise ValueError("target_ratio must be > 1.0")

    lo_gray, hi_gray = 0, 255
    best: RGB = (0, 0, 0)
    for _ in range(20):
        mid = (lo_gray + hi_gray) // 2
        candidate: RGB = (mid, mid, mid)
        ratio = contrast_ratio(bg, candidate)
        if abs(ratio - target_ratio) < 0.05:
            return candidate
        # move away from bg to increase contrast, toward it to decrease
        bg_lum = relative_luminance(bg)
        cand_lum = relative_luminance(candidate)
        if ratio < target_ratio:
            # want stronger contrast → push mid away from bg luminance
            if cand_lum < bg_lum:
                hi_gray = mid - 1
            else:
                lo_gray = mid + 1
        else:
            if cand_lum < bg_lum:
                lo_gray = mid + 1
            else:
                hi_gray = mid - 1
        best = candidate
        if lo_gray > hi_gray:
            break
    return best


def near_bg_color(bg: RGB, delta: int = 8) -> RGB:
    """Return a color that's ε away from `bg` — for BG_MIMIC_TEXT.

    Perturbs each channel by ±delta, clamped to [0, 255], picking the
    direction toward the middle of the [0, 255] range to stay renderable.
    """
    def _perturb(v: int) -> int:
        return max(0, min(255, v + (delta if v < 128 else -delta)))
    return _perturb(bg[0]), _perturb(bg[1]), _perturb(bg[2])
