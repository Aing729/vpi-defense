"""Visual feature extraction for a region.

Produces a `VisualFeatures` record: five deterministic image-based signals
that feed the risk function.

Design notes
------------
- All features are in [0, 1] so they can be fed directly to logistic
  regression without additional scaling. This matches the schema.
- Saliency uses spectral residual (Hou & Zhang 2007) implemented in plain
  numpy to avoid the `opencv-contrib-python` dependency. The map is computed
  once per image on a 512x512 downsample and re-sampled per bbox.
- `bg_separation` uses Lab color space because Euclidean distance in Lab is
  a better proxy for perceived color difference than raw sRGB.

Typical call pattern (per image, many bboxes):

    ctx = build_image_context(pil_image)
    for region in regions:
        features = extract_visual_features(region.bbox, ctx)
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from PIL import Image

from vpi_defense.schemas import BBox, VisualFeatures


# Max distance from image center (corners are furthest). Used to normalize.
_MAX_CENTER_DIST = sqrt(0.5**2 + 0.5**2)  # ≈ 0.7071


# ---------------------------------------------------------------------------
# Image context (cached per-image data)
# ---------------------------------------------------------------------------


@dataclass
class ImageContext:
    """Per-image data reused across multiple bbox feature extractions."""
    rgb: np.ndarray        # (H, W, 3), uint8
    lab: np.ndarray        # (H, W, 3), float32 in Lab
    luminance: np.ndarray  # (H, W), float32 in [0, 1]
    saliency: np.ndarray   # (H, W), float32 in [0, 1]
    width: int
    height: int


def build_image_context(image: Image.Image) -> ImageContext:
    """Precompute per-image arrays used by feature extractors."""
    rgb = np.array(image.convert("RGB"))                          # uint8
    lab = _rgb_to_lab(rgb.astype(np.float32) / 255.0)             # float32
    luminance = _rgb_to_luminance(rgb.astype(np.float32) / 255.0)  # float32
    saliency = spectral_residual_saliency(luminance)              # float32 in [0,1]
    h, w = rgb.shape[:2]
    return ImageContext(
        rgb=rgb, lab=lab, luminance=luminance, saliency=saliency,
        width=w, height=h,
    )


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------


def extract_visual_features(bbox: BBox, ctx: ImageContext) -> VisualFeatures:
    """Compute all 5 visual features for one bbox on a given image context."""
    x0, y0, x1, y1 = _bbox_to_pixels(bbox, ctx.width, ctx.height)
    if x1 <= x0 or y1 <= y0:
        # degenerate bbox — return zeros rather than crash
        return VisualFeatures(area_ratio=0.0, center_distance=1.0,
                              contrast=0.0, bg_separation=0.0, saliency=0.0)

    area_ratio = _area_ratio(bbox)
    center_distance = _center_distance(bbox)
    contrast = _rms_contrast(ctx.luminance[y0:y1, x0:x1])
    bg_separation = _bg_separation(ctx.lab, x0, y0, x1, y1)
    saliency = _region_saliency(ctx.saliency, x0, y0, x1, y1)

    return VisualFeatures(
        area_ratio=area_ratio,
        center_distance=center_distance,
        contrast=contrast,
        bg_separation=bg_separation,
        saliency=saliency,
    )


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------


def _bbox_to_pixels(b: BBox, w: int, h: int) -> tuple[int, int, int, int]:
    x0 = int(round(b.x_min * w))
    y0 = int(round(b.y_min * h))
    x1 = int(round(b.x_max * w))
    y1 = int(round(b.y_max * h))
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(x0 + 1, min(w, x1))
    y1 = max(y0 + 1, min(h, y1))
    return x0, y0, x1, y1


def _area_ratio(b: BBox) -> float:
    """bbox area / image area. Already in [0, 1] since bbox is normalized."""
    return max(0.0, min(1.0, (b.x_max - b.x_min) * (b.y_max - b.y_min)))


def _center_distance(b: BBox) -> float:
    """Normalized distance from image center to bbox center."""
    cx = (b.x_min + b.x_max) / 2
    cy = (b.y_min + b.y_max) / 2
    d = sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
    return max(0.0, min(1.0, d / _MAX_CENTER_DIST))


def _rms_contrast(patch_luminance: np.ndarray) -> float:
    """RMS contrast of a luminance patch. Multiplied by 2 to spread the
    typical natural-image range (0..0.3) into [0, 1] but clamped so we
    don't exceed 1.
    """
    if patch_luminance.size == 0:
        return 0.0
    std = float(patch_luminance.std())
    return max(0.0, min(1.0, std * 2.0))


def _bg_separation(lab: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    """Perceived color distance between region and its surrounding ring.

    The surrounding ring is the bbox expanded by 20% on each side, minus
    the inner region. Distance is euclidean in Lab space, normalized by
    a rough perceptual max (100.0) into [0, 1].
    """
    h, w = lab.shape[:2]
    pad_x = int(round((x1 - x0) * 0.2))
    pad_y = int(round((y1 - y0) * 0.2))
    ox0 = max(0, x0 - pad_x)
    oy0 = max(0, y0 - pad_y)
    ox1 = min(w, x1 + pad_x)
    oy1 = min(h, y1 + pad_y)

    inner = lab[y0:y1, x0:x1]
    if inner.size == 0:
        return 0.0
    inner_mean = inner.reshape(-1, 3).mean(axis=0)

    outer = lab[oy0:oy1, ox0:ox1]
    # Build a ring mask: outer minus inner region
    mask = np.ones(outer.shape[:2], dtype=bool)
    iy0, iy1 = y0 - oy0, y1 - oy0
    ix0, ix1 = x0 - ox0, x1 - ox0
    mask[iy0:iy1, ix0:ix1] = False
    ring_pixels = outer[mask]
    if ring_pixels.size == 0:
        # bbox touches the image edge; fall back to whole outer
        ring_pixels = outer.reshape(-1, 3)
    ring_mean = ring_pixels.mean(axis=0)

    dist = float(np.linalg.norm(inner_mean - ring_mean))
    # Empirical normalization: ~100.0 covers most natural cases
    return max(0.0, min(1.0, dist / 100.0))


def _region_saliency(saliency_map: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    """Mean saliency inside the bbox. Map is already in [0, 1]."""
    patch = saliency_map[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float(patch.mean())


# ---------------------------------------------------------------------------
# Spectral residual saliency (Hou & Zhang 2007), pure numpy
# ---------------------------------------------------------------------------


def spectral_residual_saliency(
    luminance: np.ndarray,
    target_size: int = 256,
    smoothing_kernel: int = 3,
    post_blur_sigma: float = 3.0,
) -> np.ndarray:
    """Compute a spectral-residual saliency map.

    Steps:
      1. Downsample luminance to `target_size` on the short axis (speed).
      2. FFT → log-amplitude + phase.
      3. Spectral residual = log-amplitude minus its low-pass filtered version.
      4. Inverse FFT of (residual + phase) → squared magnitude.
      5. Gaussian blur → normalize to [0, 1].
      6. Upsample back to original size.
    """
    from PIL import Image as PILImage

    h, w = luminance.shape
    # Downsample
    scale = target_size / min(h, w)
    tw = max(2, int(round(w * scale)))
    th = max(2, int(round(h * scale)))
    small = np.array(PILImage.fromarray((luminance * 255).astype(np.uint8))
                     .resize((tw, th), PILImage.BILINEAR)) / 255.0

    # FFT
    f = np.fft.fft2(small)
    amp = np.abs(f)
    phase = np.angle(f)
    log_amp = np.log(amp + 1e-8)

    # Low-pass filter log-amplitude by a box kernel
    k = smoothing_kernel
    smoothed = _box_filter(log_amp, k)

    # Spectral residual
    residual = log_amp - smoothed

    # Reconstruct
    reconstructed = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2

    # Post-blur for a smoother saliency map
    blurred = _gaussian_blur(reconstructed, sigma=post_blur_sigma)

    # Normalize to [0, 1]
    mx = blurred.max()
    normalized = blurred / mx if mx > 0 else blurred

    # Upsample to original size
    up = np.array(PILImage.fromarray((normalized * 255).astype(np.uint8))
                  .resize((w, h), PILImage.BILINEAR)) / 255.0
    return up.astype(np.float32)


def _box_filter(x: np.ndarray, k: int) -> np.ndarray:
    """Simple k×k box filter via convolution (numpy only)."""
    if k <= 1:
        return x
    kernel = np.ones((k, k), dtype=x.dtype) / (k * k)
    # Pad with edge reflection to keep shape
    pad = k // 2
    padded = np.pad(x, pad, mode="edge")
    out = np.zeros_like(x)
    for dy in range(k):
        for dx in range(k):
            out += padded[dy:dy + x.shape[0], dx:dx + x.shape[1]] * kernel[dy, dx]
    return out


def _gaussian_blur(x: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur, pure numpy."""
    if sigma <= 0:
        return x
    radius = int(round(sigma * 3))
    xs = np.arange(-radius, radius + 1)
    kernel_1d = np.exp(-(xs**2) / (2 * sigma * sigma))
    kernel_1d /= kernel_1d.sum()

    # horizontal pass
    padded = np.pad(x, ((0, 0), (radius, radius)), mode="edge")
    out = np.zeros_like(x)
    for i, k in enumerate(kernel_1d):
        out += padded[:, i:i + x.shape[1]] * k
    # vertical pass
    padded = np.pad(out, ((radius, radius), (0, 0)), mode="edge")
    out2 = np.zeros_like(x)
    for i, k in enumerate(kernel_1d):
        out2 += padded[i:i + x.shape[0], :] * k
    return out2


# ---------------------------------------------------------------------------
# Color space conversions (sRGB → luminance / Lab), pure numpy
# ---------------------------------------------------------------------------


def _rgb_to_luminance(rgb01: np.ndarray) -> np.ndarray:
    """Perceived luminance (Rec. 709). Input rgb in [0, 1]."""
    return (0.2126 * rgb01[..., 0]
            + 0.7152 * rgb01[..., 1]
            + 0.0722 * rgb01[..., 2]).astype(np.float32)


def _rgb_to_lab(rgb01: np.ndarray) -> np.ndarray:
    """sRGB [0,1] → CIELAB (approximate D65). Returns float32 (H, W, 3).

    Standard sRGB → linear → XYZ (D65) → Lab pipeline.
    """
    # 1. sRGB → linear
    linear = np.where(rgb01 <= 0.04045,
                      rgb01 / 12.92,
                      ((rgb01 + 0.055) / 1.055) ** 2.4)

    # 2. Linear RGB → XYZ (D65)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz = linear @ M.T

    # 3. Normalize by D65 white point
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    xyz[..., 0] /= xn
    xyz[..., 1] /= yn
    xyz[..., 2] /= zn

    # 4. f(t) function
    delta = 6 / 29
    def f(t):
        return np.where(t > delta ** 3,
                        np.cbrt(t),
                        t / (3 * delta * delta) + 4 / 29)
    fx, fy, fz = f(xyz[..., 0]), f(xyz[..., 1]), f(xyz[..., 2])

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return np.stack([L, a, b], axis=-1).astype(np.float32)
