"""Helpers shared by ingester implementations."""
from __future__ import annotations

import hashlib
import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


def sha256_bytes(data: bytes) -> str:
    """sha256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def make_image_id(prefix: str, index: int) -> str:
    """Build a stable image_id like 'img_docvqa_val_000042'.

    `prefix` may contain letters, digits, underscores, or hyphens.
    """
    return f"img_{prefix}_{index:06d}"


def pil_to_bytes(img: "PILImage", fmt: str = "PNG") -> bytes:
    """Serialize a PIL image to raw bytes.

    PNG is lossless (safe default). Use JPEG only when the source is already
    JPEG and disk pressure matters — re-encoding to JPEG loses info.
    """
    buf = io.BytesIO()
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    return buf.getvalue()


def coerce_pil(obj: object) -> "PILImage":
    """Best-effort coercion of an HF `Image` cell into a PIL image.

    HF `datasets` normally auto-decodes to PIL, but when `streaming=True` and
    `decode=False`, cells may arrive as {'path': ..., 'bytes': ...} dicts.
    """
    from PIL import Image  # local import: PIL is optional at import time

    if isinstance(obj, Image.Image):
        return obj
    if isinstance(obj, dict):
        if obj.get("bytes"):
            return Image.open(io.BytesIO(obj["bytes"])).copy()
        if obj.get("path"):
            return Image.open(obj["path"]).copy()
    if isinstance(obj, (bytes, bytearray)):
        return Image.open(io.BytesIO(obj)).copy()
    raise TypeError(f"cannot coerce {type(obj).__name__} to PIL image")
