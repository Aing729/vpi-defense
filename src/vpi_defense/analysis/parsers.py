"""Adapters that convert VLM responses into our RegionAnnotation model.

Two things happen here:
  1. Bbox coordinate system conversion.
     Gemini:  [y_min, x_min, y_max, x_max] in 0-1000  (Y first!)
     Ours:    BBox(x_min, y_min, x_max, y_max)         in 0-1     (X first)

  2. `RegionType` heuristic from the natural-language `kind` and `label`
     Gemini returns. Simple substring matching for now; can be replaced
     with a learned classifier later.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from vpi_defense.schemas import BBox, RegionAnnotation, RegionType


class VLMResponseError(ValueError):
    """Raised when a VLM response cannot be parsed into RegionAnnotation."""


# ---------------------------------------------------------------------------
# Bbox conversion
# ---------------------------------------------------------------------------


def gemini_bbox_to_ours(box_2d: Sequence[int | float]) -> BBox:
    """Convert Gemini `[y_min, x_min, y_max, x_max]` (0-1000) to our BBox (0-1).

    Also clamps to [0, 1] and enforces min < max so downstream schema
    validation doesn't fail on borderline coordinates.
    """
    if len(box_2d) != 4:
        raise VLMResponseError(f"box_2d must have 4 elements, got {len(box_2d)}: {box_2d}")

    y_min, x_min, y_max, x_max = (float(v) / 1000.0 for v in box_2d)

    # clamp
    x_min = max(0.0, min(1.0, x_min))
    y_min = max(0.0, min(1.0, y_min))
    x_max = max(0.0, min(1.0, x_max))
    y_max = max(0.0, min(1.0, y_max))

    # enforce strict ordering; if degenerate, nudge by 1/1000 to keep BBox valid
    if x_max <= x_min:
        x_max = min(1.0, x_min + 0.001)
    if y_max <= y_min:
        y_max = min(1.0, y_min + 0.001)

    return BBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


# ---------------------------------------------------------------------------
# RegionType inference
# ---------------------------------------------------------------------------


# Mapping from Gemini's `kind` field (see prompts.SYSTEM_INSTRUCTION_V1) to our
# RegionType enum. If Gemini returns something outside this set, we fall back
# to label-substring heuristics, then RegionType.OTHER.
_KIND_TO_REGION_TYPE: dict[str, RegionType] = {
    "main_notice":        RegionType.MAIN_NOTICE,
    "small_text":         RegionType.SMALL_TEXT,
    "qr_code":            RegionType.QR_CODE,
    "button_link":        RegionType.BUTTON_LINK,
    "background_text":    RegionType.BACKGROUND_TEXT,
    "document_margin":    RegionType.DOCUMENT_MARGIN,
    "table_caption":      RegionType.TABLE_CAPTION,
    "product_label_text": RegionType.PRODUCT_LABEL_TEXT,
    "other":              RegionType.OTHER,
}


# Substring heuristics used only when `kind` is missing/unknown. Ordered:
# more specific matches first.
_LABEL_HEURISTICS: tuple[tuple[str, RegionType], ...] = (
    ("qr code",      RegionType.QR_CODE),
    ("qr",           RegionType.QR_CODE),
    ("button",       RegionType.BUTTON_LINK),
    ("link",         RegionType.BUTTON_LINK),
    ("banner",       RegionType.BACKGROUND_TEXT),
    ("footnote",     RegionType.BACKGROUND_TEXT),
    ("footer",       RegionType.DOCUMENT_MARGIN),
    ("header",       RegionType.MAIN_NOTICE),
    ("title",        RegionType.MAIN_NOTICE),
    ("caption",      RegionType.TABLE_CAPTION),
    ("table",        RegionType.TABLE_CAPTION),
    ("small print",  RegionType.SMALL_TEXT),
    ("fine print",   RegionType.SMALL_TEXT),
    ("label",        RegionType.PRODUCT_LABEL_TEXT),
)


def infer_region_type(kind: str | None, label: str) -> RegionType:
    """Best-effort region_type inference from Gemini's response."""
    if kind:
        rt = _KIND_TO_REGION_TYPE.get(kind.strip().lower())
        if rt is not None:
            return rt

    label_low = (label or "").lower()
    for needle, rt in _LABEL_HEURISTICS:
        if needle in label_low:
            return rt
    return RegionType.OTHER


# ---------------------------------------------------------------------------
# Response envelope parsing
# ---------------------------------------------------------------------------


# Text tokens Gemini uses to indicate "no useful content" — we skip these.
_TEXT_LIKE_HINTS = (
    "text", "label", "notice", "message", "title", "header", "caption",
    "instruction", "warning", "banner", "print", "url", "link",
    # Korean cues
    "문구", "안내", "제목", "표시", "라벨", "설명",
)


def _looks_text_like(kind: str | None, label: str) -> bool:
    if kind in {"main_notice", "small_text", "background_text", "document_margin",
                "table_caption", "product_label_text", "button_link"}:
        return True
    if kind == "qr_code":
        return False  # QR is decoded, not OCR'd
    label_low = (label or "").lower()
    return any(hint in label_low for hint in _TEXT_LIKE_HINTS)


def strip_json_fence(text: str) -> str:
    """Remove a leading/trailing ```json fence if present.

    Even with `response_mime_type='application/json'`, Gemini occasionally
    emits fenced blocks. Be defensive.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # drop the opening fence
        lines = lines[1:]
        # drop trailing closing fence if any
        while lines and lines[-1].strip().startswith("```"):
            lines.pop()
        text = "\n".join(lines).strip()
    return text


@dataclass
class ParsedRegion:
    """Intermediate representation before we assign IDs and parent linkage."""
    bbox: BBox
    label: str
    kind_raw: str | None
    region_type: RegionType
    text_like: bool


def parse_gemini_response(response_text: str) -> list[ParsedRegion]:
    """Parse Gemini's JSON response into ParsedRegion records.

    Does NOT create RegionAnnotation directly because that needs a parent_id
    and a region_id which the caller supplies.

    If the raw JSON is malformed (truncated response, escape errors, etc.)
    we attempt a best-effort recovery: truncate at the last complete `}` and
    close the array. This lets us keep the well-formed regions that came
    before the corruption point instead of losing the whole call.
    """
    cleaned = strip_json_fence(response_text)
    raw = _load_json_with_recovery(cleaned)

    if not isinstance(raw, list):
        raise VLMResponseError(f"expected JSON array, got {type(raw).__name__}")

    parsed: list[ParsedRegion] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or "box_2d" not in item:
            # skip malformed entries rather than fail the whole batch
            continue
        try:
            bbox = gemini_bbox_to_ours(item["box_2d"])
        except VLMResponseError:
            continue
        label = str(item.get("label", "")).strip()
        kind_raw = item.get("kind")
        parsed.append(ParsedRegion(
            bbox=bbox,
            label=label,
            kind_raw=kind_raw,
            region_type=infer_region_type(kind_raw, label),
            text_like=_looks_text_like(kind_raw, label),
        ))
    return parsed


def _load_json_with_recovery(cleaned: str) -> object:
    """Parse JSON, or recover a partial array if the tail is corrupt."""
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Only attempt recovery if the response looks like an array.
        if not cleaned.lstrip().startswith("["):
            raise VLMResponseError(
                f"response is not valid JSON: {e}\nfirst 200 chars: {cleaned[:200]!r}"
            )
        # Truncate to the last complete top-level object and re-close the array.
        head = cleaned[: e.pos]
        last_brace = head.rfind("}")
        if last_brace == -1:
            raise VLMResponseError(
                f"response is not valid JSON and cannot be recovered: {e}\n"
                f"first 200 chars: {cleaned[:200]!r}"
            )
        recovered = head[: last_brace + 1] + "]"
        try:
            return json.loads(recovered)
        except json.JSONDecodeError as e2:
            raise VLMResponseError(
                f"response is not valid JSON (recovery also failed): {e2}\n"
                f"original error: {e}\nfirst 200 chars: {cleaned[:200]!r}"
            )


def to_region_annotations(
    parsed: list[ParsedRegion],
    parent_id: str,
    parent_kind: str,  # "image" or "variant"
    id_prefix: str,
) -> list[RegionAnnotation]:
    """Attach IDs and parent linkage to produce RegionAnnotation records.

    `id_prefix` should be a stable short slug (e.g. the base image_id
    without the 'img_' prefix). Region ids look like:
        rgn_{id_prefix}_gemini_{index:03d}
    """
    annotations: list[RegionAnnotation] = []
    for i, p in enumerate(parsed):
        annotations.append(
            RegionAnnotation(
                region_id=f"rgn_{id_prefix}_gemini_{i:03d}",
                parent_id=parent_id,
                parent_kind=parent_kind,
                bbox=p.bbox,
                region_type=p.region_type,
                text_like=p.text_like,
                ocr_text=p.label if p.text_like else None,
            )
        )
    return annotations
