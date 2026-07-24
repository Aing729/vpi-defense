"""Adapter: Label Studio export JSON -> RegionAnnotation records.

Label Studio groups per-region results by an internal `id`. A single bbox
produces one `rectanglelabels` result and 0..N companion results
(`choices`, `textarea`, `rating`) that share the same id. We regroup
those into a single RegionAnnotation.

Coordinate conversion:
    LS uses percentage (0..100) with (x, y) = top-left corner + width, height.
    Ours uses normalized (0..1) with (x_min, y_min, x_max, y_max).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from vpi_defense.schemas import BBox, RegionAnnotation, RegionType, RiskLabels


class LabelStudioParseError(ValueError):
    """Raised when an LS export record can't be turned into RegionAnnotation."""


# ---------------------------------------------------------------------------
# Coordinate + type conversion
# ---------------------------------------------------------------------------


def ls_bbox_to_ours(x: float, y: float, width: float, height: float) -> BBox:
    """(x, y, w, h) in 0..100 percent -> normalized BBox in 0..1."""
    x_min = max(0.0, min(1.0, x / 100.0))
    y_min = max(0.0, min(1.0, y / 100.0))
    x_max = max(0.0, min(1.0, (x + width) / 100.0))
    y_max = max(0.0, min(1.0, (y + height) / 100.0))
    if x_max <= x_min:
        x_max = min(1.0, x_min + 0.001)
    if y_max <= y_min:
        y_max = min(1.0, y_min + 0.001)
    return BBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def region_type_from_label(label: str) -> RegionType:
    """Map the RectangleLabels choice string to our RegionType enum."""
    try:
        return RegionType(label)
    except ValueError:
        return RegionType.OTHER


def _rating_to_confidence(rating: int | float | None) -> float:
    """Rating in 1..5 -> confidence in 0.2..1.0. None -> 1.0 (default)."""
    if rating is None:
        return 1.0
    r = max(1.0, min(5.0, float(rating)))
    return r / 5.0


# ---------------------------------------------------------------------------
# Per-region grouping
# ---------------------------------------------------------------------------


@dataclass
class _RegionAcc:
    """Mutable accumulator built up from the several per-region result rows."""
    bbox: BBox | None = None
    region_type: RegionType = RegionType.OTHER
    text_like: bool = False
    ocr_text: str | None = None
    risk_choices: list[str] = field(default_factory=list)
    confidence: float = 1.0
    notes: str | None = None


def _apply_result(acc: _RegionAcc, result: dict[str, Any]) -> None:
    """Merge one LS `result` row into a region accumulator."""
    r_type = result.get("type")
    value = result.get("value") or {}

    if r_type == "rectanglelabels":
        labels = value.get("rectanglelabels") or []
        chosen = labels[0] if labels else "other"
        acc.bbox = ls_bbox_to_ours(
            x=float(value.get("x", 0)),
            y=float(value.get("y", 0)),
            width=float(value.get("width", 0)),
            height=float(value.get("height", 0)),
        )
        acc.region_type = region_type_from_label(chosen)

    elif r_type == "choices":
        from_name = result.get("from_name")
        choices = value.get("choices") or []
        if from_name == "text_like":
            acc.text_like = "text_like" in choices
        elif from_name == "risk":
            acc.risk_choices = list(choices)

    elif r_type == "textarea":
        from_name = result.get("from_name")
        texts = value.get("text") or []
        joined = "\n".join(t for t in texts if t)
        if from_name == "ocr_text":
            acc.ocr_text = joined or None
        elif from_name == "notes":
            acc.notes = joined or None

    elif r_type == "rating":
        acc.confidence = _rating_to_confidence(value.get("rating"))


def _finalize(
    acc: _RegionAcc, region_id: str, parent_id: str, parent_kind: str,
) -> RegionAnnotation:
    if acc.bbox is None:
        raise LabelStudioParseError(f"region {region_id} has no bbox")

    # Only attach RiskLabels if the annotator touched any of the risk fields.
    labels: RiskLabels | None = None
    if acc.risk_choices or acc.notes is not None:
        labels = RiskLabels(
            is_injection="is_injection" in acc.risk_choices,
            is_normal_info="is_normal_info" in acc.risk_choices,
            confidence=acc.confidence,
            notes=acc.notes,
        )

    return RegionAnnotation(
        region_id=region_id,
        parent_id=parent_id,
        parent_kind=parent_kind,  # type: ignore[arg-type]
        bbox=acc.bbox,
        region_type=acc.region_type,
        text_like=acc.text_like,
        ocr_text=acc.ocr_text,
        risk_labels=labels,
    )


# ---------------------------------------------------------------------------
# Task-level parsing
# ---------------------------------------------------------------------------


def _task_ids(task: dict[str, Any]) -> tuple[str, str, str]:
    """Extract (parent_id, parent_kind, id_prefix) from a task's `data` block.

    Convention: our variants CLI writes parent_id (img_* or var_*) and
    parent_kind ("image"|"variant") into task data before uploading.
    Fallback: derive parent_kind from the id prefix.
    """
    data = task.get("data") or {}
    parent_id = data.get("parent_id") or data.get("image_id") or data.get("variant_id")
    if not parent_id:
        raise LabelStudioParseError("task has no parent_id/image_id/variant_id in data")

    parent_kind = data.get("parent_kind")
    if not parent_kind:
        parent_kind = "image" if parent_id.startswith("img_") else "variant"

    id_prefix = parent_id.removeprefix("img_").removeprefix("var_")
    return parent_id, parent_kind, id_prefix


def parse_task(task: dict[str, Any]) -> list[RegionAnnotation]:
    """One task -> list[RegionAnnotation]. Uses the FIRST completed annotation."""
    parent_id, parent_kind, id_prefix = _task_ids(task)

    annotations = task.get("annotations") or []
    if not annotations:
        return []

    # Prefer the last non-skipped annotation; falls back to the first.
    ann = next((a for a in reversed(annotations) if not a.get("was_cancelled")),
               annotations[0])
    results: Sequence[dict[str, Any]] = ann.get("result") or []

    # Group results by their `id` field (LS assigns one id per bbox region)
    grouped: dict[str, _RegionAcc] = {}
    for r in results:
        rid = r.get("id")
        if not rid:
            continue
        acc = grouped.setdefault(rid, _RegionAcc())
        _apply_result(acc, r)

    regions: list[RegionAnnotation] = []
    for i, (ls_id, acc) in enumerate(grouped.items()):
        if acc.bbox is None:
            # a stray choices/textarea with no matching rectanglelabels — skip
            continue
        region_id = f"rgn_{id_prefix}_ls_{i:03d}"
        regions.append(_finalize(acc, region_id, parent_id, parent_kind))
    return regions


def parse_export(path: Path) -> Iterator[tuple[str, list[RegionAnnotation]]]:
    """Iterate over all tasks in an LS JSON export.

    Yields (parent_id, regions) pairs so the caller can decide how to write.
    Accepts both single-object and array-of-objects JSON.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    tasks = payload if isinstance(payload, list) else [payload]
    for task in tasks:
        try:
            parent_id, _, _ = _task_ids(task)
        except LabelStudioParseError:
            continue
        yield parent_id, parse_task(task)
