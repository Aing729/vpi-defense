"""IoU-based matching of ground-truth injection regions to predicted regions.

Used to compute training labels for the risk function:
  - matched   : GT injection paired with a prediction (positive learning sample)
  - missed    : GT injection with no matching prediction (Gemini didn't see it)
  - extra     : predictions that don't match any GT (candidate negatives —
                usually normal image content the annotator labels non-injection)

Matching is greedy: for each GT (in fixed order) pick the highest-IoU
unmatched prediction above `iou_threshold`. Hungarian assignment would be
optimal but our per-image N is small (<50) so greedy is fine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from vpi_defense.schemas import BBox, RegionAnnotation


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------


def bbox_iou(a: BBox, b: BBox) -> float:
    """Intersection over Union of two normalized bboxes. Returns 0.0 if disjoint."""
    x_left = max(a.x_min, b.x_min)
    y_top = max(a.y_min, b.y_min)
    x_right = min(a.x_max, b.x_max)
    y_bottom = min(a.y_max, b.y_max)
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    a_area = (a.x_max - a.x_min) * (a.y_max - a.y_min)
    b_area = (b.x_max - b.x_min) * (b.y_max - b.y_min)
    union = a_area + b_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def bbox_area(b: BBox) -> float:
    return (b.x_max - b.x_min) * (b.y_max - b.y_min)


# ---------------------------------------------------------------------------
# Matching result types
# ---------------------------------------------------------------------------


@dataclass
class Match:
    """A GT-injection paired with the prediction that best covers it."""
    gt: RegionAnnotation
    pred: RegionAnnotation
    iou: float


@dataclass
class MatchReport:
    """Result of matching one image (or variant)'s GTs to predictions."""
    parent_id: str
    matched: list[Match]              # positive learning samples
    missed_gt: list[RegionAnnotation]  # GTs Gemini failed to see
    extra_pred: list[RegionAnnotation] # predictions not covering any GT

    @property
    def recall(self) -> float:
        n_gt = len(self.matched) + len(self.missed_gt)
        return len(self.matched) / n_gt if n_gt else 0.0

    @property
    def n_gt(self) -> int:
        return len(self.matched) + len(self.missed_gt)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def match_regions(
    ground_truth: list[RegionAnnotation],
    predictions: list[RegionAnnotation],
    parent_id: str,
    iou_threshold: float = 0.3,
) -> MatchReport:
    """Greedy match GT injections to predictions by IoU.

    Args:
        ground_truth : injection regions from the variant generator
                       (`RiskLabels.is_injection = True`)
        predictions  : all predicted regions for the same image
        iou_threshold: minimum IoU to consider a match

    Every GT is matched to at most one prediction. Every prediction is
    matched to at most one GT. Predictions with IoU < threshold against
    every GT go to `extra_pred`.
    """
    used_pred_idx: set[int] = set()
    matched: list[Match] = []
    missed: list[RegionAnnotation] = []

    for gt in ground_truth:
        best_i = -1
        best_iou = 0.0
        for i, pred in enumerate(predictions):
            if i in used_pred_idx:
                continue
            iou = bbox_iou(gt.bbox, pred.bbox)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_i = i
        if best_i == -1:
            missed.append(gt)
        else:
            used_pred_idx.add(best_i)
            matched.append(Match(gt=gt, pred=predictions[best_i], iou=best_iou))

    extra = [p for i, p in enumerate(predictions) if i not in used_pred_idx]

    return MatchReport(
        parent_id=parent_id,
        matched=matched,
        missed_gt=missed,
        extra_pred=extra,
    )


# ---------------------------------------------------------------------------
# Bulk match: group by parent_id, run per-image
# ---------------------------------------------------------------------------


def _group_by_parent(regions: Iterable[RegionAnnotation]) -> dict[str, list[RegionAnnotation]]:
    grouped: dict[str, list[RegionAnnotation]] = {}
    for r in regions:
        grouped.setdefault(r.parent_id, []).append(r)
    return grouped


def match_all(
    ground_truth: Iterable[RegionAnnotation],
    predictions: Iterable[RegionAnnotation],
    iou_threshold: float = 0.3,
) -> list[MatchReport]:
    """Group by parent_id and match per image. Only parents present in both
    are matched; a GT-only or pred-only parent is skipped silently. If you
    need to know which parents were skipped, do it upstream.
    """
    gts_by = _group_by_parent(ground_truth)
    preds_by = _group_by_parent(predictions)

    reports: list[MatchReport] = []
    for parent_id in sorted(gts_by.keys() & preds_by.keys()):
        reports.append(match_regions(
            ground_truth=gts_by[parent_id],
            predictions=preds_by[parent_id],
            parent_id=parent_id,
            iou_threshold=iou_threshold,
        ))
    return reports


def summarize(reports: list[MatchReport]) -> dict[str, float | int]:
    """Aggregate matching stats across many reports."""
    total_matched = sum(len(r.matched) for r in reports)
    total_missed = sum(len(r.missed_gt) for r in reports)
    total_extra = sum(len(r.extra_pred) for r in reports)
    n_gt = total_matched + total_missed
    return {
        "n_parents": len(reports),
        "n_gt_total": n_gt,
        "n_matched": total_matched,
        "n_missed": total_missed,
        "n_extra_pred": total_extra,
        "recall": total_matched / n_gt if n_gt else 0.0,
    }
