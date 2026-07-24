"""Smoke tests for M6.1 matching + M6.2 visual features."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image, ImageDraw

from vpi_defense.features import (
    bbox_iou,
    build_image_context,
    extract_visual_features,
    match_all,
    match_regions,
    spectral_residual_saliency,
    summarize,
)
from vpi_defense.schemas import (
    BBox,
    RegionAnnotation,
    RegionType,
    RiskLabels,
)


def _region(rid: str, parent: str, bbox: BBox, is_injection: bool = False,
            parent_kind: str = "variant") -> RegionAnnotation:
    return RegionAnnotation(
        region_id=rid,
        parent_id=parent,
        parent_kind=parent_kind,
        bbox=bbox,
        region_type=RegionType.SMALL_TEXT,
        text_like=True,
        risk_labels=(RiskLabels(is_injection=is_injection, is_normal_info=False)
                     if is_injection else None),
    )


# ---------------------------------------------------------------------------
# IoU math
# ---------------------------------------------------------------------------


def test_iou_identical_bbox_is_1():
    b = BBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)
    assert abs(bbox_iou(b, b) - 1.0) < 1e-6
    print("[OK] IoU(b, b) == 1")


def test_iou_disjoint_bbox_is_0():
    a = BBox(x_min=0.0, y_min=0.0, x_max=0.2, y_max=0.2)
    b = BBox(x_min=0.5, y_min=0.5, x_max=0.7, y_max=0.7)
    assert bbox_iou(a, b) == 0.0
    print("[OK] IoU of disjoint bboxes == 0")


def test_iou_half_overlap():
    """Two 0.5x0.5 boxes, shifted by 0.25 → union 3/4 of both, intersection 1/4.
    IoU should be intersection / union = 1/7."""
    a = BBox(x_min=0.0, y_min=0.0, x_max=0.5, y_max=0.5)
    b = BBox(x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75)
    iou = bbox_iou(a, b)
    # intersection = 0.25*0.25 = 0.0625
    # union = 0.25 + 0.25 - 0.0625 = 0.4375
    expected = 0.0625 / 0.4375
    assert abs(iou - expected) < 1e-6, (iou, expected)
    print(f"[OK] IoU 0.5×0.5 boxes offset by 0.25 → {iou:.4f}")


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def test_match_perfect_1to1():
    gt = [
        _region("rgn_gt1", "var_a", BBox(x_min=0.1, y_min=0.1, x_max=0.3, y_max=0.3),
                is_injection=True),
        _region("rgn_gt2", "var_a", BBox(x_min=0.7, y_min=0.7, x_max=0.9, y_max=0.9),
                is_injection=True),
    ]
    pred = [
        # exact matches, in reverse order — greedy should still pair correctly
        _region("rgn_p1", "var_a", BBox(x_min=0.7, y_min=0.7, x_max=0.9, y_max=0.9)),
        _region("rgn_p2", "var_a", BBox(x_min=0.1, y_min=0.1, x_max=0.3, y_max=0.3)),
    ]
    report = match_regions(gt, pred, parent_id="var_a")
    assert len(report.matched) == 2
    assert len(report.missed_gt) == 0
    assert len(report.extra_pred) == 0
    assert all(m.iou > 0.99 for m in report.matched)
    print(f"[OK] perfect matching: {len(report.matched)}/2, recall={report.recall:.2f}")


def test_match_missed_and_extra():
    gt = [
        _region("rgn_gt1", "var_a", BBox(x_min=0.05, y_min=0.05, x_max=0.15, y_max=0.15),
                is_injection=True),
        _region("rgn_gt2", "var_a", BBox(x_min=0.5, y_min=0.5, x_max=0.6, y_max=0.6),
                is_injection=True),
    ]
    pred = [
        # matches gt1 well
        _region("rgn_p1", "var_a", BBox(x_min=0.05, y_min=0.05, x_max=0.15, y_max=0.15)),
        # extra, doesn't match any GT
        _region("rgn_p2", "var_a", BBox(x_min=0.8, y_min=0.8, x_max=0.9, y_max=0.9)),
    ]
    report = match_regions(gt, pred, parent_id="var_a")
    assert len(report.matched) == 1
    assert len(report.missed_gt) == 1  # gt2 has no matching prediction
    assert len(report.extra_pred) == 1
    assert report.recall == 0.5
    print(f"[OK] partial match: {len(report.matched)} matched, "
          f"{len(report.missed_gt)} missed, {len(report.extra_pred)} extra")


def test_match_all_groups_by_parent():
    gt = [
        _region("rgn_gt1", "var_a", BBox(x_min=0.1, y_min=0.1, x_max=0.2, y_max=0.2),
                is_injection=True),
        _region("rgn_gt2", "var_b", BBox(x_min=0.5, y_min=0.5, x_max=0.6, y_max=0.6),
                is_injection=True),
    ]
    pred = [
        _region("rgn_p1", "var_a", BBox(x_min=0.1, y_min=0.1, x_max=0.2, y_max=0.2)),
        _region("rgn_p2", "var_b", BBox(x_min=0.5, y_min=0.5, x_max=0.6, y_max=0.6)),
        _region("rgn_p3", "var_c", BBox(x_min=0.0, y_min=0.0, x_max=0.1, y_max=0.1)),
    ]
    reports = match_all(gt, pred)
    # only var_a and var_b overlap between gt and pred; var_c is pred-only, skipped
    assert len(reports) == 2
    assert {r.parent_id for r in reports} == {"var_a", "var_b"}
    stats = summarize(reports)
    assert stats["n_matched"] == 2
    assert stats["recall"] == 1.0
    print(f"[OK] match_all: {stats}")


# ---------------------------------------------------------------------------
# Visual features
# ---------------------------------------------------------------------------


def _document_image() -> Image.Image:
    """A synthetic document image with a title band and body text — the kind
    of image DocVQA/DocLayNet contain."""
    img = Image.new("RGB", (800, 600), color=(248, 246, 240))
    d = ImageDraw.Draw(img)
    d.rectangle([60, 60, 740, 540], outline=(80, 80, 80), width=2)
    d.rectangle([100, 100, 700, 160], fill=(220, 230, 245))
    d.text((110, 118), "DOCUMENT TITLE", fill=(20, 20, 40))
    for i in range(6):
        d.text((110, 220 + i * 40), f"Body line {i+1}: dark text.", fill=(60, 60, 60))
    return img


def test_visual_features_are_in_range():
    img = _document_image()
    ctx = build_image_context(img)
    bbox = BBox(x_min=0.10, y_min=0.15, x_max=0.90, y_max=0.30)  # covers title band
    features = extract_visual_features(bbox, ctx)
    for name in ("area_ratio", "center_distance", "contrast", "bg_separation", "saliency"):
        val = getattr(features, name)
        assert 0.0 <= val <= 1.0, f"{name} out of range: {val}"
    print(f"[OK] visual features in [0,1]: {features}")


def test_area_ratio_matches_bbox_area():
    img = _document_image()
    ctx = build_image_context(img)
    # 0.5x0.4 bbox has area 0.2
    bbox = BBox(x_min=0.25, y_min=0.30, x_max=0.75, y_max=0.70)
    f = extract_visual_features(bbox, ctx)
    assert abs(f.area_ratio - 0.20) < 1e-4, f.area_ratio
    print(f"[OK] area_ratio: 0.5*0.4 bbox → {f.area_ratio:.4f}")


def test_center_distance_corner_vs_center():
    img = _document_image()
    ctx = build_image_context(img)
    center_bbox = BBox(x_min=0.45, y_min=0.45, x_max=0.55, y_max=0.55)
    corner_bbox = BBox(x_min=0.0, y_min=0.0, x_max=0.05, y_max=0.05)
    fc = extract_visual_features(center_bbox, ctx).center_distance
    fk = extract_visual_features(corner_bbox, ctx).center_distance
    assert fc < 0.1, fc
    assert fk > 0.9, fk
    print(f"[OK] center_distance: center={fc:.3f}, corner={fk:.3f}")


def test_contrast_higher_on_text_than_background():
    """Contrast should be higher on a text-rich region than a plain background."""
    img = _document_image()
    ctx = build_image_context(img)
    text_bbox = BBox(x_min=0.15, y_min=0.37, x_max=0.85, y_max=0.55)   # body text lines
    # Plain patch strictly inside the page interior, below the last body line
    # (which is at y ≈ 0.77) and above the bottom border (at y ≈ 0.9).
    plain_bbox = BBox(x_min=0.30, y_min=0.80, x_max=0.55, y_max=0.86)
    c_text = extract_visual_features(text_bbox, ctx).contrast
    c_plain = extract_visual_features(plain_bbox, ctx).contrast
    assert c_text > c_plain, (c_text, c_plain)
    print(f"[OK] contrast: text={c_text:.3f} > plain={c_plain:.3f}")


def test_bg_separation_higher_on_colored_band_than_body_text():
    """The blue title band should have higher bg_separation from the surrounding
    beige page than a plain text region does."""
    img = _document_image()
    ctx = build_image_context(img)
    title_bbox = BBox(x_min=0.15, y_min=0.18, x_max=0.85, y_max=0.26)  # blue band
    body_bbox = BBox(x_min=0.15, y_min=0.40, x_max=0.85, y_max=0.44)   # body text
    s_title = extract_visual_features(title_bbox, ctx).bg_separation
    s_body = extract_visual_features(body_bbox, ctx).bg_separation
    assert s_title > s_body, (s_title, s_body)
    print(f"[OK] bg_separation: colored band ({s_title:.3f}) > body text ({s_body:.3f})")


def test_saliency_map_shape_and_range():
    img = _document_image()
    ctx = build_image_context(img)
    assert ctx.saliency.shape == (600, 800)
    assert 0.0 <= ctx.saliency.min() and ctx.saliency.max() <= 1.0
    print(f"[OK] saliency map: shape={ctx.saliency.shape}, "
          f"range=[{ctx.saliency.min():.3f}, {ctx.saliency.max():.3f}]")


def test_degenerate_bbox_does_not_crash():
    img = _document_image()
    ctx = build_image_context(img)
    # near-zero area bbox
    bbox = BBox(x_min=0.5, y_min=0.5, x_max=0.5001, y_max=0.5001)
    f = extract_visual_features(bbox, ctx)
    # just needs to not raise, and stay in [0, 1]
    for name in ("area_ratio", "center_distance", "contrast", "bg_separation", "saliency"):
        assert 0.0 <= getattr(f, name) <= 1.0
    print("[OK] degenerate bbox handled")


if __name__ == "__main__":
    test_iou_identical_bbox_is_1()
    test_iou_disjoint_bbox_is_0()
    test_iou_half_overlap()
    test_match_perfect_1to1()
    test_match_missed_and_extra()
    test_match_all_groups_by_parent()
    test_visual_features_are_in_range()
    test_area_ratio_matches_bbox_area()
    test_center_distance_corner_vs_center()
    test_contrast_higher_on_text_than_background()
    test_bg_separation_higher_on_colored_band_than_body_text()
    test_saliency_map_shape_and_range()
    test_degenerate_bbox_does_not_crash()
    print("\nall features smoke tests passed.")
