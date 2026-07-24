"""Smoke tests for M3 variant generation."""
from __future__ import annotations

import io
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image, ImageDraw

from vpi_defense.schemas import (
    ConditionType,
    DatasetBundle,
    ImageMetadata,
    ImageType,
    Language,
    ModificationType,
    SourceDataset,
    SplitTag,
)
from vpi_defense.variants import (
    DEFAULT_MODIFICATIONS,
    OPERATIONS,
    generate_variants_for_image,
    make_variant,
)
from vpi_defense.variants.rendering import (
    color_for_target_contrast,
    contrast_ratio,
    load_font,
    relative_luminance,
)


def _fake_pil(w: int = 640, h: int = 480) -> Image.Image:
    img = Image.new("RGB", (w, h), color=(240, 240, 235))
    d = ImageDraw.Draw(img)
    # add some structure so bg sampling isn't trivial
    d.rectangle([40, 40, w - 40, h - 40], outline=(30, 30, 30), width=3)
    d.rectangle([100, 100, w - 100, 200], fill=(200, 220, 240))
    return img


def _fake_meta(image_id: str = "img_test_0001") -> ImageMetadata:
    return ImageMetadata(
        image_id=image_id,
        source_dataset=SourceDataset.CUSTOM,
        image_type=ImageType.DOCUMENT,
        language=Language.KO,
        image_path=f"custom/{image_id}.png",
        width=640,
        height=480,
        split=SplitTag.TRAIN,
    )


def test_font_loads():
    """If this fails, CJK font is missing on the system."""
    f = load_font(16)
    assert f is not None
    print("[OK] CJK font loaded")


def test_contrast_math_bounds():
    # black vs white → 21:1 (WCAG max)
    r = contrast_ratio((0, 0, 0), (255, 255, 255))
    assert 20.5 < r < 21.5, r
    # same color → 1:1
    assert contrast_ratio((128, 128, 128), (128, 128, 128)) == 1.0
    print(f"[OK] contrast math: black/white ratio = {r:.2f}")


def test_color_for_target_contrast_hits_target():
    bg = (240, 240, 235)  # near-white
    for target in (1.5, 3.0, 7.0):
        c = color_for_target_contrast(bg, target)
        actual = contrast_ratio(bg, c)
        # solver tolerance
        assert abs(actual - target) < 0.6, (target, actual, c)
    print("[OK] color_for_target_contrast hits within 0.6")


def test_each_operation_runs_and_stays_within_bounds():
    img = _fake_pil()
    w, h = img.size
    for mod, op in OPERATIONS.items():
        rng = random.Random(0)
        result = op(img, "테스트 문구 http://x", rng)
        assert result.image.size == (w, h)
        b = result.bbox
        assert 0.0 <= b.x_min < b.x_max <= 1.0
        assert 0.0 <= b.y_min < b.y_max <= 1.0
        print(f"[OK] op {mod.value}: bbox=({b.x_min:.2f},{b.y_min:.2f})–({b.x_max:.2f},{b.y_max:.2f})")


def test_make_variant_produces_valid_models():
    img = _fake_pil()
    meta = _fake_meta()
    gv = make_variant(
        base_image=img,
        base_meta=meta,
        modification=ModificationType.LOW_CONTRAST_OVERLAY,
        seed=42,
        variant_index=0,
    )
    assert gv.variant.base_image_id == meta.image_id
    assert gv.variant.condition_type == ConditionType.VISUAL_STEALTH
    assert gv.variant.modified_region_id == gv.injected_region.region_id
    assert gv.injected_region.parent_id == gv.variant.variant_id
    assert gv.injected_region.parent_kind == "variant"
    assert gv.injected_region.risk_labels.is_injection is True
    # image bytes are a valid PNG
    Image.open(io.BytesIO(gv.image_bytes)).verify()
    print(f"[OK] make_variant → variant={gv.variant.variant_id} region={gv.injected_region.region_id}")


def test_determinism():
    img = _fake_pil()
    meta = _fake_meta()
    gv1 = make_variant(img, meta, ModificationType.SMALL_EDGE_TEXT, seed=7, variant_index=1)
    gv2 = make_variant(img, meta, ModificationType.SMALL_EDGE_TEXT, seed=7, variant_index=1)
    assert gv1.image_bytes == gv2.image_bytes
    assert gv1.variant.model_dump() == gv2.variant.model_dump()
    print("[OK] determinism: same seed → identical output")


def test_bundle_integrates_variants():
    """The DatasetBundle validator should accept the variant + region together."""
    img = _fake_pil()
    meta = _fake_meta()
    generated = list(
        generate_variants_for_image(
            base_image=img,
            base_meta=meta,
            seed=42,
            n_per_image=len(DEFAULT_MODIFICATIONS),
        )
    )
    bundle = DatasetBundle(
        version="0.3.0",
        images=[meta],
        variants=[gv.variant for gv in generated],
        regions=[gv.injected_region for gv in generated],
    )
    assert len(bundle.variants) == len(DEFAULT_MODIFICATIONS)
    print(f"[OK] bundle integrates {len(bundle.variants)} variants")


def test_cli_end_to_end(tmp_path: Path | None = None):
    """Simulate the CLI path: write raw image + index, run generator, verify outputs."""
    from vpi_defense.ingest.base import save_items, IngestedItem
    from vpi_defense.ingest.utils import pil_to_bytes, sha256_bytes

    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())

    raw_root = tmp_path / "raw"
    out_root = tmp_path / "variants"

    # Persist a fake ingested item
    img = _fake_pil()
    data = pil_to_bytes(img)
    meta = _fake_meta()
    meta = meta.model_copy(update={"content_hash": sha256_bytes(data)})
    save_items([IngestedItem(metadata=meta, image_bytes=data)], out_root=raw_root)

    from vpi_defense.ingest.base import load_index

    for m in load_index(raw_root / "index.jsonl"):
        base = Image.open(raw_root / m.image_path).convert("RGB")
        for gv in generate_variants_for_image(base, m, seed=1, n_per_image=2):
            target = out_root / gv.variant.image_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(gv.image_bytes)
            assert target.exists() and target.stat().st_size > 0
    print(f"[OK] end-to-end variants written under {out_root}")


if __name__ == "__main__":
    test_font_loads()
    test_contrast_math_bounds()
    test_color_for_target_contrast_hits_target()
    test_each_operation_runs_and_stays_within_bounds()
    test_make_variant_produces_valid_models()
    test_determinism()
    test_bundle_integrates_variants()
    test_cli_end_to_end()
    print("\nall variant smoke tests passed.")
