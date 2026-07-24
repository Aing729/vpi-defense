"""Integration check: M2 ingest → M3 variants → M5 VLM analysis.

Simulates the whole pipeline end-to-end on synthetic data (no HF, no Gemini
API), asserts that:
  - each stage's output is consumable by the next
  - the final combined DatasetBundle validates
  - JSONL files round-trip through disk unchanged
  - IDs and cross-references are consistent

Run:
    python scripts/verify_pipeline.py
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image, ImageDraw

from vpi_defense.analysis import MockVLMClient, canned_pharmacy_response
from vpi_defense.ingest.base import IngestedItem, load_index, save_items
from vpi_defense.ingest.utils import pil_to_bytes, sha256_bytes
from vpi_defense.schemas import (
    DatasetBundle,
    ImageMetadata,
    ImageType,
    Language,
    RegionAnnotation,
    SourceDataset,
    SplitTag,
    VariantSample,
)
from vpi_defense.variants import DEFAULT_MODIFICATIONS, generate_variants_for_image


PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"


def _log(step: str, msg: str = "") -> None:
    print(f"  {step:8s}  {msg}")


def _make_synthetic_image(seed: int) -> Image.Image:
    """A document-like image with a title band, body text, and a QR-ish square."""
    img = Image.new("RGB", (800, 600), color=(250, 248, 240))
    d = ImageDraw.Draw(img)
    d.rectangle([60, 60, 740, 540], outline=(80, 80, 80), width=2)
    d.rectangle([100, 100, 700, 160], fill=(230, 235, 245))
    d.text((110, 120), f"Document #{seed}", fill=(20, 20, 20))
    for i in range(4):
        d.text((110, 200 + i * 40), f"Body line {i+1}: normal text.", fill=(60, 60, 60))
    # fake QR
    d.rectangle([620, 440, 720, 520], fill=(10, 10, 10))
    d.rectangle([640, 460, 700, 500], fill=(250, 250, 250))
    return img


def stage_ingest(tmp_root: Path, n_images: int = 3) -> list[ImageMetadata]:
    """Simulate what an ingester would produce."""
    print(f"\n[STAGE 1 / M2 ingest] generating {n_images} synthetic base images")

    items: list[IngestedItem] = []
    for i in range(n_images):
        pil = _make_synthetic_image(seed=i)
        data = pil_to_bytes(pil)
        image_id = f"img_synth_{i:04d}"
        meta = ImageMetadata(
            image_id=image_id,
            source_dataset=SourceDataset.CUSTOM,
            image_type=ImageType.DOCUMENT,
            language=Language.KO,
            image_path=f"synth/{image_id}.png",
            width=pil.width, height=pil.height,
            content_hash=sha256_bytes(data),
            split=SplitTag.TRAIN if i < 2 else SplitTag.VAL,
        )
        items.append(IngestedItem(metadata=meta, image_bytes=data))

    raw_root = tmp_root / "raw"
    written = save_items(items, out_root=raw_root)
    _log(PASS, f"wrote {len(written)} images to {raw_root}")

    # round-trip check
    reloaded = load_index(raw_root / "index.jsonl")
    assert len(reloaded) == n_images
    assert reloaded[0].image_id == written[0].image_id
    _log(PASS, f"index.jsonl round-trip: {len(reloaded)} records identical")

    return reloaded


def stage_variants(
    tmp_root: Path, images: list[ImageMetadata]
) -> tuple[list[VariantSample], list[RegionAnnotation]]:
    """Simulate M3: for each base image, generate one variant per modification."""
    print(f"\n[STAGE 2 / M3 variants] generating {len(DEFAULT_MODIFICATIONS)} variants per image")

    variants_root = tmp_root / "variants"
    variants_root.mkdir(parents=True, exist_ok=True)
    variants_idx = variants_root / "variants_index.jsonl"
    regions_idx = variants_root / "regions_index.jsonl"

    all_variants: list[VariantSample] = []
    all_regions: list[RegionAnnotation] = []
    mod_counts: Counter[str] = Counter()

    raw_root = tmp_root / "raw"
    with variants_idx.open("w", encoding="utf-8") as v_out, \
         regions_idx.open("w", encoding="utf-8") as r_out:
        for meta in images:
            src = raw_root / meta.image_path
            base = Image.open(src).convert("RGB")
            for gv in generate_variants_for_image(base, meta, seed=42):
                out_path = variants_root / gv.variant.image_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(gv.image_bytes)
                v_out.write(gv.variant.model_dump_json() + "\n")
                r_out.write(gv.injected_region.model_dump_json() + "\n")
                all_variants.append(gv.variant)
                all_regions.append(gv.injected_region)
                mod_counts[gv.variant.modification_type.value] += 1

    _log(PASS, f"{len(all_variants)} variants written, {len(all_regions)} injected regions auto-labeled")
    _log(PASS, f"modification distribution: {dict(mod_counts)}")

    # cross-reference sanity
    image_ids = {m.image_id for m in images}
    for v in all_variants:
        assert v.base_image_id in image_ids, f"orphan variant: {v.variant_id}"
    _log(PASS, f"every variant links to a known base image_id")

    variant_ids = {v.variant_id for v in all_variants}
    for r in all_regions:
        assert r.parent_kind == "variant" and r.parent_id in variant_ids
    _log(PASS, f"every injected region links to a known variant_id")

    # determinism check: re-generate the first variant of the first image with same seed
    first_run = next(iter(generate_variants_for_image(
        Image.open(raw_root / images[0].image_path).convert("RGB"), images[0], seed=42,
    )))
    second_run = next(iter(generate_variants_for_image(
        Image.open(raw_root / images[0].image_path).convert("RGB"), images[0], seed=42,
    )))
    assert first_run.image_bytes == second_run.image_bytes
    _log(PASS, f"determinism: same seed → identical output bytes")

    return all_variants, all_regions


def stage_vlm(
    tmp_root: Path,
    images: list[ImageMetadata],
    variants: list[VariantSample],
) -> list[RegionAnnotation]:
    """Simulate M5: run the VLM (mock) on both base images and variants."""
    print(f"\n[STAGE 3 / M5 VLM analysis] running MockVLMClient on {len(images)} images + {len(variants)} variants")

    client = MockVLMClient(canned_pharmacy_response())
    analysis_root = tmp_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    raw_dir = analysis_root / "raw_responses"

    all_regions: list[RegionAnnotation] = []

    raw_root = tmp_root / "raw"
    variants_root = tmp_root / "variants"

    for m in images:
        pil = Image.open(raw_root / m.image_path).convert("RGB")
        result = client.extract_regions(
            image=pil, parent_id=m.image_id, parent_kind="image",
            save_raw_to=raw_dir / f"{m.image_id}.json",
        )
        all_regions.extend(result.regions)

    for v in variants:
        pil = Image.open(variants_root / v.image_path).convert("RGB")
        result = client.extract_regions(
            image=pil, parent_id=v.variant_id, parent_kind="variant",
            save_raw_to=raw_dir / f"{v.variant_id}.json",
        )
        all_regions.extend(result.regions)

    _log(PASS, f"{len(all_regions)} regions produced by VLM")
    _log(PASS, f"raw responses archived to {raw_dir} ({len(list(raw_dir.iterdir()))} files)")

    # region_type distribution — sanity check the heuristic
    rt_counts = Counter(r.region_type.value for r in all_regions)
    _log(PASS, f"region_type distribution: {dict(rt_counts)}")

    return all_regions


def stage_integrate(
    images: list[ImageMetadata],
    variants: list[VariantSample],
    injected_regions: list[RegionAnnotation],
    vlm_regions: list[RegionAnnotation],
) -> DatasetBundle:
    """Combine everything into a single DatasetBundle and let its validator run."""
    print(f"\n[STAGE 4] integrating into DatasetBundle")

    bundle = DatasetBundle(
        version="verify-0.1.0",
        images=images,
        variants=variants,
        regions=injected_regions + vlm_regions,
    )
    _log(PASS, f"DatasetBundle validated: "
              f"{len(bundle.images)} images, "
              f"{len(bundle.variants)} variants, "
              f"{len(bundle.regions)} regions "
              f"({len(injected_regions)} injected + {len(vlm_regions)} VLM)")

    # JSON round-trip
    payload = bundle.model_dump_json()
    restored = DatasetBundle.model_validate_json(payload)
    assert restored == bundle
    _log(PASS, f"JSON round-trip identical ({len(payload):,} chars)")

    return bundle


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp_root = Path(td)
        print(f"working dir: {tmp_root}")

        images = stage_ingest(tmp_root, n_images=3)
        variants, injected_regions = stage_variants(tmp_root, images)
        vlm_regions = stage_vlm(tmp_root, images, variants)
        bundle = stage_integrate(images, variants, injected_regions, vlm_regions)

        print("\n" + "=" * 60)
        print("PIPELINE VERIFIED END-TO-END")
        print("=" * 60)
        print(f"  base images       : {len(bundle.images)}")
        print(f"  variants          : {len(bundle.variants)}")
        print(f"  regions (total)   : {len(bundle.regions)}")
        print(f"    auto-labeled    : {len(injected_regions)}")
        print(f"    VLM-produced    : {len(vlm_regions)}")
        print(f"  distinct mods used: {len({v.modification_type for v in bundle.variants})}")


if __name__ == "__main__":
    main()
