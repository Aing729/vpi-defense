"""Top-level variant generator.

For each base image, apply one modification, produce:
  - the variant image bytes
  - a VariantSample record
  - a RegionAnnotation for the injected region (with `is_injection=True`
    pre-populated — this is auto-labeling that saves annotator time)

The RegionAnnotation for the injected region is auto-generated because we
*know* where we placed the injection. The human annotator only needs to
label the *other* regions of the variant during M4.
"""
from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from PIL import Image

from vpi_defense.ingest.base import IngestedItem
from vpi_defense.ingest.utils import pil_to_bytes, sha256_bytes
from vpi_defense.schemas import (
    ConditionType,
    ImageMetadata,
    ModificationType,
    RegionAnnotation,
    RegionType,
    RiskLabels,
    SourceDataset,
    VariantSample,
)

from . import operations, prompts


# ---------------------------------------------------------------------------
# Static mapping: ModificationType → ConditionType.
# Kept explicit rather than inferred so a reader can see the taxonomy.
# ---------------------------------------------------------------------------

MODIFICATION_TO_CONDITION: dict[ModificationType, ConditionType] = {
    ModificationType.LOW_CONTRAST_OVERLAY: ConditionType.VISUAL_STEALTH,
    ModificationType.SMALL_EDGE_TEXT:      ConditionType.VISUAL_STEALTH,
    ModificationType.BG_MIMIC_TEXT:        ConditionType.VISUAL_STEALTH,
    ModificationType.UNRELATED_INSERTION:  ConditionType.SEMANTIC_MISMATCH,
}

DEFAULT_MODIFICATIONS: tuple[ModificationType, ...] = tuple(MODIFICATION_TO_CONDITION.keys())


@dataclass
class GeneratedVariant:
    """Everything the generator emits per variant."""
    image_bytes: bytes
    variant: VariantSample
    injected_region: RegionAnnotation


def _short_slug(mod: ModificationType) -> str:
    return {
        ModificationType.LOW_CONTRAST_OVERLAY: "lc",
        ModificationType.SMALL_EDGE_TEXT:      "edge",
        ModificationType.BG_MIMIC_TEXT:        "bgm",
        ModificationType.UNRELATED_INSERTION:  "unrel",
    }[mod]


def _language_for(base_meta: ImageMetadata) -> str:
    """Pick a prompt language. Korean-first project, so KO is the default.
    Falls back to EN only when the source image is definitely EN and KO
    prompts on it would be surprising (e.g. an English-only receipt).
    """
    # Simple heuristic: if the base image language is EN only, use EN.
    if base_meta.language.value == "en":
        return "en"
    return "ko"


def make_variant(
    base_image: Image.Image,
    base_meta: ImageMetadata,
    modification: ModificationType,
    seed: int,
    variant_index: int = 0,
    font_path: str | None = None,
) -> GeneratedVariant:
    """Produce a single variant of `base_image` under `modification`.

    `seed` and `variant_index` together decide the RNG state — passing the
    same pair twice produces the same variant.
    """
    rng = random.Random(f"{seed}:{variant_index}:{modification.value}")
    language = _language_for(base_meta)

    text = prompts.sample_prompt(modification, rng, language=language)

    op = operations.OPERATIONS[modification]
    result = op(base_image, text, rng, font_path=font_path)

    # Serialize the variant image
    image_bytes = pil_to_bytes(result.image, fmt="PNG")
    content_hash = sha256_bytes(image_bytes)

    # IDs — derive from base image id so lineage is visible at a glance
    base_stem = base_meta.image_id.removeprefix("img_")
    variant_id = f"var_{base_stem}_{_short_slug(modification)}_{variant_index:02d}"
    region_id = f"rgn_{base_stem}_{_short_slug(modification)}_{variant_index:02d}_inj"

    condition = MODIFICATION_TO_CONDITION[modification]

    variant = VariantSample(
        variant_id=variant_id,
        base_image_id=base_meta.image_id,
        condition_type=condition,
        modification_type=modification,
        modified_region_id=region_id,
        image_path=f"variants/{variant_id}.png",
        expected_effect="reduces_safety",
        generation_params={
            "seed": seed,
            "variant_index": variant_index,
            "language": language,
            "content_hash": content_hash,
            **{k: v for k, v in result.params.items() if isinstance(v, (int, float, str, bool))},
        },
    )

    injected_region = RegionAnnotation(
        region_id=region_id,
        parent_id=variant_id,
        parent_kind="variant",
        bbox=result.bbox,
        region_type=RegionType.SMALL_TEXT,
        text_like=True,
        ocr_text=result.injected_text,
        # visual/semantic features are computed downstream (M6); we know the
        # ground-truth risk label though, so seed it here.
        risk_labels=RiskLabels(
            is_injection=True,
            is_normal_info=False,
            confidence=1.0,
            notes="auto-labeled by variant generator",
        ),
    )

    return GeneratedVariant(image_bytes=image_bytes, variant=variant, injected_region=injected_region)


def generate_variants_for_image(
    base_image: Image.Image,
    base_meta: ImageMetadata,
    modifications: Sequence[ModificationType] = DEFAULT_MODIFICATIONS,
    seed: int = 42,
    n_per_image: int | None = None,
    font_path: str | None = None,
) -> Iterator[GeneratedVariant]:
    """Yield variants for one base image.

    If `n_per_image` is None, yield exactly one variant per modification in
    `modifications`. Otherwise cycle through modifications up to n_per_image.
    """
    if n_per_image is None:
        target = list(modifications)
    else:
        target = [modifications[i % len(modifications)] for i in range(n_per_image)]

    for i, mod in enumerate(target):
        yield make_variant(
            base_image=base_image,
            base_meta=base_meta,
            modification=mod,
            seed=seed,
            variant_index=i,
            font_path=font_path,
        )
