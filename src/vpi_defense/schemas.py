"""
Dataset schemas for the VPI-defense pipeline.

Three tiers mirror the report's dataset design:
  - ImageMetadata      : per-image record (source, type, language, ...)
  - RegionAnnotation   : per-region record (bbox, region_type, features, labels)
  - VariantSample      : per-variant record (base image + injected condition)

These models are the single source of truth. Label Studio exports, dataset
loaders, and the analysis pipeline all convert to/from these types.

Conventions
-----------
* IDs are prefixed to make cross-references unambiguous:
    - `img_*` for base images
    - `var_*` for variants
    - `rgn_*` for regions
* A region's `parent_id` may reference EITHER an image or a variant.
  The `parent_kind` field disambiguates this (and is validated against
  the prefix, so mismatch fails at load time).
* Train/val/test split is stored ONLY on ImageMetadata.
  Variants MUST inherit the split of their base image; the dataset loader
  is responsible for enforcing this. Do NOT put a `split` field on
  VariantSample — that invites data leakage.
* bbox coordinates are normalized to [0, 1] against image width/height.
* All feature vectors (VisualFeatures, SemanticFeatures) use [0, 1] ranges
  so they can be fed directly into the logistic risk function without
  additional scaling.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SourceDataset(str, Enum):
    VIZWIZ = "vizwiz"
    DOCVQA = "docvqa"
    DOCLAYNET = "doclaynet"
    RICO = "rico"
    TEXTOCR = "textocr"
    AIHUB_KO = "aihub_ko"
    CUSTOM = "custom"  # self-captured / synthesized


class ImageType(str, Enum):
    NATURAL_SCENE = "natural_scene"
    DOCUMENT = "document"
    MOBILE_UI = "mobile_ui"
    POSTER_WITH_QR = "poster_with_qr"
    WEB_CAPTURE = "web_capture"
    PRODUCT_LABEL = "product_label"
    SIGNAGE = "signage"
    OTHER = "other"


class Language(str, Enum):
    KO = "ko"
    EN = "en"
    MIXED = "mixed"
    NONE = "none"  # image without any text


class RegionType(str, Enum):
    MAIN_NOTICE = "main_notice"
    SMALL_TEXT = "small_text"
    QR_CODE = "qr_code"
    BUTTON_LINK = "button_link"
    BACKGROUND_TEXT = "background_text"
    DOCUMENT_MARGIN = "document_margin"
    TABLE_CAPTION = "table_caption"
    PRODUCT_LABEL_TEXT = "product_label_text"
    OTHER = "other"


class ConditionType(str, Enum):
    """Category of injection condition applied at the variant level."""
    NONE = "none"  # clean/normal variant
    VISUAL_STEALTH = "visual_stealth"       # small size, low contrast, edge/margin placement
    SEMANTIC_MISMATCH = "semantic_mismatch"  # sentence unrelated to image context
    ACTION_INDUCEMENT = "action_inducement"  # QR-adjacent, button-like, link-like
    LAYOUT_DISRUPTION = "layout_disruption"  # footnote/table/banner mimicry


class ModificationType(str, Enum):
    """Concrete transformation applied to the base image to build a variant."""
    NONE = "none"
    LOW_CONTRAST_OVERLAY = "low_contrast_overlay"
    SMALL_EDGE_TEXT = "small_edge_text"
    BG_MIMIC_TEXT = "bg_mimic_text"      # text colored like the background
    QR_ADJACENT_TEXT = "qr_adjacent_text"
    FAKE_BUTTON = "fake_button"
    UNRELATED_INSERTION = "unrelated_insertion"
    FOOTNOTE_INJECTION = "footnote_injection"
    BANNER_INJECTION = "banner_injection"


class SplitTag(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


# ---------------------------------------------------------------------------
# Feature / label sub-models
# ---------------------------------------------------------------------------


class VisualFeatures(BaseModel):
    """Deterministic visual features of a region. All values in [0, 1] unless noted."""
    model_config = ConfigDict(extra="forbid")

    area_ratio: float = Field(..., ge=0.0, le=1.0, description="region area / image area")
    center_distance: float = Field(..., ge=0.0, le=1.0, description="normalized distance from image center")
    contrast: float = Field(..., ge=0.0, le=1.0, description="Michelson or RMS contrast, normalized")
    bg_separation: float = Field(..., ge=0.0, le=1.0, description="how visually separated from background")
    saliency: float = Field(..., ge=0.0, le=1.0, description="spectral residual saliency mean over region")


class SemanticFeatures(BaseModel):
    """Semantic features. Populated during labeling or by an LLM-assisted pre-pass."""
    model_config = ConfigDict(extra="forbid")

    imperativeness: float = Field(..., ge=0.0, le=1.0, description="I: instruction-like sentence score")
    context_relevance: float = Field(..., ge=0.0, le=1.0, description="how well the text aligns with image context")
    action_inducement: float = Field(..., ge=0.0, le=1.0, description="degree it urges user action")
    url_qr_relevance: float = Field(..., ge=0.0, le=1.0, description="U: association with URL/QR")
    informativeness: float = Field(..., ge=0.0, le=1.0, description="normal information value to the user")


class RiskLabels(BaseModel):
    """Ground-truth labels for supervised learning of the risk function."""
    model_config = ConfigDict(extra="forbid")

    is_injection: bool = Field(..., description="human-labeled: is this region a VPI candidate?")
    is_normal_info: bool = Field(..., description="does this region carry information the user should keep?")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="annotator confidence")
    notes: str | None = None


# ---------------------------------------------------------------------------
# Tier 1: Image
# ---------------------------------------------------------------------------


class ImageMetadata(BaseModel):
    """Top-level record for a single source image (before any variant is applied)."""
    model_config = ConfigDict(extra="forbid")

    image_id: str = Field(..., pattern=r"^img_[a-zA-Z0-9_-]+$")
    source_dataset: SourceDataset
    image_type: ImageType
    language: Language
    user_context: str | None = Field(None, description="short human-written context, e.g. 'pharmacy label photo'")
    image_path: str = Field(..., description="path relative to data/raw/")
    original_caption: str | None = None

    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)

    content_hash: str | None = Field(
        None,
        pattern=r"^[a-f0-9]{64}$",
        description="sha256 of the image file bytes; used for dedup and drift detection",
    )
    split: SplitTag = SplitTag.TRAIN


# ---------------------------------------------------------------------------
# Tier 2: Region
# ---------------------------------------------------------------------------


class BBox(BaseModel):
    """Normalized bounding box in [0, 1] coordinates."""
    model_config = ConfigDict(extra="forbid")

    x_min: float = Field(..., ge=0.0, le=1.0)
    y_min: float = Field(..., ge=0.0, le=1.0)
    x_max: float = Field(..., ge=0.0, le=1.0)
    y_max: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_order(self) -> "BBox":
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError(f"invalid bbox: {self}")
        return self


class RegionAnnotation(BaseModel):
    """A single annotated region within a base image or a variant."""
    model_config = ConfigDict(extra="forbid")

    region_id: str = Field(..., pattern=r"^rgn_[a-zA-Z0-9_-]+$")
    parent_id: str = Field(..., description="image_id or variant_id this region belongs to")
    parent_kind: Literal["image", "variant"] = Field(
        ..., description="disambiguates parent_id; must be consistent with the id prefix"
    )
    bbox: BBox
    mask_path: str | None = Field(None, description="optional path to a binary mask PNG")
    region_type: RegionType
    text_like: bool = Field(
        ...,
        description=(
            "True if OCR could plausibly extract characters from this region. "
            "Note: QR_CODE is NOT text_like (its content is decoded, not OCR'd)."
        ),
    )
    ocr_text: str | None = Field(None, description="extracted or transcribed text if any")

    visual_features: VisualFeatures | None = None
    semantic_features: SemanticFeatures | None = None
    risk_labels: RiskLabels | None = None

    @model_validator(mode="after")
    def _check_parent_prefix(self) -> "RegionAnnotation":
        expected_prefix = "img_" if self.parent_kind == "image" else "var_"
        if not self.parent_id.startswith(expected_prefix):
            raise ValueError(
                f"region {self.region_id}: parent_kind={self.parent_kind} "
                f"but parent_id={self.parent_id!r} does not start with {expected_prefix!r}"
            )
        return self


# ---------------------------------------------------------------------------
# Tier 3: Variant
# ---------------------------------------------------------------------------


class VariantSample(BaseModel):
    """A synthesized image derived from a base image by applying an injection condition."""
    model_config = ConfigDict(extra="forbid")

    variant_id: str = Field(..., pattern=r"^var_[a-zA-Z0-9_-]+$")
    base_image_id: str = Field(..., pattern=r"^img_[a-zA-Z0-9_-]+$")
    condition_type: ConditionType
    modification_type: ModificationType
    modified_region_id: str | None = Field(
        None,
        description="region_id of the injected region (null if condition_type is NONE)",
    )
    image_path: str = Field(..., description="path to the synthesized image, relative to data/variants/")

    expected_effect: Literal["benign", "reduces_safety", "misleads_action"] = "benign"
    generation_params: dict[str, float | int | str | bool] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dataset bundle
# ---------------------------------------------------------------------------


class DatasetBundle(BaseModel):
    """A serializable snapshot of the dataset for versioning / export."""
    model_config = ConfigDict(extra="forbid")

    version: str
    images: list[ImageMetadata]
    variants: list[VariantSample]
    regions: list[RegionAnnotation]

    @model_validator(mode="after")
    def _check_references(self) -> "DatasetBundle":
        image_ids = {img.image_id for img in self.images}
        variant_ids = {var.variant_id for var in self.variants}
        valid_parents = image_ids | variant_ids

        for var in self.variants:
            if var.base_image_id not in image_ids:
                raise ValueError(f"variant {var.variant_id} references unknown base_image_id {var.base_image_id}")

        for rgn in self.regions:
            pool = image_ids if rgn.parent_kind == "image" else variant_ids
            if rgn.parent_id not in pool:
                raise ValueError(
                    f"region {rgn.region_id} references unknown "
                    f"{rgn.parent_kind}={rgn.parent_id}"
                )

        _ = valid_parents  # kept for future cross-tier lookups
        return self
