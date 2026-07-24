"""Feature extraction for the risk function."""
from __future__ import annotations

from .matching import (
    Match,
    MatchReport,
    bbox_area,
    bbox_iou,
    match_all,
    match_regions,
    summarize,
)
from .semantic import (
    SemanticContext,
    build_semantic_context,
    extract_semantic_features,
    try_load_default_embedder,
)
from .visual import (
    ImageContext,
    build_image_context,
    extract_visual_features,
    spectral_residual_saliency,
)

__all__ = [
    # matching
    "Match", "MatchReport",
    "bbox_iou", "bbox_area",
    "match_regions", "match_all", "summarize",
    # visual
    "ImageContext", "build_image_context",
    "extract_visual_features", "spectral_residual_saliency",
    # semantic
    "SemanticContext", "build_semantic_context",
    "extract_semantic_features", "try_load_default_embedder",
]
