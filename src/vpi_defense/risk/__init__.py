"""Risk function for the VPI-defense pipeline."""
from __future__ import annotations

from .dataset import (
    FEATURE_NAMES,
    LabeledDataset,
    LabeledSample,
    build_dataset,
    load_images,
    load_regions,
    load_variants,
)
from .evaluate import (
    CVReport,
    EvalMetrics,
    compute_metrics,
    cross_validate,
    format_coefficients,
    format_cv_report,
    format_metrics,
)
from .model import RiskModel

__all__ = [
    # dataset
    "FEATURE_NAMES", "LabeledDataset", "LabeledSample",
    "build_dataset", "load_images", "load_regions", "load_variants",
    # model
    "RiskModel",
    # evaluate
    "CVReport", "EvalMetrics",
    "compute_metrics", "cross_validate",
    "format_coefficients", "format_cv_report", "format_metrics",
]
