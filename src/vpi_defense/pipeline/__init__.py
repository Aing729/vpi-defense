"""End-to-end pipeline: image → safe natural-language response."""
from __future__ import annotations

from .cfm import (
    MaskMode,
    MaskSpec,
    apply_masks,
    select_masks_by_threshold,
    select_masks_top_k,
)
from .compare import ComparisonResult, compare_responses
from .metrics import BatchMetrics, MetricPair, compute_batch_metrics, format_batch_metrics
from .serve import (
    PipelineResult,
    ScoredRegion,
    format_pipeline_result,
    run_pipeline,
)

__all__ = [
    "MaskMode", "MaskSpec",
    "apply_masks", "select_masks_by_threshold", "select_masks_top_k",
    "ComparisonResult", "compare_responses",
    "BatchMetrics", "MetricPair",
    "compute_batch_metrics", "format_batch_metrics",
    "PipelineResult", "ScoredRegion",
    "run_pipeline", "format_pipeline_result",
]
