"""Analysis pipeline: VLM-based region extraction and downstream stages."""
from __future__ import annotations

from .parsers import (
    ParsedRegion,
    VLMResponseError,
    gemini_bbox_to_ours,
    infer_region_type,
    parse_gemini_response,
    to_region_annotations,
)
from .vlm_client import (
    BaseVLMClient,
    CLAUDE_REGION_TOOL_SCHEMA,
    ClaudeVLMClient,
    GeminiVLMClient,
    MockVLMClient,
    VLMCallResult,
    canned_pharmacy_response,
)

__all__ = [
    "BaseVLMClient",
    "CLAUDE_REGION_TOOL_SCHEMA",
    "ClaudeVLMClient",
    "GeminiVLMClient",
    "MockVLMClient",
    "VLMCallResult",
    "ParsedRegion",
    "VLMResponseError",
    "canned_pharmacy_response",
    "gemini_bbox_to_ours",
    "infer_region_type",
    "parse_gemini_response",
    "to_region_annotations",
]
