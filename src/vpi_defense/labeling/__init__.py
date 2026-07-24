"""Label Studio adapter for the VPI-defense pipeline."""
from __future__ import annotations

from pathlib import Path

from .adapter import (
    LabelStudioParseError,
    ls_bbox_to_ours,
    parse_export,
    parse_task,
    region_type_from_label,
)


def config_xml_path() -> Path:
    """Absolute path to the bundled Label Studio config.xml."""
    return Path(__file__).parent / "config.xml"


__all__ = [
    "LabelStudioParseError",
    "config_xml_path",
    "ls_bbox_to_ours",
    "parse_export",
    "parse_task",
    "region_type_from_label",
]
