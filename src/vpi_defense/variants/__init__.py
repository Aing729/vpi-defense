"""Variant generation for the VPI-defense pipeline."""
from __future__ import annotations

from .generator import (
    DEFAULT_MODIFICATIONS,
    GeneratedVariant,
    generate_variants_for_image,
    make_variant,
)
from .operations import OPERATIONS, OperationResult

__all__ = [
    "DEFAULT_MODIFICATIONS",
    "GeneratedVariant",
    "OPERATIONS",
    "OperationResult",
    "generate_variants_for_image",
    "make_variant",
]
