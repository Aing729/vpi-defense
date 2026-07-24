"""Dataset ingesters."""
from __future__ import annotations

from .base import BaseIngester, IngestedItem, load_index, save_items
from .doclaynet import DocLayNetIngester
from .docvqa import DocVQAIngester

INGESTERS: dict[str, type[BaseIngester]] = {
    "docvqa": DocVQAIngester,
    "doclaynet": DocLayNetIngester,
}

__all__ = [
    "BaseIngester",
    "IngestedItem",
    "DocVQAIngester",
    "DocLayNetIngester",
    "INGESTERS",
    "save_items",
    "load_index",
]
