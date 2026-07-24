"""Base interface and shared writer for dataset ingesters.

Design:
  - Each source dataset (DocVQA, DocLayNet, ...) implements `BaseIngester`.
  - `fetch()` yields `IngestedItem` (metadata + raw image bytes) lazily so we
    can stream from HuggingFace without holding everything in memory.
  - `save_items()` writes bytes to disk and appends metadata to a JSONL index.
    JSONL is chosen (over a single JSON array) because it is append-friendly
    and survives partial runs.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from vpi_defense.schemas import ImageMetadata, SourceDataset


@dataclass
class IngestedItem:
    """Metadata + raw image bytes yielded by an ingester.

    `metadata.image_path` should already be set to the intended relative path
    under the output root (e.g. 'raw/docvqa/img_docvqa_val_000042.png').
    """
    metadata: ImageMetadata
    image_bytes: bytes


class BaseIngester(ABC):
    """Abstract ingester. Subclasses set `source` and implement `fetch`."""

    source: SourceDataset

    @abstractmethod
    def fetch(self, n: int, split: str) -> Iterator[IngestedItem]:
        """Yield up to `n` items from the given split.

        Implementations should use HF `datasets` with `streaming=True` so we
        don't download the full dataset just to sample the first N pages.
        """
        ...


def save_items(
    items: Iterable[IngestedItem],
    out_root: Path,
    index_path: Path | None = None,
    overwrite: bool = False,
) -> list[ImageMetadata]:
    """Persist ingested items to disk.

    Layout under `out_root`:
        <metadata.image_path>          # image file (relative path from metadata)
        <index_path or out_root/index.jsonl>  # append-only metadata log

    Returns the list of successfully written metadata records.

    Raises:
        FileExistsError: if `overwrite=False` and a target image already exists.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if index_path is None:
        index_path = out_root / "index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)

    written: list[ImageMetadata] = []
    with index_path.open("a", encoding="utf-8") as idx:
        for item in items:
            target = out_root / item.metadata.image_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not overwrite:
                raise FileExistsError(f"target already exists: {target}")
            target.write_bytes(item.image_bytes)
            idx.write(item.metadata.model_dump_json() + "\n")
            written.append(item.metadata)
    return written


def load_index(index_path: Path) -> list[ImageMetadata]:
    """Read back a JSONL index into a list of ImageMetadata."""
    index_path = Path(index_path)
    with index_path.open("r", encoding="utf-8") as f:
        return [ImageMetadata.model_validate_json(line) for line in f if line.strip()]
