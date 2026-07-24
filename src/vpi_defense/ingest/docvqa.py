"""DocVQA ingester.

Source: https://huggingface.co/datasets/lmms-lab/DocVQA
Subset: "DocVQA" (the DocVQA config; also hosts "InfographicVQA")
Splits available: validation, test  (no train — this is an eval-only mirror)
Fields we rely on:
    image     : PIL image (auto-decoded by datasets)
    question  : str
    (additional fields like `answers`, `docId` exist but are not used here)

If you need the training split, use the official DocVQA distribution at
https://www.docvqa.org/ or a different mirror.
"""
from __future__ import annotations

from typing import Iterator

from vpi_defense.schemas import (
    ImageMetadata,
    ImageType,
    Language,
    SourceDataset,
    SplitTag,
)

from .base import BaseIngester, IngestedItem
from .utils import coerce_pil, make_image_id, pil_to_bytes, sha256_bytes


_HF_NAME = "lmms-lab/DocVQA"
_HF_SUBSET = "DocVQA"
_VALID_SPLITS = {"validation", "test"}


def _split_to_tag(split: str) -> SplitTag:
    """Map HF split names to our SplitTag. DocVQA has no train, so
    validation stays as VAL and test stays as TEST. If a caller wants
    to use validation as training data, they should re-tag downstream.
    """
    return {"validation": SplitTag.VAL, "test": SplitTag.TEST}[split]


class DocVQAIngester(BaseIngester):
    source = SourceDataset.DOCVQA

    def _convert(self, sample: dict, index: int, split: str) -> IngestedItem:
        """Convert one raw HF sample into an IngestedItem.

        Separated from `fetch` so it is unit-testable without any network.
        """
        pil = coerce_pil(sample["image"])
        image_bytes = pil_to_bytes(pil, fmt="PNG")

        question = sample.get("question") or ""
        image_id = make_image_id(f"docvqa_{split[:3]}", index)

        meta = ImageMetadata(
            image_id=image_id,
            source_dataset=SourceDataset.DOCVQA,
            image_type=ImageType.DOCUMENT,
            language=Language.EN,
            user_context=("DocVQA question: " + question)[:200] if question else None,
            image_path=f"docvqa/{image_id}.png",
            original_caption=question or None,
            width=pil.width,
            height=pil.height,
            content_hash=sha256_bytes(image_bytes),
            split=_split_to_tag(split),
        )
        return IngestedItem(metadata=meta, image_bytes=image_bytes)

    def fetch(self, n: int, split: str = "validation") -> Iterator[IngestedItem]:
        if split not in _VALID_SPLITS:
            raise ValueError(f"split must be one of {_VALID_SPLITS}, got {split!r}")

        # lazy import so schemas + tests work without `datasets` installed
        from datasets import load_dataset

        ds = load_dataset(_HF_NAME, _HF_SUBSET, split=split, streaming=True)
        for index, sample in enumerate(ds):
            if index >= n:
                break
            yield self._convert(sample, index, split)
