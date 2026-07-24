"""DocLayNet ingester.

Source: https://huggingface.co/datasets/ds4sd/DocLayNet-v1.2
Splits: train / validation / test
Fields (per HF dataset card):
    image     : PIL image (1025x1025 PNG originally)
    bboxes    : list of [x, y, w, h] boxes
    categories: list of category ids (1..11: Caption, Footnote, Formula,
                List-item, Page-footer, Page-header, Picture, Section-header,
                Table, Text, Title)
    pdf_cells : list of lists of PDF text cells matching bboxes
    metadata  : dict with doc_category, page_no, coco_url, etc.

Note: field names in v1.2 differ from the original COCO-format ds4sd/DocLayNet.
If you switch datasets, adjust `_convert()` accordingly. Field names should
be verified once against the live dataset (see README section on M2).
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


_HF_NAME = "ds4sd/DocLayNet-v1.2"
_VALID_SPLITS = {"train", "validation", "test"}


def _split_to_tag(split: str) -> SplitTag:
    return {"train": SplitTag.TRAIN, "validation": SplitTag.VAL, "test": SplitTag.TEST}[split]


class DocLayNetIngester(BaseIngester):
    source = SourceDataset.DOCLAYNET

    def _convert(self, sample: dict, index: int, split: str) -> IngestedItem:
        pil = coerce_pil(sample["image"])
        image_bytes = pil_to_bytes(pil, fmt="PNG")

        # metadata dict may or may not be present depending on dataset version
        meta_dict = sample.get("metadata") or {}
        doc_category = meta_dict.get("doc_category")
        page_no = meta_dict.get("page_no")

        context_parts = []
        if doc_category:
            context_parts.append(f"doc_category={doc_category}")
        if page_no is not None:
            context_parts.append(f"page={page_no}")
        user_context = "; ".join(context_parts) or None

        image_id = make_image_id(f"doclaynet_{split[:3]}", index)

        meta = ImageMetadata(
            image_id=image_id,
            source_dataset=SourceDataset.DOCLAYNET,
            image_type=ImageType.DOCUMENT,
            language=Language.EN,  # DocLayNet is predominantly English
            user_context=user_context,
            image_path=f"doclaynet/{image_id}.png",
            original_caption=None,
            width=pil.width,
            height=pil.height,
            content_hash=sha256_bytes(image_bytes),
            split=_split_to_tag(split),
        )
        return IngestedItem(metadata=meta, image_bytes=image_bytes)

    def fetch(self, n: int, split: str = "train") -> Iterator[IngestedItem]:
        if split not in _VALID_SPLITS:
            raise ValueError(f"split must be one of {_VALID_SPLITS}, got {split!r}")

        from datasets import load_dataset

        ds = load_dataset(_HF_NAME, split=split, streaming=True)
        for index, sample in enumerate(ds):
            if index >= n:
                break
            yield self._convert(sample, index, split)
