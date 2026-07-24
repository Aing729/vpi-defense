"""Smoke test for ingesters.

We can't hit HuggingFace from this sandbox, so we test the pure conversion
logic (`_convert`) with fake HF-style samples. The full `fetch()` path is
covered on the user's side once they run the CLI against the real dataset.
"""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image, ImageDraw

from vpi_defense.ingest import DocLayNetIngester, DocVQAIngester, load_index, save_items
from vpi_defense.schemas import ImageMetadata, ImageType, Language, SourceDataset, SplitTag


def _make_fake_pil(w: int = 640, h: int = 480, text: str = "PAGE") -> Image.Image:
    """Create a small in-memory PIL image so we don't touch disk or network."""
    img = Image.new("RGB", (w, h), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, w - 20, h - 20], outline="black", width=2)
    draw.text((40, 40), text, fill="black")
    return img


def _pil_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_docvqa_convert():
    ing = DocVQAIngester()
    sample = {
        "image": _make_fake_pil(text="DocVQA sample"),
        "question": "What is the total amount on the invoice?",
    }
    item = ing._convert(sample, index=42, split="validation")

    assert item.metadata.source_dataset == SourceDataset.DOCVQA
    assert item.metadata.image_type == ImageType.DOCUMENT
    assert item.metadata.language == Language.EN
    assert item.metadata.image_id == "img_docvqa_val_000042"
    assert item.metadata.split == SplitTag.VAL
    assert item.metadata.image_path.endswith("img_docvqa_val_000042.png")
    assert item.metadata.width == 640 and item.metadata.height == 480
    assert item.metadata.content_hash and len(item.metadata.content_hash) == 64
    assert "invoice" in (item.metadata.original_caption or "")
    print(f"[OK] docvqa convert: {item.metadata.image_id} ({item.metadata.width}x{item.metadata.height})")


def test_doclaynet_convert_with_metadata():
    ing = DocLayNetIngester()
    sample = {
        "image": _make_fake_pil(w=1025, h=1025, text="DocLayNet page"),
        "metadata": {"doc_category": "financial_reports", "page_no": 9},
        "bboxes": [[0.1, 0.2, 0.3, 0.4]],
        "categories": [10],
    }
    item = ing._convert(sample, index=7, split="train")

    assert item.metadata.source_dataset == SourceDataset.DOCLAYNET
    assert item.metadata.split == SplitTag.TRAIN
    assert item.metadata.image_id == "img_doclaynet_tra_000007"
    assert item.metadata.user_context is not None
    assert "financial_reports" in item.metadata.user_context
    assert "page=9" in item.metadata.user_context
    print(f"[OK] doclaynet convert: {item.metadata.image_id} context={item.metadata.user_context}")


def test_doclaynet_convert_without_metadata():
    ing = DocLayNetIngester()
    sample = {"image": _make_fake_pil(w=1025, h=1025)}
    item = ing._convert(sample, index=0, split="validation")
    assert item.metadata.user_context is None
    assert item.metadata.split == SplitTag.VAL
    print(f"[OK] doclaynet convert (no metadata): user_context is None")


def test_docvqa_invalid_split_rejected():
    ing = DocVQAIngester()
    try:
        # generator function — need to advance to trigger the check
        next(iter(ing.fetch(n=1, split="train")))
    except ValueError as e:
        print(f"[OK] docvqa rejects invalid split: {str(e)[:60]}...")
        return
    except Exception as e:
        raise AssertionError(f"unexpected exception type: {type(e).__name__}: {e}")
    raise AssertionError("invalid split was not rejected")


def test_save_items_and_load_index_roundtrip():
    ing = DocVQAIngester()
    samples = [
        {"image": _make_fake_pil(text=f"page {i}"), "question": f"q{i}"}
        for i in range(3)
    ]
    items = [ing._convert(s, index=i, split="validation") for i, s in enumerate(samples)]

    with tempfile.TemporaryDirectory() as td:
        out_root = Path(td) / "raw"
        index_path = out_root / "index.jsonl"

        written = save_items(items, out_root=out_root, index_path=index_path)
        assert len(written) == 3

        # every image file should exist on disk
        for m in written:
            assert (out_root / m.image_path).exists(), f"missing: {m.image_path}"

        # index should reload identically
        reloaded = load_index(index_path)
        assert len(reloaded) == 3
        assert reloaded[0].image_id == written[0].image_id
        print(f"[OK] save_items + load_index: 3 items written and reloaded from {out_root.name}")


def test_save_items_refuses_overwrite():
    ing = DocVQAIngester()
    item = ing._convert(
        {"image": _make_fake_pil(), "question": "q"}, index=0, split="validation"
    )
    with tempfile.TemporaryDirectory() as td:
        out_root = Path(td) / "raw"
        save_items([item], out_root=out_root)
        try:
            save_items([item], out_root=out_root)
        except FileExistsError as e:
            print(f"[OK] save_items refuses overwrite by default: {str(e)[:50]}...")
            return
    raise AssertionError("second save_items did not raise FileExistsError")


if __name__ == "__main__":
    test_docvqa_convert()
    test_doclaynet_convert_with_metadata()
    test_doclaynet_convert_without_metadata()
    test_docvqa_invalid_split_rejected()
    test_save_items_and_load_index_roundtrip()
    test_save_items_refuses_overwrite()
    print("\nall ingester smoke tests passed.")
