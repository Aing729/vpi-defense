"""Compare two VLM responses to decide whether masking changed the answer.

We use sentence-transformer cosine similarity because:
  - Robust to paraphrasing (a masked variant might describe the same object
    with slightly different words but the same meaning).
  - Fast (one embedding per response).
  - We already have the embedder loaded from M6.3 semantic features.

Interpretation of similarity `s`:
  - s ≥ high_threshold (default 0.85) → responses are essentially the same:
    masking removed nothing meaningful, so keep the ORIGINAL response.
  - s <  high_threshold                → masking materially changed the
    answer, so return the MASKED response as the safe answer.

The threshold is a hyperparameter — 0.85 is a reasonable starting point for
paraphrase-multilingual-MiniLM-L12-v2. Users can pass their own.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ComparisonResult:
    """Result of comparing an original and a masked response."""
    original: str
    masked: str
    similarity: float          # cosine in [-1, 1], typically [0, 1] for sentences
    changed: bool              # True if similarity < threshold
    threshold: float
    safe_response: str         # the one we hand back to the user


def compare_responses(
    original: str,
    masked: str,
    embedder: Any | None = None,
    threshold: float = 0.85,
) -> ComparisonResult:
    """Decide whether the masked response should replace the original.

    If `embedder` is None (no sentence-transformer available), we fall
    back to string-equality: unchanged text → same, any change → replace.
    That is a coarse fallback and downgrades safety guarantees, but keeps
    the pipeline running.
    """
    similarity = _cosine_similarity(original, masked, embedder)
    changed = similarity < threshold
    safe = masked if changed else original
    return ComparisonResult(
        original=original, masked=masked,
        similarity=similarity, changed=changed,
        threshold=threshold, safe_response=safe,
    )


def _cosine_similarity(a: str, b: str, embedder: Any | None) -> float:
    """Cosine similarity between two texts. Falls back to exact-match when
    no embedder is provided."""
    if embedder is None:
        return 1.0 if a.strip() == b.strip() else 0.0

    import numpy as np
    va = embedder.encode(a, convert_to_numpy=True)
    vb = embedder.encode(b, convert_to_numpy=True)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)
