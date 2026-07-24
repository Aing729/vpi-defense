"""Semantic feature extraction for a region.

Produces `SemanticFeatures` (imperativeness I, context_relevance S,
action_inducement M, url_qr_relevance U, informativeness C).

Strategy (option C from the design discussion):
  1. Rule-based scoring is always tried first — deterministic, free.
  2. If a sentence-transformer model is available, embedding-based
     similarity to anchor sentences refines the rule score.
  3. LLM fallback is optional (`llm_client` argument) and only invoked for
     ambiguous cases where rules and embeddings both give a middling score.

The extractor degrades gracefully:
  - `sentence-transformers` not installed  → skip embedding step
  - `llm_client` is None                    → skip LLM step
  - Empty text                              → returns zeros / defaults
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from vpi_defense.schemas import RegionType, SemanticFeatures

from . import text_patterns as tp


# Rule-score cutoffs. Above high_confidence we trust the rule and skip other
# steps for speed; below low_confidence we might invoke the LLM.
_HIGH_CONFIDENCE = 0.8
_LOW_CONFIDENCE = 0.3


# ---------------------------------------------------------------------------
# Context (built once per image)
# ---------------------------------------------------------------------------


@dataclass
class SemanticContext:
    """Per-image data reused across all regions."""
    image_texts: list[str] = field(default_factory=list)
    embedder: Any | None = None  # sentence_transformers.SentenceTransformer

    # Cached embeddings for anchor example sentences (lazily computed).
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)


def build_semantic_context(
    all_region_texts: Iterable[str],
    embedder: Any | None = None,
) -> SemanticContext:
    """Prepare per-image context.

    `all_region_texts` should be the OCR text of every region we know about
    on this image — including the one we're scoring, since context is by
    definition the image as a whole.
    """
    texts = [t for t in all_region_texts if t and t.strip()]
    return SemanticContext(image_texts=texts, embedder=embedder)


def try_load_default_embedder() -> Any | None:
    """Attempt to load a lightweight multilingual sentence embedder.

    Returns None if `sentence-transformers` isn't installed or the model
    can't be fetched. Callers should treat None as "skip embedding step".
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    try:
        # ~120MB, supports Korean + English + many others
        return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def extract_semantic_features(
    text: str,
    region_type: RegionType,
    ctx: SemanticContext,
    llm_client: Any | None = None,
) -> SemanticFeatures:
    """Compute all 5 semantic features for one region."""
    text = (text or "").strip()

    # Special handling for QR: the "text" here is a description, not decoded
    # payload, so bump url_qr_relevance based on region_type as well.
    is_qr = (region_type == RegionType.QR_CODE)

    if not text and not is_qr:
        return SemanticFeatures(
            imperativeness=0.0, context_relevance=0.0,
            action_inducement=0.0, url_qr_relevance=0.0,
            informativeness=0.0,
        )

    imperativeness = _score_imperativeness(text, ctx, llm_client)
    context_relevance = _score_context_relevance(text, ctx)
    action_inducement = _score_action_inducement(text, ctx, llm_client)
    url_qr_relevance = _score_url_qr(text, is_qr=is_qr)
    informativeness = _score_informativeness(text, ctx, llm_client)

    return SemanticFeatures(
        imperativeness=imperativeness,
        context_relevance=context_relevance,
        action_inducement=action_inducement,
        url_qr_relevance=url_qr_relevance,
        informativeness=informativeness,
    )


# ---------------------------------------------------------------------------
# Individual feature scorers
# ---------------------------------------------------------------------------


def _score_imperativeness(text: str, ctx: SemanticContext, llm_client: Any | None) -> float:
    rule_score = _rule_imperative(text)
    if rule_score >= _HIGH_CONFIDENCE:
        return rule_score

    emb_score = _embedding_similarity_max(text, tp.IMPERATIVE_EXAMPLES, ctx, "imperative")
    combined = max(rule_score, emb_score)
    if combined < _LOW_CONFIDENCE and llm_client is not None:
        return _llm_score_imperativeness(text, llm_client)
    return combined


def _score_context_relevance(text: str, ctx: SemanticContext) -> float:
    """How well does `text` fit the image's other content?"""
    others = [t for t in ctx.image_texts if t.strip() and t.strip() != text.strip()]
    if not others or ctx.embedder is None:
        return 0.5  # neutral when we can't compute
    return _embedding_similarity_to_bag(text, others, ctx)


def _score_action_inducement(text: str, ctx: SemanticContext, llm_client: Any | None) -> float:
    rule_score = _rule_action_inducement(text)
    if rule_score >= _HIGH_CONFIDENCE:
        return rule_score
    emb_score = _embedding_similarity_max(
        text, tp.ACTION_INDUCEMENT_EXAMPLES, ctx, "action_inducement",
    )
    combined = max(rule_score, emb_score)
    if combined < _LOW_CONFIDENCE and llm_client is not None:
        return _llm_score_action_inducement(text, llm_client)
    return combined


def _score_url_qr(text: str, is_qr: bool = False) -> float:
    """Regex-only. URL or QR mention → high."""
    if is_qr:
        return 1.0
    if tp.URL_REGEX.search(text):
        return 1.0
    if tp.QR_REGEX.search(text):
        return 0.7
    return 0.0


def _score_informativeness(text: str, ctx: SemanticContext, llm_client: Any | None) -> float:
    emb_score = _embedding_similarity_max(
        text, tp.INFORMATIVE_EXAMPLES, ctx, "informative",
    )
    # Length-based sanity: extremely short text is unlikely to be informative
    if len(text.strip()) < 3:
        return min(emb_score, 0.3)
    if emb_score < _LOW_CONFIDENCE and llm_client is not None:
        return _llm_score_informativeness(text, llm_client)
    return emb_score


# ---------------------------------------------------------------------------
# Rule scorers
# ---------------------------------------------------------------------------


def _rule_imperative(text: str) -> float:
    """Number of imperative-pattern matches → saturated score in [0, 1]."""
    if not text:
        return 0.0
    matches = tp.IMPERATIVE_REGEX.findall(text)
    n = len(matches)
    # 1 match → 0.6, 2 → 0.85, 3+ → 1.0 (saturates fast)
    return min(1.0, 0.6 + 0.2 * (n - 1)) if n >= 1 else 0.0


def _rule_action_inducement(text: str) -> float:
    if not text:
        return 0.0
    n = len(tp.ACTION_INDUCEMENT_REGEX.findall(text))
    if n >= 1:
        return min(1.0, 0.6 + 0.2 * (n - 1))
    return 0.0


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def _embed(text: str, ctx: SemanticContext):
    """Embed a single text with caching. Returns None if no embedder."""
    if ctx.embedder is None:
        return None
    cache_key = f"txt::{text}"
    if cache_key not in ctx._cache:
        ctx._cache[cache_key] = ctx.embedder.encode(text, convert_to_numpy=True)
    return ctx._cache[cache_key]


def _embed_examples(examples: Sequence[str], ctx: SemanticContext, tag: str):
    """Embed a list of anchor examples with per-tag caching."""
    if ctx.embedder is None:
        return None
    cache_key = f"anchors::{tag}"
    if cache_key not in ctx._cache:
        ctx._cache[cache_key] = ctx.embedder.encode(
            list(examples), convert_to_numpy=True
        )
    return ctx._cache[cache_key]


def _cosine(a, b) -> float:
    import numpy as np
    denom = float((np.linalg.norm(a) * np.linalg.norm(b)))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _embedding_similarity_max(
    text: str, examples: Sequence[str], ctx: SemanticContext, tag: str,
) -> float:
    """Max cosine similarity between text and any anchor. Returns 0 if no embedder."""
    txt_emb = _embed(text, ctx)
    if txt_emb is None:
        return 0.0
    anchor_embs = _embed_examples(examples, ctx, tag)
    if anchor_embs is None:
        return 0.0
    sims = [_cosine(txt_emb, a) for a in anchor_embs]
    if not sims:
        return 0.0
    # Cosine can be negative for unrelated text; clip to [0, 1]
    return max(0.0, min(1.0, max(sims)))


def _embedding_similarity_to_bag(
    text: str, others: Sequence[str], ctx: SemanticContext,
) -> float:
    """Mean cosine similarity between text and a bag of other texts."""
    txt_emb = _embed(text, ctx)
    if txt_emb is None:
        return 0.5
    embs = [_embed(o, ctx) for o in others]
    embs = [e for e in embs if e is not None]
    if not embs:
        return 0.5
    sims = [_cosine(txt_emb, e) for e in embs]
    return max(0.0, min(1.0, sum(sims) / len(sims)))


# ---------------------------------------------------------------------------
# Optional LLM fallback (only when both rule and embedding are inconclusive)
# ---------------------------------------------------------------------------


def _llm_score_imperativeness(text: str, llm_client: Any) -> float:
    prompt = (
        "Rate how imperative or instruction-like this text is on a scale of "
        "0.0 (purely informational) to 1.0 (strongly commands the reader to "
        "act). Respond with only a decimal number.\n\n"
        f"Text: {text}"
    )
    return _llm_score_call(prompt, llm_client)


def _llm_score_action_inducement(text: str, llm_client: Any) -> float:
    prompt = (
        "Rate how strongly this text urges the reader to take a specific action "
        "(click, scan, purchase, register, respond) on a scale of 0.0 to 1.0. "
        "Respond with only a decimal number.\n\n"
        f"Text: {text}"
    )
    return _llm_score_call(prompt, llm_client)


def _llm_score_informativeness(text: str, llm_client: Any) -> float:
    prompt = (
        "Rate how much useful, factual information this text conveys to a "
        "visually impaired user reading the image, on a scale of 0.0 (no "
        "information) to 1.0 (highly informative). Respond with only a "
        "decimal number.\n\n"
        f"Text: {text}"
    )
    return _llm_score_call(prompt, llm_client)


def _llm_score_call(prompt: str, llm_client: Any) -> float:
    """Ask the LLM for a scalar score. Returns 0.5 (neutral) on any error.

    llm_client is treated duck-typed: it must have `_generate_raw(image)`
    that returns text OR a `.messages.create(...)` API. We don't rely on
    either here — the caller is expected to wrap. For now this is a stub
    that returns 0.5 so the pipeline is exercised without incurring API
    calls. Wire up when we actually enable LLM fallback.
    """
    return 0.5
