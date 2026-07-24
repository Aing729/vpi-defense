"""Integrated Gradients for the risk function.

For a logistic regression, IG has a closed-form:

    F(x)   = σ(w·x + b)
    F'(x)  = σ(w·x + b) · (1 - σ(w·x + b)) · w                (chain rule)

    IG_i(x, baseline)
        = (x_i - baseline_i) · ∫₀¹ F'(baseline + α(x-baseline))_i dα

For a *linear* score s(x) = w·x + b (before sigmoid), the integral collapses
to just `w_i · (x_i - baseline_i)` — no need to sample α steps. This gives
attributions that sum to `s(x) - s(baseline)`.

We keep two views:
  - `linear_attribution` : the exact score-space IG (fast, closed-form)
  - `sigmoid_attribution`: numerical IG over σ (matches the probability F(x)).
                          Uses `n_steps` Riemann samples.

For interpreting "which feature made this region look like an injection",
linear_attribution is the standard choice — it's what SHAP would give for a
linear model, and it's decomposable (features sum to the total).

Baseline: zero vector (feature vector of a "featureless" region).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .dataset import FEATURE_NAMES
from .model import RiskModel


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class Attribution:
    """Per-feature attribution for one region."""
    feature_names: tuple[str, ...]
    values: np.ndarray                # shape (n_features,)
    baseline: np.ndarray              # shape (n_features,)
    input: np.ndarray                 # shape (n_features,)
    score_at_input: float             # w·x + b
    score_at_baseline: float          # w·baseline + b
    prob_at_input: float              # σ(w·x + b)
    prob_at_baseline: float           # σ(w·baseline + b)

    def sorted(self, by_absolute: bool = True) -> list[tuple[str, float]]:
        """Feature attributions sorted by magnitude (descending)."""
        pairs = list(zip(self.feature_names, self.values.tolist()))
        key = (lambda p: -abs(p[1])) if by_absolute else (lambda p: -p[1])
        return sorted(pairs, key=key)


# ---------------------------------------------------------------------------
# Core IG computation
# ---------------------------------------------------------------------------


def linear_attribution(
    model: RiskModel,
    x: np.ndarray,
    baseline: np.ndarray | None = None,
) -> Attribution:
    """Closed-form IG on the pre-sigmoid score. Feature attributions sum to
    `s(x) - s(baseline)`, which is exact for a linear model."""
    x = np.asarray(x, dtype=np.float32).ravel()
    if baseline is None:
        baseline = np.zeros_like(x)
    else:
        baseline = np.asarray(baseline, dtype=np.float32).ravel()

    w = model.estimator.coef_.ravel().astype(np.float32)
    b = float(model.estimator.intercept_.ravel()[0])

    values = w * (x - baseline)

    score_x = float(w @ x + b)
    score_b = float(w @ baseline + b)
    prob_x = float(_sigmoid(score_x))
    prob_b = float(_sigmoid(score_b))

    return Attribution(
        feature_names=model.feature_names,
        values=values,
        baseline=baseline,
        input=x,
        score_at_input=score_x,
        score_at_baseline=score_b,
        prob_at_input=prob_x,
        prob_at_baseline=prob_b,
    )


def sigmoid_attribution(
    model: RiskModel,
    x: np.ndarray,
    baseline: np.ndarray | None = None,
    n_steps: int = 64,
) -> Attribution:
    """Numerical IG over the probability output. Sums (approximately) to
    `F(x) - F(baseline)`. Slower than linear_attribution but matches the
    definition in the IG paper (Sundararajan et al., 2017) when you want
    attributions in probability units rather than score units.
    """
    x = np.asarray(x, dtype=np.float32).ravel()
    if baseline is None:
        baseline = np.zeros_like(x)
    else:
        baseline = np.asarray(baseline, dtype=np.float32).ravel()

    w = model.estimator.coef_.ravel().astype(np.float32)
    b = float(model.estimator.intercept_.ravel()[0])

    # Riemann sum: sample α ∈ (0, 1) at midpoints for the second-order
    # trapezoidal-ish accuracy.
    alphas = (np.arange(n_steps) + 0.5) / n_steps
    grad_sum = np.zeros_like(x)
    for a in alphas:
        point = baseline + a * (x - baseline)
        s = float(w @ point + b)
        # d/dx σ(w·x + b) = σ(s)(1-σ(s)) · w
        sig = _sigmoid(s)
        grad_sum += sig * (1 - sig) * w
    grad_avg = grad_sum / n_steps
    values = (x - baseline) * grad_avg

    score_x = float(w @ x + b)
    score_b = float(w @ baseline + b)
    return Attribution(
        feature_names=model.feature_names,
        values=values,
        baseline=baseline,
        input=x,
        score_at_input=score_x,
        score_at_baseline=score_b,
        prob_at_input=float(_sigmoid(score_x)),
        prob_at_baseline=float(_sigmoid(score_b)),
    )


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------


def format_attribution(a: Attribution, top_k: int = 10) -> str:
    lines = [
        f"score:        s(baseline)={a.score_at_baseline:+.3f}  →  s(x)={a.score_at_input:+.3f}",
        f"probability:  p(baseline)={a.prob_at_baseline:.3f}   →  p(x)={a.prob_at_input:.3f}",
        f"per-feature attribution (top {top_k} by |value|):",
    ]
    for name, val in a.sorted(by_absolute=True)[:top_k]:
        sign = "+" if val >= 0 else "-"
        bar_len = int(min(30, abs(val) * 30 / max(1e-6, abs(a.values).max())))
        bar = ("█" if val >= 0 else "░") * bar_len
        lines.append(f"  {name:<20s} {sign}{abs(val):.3f}  {bar}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sigmoid(z: float) -> float:
    # numerically stable
    if z >= 0:
        ez = np.exp(-z)
        return float(1.0 / (1.0 + ez))
    ez = np.exp(z)
    return float(ez / (1.0 + ez))
