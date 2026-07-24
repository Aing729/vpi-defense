"""Smoke tests for M6.5 Integrated Gradients on the risk model."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from vpi_defense.risk import (
    Attribution,
    RiskModel,
    format_attribution,
    linear_attribution,
    sigmoid_attribution,
)


def _train_toy_model(seed: int = 0) -> RiskModel:
    """Small linearly-separable set so we get a stable, non-degenerate model."""
    rng = np.random.default_rng(seed)
    n = 200
    n_pos = 40
    X_pos = rng.uniform(size=(n_pos, 10)).astype(np.float32) * 0.3
    X_pos[:, 5] += 0.7   # imperativeness
    X_pos[:, 8] += 0.6   # url_qr_relevance
    X_neg = rng.uniform(size=(n - n_pos, 10)).astype(np.float32) * 0.5
    X = np.clip(np.concatenate([X_pos, X_neg]), 0.0, 1.0)
    y = np.array([1] * n_pos + [0] * (n - n_pos), dtype=int)
    return RiskModel.train(X, y, random_state=0)


# ---------------------------------------------------------------------------
# Linear IG: exact closed-form properties
# ---------------------------------------------------------------------------


def test_linear_attribution_sums_to_score_delta():
    """For a linear model, ∑ attribution_i = s(x) - s(baseline). This is the
    'completeness' axiom of IG."""
    model = _train_toy_model()
    x = np.array([0.9, 0.1, 0.8, 0.05, 0.9,
                  0.85, 0.5, 0.3, 0.75, 0.6], dtype=np.float32)
    a = linear_attribution(model, x)
    total_attribution = float(a.values.sum())
    delta_score = a.score_at_input - a.score_at_baseline
    assert abs(total_attribution - delta_score) < 1e-4, (
        total_attribution, delta_score)
    print(f"[OK] ∑ IG = {total_attribution:+.4f} ≈ Δscore = {delta_score:+.4f}")


def test_linear_attribution_zero_baseline_is_zero():
    """IG(baseline, baseline) must be all zeros."""
    model = _train_toy_model()
    baseline = np.zeros(10, dtype=np.float32)
    a = linear_attribution(model, baseline, baseline=baseline)
    assert np.allclose(a.values, 0.0)
    print("[OK] IG(baseline, baseline) is zero vector")


def test_linear_attribution_direction_matches_sign():
    """A feature with positive weight and positive input value must
    contribute positively; negative weight or negative input negatively."""
    model = _train_toy_model()
    coefs = model.coefficients()
    # Take a random-ish input away from baseline
    x = np.array([0.5] * 10, dtype=np.float32)
    a = linear_attribution(model, x)
    for i, name in enumerate(a.feature_names):
        expected_sign = np.sign(coefs[name])
        actual_sign = np.sign(a.values[i])
        # if weight ≈ 0 the value is ≈ 0, sign undefined — skip
        if abs(coefs[name]) < 1e-6:
            continue
        assert expected_sign == actual_sign, (
            f"{name}: weight_sign={expected_sign}, "
            f"attribution_sign={actual_sign}")
    print("[OK] attribution sign matches weight sign for all non-zero features")


def test_linear_attribution_is_additive_across_features():
    """Attributions decompose the score change perfectly:
        IG_i = w_i · (x_i - baseline_i)
    This test recomputes them by hand and confirms."""
    model = _train_toy_model()
    x = np.array([0.6, 0.2, 0.7, 0.1, 0.8,
                  0.9, 0.4, 0.5, 0.85, 0.3], dtype=np.float32)
    a = linear_attribution(model, x)
    w = model.estimator.coef_.ravel()
    manual = w * x  # baseline is 0
    assert np.allclose(a.values, manual, atol=1e-5), (a.values, manual)
    print("[OK] linear IG matches the closed-form w · (x - baseline)")


# ---------------------------------------------------------------------------
# Sigmoid IG: numerical, should agree with F(x) - F(baseline)
# ---------------------------------------------------------------------------


def test_sigmoid_attribution_sums_to_prob_delta():
    """Numerical IG on σ(w·x + b) should sum (approximately) to
    F(x) - F(baseline) — this is completeness in probability units."""
    model = _train_toy_model()
    x = np.array([0.9, 0.1, 0.8, 0.05, 0.9,
                  0.85, 0.5, 0.3, 0.75, 0.6], dtype=np.float32)
    a = sigmoid_attribution(model, x, n_steps=128)
    total = float(a.values.sum())
    delta = a.prob_at_input - a.prob_at_baseline
    # Allow ~1% error for the Riemann approximation
    assert abs(total - delta) < 0.02, (total, delta)
    print(f"[OK] ∑ σ-IG = {total:+.4f} ≈ Δp = {delta:+.4f}")


# ---------------------------------------------------------------------------
# Sorting + pretty-print sanity
# ---------------------------------------------------------------------------


def test_attribution_sorted_by_magnitude():
    model = _train_toy_model()
    x = np.random.default_rng(0).uniform(size=10).astype(np.float32)
    a = linear_attribution(model, x)
    ordered = a.sorted(by_absolute=True)
    magnitudes = [abs(v) for _, v in ordered]
    assert magnitudes == sorted(magnitudes, reverse=True)
    print(f"[OK] .sorted() returns descending absolute values; "
          f"top: {ordered[0]!r}")


def test_format_attribution_does_not_crash():
    model = _train_toy_model()
    x = np.array([0.7] * 10, dtype=np.float32)
    a = linear_attribution(model, x)
    text = format_attribution(a, top_k=5)
    assert "attribution" in text
    assert "score:" in text
    print("[OK] format_attribution produces non-empty output")


# ---------------------------------------------------------------------------
# Custom baseline
# ---------------------------------------------------------------------------


def test_custom_baseline_shifts_attribution():
    model = _train_toy_model()
    x = np.array([0.8] * 10, dtype=np.float32)
    b1 = np.zeros(10, dtype=np.float32)
    b2 = np.full(10, 0.4, dtype=np.float32)
    a1 = linear_attribution(model, x, baseline=b1)
    a2 = linear_attribution(model, x, baseline=b2)
    # Higher baseline → smaller Δ, hence smaller absolute attributions
    assert abs(a2.values).sum() < abs(a1.values).sum()
    print(f"[OK] baseline=0.4·1 gives smaller total attribution "
          f"({abs(a2.values).sum():.3f}) than baseline=0 "
          f"({abs(a1.values).sum():.3f})")


if __name__ == "__main__":
    test_linear_attribution_sums_to_score_delta()
    test_linear_attribution_zero_baseline_is_zero()
    test_linear_attribution_direction_matches_sign()
    test_linear_attribution_is_additive_across_features()
    test_sigmoid_attribution_sums_to_prob_delta()
    test_attribution_sorted_by_magnitude()
    test_format_attribution_does_not_crash()
    test_custom_baseline_shifts_attribution()
    print("\nall IG smoke tests passed.")
