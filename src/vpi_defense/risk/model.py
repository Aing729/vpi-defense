"""Risk function: L2-regularized logistic regression on 10-dim features.

Kept as a thin wrapper around sklearn so we can add IG, per-feature
attribution, and other analyses on top without spreading model details
across the codebase.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import FEATURE_NAMES


@dataclass
class RiskModel:
    """L2 logistic regression + a copy of feature_names for provenance.

    We keep the sklearn estimator opaque; callers use `predict_proba`,
    `predict`, `coefficients`, `save`, `load`. If you need something more
    sklearn-specific, use `.estimator` directly.
    """
    estimator: Any                    # sklearn.linear_model.LogisticRegression
    feature_names: tuple[str, ...] = FEATURE_NAMES

    # -----------------------------------------------------------------
    # Training entrypoint
    # -----------------------------------------------------------------

    @classmethod
    def train(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        C: float = 1.0,
        class_weight: str | dict | None = "balanced",
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> "RiskModel":
        """Fit the logistic regression.

        Args:
            X            : (N, 10) feature matrix
            y            : (N,) binary labels
            C            : inverse of L2 regularization strength (sklearn convention)
            class_weight : 'balanced' automatically weighs by inverse frequency
            max_iter     : LBFGS iterations
            random_state : for reproducibility
        """
        from sklearn.linear_model import LogisticRegression

        est = LogisticRegression(
            penalty="l2",
            C=C,
            solver="lbfgs",
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
        )
        est.fit(X, y)
        return cls(estimator=est)

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(is_injection=1) for each row. Shape (N,)."""
        return self.estimator.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary predictions at the given probability threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)

    # -----------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------

    def coefficients(self) -> dict[str, float]:
        """Return the fitted weights per feature (positive → indicates injection)."""
        coef = self.estimator.coef_.ravel()
        return dict(zip(self.feature_names, (float(c) for c in coef)))

    def intercept(self) -> float:
        return float(self.estimator.intercept_.ravel()[0])

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({
                "estimator": self.estimator,
                "feature_names": self.feature_names,
            }, f)

    @classmethod
    def load(cls, path: Path) -> "RiskModel":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        return cls(estimator=payload["estimator"],
                   feature_names=tuple(payload["feature_names"]))
