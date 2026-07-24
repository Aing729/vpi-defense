"""Evaluation for the risk function.

Two entry points:
    - `evaluate_split`: single train/test split by group
    - `cross_validate` : grouped K-fold CV, aggregates fold metrics

Metrics reported: precision, recall, F1, ROC-AUC, confusion matrix.
Grouped by base_image_id so that variants of the same original image don't
leak into both train and test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .model import RiskModel


# ---------------------------------------------------------------------------
# Metric dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EvalMetrics:
    n_samples: int
    n_positive: int
    precision: float
    recall: float
    f1: float
    roc_auc: float
    threshold: float
    # Confusion matrix at `threshold`, rows=true, cols=pred, [[TN, FP], [FN, TP]]
    confusion: list[list[int]] = field(default_factory=list)


@dataclass
class CVReport:
    per_fold: list[EvalMetrics]
    mean_precision: float
    mean_recall: float
    mean_f1: float
    mean_roc_auc: float


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> EvalMetrics:
    """Compute all metrics at a given probability threshold.

    Falls back gracefully when the split contains only one class (ROC-AUC
    undefined → reported as NaN).
    """
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = (y_prob >= threshold).astype(int)
    n = len(y_true)
    n_pos = int(y_true.sum())

    # Precision/recall/F1 require both classes to be present in y_pred to
    # avoid warnings; `zero_division=0` returns 0 instead of raising.
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    if n_pos == 0 or n_pos == n:
        roc_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true, y_prob))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return EvalMetrics(
        n_samples=n, n_positive=n_pos,
        precision=precision, recall=recall, f1=f1,
        roc_auc=roc_auc, threshold=threshold, confusion=cm,
    )


# ---------------------------------------------------------------------------
# Grouped K-fold
# ---------------------------------------------------------------------------


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    threshold: float = 0.5,
    C: float = 1.0,
    random_state: int = 42,
) -> CVReport:
    """Grouped K-fold CV.

    Groups are base_image_ids: all rows with the same group_id stay together
    in one fold. This prevents leakage between the raw image and its variants.
    """
    from sklearn.model_selection import GroupKFold

    n_unique_groups = len(np.unique(groups))
    if n_unique_groups < 2:
        raise ValueError(
            f"cross-validation needs at least 2 groups, got {n_unique_groups}. "
            f"You probably have too few base images — collect more raw data first."
        )
    # Clamp splits to the number of groups available
    n_splits = min(n_splits, n_unique_groups)

    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics: list[EvalMetrics] = []
    for fold_i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        if len(np.unique(ytr)) < 2:
            # Degenerate: training fold has only one class. Skip.
            continue

        model = RiskModel.train(Xtr, ytr, C=C, random_state=random_state)
        yprob = model.predict_proba(Xte)
        fm = compute_metrics(yte, yprob, threshold=threshold)
        fold_metrics.append(fm)

    if not fold_metrics:
        raise RuntimeError("no valid folds — every training split had one class only")

    def _mean(attr: str) -> float:
        vals = [getattr(m, attr) for m in fold_metrics]
        # ignore NaN AUCs from single-class test folds
        vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(vals)) if vals else float("nan")

    return CVReport(
        per_fold=fold_metrics,
        mean_precision=_mean("precision"),
        mean_recall=_mean("recall"),
        mean_f1=_mean("f1"),
        mean_roc_auc=_mean("roc_auc"),
    )


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------


def format_metrics(m: EvalMetrics) -> str:
    lines = [
        f"  n={m.n_samples} (pos={m.n_positive}) @ threshold={m.threshold:.2f}",
        f"  precision = {m.precision:.3f}",
        f"  recall    = {m.recall:.3f}",
        f"  f1        = {m.f1:.3f}",
        f"  roc_auc   = {m.roc_auc:.3f}",
        f"  confusion = {m.confusion}    (rows=true, cols=pred, order [neg, pos])",
    ]
    return "\n".join(lines)


def format_cv_report(r: CVReport) -> str:
    lines = [f"grouped {len(r.per_fold)}-fold CV:"]
    for i, m in enumerate(r.per_fold):
        lines.append(f" fold {i+1}: p={m.precision:.3f} r={m.recall:.3f} "
                     f"f1={m.f1:.3f} auc={m.roc_auc:.3f}")
    lines.append(f" mean:   p={r.mean_precision:.3f} r={r.mean_recall:.3f} "
                 f"f1={r.mean_f1:.3f} auc={r.mean_roc_auc:.3f}")
    return "\n".join(lines)


def format_coefficients(model: RiskModel) -> str:
    lines = ["feature weights (positive → indicates injection):"]
    coefs = model.coefficients()
    for name, val in sorted(coefs.items(), key=lambda kv: -abs(kv[1])):
        sign = "+" if val >= 0 else "-"
        lines.append(f"  {name:<20s} {sign}{abs(val):.3f}")
    lines.append(f"  intercept            {model.intercept():+.3f}")
    return "\n".join(lines)
