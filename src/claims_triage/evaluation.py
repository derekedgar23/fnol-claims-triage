"""Metrics helpers used in notebooks, the champion gate, and tests."""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    """Threshold-free (ROC AUC, PR AUC, log loss) and thresholded (precision, recall, F1) metrics."""
    y_true = np.asarray(y_true)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-7, 1 - 1e-7)
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "log_loss": float(log_loss(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "base_rate": float(y_true.mean()),
    }


def regression_metrics_from_log(y_true_dollars, pred_log) -> dict[str, float]:
    """The model predicts log1p(dollars). Convert back to dollars BEFORE computing metrics."""
    y = np.asarray(y_true_dollars, dtype=float)
    pred = np.expm1(np.asarray(pred_log, dtype=float))
    return {
        "rmse": float(math.sqrt(np.mean((y - pred) ** 2))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        "rmse_log_scale": float(math.sqrt(np.mean((np.log1p(y) - np.asarray(pred_log)) ** 2))),
    }


def count_grid_models(param_grid: dict[str, list], folds: int, refit: bool = True) -> int:
    """Models trained by a grid search with k-fold cross-validation."""
    combos = math.prod(len(v) for v in param_grid.values())
    return combos * folds + (1 if refit else 0)


def challenger_wins(challenger: dict[str, float], champion: dict[str, float] | None,
                    metric: str = "pr_auc", min_gain: float = 0.005) -> bool:
    """Champion/challenger gate: promote only if the challenger beats the champion by min_gain."""
    if champion is None:
        return True
    return challenger[metric] >= champion[metric] + min_gain
