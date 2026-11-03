"""Fraud detection model builders: supervised classifier and unsupervised anomaly detection.

The supervised model is a GBM trained on labeled fraud data. The unsupervised models
(Isolation Forest and Local Outlier Factor) score claims without labels, surfacing
anomalies that may warrant SIU investigation.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline

from .features import ALL_FEATURES, build_preprocessor


# ---------------------------------------------------------------------------
# Supervised
# ---------------------------------------------------------------------------

def fraud_classifier(class_weight: str | None = "balanced") -> Pipeline:
    """GBM classifier for labeled fraud detection.

    Uses balanced class weights by default because fraud is rare (~2.5%).
    """
    return Pipeline([
        ("prep", build_preprocessor("tree")),
        ("model", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=30,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
    ])


# ---------------------------------------------------------------------------
# Unsupervised
# ---------------------------------------------------------------------------

def isolation_forest(contamination: float = 0.025, n_estimators: int = 300) -> Pipeline:
    """Isolation Forest: isolates anomalies by random recursive partitioning.

    Points that require fewer splits to isolate are more anomalous. Works well
    with mixed feature types after preprocessing.
    """
    return Pipeline([
        ("prep", build_preprocessor("tree")),
        ("model", IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples="auto",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def local_outlier_factor(contamination: float = 0.025, n_neighbors: int = 20) -> Pipeline:
    """Local Outlier Factor: density-based anomaly detection.

    Compares the local density of a point to its neighbors. Points in sparser
    regions relative to their neighbors are flagged as outliers. Must be used
    with novelty=True for scoring new data.
    """
    return Pipeline([
        ("prep", build_preprocessor("tree")),
        ("model", LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
            n_jobs=-1,
        )),
    ])


def anomaly_scores(model: Pipeline, X) -> np.ndarray:
    """Extract continuous anomaly scores from a fitted unsupervised pipeline.

    Returns scores where more negative = more anomalous (Isolation Forest convention).
    Normalized to [0, 1] where 1 = most anomalous for easier interpretation.
    """
    raw = model.named_steps["model"].decision_function(
        model.named_steps["prep"].transform(X)
    )
    # Invert and normalize: decision_function returns negative for outliers
    normalized = (raw.max() - raw) / (raw.max() - raw.min() + 1e-10)
    return normalized


FRAUD_FEATURES = ALL_FEATURES  # same feature set, different target
