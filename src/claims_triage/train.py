"""Model builders used by the training notebook and the retraining job (deploy-code pattern)."""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import ALL_FEATURES, build_preprocessor

DEFAULT_GBM_PARAMS = {"learning_rate": 0.06, "max_depth": 6, "max_iter": 300,
                      "min_samples_leaf": 40, "l2_regularization": 1.0}


def baseline_classifier(class_weight: str | None = "balanced") -> Pipeline:
    """Logistic regression: needs one-hot, imputation, log transforms and scaling."""
    return Pipeline([
        ("prep", build_preprocessor("linear")),
        ("model", LogisticRegression(max_iter=2000, class_weight=class_weight)),
    ])


def gbm_classifier(params: dict | None = None, class_weight: str | None = None) -> Pipeline:
    """Gradient-boosted trees: ordinal categories are enough, no scaling needed."""
    p = {**DEFAULT_GBM_PARAMS, **(params or {})}
    p["max_depth"] = int(p["max_depth"])
    p["max_iter"] = int(p["max_iter"])
    p["min_samples_leaf"] = int(p["min_samples_leaf"])
    return Pipeline([
        ("prep", build_preprocessor("tree")),
        ("model", HistGradientBoostingClassifier(class_weight=class_weight, random_state=42, **p)),
    ])


def gbm_regressor(params: dict | None = None) -> Pipeline:
    """Severity model. Train on log1p(incurred), then expm1 predictions back to dollars."""
    p = {**DEFAULT_GBM_PARAMS, **(params or {})}
    p["max_depth"] = int(p["max_depth"]) if p["max_depth"] is not None else None
    p["max_iter"] = int(p["max_iter"])
    p["min_samples_leaf"] = int(p["min_samples_leaf"])
    return Pipeline([
        ("prep", build_preprocessor("tree")),
        ("model", HistGradientBoostingRegressor(random_state=42, **p)),
    ])


def feature_frame(df):
    """Select model inputs in a fixed order (works for pandas DataFrames)."""
    return df[ALL_FEATURES]
