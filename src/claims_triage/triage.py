"""Business layer on top of the large-loss probability: triage tier and reason codes.

``assign_tier`` and ``reason_codes`` are plain functions so they can be unit tested without
MLflow. The two pyfunc classes wrap a fitted sklearn pipeline:

* ``LargeLossProbability`` returns the probability (used for batch scoring and the gate).
* ``TriageModel`` returns probability, tier and reason codes (used for the serving endpoint).

Before logging either class, call ``cloudpickle.register_pickle_by_value(claims_triage)`` so the
model is self-contained and loads without this package installed.
"""
from __future__ import annotations

import pandas as pd

from .features import ALL_FEATURES

TIER_THRESHOLDS = {"HIGH": 0.50, "ELEVATED": 0.20}   # probability cut-offs (agree these with claims leadership)


def assign_tier(prob: float) -> str:
    if prob >= TIER_THRESHOLDS["HIGH"]:
        return "HIGH"          # route to a senior adjuster on day one
    if prob >= TIER_THRESHOLDS["ELEVATED"]:
        return "ELEVATED"      # early supervisor review
    return "STANDARD"


def reason_codes(row: pd.Series) -> str:
    """Simple, readable flags for the adjuster. Not model explanations (use SHAP for that)."""
    reasons = []
    if row.get("attorney_at_fnol") == 1:
        reasons.append("ATTORNEY_AT_FNOL")
    if row.get("injury_flag") == 1:
        reasons.append("INJURY_REPORTED")
    lag = row.get("report_lag_days")
    if lag is not None and not pd.isna(lag) and lag > 30:
        reasons.append("LATE_REPORT")
    prior = row.get("prior_claims_3y")
    if prior is not None and not pd.isna(prior) and prior >= 2:
        reasons.append("FREQUENT_PRIOR_CLAIMS")
    return ",".join(reasons)


try:  # mlflow is present on Databricks; unit tests run without it
    import mlflow.pyfunc

    class LargeLossProbability(mlflow.pyfunc.PythonModel):
        """pyfunc whose predict() returns P(large loss) instead of a 0/1 label."""

        def __init__(self, sk_model):
            self.sk_model = sk_model

        def predict(self, context, model_input: pd.DataFrame, params=None):
            # Select model columns by name: scoring inputs may carry extra columns (keys, dates)
            return self.sk_model.predict_proba(model_input[ALL_FEATURES])[:, 1]

    class TriageModel(mlflow.pyfunc.PythonModel):
        """pyfunc for serving: probability + tier + reason codes."""

        def __init__(self, sk_model):
            self.sk_model = sk_model

        def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
            prob = self.sk_model.predict_proba(model_input[ALL_FEATURES])[:, 1]
            return pd.DataFrame({
                "large_loss_probability": prob.round(4),
                "triage_tier": [assign_tier(p) for p in prob],
                "reason_codes": model_input.apply(reason_codes, axis=1).astype(str),
            })
except ImportError:  # pragma: no cover
    LargeLossProbability = TriageModel = None
