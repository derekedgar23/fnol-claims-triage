"""Feature definitions and sklearn preprocessing shared by notebooks, jobs and tests.

Pipelines use only sklearn and numpy objects, so a logged model can be loaded or served
without this package installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

# Columns that live in the policy feature table (looked up by policy_id)
POLICY_FEATURES = [
    "line_of_business", "insured_state", "policy_limit", "deductible",
    "annual_revenue", "years_in_business", "total_insured_value", "prior_claims_3y",
]
# Columns known on the claim at First Notice of Loss (passed in with the claim)
FNOL_FEATURES = ["cause_of_loss", "report_lag_days", "injury_flag", "attorney_at_fnol", "loss_month"]

CATEGORICAL = ["line_of_business", "insured_state", "cause_of_loss"]
SKEWED = ["annual_revenue", "total_insured_value", "report_lag_days"]   # log1p in linear models
NUMERIC = [c for c in POLICY_FEATURES + FNOL_FEATURES if c not in CATEGORICAL]
ALL_FEATURES = CATEGORICAL + NUMERIC


def iqr_bounds(values: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Return (low, high) fences: Q1 - k*IQR and Q3 + k*IQR."""
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def std_bounds(values: pd.Series, n_std: float = 3.0) -> tuple[float, float]:
    """Return mean +/- n_std standard deviations."""
    m, s = values.mean(), values.std()
    return float(m - n_std * s), float(m + n_std * s)


def build_preprocessor(model_family: str = "tree") -> ColumnTransformer:
    """Preprocessing for a model family.

    linear: median imputation, log1p for skewed columns, scaling, one-hot categories.
    tree:   median imputation, ordinal-encoded categories (trees do not need one-hot).
    """
    cat_impute = SimpleImputer(strategy="most_frequent")          # mode for categoricals
    if model_family == "linear":
        skewed = Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                           ("scale", StandardScaler())])
        numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        cats = Pipeline([("impute", cat_impute), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
        return ColumnTransformer([
            ("skewed", skewed, SKEWED),
            ("numeric", numeric, [c for c in NUMERIC if c not in SKEWED]),
            ("cats", cats, CATEGORICAL),
        ])
    if model_family == "tree":
        cats = Pipeline([("impute", cat_impute),
                         ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])
        return ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median"), NUMERIC),
            ("cats", cats, CATEGORICAL),
        ])
    raise ValueError(f"unknown model_family: {model_family}")

