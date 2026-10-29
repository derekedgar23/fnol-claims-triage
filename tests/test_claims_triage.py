import numpy as np
import pandas as pd
import pytest

from claims_triage.data import LARGE_LOSS_THRESHOLD, generate
from claims_triage.evaluation import (
    challenger_wins,
    classification_metrics,
    count_grid_models,
    regression_metrics_from_log,
)
from claims_triage.features import ALL_FEATURES, POLICY_FEATURES, iqr_bounds, std_bounds
from claims_triage.train import baseline_classifier, gbm_classifier
from claims_triage.triage import assign_tier, reason_codes


@pytest.fixture(scope="module")
def small():
    return generate(n_policies=2_000, n_claims=3_000, seed=1)


def test_generation_is_deterministic():
    a = generate(500, 800, seed=3)[1]
    b = generate(500, 800, seed=3)[1]
    pd.testing.assert_frame_equal(a, b)


def test_tables_and_keys(small):
    policies, claims = small
    assert policies["policy_id"].is_unique
    assert claims["claim_id"].is_unique
    assert set(claims["policy_id"]) <= set(policies["policy_id"])
    assert set(POLICY_FEATURES) <= set(policies.columns)


def test_label_and_imbalance(small):
    _, claims = small
    assert ((claims["incurred_12m"] >= LARGE_LOSS_THRESHOLD) == (claims["large_loss"] == 1)).all()
    assert 0.03 < claims["large_loss"].mean() < 0.2


def test_planted_missing_values(small):
    policies, claims = small
    assert policies["annual_revenue"].isna().mean() > 0.03
    assert claims["cause_of_loss"].isna().any()


def test_outlier_bounds():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 100])
    lo, hi = iqr_bounds(s)
    assert hi < 100 and lo < 1
    lo2, hi2 = std_bounds(s, 1)
    assert lo2 < s.mean() < hi2


def test_grid_model_count_matches_exam_example():
    grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": [0.01, 0.1, 1]}
    assert count_grid_models(grid, folds=5, refit=False) == 90
    assert count_grid_models(grid, folds=5, refit=True) == 91


def test_regression_metrics_exponentiate():
    y = np.array([1000.0, 10000.0])
    m = regression_metrics_from_log(y, np.log1p(y))
    assert m["rmse"] == pytest.approx(0, abs=1e-6)


def test_classification_metrics_and_gate():
    m = classification_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert 0 <= m["pr_auc"] <= 1 and m["roc_auc"] == pytest.approx(0.75)
    assert challenger_wins({"pr_auc": 0.60}, {"pr_auc": 0.55})
    assert not challenger_wins({"pr_auc": 0.552}, {"pr_auc": 0.55})
    assert challenger_wins({"pr_auc": 0.1}, None)


def test_tiers_and_reasons():
    assert assign_tier(0.7) == "HIGH" and assign_tier(0.3) == "ELEVATED" and assign_tier(0.05) == "STANDARD"
    row = pd.Series({"attorney_at_fnol": 1, "injury_flag": 0, "report_lag_days": 45, "prior_claims_3y": np.nan})
    assert reason_codes(row) == "ATTORNEY_AT_FNOL,LATE_REPORT"


@pytest.mark.parametrize("builder", [baseline_classifier, gbm_classifier])
def test_pipelines_fit_with_missing_values(small, builder):
    policies, claims = small
    df = claims.merge(policies, on="policy_id")
    model = builder().fit(df[ALL_FEATURES], df["large_loss"])
    proba = model.predict_proba(df[ALL_FEATURES].head(50))[:, 1]
    assert proba.shape == (50,) and ((proba >= 0) & (proba <= 1)).all()


def test_split_is_deterministic_and_balanced():
    from claims_triage.splits import split_for
    ids = [f"CLM-{i:06d}" for i in range(1, 20001)]
    s = pd.Series([split_for(i) for i in ids]).value_counts(normalize=True)
    assert split_for("CLM-000123") == split_for("CLM-000123")
    assert 0.66 < s["train"] < 0.74 and 0.12 < s["val"] < 0.18 and 0.12 < s["test"] < 0.18


def test_streaming_pipeline_feature_list_matches_package():
    import ast
    import pathlib
    src = pathlib.Path(__file__).parents[1] / "src" / "pipelines" / "streaming_triage.py"
    tree = ast.parse(src.read_text())
    features = next(ast.literal_eval(n.value) for n in ast.walk(tree)
                    if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "FEATURES")
    assert features == ALL_FEATURES
