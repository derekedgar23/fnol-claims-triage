# Databricks notebook source
# MAGIC %md
# MAGIC # 09 - Model Monitoring: Drift Detection and Performance Tracking
# MAGIC A deployed model degrades when the data it sees in production drifts from what it trained on.
# MAGIC This notebook simulates a monthly monitoring cadence:
# MAGIC
# MAGIC 1. **Feature drift** - are input distributions shifting? (Population Stability Index, KS test)
# MAGIC 2. **Prediction drift** - is the model's output distribution changing?
# MAGIC 3. **Performance decay** - when labels arrive, are metrics still holding?
# MAGIC 4. **Tier distribution shift** - are more claims landing in HIGH than expected?
# MAGIC
# MAGIC In production, these checks run as a scheduled Databricks job and write metrics to a Delta
# MAGIC table. Alerts fire when PSI or metric degradation crosses a threshold.

# COMMAND ----------

# MAGIC %pip install -q "scikit-learn>=1.4" "databricks-feature-engineering>=0.12"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
from databricks.feature_engineering import FeatureEngineeringClient
from matplotlib import pyplot as plt
from scipy import stats

from claims_triage.data import generate_claims, generate_policies
from claims_triage.features import ALL_FEATURES, CATEGORICAL, NUMERIC
from claims_triage.splits import split_for
from claims_triage.triage import TIER_THRESHOLDS, assign_tier
from claims_triage.train import feature_frame

fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load champion model and reference (training) data

# COMMAND ----------

champion_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")

# Reference: training split from the feature table
ref_pdf = (spark.table(f"{FQ}.claims_training_base")
           .join(spark.table(FEATURE_TABLE), "policy_id")
           .toPandas())
ref_pdf["split"] = ref_pdf["claim_id"].apply(split_for)
ref_train = ref_pdf[ref_pdf["split"] == "train"].copy()
X_ref = feature_frame(ref_train)

ref_proba = champion_model.predict_proba(X_ref)[:, 1]
ref_train["predicted_proba"] = ref_proba
ref_train["predicted_tier"] = [assign_tier(p) for p in ref_proba]

print(f"Reference (training) set: {len(ref_train)} claims")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simulate monthly production scoring windows
# MAGIC We split the data by `loss_date` to create monthly cohorts, simulating how claims arrive
# MAGIC over time. In production, this would read from the scored claims Delta table.

# COMMAND ----------

scored = ref_pdf.copy()
X_all = feature_frame(scored)
scored["predicted_proba"] = champion_model.predict_proba(X_all)[:, 1]
scored["predicted_tier"] = [assign_tier(p) for p in scored["predicted_proba"]]
scored["loss_date"] = pd.to_datetime(scored["loss_date"])
scored["loss_month"] = scored["loss_date"].dt.to_period("M")

months = sorted(scored["loss_month"].unique())
print(f"Monitoring window: {months[0]} to {months[-1]} ({len(months)} months)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature drift: Population Stability Index (PSI)
# MAGIC PSI measures how much a feature's distribution has shifted from the reference period.
# MAGIC - PSI < 0.10: no significant shift
# MAGIC - 0.10 - 0.25: moderate shift, investigate
# MAGIC - PSI > 0.25: significant shift, retrain candidate
# MAGIC
# MAGIC We compute PSI for each numeric feature per month.

# COMMAND ----------

def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two distributions."""
    ref_clean = reference[~np.isnan(reference)]
    cur_clean = current[~np.isnan(current)]
    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return 0.0

    breakpoints = np.percentile(ref_clean, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    ref_counts = np.histogram(ref_clean, bins=breakpoints)[0] + 1  # smoothing
    cur_counts = np.histogram(cur_clean, bins=breakpoints)[0] + 1

    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


# Compute PSI for each numeric feature, for each month vs training reference
psi_records = []
for month in months:
    month_data = scored[scored["loss_month"] == month]
    X_month = feature_frame(month_data)
    for feat in NUMERIC:
        psi_val = compute_psi(X_ref[feat].values, X_month[feat].values)
        psi_records.append({"month": str(month), "feature": feat, "psi": round(psi_val, 4)})

psi_df = pd.DataFrame(psi_records)

# Pivot for display
psi_pivot = psi_df.pivot(index="feature", columns="month", values="psi")
print("PSI by feature and month (> 0.25 = significant drift):")
display(psi_pivot.style.applymap(lambda v: "background-color: #ffcccc" if v > 0.25
                                  else ("background-color: #ffffcc" if v > 0.10 else "")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature drift: Kolmogorov-Smirnov test
# MAGIC A complementary statistical test. The KS statistic measures the maximum distance between
# MAGIC two cumulative distributions. p-value < 0.05 suggests the distributions differ.

# COMMAND ----------

ks_records = []
for month in months[-6:]:  # last 6 months for brevity
    month_data = scored[scored["loss_month"] == month]
    X_month = feature_frame(month_data)
    for feat in NUMERIC:
        ref_vals = X_ref[feat].dropna().values
        cur_vals = X_month[feat].dropna().values
        if len(cur_vals) > 0:
            ks_stat, p_val = stats.ks_2samp(ref_vals, cur_vals)
            ks_records.append({"month": str(month), "feature": feat,
                               "ks_stat": round(ks_stat, 4), "p_value": round(p_val, 4)})

ks_df = pd.DataFrame(ks_records)
print("KS test results (last 6 months, p < 0.05 = significant):")
display(ks_df[ks_df["p_value"] < 0.05].sort_values(["month", "ks_stat"], ascending=[True, False]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prediction drift: score distribution over time
# MAGIC If the model's output distribution shifts, either the data changed or the model is stale.

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of predicted probabilities: reference vs recent months
axes[0].hist(ref_proba, bins=50, alpha=0.5, density=True, label="Reference (train)")
for month in months[-3:]:
    month_proba = scored[scored["loss_month"] == month]["predicted_proba"]
    axes[0].hist(month_proba, bins=50, alpha=0.3, density=True, label=str(month))
axes[0].set_xlabel("Predicted P(large loss)")
axes[0].set_ylabel("Density")
axes[0].set_title("Score Distribution: Reference vs Recent")
axes[0].legend()

# Monthly mean predicted probability
monthly_mean = scored.groupby("loss_month")["predicted_proba"].mean()
axes[1].plot(range(len(monthly_mean)), monthly_mean.values, marker="o")
axes[1].axhline(ref_proba.mean(), ls="--", c="red", label=f"Reference mean ({ref_proba.mean():.3f})")
axes[1].set_xlabel("Month index")
axes[1].set_ylabel("Mean predicted probability")
axes[1].set_title("Mean Score Over Time")
axes[1].legend()

plt.tight_layout()
display(fig)

with mlflow.start_run(run_name="monitoring_prediction_drift"):
    mlflow.log_figure(fig, "prediction_drift.png")
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Performance decay: monthly metrics (when labels arrive)
# MAGIC In practice, the `large_loss` label is known only after 12 months of development. This
# MAGIC simulates computing metrics on each monthly cohort once labels are available.

# COMMAND ----------

from claims_triage.evaluation import classification_metrics

monthly_metrics = []
for month in months:
    cohort = scored[scored["loss_month"] == month]
    if len(cohort) < 50 or cohort["large_loss"].nunique() < 2:
        continue
    m = classification_metrics(cohort["large_loss"], cohort["predicted_proba"])
    m["month"] = str(month)
    m["n_claims"] = len(cohort)
    monthly_metrics.append(m)

metrics_df = pd.DataFrame(monthly_metrics)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, metric, color in zip(axes, ["pr_auc", "roc_auc", "precision"], ["#2196F3", "#4CAF50", "#FF9800"]):
    ax.plot(range(len(metrics_df)), metrics_df[metric], marker="o", color=color)
    ref_val = metrics_df[metric].iloc[:3].mean()  # first 3 months as baseline
    ax.axhline(ref_val, ls="--", c="red", alpha=0.7, label=f"Baseline ({ref_val:.3f})")
    ax.fill_between(range(len(metrics_df)),
                    ref_val - 0.02, ref_val + 0.02, alpha=0.1, color="red")
    ax.set_xlabel("Month index")
    ax.set_ylabel(metric)
    ax.set_title(f"Monthly {metric}")
    ax.legend()

plt.suptitle("Performance Tracking Over Time", fontsize=14)
plt.tight_layout()
display(fig)

with mlflow.start_run(run_name="monitoring_performance_decay"):
    mlflow.log_figure(fig, "performance_decay.png")
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tier distribution shift
# MAGIC If more claims are landing in HIGH than the training period, it could signal data drift,
# MAGIC a change in book mix, or a real shift in loss severity.

# COMMAND ----------

ref_tier_dist = ref_train["predicted_tier"].value_counts(normalize=True).sort_index()

tier_by_month = (scored.groupby("loss_month")["predicted_tier"]
                 .value_counts(normalize=True)
                 .unstack(fill_value=0)
                 .sort_index())

fig, ax = plt.subplots(figsize=(12, 5))
tier_by_month.plot(kind="bar", stacked=True, ax=ax, color=["#4CAF50", "#FF9800", "#F44336"])
ax.set_xlabel("Month")
ax.set_ylabel("Share of claims")
ax.set_title("Triage Tier Distribution Over Time")
ax.legend(title="Tier")
# Add reference lines
for tier, color in [("HIGH", "#F44336"), ("ELEVATED", "#FF9800")]:
    if tier in ref_tier_dist.index:
        ax.axhline(ref_tier_dist.loc[tier] + ref_tier_dist.get("STANDARD", 0)
                    if tier == "ELEVATED" else ref_tier_dist.get(tier, 0),
                    ls="--", c=color, alpha=0.5)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitoring summary table
# MAGIC A single-row-per-month view for the monitoring dashboard or alerting job.

# COMMAND ----------

summary_rows = []
for month in months:
    cohort = scored[scored["loss_month"] == month]
    X_month = feature_frame(cohort)
    row = {"month": str(month), "n_claims": len(cohort)}

    # Mean PSI across numeric features
    psi_vals = [compute_psi(X_ref[f].values, X_month[f].values) for f in NUMERIC]
    row["mean_psi"] = round(np.mean(psi_vals), 4)
    row["max_psi"] = round(np.max(psi_vals), 4)
    row["max_psi_feature"] = NUMERIC[np.argmax(psi_vals)]

    # Score drift
    row["mean_score"] = round(cohort["predicted_proba"].mean(), 4)
    row["score_psi"] = round(compute_psi(ref_proba, cohort["predicted_proba"].values), 4)

    # Tier shares
    tier_counts = cohort["predicted_tier"].value_counts(normalize=True)
    row["pct_high"] = round(tier_counts.get("HIGH", 0), 4)
    row["pct_elevated"] = round(tier_counts.get("ELEVATED", 0), 4)

    # Performance (if labels available)
    if cohort["large_loss"].nunique() >= 2 and len(cohort) >= 50:
        m = classification_metrics(cohort["large_loss"], cohort["predicted_proba"])
        row["pr_auc"] = round(m["pr_auc"], 4)
        row["roc_auc"] = round(m["roc_auc"], 4)

    summary_rows.append(row)

monitoring_df = pd.DataFrame(summary_rows)
print("Monthly monitoring summary:")
display(monitoring_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Alerting thresholds
# MAGIC In a production deployment, the monitoring job writes `monitoring_df` to a Delta table and an
# MAGIC alert fires when:
# MAGIC
# MAGIC | Condition | Action |
# MAGIC |---|---|
# MAGIC | Any feature PSI > 0.25 | Investigate feature; check upstream data pipeline |
# MAGIC | Score PSI > 0.20 | Compare input distributions; consider retraining |
# MAGIC | PR AUC drops > 0.03 from baseline | Trigger champion/challenger re-evaluation |
# MAGIC | HIGH tier share > 2x training rate | Check for book mix change or data quality issue |
# MAGIC
# MAGIC These thresholds are starting points. Calibrate them against your claims team's tolerance for
# MAGIC false alerts vs missed degradation.
# MAGIC
# MAGIC Next: `10_triage_dashboard` (Streamlit app for claims triage visualization).
