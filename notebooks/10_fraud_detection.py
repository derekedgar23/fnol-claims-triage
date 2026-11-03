# Databricks notebook source
# MAGIC %md
# MAGIC # 10 - Fraud Detection: Supervised and Unsupervised Approaches
# MAGIC Claims fraud costs the P&C industry tens of billions annually. Catching it early -- ideally at
# MAGIC FNOL or during the first adjuster review -- saves investigation costs and reduces paid losses.
# MAGIC
# MAGIC This notebook builds two complementary fraud detection approaches:
# MAGIC
# MAGIC 1. **Supervised GBM classifier** trained on labeled fraud data (when labels exist)
# MAGIC 2. **Unsupervised anomaly detection** using Isolation Forest and Local Outlier Factor
# MAGIC    (when labels are scarce or unavailable)
# MAGIC
# MAGIC In practice, fraud labels are rare and delayed -- SIU investigations take months, and many
# MAGIC fraudulent claims are never confirmed. Unsupervised methods fill this gap by surfacing
# MAGIC statistically unusual claims for human review without requiring historical labels.
# MAGIC
# MAGIC | Approach | Strengths | Limitations |
# MAGIC |---|---|---|
# MAGIC | Supervised (GBM) | High precision when labels are reliable | Needs labeled data; misses novel schemes |
# MAGIC | Isolation Forest | Finds global outliers; fast; no labels needed | No fraud-specific signal; may flag legitimate edge cases |
# MAGIC | Local Outlier Factor | Finds local density anomalies; catches cluster-edge fraud | Sensitive to neighbor count; slower on large data |

# COMMAND ----------

# MAGIC %pip install -q "scikit-learn>=1.4" "shap>=0.43" "databricks-feature-engineering>=0.12"

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
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    roc_auc_score,
    RocCurveDisplay,
)

from claims_triage.data import generate
from claims_triage.features import ALL_FEATURES, CATEGORICAL, NUMERIC
from claims_triage.fraud import (
    anomaly_scores,
    fraud_classifier,
    isolation_forest,
    local_outlier_factor,
)
from claims_triage.splits import split_for
from claims_triage.train import feature_frame

fe = FeatureEngineeringClient()
mlflow.sklearn.autolog(disable=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data
# MAGIC The synthetic generator produces a `is_fraud` column (~2.5% base rate) driven by
# MAGIC realistic risk factors: attorney involvement on new businesses, injury claims on
# MAGIC non-injury lines, delayed reporting, and suspiciously round amounts.

# COMMAND ----------

policies, claims = generate()
merged = claims.merge(policies, on="policy_id", how="left")
merged["split"] = merged["claim_id"].apply(split_for)

train = merged[merged["split"] == "train"].copy()
val = merged[merged["split"] == "val"].copy()
test = merged[merged["split"] == "test"].copy()

X_train, X_val, X_test = feature_frame(train), feature_frame(val), feature_frame(test)
y_train, y_val, y_test = train["is_fraud"], val["is_fraud"], test["is_fraud"]

print(f"Train: {len(train):,} claims | Val: {len(val):,} | Test: {len(test):,}")
print(f"Fraud rate -- train: {y_train.mean():.3f} | val: {y_val.mean():.3f} | test: {y_test.mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 1: Unsupervised Anomaly Detection
# MAGIC No labels required. The model learns what "normal" looks like and flags deviations.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Isolation Forest
# MAGIC Isolation Forest works by randomly selecting a feature and a split value, then counting
# MAGIC how many splits it takes to isolate each data point. Anomalies are isolated in fewer splits
# MAGIC because they sit in sparse regions of the feature space.
# MAGIC
# MAGIC Key intuition: a fraudulent claim with an unusual combination of features (e.g., attorney
# MAGIC involved + new business + non-injury line claiming injury) will be easier to separate from
# MAGIC the bulk of normal claims.

# COMMAND ----------

iso_model = isolation_forest(contamination=0.025, n_estimators=300)
iso_model.fit(X_train)

iso_train_scores = anomaly_scores(iso_model, X_train)
iso_val_scores = anomaly_scores(iso_model, X_val)
iso_test_scores = anomaly_scores(iso_model, X_test)

# Flag top 2.5% as anomalies
iso_threshold = np.percentile(iso_train_scores, 97.5)
iso_val_pred = (iso_val_scores >= iso_threshold).astype(int)

print(f"Isolation Forest anomaly threshold: {iso_threshold:.4f}")
print(f"Anomalies flagged in validation: {iso_val_pred.sum()} / {len(iso_val_pred)} "
      f"({iso_val_pred.mean():.1%})")

with mlflow.start_run(run_name="isolation_forest_fraud"):
    mlflow.log_params({"model": "IsolationForest", "contamination": 0.025,
                        "n_estimators": 300, "approach": "unsupervised"})
    mlflow.log_metrics({
        "val_roc_auc": roc_auc_score(y_val, iso_val_scores),
        "val_pr_auc": average_precision_score(y_val, iso_val_scores),
        "val_anomalies_flagged": int(iso_val_pred.sum()),
        "val_fraud_in_anomalies": int((iso_val_pred & y_val).sum()),
    })

# COMMAND ----------

# MAGIC %md
# MAGIC ### How well does Isolation Forest find fraud?
# MAGIC Since we have labels (synthetic), we can evaluate. In production without labels, you would
# MAGIC review the top-scored claims manually and track the SIU referral hit rate.

# COMMAND ----------

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Score distribution by fraud status
axes[0].hist(iso_val_scores[y_val == 0], bins=50, alpha=0.5, density=True, label="Legitimate", color="#4CAF50")
axes[0].hist(iso_val_scores[y_val == 1], bins=50, alpha=0.5, density=True, label="Fraud", color="#F44336")
axes[0].axvline(iso_threshold, ls="--", c="black", label=f"Threshold ({iso_threshold:.2f})")
axes[0].set_xlabel("Anomaly Score (higher = more anomalous)")
axes[0].set_ylabel("Density")
axes[0].set_title("Isolation Forest: Score Distribution")
axes[0].legend()

# 2. ROC curve
RocCurveDisplay.from_predictions(y_val, iso_val_scores, ax=axes[1], name="Isolation Forest")
axes[1].set_title("ROC Curve (Unsupervised)")
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3)

# 3. Precision-Recall curve
PrecisionRecallDisplay.from_predictions(y_val, iso_val_scores, ax=axes[2], name="Isolation Forest")
axes[2].axhline(y_val.mean(), ls="--", c="grey", label=f"Base rate ({y_val.mean():.3f})")
axes[2].set_title("Precision-Recall Curve (Unsupervised)")
axes[2].legend()

plt.suptitle("Isolation Forest Anomaly Detection", fontsize=14, y=1.02)
plt.tight_layout()
display(fig)

with mlflow.start_run(run_name="isolation_forest_evaluation"):
    mlflow.log_figure(fig, "isolation_forest_evaluation.png")
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Enrichment analysis: are anomalies enriched for fraud?
# MAGIC Even if the unsupervised model does not achieve high precision, what matters is whether
# MAGIC the flagged population has a meaningfully higher fraud rate than the general population.

# COMMAND ----------

val_with_scores = val.copy()
val_with_scores["iso_anomaly_score"] = iso_val_scores
val_with_scores["iso_flagged"] = iso_val_pred

# Enrichment by decile
val_with_scores["score_decile"] = pd.qcut(iso_val_scores, 10, labels=False, duplicates="drop") + 1
enrichment = (val_with_scores.groupby("score_decile")
              .agg(claims=("is_fraud", "size"),
                   frauds=("is_fraud", "sum"))
              .assign(fraud_rate=lambda d: d["frauds"] / d["claims"],
                      lift=lambda d: (d["frauds"] / d["claims"]) / y_val.mean()))

print("Fraud enrichment by anomaly score decile (10 = most anomalous):")
display(enrichment.round(3))

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(enrichment.index, enrichment["lift"], color="#2196F3")
ax.axhline(1.0, ls="--", c="red", label="No lift (random)")
ax.set_xlabel("Anomaly Score Decile (10 = most anomalous)")
ax.set_ylabel("Lift over base fraud rate")
ax.set_title("Isolation Forest: Fraud Enrichment by Decile")
ax.legend()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Local Outlier Factor (LOF)
# MAGIC LOF measures the local density deviation of a data point relative to its neighbors.
# MAGIC A point surrounded by dense clusters but sitting on the edge (lower local density) gets
# MAGIC a high LOF score. This catches fraud patterns that are locally unusual even if they
# MAGIC are not global outliers.

# COMMAND ----------

lof_model = local_outlier_factor(contamination=0.025, n_neighbors=20)
lof_model.fit(X_train)

lof_val_scores = anomaly_scores(lof_model, X_val)

lof_roc = roc_auc_score(y_val, lof_val_scores)
lof_pr = average_precision_score(y_val, lof_val_scores)

print(f"LOF validation -- ROC AUC: {lof_roc:.3f} | PR AUC: {lof_pr:.3f}")

with mlflow.start_run(run_name="lof_fraud"):
    mlflow.log_params({"model": "LocalOutlierFactor", "contamination": 0.025,
                        "n_neighbors": 20, "approach": "unsupervised"})
    mlflow.log_metrics({"val_roc_auc": lof_roc, "val_pr_auc": lof_pr})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensemble: combining Isolation Forest and LOF
# MAGIC Averaging the anomaly scores from both models creates a more robust signal. Claims flagged
# MAGIC by both methods are stronger candidates for SIU referral.

# COMMAND ----------

ensemble_scores = (iso_val_scores + lof_val_scores) / 2
ensemble_roc = roc_auc_score(y_val, ensemble_scores)
ensemble_pr = average_precision_score(y_val, ensemble_scores)

print(f"Ensemble (IF + LOF) -- ROC AUC: {ensemble_roc:.3f} | PR AUC: {ensemble_pr:.3f}")

# Agreement analysis
iso_flags = (iso_val_scores >= np.percentile(iso_val_scores, 97.5)).astype(int)
lof_flags = (lof_val_scores >= np.percentile(lof_val_scores, 97.5)).astype(int)
both_flagged = iso_flags & lof_flags

print(f"\nBoth models flag: {both_flagged.sum()} claims")
if both_flagged.sum() > 0:
    print(f"Fraud rate in dual-flagged: {y_val[both_flagged == 1].mean():.1%}")
print(f"Fraud rate in general population: {y_val.mean():.1%}")

with mlflow.start_run(run_name="unsupervised_ensemble_fraud"):
    mlflow.log_params({"model": "IF+LOF_ensemble", "approach": "unsupervised"})
    mlflow.log_metrics({"val_roc_auc": ensemble_roc, "val_pr_auc": ensemble_pr,
                        "dual_flagged_count": int(both_flagged.sum()),
                        "dual_flagged_fraud_rate": float(y_val[both_flagged == 1].mean()) if both_flagged.sum() > 0 else 0.0})

# COMMAND ----------

# MAGIC %md
# MAGIC ### What makes an anomaly anomalous?
# MAGIC Profile the flagged claims vs the general population to understand what the unsupervised
# MAGIC models are picking up on.

# COMMAND ----------

val_with_scores["ensemble_score"] = ensemble_scores
val_with_scores["ensemble_flagged"] = (ensemble_scores >= np.percentile(ensemble_scores, 97.5)).astype(int)

flagged = val_with_scores[val_with_scores["ensemble_flagged"] == 1]
not_flagged = val_with_scores[val_with_scores["ensemble_flagged"] == 0]

profile = pd.DataFrame({
    "Flagged (anomalies)": {
        "count": len(flagged),
        "fraud_rate": flagged["is_fraud"].mean(),
        "attorney_rate": flagged["attorney_at_fnol"].mean(),
        "injury_rate": flagged["injury_flag"].mean(),
        "mean_report_lag": flagged["report_lag_days"].mean(),
        "mean_prior_claims": flagged["prior_claims_3y"].mean(),
        "mean_years_in_business": flagged["years_in_business"].mean(),
        "mean_incurred": flagged["incurred_12m"].mean(),
        "large_loss_rate": flagged["large_loss"].mean(),
    },
    "Not flagged": {
        "count": len(not_flagged),
        "fraud_rate": not_flagged["is_fraud"].mean(),
        "attorney_rate": not_flagged["attorney_at_fnol"].mean(),
        "injury_rate": not_flagged["injury_flag"].mean(),
        "mean_report_lag": not_flagged["report_lag_days"].mean(),
        "mean_prior_claims": not_flagged["prior_claims_3y"].mean(),
        "mean_years_in_business": not_flagged["years_in_business"].mean(),
        "mean_incurred": not_flagged["incurred_12m"].mean(),
        "large_loss_rate": not_flagged["large_loss"].mean(),
    },
}).round(4)

print("Profile: anomaly-flagged claims vs general population")
display(profile)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature distributions: flagged vs not flagged

# COMMAND ----------

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

plot_features = ["report_lag_days", "prior_claims_3y", "years_in_business",
                 "total_insured_value", "policy_limit", "deductible"]

for ax, feat in zip(axes.ravel(), plot_features):
    ax.hist(not_flagged[feat].dropna(), bins=40, alpha=0.5, density=True,
            label="Not flagged", color="#4CAF50")
    ax.hist(flagged[feat].dropna(), bins=40, alpha=0.5, density=True,
            label="Flagged", color="#F44336")
    ax.set_title(feat)
    ax.legend(fontsize=8)

plt.suptitle("Feature Distributions: Anomaly-Flagged vs Not Flagged", fontsize=14, y=1.02)
plt.tight_layout()
display(fig)

with mlflow.start_run(run_name="anomaly_profiles"):
    mlflow.log_figure(fig, "anomaly_feature_distributions.png")
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 2: Supervised Fraud Classifier
# MAGIC When labeled fraud data is available, a supervised model will outperform unsupervised
# MAGIC methods. The tradeoff: it can only catch patterns similar to historically confirmed fraud.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train the supervised GBM

# COMMAND ----------

# GBM uses balanced sample weights internally since fraud is rare
sup_model = fraud_classifier(class_weight="balanced")
sup_model.fit(X_train, y_train)

sup_val_proba = sup_model.predict_proba(X_val)[:, 1]
sup_test_proba = sup_model.predict_proba(X_test)[:, 1]

sup_val_roc = roc_auc_score(y_val, sup_val_proba)
sup_val_pr = average_precision_score(y_val, sup_val_proba)
sup_test_roc = roc_auc_score(y_test, sup_test_proba)
sup_test_pr = average_precision_score(y_test, sup_test_proba)

print(f"Supervised GBM -- Val ROC AUC: {sup_val_roc:.3f} | Val PR AUC: {sup_val_pr:.3f}")
print(f"                 Test ROC AUC: {sup_test_roc:.3f} | Test PR AUC: {sup_test_pr:.3f}")

with mlflow.start_run(run_name="supervised_fraud_gbm"):
    mlflow.log_params({"model": "GradientBoostingClassifier", "approach": "supervised",
                        "class_weight": "balanced"})
    mlflow.log_metrics({"val_roc_auc": sup_val_roc, "val_pr_auc": sup_val_pr,
                        "test_roc_auc": sup_test_roc, "test_pr_auc": sup_test_pr})
    mlflow.sklearn.log_model(sup_model, "fraud_classifier")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Supervised model evaluation

# COMMAND ----------

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Confusion matrix at 0.5 threshold
sup_val_pred = (sup_val_proba >= 0.5).astype(int)
ConfusionMatrixDisplay.from_predictions(y_val, sup_val_pred, ax=axes[0], colorbar=False)
axes[0].set_title("Confusion Matrix @ 0.5")

# ROC curve
RocCurveDisplay.from_predictions(y_val, sup_val_proba, ax=axes[1], name="Supervised GBM")
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3)
axes[1].set_title("ROC Curve")

# PR curve
PrecisionRecallDisplay.from_predictions(y_val, sup_val_proba, ax=axes[2], name="Supervised GBM")
axes[2].axhline(y_val.mean(), ls="--", c="grey", label=f"Base rate ({y_val.mean():.3f})")
axes[2].set_title("Precision-Recall Curve")
axes[2].legend()

plt.suptitle("Supervised Fraud Classifier", fontsize=14, y=1.02)
plt.tight_layout()
display(fig)

with mlflow.start_run(run_name="supervised_fraud_evaluation"):
    mlflow.log_figure(fig, "supervised_fraud_evaluation.png")
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature importance: what the supervised model learned

# COMMAND ----------

import shap

preprocessor = sup_model.named_steps["prep"]
tree_model = sup_model.named_steps["model"]

X_val_transformed = preprocessor.transform(X_val)
explainer = shap.TreeExplainer(tree_model)
shap_values = explainer.shap_values(X_val_transformed)

try:
    feature_names = list(preprocessor.get_feature_names_out())
except AttributeError:
    feature_names = ALL_FEATURES

fig, ax = plt.subplots(figsize=(9, 6))
shap.summary_plot(shap_values, X_val_transformed, feature_names=feature_names,
                  plot_type="bar", show=False, max_display=12)
plt.title("Supervised Fraud Model: Feature Importance (SHAP)")
plt.tight_layout()
display(fig)

with mlflow.start_run(run_name="supervised_fraud_shap"):
    mlflow.log_figure(fig, "fraud_shap_importance.png")
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Part 3: Head-to-Head Comparison
# MAGIC How do supervised and unsupervised approaches compare? Where does each excel?

# COMMAND ----------

# MAGIC %md
# MAGIC ## Side-by-side ROC and PR curves

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name, scores, color in [
    ("Supervised GBM", sup_val_proba, "#2196F3"),
    ("Isolation Forest", iso_val_scores, "#FF9800"),
    ("LOF", lof_val_scores, "#9C27B0"),
    ("IF + LOF Ensemble", ensemble_scores, "#F44336"),
]:
    RocCurveDisplay.from_predictions(y_val, scores, ax=axes[0], name=name)
    PrecisionRecallDisplay.from_predictions(y_val, scores, ax=axes[1], name=name)

axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
axes[0].set_title("ROC Curves: All Models")
axes[1].axhline(y_val.mean(), ls="--", c="grey", label=f"Base rate ({y_val.mean():.3f})")
axes[1].set_title("Precision-Recall Curves: All Models")
axes[1].legend(fontsize=8)

plt.tight_layout()
display(fig)

with mlflow.start_run(run_name="fraud_model_comparison"):
    mlflow.log_figure(fig, "fraud_model_comparison.png")
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary metrics table

# COMMAND ----------

comparison = pd.DataFrame({
    "Supervised GBM": {
        "approach": "supervised",
        "ROC AUC": sup_val_roc,
        "PR AUC": sup_val_pr,
        "requires_labels": True,
        "catches_novel_fraud": False,
    },
    "Isolation Forest": {
        "approach": "unsupervised",
        "ROC AUC": roc_auc_score(y_val, iso_val_scores),
        "PR AUC": average_precision_score(y_val, iso_val_scores),
        "requires_labels": False,
        "catches_novel_fraud": True,
    },
    "LOF": {
        "approach": "unsupervised",
        "ROC AUC": roc_auc_score(y_val, lof_val_scores),
        "PR AUC": average_precision_score(y_val, lof_val_scores),
        "requires_labels": False,
        "catches_novel_fraud": True,
    },
    "IF + LOF Ensemble": {
        "approach": "unsupervised",
        "ROC AUC": ensemble_roc,
        "PR AUC": ensemble_pr,
        "requires_labels": False,
        "catches_novel_fraud": True,
    },
}).T

display(comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Combined scoring: supervised + unsupervised
# MAGIC The most robust production setup uses both: the supervised model catches known fraud patterns,
# MAGIC while anomaly scores flag novel schemes the classifier has never seen. Claims scoring high on
# MAGIC both warrant priority SIU review.

# COMMAND ----------

combined = val.copy()
combined["supervised_score"] = sup_val_proba
combined["anomaly_score"] = ensemble_scores

# Quadrant analysis
sup_thresh = 0.3   # lower than 0.5 to catch more candidates for review
anom_thresh = np.percentile(ensemble_scores, 95)  # top 5%

combined["quadrant"] = "Low risk"
combined.loc[(combined["supervised_score"] >= sup_thresh) &
             (combined["anomaly_score"] < anom_thresh), "quadrant"] = "Known patterns"
combined.loc[(combined["supervised_score"] < sup_thresh) &
             (combined["anomaly_score"] >= anom_thresh), "quadrant"] = "Novel anomalies"
combined.loc[(combined["supervised_score"] >= sup_thresh) &
             (combined["anomaly_score"] >= anom_thresh), "quadrant"] = "Priority SIU"

quadrant_summary = (combined.groupby("quadrant")
                    .agg(claims=("is_fraud", "size"),
                         frauds=("is_fraud", "sum"))
                    .assign(fraud_rate=lambda d: d["frauds"] / d["claims"],
                            share_of_fraud=lambda d: d["frauds"] / d["frauds"].sum()))

print("Combined scoring quadrant analysis:")
display(quadrant_summary.round(4))

# COMMAND ----------

fig, ax = plt.subplots(figsize=(9, 7))

colors = {"Low risk": "#4CAF50", "Known patterns": "#2196F3",
          "Novel anomalies": "#FF9800", "Priority SIU": "#F44336"}

for quad, color in colors.items():
    mask = combined["quadrant"] == quad
    fraud_mask = combined["is_fraud"] == 1
    # Plot non-fraud
    ax.scatter(combined.loc[mask & ~fraud_mask, "supervised_score"],
               combined.loc[mask & ~fraud_mask, "anomaly_score"],
               alpha=0.15, s=10, c=color, label=f"{quad} (legit)")
    # Plot fraud with markers
    ax.scatter(combined.loc[mask & fraud_mask, "supervised_score"],
               combined.loc[mask & fraud_mask, "anomaly_score"],
               alpha=0.8, s=30, c=color, marker="x", label=f"{quad} (fraud)")

ax.axvline(sup_thresh, ls="--", c="grey", alpha=0.5)
ax.axhline(anom_thresh, ls="--", c="grey", alpha=0.5)
ax.set_xlabel("Supervised Score (P(fraud))")
ax.set_ylabel("Anomaly Score (higher = more anomalous)")
ax.set_title("Combined Fraud Detection: Supervised vs Unsupervised")
ax.legend(fontsize=7, loc="upper left", ncol=2)

plt.tight_layout()
display(fig)

with mlflow.start_run(run_name="combined_fraud_quadrant"):
    mlflow.log_figure(fig, "fraud_quadrant_analysis.png")
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Takeaways
# MAGIC
# MAGIC **Unsupervised methods (Isolation Forest + LOF):**
# MAGIC - Work without any labeled fraud data, making them deployable on day one
# MAGIC - The anomaly-flagged population shows a meaningfully higher fraud rate than the base
# MAGIC   population (see the enrichment/lift analysis above)
# MAGIC - Best used to generate an SIU referral queue ranked by anomaly score
# MAGIC - Claims flagged by both IF and LOF are strongest candidates
# MAGIC
# MAGIC **Supervised GBM:**
# MAGIC - Higher overall discrimination when labels are available
# MAGIC - Learns the specific fraud signatures in your historical data
# MAGIC - Blind spot: cannot catch fraud schemes it has never seen in training
# MAGIC
# MAGIC **Combined approach (recommended for production):**
# MAGIC - **Priority SIU**: high on both scores -- strong fraud signal from multiple angles
# MAGIC - **Known patterns**: high supervised, low anomaly -- matches historical fraud profiles
# MAGIC - **Novel anomalies**: low supervised, high anomaly -- statistically unusual claims that
# MAGIC   the classifier missed, worth investigating as potential new schemes
# MAGIC - Track SIU outcomes to continuously improve both models
# MAGIC
# MAGIC This mirrors how fraud teams actually operate: the supervised model handles the bulk of
# MAGIC referrals for known patterns, while anomaly detection serves as a safety net for emerging
# MAGIC fraud schemes.
