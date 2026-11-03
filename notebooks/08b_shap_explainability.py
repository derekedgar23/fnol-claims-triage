# Databricks notebook source
# MAGIC %md
# MAGIC # 08b - SHAP Explainability for the Large-Loss Classifier
# MAGIC Why did the model flag this claim? SHAP (SHapley Additive exPlanations) decomposes each
# MAGIC prediction into per-feature contributions, giving adjusters and claims managers a transparent
# MAGIC view of what drives the triage tier.
# MAGIC
# MAGIC This notebook covers:
# MAGIC - Global feature importance (which features matter most across all claims)
# MAGIC - Dependence plots (how a single feature shifts predictions)
# MAGIC - Per-claim waterfall explanations (why *this* claim scored HIGH)

# COMMAND ----------

# MAGIC %pip install -q "shap>=0.43" "scikit-learn>=1.4" "databricks-feature-engineering>=0.12"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
import shap
from databricks.feature_engineering import FeatureEngineeringClient
from matplotlib import pyplot as plt

from claims_triage.tracking import load_training_data
from claims_triage.features import ALL_FEATURES, CATEGORICAL
from claims_triage.triage import TIER_THRESHOLDS, assign_tier

fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the champion model and validation data

# COMMAND ----------

data = load_training_data(spark, fe, FQ, FEATURE_TABLE)

client = mlflow.tracking.MlflowClient()
champion_version = client.get_model_version_by_alias(MODEL_NAME, "champion")
champion_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")

print(f"Champion model: {MODEL_NAME}@champion (run {champion_version.run_id})")
print(f"Validation set: {len(data.X_val)} claims, large-loss rate {data.y_val.mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build SHAP explainer
# MAGIC We use `TreeExplainer` for the GBM pipeline. The preprocessor transforms raw features into the
# MAGIC model's internal representation, then SHAP attributes contributions back to those transformed
# MAGIC features.

# COMMAND ----------

preprocessor = champion_model.named_steps["prep"]
tree_model = champion_model.named_steps["model"]

X_val_transformed = preprocessor.transform(data.X_val)

# Get feature names after preprocessing
try:
    feature_names = preprocessor.get_feature_names_out()
except AttributeError:
    feature_names = ALL_FEATURES

explainer = shap.TreeExplainer(tree_model)
shap_values = explainer.shap_values(X_val_transformed)

# For binary classifiers, shap_values may be a list [class_0, class_1]
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # class 1 = large loss

print(f"SHAP matrix: {shap_values.shape[0]} claims x {shap_values.shape[1]} features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Global feature importance
# MAGIC Mean absolute SHAP value per feature: which features shift predictions the most, on average.

# COMMAND ----------

fig, ax = plt.subplots(figsize=(9, 6))
shap.summary_plot(shap_values, X_val_transformed, feature_names=list(feature_names),
                  plot_type="bar", show=False, max_display=15)
plt.title("Global Feature Importance (mean |SHAP value|)")
plt.tight_layout()
display(fig)

with mlflow.start_run(run_name="shap_global_importance"):
    mlflow.log_figure(fig, "shap_global_importance.png")
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## SHAP beeswarm plot
# MAGIC Each dot is one claim. Color is the feature value (red = high, blue = low). Position on the
# MAGIC x-axis is the SHAP value (positive = pushes prediction toward large loss).

# COMMAND ----------

fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, X_val_transformed, feature_names=list(feature_names),
                  show=False, max_display=15)
plt.title("SHAP Beeswarm: Feature Impact on Large-Loss Prediction")
plt.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dependence plots for key features
# MAGIC How does a single feature's value relate to its SHAP contribution? SHAP auto-selects an
# MAGIC interaction feature for the color axis.

# COMMAND ----------

# Map readable names to column indices
name_to_idx = {str(n): i for i, n in enumerate(feature_names)}

key_features = ["attorney_at_fnol", "injury_flag", "report_lag_days", "prior_claims_3y"]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, feat_name in zip(axes.ravel(), key_features):
    # Find the column index matching (or containing) this feature name
    idx = None
    for col_name, col_idx in name_to_idx.items():
        if feat_name in col_name:
            idx = col_idx
            break
    if idx is not None:
        shap.dependence_plot(idx, shap_values, X_val_transformed,
                             feature_names=list(feature_names), ax=ax, show=False)

plt.suptitle("SHAP Dependence Plots for Key FNOL Features", fontsize=14, y=1.02)
plt.tight_layout()
display(fig)

with mlflow.start_run(run_name="shap_dependence"):
    mlflow.log_figure(fig, "shap_dependence_plots.png")
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Per-claim waterfall: why was this claim flagged HIGH?
# MAGIC Pick a HIGH-tier claim from the validation set and show the SHAP waterfall: the base value
# MAGIC (average prediction) plus each feature's push toward or away from "large loss".

# COMMAND ----------

proba = champion_model.predict_proba(data.X_val)[:, 1]
tiers = pd.Series([assign_tier(p) for p in proba], index=data.X_val.index)

high_mask = tiers == "HIGH"
print(f"HIGH tier claims in validation: {high_mask.sum()} / {len(tiers)}")

# Pick the first HIGH-tier claim
high_idx = data.X_val.index[high_mask][0]
claim_pos = list(data.X_val.index).index(high_idx)

print(f"\nClaim index: {high_idx}")
print(f"Predicted probability: {proba[claim_pos]:.3f}")
print(f"Actual large_loss: {data.y_val.iloc[claim_pos]}")
print(f"\nRaw features:")
display(data.X_val.loc[[high_idx]])

# COMMAND ----------

explanation = shap.Explanation(
    values=shap_values[claim_pos],
    base_values=explainer.expected_value if not isinstance(explainer.expected_value, list)
                else explainer.expected_value[1],
    data=X_val_transformed[claim_pos] if hasattr(X_val_transformed, '__getitem__')
         else X_val_transformed[claim_pos:claim_pos+1].toarray().ravel(),
    feature_names=list(feature_names),
)

fig, ax = plt.subplots(figsize=(10, 6))
shap.waterfall_plot(explanation, max_display=12, show=False)
plt.title(f"SHAP Waterfall: Claim {high_idx} (P(large loss) = {proba[claim_pos]:.3f})")
plt.tight_layout()
display(fig)

with mlflow.start_run(run_name="shap_waterfall_example"):
    mlflow.log_figure(fig, "shap_waterfall_high_tier.png")
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Per-claim waterfall: a STANDARD-tier claim for comparison

# COMMAND ----------

std_mask = tiers == "STANDARD"
std_idx = data.X_val.index[std_mask][0]
std_pos = list(data.X_val.index).index(std_idx)

print(f"Claim index: {std_idx}")
print(f"Predicted probability: {proba[std_pos]:.3f}")
print(f"Actual large_loss: {data.y_val.iloc[std_pos]}")
display(data.X_val.loc[[std_idx]])

explanation_std = shap.Explanation(
    values=shap_values[std_pos],
    base_values=explainer.expected_value if not isinstance(explainer.expected_value, list)
                else explainer.expected_value[1],
    data=X_val_transformed[std_pos] if hasattr(X_val_transformed, '__getitem__')
         else X_val_transformed[std_pos:std_pos+1].toarray().ravel(),
    feature_names=list(feature_names),
)

fig, ax = plt.subplots(figsize=(10, 6))
shap.waterfall_plot(explanation_std, max_display=12, show=False)
plt.title(f"SHAP Waterfall: Claim {std_idx} (P(large loss) = {proba[std_pos]:.3f})")
plt.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary table: top SHAP drivers for all HIGH-tier claims
# MAGIC For each HIGH-tier claim, which feature contributed the most to the prediction?

# COMMAND ----------

high_positions = [list(data.X_val.index).index(idx) for idx in data.X_val.index[high_mask]]
high_shap = shap_values[high_positions]

top_driver_idx = np.argmax(np.abs(high_shap), axis=1)
top_drivers = [str(feature_names[i]) for i in top_driver_idx]

summary = pd.DataFrame({
    "claim_index": data.X_val.index[high_mask],
    "probability": proba[high_positions],
    "actual_large_loss": data.y_val.iloc[high_positions].values,
    "top_shap_driver": top_drivers,
    "top_shap_value": [high_shap[i, top_driver_idx[i]] for i in range(len(high_positions))],
})

print(f"Top SHAP drivers for {len(summary)} HIGH-tier claims:")
display(summary.head(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## What this means for the claims team
# MAGIC - **Global importance** tells management which data fields matter most for triage accuracy.
# MAGIC   If attorney and injury flags dominate, that validates existing adjuster intuition.
# MAGIC - **Dependence plots** show non-linear effects (e.g., report lag between 15-60 days raises
# MAGIC   risk more than very long lags, which are often administrative).
# MAGIC - **Waterfall plots** give adjusters a per-claim explanation they can attach to the file:
# MAGIC   "This claim scored HIGH because attorney involvement (+0.8) and injury flag (+0.5) both
# MAGIC   pushed the prediction well above average."
# MAGIC - These explanations support regulatory and audit requirements around model transparency.
# MAGIC
# MAGIC Next: `09_model_monitoring` (drift detection and performance tracking over time).
