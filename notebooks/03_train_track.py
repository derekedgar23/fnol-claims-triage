# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Train and track the large-loss classifier
# MAGIC **Exam Sections 1 and 3.** Build a training set from the feature table, train three candidates, and log
# MAGIC parameters, metrics, figures and models to MLflow. Covers class imbalance, metric choice, and finding the best run.
# MAGIC
# MAGIC | Candidate | Why it is here |
# MAGIC |---|---|
# MAGIC | Logistic regression, balanced class weights | Interpretable baseline; needs one-hot, imputation, scaling |
# MAGIC | Gradient-boosted trees, no weights | Captures interactions and step effects |
# MAGIC | Gradient-boosted trees, balanced weights | Cost-sensitive learning for the ~5% minority class |

# COMMAND ----------

# MAGIC %pip install -q "databricks-feature-engineering>=0.12" "scikit-learn>=1.4"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

import cloudpickle
import mlflow
import pandas as pd
from databricks.feature_engineering import FeatureEngineeringClient

import claims_triage
from claims_triage.tracking import load_training_data, log_candidate
from claims_triage.train import baseline_classifier, gbm_classifier
from claims_triage.triage import TIER_THRESHOLDS

cloudpickle.register_pickle_by_value(claims_triage)   # logged pyfunc models load without this package
mlflow.sklearn.autolog(disable=True)                   # we log explicitly so every run looks the same
fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training sets from the feature table
# MAGIC `load_training_data()` in `src/claims_triage/tracking.py` does this for each split:
# MAGIC
# MAGIC ```python
# MAGIC lookups = [FeatureLookup(table_name=FEATURE_TABLE, lookup_key="policy_id", feature_names=POLICY_FEATURES)]
# MAGIC ts = fe.create_training_set(df=claims_for_split,          # claim_id, policy_id, FNOL fields, large_loss
# MAGIC                             feature_lookups=lookups,
# MAGIC                             label="large_loss",
# MAGIC                             exclude_columns=["claim_id"])  # policy_id stays: it is the lookup key when scoring
# MAGIC pdf = ts.load_df().toPandas()
# MAGIC ```
# MAGIC The split is deterministic by `claim_id` (CRC-32), so every notebook and the retraining job agree on it.

# COMMAND ----------

data = load_training_data(spark, fe, FQ, FEATURE_TABLE)
print(f"train {len(data.X_train)} | val {len(data.X_val)} | test {len(data.X_test)} "
      f"| large-loss rate {data.y_train.mean():.3f}")
display(data.X_train.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train, evaluate, log
# MAGIC `log_candidate()` fits the model and logs, in one MLflow run:
# MAGIC - **params**: model family, class weights, hyperparameters (`mlflow.log_params`)
# MAGIC - **metrics**: ROC AUC, **PR AUC** (primary, because positives are rare), log loss, precision, recall, F1, on
# MAGIC   validation and test (`mlflow.log_metrics`)
# MAGIC - **figure**: precision-recall curve and confusion matrix (`mlflow.log_figure`)
# MAGIC - **models**: `sklearn_model` (raw features), `proba_model` (pyfunc returning probabilities, for `spark_udf`), and
# MAGIC   `fs_model` (logged with `fe.log_model`, so it looks up policy features by itself at scoring time)

# COMMAND ----------

results = {}
for name, model, family, cw in [
    ("logreg_balanced", baseline_classifier("balanced"), "logistic_regression", "balanced"),
    ("gbm_unweighted", gbm_classifier(class_weight=None), "hist_gbm", None),
    ("gbm_balanced", gbm_classifier(class_weight="balanced"), "hist_gbm", "balanced"),
]:
    est_params = {k: v for k, v in model.named_steps["model"].get_params().items()
                  if k in ("C", "learning_rate", "max_depth", "min_samples_leaf", "l2_regularization", "max_iter")}
    run_id, m = log_candidate(fe, data, model, name,
                              tags={"model_family": family, "class_weight": str(cw)},
                              params={"model_family": family, "class_weight": str(cw), **est_params})
    results[name] = {"run_id": run_id, **{k: round(v, 3) for k, v in m.items()}}

display(pd.DataFrame(results).T)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading the results
# MAGIC - **Accuracy is not reported on purpose.** With a 5% base rate, predicting "not large" for every claim is 95% accurate.
# MAGIC - **PR AUC** is the primary metric: it measures how well the model ranks the rare positives. ROC AUC looks high for
# MAGIC   every model on imbalanced data.
# MAGIC - **Balanced class weights** (cost-sensitive learning) raise recall at the 0.5 threshold and lower precision. They
# MAGIC   barely change PR AUC (ranking), but they make probabilities less calibrated, which shows up as worse **log loss**.
# MAGIC   Because this project turns probabilities into tiers with business thresholds, the unweighted model plus a chosen
# MAGIC   threshold is usually the better production choice. Moving the threshold is another way to handle imbalance.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find the best run with the MLflow Client API

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
exp_id = mlflow.get_experiment_by_name(EXPERIMENT_PATH).experiment_id
best = client.search_runs(
    experiment_ids=[exp_id],
    filter_string="tags.candidate = 'true'",
    order_by=["metrics.val_pr_auc DESC"],
    max_results=1,
)[0]
print("best run:", best.info.run_name, best.info.run_id, "val_pr_auc =", round(best.data.metrics["val_pr_auc"], 4))

# The same search as a pandas DataFrame:
display(mlflow.search_runs(experiment_ids=[exp_id], filter_string="tags.candidate = 'true'",
                           order_by=["metrics.val_pr_auc DESC"])
        [["tags.mlflow.runName", "metrics.val_pr_auc", "metrics.val_roc_auc", "metrics.val_log_loss",
          "metrics.val_recall", "metrics.val_precision"]].head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## What the tiers mean for the claims team
# MAGIC Using the best model's validation predictions: how many claims land in each tier, and what share of all large
# MAGIC losses each tier catches.

# COMMAND ----------

best_model = mlflow.sklearn.load_model(f"runs:/{best.info.run_id}/sklearn_model")
p = best_model.predict_proba(data.X_val)[:, 1]
tiers = pd.cut(p, [-0.01, TIER_THRESHOLDS["ELEVATED"], TIER_THRESHOLDS["HIGH"], 1.01],
               labels=["STANDARD", "ELEVATED", "HIGH"])
summary = (pd.DataFrame({"tier": tiers, "large_loss": data.y_val.to_numpy()})
           .groupby("tier", observed=False)
           .agg(claims=("large_loss", "size"), large_losses=("large_loss", "sum")))
summary["share_of_claims"] = (summary["claims"] / summary["claims"].sum()).round(3)
summary["share_of_large_losses"] = (summary["large_losses"] / summary["large_losses"].sum()).round(3)
display(summary.reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC Open the experiment (left sidebar **Experiments**, or the link printed by each run) and look at what the MLflow UI
# MAGIC shows for a run: parameters, metrics and charts, the `evaluation.png` artifact, the three logged models with their
# MAGIC signatures and input examples, tags, the source notebook, user, start time and duration. Select runs and click
# MAGIC **Compare** to chart them side by side.
# MAGIC
# MAGIC Next: `03b_severity_regression` (optional, regression metrics and bias/variance) or `04_tuning`.
