# Databricks notebook source
# MAGIC %md
# MAGIC # Job task: champion_gate
# MAGIC Scores the held-out test claims with `@champion` and `@challenger` and promotes the challenger only if it beats
# MAGIC the champion's PR AUC by at least `min_gain`. Publishes `promoted` as a task value.

# COMMAND ----------

# MAGIC %pip install -q "databricks-feature-engineering>=0.12" "scikit-learn>=1.4"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../notebooks/_bootstrap

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from mlflow.tracking import MlflowClient

from claims_triage.evaluation import challenger_wins
from claims_triage.tracking import evaluate_alias

dbutils.widgets.text("min_gain", "0.005")
min_gain = float(dbutils.widgets.get("min_gain"))
fe, client = FeatureEngineeringClient(), MlflowClient()

challenger = client.get_model_version_by_alias(MODEL_NAME, "challenger")
champ_m = evaluate_alias(fe, spark, FQ, MODEL_NAME, "champion")
chall_m = evaluate_alias(fe, spark, FQ, MODEL_NAME, "challenger")
print("champion  :", champ_m)
print("challenger:", chall_m)

promoted = challenger_wins(chall_m, champ_m, metric="pr_auc", min_gain=min_gain)
if promoted:
    if champ_m is not None:
        old = client.get_model_version_by_alias(MODEL_NAME, "champion")
        client.set_registered_model_alias(MODEL_NAME, "previous_champion", old.version)
    client.set_registered_model_alias(MODEL_NAME, "champion", challenger.version)
    client.set_model_version_tag(MODEL_NAME, challenger.version, "validation_status", "passed")
else:
    client.set_model_version_tag(MODEL_NAME, challenger.version, "validation_status", "rejected")
client.delete_registered_model_alias(MODEL_NAME, "challenger")
dbutils.jobs.taskValues.set(key="promoted", value=promoted)
print("promoted:", promoted)
