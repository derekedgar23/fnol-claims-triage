# Databricks notebook source
# MAGIC %md
# MAGIC # Job task: retrain_challenger
# MAGIC Deploy-code pattern: this same code runs in dev and prod and retrains on that environment's data.
# MAGIC Trains with the tuned parameters in `conf/best_params.json`, logs the run, registers a new version,
# MAGIC and points the `@challenger` alias at it.

# COMMAND ----------

# MAGIC %pip install -q "databricks-feature-engineering>=0.12" "scikit-learn>=1.4"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../notebooks/_bootstrap

# COMMAND ----------

import json
import os
from datetime import date

import cloudpickle
import mlflow
from databricks.feature_engineering import FeatureEngineeringClient
from mlflow.tracking import MlflowClient

import claims_triage
from claims_triage.tracking import load_training_data, log_candidate
from claims_triage.train import gbm_classifier

cloudpickle.register_pickle_by_value(claims_triage)
mlflow.sklearn.autolog(disable=True)
fe, client = FeatureEngineeringClient(), MlflowClient()

params = json.load(open(os.path.join(os.path.dirname(SRC), "conf", "best_params.json")))
data = load_training_data(spark, fe, FQ, FEATURE_TABLE)
run_id, m = log_candidate(fe, data, gbm_classifier(params), f"retrain_{date.today().isoformat()}",
                          tags={"model_family": "hist_gbm", "trigger": "scheduled_retrain"},
                          params={"model_family": "hist_gbm", **params})

mv = mlflow.register_model(f"runs:/{run_id}/fs_model", MODEL_NAME)
client.set_registered_model_alias(MODEL_NAME, "challenger", mv.version)
client.set_model_version_tag(MODEL_NAME, mv.version, "validation_status", "pending")
dbutils.jobs.taskValues.set(key="challenger_version", value=int(mv.version))
print(f"registered {MODEL_NAME} v{mv.version} as @challenger, val PR AUC {m['pr_auc']:.4f}")
