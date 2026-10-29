# Databricks notebook source
# MAGIC %md
# MAGIC # Bootstrap (loaded by every notebook with `%run ./_bootstrap`)
# MAGIC Puts `src/` on the Python path, reads the catalog/schema widgets, and sets the shared names.

# COMMAND ----------

import os
import sys


def _find_src():
    # Serverless does not guarantee the working directory, so also start from the notebook's own path.
    starts = [os.getcwd()]
    try:
        nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        starts.append(os.path.dirname(nb if nb.startswith("/Workspace") else "/Workspace" + nb))
    except Exception:
        pass
    for p in starts:
        while p not in ("/", ""):
            if os.path.isdir(os.path.join(p, "src", "claims_triage")):
                return os.path.join(p, "src")
            p = os.path.dirname(p)
    raise FileNotFoundError("Could not find src/claims_triage. Open the notebooks from the cloned Git folder.")


SRC = _find_src()
if SRC not in sys.path:
    sys.path.insert(0, SRC)

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "claims_ml")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
FQ = f"{CATALOG}.{SCHEMA}"

FEATURE_TABLE = f"{FQ}.policy_features"                  # offline feature table (primary key policy_id)
MODEL_NAME = f"{FQ}.fnol_large_loss"                      # UC registered model used for batch scoring
SERVICE_MODEL_NAME = f"{FQ}.fnol_triage_service"          # UC registered model behind the serving endpoint
ENDPOINT_NAME = "fnol-triage"

USER = spark.sql("SELECT current_user()").first()[0]
EXPERIMENT_PATH = f"/Users/{USER}/fnol-claims-triage"

import mlflow  # noqa: E402

mlflow.set_registry_uri("databricks-uc")                  # default with MLflow 3; explicit for older clients
mlflow.set_experiment(EXPERIMENT_PATH)

print(f"catalog.schema = {FQ}\nfeature table  = {FEATURE_TABLE}\nmodel          = {MODEL_NAME}\nexperiment     = {EXPERIMENT_PATH}")
