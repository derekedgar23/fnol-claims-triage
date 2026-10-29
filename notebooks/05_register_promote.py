# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Register in Unity Catalog and promote with aliases
# MAGIC **Exam Section 1.** Register model versions with three-level names, tag them, and run a champion/challenger
# MAGIC promotion with **aliases** (Unity Catalog does not use the old Staging/Production stages).
# MAGIC
# MAGIC Scenario: the logistic regression baseline is already in production (`@champion`). The best tree model is the
# MAGIC `@challenger`. Both are scored on the same held-out test claims; the challenger is promoted only if it wins on PR AUC.

# COMMAND ----------

# MAGIC %pip install -q "databricks-feature-engineering>=0.12" "scikit-learn>=1.4"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

import mlflow
import pandas as pd
from databricks.feature_engineering import FeatureEngineeringClient
from mlflow.tracking import MlflowClient

from claims_triage.evaluation import challenger_wins
from claims_triage.tracking import evaluate_alias

client = MlflowClient()
fe = FeatureEngineeringClient()
exp_id = mlflow.get_experiment_by_name(EXPERIMENT_PATH).experiment_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pick runs with the Client API

# COMMAND ----------

runs = client.search_runs([exp_id], filter_string="tags.candidate = 'true'", order_by=["metrics.val_pr_auc DESC"])
best = runs[0]
baseline = next(r for r in runs if r.data.tags.get("model_family") == "logistic_regression")
print("best      :", best.info.run_name, round(best.data.metrics["val_pr_auc"], 4))
print("baseline  :", baseline.info.run_name, round(baseline.data.metrics["val_pr_auc"], 4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register versions
# MAGIC `mlflow.register_model("runs:/<run_id>/<artifact>", "catalog.schema.model")` creates a new **version** each call.
# MAGIC We register `fs_model`, the feature-store-aware model, so batch scoring only needs `policy_id` plus FNOL fields.

# COMMAND ----------

v_base = mlflow.register_model(f"runs:/{baseline.info.run_id}/fs_model", MODEL_NAME)
client.set_registered_model_alias(MODEL_NAME, "champion", v_base.version)      # simulate current production

v_new = mlflow.register_model(f"runs:/{best.info.run_id}/fs_model", MODEL_NAME)
client.set_registered_model_alias(MODEL_NAME, "challenger", v_new.version)

client.update_registered_model(MODEL_NAME, description="P(claim incurred >= $50k at 12 months) from FNOL data. "
                                                      "Synthetic portfolio. Drives claims triage tiers.")
client.set_registered_model_tag(MODEL_NAME, "owner", "claims-analytics")
client.set_registered_model_tag(MODEL_NAME, "use_case", "fnol_triage")
client.set_model_version_tag(MODEL_NAME, v_new.version, "validation_status", "pending")
client.set_model_version_tag(MODEL_NAME, v_new.version, "source_run", best.info.run_name)
print(f"champion = v{v_base.version}, challenger = v{v_new.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Champion vs challenger on the same test claims
# MAGIC `evaluate_alias` calls `fe.score_batch(model_uri="models:/<name>@<alias>", df=keys_and_fnol_fields)`. The model
# MAGIC looks up the policy features from the feature table by itself.

# COMMAND ----------

champ = evaluate_alias(fe, spark, FQ, MODEL_NAME, "champion")
chall = evaluate_alias(fe, spark, FQ, MODEL_NAME, "challenger")
display(pd.DataFrame({"champion": champ, "challenger": chall}).round(4))

# COMMAND ----------

if challenger_wins(chall, champ, metric="pr_auc", min_gain=0.005):
    client.set_registered_model_alias(MODEL_NAME, "previous_champion", v_base.version)   # one-step rollback
    client.set_registered_model_alias(MODEL_NAME, "champion", v_new.version)
    client.set_model_version_tag(MODEL_NAME, v_new.version, "validation_status", "passed")
    print(f"Promoted v{v_new.version} to @champion. Rollback target: @previous_champion = v{v_base.version}")
else:
    client.set_model_version_tag(MODEL_NAME, v_new.version, "validation_status", "rejected")
    print("Challenger did not beat the champion by the required margin. Champion unchanged.")

client.delete_registered_model_alias(MODEL_NAME, "challenger")    # the comparison is over

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load by alias, and remove a tag

# COMMAND ----------

mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
print("champion is version", mv.version, "| tags:", mv.tags, "| aliases:", mv.aliases)
champion_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")   # how production code loads it

client.set_model_version_tag(MODEL_NAME, mv.version, "temp_note", "delete me")
client.delete_model_version_tag(MODEL_NAME, mv.version, "temp_note")
print("tags after delete:", client.get_model_version(MODEL_NAME, mv.version).tags)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Why the Unity Catalog registry
# MAGIC - Three-level names, usable from any workspace on the metastore.
# MAGIC - Unity Catalog permissions: `EXECUTE` on the model (plus `USE CATALOG` / `USE SCHEMA`) to load it.
# MAGIC - Lineage: Catalog Explorer shows which feature table and run each version came from.
# MAGIC - **Aliases** are mutable pointers (`@champion`), so production code never changes when you promote.
# MAGIC   `transition_model_version_stage` is the legacy workspace-registry API.
# MAGIC
# MAGIC Rerunning this notebook registers new versions each time. That is expected; versions are cheap and immutable.
# MAGIC
# MAGIC Next: `06_batch_streaming_inference`.
