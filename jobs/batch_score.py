# Databricks notebook source
# MAGIC %md
# MAGIC # Job task: batch_score
# MAGIC Scores new FNOL claims with `@champion` (whichever version that is after the gate) and writes triage tiers.

# COMMAND ----------

# MAGIC %pip install -q "databricks-feature-engineering>=0.12" "scikit-learn>=1.4"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../notebooks/_bootstrap

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

from claims_triage.tracking import score_with_alias, tier_column

fe = FeatureEngineeringClient()
champ = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
scored = (score_with_alias(fe, spark, MODEL_NAME, "champion", spark.table(f"{FQ}.claims_to_score"), FEATURE_TABLE)
          .withColumnRenamed("prediction", "large_loss_probability")
          .withColumn("triage_tier", tier_column("large_loss_probability"))
          .withColumn("model_version", F.lit(int(champ.version)))
          .withColumn("scored_at", F.current_timestamp())
          .select("claim_id", "policy_id", "loss_date", "large_loss_probability", "triage_tier",
                  "model_version", "scored_at"))
scored.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{FQ}.claims_triage_scores")
display(spark.table(f"{FQ}.claims_triage_scores").groupBy("triage_tier").count())
