# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Batch and streaming inference
# MAGIC **Exam Section 4.** Three ways to score the 2026 FNOL claims with the champion model:
# MAGIC
# MAGIC | Method | Scale | When |
# MAGIC |---|---|---|
# MAGIC | pandas: `load_model(...).predict(pdf)` | One machine | Small batches, notebooks, tests |
# MAGIC | `mlflow.pyfunc.spark_udf` | Cluster | Large tables, and inside streaming pipelines |
# MAGIC | `fe.score_batch` | Cluster | Feature store models: pass keys, features are looked up |
# MAGIC
# MAGIC Batch is the right pattern here: claims are triaged daily, there is no millisecond requirement, and it is the
# MAGIC cheapest option. Notebook 07 covers the real-time case.

# COMMAND ----------

# MAGIC %pip install -q "databricks-feature-engineering>=0.12" "scikit-learn>=1.4"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

from claims_triage.features import ALL_FEATURES
from claims_triage.tracking import score_with_alias, tier_column

client = MlflowClient()
fe = FeatureEngineeringClient()
champ = client.get_model_version_by_alias(MODEL_NAME, "champion")
run_id = champ.run_id
print(f"champion = {MODEL_NAME} v{champ.version} (run {run_id})")

to_score = spark.table(f"{FQ}.claims_to_score")
with_features = to_score.join(spark.table(FEATURE_TABLE), "policy_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. pandas batch inference

# COMMAND ----------

pdf = with_features.limit(1000).toPandas()
proba_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/proba_model")
pdf["large_loss_probability"] = proba_model.predict(pdf[ALL_FEATURES])
display(pdf[["claim_id", "line_of_business", "cause_of_loss", "attorney_at_fnol", "large_loss_probability"]]
        .sort_values("large_loss_probability", ascending=False).head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Spark UDF (distributed; the same pattern runs inside a streaming pipeline)

# COMMAND ----------

predict = mlflow.pyfunc.spark_udf(spark, f"runs:/{run_id}/proba_model", result_type="double", env_manager="local")
scored_udf = with_features.withColumn("large_loss_probability", predict(F.struct(*ALL_FEATURES)))
display(scored_udf.select("claim_id", "large_loss_probability").orderBy(F.desc("large_loss_probability")).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feature store batch scoring (production path)
# MAGIC Only `policy_id` and the FNOL fields go in. The model fetches policy features from the feature table, so scoring
# MAGIC can never drift from the features the model was trained on.
# MAGIC
# MAGIC `score_with_alias()` calls `fe.score_batch(model_uri="models:/...@champion", df=keys_and_fnol)`. If that path fails
# MAGIC in your environment it prints why and falls back to a feature join plus the `proba_model` Spark UDF.

# COMMAND ----------

scored = (score_with_alias(fe, spark, MODEL_NAME, "champion", to_score, FEATURE_TABLE)
          .withColumnRenamed("prediction", "large_loss_probability")
          .withColumn("triage_tier", tier_column("large_loss_probability"))
          .withColumn("model_version", F.lit(int(champ.version)))
          .withColumn("scored_at", F.current_timestamp())
          .select("claim_id", "policy_id", "loss_date", "large_loss_probability", "triage_tier",
                  "model_version", "scored_at"))
scored.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{FQ}.claims_triage_scores")

display(spark.sql(f"""
  SELECT triage_tier, count(*) AS claims, round(avg(large_loss_probability), 3) AS avg_probability
  FROM {FQ}.claims_triage_scores GROUP BY triage_tier ORDER BY avg_probability DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Streaming inference with a declarative pipeline
# MAGIC For continuous FNOL feeds, apply the same Spark UDF inside a Lakeflow Spark Declarative Pipeline (the exam guide
# MAGIC calls it Delta Live Tables). Pipelines autoscale their compute as event volume changes, which is why the exam
# MAGIC prefers this over calling a serving endpoint from a stream.
# MAGIC
# MAGIC See `src/pipelines/streaming_triage.py`. To try it, create an ETL pipeline with that file as source code,
# MAGIC default catalog `workspace` and schema `claims_ml`, add `mlflow` and `scikit-learn` to the pipeline environment,
# MAGIC and set the configuration key `model_uri` to the `proba_model` URI printed below.
# MAGIC Then append rows to `fnol_stream_source` and run the pipeline.

# COMMAND ----------

spark.sql(f"""
  CREATE TABLE IF NOT EXISTS {FQ}.fnol_stream_source AS
  SELECT * FROM {FQ}.claims_to_score LIMIT 0
""")
spark.sql(f"INSERT INTO {FQ}.fnol_stream_source SELECT * FROM {FQ}.claims_to_score LIMIT 200")
print("fnol_stream_source rows:", spark.table(f"{FQ}.fnol_stream_source").count())
print("proba_model URI for the pipeline config:", f"runs:/{run_id}/proba_model")

# COMMAND ----------

# MAGIC %md
# MAGIC Next: `07_model_serving` (real-time endpoint, custom pyfunc, traffic split).
