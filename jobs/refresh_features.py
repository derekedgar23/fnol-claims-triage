# Databricks notebook source
# MAGIC %md
# MAGIC # Job task: refresh_features
# MAGIC Upserts the policy feature table from the latest policy data (`write_table(mode="merge")`).
# MAGIC Creates the raw data and the table on the first run in a new schema.

# COMMAND ----------

# MAGIC %pip install -q "databricks-feature-engineering>=0.12" "scikit-learn>=1.4"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../notebooks/_bootstrap

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

from claims_triage.features import POLICY_FEATURES

fe = FeatureEngineeringClient()
if not spark.catalog.tableExists(f"{FQ}.raw_policies"):
    # First run in a new schema (for example the bundle's dev target): build the synthetic data here.
    from pyspark.sql import functions as F

    from claims_triage.data import generate
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FQ}")
    pol, clm = generate()
    pol_sdf = spark.createDataFrame(pol)
    for c in ["annual_revenue", "years_in_business"]:
        pol_sdf = pol_sdf.withColumn(c, F.when(F.isnan(c), None).otherwise(F.col(c)))
    pol_sdf.write.mode("overwrite").saveAsTable(f"{FQ}.raw_policies")
    spark.createDataFrame(clm).write.mode("overwrite").saveAsTable(f"{FQ}.raw_claims")
    spark.sql(f"CREATE OR REPLACE TABLE {FQ}.claims_training_base AS "
              f"SELECT * FROM {FQ}.raw_claims WHERE loss_date < '2026-01-01'")
    spark.sql(f"CREATE OR REPLACE TABLE {FQ}.claims_to_score AS SELECT claim_id, policy_id, loss_date, loss_month, "
              f"cause_of_loss, report_lag_days, injury_flag, attorney_at_fnol FROM {FQ}.raw_claims "
              f"WHERE loss_date >= '2026-01-01'")

src = spark.table(f"{FQ}.raw_policies").select("policy_id", *POLICY_FEATURES)
if spark.catalog.tableExists(FEATURE_TABLE):
    fe.write_table(name=FEATURE_TABLE, df=src, mode="merge")
else:
    fe.create_table(name=FEATURE_TABLE, primary_keys=["policy_id"], df=src,
                    description="Policy-level features for claims models. Synthetic data.")
print("feature rows:", spark.table(FEATURE_TABLE).count())
