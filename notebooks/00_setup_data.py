# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Generate the synthetic portfolio
# MAGIC Creates the schema and two raw tables from the deterministic generator in `src/claims_triage/data.py`:
# MAGIC
# MAGIC | Table | Grain | Rows |
# MAGIC |---|---|---|
# MAGIC | `raw_policies` | one row per policy | 20,000 |
# MAGIC | `raw_claims` | one row per claim, FNOL fields + 12-month incurred | 40,000 |
# MAGIC
# MAGIC It then splits claims into a **training base** (losses before 2026, labels known) and **claims to score**
# MAGIC (2026 losses, labels hidden), which is how the model would be used in practice: trained on mature claims,
# MAGIC applied to new First Notice of Loss reports.
# MAGIC
# MAGIC All data is fictional. Safe to rerun.

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

from claims_triage.data import LARGE_LOSS_THRESHOLD, generate

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FQ}")
policies_pdf, claims_pdf = generate()
print(len(policies_pdf), "policies |", len(claims_pdf), "claims | large-loss rate",
      round(claims_pdf["large_loss"].mean(), 4), f"(threshold ${LARGE_LOSS_THRESHOLD:,})")

# COMMAND ----------

from pyspark.sql import functions as F

policies_sdf = spark.createDataFrame(policies_pdf)
# pandas NaN can arrive in Spark as NaN rather than NULL. Make missing values real NULLs.
for c in ["annual_revenue", "years_in_business"]:
    policies_sdf = policies_sdf.withColumn(c, F.when(F.isnan(c), None).otherwise(F.col(c)))

(policies_sdf
 .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{FQ}.raw_policies"))
(spark.createDataFrame(claims_pdf)
 .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{FQ}.raw_claims"))

spark.sql(f"COMMENT ON TABLE {FQ}.raw_policies IS 'Synthetic commercial policies (fictional)'")
spark.sql(f"COMMENT ON TABLE {FQ}.raw_claims IS 'Synthetic claims with FNOL fields and 12-month incurred (fictional)'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training base vs claims to score

# COMMAND ----------

spark.sql(f"""
  CREATE OR REPLACE TABLE {FQ}.claims_training_base AS
  SELECT * FROM {FQ}.raw_claims WHERE loss_date < '2026-01-01'
""")
spark.sql(f"""
  CREATE OR REPLACE TABLE {FQ}.claims_to_score AS
  SELECT claim_id, policy_id, loss_date, loss_month, cause_of_loss, report_lag_days, injury_flag, attorney_at_fnol
  FROM {FQ}.raw_claims WHERE loss_date >= '2026-01-01'
""")
display(spark.sql(f"""
  SELECT 'claims_training_base' AS table_name, count(*) AS rows, round(avg(large_loss), 4) AS large_loss_rate
  FROM {FQ}.claims_training_base
  UNION ALL
  SELECT 'claims_to_score', count(*), NULL FROM {FQ}.claims_to_score
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC Next: `01_eda_data_prep`.
