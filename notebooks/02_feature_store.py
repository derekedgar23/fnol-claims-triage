# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Policy feature table in Unity Catalog
# MAGIC **Exam Section 1.** Create a feature table, write to it with merge, and understand offline vs online features.
# MAGIC
# MAGIC Design: policy attributes change slowly and are reused by many models (triage, severity, pricing), so they live in
# MAGIC a governed feature table keyed by `policy_id`. Claim-level FNOL fields arrive with each claim and are passed in at
# MAGIC training and scoring time.

# COMMAND ----------

# MAGIC %pip install -q "databricks-feature-engineering>=0.12" "scikit-learn>=1.4"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import functions as F

from claims_triage.features import POLICY_FEATURES

fe = FeatureEngineeringClient()
policy_features = spark.table(f"{FQ}.raw_policies").select("policy_id", *POLICY_FEATURES)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the feature table
# MAGIC `FeatureEngineeringClient.create_table` creates a Unity Catalog Delta table with a **primary key**. There is no
# MAGIC `AS FEATURE STORE` SQL clause, and the old workspace `FeatureStoreClient` is legacy. Any UC Delta table with a
# MAGIC primary key constraint can also be used as a feature table.

# COMMAND ----------

if not spark.catalog.tableExists(FEATURE_TABLE):
    fe.create_table(
        name=FEATURE_TABLE,
        primary_keys=["policy_id"],
        df=policy_features,
        description="Policy-level features for claims models. Synthetic data. Key: policy_id.",
        tags={"domain": "claims", "pii": "none"},
    )
    print("created", FEATURE_TABLE)
else:
    fe.write_table(name=FEATURE_TABLE, df=policy_features, mode="merge")
    print("merged into existing", FEATURE_TABLE)

display(spark.sql(f"DESCRIBE TABLE EXTENDED {FEATURE_TABLE}").where("col_name IN ('Type', 'Table Properties', 'Comment')"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write updates with merge
# MAGIC Five policies report a new prior claim. `write_table(mode="merge")` upserts on the primary key: matching rows are
# MAGIC updated, new keys are inserted, everything else is untouched.

# COMMAND ----------

updates = (spark.table(FEATURE_TABLE).orderBy("policy_id").limit(5)
           .withColumn("prior_claims_3y", F.col("prior_claims_3y") + 1))
fe.write_table(name=FEATURE_TABLE, df=updates, mode="merge")

display(spark.table(FEATURE_TABLE).orderBy("policy_id").limit(5).select("policy_id", "prior_claims_3y"))
display(spark.sql(f"DESCRIBE HISTORY {FEATURE_TABLE}").select("version", "operation", "timestamp").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Why Unity Catalog (account level) instead of the workspace feature store
# MAGIC - One table, usable from every workspace attached to the metastore.
# MAGIC - Same permissions model as every other table (`GRANT SELECT ON TABLE ...`).
# MAGIC - Lineage from the feature table to the models trained on it (see the **Lineage** tab in Catalog Explorer after notebook 03).
# MAGIC - Discoverable in Catalog Explorer, with tags and descriptions.
# MAGIC
# MAGIC ## Offline vs online
# MAGIC | | Offline feature table (this notebook) | Online feature store |
# MAGIC |---|---|---|
# MAGIC | Storage | Delta table in UC | Low-latency store (Lakebase-backed Online Feature Store) |
# MAGIC | Used for | Training, batch scoring | Real-time serving lookups in milliseconds |
# MAGIC | How it is filled | `create_table` / `write_table` | `fe.publish_table(...)` from the offline table |
# MAGIC
# MAGIC This project serves a model that receives all its inputs in the request (notebook 07), so no online store is needed.
# MAGIC If the endpoint received only `policy_id`, you would publish this table to an online store so the endpoint could
# MAGIC look the features up. Sketch (check current docs before running; online stores may not be available on Free Edition):
# MAGIC
# MAGIC ```python
# MAGIC store = fe.create_online_store(name="claims-online", capacity="CU_1")
# MAGIC fe.publish_table(online_store=store, source_table_name=FEATURE_TABLE,
# MAGIC                  online_table_name=f"{FQ}.policy_features_online")
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exam check
# MAGIC 1. Which client and method create a feature table in UC?
# MAGIC 2. What does `mode="merge"` do?
# MAGIC 3. Name two benefits of UC feature tables over the workspace feature store.
# MAGIC 4. Offline or online: training? A 50 ms quoting API?
# MAGIC
# MAGIC Next: `03_train_track`.
