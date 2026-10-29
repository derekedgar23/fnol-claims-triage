# Databricks notebook source
# MAGIC %md
# MAGIC # 08 · AutoML baseline (reference)
# MAGIC **Exam Section 1.** AutoML gives a fast, glass-box baseline: it prepares the data, tries several algorithms and
# MAGIC hyperparameters, logs every trial to an MLflow experiment, and generates a **data exploration notebook** and an
# MAGIC editable **best-trial notebook** you can read and modify.
# MAGIC
# MAGIC **Compute note:** AutoML classification and regression run on classic compute with a Databricks Runtime for ML
# MAGIC (dedicated access mode). Free Edition is serverless only, so run this on a work workspace, or read it as reference.
# MAGIC From DBR 18.0 ML, AutoML is no longer a built-in library and must be installed.

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

dataset = (spark.table(f"{FQ}.claims_training_base").join(spark.table(FEATURE_TABLE), "policy_id")
           .drop("claim_id", "policy_id", "loss_date", "incurred_12m"))

try:
    from databricks import automl
    summary = automl.classify(
        dataset=dataset,
        target_col="large_loss",
        primary_metric="f1",            # options include f1, log_loss, precision, accuracy, roc_auc
        timeout_minutes=15,
        experiment_name=f"fnol-automl-{SCHEMA}",
    )
    print("best trial notebook:", summary.best_trial.notebook_url)
    print("best trial metrics :", summary.best_trial.metrics)
except ImportError:
    print("databricks.automl is not available on this compute (needs classic compute with an ML runtime).")

# COMMAND ----------

# MAGIC %md
# MAGIC Compare the AutoML best trial with the hand-built models in notebook 03. Where AutoML helps: a strong baseline in
# MAGIC minutes, a view of which algorithms and features matter, and generated code to start from. Where hand-built wins:
# MAGIC custom preprocessing, business metrics like PR AUC and tier capture, feature-store packaging, and the serving wrapper.
