# Databricks notebook source
# MAGIC %md
# MAGIC # 03b · Severity regression, log targets, and bias vs variance
# MAGIC **Exam Section 3.** Predict 12-month incurred dollars for each claim. Shows why you train on a log target, why you
# MAGIC must exponentiate predictions before computing RMSE/MAE/R², and how model complexity trades bias for variance.

# COMMAND ----------

# MAGIC %pip install -q "scikit-learn>=1.4"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from claims_triage.evaluation import regression_metrics_from_log
from claims_triage.features import build_preprocessor
from claims_triage.splits import spark_split_col
from claims_triage.train import feature_frame, gbm_regressor

mlflow.sklearn.autolog(disable=True)
pdf = (spark.table(f"{FQ}.claims_training_base").join(spark.table(FEATURE_TABLE), "policy_id")
       .withColumn("split", spark_split_col("claim_id")).toPandas())
tr, va = pdf[pdf.split == "train"], pdf[pdf.split == "val"]
X_tr, X_va = feature_frame(tr), feature_frame(va)
y_tr_log = np.log1p(tr["incurred_12m"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train on log1p(incurred), report in dollars

# COMMAND ----------

with mlflow.start_run(run_name="severity_gbm_log_target"):
    reg = gbm_regressor().fit(X_tr, y_tr_log)
    pred_log = reg.predict(X_va)

    right = regression_metrics_from_log(va["incurred_12m"], pred_log)          # expm1 first, then metrics
    wrong_rmse = float(np.sqrt(np.mean((va["incurred_12m"] - pred_log) ** 2)))   # comparing dollars to log units
    mlflow.log_metrics({f"val_{k}": v for k, v in right.items()})
    mlflow.set_tags({"model_family": "hist_gbm_regressor", "target": "log1p(incurred_12m)"})

print({k: round(v, 3) for k, v in right.items()})
print(f"Without exponentiating, 'RMSE' would be {wrong_rmse:,.0f}: meaningless, it compares dollars to log units.")

# COMMAND ----------

# MAGIC %md
# MAGIC - **RMSE** is in dollars and is dominated by the few largest claims (squared errors).
# MAGIC - **MAE** is the typical miss in dollars and is less sensitive to those outliers.
# MAGIC - **R²** is the share of variance explained (1 is perfect, 0 is no better than predicting the mean).
# MAGIC - `rmse_log_scale` is useful for comparing models, but only dollar metrics mean anything to a claims manager.
# MAGIC
# MAGIC Note: `expm1` of a mean log prediction is closer to a median than a mean, so dollar predictions run low on average.
# MAGIC Actuarial work adds a smearing correction; it is out of scope here.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bias vs variance: one decision tree, increasing depth

# COMMAND ----------

rows = []
for depth in [1, 2, 4, 6, 8, 10, 14, 20]:
    tree = Pipeline([("prep", build_preprocessor("tree")), ("model", DecisionTreeRegressor(max_depth=depth, random_state=0))])
    tree.fit(X_tr, y_tr_log)
    rows.append({"max_depth": depth,
                 "train_rmse_log": regression_metrics_from_log(tr["incurred_12m"], tree.predict(X_tr))["rmse_log_scale"],
                 "val_rmse_log": regression_metrics_from_log(va["incurred_12m"], tree.predict(X_va))["rmse_log_scale"]})
curve = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(curve.max_depth, curve.train_rmse_log, marker="o", label="training error")
ax.plot(curve.max_depth, curve.val_rmse_log, marker="o", label="validation error")
best = curve.loc[curve.val_rmse_log.idxmin()]
ax.axvline(best.max_depth, ls="--", c="grey")
ax.set_xlabel("max_depth (model complexity)")
ax.set_ylabel("RMSE on log1p(incurred)")
ax.set_title("Bias-variance tradeoff")
ax.legend()
display(fig)
with mlflow.start_run(run_name="bias_variance_curve"):
    mlflow.log_figure(fig, "bias_variance.png")
display(curve.round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC - **Shallow trees (left): high bias.** Training and validation error are both high. The model is too simple.
# MAGIC - **Deep trees (right): high variance.** Training error keeps falling while validation error rises: overfitting.
# MAGIC - The sweet spot is the lowest validation error. Remedies for overfitting: limit depth, require more samples per
# MAGIC   leaf, regularize, add data, or average many trees (random forest, boosting with shrinkage).
# MAGIC
# MAGIC Next: `04_tuning`.
