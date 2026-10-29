# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Explore and prepare the data
# MAGIC **Exam Section 2: Data Processing (19%).** Summary statistics, outliers (std dev vs IQR), visualizing and
# MAGIC comparing features, mean/median/mode imputation, one-hot encoding, and log transforms.
# MAGIC
# MAGIC The modeling pipelines in `src/claims_triage/features.py` encode the decisions made here, so the model applies
# MAGIC exactly the same preparation at training and scoring time.

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import functions as F

from claims_triage.features import iqr_bounds, std_bounds

policies = spark.table(f"{FQ}.raw_policies")
claims = spark.table(f"{FQ}.claims_training_base")
df = claims.join(policies, "policy_id")
print(df.count(), "claims with policy attributes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Summary statistics
# MAGIC `describe()` gives count, mean, stddev, min, max. `summary()` adds the 25/50/75% percentiles.
# MAGIC `dbutils.data.summarize()` adds histograms and missing-value counts (also on the **Data Profile** tab of `display`).

# COMMAND ----------

num_cols = ["incurred_12m", "annual_revenue", "total_insured_value", "years_in_business", "report_lag_days", "prior_claims_3y"]
display(df.select(num_cols).summary())

# COMMAND ----------

try:
    dbutils.data.summarize(df.select(num_cols + ["line_of_business", "cause_of_loss", "large_loss"]))
except Exception as e:
    print("dbutils.data.summarize not available on this compute; use the Data Profile tab of display():", str(e)[:200])

# COMMAND ----------

# MAGIC %md
# MAGIC **What to notice:** `incurred_12m`, `annual_revenue`, `total_insured_value` and `report_lag_days` have a mean far
# MAGIC above the median, which means a long right tail. `annual_revenue` and `years_in_business` have missing values.
# MAGIC Only about 5% of claims are large losses: the label is imbalanced.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Visualize categorical and continuous features
# MAGIC Bar charts for categories, histograms for continuous values. (Matplotlib is used so the charts also render on GitHub.)

# COMMAND ----------

pdf = df.select("line_of_business", "cause_of_loss", "incurred_12m", "report_lag_days", "annual_revenue",
                "total_insured_value", "large_loss", "injury_flag", "attorney_at_fnol").toPandas()

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
pdf["line_of_business"].value_counts().plot.barh(ax=axes[0], title="Claims by line of business")
axes[1].hist(pdf["incurred_12m"], bins=80)
axes[1].set_title("incurred_12m (raw dollars)")
axes[2].hist(np.log1p(pdf["incurred_12m"]), bins=80)
axes[2].set_title("log1p(incurred_12m)")
plt.tight_layout()
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC The raw severity is squashed against zero with a tail out to the policy limit; on the log scale it is close to
# MAGIC symmetric. That is the textbook case for a **log transform**: right-skewed and spanning several orders of magnitude.
# MAGIC The severity model in notebook 03b trains on `log1p(incurred)` and converts predictions back with `expm1`
# MAGIC before computing dollar metrics.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Compare two features
# MAGIC - Two continuous: scatter + correlation. Pearson measures linear association; Spearman (rank) handles skew.
# MAGIC - Two categorical: contingency table + chi-square test.

# COMMAND ----------

print("Pearson  revenue vs TIV:", round(df.stat.corr("annual_revenue", "total_insured_value"), 3))
ranked = pdf[["annual_revenue", "total_insured_value"]].dropna().rank()
print("Spearman revenue vs TIV:", round(ranked.corr().iloc[0, 1], 3))

# COMMAND ----------

from scipy.stats import chi2_contingency

ct = df.stat.crosstab("line_of_business", "large_loss")
display(ct)
table = pdf.groupby(["line_of_business", "large_loss"]).size().unstack(fill_value=0)
chi2, p, dof, _ = chi2_contingency(table)
print(f"chi-square={chi2:.1f}, dof={dof}, p={p:.2e}  -> large-loss rate depends on line of business")

# COMMAND ----------

display(pdf.groupby(["injury_flag", "attorney_at_fnol"])["large_loss"].agg(["mean", "count"]).reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC Injury plus attorney involvement at FNOL is a much higher large-loss rate than either alone. That interaction is
# MAGIC the kind of pattern tree models pick up without being told.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Outliers: standard deviation vs IQR
# MAGIC `report_lag_days` has a few claims reported 4 months to 2 years late.

# COMMAND ----------

lag = pdf["report_lag_days"]
lo_s, hi_s = std_bounds(lag, 3)
lo_i, hi_i = iqr_bounds(lag, 1.5)
print(f"mean ± 3 sd fences: ({lo_s:.1f}, {hi_s:.1f}) -> flags {((lag < lo_s) | (lag > hi_s)).sum()} rows")
print(f"IQR fences:         ({lo_i:.1f}, {hi_i:.1f}) -> flags {((lag < lo_i) | (lag > hi_i)).sum()} rows")

# Same thing in Spark, which is what you would do on a big table:
q1, q3 = df.approxQuantile("report_lag_days", [0.25, 0.75], 0.01)
iqr = q3 - q1
spark_kept = df.where(F.col("report_lag_days").between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)).count()
print("Spark IQR filter keeps", spark_kept, "of", df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC **Decision:** do not delete these claims. Late reports are real and predictive (they are a business signal, not a
# MAGIC data error), and the model needs to score them at FNOL. The IQR rule flags far more rows than ±3 sd because the
# MAGIC distribution is skewed; on skewed data the mean and sd are themselves pulled by the tail.
# MAGIC Removing outliers is right for data errors; for genuine extreme values, prefer robust models or transforms.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Missing values: mean vs median vs mode

# COMMAND ----------

rev = policies.select("annual_revenue")
stats = rev.agg(F.mean("annual_revenue").alias("mean"),
                F.expr("percentile_approx(annual_revenue, 0.5)").alias("median"),
                F.sum(F.col("annual_revenue").isNull().cast("int")).alias("missing")).first()
print(f"annual_revenue missing={stats['missing']}  mean={stats['mean']:,.0f}  median={stats['median']:,.0f}")

mode_row = (claims.where("cause_of_loss IS NOT NULL").groupBy("cause_of_loss").count()
            .orderBy(F.desc("count")).first())
print("cause_of_loss mode:", mode_row["cause_of_loss"])

# COMMAND ----------

# MAGIC %md
# MAGIC The mean revenue is well above the median because of the right tail, so mean imputation would assign every missing
# MAGIC insured an unusually large revenue. **Median** for skewed numeric features, **mode** for categorical ones. That is
# MAGIC what `build_preprocessor()` does (`SimpleImputer(strategy="median")` and `"most_frequent"`).
# MAGIC
# MAGIC Spark ML equivalent (needs `pyspark.ml` support on your compute; sklearn is used in the model pipelines):

# COMMAND ----------

try:
    from pyspark.ml.feature import Imputer
    imp = Imputer(strategy="median", inputCols=["annual_revenue", "years_in_business"],
                  outputCols=["annual_revenue_imp", "years_in_business_imp"])
    display(imp.fit(policies).transform(policies)
            .select("annual_revenue", "annual_revenue_imp", "years_in_business", "years_in_business_imp").limit(10))
except Exception as e:
    print("pyspark.ml not available on this compute:", str(e)[:200])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. One-hot encoding: when and when not
# MAGIC | Model | Categorical handling used in this project | Why |
# MAGIC |---|---|---|
# MAGIC | Logistic regression | One-hot (`OneHotEncoder(handle_unknown="ignore")`) | A linear model would otherwise read category codes as ordered numbers |
# MAGIC | Gradient-boosted trees | Ordinal codes (`OrdinalEncoder`) | Trees split on codes fine; one-hot would add many sparse columns for nothing |
# MAGIC
# MAGIC One-hot is a poor fit for very high-cardinality features (for example ZIP code), where it creates thousands of
# MAGIC sparse columns; consider grouping, target encoding, or hashing.
# MAGIC
# MAGIC Spark ML version for reference: `StringIndexer` then `OneHotEncoder` then `VectorAssembler`.

# COMMAND ----------

try:
    from pyspark.ml import Pipeline as SparkPipeline
    from pyspark.ml.feature import OneHotEncoder, StringIndexer
    enc = SparkPipeline(stages=[
        StringIndexer(inputCol="line_of_business", outputCol="lob_idx", handleInvalid="keep"),
        OneHotEncoder(inputCols=["lob_idx"], outputCols=["lob_ohe"]),
    ]).fit(policies)
    display(enc.transform(policies).select("line_of_business", "lob_idx", "lob_ohe").distinct())
except Exception as e:
    print("pyspark.ml not available on this compute:", str(e)[:200])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exam check
# MAGIC 1. Which call adds percentiles to Spark summary statistics?
# MAGIC 2. Why did ±3 sd and IQR flag different numbers of late reports?
# MAGIC 3. Mean or median for `annual_revenue`, and why?
# MAGIC 4. Which of the two models here needs one-hot encoding?
# MAGIC 5. After training on `log1p(incurred)`, what must you do before computing RMSE in dollars?
# MAGIC
# MAGIC Next: `02_feature_store`.
