# Lakeflow Spark Declarative Pipeline (formerly Delta Live Tables): streaming FNOL triage.
#
# Pipeline settings:
#   default catalog/schema : workspace / claims_ml
#   environment            : add mlflow and scikit-learn as dependencies
#   configuration          : model_uri = runs:/<run_id>/proba_model   (printed by notebook 06)
#
# New FNOL rows appended to fnol_stream_source are joined to the policy features (stream-static join) and scored
# with the model loaded as a Spark UDF. The pipeline autoscales with event volume.
import mlflow
from pyspark import pipelines as dp
from pyspark.sql import functions as F

MODEL_URI = spark.conf.get("model_uri")
FEATURES = ["line_of_business", "insured_state", "cause_of_loss", "policy_limit", "deductible", "annual_revenue",
            "years_in_business", "total_insured_value", "prior_claims_3y", "report_lag_days", "injury_flag",
            "attorney_at_fnol", "loss_month"]   # must match claims_triage.features.ALL_FEATURES

predict = mlflow.pyfunc.spark_udf(spark, MODEL_URI, result_type="double", env_manager="local")


@dp.table(name="fnol_triage_stream", comment="FNOL claims scored as they arrive")
def fnol_triage_stream():
    claims = spark.readStream.table("fnol_stream_source")
    policies = spark.read.table("policy_features")
    p = F.col("large_loss_probability")
    return (claims.join(policies, "policy_id")
            .withColumn("large_loss_probability", predict(F.struct(*FEATURES)))
            .withColumn("triage_tier", F.when(p >= 0.5, "HIGH").when(p >= 0.2, "ELEVATED").otherwise("STANDARD"))
            .withColumn("scored_at", F.current_timestamp()))
