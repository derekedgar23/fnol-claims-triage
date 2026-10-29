"""Databricks-side helpers shared by notebooks and the retraining job.

These import Databricks/MLflow packages lazily, so the rest of the package (and the unit tests)
work without them.
"""
from __future__ import annotations

from dataclasses import dataclass

from .evaluation import classification_metrics
from .features import FNOL_FEATURES, POLICY_FEATURES
from .train import feature_frame


@dataclass
class TrainingData:
    ts_train: object          # FeatureEngineering TrainingSet for the train split (used when logging)
    X_train: object
    y_train: object
    X_val: object
    y_val: object
    X_test: object
    y_test: object


def load_training_data(spark, fe, fq: str, feature_table: str) -> TrainingData:
    """Claims + policy features via FeatureLookup, split deterministically by claim_id."""
    from databricks.feature_engineering import FeatureLookup
    from pyspark.sql import functions as F

    from .splits import spark_split_col

    lookups = [FeatureLookup(table_name=feature_table, lookup_key="policy_id", feature_names=POLICY_FEATURES)]
    labels = (spark.table(f"{fq}.claims_training_base")
              .select("claim_id", "policy_id", *FNOL_FEATURES, "large_loss")
              .withColumn("split", spark_split_col("claim_id")))

    def ts(split):
        return fe.create_training_set(df=labels.where(F.col("split") == split).drop("split"),
                                      feature_lookups=lookups, label="large_loss", exclude_columns=["claim_id"])

    ts_train = ts("train")
    frames = [t.load_df().toPandas() for t in (ts_train, ts("val"), ts("test"))]
    (tr, va, te) = frames
    return TrainingData(ts_train, feature_frame(tr), tr["large_loss"], feature_frame(va), va["large_loss"],
                        feature_frame(te), te["large_loss"])


def log_candidate(fe, data: TrainingData, model, run_name: str, tags: dict | None = None,
                  params: dict | None = None) -> tuple[str, dict]:
    """Fit, evaluate on val and test, and log metrics, an evaluation figure and three model flavors.

    Logged models:
      sklearn_model : raw sklearn pipeline (pandas batch inference, serving wrapper)
      proba_model   : pyfunc returning probabilities (mlflow.pyfunc.spark_udf)
      fs_model      : feature-store-aware model (fe.score_batch looks up policy features itself)
    """
    import matplotlib.pyplot as plt
    import mlflow
    from mlflow.models import infer_signature
    from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

    from .triage import LargeLossProbability

    with mlflow.start_run(run_name=run_name) as run:
        model.fit(data.X_train, data.y_train)
        val_p = model.predict_proba(data.X_val)[:, 1]
        test_p = model.predict_proba(data.X_test)[:, 1]
        val_m, test_m = classification_metrics(data.y_val, val_p), classification_metrics(data.y_test, test_p)

        mlflow.set_tags({"candidate": "true", **(tags or {})})
        if params:
            mlflow.log_params(params)
        mlflow.log_metrics({**{f"val_{k}": v for k, v in val_m.items()},
                            **{f"test_{k}": v for k, v in test_m.items()}})

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        PrecisionRecallDisplay.from_predictions(data.y_val, val_p, ax=ax[0], name=run_name)
        ax[0].axhline(float(data.y_val.mean()), ls="--", c="grey", label="base rate")
        ax[0].legend()
        ConfusionMatrixDisplay.from_predictions(data.y_val, (val_p >= 0.5).astype(int), ax=ax[1], colorbar=False)
        ax[1].set_title("Validation confusion matrix @ 0.5")
        plt.tight_layout()
        mlflow.log_figure(fig, "evaluation.png")
        plt.close(fig)

        sig = infer_signature(data.X_val, val_p)
        example = data.X_val.head(3)
        mlflow.sklearn.log_model(model, "sklearn_model", signature=sig, input_example=example)
        mlflow.pyfunc.log_model("proba_model", python_model=LargeLossProbability(model), signature=sig,
                                input_example=example)
        fe.log_model(model=LargeLossProbability(model), artifact_path="fs_model", flavor=mlflow.pyfunc,
                     training_set=data.ts_train)
        return run.info.run_id, val_m


def _test_claims(spark, fq: str):
    from .features import FNOL_FEATURES
    from .splits import spark_split_col
    return (spark.table(f"{fq}.claims_training_base")
            .withColumn("split", spark_split_col("claim_id")).where("split = 'test'")
            .select("claim_id", "policy_id", *FNOL_FEATURES, "large_loss"))


def score_with_alias(fe, spark, model_name: str, alias: str, df, feature_table: str):
    """Score df (claim keys + FNOL fields) with models:/<name>@<alias>, returning a 'prediction' column.

    Production path: fe.score_batch, which looks up policy features itself. If the feature-store
    scoring path fails in your environment (for example a serialization issue with custom pyfunc
    classes on remote workers), fall back to joining the feature table and applying the same run's
    proba_model as a Spark UDF, and say so.
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    from pyspark.sql import functions as F

    from .features import ALL_FEATURES

    try:
        out = fe.score_batch(model_uri=f"models:/{model_name}@{alias}", df=df, result_type="double")
        out.limit(1).collect()          # force execution so failures surface here
        return out
    except Exception as e:              # noqa: BLE001
        print(f"fe.score_batch failed ({str(e)[:160]}...). Falling back to feature join + spark_udf.")
        run_id = MlflowClient().get_model_version_by_alias(model_name, alias).run_id
        udf = mlflow.pyfunc.spark_udf(spark, f"runs:/{run_id}/proba_model", result_type="double",
                                      env_manager="local")
        feats = df.join(spark.table(feature_table), "policy_id")
        return feats.withColumn("prediction", udf(F.struct(*ALL_FEATURES)))


def evaluate_alias(fe, spark, fq: str, model_name: str, alias: str, feature_table: str | None = None) -> dict | None:
    """Score the held-out test claims with models:/<name>@<alias>.

    Only keys and FNOL fields are passed in; the model looks up policy features itself.
    Returns None if the alias does not exist yet.
    """
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    try:
        MlflowClient().get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return None
    test = _test_claims(spark, fq)
    scored = score_with_alias(fe, spark, model_name, alias, test.drop("large_loss"),
                              feature_table or f"{fq}.policy_features")
    pdf = (scored.select("claim_id", "prediction").join(test.select("claim_id", "large_loss"), "claim_id")
           .toPandas())
    return classification_metrics(pdf["large_loss"], pdf["prediction"])


def tier_column(prob_col: str = "prediction"):
    """Spark Column with the same tiers as triage.assign_tier."""
    from pyspark.sql import functions as F

    from .triage import TIER_THRESHOLDS
    p = F.col(prob_col)
    return (F.when(p >= TIER_THRESHOLDS["HIGH"], "HIGH")
            .when(p >= TIER_THRESHOLDS["ELEVATED"], "ELEVATED").otherwise("STANDARD"))
