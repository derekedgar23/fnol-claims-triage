# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Hyperparameter tuning and cross-validation
# MAGIC **Exam Section 3.** Grid search with k-fold cross-validation (and how many models it trains), Bayesian search with
# MAGIC Hyperopt `fmin`, the same search in Optuna, and Spark ML estimators/transformers with `CrossValidator`.
# MAGIC The best configuration is logged as a new candidate run.

# COMMAND ----------

# MAGIC %pip install -q "databricks-feature-engineering>=0.12" "scikit-learn>=1.4" hyperopt optuna

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

import json

import cloudpickle
import mlflow
import numpy as np
import pandas as pd
from databricks.feature_engineering import FeatureEngineeringClient
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import claims_triage
from claims_triage.evaluation import count_grid_models
from claims_triage.tracking import load_training_data, log_candidate
from claims_triage.train import gbm_classifier

cloudpickle.register_pickle_by_value(claims_triage)
mlflow.sklearn.autolog(disable=True)
fe = FeatureEngineeringClient()
data = load_training_data(spark, fe, FQ, FEATURE_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Grid search + cross-validation: count the models
# MAGIC Grid: 2 learning rates x 2 depths x 2 leaf sizes = **8 combinations**. With **3 folds** that is 24 fits, plus one
# MAGIC final refit on all the training data with the best combination = **25 models**.
# MAGIC
# MAGIC Cross-validation vs a single train/validation split: CV gives a more reliable estimate and uses every row for both
# MAGIC training and validation, at about k times the cost.

# COMMAND ----------

grid = {"model__learning_rate": [0.05, 0.1], "model__max_depth": [3, 6], "model__min_samples_leaf": [20, 80]}
folds = 3
print("expected fits:", count_grid_models(grid, folds=folds, refit=True))

search = GridSearchCV(gbm_classifier({"max_iter": 150}), grid, cv=StratifiedKFold(folds, shuffle=True, random_state=0),
                      scoring="average_precision", refit=True, n_jobs=-1)
search.fit(data.X_train, data.y_train)

n_combos = len(search.cv_results_["params"])
print(f"combinations={n_combos}, folds={folds}, fits={n_combos * folds} + 1 refit")
print("best:", search.best_params_, "CV PR AUC:", round(search.best_score_, 4))
display(pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
        .sort_values("rank_test_score"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Bayesian search with Hyperopt
# MAGIC - `fmin` **minimizes** the `loss` the objective returns, so return **negative** PR AUC.
# MAGIC - The objective returns `{"loss": ..., "status": STATUS_OK}`.
# MAGIC - `tpe.suggest` is Bayesian (Tree-structured Parzen Estimator); `rand.suggest` is random search.
# MAGIC - Each trial is logged as a **nested** MLflow run under one parent run.
# MAGIC
# MAGIC **Parallelism:** on classic compute with an ML runtime, `trials=SparkTrials(parallelism=4)` runs trials of
# MAGIC single-node models (like this sklearn model) in parallel on the workers. Use plain `Trials()` for models that are
# MAGIC already distributed (Spark MLlib). Serverless compute has no SparkContext for `SparkTrials`, so this notebook uses
# MAGIC `Trials()`. Note: open-source Hyperopt is no longer maintained (last bundled in DBR 16.4 LTS ML); Databricks now
# MAGIC recommends Optuna (section 3) for single-node tuning. The exam still asks about `fmin`.

# COMMAND ----------

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

space = {
    "learning_rate": hp.loguniform("learning_rate", np.log(0.02), np.log(0.3)),
    "max_depth": hp.quniform("max_depth", 3, 10, 1),
    "min_samples_leaf": hp.quniform("min_samples_leaf", 10, 200, 10),
    "l2_regularization": hp.uniform("l2_regularization", 0.0, 5.0),
    "max_iter": hp.quniform("max_iter", 100, 500, 50),
}


def objective(params):
    with mlflow.start_run(nested=True):
        model = gbm_classifier(params).fit(data.X_train, data.y_train)
        pr_auc = average_precision_score(data.y_val, model.predict_proba(data.X_val)[:, 1])
        mlflow.log_params(params)
        mlflow.log_metric("val_pr_auc", pr_auc)
        return {"loss": -pr_auc, "status": STATUS_OK}


with mlflow.start_run(run_name="hyperopt_tpe_search"):
    trials = Trials()
    best_raw = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials,
                    rstate=np.random.default_rng(42))
    best_hyperopt = {k: (int(v) if k in ("max_depth", "min_samples_leaf", "max_iter") else float(v))
                     for k, v in best_raw.items()}
    mlflow.log_dict(best_hyperopt, "best_params.json")
    mlflow.log_metric("best_val_pr_auc", -min(trials.losses()))

print("Hyperopt best:", best_hyperopt, "val PR AUC:", round(-min(trials.losses()), 4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. The same search in Optuna (the current Databricks recommendation)
# MAGIC Optuna maximizes or minimizes (`direction=`), and the search space is defined inside the objective.

# COMMAND ----------

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


def optuna_objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200, step=10),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 5.0),
        "max_iter": trial.suggest_int("max_iter", 100, 500, step=50),
    }
    model = gbm_classifier(params).fit(data.X_train, data.y_train)
    return average_precision_score(data.y_val, model.predict_proba(data.X_val)[:, 1])


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(optuna_objective, n_trials=20)
print("Optuna best:", study.best_params, "val PR AUC:", round(study.best_value, 4))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Spark ML: estimators, transformers, Pipeline, CrossValidator
# MAGIC - **Transformer**: `.transform(df)` returns a new DataFrame (`VectorAssembler`, any fitted model).
# MAGIC - **Estimator**: `.fit(df)` returns a fitted model (`StringIndexer`, `Imputer`, `LogisticRegression`, and `Pipeline` itself).
# MAGIC - `CrossValidator` with a 2 x 2 grid and 3 folds trains 12 models, then refits the best on all the data.
# MAGIC
# MAGIC Needs `pyspark.ml` support on your compute. If it is unavailable, the cell says so and you can skip it.

# COMMAND ----------

try:
    from pyspark.ml import Pipeline as SparkPipeline
    from pyspark.ml.classification import LogisticRegression as SparkLR
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

    sdf = (spark.table(f"{FQ}.claims_training_base").join(spark.table(FEATURE_TABLE), "policy_id")
           .fillna({"cause_of_loss": "UNKNOWN"}))
    train_sdf, test_sdf = sdf.randomSplit([0.8, 0.2], seed=42)
    num = ["annual_revenue", "years_in_business", "report_lag_days", "prior_claims_3y", "injury_flag", "attorney_at_fnol"]
    lr = SparkLR(labelCol="large_loss", featuresCol="features")
    pipe = SparkPipeline(stages=[
        StringIndexer(inputCols=["line_of_business", "cause_of_loss"], outputCols=["lob_idx", "cause_idx"],
                      handleInvalid="keep"),                                                  # estimator
        OneHotEncoder(inputCols=["lob_idx", "cause_idx"], outputCols=["lob_ohe", "cause_ohe"]),  # estimator
        Imputer(strategy="median", inputCols=num, outputCols=[f"{c}_imp" for c in num]),       # estimator
        VectorAssembler(inputCols=["lob_ohe", "cause_ohe"] + [f"{c}_imp" for c in num],
                        outputCol="features"),                                                # transformer
        lr,                                                                                   # estimator
    ])
    param_grid = ParamGridBuilder().addGrid(lr.regParam, [0.0, 0.01]).addGrid(lr.elasticNetParam, [0.0, 0.5]).build()
    evaluator = BinaryClassificationEvaluator(labelCol="large_loss", metricName="areaUnderPR")
    cv = CrossValidator(estimator=pipe, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3, seed=42)
    print("models trained during CV:", len(param_grid) * 3, "(+1 refit)")
    cv_model = cv.fit(train_sdf)
    print("CV PR AUC per param set:", [round(m, 4) for m in cv_model.avgMetrics])
    print("Test PR AUC:", round(evaluator.evaluate(cv_model.transform(test_sdf)), 4))
except Exception as e:
    print("Skipped Spark ML section:", str(e)[:300])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Log the tuned model as a candidate
# MAGIC Uses the Hyperopt result, evaluated on validation and test like the runs in notebook 03. The params are also saved
# MAGIC to `conf/best_params.json` content below, which the retraining job uses (copy them into the repo file).

# COMMAND ----------

run_id, m = log_candidate(fe, data, gbm_classifier(best_hyperopt), "gbm_tuned_hyperopt",
                          tags={"model_family": "hist_gbm", "class_weight": "None", "tuned": "hyperopt"},
                          params={"model_family": "hist_gbm", **best_hyperopt})
print(run_id, {k: round(v, 4) for k, v in m.items()})
print("For conf/best_params.json:\n" + json.dumps(best_hyperopt, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exam check
# MAGIC 1. 3 x 2 x 3 grid with 5 folds: how many models, with and without the final refit?
# MAGIC 2. Why does the Hyperopt objective return negative PR AUC?
# MAGIC 3. `SparkTrials` or `Trials` for tuning Spark MLlib models? For sklearn models on a cluster?
# MAGIC 4. Grid vs random vs Bayesian search: which uses earlier results to choose the next trial?
# MAGIC 5. Is a Spark `Pipeline` an estimator or a transformer? What does `fit` return?
# MAGIC
# MAGIC Next: `05_register_promote`.
