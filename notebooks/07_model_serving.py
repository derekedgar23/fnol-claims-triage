# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · Real-time serving with a custom model and a traffic split
# MAGIC **Exam Section 4.** Wrap the model in a custom `mlflow.pyfunc.PythonModel` that returns a probability, a triage
# MAGIC tier and reason codes; register it; serve it on a Model Serving endpoint; query it; and split traffic 90/10
# MAGIC between the champion and the baseline.
# MAGIC
# MAGIC Real time fits when a claims intake app needs the tier the moment a claim is entered. The request carries all
# MAGIC model inputs, so no online feature store is needed (see notebook 02 for the alternative).
# MAGIC
# MAGIC **Cost note:** endpoints use compute while running. This notebook enables scale to zero and deletes the endpoint
# MAGIC at the end. Free Edition limits the number of active endpoints; if creation fails with a quota message, read the
# MAGIC cells as reference.

# COMMAND ----------

# MAGIC %pip install -q "scikit-learn>=1.4" "databricks-sdk>=0.40"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

import time

import cloudpickle
import mlflow
import numpy as np
import pandas as pd
import sklearn
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, Route, ServedEntityInput, TrafficConfig
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

import claims_triage
from claims_triage.features import ALL_FEATURES
from claims_triage.triage import TriageModel

cloudpickle.register_pickle_by_value(claims_triage)   # the served model must not need this repo installed
client = MlflowClient()
w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build and register the custom pyfunc
# MAGIC One service version wraps the current champion; another wraps the previous champion (the baseline), so we can
# MAGIC split traffic between them.

# COMMAND ----------

# Infer the signature from enough rows to include missing values, so nullable inputs are marked optional
sig_sample = (spark.table(f"{FQ}.claims_to_score").join(spark.table(FEATURE_TABLE), "policy_id")
              .limit(1000).toPandas()[ALL_FEATURES])
example = sig_sample.head(5)


def register_service_version(alias):
    run_id = client.get_model_version_by_alias(MODEL_NAME, alias).run_id
    sk_model = mlflow.sklearn.load_model(f"runs:/{run_id}/sklearn_model")
    wrapper = TriageModel(sk_model)
    sig = infer_signature(sig_sample, wrapper.predict(None, sig_sample))
    with mlflow.start_run(run_name=f"triage_service_from_{alias}") as run:
        mlflow.pyfunc.log_model("triage_model", python_model=wrapper, signature=sig, input_example=example,
                                pip_requirements=[f"scikit-learn=={sklearn.__version__}",
                                                  f"pandas=={pd.__version__}", f"numpy=={np.__version__}",
                                                  f"cloudpickle=={cloudpickle.__version__}"],
                                registered_model_name=SERVICE_MODEL_NAME)
    versions = client.search_model_versions(f"name = '{SERVICE_MODEL_NAME}'")
    return max(int(v.version) for v in versions if v.run_id == run.info.run_id)


try:
    previous_alias = "previous_champion"
    client.get_model_version_by_alias(MODEL_NAME, previous_alias)
except Exception:
    previous_alias = "champion"     # nothing was promoted in notebook 05; serve the same model twice

v_champion = register_service_version("champion")
v_previous = register_service_version(previous_alias)
client.set_registered_model_alias(SERVICE_MODEL_NAME, "champion", v_champion)
print(f"{SERVICE_MODEL_NAME}: champion v{v_champion}, previous v{v_previous}")

# Local check before deploying anything
local = mlflow.pyfunc.load_model(f"models:/{SERVICE_MODEL_NAME}@champion")
display(local.predict(example))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the endpoint with a 90/10 traffic split
# MAGIC Two **served entities** on one endpoint, and `traffic_config.routes` with a percentage for each. All served models
# MAGIC on one endpoint must be the same model type. Creation takes several minutes.

# COMMAND ----------

def served(name, version):
    return ServedEntityInput(name=name, entity_name=SERVICE_MODEL_NAME, entity_version=str(version),
                             workload_size="Small", scale_to_zero_enabled=True)


config = EndpointCoreConfigInput(
    name=ENDPOINT_NAME,                  # required by current databricks-sdk versions
    served_entities=[served("champion", v_champion), served("previous", v_previous)],
    traffic_config=TrafficConfig(routes=[Route(served_model_name="champion", traffic_percentage=90),
                                         Route(served_model_name="previous", traffic_percentage=10)]),
)

existing = [e.name for e in w.serving_endpoints.list()]
if ENDPOINT_NAME in existing:
    w.serving_endpoints.update_config(name=ENDPOINT_NAME, served_entities=config.served_entities,
                                      traffic_config=config.traffic_config)
else:
    w.serving_endpoints.create(name=ENDPOINT_NAME, config=config)

for _ in range(60):                      # wait up to ~30 minutes
    state = w.serving_endpoints.get(ENDPOINT_NAME).state
    ready = getattr(state.ready, "value", state.ready)                  # "READY" or "NOT_READY"
    update = getattr(state.config_update, "value", state.config_update)  # "IN_PROGRESS", "UPDATE_FAILED", ...
    print(time.strftime("%H:%M:%S"), ready, update)
    if update == "UPDATE_FAILED":
        raise RuntimeError("Endpoint update failed. Open Serving in the sidebar and check the build/service logs.")
    if ready == "READY" and update != "IN_PROGRESS":
        break
    time.sleep(30)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query it
# MAGIC SDK below. Over REST it is a POST to `https://<workspace-host>/serving-endpoints/fnol-triage/invocations` with a
# MAGIC bearer token and a body such as `{"dataframe_records": [{...one claim...}]}` (`dataframe_split`, `inputs` and
# MAGIC `instances` are also accepted).

# COMMAND ----------

records = example.head(3).astype(object).where(example.head(3).notna(), None).to_dict(orient="records")
response = w.serving_endpoints.query(name=ENDPOINT_NAME, dataframe_records=records)
display(pd.DataFrame(response.predictions))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Shift traffic (canary rollout) and clean up

# COMMAND ----------

w.serving_endpoints.update_config(
    name=ENDPOINT_NAME,
    served_entities=config.served_entities,
    traffic_config=TrafficConfig(routes=[Route(served_model_name="champion", traffic_percentage=100),
                                         Route(served_model_name="previous", traffic_percentage=0)]),
)
print("traffic now 100/0. Delete the endpoint when you are done:")

# COMMAND ----------

w.serving_endpoints.delete(ENDPOINT_NAME)
print("deleted", ENDPOINT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exam check
# MAGIC 1. Why a custom pyfunc instead of serving the sklearn model directly?
# MAGIC 2. What two pieces of endpoint config split traffic between models?
# MAGIC 3. What does scale to zero trade off?
# MAGIC 4. If the request only contained `policy_id`, what would the endpoint need?
