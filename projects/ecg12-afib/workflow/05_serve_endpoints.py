# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Model Serving: create / update the 12-lead endpoints the app calls, from `champion`
# MAGIC
# MAGIC **SETUP job, task 5 of 5** (after `04_validate_and_promote`, so it runs only when both
# MAGIC challengers passed the gate). Serves the two UC models at their **`champion`** versions as
# MAGIC scale-to-zero Model Serving endpoints, each with an **AI Gateway inference table**, then
# MAGIC smoke-queries each with synthetic 12-lead ECGs.
# MAGIC
# MAGIC | Endpoint | Serves | Trained by | Promoted by |
# MAGIC |---|---|---|---|
# MAGIC | `bio_ecg12_arr` (triage) | `bio_ecg12_arr@champion`, pinned to its version | `02_` | `04_` |
# MAGIC | `bio_ecg12_afib` (classify) | `bio_ecg12_afib@champion`, pinned to its version | `03_` | `04_` |
# MAGIC
# MAGIC Each endpoint has the same name as its model (`common/config.py`). Endpoint names are
# MAGIC workspace-wide, so these names are this demo's own: the first run creates both endpoints, and a
# MAGIC re-run after a retrain + promotion moves each one to the new champion version. Inference tables
# MAGIC are `<endpoint>_payload` in the same schema (`bio_ecg12_arr_payload`, `bio_ecg12_afib_payload`,
# MAGIC read by the dashboard's live-serving section). Never touched: the first 12-lead run's endpoints
# MAGIC `arr_model_12l` / `afib_demo_12l` (schema `ekg_afib`) and the live 8-lead `arr_model_mmt` /
# MAGIC `afib_demo_mmt`.

# COMMAND ----------

# DBTITLE 1,Where the mlflow-skinny / databricks-sdk pins come from
# MAGIC %md
# MAGIC Pins = what serverless environment 5 and the AI v5 environment (`databricks_ai_v5`, used by the
# MAGIC training and gate tasks) ship: `mlflow-skinny==3.8.1`, `databricks-sdk==0.67.0` (the ML v5
# MAGIC environment of the batch scoring tasks ships the same mlflow-skinny 3.8.1). Source: the
# MAGIC [AI Runtime environment 5 release notes](https://docs.databricks.com/aws/en/release-notes/serverless/environment-version/five-gpu)
# MAGIC and its package list `requirements-ai-gpu-5.txt` (checked 2026-10-29). They were checked locally to
# MAGIC provide every API this notebook calls: `mlflow.deployments` (create / update / list / predict),
# MAGIC `MlflowClient.get_model_version_by_alias`, and the SDK's `put_ai_gateway(inference_table_config=...)`.
# MAGIC On an environment that already ships them, the install changes nothing.

# COMMAND ----------

# DBTITLE 1,Install pinned mlflow-skinny + databricks-sdk and the generator's deps
# MAGIC %pip install "mlflow-skinny==3.8.1" "databricks-sdk==0.67.0"
# MAGIC %pip install --no-deps "neurokit2==0.2.12" "PyWavelets==1.4.1"

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Shared config (common/ is one folder up from workflow/)
import os
import sys

sys.path.insert(0, os.path.abspath(".."))   # notebook working dir = workflow/

from common import config, scoring
from common.ecg_signal_gen import generate_ecg_batch

# COMMAND ----------

# DBTITLE 1,Parameters
dbutils.widgets.text("catalog_name", "", label="UC_catalog_name (required)")
dbutils.widgets.text("schema_name", config.SCHEMA, label="UC_schema_name")
catalog, schema = config.uc_location(dbutils.widgets.get("catalog_name"), dbutils.widgets.get("schema_name"))

# (model, endpoint) pairs, triage first.
SERVED = (
    (config.ARR_MODEL_NAME, config.ARR_ENDPOINT),
    (config.AFIB_MODEL_NAME, config.AFIB_ENDPOINT),
)
print(f"{catalog}.{schema}: {SERVED}")

# COMMAND ----------

# DBTITLE 1,Clients
import time
import mlflow
from mlflow import MlflowClient
from mlflow.deployments import get_deploy_client
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import AiGatewayInferenceTableConfig

mlflow.set_registry_uri("databricks-uc")
deploy_client = get_deploy_client("databricks")
w = WorkspaceClient()
mc = MlflowClient(registry_uri="databricks-uc")

# COMMAND ----------

# DBTITLE 1,Helpers — wait for an endpoint, create or update one at the champion version
def wait_for_endpoint_ready(name, timeout=2400, poll=30):
    start = time.time()
    while time.time() - start < timeout:
        ep = w.serving_endpoints.get(name=name)
        if ep.pending_config is None:
            state = ep.state.ready.value if (ep.state and ep.state.ready) else str(ep.state)
            print(f"  '{name}' settled (state: {state})")
            return ep
        print(f"  '{name}' updating... {int(time.time() - start)}s")
        time.sleep(poll)
    raise TimeoutError(f"'{name}' not ready after {timeout}s")


def deploy(model_name, endpoint_name):
    full = config.uc_name(model_name, catalog, schema)
    ver = scoring.resolve_alias(mc, full, config.CHAMPION_ALIAS).version
    served = f"{endpoint_name}-{ver}"
    endpoint_config = {
        "served_entities": [{
            "name": served, "entity_name": full, "entity_version": ver,
            "workload_size": "Small", "scale_to_zero_enabled": True,
        }],
        "traffic_config": {"routes": [{"served_model_name": served, "traffic_percentage": 100}]},
    }
    existing = [e["name"] for e in deploy_client.list_endpoints()]
    if endpoint_name in existing:
        wait_for_endpoint_ready(endpoint_name)
        print(f"updating {endpoint_name} -> {full} v{ver}")
        deploy_client.update_endpoint(endpoint=endpoint_name, config=endpoint_config)
    else:
        print(f"creating {endpoint_name} -> {full} v{ver}")
        deploy_client.create_endpoint(name=endpoint_name, config=endpoint_config)
    return full, ver

# COMMAND ----------

# DBTITLE 1,Issue both deploys (they build in parallel)
for model_name, endpoint_name in SERVED:
    full, ver = deploy(model_name, endpoint_name)
    print(f"{endpoint_name}: {full} v{ver}")

# COMMAND ----------

# DBTITLE 1,Wait for both ready + enable AI Gateway inference tables (<endpoint>_payload)
for _, endpoint_name in SERVED:
    wait_for_endpoint_ready(endpoint_name, timeout=3000)
    w.serving_endpoints.put_ai_gateway(
        name=endpoint_name,
        inference_table_config=AiGatewayInferenceTableConfig(
            enabled=True, catalog_name=catalog, schema_name=schema, table_name_prefix=endpoint_name,
        ),
    )
    print(f"AI Gateway inference table enabled for '{endpoint_name}' -> "
          f"{config.payload_table(endpoint_name, catalog, schema)}")

# COMMAND ----------

# DBTITLE 1,Smoke-query both endpoints with synthetic 12-lead ECGs (same payload as the app)
# One NSR + one AFib sample from the shared generator, sent ONE ECG PER REQUEST as a (1, 12, 2500)
# tensor, like the app. One ECG is ~650 KB of JSON; the AI Gateway inference table does not log a
# request larger than 1 MiB (it records MAX_REQUEST_SIZE_EXCEEDED instead), so a 2-ECG request would
# be served but not logged.
ecg, labels, _, lead_names = generate_ecg_batch(2, frac_afib=0.5, frac_other=0.0, seed=7, n_leads=config.N_LEADS)
print("sample ecg:", ecg.shape, "labels:", labels.tolist(), "leads:", lead_names)

for _, endpoint_name in SERVED:
    for one_ecg in ecg:
        resp = deploy_client.predict(endpoint=endpoint_name, inputs={"inputs": one_ecg[None].tolist()})
        print(f"{endpoint_name} ->", resp)
