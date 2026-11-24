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
# MAGIC | `<endpoint_prefix>_arr` (triage) | `bio_ecg12_arr@champion`, pinned to its version | `02_` | `04_` |
# MAGIC | `<endpoint_prefix>_afib` (classify) | `bio_ecg12_afib@champion`, pinned to its version | `03_` | `04_` |
# MAGIC
# MAGIC **Endpoint names are workspace-wide**, unlike the models and tables (which live in this
# MAGIC deployment's catalog.schema: the SETUP job's first task, `01_`, has already stopped the job if
# MAGIC another deployment owns that schema), so they come from the `endpoint_prefix` parameter (the bundle
# MAGIC variable of the same name, default `bio_ecg12`; `config.endpoint_names`). The first run creates
# MAGIC both endpoints, and a re-run after a retrain + promotion moves each one to the new champion
# MAGIC version. A re-run with nothing new sends no update: an endpoint that already serves the champion
# MAGIC version (same served model, size, scale-to-zero and traffic), no update in progress and no
# MAGIC pending config, gets no new served-model config. The AI Gateway step below still runs every time;
# MAGIC the first time it turns on the inference table and usage tracking, the endpoint can redeploy once
# MAGIC (`docs/express-deployment.md`, What to expect). One whose last update failed is updated again,
# MAGIC and an update that fails stops the task with Model Serving's message.
# MAGIC **An existing endpoint is updated only if it already serves this deployment's model**
# MAGIC (`<catalog>.<schema>.<model>`): one that serves anything else belongs to another deployment, and
# MAGIC the task stops before changing either endpoint, asking for a unique `endpoint_prefix`.
# MAGIC Inference tables are `<model>_payload` in the same schema (`bio_ecg12_arr_payload`,
# MAGIC `bio_ecg12_afib_payload`, read by the dashboard's live-serving section): AI Gateway's table
# MAGIC prefix is the model name, so the table names do not depend on the endpoint names. AI Gateway
# MAGIC logs the endpoints' requests (the whole ECG) and responses (the scores) there, with a delay ("You
# MAGIC can expect logs to be available within 1 hour of a request") and with documented gaps: a request
# MAGIC or response over 1 MiB is not logged, and for custom model endpoints such as these "logs may not
# MAGIC be recorded for any 4xx or 5xx errors"
# MAGIC ([inference tables for serving endpoints](https://docs.databricks.com/aws/en/ai-gateway/inference-tables-serving-endpoints)). So these
# MAGIC tables hold the ECGs from logged requests; with real data, treat them as patient data. AI Gateway **usage tracking** is on too: usage records for the endpoints' requests land in
# MAGIC the system table `system.serving.endpoint_usage` (the endpoints' served models in
# MAGIC `system.serving.served_entities`);
# MAGIC only account admins can query these, unless one grants access to them.
# MAGIC
# MAGIC With a `remove_after` date (`YYYY-MM-DD`, the bundle variable of the same name), both endpoints
# MAGIC also get the tag `removeAfter` with that date, as the jobs do; empty (the default) adds no tag
# MAGIC and leaves an existing one as it is.
# MAGIC
# MAGIC **Express deployment.** On AWS and Azure, Model Serving uses
# MAGIC [express deployment](https://docs.databricks.com/aws/en/machine-learning/model-serving/express-deployments)
# MAGIC for these endpoints when the models are eligible (they were in our tests on both clouds): it
# MAGIC prepares each model's Python environment once (shared by models with identical requirements)
# MAGIC instead of building a container image, which took about 15 of the 21 minutes an endpoint needed
# MAGIC when this demo was tested on that path. There is no switch: it is automatic for an eligible
# MAGIC model (logged with MLflow from Databricks compute, with a recorded runtime, Python version and
# MAGIC dependencies that suit the serving compute). `02_` / `03_` log the models with
# MAGIC `MODEL_PIP_REQUIREMENTS` (`common/config.py`), plain PyPI versions with no CUDA tag, which avoids
# MAGIC the dependency-related blockers the docs list (a workspace setting such as a customer-managed
# MAGIC key is outside the project's control). Because it is automatic, the platform's own records tell
# MAGIC which path an endpoint took: the served model's build log and the endpoint's events, read below.
# MAGIC Tested on AWS and Azure; the Google Cloud page still requires `env_pack` at registration, which
# MAGIC this project doesn't use, so expect the standard path there (not yet tested). Once both
# MAGIC endpoints are ready this notebook prints the path each served model took, for example
# MAGIC `bio_ecg12_arr-2: EXPRESS deployment (environment prepared in 4m00s; no container image build)`,
# MAGIC from its build log and the endpoint's events (`common/deployment_path.py`), and tags the endpoint
# MAGIC `deployment` = `express`, `standard` (a container image was built: the endpoint serves, but
# MAGIC updates are slower) or `unknown` (the build log or the events could not be read, or they
# MAGIC named neither path; not evidence against express). Either is a warning, not a failure. In
# MAGIC the Serving UI, the endpoint's **Logs > Build logs** (pick the served model, e.g.
# MAGIC `<endpoint>-2`) start with "Express deployment is preparing ..." or "This environment was
# MAGIC prepared ahead of time by express deployment ...". The events (`databricks api get
# MAGIC /api/2.0/serving-endpoints/<name>/events`) show *Model artifact preparation* for express and
# MAGIC *Container image creation* for an image build. More in `docs/express-deployment.md`.
# MAGIC
# MAGIC Parameters: `catalog_name` (required), `schema_name`, `endpoint_prefix`, `remove_after`.

# COMMAND ----------

# DBTITLE 1,Why this notebook installs mlflow-skinny 3.13.0 and no SDK
# MAGIC %md
# MAGIC This task runs on standard serverless **environment 6** (the SETUP job names it, so it does not move
# MAGIC with the workspace default). Environment 6 ships `mlflow-skinny` 3.12.0, `databricks-sdk` 0.122.0
# MAGIC and `protobuf` 6.33.5. The notebook installs the ECG generator's two packages and moves
# MAGIC `mlflow-skinny` up to 3.13.0, the version the models are logged with (`MODEL_PIP_REQUIREMENTS`
# MAGIC in `common/config.py`); 3.13.0 accepts the environment's SDK and protobuf (it requires
# MAGIC `databricks-sdk` >=0.20.0,<1 and `protobuf` >=3.12.0,<8), so pip changes no other package. Pinning
# MAGIC older SDK or protobuf versions here would downgrade core packages the Python kernel itself uses,
# MAGIC and the kernel does not come back after the restart. Express deployment does not depend on
# MAGIC this notebook: Model Serving decides it from what `02_` / `03_` recorded when they logged the models.
# MAGIC Source: the [serverless environment 6](https://docs.databricks.com/aws/en/release-notes/serverless/environment-version/six)
# MAGIC release notes and their package list (checked 2026-11-23). The `mlflow.deployments` module is
# MAGIC the same in mlflow-skinny 3.12.0 and 3.13.0, and these versions provide every API this notebook
# MAGIC calls: `mlflow.deployments` (create / update / list / predict), `MlflowClient.get_model_version_by_alias`,
# MAGIC and the SDK's `serving_endpoints.get`, `put_ai_gateway(inference_table_config=...,
# MAGIC usage_tracking_config=...)` and `patch(add_tags=[EndpointTag(...)])`.

# COMMAND ----------

# DBTITLE 1,Install the generator's deps (--no-deps keeps the environment's numpy intact)
# MAGIC %pip install --no-deps "neurokit2==0.2.12" "PyWavelets==1.4.1"

# COMMAND ----------

# DBTITLE 1,Install mlflow-skinny 3.13.0, the version the models are logged with (this environment ships 3.12.0)
# MAGIC %pip install "mlflow-skinny==3.13.0"

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Shared config (common/ is one folder up from workflow/)
import os
import re
import sys

sys.path.insert(0, os.path.abspath(".."))   # notebook working dir = workflow/

from common import config, deployment_path, scoring
from common.ecg_signal_gen import generate_ecg_batch

# COMMAND ----------

# DBTITLE 1,Parameters
dbutils.widgets.text("catalog_name", "", label="UC_catalog_name (required)")
dbutils.widgets.text("schema_name", config.SCHEMA, label="UC_schema_name")
dbutils.widgets.text("endpoint_prefix", config.ENDPOINT_PREFIX, label="endpoint_prefix")
catalog, schema = config.uc_location(dbutils.widgets.get("catalog_name"), dbutils.widgets.get("schema_name"))
arr_endpoint, afib_endpoint = config.endpoint_names(dbutils.widgets.get("endpoint_prefix"))
# Optional expiry date for the endpoints' removeAfter tag. Checked here, before any endpoint changes.
dbutils.widgets.text("remove_after", "", label="remove_after (YYYY-MM-DD, optional)")
remove_after = dbutils.widgets.get("remove_after").strip()
if remove_after and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", remove_after):
    raise ValueError(f"remove_after takes a date YYYY-MM-DD (or nothing), not '{remove_after}'")

# (model, endpoint) pairs, triage first.
SERVED = (
    (config.ARR_MODEL_NAME, arr_endpoint),
    (config.AFIB_MODEL_NAME, afib_endpoint),
)
print(f"{catalog}.{schema}: {SERVED}")

# COMMAND ----------

# DBTITLE 1,Clients
import time

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    AiGatewayInferenceTableConfig,
    AiGatewayUsageTrackingConfig,
    EndpointStateConfigUpdate,
    EndpointTag,
)
from mlflow import MlflowClient
from mlflow.deployments import get_deploy_client

mlflow.set_registry_uri("databricks-uc")
deploy_client = get_deploy_client("databricks")
w = WorkspaceClient()
mc = MlflowClient(registry_uri="databricks-uc")

# COMMAND ----------

# DBTITLE 1,Helpers — wait for an endpoint, create or update one at the champion version (only when it differs)
SETTLED_OK = EndpointStateConfigUpdate.NOT_UPDATING
SETTLED_BAD = (EndpointStateConfigUpdate.UPDATE_FAILED, EndpointStateConfigUpdate.UPDATE_CANCELED)


def config_versions(ep):
    """Config versions of an endpoint's current and pending configs."""
    return [c.config_version for c in (ep.config, ep.pending_config) if c is not None and c.config_version is not None]


def pending_messages(ep):
    """'<served entity> <deployment state>: <message>' for each served entity of the pending config."""
    entities = (ep.pending_config.served_entities or []) if ep.pending_config is not None else []
    return "; ".join(f"{e.name} {e.state.deployment.value if e.state and e.state.deployment else '?'}: "
                     f"{(e.state.deployment_state_message if e.state else None) or 'no message'}" for e in entities)


def update_settled(ep, newer_than, target, seen_in_progress):
    """After an update to ``target`` was sent (``newer_than``: the latest config version before it),
    whether the endpoint's state, no longer IN_PROGRESS, is that update's result rather than what was
    there before (a leftover UPDATE_FAILED from an earlier attempt at the same target, say): a newer
    config version shows; or the update completed (NOT_UPDATING, current config the target, nothing
    pending); or it was seen IN_PROGRESS since and its config (pending or current) is the target. The
    last two cover a config version Model Serving reuses."""
    if max(config_versions(ep), default=-1) > newer_than:
        return True
    if ep.state.config_update == SETTLED_OK and ep.pending_config is None and serves_exactly(ep.config, target):
        return True
    return seen_in_progress and any(serves_exactly(c, target) for c in (ep.config, ep.pending_config))


def events_api(name):
    """The endpoint's events API path, the one place it is written (read_events, and the hint 05_ prints)."""
    return f"/api/2.0/serving-endpoints/{name}/events"


def read_events(name):
    """The endpoint's events (deployment_path.events_of), or None when they could not be read: the call
    raised in any way (an API error, a timeout, a dropped connection) or the response is not
    {"events": [...]}; the reason is printed. The only read of the events API in this notebook: the
    update wait and the deployment-path report both use it."""
    try:
        response = w.api_client.do("GET", events_api(name))
    except Exception as error:  # noqa: BLE001 (any failure: the caller goes on without the events)
        print(f"  {name}: events not readable ({type(error).__name__}: {error})")
        return None
    events = deployment_path.events_of(response)
    if events is None:
        print(f"  {name}: events not readable (unexpected response: {str(response)[:200]})")
    return events


def wait_for_endpoint_ready(name, timeout=2400, poll=30, newer_than=None, target=None, before=None):
    """Poll until the endpoint's config update has settled (state.config_update is no longer
    IN_PROGRESS) and return it: NOT_UPDATING, or UPDATE_FAILED / UPDATE_CANCELED for the caller to
    act on. A failed update leaves its pending_config behind, so pending_config is not a sign of an
    update in progress. After an update (``newer_than`` and ``target`` set, from deploy()), only a
    state that is that update's result counts (update_settled), so an earlier failure is not taken for
    this one. While the state still looks like an earlier failure, the endpoint's events decide, when
    ``before`` holds the fingerprints of the events read just before the update was sent: a new
    "Endpoint update failed" event (beyond those counted then) is this update's failure, and raises with
    Model Serving's messages (an update that reuses the config version and fails before any poll sees
    it IN_PROGRESS looks exactly like the earlier failure otherwise). No clock is compared: the
    events' timestamps are Model Serving's. Events that cannot be read now leave the wait on the
    state alone."""
    start, seen_in_progress = time.time(), False
    while time.time() - start < timeout:
        ep = w.serving_endpoints.get(name=name)
        update = ep.state.config_update if ep.state else None
        seen_in_progress = seen_in_progress or update == EndpointStateConfigUpdate.IN_PROGRESS
        if update is not None and update != EndpointStateConfigUpdate.IN_PROGRESS and (
                newer_than is None or update_settled(ep, newer_than, target, seen_in_progress)):
            ready = ep.state.ready.value if ep.state.ready else "?"
            if update in SETTLED_BAD:
                print(f"  '{name}' last config update {update.value} (state: {ready}; pending config "
                      f"v{ep.pending_config.config_version if ep.pending_config else '?'}: {pending_messages(ep)})")
            else:
                print(f"  '{name}' settled (state: {ready})")
            return ep
        if before is not None and update in SETTLED_BAD:
            failure = deployment_path.update_failure(read_events(name), name, before)
            if failure:
                (want,) = target["served_entities"]
                raise RuntimeError(
                    f"{name}: the config update to {want['entity_name']} v{want['entity_version']} failed: "
                    f"{' '.join(failure)} The endpoint keeps serving its previous config; see its Logs > Build "
                    "logs, then run the task again.")
        print(f"  '{name}' updating... {int(time.time() - start)}s")
        time.sleep(poll)
    raise TimeoutError(f"'{name}' not ready after {timeout}s")


def served_models(endpoint):
    """UC model full names an existing endpoint serves, in its current and pending config."""
    names = set()
    for endpoint_config in (endpoint.config, endpoint.pending_config):
        if endpoint_config is None:
            continue
        names |= {e.entity_name for e in (endpoint_config.served_entities or []) if e.entity_name}
        names |= {m.model_name for m in (endpoint_config.served_models or []) if m.model_name}
    return names


def serves_exactly(endpoint_config, target):
    """True when an endpoint config (current or pending) is ``target`` (the config deploy() sends): one
    served entity with the same name, model, version, size and scale-to-zero, and one route sending it
    all traffic."""
    if endpoint_config is None:
        return False
    (want,) = target["served_entities"]
    entities = endpoint_config.served_entities or []
    if len(entities) != 1 or any(str(getattr(entities[0], key)) != str(value) for key, value in want.items()):
        return False
    routes = (endpoint_config.traffic_config.routes or []) if endpoint_config.traffic_config else []
    return len(routes) == 1 and (routes[0].served_model_name or routes[0].served_entity_name) == want["name"] \
        and routes[0].traffic_percentage == 100


def deploy(model_name, endpoint_name, existing):
    """Create the endpoint, or update it to the champion version, unless it already serves exactly
    that (config_update NOT_UPDATING, no pending config). Returns (full name, version, target config, the latest
    config version before the update; None when created, or when nothing was sent; the fingerprints
    of the endpoint's events read just before the update was sent, deployment_path.fingerprints;
    None when created, nothing was sent, or the events could not be read)."""
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
    if endpoint_name not in existing:
        print(f"creating {endpoint_name} -> {full} v{ver}")
        deploy_client.create_endpoint(name=endpoint_name, config=endpoint_config)
        return full, ver, endpoint_config, None, None
    ep = wait_for_endpoint_ready(endpoint_name)
    if ep.state.config_update == SETTLED_OK and ep.pending_config is None and serves_exactly(ep.config, endpoint_config):
        print(f"{endpoint_name} already serves {full} v{ver}; no update")
        return full, ver, endpoint_config, None, None
    print(f"updating {endpoint_name} -> {full} v{ver} (last config update: {ep.state.config_update.value})")
    # The events as they are before the update: afterwards, a failure event not among them is this
    # update's (wait_for_endpoint_ready). Unreadable: the wait judges the update by its state alone.
    before = deployment_path.fingerprints(read_events(endpoint_name))
    if before is None:
        print(f"  note: {endpoint_name}'s events could not be read before the update, so its result is judged "
              "from the endpoint's state alone (a failure that looks like an earlier one waits for the timeout)")
    deploy_client.update_endpoint(endpoint=endpoint_name, config=endpoint_config)
    return full, ver, endpoint_config, max(config_versions(ep), default=-1), before

# COMMAND ----------

# DBTITLE 1,Refuse an existing endpoint that serves another deployment's model (before changing either)
existing = {e["name"] for e in deploy_client.list_endpoints()}
for model_name, endpoint_name in SERVED:
    if endpoint_name not in existing:
        continue
    full = config.uc_name(model_name, catalog, schema)
    served = served_models(w.serving_endpoints.get(name=endpoint_name))
    if served != {full}:
        raise RuntimeError(
            f"Serving endpoint {endpoint_name} already exists and serves {sorted(served) or 'no model'}, not {full}. "
            "Endpoint names are workspace-wide and it belongs to another deployment, so it is left as it is. "
            "Set the bundle variable endpoint_prefix to a unique value and run the SETUP job again.")
    print(f"{endpoint_name}: exists and serves {full} (this deployment's model), will be updated if needed")

# COMMAND ----------

# DBTITLE 1,Issue both deploys (they build in parallel); an endpoint already at the champion is left as it is
deployed = {}
for model_name, endpoint_name in SERVED:
    deployed[endpoint_name] = deploy(model_name, endpoint_name, existing)
    print(f"{endpoint_name}: {deployed[endpoint_name][0]} v{deployed[endpoint_name][1]}")

# COMMAND ----------

# DBTITLE 1,Wait for both ready + enable AI Gateway inference tables (<model>_payload) + usage tracking
# The table prefix is the model name (config.payload_table_name), not the endpoint name: the
# dashboard reads these tables by fixed bare names, whatever endpoint_prefix is.
# put_ai_gateway replaces the endpoint's whole AI Gateway config, so each call passes every feature
# this demo uses: one left out would be switched off.
# An update (or create) that ends UPDATE_FAILED / UPDATE_CANCELED stops the task here with Model
# Serving's message; the endpoint keeps serving its previous config meanwhile.
for model_name, endpoint_name in SERVED:
    full, ver, endpoint_config, newer_than, before = deployed[endpoint_name]
    ep = wait_for_endpoint_ready(endpoint_name, timeout=3000, newer_than=newer_than, target=endpoint_config,
                                 before=before)
    if ep.state.config_update != SETTLED_OK or not serves_exactly(ep.config, endpoint_config):
        raise RuntimeError(
            f"{endpoint_name}: the config update to {full} v{ver} did not complete (config update "
            f"{ep.state.config_update.value}): {pending_messages(ep) or 'no pending config'}. The endpoint "
            "keeps serving its previous config; see its Logs > Build logs, then run the task again.")
    w.serving_endpoints.put_ai_gateway(
        name=endpoint_name,
        inference_table_config=AiGatewayInferenceTableConfig(
            enabled=True, catalog_name=catalog, schema_name=schema, table_name_prefix=model_name,
        ),
        usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
    )
    print(f"AI Gateway inference table enabled for '{endpoint_name}' -> "
          f"{config.payload_table(model_name, catalog, schema)}; usage tracking enabled "
          "(system.serving.endpoint_usage)")

# COMMAND ----------

# DBTITLE 1,Report each endpoint's deployment path (express or container image build) + tag it deployment
# Express deployment is automatic for an eligible model (the first cell of this notebook says how ours
# avoid what prevents it); this shows which path each endpoint's served model took, from its build log and the endpoint's
# events. Anything but express is a loud warning, not a failure: eligibility is Model Serving's
# decision, it can change with platform updates, and the endpoint serves either way. Reading the
# build log or the events can fail in any way (an API error, a timeout, a dropped connection, an
# unexpected response): the path is then `unknown`, since without the events a container image
# build (express that fell back) could hide behind an express build-log line.
for _, endpoint_name in SERVED:
    (entity,) = w.serving_endpoints.get(name=endpoint_name).config.served_entities   # deploy() serves one
    try:
        build_log = w.serving_endpoints.build_logs(name=endpoint_name, served_model_name=entity.name).logs
    except Exception as error:  # noqa: BLE001 (any failure: path unknown)
        build_log = None
        print(f"  {entity.name}: build log not readable ({type(error).__name__}: {error})")
    events = read_events(endpoint_name)             # None (reason printed) when not readable: path unknown
    found = deployment_path.classify(entity.name, build_log, events)
    print(found.line)
    where = f"the endpoint's Logs > Build logs and its events (databricks api get {events_api(endpoint_name)})"
    if found.path == deployment_path.STANDARD:
        print(f"WARNING: {endpoint_name} did not use express deployment (it built a container image), so its "
              f"updates take longer. Check {where}, and that 02_ / 03_ logged the model with "
              "config.MODEL_PIP_REQUIREMENTS (docs/architecture.md, Express deployment).")
    elif found.path == deployment_path.UNKNOWN:
        reason = (f"could not read {found.unread}" if found.unread
                  else "its build log and events named neither path")
        print(f"WARNING: the deployment path of {endpoint_name} could not be determined ({reason}). This "
              f"does not mean express was not used; check {where}.")
    w.serving_endpoints.patch(
        name=endpoint_name, add_tags=[EndpointTag(key=deployment_path.DEPLOYMENT_TAG, value=found.path)])
    print(f"'{endpoint_name}' tagged {deployment_path.DEPLOYMENT_TAG}={found.path}")

# COMMAND ----------

# DBTITLE 1,Tag both endpoints removeAfter=<remove_after> (only when the parameter is set)
# The bundle's presets.tags reaches only bundle resources, not these SDK-created endpoints, so the
# SETUP job passes the date here. patch(add_tags=...) sets the key, so a re-run sets the same tag
# again. Empty: no tag is added or deleted.
if remove_after:
    for _, endpoint_name in SERVED:
        w.serving_endpoints.patch(name=endpoint_name, add_tags=[EndpointTag(key="removeAfter", value=remove_after)])
        print(f"'{endpoint_name}' tagged removeAfter={remove_after}")
else:
    print("remove_after is empty: endpoint tags left as they are")

# COMMAND ----------

# DBTITLE 1,Smoke-query both endpoints with synthetic 12-lead ECGs (same payload as the app)
# One NSR + one AFib sample from the shared generator, sent ONE ECG PER REQUEST as a (1, 12, 2500)
# tensor, like the app. One ECG is about 650 KB of JSON; the AI Gateway inference table does not log a
# request larger than 1 MiB (it records MAX_REQUEST_SIZE_EXCEEDED instead; https://docs.databricks.com/aws/en/ai-gateway/inference-tables-serving-endpoints),
# so a 2-ECG request would be served but not logged.
ecg, labels, _, lead_names = generate_ecg_batch(2, frac_afib=0.5, frac_other=0.0, seed=7, n_leads=config.N_LEADS)
print("sample ecg:", ecg.shape, "labels:", labels.tolist(), "leads:", lead_names)

for _, endpoint_name in SERVED:
    for one_ecg in ecg:
        resp = deploy_client.predict(endpoint=endpoint_name, inputs={"inputs": one_ecg[None].tolist()})
        print(f"{endpoint_name} ->", resp)
