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
# MAGIC and an update that fails stops the task with Model Serving's message (one whose result looks
# MAGIC exactly like that earlier failure, or that shows no state, stops it once that has lasted 5
# MAGIC minutes, saying so and pointing at the endpoint's events; the stop comes at the next check of
# MAGIC that endpoint, which the other endpoint's AI Gateway setup can delay, and an endpoint that
# MAGIC shows the update in progress is always waited for, up to the task's timeout).
# MAGIC **One writer at a time.** The wait after each update assumes this task is the only one changing
# MAGIC these endpoints while it runs (no second SETUP run, no edit in the Serving UI). It still checks: a
# MAGIC pending config that definitely serves something else than this run sent (a served-entity name,
# MAGIC model or version that is there and differs) is another writer's, and the task stops saying
# MAGIC another update is in progress (or replaced this one), instead of reporting that update's result
# MAGIC as its own. Any field the API leaves out (in a pending config or in the current config) is
# MAGIC missing, never a value: missing data alone never stops the task as a failure, another writer's
# MAGIC update or "did not complete"; the wait counts it as unclear until a poll shows it whole. Both
# MAGIC endpoints are waited for together, under one deadline, so a failure on either stops the task as
# MAGIC soon as it shows.
# MAGIC **An existing endpoint is updated only if it already serves this deployment's model**
# MAGIC (`<catalog>.<schema>.<model>`), fail safe: its current config must be readable and serve only that
# MAGIC model, and a pending config it has must name only that model too. One that serves anything else
# MAGIC belongs to another deployment, and the task stops before changing either endpoint, asking for a
# MAGIC unique `endpoint_prefix`; one with anything unknown (no current config, a config that lists no
# MAGIC served model, a model name left out) stops it the same way, saying it could not tell which model
# MAGIC the endpoint serves. The check runs for both endpoints first, and again on the read of the
# MAGIC endpoint that its update is sent from. That narrows the window between the check and the update
# MAGIC but does not close it, so the one-writer-at-a-time assumption above still applies.
# MAGIC Inference tables are `<model>_payload` in the same schema (`bio_ecg12_arr_payload`,
# MAGIC `bio_ecg12_afib_payload`, read by the dashboard's live-serving section): AI Gateway's table
# MAGIC prefix is the model name, so the table names do not depend on the endpoint names. AI Gateway
# MAGIC logs the endpoints' requests (the whole ECG) and responses (the scores) there, with a delay: the
# MAGIC docs say that for CPU custom-model endpoints such as these the table is a view over a telemetry
# MAGIC table and "rows are typically available within seconds" (the "within 1 hour" best-effort
# MAGIC delivery is for Foundation Model API, external-model and agent endpoints), but in the tested
# MAGIC deployments rows arrived 20 to 60 minutes after a request (in the tested AWS deployment the
# MAGIC `_payload` tables were plain Delta tables, not that view; checked 2026-11-24). The docs also list gaps: a request or response over 1 MiB is
# MAGIC "logged as `null`", and for custom model endpoints "logs may not be recorded for any 4xx or 5xx
# MAGIC errors"
# MAGIC ([inference tables for serving endpoints](https://docs.databricks.com/aws/en/ai-gateway/inference-tables-serving-endpoints)). So these
# MAGIC tables hold the ECGs from logged requests; with real data, treat them as patient data. AI Gateway **usage tracking** is on too: usage records for the endpoints' requests land in
# MAGIC the system table `system.serving.endpoint_usage` (the endpoints' served models in
# MAGIC `system.serving.served_entities`);
# MAGIC only account admins can query these, unless one grants access to them.
# MAGIC
# MAGIC With a `remove_after` date (`YYYY-MM-DD`, the bundle variable of the same name), both endpoints
# MAGIC also get the tag `removeAfter` with that date, as the jobs do; empty (the default) adds no tag
# MAGIC and leaves an existing one as it is. A tag call that fails (no permission, a tag policy
# MAGIC refusing the value) prints a warning with the tag to add by hand and does not fail the task,
# MAGIC for this tag and for `deployment` below: the endpoints serve either way.
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
# MAGIC `deployment` = `express` (its environment preparation completed, or was prepared earlier and
# MAGIC reused), `standard` (a container image build finished: the endpoint serves, but updates are
# MAGIC slower) or `unknown` (the build log or the events could not be read, a build had started and
# MAGIC not finished, or they named neither path; not evidence against express). Either is a warning,
# MAGIC not a failure. In the Serving UI, the endpoint's **Logs > Build logs** (pick the served model,
# MAGIC e.g. `<endpoint>-2`) started with "Express deployment is preparing ..." or "This environment was
# MAGIC prepared ahead of time by express deployment ..." in our tests. The events (`databricks api get
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
from typing import NamedTuple

SETTLED_OK = EndpointStateConfigUpdate.NOT_UPDATING
SETTLED_BAD = (EndpointStateConfigUpdate.UPDATE_FAILED, EndpointStateConfigUpdate.UPDATE_CANCELED)


class Deployment(NamedTuple):
    """What deploy() did for one endpoint, for the wait that follows (update_verdict)."""
    full_name: str              # <catalog>.<schema>.<model>
    version: str                # the champion version
    target: dict                # the endpoint config deploy() sends (or found already served)
    before_version: object      # the current config's version before the update: CREATED (no endpoint), None (left out)
    before_pending: tuple | None = None   # pending_signature of the pending config left before the update


# Deployment.before_version of a create: there was no endpoint, so any config after it is new. Kept
# apart from None, the API leaving an existing config's version out (unknown, not "none").
CREATED = "created"


def current_version(ep):
    """The endpoint's current config version, None when it has no current config or the API left the
    version out."""
    return ep.config.config_version if ep.config is not None else None


def pending_messages(ep):
    """'<served entity> <deployment state>: <message>' for each served entity of the pending config."""
    entities = (ep.pending_config.served_entities or []) if ep.pending_config is not None else []
    return "; ".join(f"{e.name} {e.state.deployment.value if e.state and e.state.deployment else '?'}: "
                     f"{(e.state.deployment_state_message if e.state else None) or 'no message'}" for e in entities)


def entity_states(pending):
    """The deployment state of each served entity of a pending config (None for one the API gave no
    state: neither progress nor a failure); no states when it lists no served entity."""
    return [e.state.deployment.value if e.state and e.state.deployment else None for e in (pending.served_entities or [])]


def pending_signature(pending):
    """What an update sets in its pending config, for telling it from another one (new_attempt):
    config version, start_time and its served entities (IDENTITY), as the API gave them: a field it
    left out stays None (MISSING to compare), never a value. Not their deployment states or
    messages, which can still change on an earlier attempt's config. None for no pending config."""
    if pending is None:
        return None
    entities = None if pending.served_entities is None else tuple(
        tuple(getattr(e, key, None) for key in IDENTITY) for e in pending.served_entities)
    return (pending.config_version, pending.start_time, entities)


# How 05_ tells the update it sent from what was there before (update_verdict).
# Every comparison of a field the API gives goes through compare(), which has three answers: MATCH,
# DIFFERS (both values there and different) and MISSING (either left out: the SDK models nearly
# every field as optional). combine() makes one answer of several: any DIFFERS is definite, else any
# MISSING is inconclusive, else MATCH. Missing data alone therefore never makes a definite answer:
# an inconclusive one counts as unclear (below). The three comparisons:
# * new_attempt: before sending, deploy() waits for the endpoint to settle and keeps the signature of
#   the pending config left then (none, or an earlier failure's; pending_signature: config version,
#   start_time, served entities). After the send, a pending config is a new attempt when there was
#   none before, or when a value that is there in both differs (Model Serving can reuse a config
#   version, so start_time can be what differs). A value present before and left out now (or the
#   other way round) does not make a new attempt.
# * identity: whether a config's served entity is the one this update sent (name, model, version);
#   more than one served entity differs.
# * target_match: whether a config is exactly what deploy() sends (every served-entity field, one
#   route with all traffic). It alone decides "at the target", for deploy() and for the wait.
# A new attempt whose identity matches is this update's. A new attempt whose identity differs is
# another writer's (this notebook assumes it is the only writer while it runs, and checks): the task
# stops at once saying another update is in progress (OTHER). With nothing pending, a current config
# at the target (target_match MATCH) is this update's success (DONE). Otherwise it counts only when
# it is another config than before the send (config_changed: its version compared with the one
# deploy() read; any config after a create): then another update's when its identity differs
# (REPLACED), a result that is not the target when another field differs (FAILED). Anything missing
# (including a version deploy() could not read: unknown, not "no endpoint yet") is unclear.
# A failed state counts as this update's failure (FAILED) only when the pending config is this
# update's and every one of its served entities ended DEPLOYMENT_FAILED or DEPLOYMENT_ABORTED. While
# this update's served entity is being created, recovering, or ready with the traffic not switched
# yet (PROGRESS_STATES), the wait goes on, whatever the endpoint-level state says (a poll can show
# the new pending config next to the earlier UPDATE_FAILED), so a successful update is never
# reported as failed. A poll with no endpoint state, or a served entity with no state, says nothing
# either way: it counts as unclear, not as progress, except that an endpoint-level IN_PROGRESS is
# progress whatever its served entity shows.
# What is left is unclear: a state that cannot be told apart from the one before the send (the
# update not registered yet, or a failure identical to the earlier one) or one with data missing.
# The wait does not claim a failure, and does not wait out its timeout either. Once such polls have
# gone on for UNCLEAR_LIMIT_S with no progress in between it stops at its next poll of that
# endpoint, with a message saying so and pointing at the endpoint's events; the AI Gateway setup the
# wait cell does for an endpoint that is ready runs between polls, so it can delay that poll.
# Not used: the create / update call's own response (mlflow returns the endpoint as the API answers;
# its documented example has no pending config, and checking would take a real update), because it
# can only carry the same fields this GET shows, so it cannot separate the cases these cannot; and
# a baseline of the endpoint's events, which an earlier failure's late events made unreliable.
UNCLEAR_LIMIT_S = 300
# Verdicts of a poll. SETTLED: the endpoint's config update is no longer in progress (deploy()'s wait
# before it sends anything); the others judge a sent update (update_verdict).
SETTLED, DONE, FAILED, REPLACED = "settled", "done", "failed", "replaced"
WAITING, UNCLEAR, OTHER = "waiting", "unclear", "other"
# Deployment states (ServedModelStateDeployment) of this update's served entity that mean progress.
PROGRESS_STATES = ("DEPLOYMENT_CREATING", "DEPLOYMENT_RECOVERING", "DEPLOYMENT_READY")
MATCH, MISSING, DIFFERS = "match", "missing", "differs"
IDENTITY = ("name", "entity_name", "entity_version")      # what makes a served entity this update's


def compare(expected, actual):
    """One field: MISSING when either value is None (the API left it out), else MATCH or DIFFERS,
    compared as text (the API may give a version as a string and deploy() as a number)."""
    if expected is None or actual is None:
        return MISSING
    return MATCH if str(expected) == str(actual) else DIFFERS


def combine(answers):
    """One answer of several: DIFFERS if any differs, else MISSING if any is missing, else MATCH."""
    answers = list(answers)
    return DIFFERS if DIFFERS in answers else MISSING if MISSING in answers else MATCH


def compare_items(expected, actual, compare_one):
    """Two lists (served entities, routes) item by item with ``compare_one``: MISSING when either is
    left out or empty, DIFFERS when their lengths differ."""
    if not expected or not actual:
        return MISSING
    if len(expected) != len(actual):
        return DIFFERS
    return combine(compare_one(a, b) for a, b in zip(expected, actual))


def new_attempt(before, pending):
    """Whether ``pending`` (a pending config) is another attempt than the one whose signature deploy()
    kept (``before``, None when there was no pending config): DIFFERS (a new attempt: none before, or
    a value there in both differs), MATCH (the same) or MISSING (cannot tell)."""
    if before is None:
        return DIFFERS
    now = pending_signature(pending)
    return combine([compare(before[0], now[0]), compare(before[1], now[1]),
                    compare_items(before[2], now[2], lambda b, n: combine(map(compare, b, n)))])


def identity(endpoint_config, target):
    """Whether a config's served entity is the one ``target`` (the config deploy() sends) names
    (IDENTITY): MATCH, DIFFERS (a field differs, or more than one served entity) or MISSING."""
    entities = endpoint_config.served_entities if endpoint_config is not None else None
    return compare_items(target["served_entities"], entities,
                         lambda want, e: combine(compare(want[key], getattr(e, key, None)) for key in IDENTITY))


def target_match(endpoint_config, target):
    """Whether a config (current or pending) is exactly ``target``: every field deploy() sends for its
    one served entity (name, model, version, size, scale-to-zero), and one route sending that entity
    all traffic. MATCH, DIFFERS or MISSING."""
    if endpoint_config is None:
        return MISSING
    traffic = endpoint_config.traffic_config
    return combine([
        compare_items(target["served_entities"], endpoint_config.served_entities,
                      lambda want, e: combine(compare(value, getattr(e, key, None)) for key, value in want.items())),
        compare_items(target["traffic_config"]["routes"], traffic.routes if traffic is not None else None,
                      lambda want, route: combine([
                          compare(want["served_model_name"], route.served_model_name or route.served_entity_name),
                          compare(want["traffic_percentage"], route.traffic_percentage)])),
    ])


def config_changed(sent, ep):
    """Whether the endpoint's current config is another one than before the send (``sent``, its
    Deployment): after a create, DIFFERS for any config; otherwise compare() of the config versions
    before and now (MISSING when either is unknown)."""
    if sent.before_version == CREATED:
        return DIFFERS if ep.config is not None else MISSING
    return compare(sent.before_version, current_version(ep))


def describe(endpoint_config):
    """'<served entity> (<model> v<version>)' for each served entity of a config, or 'nothing'."""
    entities = (endpoint_config.served_entities or []) if endpoint_config is not None else []
    return ", ".join(f"{e.name or '?'} ({e.entity_name or '?'} v{e.entity_version or '?'})" for e in entities) or "nothing"


def update_verdict(ep, sent):
    """After deploy() sent an update or a create, or found the endpoint at the target (``sent``, its
    Deployment), what a poll says (see the comment above): DONE (at the target), FAILED (this
    update's failure, or a changed config that definitely is not the target), REPLACED (a changed
    config for something else), OTHER (another writer's update pending), WAITING (in progress, or this
    update's pending config with its served model in a PROGRESS_STATES state) or UNCLEAR (no endpoint
    state, nothing that tells this update's state apart from the one before, or data missing)."""
    pending = ep.pending_config
    attempt = new_attempt(sent.before_pending, pending) if pending is not None else None
    who = identity(pending, sent.target) if pending is not None else None
    if attempt == DIFFERS and who == DIFFERS:
        return OTHER
    update = ep.state.config_update if ep.state else None
    if update is None:
        return UNCLEAR
    if update == EndpointStateConfigUpdate.IN_PROGRESS:
        return WAITING
    ours = attempt == DIFFERS and who == MATCH
    progressing = ours and any(state in PROGRESS_STATES for state in entity_states(pending))
    if update == SETTLED_OK:
        if pending is None:
            current = target_match(ep.config, sent.target)
            if current == MATCH:
                return DONE
            if current == MISSING or config_changed(sent, ep) != DIFFERS:
                return UNCLEAR              # data missing, or the config from before (not registered yet)
            return REPLACED if identity(ep.config, sent.target) == DIFFERS else FAILED
        return WAITING if progressing else UNCLEAR
    if ours and entity_states(pending) and all(
            state in ("DEPLOYMENT_FAILED", "DEPLOYMENT_ABORTED") for state in entity_states(pending)):
        return FAILED
    return WAITING if progressing else UNCLEAR


def events_api(name):
    """The endpoint's events API path, the one place it is written (read_events, and the hint 05_ prints)."""
    return f"/api/2.0/serving-endpoints/{name}/events"


def read_events(name):
    """The endpoint's events (deployment_path.events_of), or None when they could not be read: the call
    raised in any way (an API error, a timeout, a dropped connection) or the response is not
    {"events": [...]}; the reason is printed. The only read of the events API in this notebook (the
    deployment-path report; the update wait does not need it)."""
    try:
        response = w.api_client.do("GET", events_api(name))
    except Exception as error:  # noqa: BLE001 (any failure: the caller goes on without the events)
        print(f"  {name}: events not readable ({type(error).__name__}: {error})")
        return None
    events = deployment_path.events_of(response)
    if events is None:
        print(f"  {name}: events not readable (unexpected response: {str(response)[:200]})")
    return events


def tag_endpoint(name, key, value):
    """Set one endpoint tag (patch(add_tags=...) sets the key and keeps the others). The tags 05_ sets
    are informational (the deployment path, the removeAfter date), so a call that fails in any way
    (no permission, a tag policy refusing the value) prints a warning with the tag to add by hand
    instead of failing the task: the endpoints serve either way."""
    try:
        w.serving_endpoints.patch(name=name, add_tags=[EndpointTag(key=key, value=value)])
    except Exception as error:  # noqa: BLE001 (any failure: warn, the endpoint serves either way)
        print(f"WARNING: could not tag '{name}' {key}={value} ({type(error).__name__}: {error}); add the tag in "
              "the Serving UI (the endpoint's Tags) if you need it.")
        return
    print(f"'{name}' tagged {key}={value}")


def wait_for_endpoints(sent_by_name, timeout=2400, poll=30):
    """Poll every endpoint of ``sent_by_name`` ({endpoint name: its Deployment, or None}) in each
    round, under one deadline, and yield (name, endpoint, verdict) for each as soon as it has a
    result, so the caller acts on a failure of any of them without waiting for the others. With
    None (deploy(), before it sends anything) the result is SETTLED: state.config_update is no longer
    IN_PROGRESS. A failed update leaves its pending_config behind, so pending_config is not a sign of
    an update in progress. With a Deployment only that update's result counts (update_verdict: DONE,
    FAILED or REPLACED, for the caller to act on); another writer's update stops the task at once, and an unclear state (one that cannot be told
    apart from the one before the send, or an incomplete one) stops it at the first poll of that
    endpoint after UNCLEAR_LIMIT_S of unclear polls; the caller's work between polls (AI Gateway
    setup) can delay that poll."""
    start, waiting, unclear_since = time.time(), dict(sent_by_name), {}
    while waiting:
        if time.time() - start >= timeout:
            raise TimeoutError(f"{', '.join(repr(name) for name in waiting)} not ready after {timeout}s")
        for name, sent in list(waiting.items()):
            ep = w.serving_endpoints.get(name=name)
            update = ep.state.config_update if ep.state else None
            if sent is not None:
                verdict = update_verdict(ep, sent)
            else:
                verdict = WAITING if update in (None, EndpointStateConfigUpdate.IN_PROGRESS) else SETTLED
            if verdict == OTHER:
                raise RuntimeError(
                    f"{name}: another update is in progress on this endpoint: its pending config serves "
                    f"{describe(ep.pending_config)} (config update {update.value if update else 'not reported'}), "
                    f"not this run's update to {sent.full_name} v{sent.version}. Another SETUP run or a change in "
                    "the Serving UI may be updating it; wait until it is done, then run the task again.")
            if verdict in (SETTLED, DONE, FAILED, REPLACED):
                ready = ep.state.ready.value if ep.state.ready else "?"
                if update in SETTLED_BAD:
                    print(f"  '{name}' last config update {update.value} (state: {ready}; pending config "
                          f"v{ep.pending_config.config_version if ep.pending_config else '?'}: {pending_messages(ep)})")
                else:
                    print(f"  '{name}' settled (state: {ready})")
                del waiting[name]
                yield name, ep, verdict
                continue
            if verdict == WAITING:
                unclear_since.pop(name, None)
            elif time.time() - unclear_since.setdefault(name, time.time()) >= UNCLEAR_LIMIT_S:
                raise RuntimeError(
                    f"{name}: could not tell whether the config update to {sent.full_name} v{sent.version} has "
                    f"started or how it ended: for {UNCLEAR_LIMIT_S}s the endpoint has shown neither that update "
                    "in progress nor a result that is clearly its own (last poll: config update "
                    f"{update.value if update else 'not reported'}; pending config: {pending_messages(ep) or 'none'}). "
                    f"The update may still be running. Check the endpoint's events (databricks api get "
                    f"{events_api(name)}) and its Logs > Build logs, then run the task again.")
            print(f"  '{name}' updating... {int(time.time() - start)}s")
        if waiting:
            time.sleep(poll)


def wait_for_endpoint_ready(name, timeout=2400, poll=30):
    """wait_for_endpoints for one endpoint with nothing sent yet (deploy(), before its update):
    returns the endpoint once its config update has settled."""
    for _, ep, _ in wait_for_endpoints({name: None}, timeout, poll):
        return ep


# Ownership of an existing endpoint (endpoint names are workspace-wide), fail safe: it is this
# deployment's only when its current config can be read and serves only this deployment's model,
# and a pending config it has names only that model too. Checked in the ownership cell (both
# endpoints, before either changes) and again in deploy() on the snapshot it sends the update from.
# That narrows the window between the check (a GET) and the update but does not remove it: another
# writer could still change the endpoint in between, so the single-writer assumption (one SETUP run
# or one person updating these endpoints at a time) still applies.
OURS, THEIRS, UNKNOWN_OWNER = "ours", "theirs", "unknown"


def ownership(endpoint, full):
    """(verdict, model names found) for an existing endpoint and this deployment's model ``full``
    (<catalog>.<schema>.<model>): THEIRS when a model name that is there is anything else; else
    UNKNOWN_OWNER when the current config is missing, or the current or a pending config lists no
    served model (served_entities, or the legacy served_models) or leaves a model name out; else
    OURS."""
    names, unknown = set(), endpoint.config is None
    for endpoint_config in (endpoint.config, endpoint.pending_config):
        if endpoint_config is None:
            continue
        listed = [e.entity_name for e in endpoint_config.served_entities or []] + [
            m.model_name for m in endpoint_config.served_models or []]
        unknown = unknown or not listed or None in listed
        names |= {name for name in listed if name is not None}
    if names - {full}:
        return THEIRS, names
    return (UNKNOWN_OWNER if unknown else OURS), names


def check_ownership(endpoint_name, endpoint, full):
    """Stop the task unless the existing endpoint is this deployment's (ownership OURS)."""
    verdict, names = ownership(endpoint, full)
    if verdict == THEIRS:
        raise RuntimeError(
            f"Serving endpoint {endpoint_name} already exists and serves {sorted(names)}, not only {full}. "
            "Endpoint names are workspace-wide and it belongs to another deployment, so it is left as it is. "
            "Set the bundle variable endpoint_prefix to a unique value and run the SETUP job again.")
    if verdict == UNKNOWN_OWNER:
        # Missing data is not evidence of another deployment, nor of this one: stop before changing it.
        raise RuntimeError(
            f"Serving endpoint {endpoint_name} already exists, and 05_ could not tell which model it serves (its "
            "current config, or a pending config, is missing, lists no served model or gives no model name), so "
            "it is left as it is. Check the endpoint in the Serving UI, then run the SETUP job again.")


def at_target(ep, target):
    """True when the endpoint already serves exactly ``target`` and nothing else is going on: its
    last config update completed (NOT_UPDATING), no pending config is left, and its current config is
    ``target`` (target_match MATCH: a field left out is not a match, so deploy() sends the update)."""
    return (ep.state is not None and ep.state.config_update == SETTLED_OK and ep.pending_config is None
            and target_match(ep.config, target) == MATCH)


def deploy(model_name, endpoint_name, existing):
    """Create the endpoint, or update it to the champion version, unless it already serves exactly
    that (at_target). Returns a Deployment."""
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
        return Deployment(full, ver, endpoint_config, CREATED)          # no config before: any is new
    ep = wait_for_endpoint_ready(endpoint_name)
    check_ownership(endpoint_name, ep, full)        # on the snapshot the update is sent from
    if at_target(ep, endpoint_config):
        print(f"{endpoint_name} already serves {full} v{ver}; no update")
        return Deployment(full, ver, endpoint_config, current_version(ep))
    print(f"updating {endpoint_name} -> {full} v{ver} (last config update: {ep.state.config_update.value})")
    deploy_client.update_endpoint(endpoint=endpoint_name, config=endpoint_config)
    return Deployment(full, ver, endpoint_config, current_version(ep), pending_signature(ep.pending_config))

# COMMAND ----------

# DBTITLE 1,Refuse an existing endpoint that serves another deployment's model (before changing either)
existing = {e["name"] for e in deploy_client.list_endpoints()}
for model_name, endpoint_name in SERVED:
    if endpoint_name not in existing:
        continue
    full = config.uc_name(model_name, catalog, schema)
    check_ownership(endpoint_name, w.serving_endpoints.get(name=endpoint_name), full)
    print(f"{endpoint_name}: exists and serves {full} (this deployment's model), will be updated if needed")

# COMMAND ----------

# DBTITLE 1,Issue both deploys (they build in parallel); an endpoint already at the champion is left as it is
deployed = {}
for model_name, endpoint_name in SERVED:
    deployed[endpoint_name] = deploy(model_name, endpoint_name, existing)
    print(f"{endpoint_name}: {deployed[endpoint_name].full_name} v{deployed[endpoint_name].version}")

# COMMAND ----------

# DBTITLE 1,Wait for both ready + enable AI Gateway inference tables (<model>_payload) + usage tracking
# The table prefix is the model name (config.payload_table_name), not the endpoint name: the
# dashboard reads these tables by fixed bare names, whatever endpoint_prefix is.
# put_ai_gateway replaces the endpoint's whole AI Gateway config, so each call passes every feature
# this demo uses: one left out would be switched off.
# An update (or create) that ends UPDATE_FAILED / UPDATE_CANCELED stops the task here with Model
# Serving's message; the endpoint keeps serving its previous config meanwhile. Both endpoints are
# polled together (wait_for_endpoints, one deadline), so either one's failure stops the task as soon
# as it shows, and each gets its AI Gateway config as soon as it is ready. That setup runs between
# polls, so it can delay the other endpoint's next poll (and so a stop at UNCLEAR_LIMIT_S).
model_of = {endpoint_name: model_name for model_name, endpoint_name in SERVED}
for endpoint_name, ep, verdict in wait_for_endpoints({name: deployed[name] for name in model_of}, timeout=3000):
    sent = deployed[endpoint_name]
    if verdict == REPLACED:
        raise RuntimeError(
            f"{endpoint_name}: another update replaced this run's update to {sent.full_name} v{sent.version}: the "
            f"endpoint now serves {describe(ep.config)}. Another SETUP run or a change in the Serving UI may have "
            "updated it; run the task again once it is done.")
    if verdict != DONE:
        raise RuntimeError(
            f"{endpoint_name}: the config update to {sent.full_name} v{sent.version} did not complete (config update "
            f"{ep.state.config_update.value}): {pending_messages(ep) or 'no pending config'}. The endpoint "
            "keeps serving its previous config; see its Logs > Build logs, then run the task again.")
    model_name = model_of[endpoint_name]
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
# unexpected response): the path is then `unknown`. Without the events a container image build
# could hide behind an express build-log line, on the assumption (not documented) that express which
# falls back may still have written one.
for _, endpoint_name in SERVED:
    current = w.serving_endpoints.get(name=endpoint_name).config
    entities = (current.served_entities if current is not None else None) or []
    if len(entities) != 1:                         # deploy() serves one; the API left it out (or more)
        reason = "the endpoint's config names no single served entity"
        print(f"WARNING: the deployment path of {endpoint_name} could not be determined ({reason}).")
        tag_endpoint(endpoint_name, deployment_path.DEPLOYMENT_TAG, deployment_path.UNKNOWN)
        continue
    (entity,) = entities
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
        print(f"WARNING: the deployment path of {endpoint_name} could not be determined ({found.reason}). This "
              f"does not mean express was not used; check {where}.")
    tag_endpoint(endpoint_name, deployment_path.DEPLOYMENT_TAG, found.path)

# COMMAND ----------

# DBTITLE 1,Tag both endpoints removeAfter=<remove_after> (only when the parameter is set)
# The bundle's presets.tags reaches only bundle resources, not these SDK-created endpoints, so the
# SETUP job passes the date here. patch(add_tags=...) sets the key, so a re-run sets the same tag
# again. Empty: no tag is added or deleted. A tag call that fails only warns (tag_endpoint).
if remove_after:
    for _, endpoint_name in SERVED:
        tag_endpoint(endpoint_name, "removeAfter", remove_after)
else:
    print("remove_after is empty: endpoint tags left as they are")

# COMMAND ----------

# DBTITLE 1,Smoke-query both endpoints with synthetic 12-lead ECGs (same payload as the app)
# One NSR + one AFib sample from the shared generator, sent ONE ECG PER REQUEST as a (1, 12, 2500)
# tensor, like the app. One ECG is about 650 KB of JSON; the AI Gateway inference table logs a request
# larger than 1 MiB with a null payload and MAX_REQUEST_SIZE_EXCEEDED
# (https://docs.databricks.com/aws/en/ai-gateway/inference-tables-serving-endpoints), so a 2-ECG
# request would be served but its ECGs not logged.
ecg, labels, _, lead_names = generate_ecg_batch(2, frac_afib=0.5, frac_other=0.0, seed=7, n_leads=config.N_LEADS)
print("sample ecg:", ecg.shape, "labels:", labels.tolist(), "leads:", lead_names)

for _, endpoint_name in SERVED:
    for one_ecg in ecg:
        resp = deploy_client.predict(endpoint=endpoint_name, inputs={"inputs": one_ecg[None].tolist()})
        print(f"{endpoint_name} ->", resp)
