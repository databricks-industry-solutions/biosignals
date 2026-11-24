"""Static checks of the workflow notebooks (they are NOT executed on Databricks here; the SETUP and
DAILY flows run locally with stand-ins in test_setup_flow_local.py / test_daily_flow_local.py),
plus a serving check of the training notebooks' inline model code: it is pickled the way MLflow
pickles it and loaded in a fresh process where the `common` package does not exist (as on a
serving endpoint)."""
import ast
import contextlib
import enum
import io
import re
import subprocess
import sys
import textwrap
import tokenize
import types
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import DEMO_ROOT, GOVERNED_TAG_VALUES, PIP_LINES, pip_lines

from common import config

WORKFLOW = DEMO_ROOT / "workflow"
NOTEBOOKS = sorted(WORKFLOW.glob("*.py"))
# Run order: SETUP 01-05 (data -> train x2 -> validation gate -> serve), DAILY 06-08.
EXPECTED = ["00_overview.py", "01_generate_training_data.py", "02_train_arr_model.py", "03_train_afib_model.py",
            "04_validate_and_promote.py", "05_serve_endpoints.py", "06_generate_batch_data.py",
            "07_arr_triage.py", "08_afib_classify.py"]
TRAINING = {"02_train_arr_model.py": ("ArrCNN", "ArrModelPyfunc"),
            "03_train_afib_model.py": ("AfibCNN", "AfibModelPyfunc")}
GATE = "04_validate_and_promote.py"
# Names that would make MLflow pickle a reference to the `common` package (absent when serving).
SHARED_NAMES = {"common", "config", "scoring", "uc_io", "patient_info", "training", "training_data",
                "evaluation", "ecg_plots", "eval_plots", "generate_ecg_batch", "derive_arr_flag",
                "derive_afib_truth", "LABEL_NAMES", "describe_lead"}
BATCH = ("07_arr_triage.py", "08_afib_classify.py")
# The notebooks that log, load or serve the models: each runs the models' MLflow version, from its
# own %pip line or from its environment (test_repo_consistency.py checks which).
MLFLOW_NOTEBOOKS = (*TRAINING, GATE, "05_serve_endpoints.py", *BATCH)


def _cells(path):
    return path.read_text().split("# COMMAND ----------")


def _python_source(path):
    """Notebook source with magic (# MAGIC) lines blanked, so it parses as plain Python."""
    lines = path.read_text().splitlines()
    return "\n".join("" if line.startswith("# MAGIC") else line for line in lines)


def _code_tokens(path, names=True):
    """STRING (and, by default, NAME) tokens of the notebook's code; markdown and comments excluded."""
    tokens = tokenize.generate_tokens(io.StringIO(_python_source(path)).readline)
    kinds = {tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", tokenize.STRING)} | ({tokenize.NAME} if names else set())
    return [t.string for t in tokens if t.type in kinds]


def test_run_order_names():
    assert [p.name for p in NOTEBOOKS] == EXPECTED


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_databricks_source(path):
    assert path.read_text().startswith("# Databricks notebook source\n")
    compile(_python_source(path), str(path), "exec")


@pytest.mark.parametrize("path", [p for p in NOTEBOOKS if p.name != "00_overview.py"], ids=lambda p: p.name)
def test_catalog_is_a_required_parameter(path):
    """No catalog default anywhere: the catalog_name widget starts empty and goes through
    config.uc_location, which fails clearly when the job did not pass it."""
    code = _python_source(path)
    assert 'dbutils.widgets.text("catalog_name", "", label="UC_catalog_name (required)")' in code
    assert 'dbutils.widgets.text("schema_name", config.SCHEMA, label="UC_schema_name")' in code
    assert 'config.uc_location(dbutils.widgets.get("catalog_name")' in code
    assert code.count('dbutils.widgets.get("catalog_name")') == 1


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_aliases_schema_and_tables_come_from_config(path):
    """No alias, schema or training-table name typed into code: they come from common/config.py."""
    for token in _code_tokens(path, names=False):
        literal = token.strip("'\"")
        assert literal not in {"champion", "Champion", "challenger", config.SCHEMA,
                               config.TRAINING_TABLE}, f"{path.name}: {token}"
        assert "@champion" not in token and "@challenger" not in token, f"{path.name}: {token}"


def test_only_the_gate_promotes_and_only_it_reads_the_holdout():
    """Training registers challengers; only 04_ sets champion, and only 04_ reads the hold-out split."""
    for path in NOTEBOOKS:
        code = _python_source(path)
        sets_champion = "set_registered_model_alias(resolved.full_name, config.CHAMPION_ALIAS" in code
        assert sets_champion == (path.name == GATE), path.name
        assert ("config.HOLDOUT_SPLIT" in code) == (path.name == GATE), path.name
        if path.name in TRAINING:
            assert "set_registered_model_alias(full_name, config.CHALLENGER_ALIAS, version)" in code
            assert "CHAMPION_ALIAS" not in code
    gate = _python_source(WORKFLOW / GATE)
    # Promotion happens after the failure check, and the failure check raises.
    assert gate.index("if not passed:") < gate.index("set_registered_model_alias(")
    assert "raise RuntimeError" in gate[gate.index("if not passed:"):gate.index("set_registered_model_alias(")]
    assert "uc_io.read_table_version(spark, table, version)" in gate           # the challenger's data version


def _governed_tag_writes(tree):
    """(key, value source, possible values) for every value a governed tag key (GOVERNED_TAG_VALUES)
    gets in a notebook: dict entries and set_tag / set_model_version_tag arguments. The possible
    values follow string literals, ``a if c else b``, variables and the notebook's own functions
    (their return values); they are None when the value is not known statically."""
    functions = {f.name: f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)}
    assigned = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned.setdefault(target.id, []).append(node.value)

    def union(options):
        return None if not options or None in options else frozenset().union(*options)

    def values(node, seen):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return frozenset({node.value})
        if isinstance(node, ast.IfExp):
            return union([values(node.body, seen), values(node.orelse, seen)])
        if isinstance(node, ast.Name) and node.id in assigned and node.id not in seen:
            return union([values(v, seen | {node.id}) for v in assigned[node.id]])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions \
                and node.func.id not in seen:
            returns = [r.value for r in ast.walk(functions[node.func.id]) if isinstance(r, ast.Return) and r.value]
            return union([values(r, seen | {node.func.id}) for r in returns])
        return None

    for node in ast.walk(tree):
        pairs = []
        if isinstance(node, ast.Dict):
            pairs = [(k, v) for k, v in zip(node.keys, node.values) if k is not None]
        elif isinstance(node, ast.Call) and getattr(node.func, "attr", None) in {"set_tag", "set_model_version_tag"} \
                and len(node.args) >= 2:
            pairs = [tuple(node.args[-2:])]                  # (key, value) are the last two arguments
        for key, value in pairs:
            if isinstance(key, ast.Constant) and key.value in GOVERNED_TAG_VALUES:
                yield key.value, ast.unparse(value), values(value, frozenset())


def test_tag_writes_follow_the_workspace_tag_policies():
    """No notebook writes a governed tag (conftest.GOVERNED_TAG_VALUES) with a value its tag policy does
    not allow. Statically: every value a governed key gets is known and allowed. Only the SETUP
    notebooks 02_-04_ write tags at all, and test_setup_flow_local.py runs them against a registry
    that refuses a disallowed value as Unity Catalog does, in both gate outcomes."""
    writers, found = set(), {}
    for path in NOTEBOOKS:
        tokens = _code_tokens(path)
        if {"set_tag", "set_tags", "set_model_version_tag", "set_registered_model_tag"} & set(tokens) \
                or any("SET TAGS" in t.upper() for t in tokens):
            writers.add(path.name)
        found[path.name] = set()
        for key, source, possible in _governed_tag_writes(ast.parse(_python_source(path))):
            assert possible is not None, f"{path.name}: {key}={source} is not a statically known value"
            assert possible <= GOVERNED_TAG_VALUES[key], \
                f"{path.name}: {key}={source} can be {sorted(possible)}, allowed {sorted(GOVERNED_TAG_VALUES[key])}"
            found[path.name] |= {(key, value) for value in possible}
    assert writers == {*TRAINING, GATE}
    # Only the gate sets validation_status (both values); the training notebooks leave it unset.
    assert found[GATE] == {("validation_status", "valid"), ("validation_status", "invalid")}
    assert not any(found[name] for name in TRAINING)


@pytest.mark.parametrize("name", sorted(TRAINING))
def test_training_reads_the_persisted_dataset_and_logs_thoroughly(name):
    """Data-first: train/val come from training_data_12l at one Delta version (no in-memory draw);
    MLflow gets params, per-epoch + final metrics, tags, and the model with signature + example."""
    code = _python_source(WORKFLOW / name)
    assert "generate_ecg_batch" not in code and pip_lines(WORKFLOW / name) == PIP_LINES[name]
    assert "data_version = uc_io.table_version(spark, train_table)" in code
    for split in ("config.TRAIN_SPLIT", "config.VAL_SPLIT"):
        assert f"training_data.load_split(spark, train_table, data_version, {split}, N_LEADS)" in code
    for logged in ("mlflow.log_params(", "**P,", '"training_table_version"', "step=m.epoch",
                   'mlflow.log_metrics({f"final_val_{k}": v', "mlflow.set_tags(", 'mlflow.set_tag("challenger_version"',
                   "signature=signature", "input_example=ex_in", "registered_model_name=full_name",
                   "client.set_model_version_tag(", '"n_parameters": n_params',
                   'mlflow.log_text(arch_md, "model_architecture.md")'):
        assert logged in code, f"{name}: {logged}"
    assert "artifact_path=" not in code                       # deprecated in MLflow 3; name= is used


def test_every_mlflow_run_logs_to_the_project_experiment():
    """The notebooks that start MLflow runs (02_, 03_, 04_) read the mlflow_experiment parameter and
    make it the active experiment with tracking.use_experiment (which creates its folder, for an
    identity without a home folder, see test_tracking.py) before their first run, so no run lands in
    the notebook's own experiment (deleted with the bundle). No notebook calls MLflow's experiment
    functions itself, and none autologs, which would start runs of its own."""
    sources = {path.name: _python_source(path) for path in NOTEBOOKS}
    assert {name for name, code in sources.items() if "mlflow.start_run(" in code} == {*TRAINING, GATE}
    for name in (*sorted(TRAINING), GATE):
        code = sources[name]
        assert ('dbutils.widgets.text(config.MLFLOW_EXPERIMENT_PARAM, "", '
                'label="mlflow_experiment (required, set by the job)")') in code, name
        resolved = code.index("mlflow_experiment = config.resolve_mlflow_experiment("
                              "dbutils.widgets.get(config.MLFLOW_EXPERIMENT_PARAM))")
        use = "tracking.use_experiment(WorkspaceClient(), mlflow_experiment)"
        assert code.count(use) == 1 and "from databricks.sdk import WorkspaceClient" in code, name
        assert resolved < code.index(use) < code.index("mlflow.start_run("), name
    for name, code in sources.items():
        for call in ("autolog(", "set_experiment(", "create_experiment("):
            assert call not in code, f"{name}: {call}"


@pytest.mark.parametrize("name", BATCH)
def test_batch_scoring_is_driver_side_at_the_pinned_champion(name):
    """07_ / 08_ resolve champion to a version, load exactly that version and score on the driver
    with score_driver_side, as the gate does."""
    code = _python_source(WORKFLOW / name)
    kind = "arr" if name.startswith("07") else "afib"
    assert f"{kind} = scoring.resolve_alias(MlflowClient(registry_uri=\"databricks-uc\"), {kind}_model_full)" in code
    assert f"{kind}_model = mlflow.pyfunc.load_model({kind}.uri)" in code
    assert code.count("scoring.score_driver_side(") == 1 and f"model_info={kind}.info" in code
    assert pip_lines(WORKFLOW / name) == PIP_LINES[name]        # mlflow-skinny + torch: environment 6 has no torch


def test_triage_writes_the_history_slice_the_dashboard_reads():
    """07_ writes every day's triage rows, without the ECG arrays, as one date slice of the fixed
    table the dashboard reads (a per-date table cannot be a dashboard dataset)."""
    code = _python_source(WORKFLOW / "07_arr_triage.py")
    assert "history_table = config.uc_name(config.TRIAGE_HISTORY_TABLE, catalog_name, schema_name)" in code
    assert ("uc_io.write_date_slice(spark, spark.table(write_table).drop(*ecg_cols), history_table, run_date)"
            in code)


def test_serving_never_takes_over_another_deployments_endpoint():
    """05_ names the endpoints from the endpoint_prefix parameter, refuses (before changing either)
    an existing endpoint that serves anything but this deployment's <catalog>.<schema>.<model>, and
    names the inference tables by model, not by endpoint."""
    code = _python_source(WORKFLOW / "05_serve_endpoints.py")
    assert 'dbutils.widgets.text("endpoint_prefix", config.ENDPOINT_PREFIX, label="endpoint_prefix")' in code
    assert 'arr_endpoint, afib_endpoint = config.endpoint_names(dbutils.widgets.get("endpoint_prefix"))' in code
    assert "served = served_models(w.serving_endpoints.get(name=endpoint_name))" in code
    assert "if served != {full}:" in code and "raise RuntimeError(" in code and "endpoint_prefix" in code
    # The check runs for both endpoints (its own cell) before the first create / update call.
    assert code.index("if served != {full}:") < code.index("deployed[endpoint_name] = deploy(model_name, endpoint_name, existing)")
    assert "table_name_prefix=model_name" in code and "table_name_prefix=endpoint_name" not in code
    assert "config.payload_table(model_name, catalog, schema)" in code


def test_serving_enables_usage_tracking_in_the_one_ai_gateway_call():
    """put_ai_gateway replaces the endpoint's whole AI Gateway config, so 05_'s single call per
    endpoint passes both the inference table and usage tracking (enabled); a second call with only
    one of them would switch the other off."""
    code = _python_source(WORKFLOW / "05_serve_endpoints.py")
    assert code.count("w.serving_endpoints.put_ai_gateway(") == 1
    call = code[code.index("w.serving_endpoints.put_ai_gateway("):]
    call = call[:call.index("\n    )\n")]
    assert "inference_table_config=AiGatewayInferenceTableConfig(" in call
    assert "usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True)" in call
    assert "rate_limits=" not in call and "guardrails=" not in call


# databricks-sdk 0.122.0 (databricks/sdk/service/serving.py): the endpoint and served-entity states.
_ConfigUpdate = enum.Enum("EndpointStateConfigUpdate", {v: v for v in (
    "IN_PROGRESS", "NOT_UPDATING", "UPDATE_CANCELED", "UPDATE_FAILED")})
_Ready = enum.Enum("EndpointStateReady", {v: v for v in ("NOT_READY", "READY")})
_Deployment = enum.Enum("ServedModelStateDeployment", {v: v for v in (
    "DEPLOYMENT_ABORTED", "DEPLOYMENT_CREATING", "DEPLOYMENT_FAILED", "DEPLOYMENT_READY", "DEPLOYMENT_RECOVERING")})
ABORTED = ("Served entity creation aborted for served entity `{name}`, config version {version}, "
           "since the update timed out")


class _Clock:
    """``time`` for 05_'s wait loop: sleep() moves the clock instead of waiting."""

    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class _ServingEndpoints:
    """Model Serving for 05_'s deploy cells. Each endpoint has its current config (version, served
    model version), its config_update state and a leftover pending config. An update or create
    makes a new pending config, IN_PROGRESS for ``polls`` gets, then ends as ``outcome`` ("succeed":
    it becomes the current config; "fail": UPDATE_FAILED with the entity aborted; a (deployment state,
    message) pair: UPDATE_FAILED with that). ``lag``: gets that still return the endpoint as it was
    before the update (the update not registered yet). ``reuse_version``: the new pending config keeps
    the latest version number instead of the next one. ``reject``: update_endpoint raises it.
    ``events``: the endpoints' events API (``api_client.do``) answers with these, plus, when an update
    fails, the two events Model Serving writes then (seen live), stamped with the clock's time in
    milliseconds plus ``event_offset_ms`` (Model Serving's clock need not match the notebook's);
    ``events_error``: the events call raises it, on every call or (``events_error_calls``) on the
    first n; ``late_events``: added to the events from the second read on (delivered late). The clock
    is _run_deploy_cells'."""

    def __init__(self, endpoints, outcome="succeed", polls=2, lag=0, reuse_version=False, reject=None,
                 events=(), events_error=None, events_error_calls=None, event_offset_ms=0, late_events=()):
        self.endpoints, self.outcome, self.polls, self.lag = endpoints, outcome, polls, lag
        self.reuse_version, self.reject = reuse_version, reject
        self.events, self.events_error, self.clock = list(events), events_error, None
        self.events_error_calls, self.event_offset_ms, self.event_reads = events_error_calls, event_offset_ms, 0
        self.late_events = list(late_events)
        self.updates, self.creates, self.gateways, self.gets = [], [], [], 0
        self.serving_endpoints = SimpleNamespace(get=self.get, put_ai_gateway=self.put_ai_gateway)
        self.api_client = SimpleNamespace(do=self.do)
        self.deploy_client = SimpleNamespace(update_endpoint=self.update_endpoint, create_endpoint=self.create_endpoint,
                                             list_endpoints=lambda: [{"name": n} for n in self.endpoints])

    @staticmethod
    def _config(version, entity, deployment="DEPLOYMENT_READY", message="Scaled to zero"):
        name, full, ver = entity
        return SimpleNamespace(
            config_version=version, served_models=None,
            served_entities=[SimpleNamespace(name=name, entity_name=full, entity_version=ver, workload_size="Small",
                                             scale_to_zero_enabled=True, state=SimpleNamespace(
                                                 deployment=_Deployment[deployment], deployment_state_message=message))],
            traffic_config=SimpleNamespace(routes=[SimpleNamespace(served_model_name=name, served_entity_name=None,
                                                                   traffic_percentage=100)]))

    def get(self, name):
        self.gets += 1
        e = self.endpoints[name]
        if e.get("lag"):
            e["lag"] -= 1
            snapshot = e["before"]
        else:
            if e.get("remaining") is not None:
                if e["remaining"] > 0:
                    e["remaining"] -= 1
                else:
                    e["remaining"] = None
                    if self.outcome == "succeed":
                        e.update(version=e["pending"][0], entity=e["pending"][1], pending=None, update="NOT_UPDATING")
                    else:
                        deployment, message = ("DEPLOYMENT_ABORTED", None) if self.outcome == "fail" else self.outcome
                        e.update(update="UPDATE_FAILED", pending=(*e["pending"][:2], deployment, message))
                        version, (entity_name, *_) = e["pending"][:2]
                        now_ms = self.clock.now * 1000 + self.event_offset_ms
                        self.events += [
                            {"event_type": "SERVED_ENTITY_SERVICE_EVENT", "timestamp": now_ms, "served_entity_name": entity_name,
                             "message": ABORTED.format(name=entity_name, version=version) + "."},
                            {"event_type": "ENDPOINT_UPDATE_EVENT", "timestamp": now_ms + 50, "served_entity_name": None,
                             "message": f"Endpoint update failed for endpoint `{name}`, config version {version}."}]
            snapshot = e
        current = self._config(snapshot["version"], snapshot["entity"]) if snapshot.get("entity") else None
        pending = None
        if snapshot.get("pending"):
            version, entity, deployment, *custom = snapshot["pending"]
            message = (custom[0] if custom and custom[0] else ABORTED.format(name=entity[0], version=version)
                       if deployment == "DEPLOYMENT_ABORTED" else "Deploying")
            pending = self._config(version, entity, deployment, message)
        return SimpleNamespace(name=name, config=current, pending_config=pending, state=SimpleNamespace(
            config_update=_ConfigUpdate[snapshot["update"]], ready=_Ready["READY" if current else "NOT_READY"]))

    def _start(self, name, endpoint_config):
        e = self.endpoints.setdefault(name, {"version": 0, "entity": None, "update": "NOT_UPDATING", "pending": None})
        (entity,) = endpoint_config["served_entities"]
        e["before"] = {k: e[k] for k in ("version", "entity", "update", "pending")}
        latest = max([e["version"]] + ([e["pending"][0]] if e["pending"] else []))
        e.update(pending=(latest + (0 if self.reuse_version else 1),
                          (entity["name"], entity["entity_name"], entity["entity_version"]),
                          "DEPLOYMENT_CREATING"), update="IN_PROGRESS", remaining=self.polls, lag=self.lag)

    def update_endpoint(self, endpoint, config):
        self.updates.append((endpoint, config))
        if self.reject is not None:
            raise self.reject
        self._start(endpoint, config)

    def create_endpoint(self, name, config):
        self.creates.append((name, config))
        self._start(name, config)

    def put_ai_gateway(self, name, **kwargs):
        self.gateways.append(name)

    def do(self, method, path):
        assert method == "GET" and path.endswith("/events")
        self.event_reads += 1
        if self.event_reads > 1:
            self.events, self.late_events = self.events + self.late_events, []
        if self.events_error is not None and (self.events_error_calls is None or self.event_reads <= self.events_error_calls):
            raise self.events_error
        return {"events": [e for e in self.events if path == f"/api/2.0/serving-endpoints/{e.get('endpoint', 'ep')}/events"]}


def _endpoint(version, served_version, update="NOT_UPDATING", pending=None, pending_served=None):
    """An existing endpoint ``ep`` at config ``version`` serving model version ``served_version``, with
    an aborted pending config ``pending`` (its version) for model version ``pending_served`` (default:
    the served one)."""
    full = config.uc_name(config.ARR_MODEL_NAME, "cat", "sch")
    left = pending_served or served_version
    return {"version": version, "entity": (f"ep-{served_version}", full, served_version), "update": update,
            "pending": None if pending is None else (pending, (f"ep-{left}", full, left), "DEPLOYMENT_ABORTED")}


def _run_deploy_cells(serving, champion="2", clock=None):
    """05_'s helper, issue and wait cells on the stand-in, for one endpoint ``ep`` serving the ARR
    model; returns (output, clock)."""
    from common import deployment_path

    clock = clock or _Clock()
    serving.clock = clock
    cells = _cells(WORKFLOW / "05_serve_endpoints.py")
    (helpers,) = [c for c in cells if "def wait_for_endpoint_ready(" in c]
    (issue,) = [c for c in cells if "# DBTITLE 1,Issue both deploys" in c]
    (wait,) = [c for c in cells if "# DBTITLE 1,Wait for both ready" in c]
    namespace = {
        "w": SimpleNamespace(serving_endpoints=serving.serving_endpoints, api_client=serving.api_client),
        "deploy_client": serving.deploy_client, "deployment_path": deployment_path,
        "time": clock, "config": config, "catalog": "cat", "schema": "sch", "mc": None,
        "scoring": SimpleNamespace(resolve_alias=lambda client, full, alias: SimpleNamespace(version=champion)),
        "EndpointStateConfigUpdate": _ConfigUpdate, "SERVED": ((config.ARR_MODEL_NAME, "ep"),),
        "existing": set(serving.endpoints), "AiGatewayInferenceTableConfig": lambda **kw: kw,
        "AiGatewayUsageTrackingConfig": lambda **kw: kw}
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        for cell in (helpers, issue, wait):
            exec(compile(cell, "05_serve_endpoints.py", "exec"), namespace)   # noqa: S102 (the repo's own notebook cells)
    return printed.getvalue(), clock


def test_serving_rerun_at_the_served_champion_sends_no_update():
    """An endpoint that already serves the champion version with nothing pending gets no update (no
    new served-model config); the AI Gateway step still runs."""
    serving = _ServingEndpoints({"ep": _endpoint(4, "2")})
    out, clock = _run_deploy_cells(serving)
    assert serving.updates == [] and serving.creates == []
    assert f"ep already serves {config.uc_name(config.ARR_MODEL_NAME, 'cat', 'sch')} v2; no update" in out
    assert serving.gateways == ["ep"] and clock.now == 0


def test_serving_after_a_failed_update_settles_at_once_and_updates_again():
    """A failed update leaves UPDATE_FAILED and its aborted pending config behind (seen live): the
    wait returns at once with that state (it used to wait the whole timeout for pending_config to go
    away), the update is sent again, and an earlier version still showing UPDATE_FAILED right after
    it (not registered yet) does not count as the new update's result."""
    serving = _ServingEndpoints({"ep": _endpoint(4, "2", update="UPDATE_FAILED", pending=5)}, lag=1)
    out, clock = _run_deploy_cells(serving)
    assert len(serving.updates) == 1 and serving.endpoints["ep"]["version"] == 6
    assert "'ep' last config update UPDATE_FAILED (state: READY; pending config v5: ep-2 DEPLOYMENT_ABORTED: " in out
    assert "since the update timed out" in out and "(last config update: UPDATE_FAILED)" in out
    assert clock.now <= 4 * 30, clock.now         # a few polls, not the 2400 s timeout


def test_serving_updates_an_endpoint_at_the_champion_that_still_has_a_pending_config():
    """Current config the champion and NOT_UPDATING, but a pending config left behind: that is not
    "nothing to do", so the update is sent."""
    serving = _ServingEndpoints({"ep": _endpoint(4, "2", update="NOT_UPDATING", pending=5)})
    out, _ = _run_deploy_cells(serving)
    assert len(serving.updates) == 1 and "no update" not in out
    assert serving.endpoints["ep"]["pending"] is None and serving.endpoints["ep"]["update"] == "NOT_UPDATING"


@pytest.mark.parametrize("outcome", ["succeed", ("DEPLOYMENT_FAILED", "Container failed to start: model load error")],
                         ids=["succeed", "fail"])
def test_serving_update_that_reuses_the_config_version_settles_without_the_timeout(outcome):
    """Should Model Serving keep the config version number for the update, its result still settles
    the wait (completed on the champion config, or seen IN_PROGRESS and then ended on it), while the
    earlier UPDATE_FAILED at the same champion config, still showing right after the update call
    (``lag``), is not taken for it: a failure stops the task with the new failure's message, not the
    earlier one's (DEPLOYMENT_ABORTED)."""
    serving = _ServingEndpoints({"ep": _endpoint(4, "1", update="UPDATE_FAILED", pending=5, pending_served="2")},
                                outcome=outcome,
                                lag=1, reuse_version=True)
    clock = _Clock()
    if outcome == "succeed":
        _run_deploy_cells(serving, clock=clock)
        assert serving.endpoints["ep"]["entity"][0] == "ep-2" and serving.endpoints["ep"]["version"] == 5   # reused
    else:
        with pytest.raises(RuntimeError, match=r"did not complete \(config update UPDATE_FAILED\): ep-2 "
                                               r"DEPLOYMENT_FAILED: Container failed to start: model load error\."):
            _run_deploy_cells(serving, clock=clock)
    assert len(serving.updates) == 1 and clock.now <= 5 * 30, clock.now


FAR = 10 ** 12   # ms, about 30 years: Model Serving's event times far from the notebook's clock either way


@pytest.mark.parametrize("event_offset_ms", [0, -FAR, FAR], ids=["same-clock", "events-far-past", "events-far-future"])
def test_serving_update_that_reuses_the_config_version_and_fails_before_in_progress_fails_promptly(event_offset_ms):
    """A reused config version whose update fails before any poll sees it IN_PROGRESS looks, in the
    endpoint's state, like an earlier failure at that config. The endpoint's events tell them apart:
    an "Endpoint update failed" event that was not there when the update was sent stops the task at
    the first poll, with Model Serving's messages (it used to wait the whole 3000 s timeout), whatever
    the events' timestamps say compared with the notebook's clock."""
    serving = _ServingEndpoints({"ep": _endpoint(4, "1")}, outcome="fail", polls=0, reuse_version=True,
                                event_offset_ms=event_offset_ms)
    clock = _Clock()
    clock.now = 1000.0
    with pytest.raises(RuntimeError, match=r"^ep: the config update to cat\.sch\.bio_ecg12_arr v2 failed: Served "
                                           r"entity creation aborted for served entity `ep-2`, config version 4, since "
                                           r"the update timed out\. Endpoint update failed for endpoint `ep`, config "
                                           r"version 4\. The endpoint keeps serving its previous config"):
        _run_deploy_cells(serving, clock=clock)
    assert len(serving.updates) == 1 and serving.endpoints["ep"]["version"] == 4 and clock.now == 1000.0, clock.now
    assert serving.endpoints["ep"]["update"] == "UPDATE_FAILED" and serving.gateways == []


# A failure event from an earlier attempt, already there before the update is sent.
STALE = {"event_type": "ENDPOINT_UPDATE_EVENT", "timestamp": 400_000, "served_entity_name": None,
         "message": "Endpoint update failed for endpoint `ep`, config version 5."}


@pytest.mark.parametrize("stale_timestamp", [400_000, 1_000_000 + FAR], ids=["stale-before-clock", "stale-after-clock"])
def test_serving_ignores_a_failed_update_event_from_before_the_update(stale_timestamp):
    """An "Endpoint update failed" event from an earlier attempt (already among the events read just
    before this update was sent) is not this update's failure, even when its timestamp is later than
    the notebook's clock: with the earlier UPDATE_FAILED still showing right after the update call,
    the wait goes on, and the new update completes."""
    serving = _ServingEndpoints({"ep": _endpoint(4, "1", update="UPDATE_FAILED", pending=5, pending_served="2")},
                                lag=1, reuse_version=True, events=[{**STALE, "timestamp": stale_timestamp}])
    clock = _Clock()
    clock.now = 1000.0                                  # the notebook's clock: 1,000,000 ms when the update is sent
    out, _ = _run_deploy_cells(serving, clock=clock)
    assert serving.endpoints["ep"]["entity"][0] == "ep-2" and serving.endpoints["ep"]["update"] == "NOT_UPDATING"
    assert serving.gateways == ["ep"] and clock.now > 1000.0 and "failed:" not in out
    assert serving.event_reads >= 2                     # read before the send, and again while the stale state showed


def test_serving_accepted_gap_a_stale_failure_event_delivered_late_fails_the_task():
    """The accepted gap (deployment_path.update_failure): an earlier failure event missing from the
    read before the send (delivered late) but present in the first read after it, while the earlier
    UPDATE_FAILED still shows, is taken for this update's failure. The task stops with Model Serving's
    message, a false failure to run again; it changes nothing the endpoint serves: the update was
    sent once, nothing else is sent, and the AI Gateway step does not run."""
    serving = _ServingEndpoints({"ep": _endpoint(4, "1", update="UPDATE_FAILED", pending=5, pending_served="2")},
                                lag=1, reuse_version=True, late_events=[STALE])
    with pytest.raises(RuntimeError, match=r"^ep: the config update to cat\.sch\.bio_ecg12_arr v2 failed: Endpoint "
                                           r"update failed for endpoint `ep`, config version 5\."):
        _run_deploy_cells(serving)
    assert len(serving.updates) == 1 and serving.creates == [] and serving.gateways == [] and serving.event_reads == 2


def test_serving_with_events_unreadable_before_the_update_judges_by_the_state_alone():
    """The events read before the update fails: a note says the result is judged from the state alone,
    and the events are not read again, so the stale failure event (readable later) is never taken
    for this update's: the update completes."""
    serving = _ServingEndpoints({"ep": _endpoint(4, "1", update="UPDATE_FAILED", pending=5, pending_served="2")},
                                lag=1, reuse_version=True, events=[STALE],
                                events_error=ConnectionError("connection reset by peer"), events_error_calls=1)
    out, _ = _run_deploy_cells(serving)
    assert "ep: events not readable (ConnectionError: connection reset by peer)" in out
    assert "note: ep's events could not be read before the update, so its result is judged from the endpoint's state alone" in out
    assert serving.endpoints["ep"]["update"] == "NOT_UPDATING" and serving.gateways == ["ep"] and serving.event_reads == 1


def test_serving_update_failing_before_in_progress_with_unreadable_events_waits_for_the_timeout():
    """Events that cannot be read do not break the wait: it goes on with the endpoint's state alone,
    and the case the state cannot settle (a reused config version that fails before any poll sees it
    IN_PROGRESS) then ends at the wait cell's timeout (3000 s), not before and not later."""
    serving = _ServingEndpoints({"ep": _endpoint(4, "1")}, outcome="fail", polls=0, reuse_version=True,
                                events_error=ConnectionError("connection reset by peer"))
    clock = _Clock()
    with pytest.raises(TimeoutError, match=r"^'ep' not ready after 3000s$"):
        _run_deploy_cells(serving, clock=clock)
    assert len(serving.updates) == 1 and clock.now == 3000, clock.now
    assert serving.endpoints["ep"]["update"] == "UPDATE_FAILED" and serving.gateways == []


def test_serving_update_rejected_by_the_api_fails_at_once():
    """An update_endpoint call Model Serving refuses raises straight away; nothing waits for it."""
    serving = _ServingEndpoints({"ep": _endpoint(4, "1")}, reject=RuntimeError("INVALID_PARAMETER_VALUE: refused"))
    clock = _Clock()
    with pytest.raises(RuntimeError, match="INVALID_PARAMETER_VALUE: refused"):
        _run_deploy_cells(serving, clock=clock)
    assert clock.now == 0 and serving.gateways == []


def test_serving_deployment_failed_stops_the_task_with_its_message():
    serving = _ServingEndpoints({"ep": _endpoint(4, "1")},
                                outcome=("DEPLOYMENT_FAILED", "Container failed to start: model load error"))
    with pytest.raises(RuntimeError, match=r"UPDATE_FAILED\): ep-2 DEPLOYMENT_FAILED: Container failed to start: "
                                           r"model load error"):
        _run_deploy_cells(serving)


def test_serving_update_that_ends_failed_stops_the_task_with_the_message():
    serving = _ServingEndpoints({"ep": _endpoint(4, "1")}, outcome="fail")
    with pytest.raises(RuntimeError, match=r"ep: the config update to .* v2 did not complete \(config update "
                                           r"UPDATE_FAILED\): ep-2 DEPLOYMENT_ABORTED: Served entity creation "
                                           r"aborted .* since the update timed out"):
        _run_deploy_cells(serving)
    assert len(serving.updates) == 1 and serving.gateways == []


def test_serving_new_champion_version_is_updated_and_a_new_endpoint_created():
    serving = _ServingEndpoints({"ep": _endpoint(4, "1")})
    out, _ = _run_deploy_cells(serving)
    ((name, sent),) = serving.updates
    assert name == "ep" and sent["served_entities"][0]["entity_version"] == "2"
    assert serving.endpoints["ep"]["entity"][0] == "ep-2" and "updating ep -> " in out
    serving = _ServingEndpoints({})
    _run_deploy_cells(serving)
    assert [n for n, _ in serving.creates] == ["ep"] and serving.updates == [] and serving.gateways == ["ep"]


def test_serves_exactly_compares_every_field_deploy_sends():
    """A different size, scale-to-zero, model, version, entity name, extra entity or split traffic
    is not the champion config, so it is updated."""
    serving = _ServingEndpoints({})
    namespace = {"EndpointStateConfigUpdate": _ConfigUpdate}
    (helpers,) = [c for c in _cells(WORKFLOW / "05_serve_endpoints.py") if "def wait_for_endpoint_ready(" in c]
    exec(compile(helpers, "05_serve_endpoints.py", "exec"), namespace)   # noqa: S102 (the repo's own notebook cell)
    serves_exactly = namespace["serves_exactly"]
    target = {"served_entities": [{"name": "ep-2", "entity_name": "c.s.m", "entity_version": "2",
                                   "workload_size": "Small", "scale_to_zero_enabled": True}],
              "traffic_config": {"routes": [{"served_model_name": "ep-2", "traffic_percentage": 100}]}}
    good = serving._config(3, ("ep-2", "c.s.m", "2"))
    assert serves_exactly(good, target) and not serves_exactly(None, target)
    for field, value in (("workload_size", "Medium"), ("scale_to_zero_enabled", False), ("entity_name", "c.s.x"),
                         ("entity_version", "3"), ("name", "ep-3")):
        changed = serving._config(3, ("ep-2", "c.s.m", "2"))
        setattr(changed.served_entities[0], field, value)
        assert not serves_exactly(changed, target), field
    split = serving._config(3, ("ep-2", "c.s.m", "2"))
    split.traffic_config.routes[0].traffic_percentage = 50
    two = serving._config(3, ("ep-2", "c.s.m", "2"))
    two.served_entities.append(two.served_entities[0])
    assert not serves_exactly(split, target) and not serves_exactly(two, target)


def _deployment_path_cell():
    (cell,) = [c for c in _cells(WORKFLOW / "05_serve_endpoints.py") if "deployment_path.classify(" in c]
    return cell


def test_serving_reports_each_endpoints_deployment_path_and_tags_it_before_the_smoke_queries():
    """Once both endpoints are ready (after the wait + AI Gateway cell), 05_ reads each served model's
    build log and the endpoint's events, prints the path (common/deployment_path.py) and tags the
    endpoint `deployment`, before the smoke queries; then the removeAfter tag and the queries."""
    code = _python_source(WORKFLOW / "05_serve_endpoints.py")
    cell = _deployment_path_cell()
    assert "from common import config, deployment_path, scoring" in code
    for call in ("w.serving_endpoints.build_logs(name=endpoint_name, served_model_name=entity.name).logs",
                 "events = read_events(endpoint_name)",
                 "found = deployment_path.classify(entity.name, build_log, events)", "print(found.line)",
                 "add_tags=[EndpointTag(key=deployment_path.DEPLOYMENT_TAG, value=found.path)]"):
        assert call in cell, call
    # One read of the events API in the notebook, in the helper the update wait uses too.
    (helpers,) = [c for c in _cells(WORKFLOW / "05_serve_endpoints.py") if "def wait_for_endpoint_ready(" in c]
    assert code.count("api_client.do(") == 1 and 'w.api_client.do("GET", events_api(name))' in helpers
    assert code.count("/api/2.0/serving-endpoints/{") == 1 and "deployment_path.events_of(response)" in helpers
    assert code.index("w.serving_endpoints.put_ai_gateway(") < code.index("deployment_path.classify(")
    assert code.index("deployment_path.classify(") < code.index("deploy_client.predict(")


class _ServingStandIn:
    """``w.serving_endpoints`` + ``w.api_client`` for 05_'s tag cells. Each endpoint ``<x>`` serves
    ``<x>-1``; ``logs`` (by served entity) and ``events`` (the events API response, by endpoint) hold
    what Model Serving returns, an exception instance to raise instead. ``patch(add_tags=...)`` adds or
    updates the given keys and keeps the others, as the API does."""

    def __init__(self, logs, events):
        self.logs, self.events, self.tags = logs, events, {}
        self.serving_endpoints = SimpleNamespace(get=self.get, build_logs=self.build_logs, patch=self.patch)
        self.api_client = SimpleNamespace(do=self.do)

    def get(self, name):
        return SimpleNamespace(config=SimpleNamespace(served_entities=[SimpleNamespace(name=f"{name}-1")]))

    def build_logs(self, name, served_model_name):
        log = self.logs.get(served_model_name, "")
        if isinstance(log, BaseException):
            raise log
        return SimpleNamespace(logs=log)

    def do(self, method, path):
        assert method == "GET" and path.startswith("/api/2.0/serving-endpoints/") and path.endswith("/events")
        response = self.events.get(path.split("/")[-2], {"events": []})
        if isinstance(response, BaseException):
            raise response
        return response

    def patch(self, name, add_tags):
        self.tags.setdefault(name, {}).update({t.key: t.value for t in add_tags})


def _run_tag_cells(stand_in, endpoints, remove_after=""):
    """Run 05_'s helper cell (read_events), its deployment-path cell and then its removeAfter cell on
    the stand-in; returns the output."""
    from common import deployment_path as dp
    cells = _cells(WORKFLOW / "05_serve_endpoints.py")
    (helpers,) = [c for c in cells if "def wait_for_endpoint_ready(" in c]
    (remove_after_cell,) = [c for c in cells if 'key="removeAfter"' in c]
    namespace = {"SERVED": tuple(("m", e) for e in endpoints), "deployment_path": dp, "w": stand_in,
                 "remove_after": remove_after, "EndpointStateConfigUpdate": _ConfigUpdate,
                 "EndpointTag": lambda key, value: SimpleNamespace(key=key, value=value)}
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        for cell in (helpers, _deployment_path_cell(), remove_after_cell):
            exec(compile(cell, "05_serve_endpoints.py", "exec"), namespace)   # noqa: S102 (the repo's own notebook cell)
    return printed.getvalue()


EXPRESS_LOG = ("This environment was prepared ahead of time by express deployment and is shared by models with "
               "identical requirements.")


def test_serving_warns_but_never_fails_on_a_path_other_than_express():
    """The deployment-path cell with one express, one standard and one unreadable endpoint: it tags
    each with its path and warns for the two that are not express, each in its own words (unknown is
    missing evidence, not a container image build), without raising: eligibility is Model Serving's
    decision and the endpoint serves either way."""
    from common import deployment_path as dp
    assert not {"raise", "assert", "exit"} & set(_code_tokens_of(_deployment_path_cell()))
    stand_in = _ServingStandIn(
        logs={"a-1": EXPRESS_LOG, "b-1": "#18 0.382 Applying Databricks-managed pip constraints:",
              "c-1": RuntimeError("no build log")},
        events={"b": {"events": [{"event_type": dp.CONTAINER_EVENT, "served_entity_name": "b-1", "timestamp": 0,
                                  "message": dp.IMAGE_STARTED}]},
                "c": RuntimeError("no events")})
    out = _run_tag_cells(stand_in, ("a", "b", "c"))
    assert stand_in.tags == {"a": {"deployment": "express"}, "b": {"deployment": "standard"},
                             "c": {"deployment": "unknown"}}
    assert "a-1: EXPRESS deployment (environment prepared earlier and reused; no container image build)" in out
    warnings = [line for line in out.splitlines() if line.startswith("WARNING: ")]
    assert len(warnings) == 2
    assert warnings[0].startswith("WARNING: b did not use express deployment (it built a container image)")
    assert warnings[1].startswith("WARNING: the deployment path of c could not be determined (could not read "
                                  "its build log and the endpoint's events)")
    assert "did not use express" not in warnings[1] and "longer" not in warnings[1]


@pytest.mark.parametrize("log, events", [
    (TimeoutError("build-logs request timed out"), None),
    (None, ConnectionError("connection reset by peer")),
    (TimeoutError("timed out"), ConnectionError("connection reset by peer")),
], ids=["build-log-timeout", "events-connection-error", "both"])
def test_serving_reads_that_fail_in_any_way_leave_the_path_unknown_and_never_raise(log, events):
    """Not only API errors: a timeout or a dropped connection while reading the build log or the
    events (printed with its type) makes the path unknown."""
    stand_in = _ServingStandIn(logs={"a-1": log} if log is not None else {},
                               events={"a": events} if events is not None else {})
    out = _run_tag_cells(stand_in, ("a",))
    assert stand_in.tags == {"a": {"deployment": "unknown"}}
    for failure in (log, events):
        if failure is not None:
            assert f"not readable ({type(failure).__name__}: {failure})" in out


UNREADABLE_EVENTS = [TimeoutError("events request timed out"), ConnectionError("connection reset by peer"),
                     AttributeError("'NoneType' object has no attribute 'get'"),
                     {}, {"events": None}, {"events": "none"}, "<html>sign in</html>", None, ["not", "a", "mapping"],
                     {"events": ["not an event", 7]}]


@pytest.mark.parametrize("events", UNREADABLE_EVENTS, ids=[
    "timeout", "connection-error", "attribute-error", "no-events-key", "events-null", "events-string", "html",
    "none", "list", "bad-items"])
def test_serving_express_build_log_with_unreadable_events_is_unknown(events):
    """Events that could not be read (the call raised, or its response is not {"events": [...]}) make
    the path unknown even with an express build log: without them a container image build (express
    that fell back) would go unseen. The warning says what could not be read and never raises."""
    stand_in = _ServingStandIn(logs={"a-1": EXPRESS_LOG}, events={"a": events})
    out = _run_tag_cells(stand_in, ("a",))
    assert stand_in.tags == {"a": {"deployment": "unknown"}}
    assert "a-1: UNKNOWN deployment path (could not read the endpoint's events)" in out
    assert ("WARNING: the deployment path of a could not be determined (could not read the endpoint's events). "
            "This does not mean express was not used") in out
    expected = (f"{type(events).__name__}: {events}" if isinstance(events, BaseException)
                else "unexpected response: ")
    assert f"a: events not readable ({expected}" in out


def test_serving_valid_events_with_none_for_the_entity_keep_express_from_the_build_log():
    """A valid events response without events for this served entity (an update that reused its
    environment) is not an unreadable one: the express build log decides."""
    stand_in = _ServingStandIn(logs={"a-1": EXPRESS_LOG}, events={"a": {"events": []}})
    out = _run_tag_cells(stand_in, ("a",))
    assert stand_in.tags == {"a": {"deployment": "express"}} and "WARNING" not in out


def test_serving_tag_and_query_errors_still_fail_the_task():
    """Only reading the evidence is guarded: a failing tag call raises, and the smoke queries have no
    error handling of their own."""
    stand_in = _ServingStandIn(logs={"a-1": EXPRESS_LOG}, events={})

    def patch(name, add_tags):
        raise PermissionError("cannot tag the endpoint")

    stand_in.serving_endpoints.patch = patch
    with pytest.raises(PermissionError):
        _run_tag_cells(stand_in, ("a",))
    (smoke,) = [c for c in _cells(WORKFLOW / "05_serve_endpoints.py") if "deploy_client.predict(" in c]
    assert not {"try", "except"} & set(_code_tokens_of(smoke))


def test_serving_keeps_both_endpoint_tags():
    """05_ tags deployment and then, with a remove_after date, removeAfter: the second patch adds
    its key and both tags remain."""
    expiry = "2099-12-31"           # a stand-in date (test_repo_consistency.py: no real date committed)
    stand_in = _ServingStandIn(logs={"a-1": EXPRESS_LOG}, events={})
    _run_tag_cells(stand_in, ("a",), remove_after=expiry)
    assert stand_in.tags == {"a": {"deployment": "express", "removeAfter": expiry}}


def _code_tokens_of(source):
    """NAME tokens of a Python source string."""
    return [t.string for t in tokenize.generate_tokens(io.StringIO(source).readline) if t.type == tokenize.NAME]


def test_gate_logs_holdout_evaluation_artifacts_with_the_synthetic_caption():
    """04_ logs a confusion matrix, ROC + precision-recall curves, a per-class table and a README per
    model from the scores it already computed (one scoring pass), each carrying the caption."""
    code = _python_source(WORKFLOW / GATE)
    assert code.count("mlflow.pyfunc.load_model(") == 1 and code.count("scoring.score_driver_side(") == 1
    for logged in ("mlflow.log_figure(", "mlflow.log_dict(", "mlflow.log_table(", "mlflow.log_text(",
                   "eval_plots.confusion_matrix(", "eval_plots.roc_curve(", "eval_plots.precision_recall_curve(",
                   "evaluation.per_class_report(", "holdout_scores[model_name]", '"holdout_eval_artifacts"'):
        assert logged in code, logged
    assert code.count("EVAL_CAPTION") >= 6                    # defined once; on 3 figures, the JSON, the README
    assert "separable by design" in code


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_pip_installs_are_pinned(path):
    """Every %pip install names exact versions (==) and never upgrades to whatever is latest."""
    for line in path.read_text().splitlines():
        if not line.startswith("# MAGIC %pip install"):
            continue
        args = line.removeprefix("# MAGIC %pip install").split()
        flags = [a for a in args if a.startswith("-")]
        packages = [a.strip('"\'') for a in args if not a.startswith("-")]
        assert set(flags) <= {"--no-deps"}, f"{path.name}: {line}"
        assert packages and all("==" in p for p in packages), f"{path.name}: {line}"


def test_pip_lines_are_exactly_the_recorded_ones_and_run_before_any_code():
    """Each notebook's %pip lines are exactly PIP_LINES (conftest.py), so no version changes without
    the check against its environment; only the notebooks that use MLflow name it. The install cells
    come first (only markdown before them) and the restart right after them, so no code (no import)
    runs before the pinned versions are in place; a notebook that installs nothing does not restart."""
    assert set(PIP_LINES) == {p.name for p in NOTEBOOKS}
    for path in NOTEBOOKS:
        lines = pip_lines(path)
        assert lines == PIP_LINES[path.name], path.name
        imports_mlflow = "mlflow" in _code_tokens(path)
        assert imports_mlflow == (path.name in MLFLOW_NOTEBOOKS), path.name
        if not imports_mlflow:
            assert not [line for line in lines if "mlflow" in line], path.name
        if not lines:
            assert "restartPython()" not in path.read_text(), f"{path.name}: restarts with nothing installed"
            continue
        cells = _cells(path)
        pip_cells = [i for i, cell in enumerate(cells) if "# MAGIC %pip install" in cell]
        assert all("# MAGIC %md" in cell for cell in cells[:pip_cells[0]]), path.name
        assert pip_cells == list(range(pip_cells[0], pip_cells[-1] + 1)), path.name
        assert cells[pip_cells[-1] + 1].strip().endswith("dbutils.library.restartPython()"), path.name


@pytest.mark.parametrize("name", sorted(TRAINING))
def test_training_logs_the_model_with_the_shared_exact_pip_requirements(name):
    """Both models are logged with pip_requirements=config.MODEL_PIP_REQUIREMENTS (one list for
    both), not MLflow's inferred list, a conda env or extra requirements."""
    calls = [node for node in ast.walk(ast.parse(_python_source(WORKFLOW / name)))
             if isinstance(node, ast.Call) and ast.unparse(node.func) == "mlflow.pyfunc.log_model"]
    assert len(calls) == 1, name
    keywords = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
    assert keywords.get("pip_requirements") == "config.MODEL_PIP_REQUIREMENTS", name
    assert not {"conda_env", "extra_pip_requirements", "code_paths"} & set(keywords), name


def test_model_pip_requirements_are_exact_plain_pypi_pins():
    """Exact name==version pins from the default package index, only for what the pyfunc imports at
    serving time (the training versions, so the pickled model loads), and nothing express deployment
    lists as a blocker (no URL, index option, editable or git entry, no CUDA or other local version tag)."""
    reqs = config.MODEL_PIP_REQUIREMENTS
    assert all(re.fullmatch(r"[a-z0-9-]+==\d+(\.\d+)*", r) for r in reqs), reqs
    assert [r.split("==")[0] for r in reqs] == ["mlflow", "cloudpickle", "numpy", "pandas", "torch"]
    # Locked literally: numpy, pandas, torch, cloudpickle and mlflow must equal the AI Runtime
    # environment 6 versions the models are trained (and cloudpickled and logged) in (conftest.PREINSTALLED
    # for mlflow-skinny and torch). A bump goes here, in config.py and, for mlflow and torch, in
    # PIP_LINES and the notebooks' %pip lines (test_repo_consistency.py checks each environment).
    assert reqs == ["mlflow==3.13.0", "cloudpickle==3.1.1", "numpy==2.3.4", "pandas==2.3.3", "torch==2.11.0"]


def _model_cell(path):
    """The training notebook's 'Model + inference wrapper' cell, as Python source."""
    (cell,) = [c for c in _cells(path) if "Model + inference wrapper" in c]
    return cell


@pytest.mark.parametrize("name", sorted(TRAINING))
def test_model_code_is_inline_and_self_contained(name):
    path = WORKFLOW / name
    cnn_name, pyfunc_name = TRAINING[name]
    module = ast.parse(_python_source(path))
    top = {node.name: node for node in module.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    for required in (cnn_name, pyfunc_name, "_coerce_ecg"):
        assert required in top, f"{name}: {required} must be defined in the notebook"
        used = {n.id for n in ast.walk(top[required]) if isinstance(n, ast.Name)}
        assert not used & SHARED_NAMES, f"{name}: {required} uses {used & SHARED_NAMES}"
    # First conv layer takes n_leads channels, defaulting to the notebook's N_LEADS (= 12).
    init = next(f for f in top[cnn_name].body if isinstance(f, ast.FunctionDef) and f.name == "__init__")
    assert ast.unparse(init.args.defaults[0]) == "N_LEADS"
    first_conv = next(c for c in ast.walk(top[cnn_name]) if isinstance(c, ast.Call) and ast.unparse(c.func) == "nn.Conv1d")
    assert ast.unparse(first_conv.args[0]) == "n_leads"
    assert "N_LEADS = int(config.N_LEADS)" in path.read_text()
    if pyfunc_name == "AfibModelPyfunc":
        assert "AFIB_TEMPERATURE = float(config.AFIB_TEMPERATURE)" in path.read_text()


@pytest.mark.parametrize("name", sorted(TRAINING))
def test_pyfunc_picks_device_at_load(name):
    """The wrapper loads onto whichever device is present (GPU if any, else CPU) instead of a
    hard-coded one, says which it picked once loaded, moves inputs there, and brings outputs back
    to CPU before numpy."""
    _, pyfunc_name = TRAINING[name]
    module = ast.parse(_python_source(WORKFLOW / name))
    (cls,) = [n for n in module.body if isinstance(n, ast.ClassDef) and n.name == pyfunc_name]
    methods = {f.name: ast.unparse(f) for f in cls.body if isinstance(f, ast.FunctionDef)}
    load, predict = methods["load_context"], methods["predict"]
    assert "self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')" in load
    assert "map_location=self._device" in load and ".to(self._device)" in load
    loaded = "print(f'{type(self).__name__} loaded on {self._device}')"
    assert load.index(".to(self._device).eval()") < load.index(loaded)    # logged after the move
    assert ".to(self._device)" in predict and ".cpu().numpy()" in predict
    device_strings = [n.value for n in ast.walk(cls) if isinstance(n, ast.Constant) and n.value in ("cpu", "cuda")]
    assert device_strings == ["cuda", "cpu"], f"{pyfunc_name}: device hard-coded outside the load-time pick"
    assert ".cuda()" not in ast.unparse(cls)


# --- serving check: pickle by value, load where `common` does not exist --------------------

_LOAD_AND_PREDICT = textwrap.dedent("""
    import sys, types, pickle
    from types import SimpleNamespace
    import numpy as np, pandas as pd
    assert "common" not in sys.modules
    try:
        import common  # noqa: F401
        raise SystemExit("common must not be importable here")
    except ImportError:
        pass
    mlflow = types.ModuleType("mlflow"); mlflow.pyfunc = types.ModuleType("mlflow.pyfunc")
    class PythonModel: pass
    mlflow.pyfunc.PythonModel = PythonModel
    sys.modules.update({"mlflow": mlflow, "mlflow.pyfunc": mlflow.pyfunc})
    model = pickle.loads(open(sys.argv[1], "rb").read())
    model.load_context(SimpleNamespace(artifacts={"state_dict": sys.argv[2]}))
    x = np.random.default_rng(0).normal(size=(3, 12, 2500)).astype("float32")
    out = model.predict(None, x)
    flat = model.predict(None, pd.DataFrame({"ecg": list(x.reshape(3, -1))}))
    if isinstance(out, pd.DataFrame):
        print("frame", list(out.columns), out.shape, sorted(set(out["arr_flag"].tolist()) | {0.0, 1.0}))
    else:
        print("array", np.asarray(out).shape, bool(np.allclose(np.asarray(out).sum(axis=1), 1.0)))
    print("flat_equal", bool(np.allclose(np.asarray(out), np.asarray(flat))))
    import torch
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    params = {p.device.type for p in model._model.parameters()}
    print("device_ok", model._device.type == expected and params == {expected})
    arrays = [out[c].to_numpy() for c in out.columns] if isinstance(out, pd.DataFrame) else [out]
    print("numpy_out", sorted({f"{type(a).__name__} {a.dtype}" for a in arrays}))
""")


@pytest.mark.parametrize("name", sorted(TRAINING))
def test_inline_model_serves_without_common(name, tmp_path):
    torch = pytest.importorskip("torch")
    cloudpickle = pytest.importorskip("cloudpickle")
    cnn_name, pyfunc_name = TRAINING[name]

    # Stand-in for the notebook: its constants + the model cell, in a module pickled by value.
    stub = types.ModuleType("mlflow")
    stub.pyfunc = types.SimpleNamespace(PythonModel=type("PythonModel", (), {}))
    nb = types.ModuleType(f"notebook_{cnn_name.lower()}")
    nb.__dict__.update(np=np, pd=__import__("pandas"), torch=torch, nn=torch.nn, mlflow=stub,
                       N_LEADS=12, N_SAMPLES=2500, ARR_FLAG_THRESHOLD=0.5, AFIB_TEMPERATURE=2.0)
    sys.modules[nb.__name__] = nb
    try:
        exec(compile(_model_cell(WORKFLOW / name), name, "exec"), nb.__dict__)   # noqa: S102 (the repo's own model cell)
        cloudpickle.register_pickle_by_value(nb)
        state_dict = tmp_path / "state_dict.pt"
        torch.save(getattr(nb, cnn_name)().state_dict(), state_dict)
        pickled = tmp_path / "model.pkl"
        python_model = getattr(nb, pyfunc_name)()
        assert not hasattr(python_model, "_device")       # the device is picked at load time, never pickled
        pickled.write_bytes(cloudpickle.dumps(python_model))
    finally:
        sys.modules.pop(nb.__name__, None)

    # Fresh interpreter, cwd and sys.path without projects/ecg12-afib: `common` cannot be imported.
    result = subprocess.run([sys.executable, "-I", "-c", _LOAD_AND_PREDICT, str(pickled), str(state_dict)],
                            capture_output=True, text=True, cwd=tmp_path, timeout=300, check=False)
    assert result.returncode == 0, result.stderr
    if pyfunc_name == "ArrModelPyfunc":
        assert "frame ['arr_prob', 'arr_flag'] (3, 2) [0.0, 1.0]" in result.stdout
    else:
        assert "array (3, 2) True" in result.stdout
    assert "flat_equal True" in result.stdout
    assert "device_ok True" in result.stdout                      # model sits on the device it picked
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert f"{pyfunc_name} loaded on {expected}" in result.stdout.splitlines()   # load_context says so
    assert "numpy_out ['ndarray float32']" in result.stdout        # outputs come back as CPU numpy
