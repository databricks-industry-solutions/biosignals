"""Static checks of the workflow notebooks (they are NOT executed on Databricks here; the SETUP and
DAILY flows run locally with stand-ins in test_setup_flow_local.py / test_daily_flow_local.py),
plus a serving check of the training notebooks' inline model code: it is pickled the way MLflow
pickles it and loaded in a fresh process where the `common` package does not exist (as on a
serving endpoint)."""
import ast
import contextlib
import enum
import fractions
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
from conftest import DEMO_ROOT, GOVERNED_TAG_VALUES, PIP_LINES, TORCH_CPU_INDEX, pip_lines

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
    assert "check_ownership(endpoint_name, w.serving_endpoints.get(name=endpoint_name), full)" in code
    assert "raise RuntimeError(" in code and "endpoint_prefix" in code
    # The check runs for both endpoints (its own cell) before the first create / update call, and
    # again in deploy() on the snapshot it sends the update from.
    assert code.index("check_ownership(endpoint_name, w.serving_endpoints") < code.index(
        "deployed[endpoint_name] = deploy(model_name, endpoint_name, existing)")
    assert "check_ownership(endpoint_name, ep, full)" in code
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
    before the update (the update not registered yet); ``idle``: gets after those that return the new
    pending config with the state still NOT_UPDATING (registered, not started yet). ``reuse_version``:
    the new pending config keeps the latest version number instead of the next one. ``reject``:
    update_endpoint raises it. A pending config carries ``start_time``, Model Serving's timestamp of
    when that update started: the clock's time in milliseconds plus ``start_offset_ms`` (Model
    Serving's clock need not match the notebook's), or ``start_time`` when given (to make it equal an
    earlier one); ``start_times=False``: the API leaves it out. ``stale``: gets after the update
    registers that show its new pending config with the state still the one from before the update
    (a snapshot whose parts were read at different moments).
    ``events_error``: the endpoints' events API (``api_client.do``, which the wait must not need)
    raises it; ``event_reads`` counts its calls. ``progress_state``: the deployment state the new
    pending config's served entity shows until the update ends (None: no state). ``rival``: another
    writer's served entity (name, model, version) that replaces this update's in the pending config
    once it registers. ``no_state``: gets after the update registers that show no endpoint state.
    ``per``: per-endpoint ``outcome`` / ``polls``. ``missing``: fields of the new pending config's
    served entity the API leaves out (None; the SDK models them as optional). ``current_missing``:
    (fields, gets): once the update has completed, the first ``gets`` gets leave those fields out of
    the current config. ``baseline_missing``: (fields, when): the pending config left before the
    update leaves them out when deploy() reads it ("read") or in the polls after the update that
    still show it ("later"); "current.<field>" names a field of the current config instead (left
    out when deploy() reads it, given in later polls). A field is a config's (config_version, start_time, traffic_config), its
    route's ("route.<name>"), "state" (the served entity's deployment state) or its served entity's
    (see _drop). ``edited``: {field: value} set on the current config once the update has completed
    (someone changed it in the Serving UI, say). The clock is _run_deploy_cells'."""

    def __init__(self, endpoints, outcome="succeed", polls=2, lag=0, reuse_version=False, reject=None,
                 events_error=None, idle=0, start_offset_ms=0, start_times=True, start_time=None, stale=0,
                 progress_state="DEPLOYMENT_CREATING", rival=None, no_state=0, per=None, missing=(),
                 current_missing=((), 0), baseline_missing=((), None), edited=None):
        self.endpoints, self.outcome, self.polls, self.lag = endpoints, outcome, polls, lag
        self.progress_state, self.rival, self.no_state, self.per = progress_state, rival, no_state, per or {}
        self.missing, self.current_missing, self.baseline_missing = missing, current_missing, baseline_missing
        self.edited = edited or {}
        self.idle, self.start_offset_ms, self.start_times = idle, start_offset_ms, start_times
        self.start_time, self.stale = start_time, stale
        self.reuse_version, self.reject = reuse_version, reject
        self.events_error, self.clock, self.event_reads = events_error, None, 0
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
                                             scale_to_zero_enabled=True, state=None if deployment is None else SimpleNamespace(
                                                 deployment=_Deployment[deployment], deployment_state_message=message))],
            traffic_config=SimpleNamespace(routes=[SimpleNamespace(served_model_name=name, served_entity_name=None,
                                                                   traffic_percentage=100)]))

    @staticmethod
    def _set(endpoint_config, field, value):
        """Set ``field`` of a config (see _drop) to ``value``."""
        if field in ("config_version", "start_time", "traffic_config", "served_entities"):
            setattr(endpoint_config, field, value)
        elif field.startswith("route."):
            setattr(endpoint_config.traffic_config.routes[0], field[len("route."):], value)
        else:
            setattr(endpoint_config.served_entities[0], field, value)

    @staticmethod
    def _drop(endpoint_config, field):
        """Leave ``field`` out of a config (None, as the SDK gives a field the API did not return)."""
        if field == "served_entities[]":                         # listed, but empty
            endpoint_config.served_entities = []
        else:
            _ServingEndpoints._set(endpoint_config, field, None)

    def get(self, name):
        self.gets += 1
        e = self.endpoints[name]
        if e.get("lag"):
            e["lag"] -= 1
            snapshot = e["before"]
        elif e.get("no_state"):
            e["no_state"] -= 1
            return SimpleNamespace(name=name, config=None, pending_config=None, state=None)
        elif e.get("idle"):
            e["idle"] -= 1
            snapshot = {**e, "update": "NOT_UPDATING"}
        elif e.get("stale"):
            e["stale"] -= 1
            snapshot = {**e, "update": e["before"]["update"]}
        else:
            if e.get("remaining") is not None:
                if e["remaining"] > 0:
                    e["remaining"] -= 1
                else:
                    e["remaining"] = None
                    if e["outcome"] == "succeed":
                        e.update(version=e["pending"][0], entity=e["pending"][1], pending=None, pending_start=None,
                                 update="NOT_UPDATING", current_missing=self.current_missing[1])
                    else:
                        deployment, message = ("DEPLOYMENT_ABORTED", None) if e["outcome"] == "fail" else e["outcome"]
                        e.update(update="UPDATE_FAILED", pending=(*e["pending"][:2], deployment, message))
            snapshot = e
        current = self._config(snapshot["version"], snapshot["entity"]) if snapshot.get("entity") else None
        if current is not None and "before" not in e and self.baseline_missing[1] == "read":
            for field in self.baseline_missing[0]:
                if field.startswith("current."):
                    self._drop(current, field[len("current."):])
        if snapshot is e and e.get("current_missing"):
            e["current_missing"] -= 1
            for field in self.current_missing[0]:
                self._drop(current, field)
        if snapshot is e and "before" in e and e.get("pending") is None:      # the update has completed
            for field, value in self.edited.items():
                self._set(current, field, value)
        pending = None
        if snapshot.get("pending"):
            version, entity, deployment, *custom = snapshot["pending"]
            message = (custom[0] if custom and custom[0] else ABORTED.format(name=entity[0], version=version)
                       if deployment == "DEPLOYMENT_ABORTED" else "Deploying")
            pending = self._config(version, entity, deployment, message)
            pending.start_time = snapshot.get("pending_start") if self.start_times else None
            for field in snapshot.get("missing", ()):
                self._drop(pending, field)
            fields, when = self.baseline_missing
            if ("before" not in e and when == "read") or (when == "later" and snapshot is e.get("before")):
                for field in fields:
                    if not field.startswith("current."):
                        self._drop(pending, field)
        return SimpleNamespace(name=name, config=current, pending_config=pending, state=SimpleNamespace(
            config_update=_ConfigUpdate[snapshot["update"]], ready=_Ready["READY" if current else "NOT_READY"]))

    def _start(self, name, endpoint_config):
        e = self.endpoints.setdefault(name, {"version": 0, "entity": None, "update": "NOT_UPDATING", "pending": None})
        (entity,) = endpoint_config["served_entities"]
        e["before"] = {k: e.get(k) for k in ("version", "entity", "update", "pending", "pending_start")}
        latest = max([e["version"]] + ([e["pending"][0]] if e["pending"] else []))
        per = self.per.get(name, {})
        e.update(pending=(latest + (0 if self.reuse_version else 1),
                          self.rival or (entity["name"], entity["entity_name"], entity["entity_version"]),
                          self.progress_state), update="IN_PROGRESS", remaining=per.get("polls", self.polls),
                 outcome=per.get("outcome", self.outcome), lag=self.lag, idle=self.idle, stale=self.stale,
                 no_state=self.no_state, missing=self.missing,
                 pending_start=self.start_time if self.start_time is not None
                 else round(self.clock.now * 1000) + self.start_offset_ms)

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
        if self.events_error is not None:
            raise self.events_error
        return {"events": []}


def _endpoint(version, served_version, update="NOT_UPDATING", pending=None, pending_served=None):
    """An existing endpoint ``ep`` at config ``version`` serving model version ``served_version``, with
    an aborted pending config ``pending`` (its version) for model version ``pending_served`` (default:
    the served one), left by an update that started at 400,000 ms on Model Serving's clock."""
    full = config.uc_name(config.ARR_MODEL_NAME, "cat", "sch")
    left = pending_served or served_version
    return {"version": version, "entity": (f"ep-{served_version}", full, served_version), "update": update,
            "pending": None if pending is None else (pending, (f"ep-{left}", full, left), "DEPLOYMENT_ABORTED"),
            "pending_start": None if pending is None else 400_000}


def _run_deploy_cells(serving, champion="2", clock=None, served=((config.ARR_MODEL_NAME, "ep"),), refusal=False):
    """05_'s helper, issue and wait cells on the stand-in, for the (model, endpoint) pairs ``served``
    (default: one endpoint ``ep`` serving the ARR model), with ``refusal`` its ownership-check cell
    first; returns (output, clock). The cells' namespace is left on ``serving.namespace``."""
    from common import deployment_path

    clock = clock or _Clock()
    serving.clock = clock
    cells = _cells(WORKFLOW / "05_serve_endpoints.py")
    (helpers,) = [c for c in cells if "def wait_for_endpoint_ready(" in c]
    (issue,) = [c for c in cells if "# DBTITLE 1,Issue both deploys" in c]
    (wait,) = [c for c in cells if "# DBTITLE 1,Wait for both ready" in c]
    (check,) = [c for c in cells if "# DBTITLE 1,Refuse an existing endpoint" in c]
    namespace = {
        "w": SimpleNamespace(serving_endpoints=serving.serving_endpoints, api_client=serving.api_client),
        "deploy_client": serving.deploy_client, "deployment_path": deployment_path,
        "time": clock, "config": config, "catalog": "cat", "schema": "sch", "mc": None,
        "scoring": SimpleNamespace(resolve_alias=lambda client, full, alias: SimpleNamespace(version=champion)),
        "EndpointStateConfigUpdate": _ConfigUpdate, "SERVED": served,
        "existing": set(serving.endpoints), "AiGatewayInferenceTableConfig": lambda **kw: kw,
        "AiGatewayUsageTrackingConfig": lambda **kw: kw}
    serving.namespace = namespace
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        for cell in (helpers, check, issue, wait) if refusal else (helpers, issue, wait):
            exec(compile(cell, "05_serve_endpoints.py", "exec"), namespace)   # noqa: S102 (the repo's own notebook cells)
    return printed.getvalue(), clock


def _run_refusal_cell(endpoint):
    """05_'s helper cell and its refusal cell for one existing endpoint ``ep`` (a stand-in config)."""
    cells = _cells(WORKFLOW / "05_serve_endpoints.py")
    (helpers,) = [c for c in cells if "def wait_for_endpoint_ready(" in c]
    (refusal,) = [c for c in cells if "# DBTITLE 1,Refuse an existing endpoint" in c]
    namespace = {"EndpointStateConfigUpdate": _ConfigUpdate, "config": config, "catalog": "cat", "schema": "sch",
                 "SERVED": ((config.ARR_MODEL_NAME, "ep"),),
                 "w": SimpleNamespace(serving_endpoints=SimpleNamespace(get=lambda name: endpoint)),
                 "deploy_client": SimpleNamespace(list_endpoints=lambda: [{"name": "ep"}])}
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        for cell in (helpers, refusal):
            exec(compile(cell, "05_serve_endpoints.py", "exec"), namespace)   # noqa: S102 (the repo's own notebook cells)
    return printed.getvalue()


def test_serving_refusal_never_takes_missing_model_names_for_another_deployment():
    """An existing endpoint whose served entity leaves out its model name is not "another
    deployment's": the task stops saying it could not tell which model the endpoint serves. One that
    names another model is refused as another deployment's; this deployment's goes on."""
    full = config.uc_name(config.ARR_MODEL_NAME, "cat", "sch")
    ours = _ServingEndpoints._config(4, ("ep-1", full, "1"))
    endpoint = SimpleNamespace(config=ours, pending_config=None)
    assert "exists and serves cat.sch.bio_ecg12_arr" in _run_refusal_cell(endpoint)
    unnamed = _ServingEndpoints._config(4, ("ep-1", None, "1"))
    with pytest.raises(RuntimeError) as raised:
        _run_refusal_cell(SimpleNamespace(config=unnamed, pending_config=None))
    assert "could not tell which model" in str(raised.value) and "another deployment" not in str(raised.value)
    other = _ServingEndpoints._config(4, ("ep-1", "c.s.other", "1"))
    with pytest.raises(RuntimeError, match="belongs to another deployment"):
        _run_refusal_cell(SimpleNamespace(config=other, pending_config=None))


OURS = config.uc_name(config.ARR_MODEL_NAME, "cat", "sch")


def _ownership_case(case):
    """An existing endpoint ``ep`` for the ownership guard. "own": this deployment's model, current
    and pending. "current-unknown-*": the current config left out, or listing no served model, while a
    pending config names this deployment's model. "pending-unreadable-*": the current config names
    it, and a pending config is there but its served models are left out or empty. "other": the
    current config serves another deployment's model."""
    leftover = _endpoint(4, "1", update="UPDATE_FAILED", pending=5, pending_served="2")
    if case == "own":
        return _ServingEndpoints({"ep": leftover})
    if case == "current-unknown-missing":
        return _ServingEndpoints({"ep": {**leftover, "entity": None}})
    if case == "current-unknown-empty":
        return _ServingEndpoints({"ep": leftover}, baseline_missing=(("current.served_entities[]",), "read"))
    if case.startswith("pending-unreadable"):
        field = "served_entities" if case.endswith("missing") else "served_entities[]"
        return _ServingEndpoints({"ep": leftover}, baseline_missing=((field,), "read"))
    return _ServingEndpoints({"ep": {**leftover, "entity": ("ep-1", "c.s.another_model", "1")}})


@pytest.mark.parametrize("refusal", [True, False], ids=["ownership-cell", "deploy-before-sending"])
@pytest.mark.parametrize("case", ["current-unknown-missing", "current-unknown-empty", "pending-unreadable-missing",
                                  "pending-unreadable-empty", "other", "own"])
def test_serving_ownership_guard_fails_safe_before_every_update(case, refusal):
    """The ownership guard, fail safe, in the ownership cell (both endpoints before either changes)
    and in deploy() on the very snapshot it updates from: the current config must be readable and
    serve only this deployment's model, and a pending config there must name only that model too.
    Anything unknown stops the task with "could not tell which model", another model as another
    deployment's, and neither ever sends the update; this deployment's own endpoint is updated (before:
    a pending config naming this model passed for an unknown current config, and a known current
    model hid an unreadable pending config)."""
    serving = _ownership_case(case)
    if case == "own":
        _run_deploy_cells(serving, refusal=refusal)
        assert len(serving.updates) == 1 and serving.endpoints["ep"]["entity"][0] == "ep-2"
        return
    with pytest.raises(RuntimeError) as raised:
        _run_deploy_cells(serving, refusal=refusal)
    message = str(raised.value)
    if case == "other":
        assert "belongs to another deployment" in message and "c.s.another_model" in message, message
    else:
        assert "could not tell which model" in message and "another deployment" not in message, message
    assert serving.updates == [] and serving.creates == []


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
    the wait (completed on the champion config, or a failure of the pending config that is not the
    one left before), while the earlier UPDATE_FAILED at the same champion config, still showing right
    after the update call (``lag``), is not taken for it: a failure stops the task with the new
    failure's message, not the earlier one's (DEPLOYMENT_ABORTED)."""
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
    assert len(serving.updates) == 1 and clock.now <= 5 * 30 and serving.event_reads == 0, clock.now


FAR = 10 ** 12   # ms, about 30 years: Model Serving's event times far from the notebook's clock either way


@pytest.mark.parametrize("start_offset_ms", [0, -FAR, FAR], ids=["same-clock", "server-far-past", "server-far-future"])
@pytest.mark.parametrize("leftover", [False, True], ids=["no-earlier-failure", "earlier-failure-at-the-same-config"])
def test_serving_reused_config_version_failing_before_in_progress_fails_promptly_without_events(leftover, start_offset_ms):
    """An update that keeps the config version number and fails before any poll sees it IN_PROGRESS,
    with the endpoint's events unreadable, stops the task at the first poll with Model Serving's
    message (it used to wait the whole 3000 s timeout). Even after an earlier failure at the very same
    config (same version, same served entity, same message), the pending config is not the one left
    before the update: here its start_time, Model Serving's own timestamp of when the update started,
    differs. Only compared with that earlier value, never with the notebook's clock, and the wait
    reads no events."""
    endpoint = (_endpoint(4, "1", update="UPDATE_FAILED", pending=4, pending_served="2") if leftover
                else _endpoint(4, "1"))
    serving = _ServingEndpoints({"ep": endpoint}, outcome="fail", polls=0, reuse_version=True,
                                start_offset_ms=start_offset_ms, events_error=ConnectionError("connection reset by peer"))
    clock = _Clock()
    clock.now = 1000.0
    with pytest.raises(RuntimeError, match=r"^ep: the config update to cat\.sch\.bio_ecg12_arr v2 did not complete "
                                           r"\(config update UPDATE_FAILED\): ep-2 DEPLOYMENT_ABORTED: Served entity "
                                           r"creation aborted for served entity `ep-2`, config version 4, since the "
                                           r"update timed out\. The endpoint keeps serving its previous config"):
        _run_deploy_cells(serving, clock=clock)
    assert len(serving.updates) == 1 and serving.endpoints["ep"]["version"] == 4 and clock.now == 1000.0, clock.now
    assert serving.gateways == [] and serving.event_reads == 0


@pytest.mark.parametrize("existing", [False, True], ids=["create", "update"])
@pytest.mark.parametrize("early", ["lag", "idle"], ids=["not-registered-yet", "registered-not-started"])
def test_serving_first_poll_before_in_progress_keeps_waiting(existing, early):
    """The first poll after a create or an update can come before the endpoint shows it IN_PROGRESS:
    the endpoint as it was (for a create: no config at all), or the new pending config with the state
    still NOT_UPDATING. Neither is taken for the result: the wait goes on and the update completes
    (a new endpoint used to stop the task at once with "did not complete")."""
    serving = _ServingEndpoints({"ep": _endpoint(4, "1")} if existing else {}, **{early: 1})
    out, clock = _run_deploy_cells(serving)
    assert len(serving.updates if existing else serving.creates) == 1
    assert serving.endpoints["ep"]["entity"][0] == "ep-2" and serving.endpoints["ep"]["update"] == "NOT_UPDATING"
    assert serving.gateways == ["ep"] and "did not complete" not in out and clock.now > 0


def test_serving_new_endpoint_that_fails_stops_the_task_with_the_message():
    """Regression guard (the code before the start_time change got this right too): a create that
    ends UPDATE_FAILED is that create's failure, since there was no endpoint before."""
    serving = _ServingEndpoints({}, outcome=("DEPLOYMENT_FAILED", "Container failed to start: model load error"))
    with pytest.raises(RuntimeError, match=r"did not complete \(config update UPDATE_FAILED\): ep-2 DEPLOYMENT_FAILED: "
                                           r"Container failed to start: model load error"):
        _run_deploy_cells(serving)
    assert len(serving.creates) == 1 and serving.gateways == []


@pytest.mark.parametrize("start", ["missing", "identical"])
def test_serving_failure_that_looks_like_the_earlier_one_stops_soon_with_a_clear_message(start):
    """After an earlier failure at the very same config, an update that keeps the config version and
    fails before any poll sees it IN_PROGRESS can leave a pending config identical to the earlier one:
    no start_time from the API, or the same one. Nothing in the endpoint's state then says whether
    the update failed or has not started, so the task neither claims a failure nor waits out the
    3000 s timeout: once the endpoint has shown no progress and no result clearly its own for the
    wait's limit for that (UNCLEAR_LIMIT_S), it stops with a message that says so and where to look."""
    serving = _ServingEndpoints({"ep": _endpoint(4, "1", update="UPDATE_FAILED", pending=4, pending_served="2")},
                                outcome="fail", polls=0, reuse_version=True,
                                **({"start_times": False} if start == "missing" else {"start_time": 400_000}))
    clock = _Clock()
    with pytest.raises(RuntimeError) as raised:
        _run_deploy_cells(serving, clock=clock)
    message = str(raised.value)
    assert message.startswith("ep: could not tell whether the config update to cat.sch.bio_ecg12_arr v2 has "
                              "started or how it ended"), message
    assert "databricks api get /api/2.0/serving-endpoints/ep/events" in message and "did not complete" not in message
    limit = serving.namespace["UNCLEAR_LIMIT_S"]
    assert limit <= 300 and limit <= clock.now <= limit + 30, clock.now
    assert len(serving.updates) == 1 and serving.gateways == []


def test_serving_reused_version_without_start_time_and_no_earlier_pending_config_fails_promptly():
    """Without a start_time from the API but with no pending config left before the update, any
    pending config after it is this update's: one that fails before any poll sees it IN_PROGRESS
    stops the task at the first poll with Model Serving's message (it used to wait 3000 s)."""
    serving = _ServingEndpoints({"ep": _endpoint(4, "1")}, outcome="fail", polls=0, reuse_version=True,
                                start_times=False)
    clock = _Clock()
    with pytest.raises(RuntimeError, match=r"^ep: the config update to cat\.sch\.bio_ecg12_arr v2 did not complete "
                                           r"\(config update UPDATE_FAILED\): ep-2 DEPLOYMENT_ABORTED: "):
        _run_deploy_cells(serving, clock=clock)
    assert len(serving.updates) == 1 and clock.now == 0, clock.now


def test_serving_new_pending_config_with_the_earlier_failed_state_is_not_a_failure():
    """A poll can show the update's new pending config (new start_time, its served model still being
    created) while the endpoint's state still says the earlier UPDATE_FAILED. That is not this
    update's failure: the wait goes on, and the update that then completes is reported as done."""
    serving = _ServingEndpoints({"ep": _endpoint(4, "1", update="UPDATE_FAILED", pending=5, pending_served="2")},
                                stale=1)
    out, _ = _run_deploy_cells(serving)
    assert serving.endpoints["ep"]["entity"][0] == "ep-2" and serving.endpoints["ep"]["update"] == "NOT_UPDATING"
    assert serving.gateways == ["ep"] and "did not complete" not in out


@pytest.mark.parametrize("progress", ["in-progress", "registered-creating", "new-pending-with-stale-failed"])
def test_serving_long_update_that_keeps_showing_progress_is_never_stopped_early(progress):
    """Regression guard: an update that keeps showing progress for far longer than UNCLEAR_LIMIT_S
    (25 polls, 12.5 minutes) is waited for and then completes at the target, with no early stop:
    the endpoint IN_PROGRESS; this update's pending config with its served model being created while
    the state says NOT_UPDATING; and that pending config next to the earlier UPDATE_FAILED."""
    polls = 25
    if progress == "new-pending-with-stale-failed":
        serving = _ServingEndpoints({"ep": _endpoint(4, "1", update="UPDATE_FAILED", pending=5, pending_served="2")},
                                    polls=0, stale=polls)
    else:
        serving = _ServingEndpoints({"ep": _endpoint(4, "1")},
                                    **({"polls": polls} if progress == "in-progress" else {"polls": 0, "idle": polls}))
    out, clock = _run_deploy_cells(serving)
    assert clock.now >= polls * 30 > 2 * serving.namespace["UNCLEAR_LIMIT_S"], clock.now
    assert serving.endpoints["ep"]["entity"][0] == "ep-2" and serving.endpoints["ep"]["update"] == "NOT_UPDATING"
    assert serving.endpoints["ep"]["pending"] is None and serving.gateways == ["ep"] and len(serving.updates) == 1
    assert "could not tell" not in out and "did not complete" not in out


@pytest.mark.parametrize("state", ["DEPLOYMENT_RECOVERING", "DEPLOYMENT_READY"])
@pytest.mark.parametrize("phase", ["registered-not-updating", "next-to-the-earlier-failed-state"])
def test_serving_recovering_or_ready_served_model_of_this_update_is_progress(state, phase):
    """This update's pending config whose served model is recovering, or already ready (traffic not
    switched yet), is progress like one being created (ServedModelStateDeployment in databricks-sdk
    0.122.0, serving.py): with the state still NOT_UPDATING, or next to the earlier UPDATE_FAILED,
    for far longer than UNCLEAR_LIMIT_S, the wait goes on and the update completes."""
    polls = 25
    if phase == "registered-not-updating":
        serving = _ServingEndpoints({"ep": _endpoint(4, "1")}, polls=0, idle=polls, progress_state=state)
    else:
        serving = _ServingEndpoints({"ep": _endpoint(4, "1", update="UPDATE_FAILED", pending=5, pending_served="2")},
                                    polls=0, stale=polls, progress_state=state)
    out, clock = _run_deploy_cells(serving)
    assert clock.now >= polls * 30 > 2 * serving.namespace["UNCLEAR_LIMIT_S"], clock.now
    assert serving.endpoints["ep"]["entity"][0] == "ep-2" and serving.gateways == ["ep"]
    assert "could not tell" not in out and "did not complete" not in out


RIVAL = ("ep-3", config.uc_name(config.ARR_MODEL_NAME, "cat", "sch"), "3")


@pytest.mark.parametrize("rival", ["in-progress", "failed", "completed"])
def test_serving_another_writers_update_is_reported_as_such_not_as_this_ones(rival):
    """Another writer's update (a pending config for a different model version than this update
    sent) is not this update's: in progress, or failed, the task stops at once saying another update
    is in progress on the endpoint, not that this update failed; completed (the endpoint now serves
    the other version), the task stops saying another update replaced this one."""
    outcome = {"in-progress": {"polls": 25}, "failed": {"polls": 0, "outcome": "fail"}, "completed": {"polls": 0}}[rival]
    serving = _ServingEndpoints({"ep": _endpoint(4, "1")}, rival=RIVAL, **outcome)
    clock = _Clock()
    with pytest.raises(RuntimeError) as raised:
        _run_deploy_cells(serving, clock=clock)
    message = str(raised.value)
    assert "another update" in message and "ep-3" in message and "cat.sch.bio_ecg12_arr v2" in message, message
    if rival != "completed":
        assert "is in progress" in message and "did not complete" not in message, message
    assert clock.now == 0 and len(serving.updates) == 1 and serving.gateways == [], clock.now


@pytest.mark.parametrize("field", ["entity_version", "entity_name", "name"])
def test_serving_pending_config_with_an_incomplete_identity_is_not_another_writers(field):
    """The SDK models a served entity's name, entity_name and entity_version as optional. A new pending
    config whose served entity leaves one out is inconclusive, not another writer's: while the
    endpoint shows the update IN_PROGRESS the wait goes on, and the update completes (before, a
    missing entity_version compared as "None" and stopped the task as another writer's update)."""
    serving = _ServingEndpoints({"ep": _endpoint(4, "1")}, polls=3, missing=(field,))
    out, _ = _run_deploy_cells(serving)
    assert serving.endpoints["ep"]["entity"][0] == "ep-2" and serving.gateways == ["ep"]
    assert "another update" not in out and "could not tell" not in out


@pytest.mark.parametrize("field", ["entity_version", "entity_name", "name"])
def test_serving_incomplete_identity_without_progress_stops_as_unclear(field):
    """The same pending config with the endpoint NOT_UPDATING says nothing about whose update it is:
    unclear, so after UNCLEAR_LIMIT_S the task stops saying it could not tell, never that another
    update is in progress."""
    serving = _ServingEndpoints({"ep": _endpoint(4, "1")}, polls=0, idle=25, missing=(field,))
    clock = _Clock()
    with pytest.raises(RuntimeError, match=r"^ep: could not tell whether the config update"):
        _run_deploy_cells(serving, clock=clock)
    limit = serving.namespace["UNCLEAR_LIMIT_S"]
    assert limit <= clock.now <= limit + 30, clock.now


@pytest.mark.parametrize("field", ["entity_version", "entity_name", "name"])
def test_serving_completed_config_with_an_incomplete_identity_is_waited_on_until_it_shows_the_target(field):
    """The update completed (NOT_UPDATING, no pending config, a newer current config), but the first
    poll's current config leaves out a served-entity field: that is not yet this update's success nor
    a failure, so the wait goes on, and the next poll, which shows the target, completes it (before:
    the task stopped with "did not complete")."""
    serving = _ServingEndpoints({"ep": _endpoint(4, "1")}, polls=0, current_missing=((field,), 1))
    out, clock = _run_deploy_cells(serving)
    assert serving.endpoints["ep"]["entity"][0] == "ep-2" and serving.gateways == ["ep"]
    assert "did not complete" not in out and "another update" not in out and clock.now == 30, clock.now


def test_serving_completed_config_whose_identity_never_resolves_stops_as_unclear():
    """A newer current config that keeps leaving out entity_version never counts as done or failed:
    after UNCLEAR_LIMIT_S the task stops saying it could not tell, never "did not complete"."""
    serving = _ServingEndpoints({"ep": _endpoint(4, "1")}, polls=0, current_missing=(("entity_version",), 25))
    clock = _Clock()
    with pytest.raises(RuntimeError) as raised:
        _run_deploy_cells(serving, clock=clock)
    message = str(raised.value)
    assert message.startswith("ep: could not tell whether the config update") and "did not complete" not in message
    limit = serving.namespace["UNCLEAR_LIMIT_S"]
    assert limit <= clock.now <= limit + 30 and serving.gateways == [], clock.now


# Every field of the Model Serving API the update wait reads, by where it reads it: the pending config
# left before the update (its signature, read by deploy() and then seen again in later polls), the
# update's new pending config (signature, served-entity identity, deployment state) and the newer
# current config once the update completed (everything the at-target check compares).
SIGNATURE = ["config_version", "start_time", "name", "entity_name", "entity_version"]
WAIT_FIELDS = {
    "baseline-at-read": SIGNATURE, "baseline-later": SIGNATURE, "baseline-current-at-read": ["config_version"],
    "new-pending": [*SIGNATURE, "state"],
    "current": ["config_version", "name", "entity_name", "entity_version", "workload_size", "scale_to_zero_enabled",
                "traffic_config", "route.served_model_name", "route.traffic_percentage"],
}


def _missing_field_case(where, field, lasting):
    """The stand-in for one field left out at one place: briefly (a later poll is whole) or lasting."""
    if where == "current":
        return _ServingEndpoints({"ep": _endpoint(4, "1")}, polls=0, current_missing=((field,), 25 if lasting else 1))
    if where == "baseline-current-at-read":
        # deploy() reads the current config without its version; the polls after the send first show
        # the unchanged old config, now with its version (the update not registered yet).
        return _ServingEndpoints({"ep": _endpoint(4, "1")}, polls=1, lag=25 if lasting else 2,
                                 baseline_missing=((f"current.{field}",), "read"))
    # An earlier failed update at the same target (served ep-2, aborted) left its pending config, and
    # Model Serving keeps the config version number for the retry: only start_time tells them apart.
    leftover = _endpoint(4, "1", update="UPDATE_FAILED", pending=5, pending_served="2")
    if where == "new-pending":
        options = {"progress_state": None} if field == "state" else {"missing": (field,)}
        return _ServingEndpoints({"ep": leftover}, polls=1, idle=25 if lasting else 2, reuse_version=True, **options)
    when = "read" if where == "baseline-at-read" else "later"
    return _ServingEndpoints({"ep": leftover}, polls=1, reuse_version=True, baseline_missing=((field,), when),
                             **({"idle": 25} if lasting and when == "read" else {"lag": 25 if lasting else 2}))


def test_serving_unknown_baseline_version_and_the_old_config_seen_again_is_never_replaced():
    """deploy() reads the existing endpoint's config without its version (left out by the API): that
    baseline is unknown, not "no endpoint yet". The first poll after the send shows the old config
    with its version (the update not registered yet): not another update's result. The update then
    registers and completes: DONE (before, the unknown baseline became -1, the old config's version
    counted as newer, and the task stopped saying another update replaced this one)."""
    serving = _ServingEndpoints({"ep": _endpoint(4, "1")}, polls=1, lag=1,
                                baseline_missing=(("current.config_version",), "read"))
    out, _ = _run_deploy_cells(serving)
    assert serving.endpoints["ep"]["entity"][0] == "ep-2" and serving.gateways == ["ep"]
    assert "another update" not in out and "did not complete" not in out


@pytest.mark.parametrize("lasting", [False, True], ids=["later-poll-whole", "never-whole"])
@pytest.mark.parametrize("where, field", [(where, field) for where, fields in WAIT_FIELDS.items() for field in fields])
def test_serving_missing_api_fields_never_cause_a_false_failure(where, field, lasting):
    """Missing data alone never makes the wait report a failure, another writer's update or "did not
    complete": for each field the wait reads, left out where it reads it, the update completes once
    a later poll is whole, and when none is, the task stops saying it could not tell. The one field
    the ownership guard also reads before the update, the model name (entity_name) of the pending
    config deploy() reads, stops the task before any update is sent, saying it could not tell which
    model the endpoint serves (fail safe)."""
    serving = _missing_field_case(where, field, lasting)
    clock = _Clock()
    try:
        out, _ = _run_deploy_cells(serving, clock=clock)
    except RuntimeError as error:
        if (where, field) == ("baseline-at-read", "entity_name"):
            assert "could not tell which model" in str(error) and serving.updates == [], str(error)
            return
        assert lasting and str(error).startswith("ep: could not tell whether the config update"), str(error)
        assert serving.gateways == []
        return
    assert (where, field) != ("baseline-at-read", "entity_name"), "the ownership guard let an unknown model through"
    assert serving.endpoints["ep"]["entity"][0] == "ep-2" and serving.gateways == ["ep"], out
    assert "did not complete" not in out and "another update" not in out


@pytest.mark.parametrize("missing", ["endpoint-state", "served-model-state"])
def test_serving_missing_state_is_unclear_not_progress(missing):
    """A poll with no endpoint state, or this update's pending config with no state for its served
    model under NOT_UPDATING, says nothing about the update: it neither counts as progress (waiting
    out the timeout) nor as a result; after UNCLEAR_LIMIT_S the task stops saying it could not tell."""
    polls = 25
    options = {"no_state": polls} if missing == "endpoint-state" else {"idle": polls, "progress_state": None}
    serving = _ServingEndpoints({"ep": _endpoint(4, "1")}, polls=0, **options)
    clock = _Clock()
    with pytest.raises(RuntimeError, match=r"^ep: could not tell whether the config update to cat\.sch\.bio_ecg12_arr v2"):
        _run_deploy_cells(serving, clock=clock)
    limit = serving.namespace["UNCLEAR_LIMIT_S"]
    assert limit <= clock.now <= limit + 30 and serving.gateways == [], clock.now


def test_serving_endpoint_state_missing_for_a_few_polls_then_completes():
    """A few polls without an endpoint state, then the update's progress: the wait goes on and the
    update completes."""
    serving = _ServingEndpoints({"ep": _endpoint(4, "1")}, no_state=2)
    out, _ = _run_deploy_cells(serving)
    assert serving.endpoints["ep"]["entity"][0] == "ep-2" and serving.gateways == ["ep"] and "could not tell" not in out


BOTH = ((config.ARR_MODEL_NAME, "ep"), (config.AFIB_MODEL_NAME, "ep2"))


def test_serving_waits_for_both_endpoints_under_one_deadline_and_reports_an_early_failure_at_once():
    """Both endpoints are polled together under one deadline: the second endpoint's update failing at
    once stops the task at the first round with its message, while the first is still in progress
    (a serial wait would have waited out the first one, 25 polls, before looking at the second)."""
    serving = _ServingEndpoints({}, per={"ep": {"polls": 25},
                                         "ep2": {"polls": 0, "outcome": ("DEPLOYMENT_FAILED", "model load error")}})
    clock = _Clock()
    with pytest.raises(RuntimeError, match=r"^ep2: the config update to .* did not complete .* model load error"):
        _run_deploy_cells(serving, clock=clock, served=BOTH)
    assert clock.now == 0 and serving.gateways == [] and [n for n, _ in serving.creates] == ["ep", "ep2"], clock.now


def test_serving_two_updates_take_as_long_as_the_slower_one():
    """Two updates of 10 polls each finish in about 10 polls, not 20, and each endpoint gets its AI
    Gateway step once it is ready."""
    serving = _ServingEndpoints({}, polls=10)
    out, clock = _run_deploy_cells(serving, served=BOTH)
    assert sorted(serving.gateways) == ["ep", "ep2"] and clock.now <= 12 * 30, clock.now
    assert "did not complete" not in out


@pytest.mark.parametrize("field, value, named", [
    ("workload_size", "Medium", "workload_size Medium (submitted Small)"),
    ("scale_to_zero_enabled", False, "scale_to_zero_enabled False (submitted True)"),
    ("route.traffic_percentage", 50, "traffic ep-2 50% (submitted ep-2 100%)")])
def test_serving_completed_config_with_other_settings_says_it_differs_not_that_the_update_failed(field, value, named):
    """The update completed and the endpoint's current config serves this update's served entity
    (name, model, version), but another setting differs from the config 05_ submitted (changed in
    the Serving UI, say): the task stops saying the current config differs from the submitted one,
    naming what differs, and that it may have been changed in the Serving UI; not that the update did
    not complete and the endpoint keeps serving its previous config, which may be false."""
    serving = _ServingEndpoints({"ep": _endpoint(4, "1")}, polls=0, edited={field: value})
    with pytest.raises(RuntimeError) as raised:
        _run_deploy_cells(serving)
    message = str(raised.value)
    assert message.startswith("ep: the endpoint's current config serves cat.sch.bio_ecg12_arr v2, but differs from "
                              "the config this run submitted: "), message
    assert named in message and "may have been changed in the Serving UI" in message, message
    assert "did not complete" not in message and "previous config" not in message, message
    assert len(serving.updates) == 1 and serving.gateways == []


@pytest.mark.parametrize("lasting", [False, True], ids=["later-poll-whole", "never-whole"])
@pytest.mark.parametrize("missing", ["name", "entity_name", "entity_version"])
@pytest.mark.parametrize("field, value", [("workload_size", "Medium"), ("scale_to_zero_enabled", False),
                                          ("route.traffic_percentage", 50)])
def test_serving_changed_settings_with_an_incomplete_identity_are_unclear_not_a_changed_config(
        field, value, missing, lasting):
    """A completed config with a changed setting whose served entity leaves out its name, model or
    version is not known to serve this update's served entity: inconclusive (UNCLEAR), never the
    definite "may have been changed in the Serving UI" message, nor "did not complete". The wait goes
    on: once a poll shows the identity whole, the changed setting is reported as such; when none
    does, the task stops at UNCLEAR_LIMIT_S saying it could not tell (before, the changed setting
    outweighed the missing identity and the task stopped at the first poll as a changed config)."""
    serving = _ServingEndpoints({"ep": _endpoint(4, "1")}, polls=0, edited={field: value},
                                current_missing=((missing,), 25 if lasting else 1))
    clock = _Clock()
    with pytest.raises(RuntimeError) as raised:
        _run_deploy_cells(serving, clock=clock)
    message = str(raised.value)
    assert "did not complete" not in message and "previous config" not in message, message
    if lasting:
        assert message.startswith("ep: could not tell whether the config update"), message
        assert "may have been changed in the Serving UI" not in message, message
        limit = serving.namespace["UNCLEAR_LIMIT_S"]
        assert limit <= clock.now <= limit + 30, clock.now
    else:
        assert "may have been changed in the Serving UI" in message and clock.now == 30, (message, clock.now)
    assert serving.gateways == []


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
    updated = serving.namespace["deployed"]["ep"]           # deploy() returns a Deployment, by field name
    assert (updated.full_name, updated.version, updated.target) == (config.uc_name(config.ARR_MODEL_NAME, "cat", "sch"),
                                                                    "2", sent)
    serving = _ServingEndpoints({})
    _run_deploy_cells(serving)
    assert [n for n, _ in serving.creates] == ["ep"] and serving.updates == [] and serving.gateways == ["ep"]


def test_target_match_compares_every_field_deploy_sends_and_never_takes_a_missing_one_for_a_value():
    """target_match, the one at-target check: a different size, scale-to-zero, model, version,
    entity name, extra entity or split traffic DIFFERS (so deploy() updates it, and the wait does not
    take it for this update's success); every field left out (None) is MISSING, never a value."""
    serving = _ServingEndpoints({})
    namespace = {"EndpointStateConfigUpdate": _ConfigUpdate}
    (helpers,) = [c for c in _cells(WORKFLOW / "05_serve_endpoints.py") if "def wait_for_endpoint_ready(" in c]
    exec(compile(helpers, "05_serve_endpoints.py", "exec"), namespace)   # noqa: S102 (the repo's own notebook cell)
    target_match, MATCH, MISSING, DIFFERS = (namespace[k] for k in ("target_match", "MATCH", "MISSING", "DIFFERS"))
    target = {"served_entities": [{"name": "ep-2", "entity_name": "c.s.m", "entity_version": "2",
                                   "workload_size": "Small", "scale_to_zero_enabled": True}],
              "traffic_config": {"routes": [{"served_model_name": "ep-2", "traffic_percentage": 100}]}}
    assert target_match(serving._config(3, ("ep-2", "c.s.m", "2")), target) == MATCH
    assert target_match(None, target) == MISSING
    for field, value in (("workload_size", "Medium"), ("scale_to_zero_enabled", False), ("entity_name", "c.s.x"),
                         ("entity_version", "3"), ("name", "ep-3")):
        changed = serving._config(3, ("ep-2", "c.s.m", "2"))
        setattr(changed.served_entities[0], field, value)
        assert target_match(changed, target) == DIFFERS, field
    for field in WAIT_FIELDS["current"][1:]:
        missing = serving._config(3, ("ep-2", "c.s.m", "2"))
        serving._drop(missing, field)
        assert target_match(missing, target) == MISSING, field
    split = serving._config(3, ("ep-2", "c.s.m", "2"))
    split.traffic_config.routes[0].traffic_percentage = 50
    two = serving._config(3, ("ep-2", "c.s.m", "2"))
    two.served_entities.append(two.served_entities[0])
    assert target_match(split, target) == DIFFERS and target_match(two, target) == DIFFERS


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
                 "tag_endpoint(endpoint_name, deployment_path.DEPLOYMENT_TAG, found.path)"):
        assert call in cell, call
    # One read of the events API in the notebook (read_events), used by this report only: the update
    # wait judges from the endpoint's state.
    (helpers,) = [c for c in _cells(WORKFLOW / "05_serve_endpoints.py") if "def wait_for_endpoint_ready(" in c]
    assert code.count("read_events(") == 2 and "events = read_events(endpoint_name)" in cell
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
    (remove_after_cell,) = [c for c in cells if '"removeAfter", remove_after)' in c]
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
                                  "message": dp.IMAGE_STARTED},
                                 {"event_type": dp.CONTAINER_EVENT, "served_entity_name": "b-1", "timestamp": 1,
                                  "message": dp.IMAGE_DONE}]},
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


@pytest.mark.parametrize("error", [PermissionError("PERMISSION_DENIED: cannot tag the endpoint"),
                                   RuntimeError("INVALID_PARAMETER_VALUE: tag value not allowed by the tag policy")],
                         ids=["permissions", "tag-policy"])
def test_serving_tag_failures_only_warn_and_query_errors_still_fail_the_task(error):
    """The `deployment` and removeAfter tags are informational: a tag call that fails (no permission,
    a tag policy refusing the value) prints a warning with the error and the tag to add by hand, and
    the task goes on (the removeAfter tag is still tried after a failed `deployment` tag). The smoke
    queries have no error handling of their own, so a failing query still fails the task."""
    stand_in = _ServingStandIn(logs={"a-1": EXPRESS_LOG}, events={})
    attempts = []

    def patch(name, add_tags):
        attempts.append([t.key for t in add_tags])
        raise error

    stand_in.serving_endpoints.patch = patch
    expiry = "2099-12-31"           # a stand-in date (test_repo_consistency.py: no real date committed)
    out = _run_tag_cells(stand_in, ("a",), remove_after=expiry)
    assert attempts == [["deployment"], ["removeAfter"]]
    warnings = [line for line in out.splitlines() if line.startswith("WARNING: could not tag")]
    assert len(warnings) == 2 and all(str(error) in w for w in warnings), out
    assert "deployment=express" in warnings[0] and f"removeAfter={expiry}" in warnings[1]
    assert "tagged" not in out.replace("could not tag", "")
    (smoke,) = [c for c in _cells(WORKFLOW / "05_serve_endpoints.py") if "deploy_client.predict(" in c]
    assert not {"try", "except"} & set(_code_tokens_of(smoke))


@pytest.mark.parametrize("entities", [None, []], ids=["left-out", "empty"])
def test_serving_deployment_path_of_an_endpoint_without_its_served_entity_is_unknown(entities):
    """The deployment-path cell reads the endpoint's served entity; when the API leaves it out the
    path is unknown with a warning, and the cell never raises (before: a ValueError or TypeError
    unpacking it, after both endpoints were serving)."""
    stand_in = _ServingStandIn(logs={}, events={})
    stand_in.serving_endpoints.get = lambda name: SimpleNamespace(config=SimpleNamespace(served_entities=entities))
    out = _run_tag_cells(stand_in, ("a",))
    assert stand_in.tags == {"a": {"deployment": "unknown"}}
    assert "WARNING: the deployment path of a could not be determined (the endpoint's config names no single served" in out


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
    """Every %pip install names exact versions (==) and never upgrades to whatever is latest. The
    only other index is PyTorch's CPU-only one, on a line of its own that installs torch alone, so
    every other package still comes from the default index."""
    for line in path.read_text().splitlines():
        if not line.startswith("# MAGIC %pip install"):
            continue
        args = line.removeprefix("# MAGIC %pip install").split()
        flags = [a for a in args if a.startswith("-")]
        packages = [a.strip('"\'') for a in args if not a.startswith("-")]
        assert set(flags) <= {"--no-deps", f"--index-url={TORCH_CPU_INDEX}"}, f"{path.name}: {line}"
        assert packages and all("==" in p for p in packages), f"{path.name}: {line}"
        if f"--index-url={TORCH_CPU_INDEX}" in flags:
            assert flags == [f"--index-url={TORCH_CPU_INDEX}"] and [p.split("==")[0] for p in packages] == ["torch"]


def test_cpu_notebooks_install_the_cpu_build_of_torch():
    """07_ / 08_ run on CPU serverless: torch comes from PyTorch's CPU-only index, on its own %pip
    line, and mlflow-skinny from the default index; the training notebooks (GPU) install no torch."""
    for name in ("07_arr_triage.py", "08_afib_classify.py"):
        lines = pip_lines(WORKFLOW / name)
        torch_lines = [line for line in lines if "torch==" in line]
        assert torch_lines == [f'# MAGIC %pip install --index-url={TORCH_CPU_INDEX} "torch==2.11.0"'], name
        assert '# MAGIC %pip install "mlflow-skinny==3.18.0"' in lines, name
    for path in NOTEBOOKS:
        if path.name not in ("07_arr_triage.py", "08_afib_classify.py"):
            assert not [line for line in pip_lines(path) if "torch" in line], path.name


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
    # Locked literally: numpy, pandas, torch, cloudpickle and mlflow must equal the versions the models
    # are trained (and cloudpickled and logged) with: AI Runtime environment 6's numpy, pandas, torch and
    # cloudpickle (conftest.PREINSTALLED for torch and cloudpickle) and the mlflow-skinny the training
    # notebooks' %pip line installs over it. A bump goes here, in config.py and, for mlflow and torch, in
    # PIP_LINES and the notebooks' %pip lines (test_repo_consistency.py checks each environment).
    assert reqs == ["mlflow==3.18.0", "cloudpickle==3.1.1", "numpy==2.3.4", "pandas==2.3.3", "torch==2.11.0"]


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
    import torch
    weights_only, real_load = [], torch.load
    def load(*args, **kwargs):
        weights_only.append(kwargs.get("weights_only"))
        return real_load(*args, **kwargs)
    torch.load = load
    model = pickle.loads(open(sys.argv[1], "rb").read())
    model.load_context(SimpleNamespace(artifacts={"state_dict": sys.argv[2]}))
    print("torch", torch.__version__, "weights_only", weights_only)
    x = np.random.default_rng(0).normal(size=(3, 12, 2500)).astype("float32")
    out = model.predict(None, x)
    flat = model.predict(None, pd.DataFrame({"ecg": list(x.reshape(3, -1))}))
    if isinstance(out, pd.DataFrame):
        print("frame", list(out.columns), out.shape, sorted(set(out["arr_flag"].tolist()) | {0.0, 1.0}))
    else:
        print("array", np.asarray(out).shape, bool(np.allclose(np.asarray(out).sum(axis=1), 1.0)))
    print("flat_equal", bool(np.allclose(np.asarray(out), np.asarray(flat))))
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    params = {p.device.type for p in model._model.parameters()}
    print("device_ok", model._device.type == expected and params == {expected})
    arrays = [out[c].to_numpy() for c in out.columns] if isinstance(out, pd.DataFrame) else [out]
    print("numpy_out", sorted({f"{type(a).__name__} {a.dtype}" for a in arrays}))
    try:
        model.load_context(SimpleNamespace(artifacts={"state_dict": sys.argv[3]}))
        print("refuses_objects False")
    except Exception as e:      # noqa: BLE001 (any other error is a failure the test reports)
        print("refuses_objects", isinstance(e, pickle.UnpicklingError) and str(e).startswith("Weights only load failed"))
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
        not_weights = tmp_path / "not_weights.pt"               # an object, not tensors: refused
        torch.save({"weight": fractions.Fraction(1, 2)}, not_weights)
        pickled = tmp_path / "model.pkl"
        python_model = getattr(nb, pyfunc_name)()
        assert not hasattr(python_model, "_device")       # the device is picked at load time, never pickled
        pickled.write_bytes(cloudpickle.dumps(python_model))
    finally:
        sys.modules.pop(nb.__name__, None)

    # Fresh interpreter, cwd and sys.path without projects/ecg12-afib: `common` cannot be imported.
    result = subprocess.run([sys.executable, "-I", "-c", _LOAD_AND_PREDICT, str(pickled), str(state_dict),
                             str(not_weights)],
                            capture_output=True, text=True, cwd=tmp_path, timeout=300, check=False)
    assert result.returncode == 0, result.stderr
    # The weights load with weights_only=True (this torch's, the models' with run_tests_python311.sh),
    # which refuses a file holding anything but tensors.
    assert f"torch {torch.__version__} weights_only [True]" in result.stdout.splitlines()
    assert "refuses_objects True" in result.stdout
    if pyfunc_name == "ArrModelPyfunc":
        assert "frame ['arr_prob', 'arr_flag'] (3, 2) [0.0, 1.0]" in result.stdout
    else:
        assert "array (3, 2) True" in result.stdout
    assert "flat_equal True" in result.stdout
    assert "device_ok True" in result.stdout                      # model sits on the device it picked
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert f"{pyfunc_name} loaded on {expected}" in result.stdout.splitlines()   # load_context says so
    assert "numpy_out ['ndarray float32']" in result.stdout        # outputs come back as CPU numpy
