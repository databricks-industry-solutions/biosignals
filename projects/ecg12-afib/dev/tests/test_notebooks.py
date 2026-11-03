"""Static checks of the workflow notebooks (they are NOT executed on Databricks here; the SETUP and
DAILY flows run locally with stand-ins in test_setup_flow_local.py / test_daily_flow_local.py),
plus a serving check of the training notebooks' inline model code: it is pickled the way MLflow
pickles it and loaded in a fresh process where the `common` package does not exist (as on a
serving endpoint)."""
import ast
import io
import subprocess
import sys
import textwrap
import tokenize
import types

import numpy as np
import pytest

from common import config
from conftest import DEMO_ROOT, GOVERNED_TAG_VALUES

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
    assert "generate_ecg_batch" not in code and "# MAGIC %pip" not in (WORKFLOW / name).read_text()
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


@pytest.mark.parametrize("name", BATCH)
def test_batch_scoring_is_driver_side_at_the_pinned_champion(name):
    """07_ / 08_ resolve champion to a version, load exactly that version and score on the driver
    with score_driver_side, as the gate does."""
    code = _python_source(WORKFLOW / name)
    kind = "arr" if name.startswith("07") else "afib"
    assert f"{kind} = scoring.resolve_alias(MlflowClient(registry_uri=\"databricks-uc\"), {kind}_model_full)" in code
    assert f"{kind}_model = mlflow.pyfunc.load_model({kind}.uri)" in code
    assert code.count("scoring.score_driver_side(") == 1 and f"model_info={kind}.info" in code
    assert "# MAGIC %pip" not in (WORKFLOW / name).read_text()       # the ML v5 environment has torch


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
    assert code.index("if served != {full}:") < code.index("full, ver = deploy(model_name, endpoint_name, existing)")
    assert "table_name_prefix=model_name" in code and "table_name_prefix=endpoint_name" not in code
    assert "config.payload_table(model_name, catalog, schema)" in code


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
        exec(compile(_model_cell(WORKFLOW / name), name, "exec"), nb.__dict__)
        cloudpickle.register_pickle_by_value(nb)
        state_dict = tmp_path / "state_dict.pt"
        torch.save(getattr(nb, cnn_name)().state_dict(), state_dict)
        pickled = tmp_path / "model.pkl"
        pickled.write_bytes(cloudpickle.dumps(getattr(nb, pyfunc_name)()))
    finally:
        sys.modules.pop(nb.__name__, None)

    # Fresh interpreter, cwd and sys.path without projects/ecg12-afib: `common` cannot be imported.
    result = subprocess.run([sys.executable, "-I", "-c", _LOAD_AND_PREDICT, str(pickled), str(state_dict)],
                            capture_output=True, text=True, cwd=tmp_path, timeout=300)
    assert result.returncode == 0, result.stderr
    if pyfunc_name == "ArrModelPyfunc":
        assert "frame ['arr_prob', 'arr_flag'] (3, 2) [0.0, 1.0]" in result.stdout
    else:
        assert "array (3, 2) True" in result.stdout
    assert "flat_equal True" in result.stdout
