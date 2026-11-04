"""Runs the SETUP notebooks 01 -> 02 + 03 -> 04 on a local Spark session, with real (tiny) PyTorch
training, and checks the data-first flow and the challenger -> validation gate -> champion cycle.

Stand-ins: notebook globals (dbutils, display, spark), Unity Catalog tables with Delta versions (an
in-memory list of versions per table, behind the same DESCRIBE HISTORY / VERSION AS OF SQL that
common/uc_io.py sends), and MLflow (a fake tracking store + model registry; a logged pyfunc model is
cloudpickled and loaded back, so 04_ scores the very models 02_ / 03_ trained; like Unity Catalog
under a tag policy, it refuses a tag value conftest.GOVERNED_TAG_VALUES does not allow).
05_serve_endpoints is not run (it creates serving endpoints). The splits and epochs are shrunk to
keep the run short.
Skipped without a local pyspark + Java, torch or cloudpickle."""
import json
import re
import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from common import config, uc_io
from conftest import DEMO_ROOT, GOVERNED_TAG_VALUES, TEST_CATALOG

SMALL_SPLITS = {config.TRAIN_SPLIT: (24, 11), config.VAL_SPLIT: (12, 99), config.HOLDOUT_SPLIT: (12, 314)}
LENIENT = {"min_accuracy": 0.0, "min_roc_auc": 0.0}


class VersionedUC:
    """Spark session whose tables keep every version, read back through uc_io's SQL."""

    def __init__(self, session):
        self._session = session
        self.versions = {}
        self.reads = []
        self.schemas = []
        self.catalog = SimpleNamespace(tableExists=lambda name: name in self.versions)

    def table(self, name):
        return self.versions[name][-1]

    def sql(self, statement):
        if statement.startswith("OPTIMIZE "):
            return None
        if m := re.fullmatch(r"CREATE SCHEMA IF NOT EXISTS `([^`]+)`\.`([^`]+)`", statement):
            self.schemas.append((m[1], m[2]))
            return None
        if m := re.fullmatch(r"DESCRIBE HISTORY (\S+) LIMIT 1", statement):
            return SimpleNamespace(collect=lambda: [{"version": len(self.versions[m[1]]) - 1}])
        if m := re.fullmatch(r"SELECT \* FROM (\S+) VERSION AS OF (\d+)", statement):
            self.reads.append((m[1], int(m[2])))
            return self.versions[m[1]][int(m[2])]
        raise AssertionError(f"unexpected SQL: {statement}")

    def __getattr__(self, name):
        return getattr(self._session, name)


def _check_tag_policy(key, value):
    """Refuse a governed tag value its policy does not allow, as Unity Catalog does."""
    allowed = GOVERNED_TAG_VALUES.get(key)
    if allowed is not None and str(value) not in allowed:
        raise AssertionError(f"tag value {value!r} is not allowed by the tag policy for {key!r}: {sorted(allowed)}")


class FakeMlflow:
    """Tracking runs (params, metrics, tags) + a UC-like registry (versions, aliases, version tags).
    Run and model-version tags are checked against the workspace's tag policies
    (``GOVERNED_TAG_VALUES``)."""

    def __init__(self, cloudpickle):
        self._cloudpickle = cloudpickle
        self.runs, self.registry = {}, {}
        self._active = []

    @contextmanager
    def start_run(self, run_name=None, run_id=None):
        run_id = run_id or f"run{len(self.runs)}"
        self.runs.setdefault(run_id, {"name": run_name, "params": {}, "metrics": [], "tags": {}, "artifacts": {}})
        self._active.append(run_id)
        try:
            yield SimpleNamespace(info=SimpleNamespace(run_id=run_id))
        finally:
            self._active.pop()

    def _run(self):
        assert self._active, "no active MLflow run"
        return self.runs[self._active[-1]]

    def log_params(self, params):
        run = self._run()
        for key, value in params.items():                   # MLflow refuses to change a logged param
            assert run["params"].get(key, str(value)) == str(value), key
            run["params"][key] = str(value)

    def log_metrics(self, metrics, step=None):
        self._run()["metrics"] += [(k, float(v), step) for k, v in metrics.items()]

    def set_tags(self, tags):
        for key, value in tags.items():
            _check_tag_policy(key, value)
        self._run()["tags"].update({k: str(v) for k, v in tags.items()})

    def set_tag(self, key, value):
        self.set_tags({key: value})

    def _artifact(self, path, value):
        artifacts = self._run()["artifacts"]
        assert path not in artifacts, f"artifact {path} logged twice"
        artifacts[path] = value

    def log_text(self, text, artifact_file):
        self._artifact(artifact_file, str(text))

    def log_dict(self, dictionary, artifact_file):
        self._artifact(artifact_file, json.loads(json.dumps(dictionary)))      # must be JSON-serializable

    def log_table(self, data, artifact_file):
        self._artifact(artifact_file, data.copy())

    def log_figure(self, figure, artifact_file):
        from matplotlib.figure import Figure

        assert isinstance(figure, Figure) and artifact_file.endswith(".png"), artifact_file
        self._artifact(artifact_file, [t.get_text() for t in figure.texts] + [ax.get_title(loc="left") for ax in figure.axes])

    def log_model(self, name, python_model, artifacts, signature, input_example, registered_model_name):
        assert signature is not None and input_example.shape[0] == 1
        model = self.registry.setdefault(registered_model_name, {"versions": {}, "aliases": {}})
        version = str(len(model["versions"]) + 1)
        model["versions"][version] = {"pickle": self._cloudpickle.dumps(python_model),
                                      "artifacts": dict(artifacts), "tags": {}}
        return SimpleNamespace(registered_model_version=version)

    def load_model(self, uri):
        name, version = re.fullmatch(r"models:/(.+)/(\d+)", uri).groups()   # always a pinned version
        entry = self.registry[name]["versions"][version]
        model = self._cloudpickle.loads(entry["pickle"])
        model.load_context(SimpleNamespace(artifacts=entry["artifacts"]))
        return SimpleNamespace(predict=lambda X: model.predict(None, X))

    def client(self, registry_uri=None):
        registry = self.registry

        def version_of(name, alias):
            return SimpleNamespace(version=registry[name]["aliases"][alias])

        def set_model_version_tag(name, version, key, value):
            _check_tag_policy(key, value)
            registry[name]["versions"][str(version)]["tags"][key] = str(value)

        return SimpleNamespace(
            set_registered_model_alias=lambda name, alias, version: registry[name]["aliases"].__setitem__(alias, str(version)),
            delete_registered_model_alias=lambda name, alias: registry[name]["aliases"].pop(alias),
            get_model_version_by_alias=version_of,
            set_model_version_tag=set_model_version_tag,
            get_model_version=lambda name, version: SimpleNamespace(tags=dict(registry[name]["versions"][str(version)]["tags"])),
            get_registered_model=lambda name: SimpleNamespace(aliases=dict(registry[name]["aliases"])),
        )

    def install(self, monkeypatch):
        mlflow = types.ModuleType("mlflow")
        pyfunc = types.ModuleType("mlflow.pyfunc")
        pyfunc.PythonModel = type("PythonModel", (), {})
        pyfunc.log_model, pyfunc.load_model = self.log_model, self.load_model
        models = types.ModuleType("mlflow.models")
        models.infer_signature = lambda model_input, model_output: ("signature", model_input.shape)
        mlflow.pyfunc, mlflow.models = pyfunc, models
        mlflow.set_registry_uri = lambda uri: None
        mlflow.MlflowClient = self.client
        for name in ("start_run", "log_params", "log_metrics", "set_tags", "set_tag", "log_text", "log_dict",
                     "log_table", "log_figure"):
            setattr(mlflow, name, getattr(self, name))
        for name, module in {"mlflow": mlflow, "mlflow.pyfunc": pyfunc, "mlflow.models": models}.items():
            monkeypatch.setitem(sys.modules, name, module)


def _overwrite_versioned(spark, df, table, optimize=True):
    spark.versions.setdefault(table, []).append(spark.createDataFrame(df.toPandas(), df.schema))


class NotebookExit(Exception):
    """dbutils.notebook.exit(value): ends the notebook run (successfully) with ``value``."""


def _exit(value):
    raise NotebookExit(value)


def _run_notebook(name, uc):
    """Runs the notebook; returns its dbutils.notebook.exit value, or None when it runs to the end."""
    import matplotlib.pyplot as plt

    values = {"catalog_name": TEST_CATALOG}                 # the job parameter; the widget has no default
    dbutils = SimpleNamespace(
        widgets=SimpleNamespace(text=lambda key, default, label=None: values.setdefault(key, default),
                                get=lambda key: values[key]),
        library=SimpleNamespace(restartPython=lambda: None),
        notebook=SimpleNamespace(exit=_exit))
    source = "\n".join("" if line.startswith("# MAGIC") else line
                       for line in (DEMO_ROOT / "workflow" / name).read_text().splitlines())
    try:
        exec(compile(source, name, "exec"),
             {"__name__": "__main__", "spark": uc, "dbutils": dbutils, "display": lambda df: df.collect()})
    except NotebookExit as stop:
        return stop.args[0]
    finally:
        plt.close("all")


def test_setup_notebooks_gate_promotes_both_or_neither(spark, monkeypatch):
    pytest.importorskip("torch")
    fake = FakeMlflow(pytest.importorskip("cloudpickle"))
    monkeypatch.chdir(DEMO_ROOT / "workflow")
    monkeypatch.setattr(uc_io, "overwrite_table", _overwrite_versioned)
    monkeypatch.setattr(config, "TRAINING_SPLITS", SMALL_SPLITS)
    monkeypatch.setattr(config, "TRAIN_PARAMS", {**config.TRAIN_PARAMS, "epochs": 2, "batch_size": 8})
    monkeypatch.setattr(config, "VALIDATION_MIN_ROWS", SMALL_SPLITS[config.HOLDOUT_SPLIT][0])
    monkeypatch.setattr(config, "VALIDATION_THRESHOLDS", {config.ARR_MODEL_NAME: LENIENT, config.AFIB_MODEL_NAME: LENIENT})
    fake.install(monkeypatch)
    uc = VersionedUC(spark)
    table = config.uc_name(config.TRAINING_TABLE, TEST_CATALOG)
    arr = config.uc_name(config.ARR_MODEL_NAME, TEST_CATALOG)
    afib = config.uc_name(config.AFIB_MODEL_NAME, TEST_CATALOG)

    # 01: the schema (created if missing), then the training dataset, persisted with its split
    # column (Delta version 0).
    _run_notebook("01_generate_training_data.py", uc)
    assert uc.schemas == [(TEST_CATALOG, config.SCHEMA)]
    data = uc.versions[table][0].toPandas()
    assert data.groupby(config.SPLIT_COL).size().to_dict() == {s: n for s, (n, _) in SMALL_SPLITS.items()}
    assert data[config.SAMPLE_ID_COL].is_unique and len(data["ecg12"].iloc[0]) == 2500

    # 02 + 03: train on train/val at version 0, register challengers, never champion. No
    # validation_status yet: the challenger alias marks them as awaiting the gate.
    _run_notebook("02_train_arr_model.py", uc)
    _run_notebook("03_train_afib_model.py", uc)
    for name in (arr, afib):
        model = fake.registry[name]
        assert model["aliases"] == {config.CHALLENGER_ALIAS: "1"}
        tags = model["versions"]["1"]["tags"]
        assert {"training_table": table, "training_table_version": "0"}.items() <= tags.items()
        assert "validation_status" not in tags
    assert set(uc.reads) == {(table, 0)}                             # read at the recorded version
    runs = {run["name"]: run for run in fake.runs.values()}
    for run_name in (f"{config.ARR_MODEL_NAME}_train", f"{config.AFIB_MODEL_NAME}_train"):
        run = runs[run_name]
        assert {"learning_rate", "epochs", "batch_size", "seed", "conv_channels", "n_leads",
                "training_table", "training_table_version", "n_train", "n_val"} <= set(run["params"])
        assert run["params"]["conv_channels"] == "16,32,64" and run["params"]["training_table_version"] == "0"
        per_epoch = {k: [step for key, _, step in run["metrics"] if key == k] for k in ("train_loss", "val_loss", "val_accuracy")}
        assert per_epoch == {k: [0, 1] for k in per_epoch}
        final = {key for key, _, step in run["metrics"] if step is None}            # no collision with per-epoch keys
        assert {"final_val_accuracy", "final_val_roc_auc", "final_val_sensitivity", "final_val_specificity",
                "final_val_loss", "final_train_loss"} <= final and not final & set(per_epoch)
        assert run["tags"]["challenger_version"] == "1" and run["tags"]["training_table_version"] == "0"
        # The architecture: layer table + total parameters; the first conv layer reads the 12 leads.
        arch = run["artifacts"]["model_architecture.md"]
        assert "`Conv1d(12, 16, kernel_size=(7,)" in arch and "| (1, 12, 2500) |" in arch
        assert run["params"]["n_parameters"] == "10514" and "10,514 trainable parameters" in arch
    assert runs[f"{config.AFIB_MODEL_NAME}_train"]["params"]["label_smoothing"] == str(config.AFIB_LABEL_SMOOTHING)

    # 04, passing thresholds: both challengers become champion, then challenger is cleared (only
    # champion remains); outcome recorded.
    uc.reads.clear()
    _run_notebook("04_validate_and_promote.py", uc)
    assert uc.reads == [(table, 0), (table, 0)]                      # hold-out at the trained-beside version
    for name in (arr, afib):
        assert fake.registry[name]["aliases"] == {config.CHAMPION_ALIAS: "1"}
        tags = fake.registry[name]["versions"]["1"]["tags"]
        assert (tags["validation_status"], tags["gate_result"]) == ("valid", "passed") and "holdout_roc_auc" in tags
    gate = runs_by_name(fake)["validate_and_promote"]
    assert gate["tags"]["validation_status"] == "valid"
    assert {f"{config.ARR_MODEL_NAME}.holdout_accuracy", f"{config.AFIB_MODEL_NAME}.holdout_n",
            f"{config.AFIB_MODEL_NAME}.holdout_average_precision"} <= {k for k, _, _ in gate["metrics"]}
    # Evaluation artifacts per model, from the same hold-out scores, each with the synthetic-data caption.
    n_holdout = SMALL_SPLITS[config.HOLDOUT_SPLIT][0]
    for model in (config.ARR_MODEL_NAME, config.AFIB_MODEL_NAME):
        folder = f"holdout_eval/{model}"
        assert {p for p in gate["artifacts"] if p.startswith(folder)} == {
            f"{folder}/{name}" for name in ("confusion_matrix.png", "confusion_matrix.json", "roc_curve.png",
                                            "pr_curve.png", "per_class_metrics.json", "README.md")}
        confusion = gate["artifacts"][f"{folder}/confusion_matrix.json"]
        assert sum(map(sum, confusion["matrix"])) == confusion["n"] == n_holdout
        assert sum(sum(c.values()) for c in confusion["by_true_rhythm"].values()) == n_holdout
        per_class = gate["artifacts"][f"{folder}/per_class_metrics.json"]
        assert list(per_class.columns) == ["class", "precision", "recall", "f1", "support"] and len(per_class) == 4
        assert per_class["support"].iloc[:2].sum() == n_holdout
        for figure in ("confusion_matrix.png", "roc_curve.png", "pr_curve.png"):
            assert any("separable by design" in text for text in gate["artifacts"][f"{folder}/{figure}"]), figure
        assert "separable by design" in gate["artifacts"][f"{folder}/README.md"] and "separable by design" in confusion["note"]
        assert fake.registry[{config.ARR_MODEL_NAME: arr, config.AFIB_MODEL_NAME: afib}[model]]["versions"]["1"]["tags"][
            "holdout_eval_artifacts"] == f"runs:/{runs_id_by_name(fake)['validate_and_promote']}/{folder}"

    # Re-running 04 alone now finds no challenger: it exits cleanly, reading and changing nothing.
    uc.reads.clear()
    n_runs = len(fake.runs)
    note = _run_notebook("04_validate_and_promote.py", uc)
    assert note and "Nothing to validate" in note and uc.reads == [] and len(fake.runs) == n_runs
    assert all(fake.registry[name]["aliases"] == {config.CHAMPION_ALIAS: "1"} for name in (arr, afib))

    # Retrain the triage model on a regenerated table (Delta version 1), then gate with a classify
    # threshold no model can meet: afib has no challenger, so it is re-checked at its champion beside
    # arr's challenger; the gate fails, nothing is promoted (not even the passing arr v2).
    _run_notebook("01_generate_training_data.py", uc)
    _run_notebook("02_train_arr_model.py", uc)
    assert fake.registry[arr]["aliases"] == {config.CHALLENGER_ALIAS: "2", config.CHAMPION_ALIAS: "1"}
    assert "validation_status" not in fake.registry[arr]["versions"]["2"]["tags"]
    monkeypatch.setattr(config, "VALIDATION_THRESHOLDS",
                        {config.ARR_MODEL_NAME: LENIENT, config.AFIB_MODEL_NAME: {"max_called_positive_rate_other": -1.0}})
    uc.reads.clear()
    with pytest.raises(RuntimeError, match="Validation gate FAILED, nothing promoted"):
        _run_notebook("04_validate_and_promote.py", uc)
    assert sorted(uc.reads) == [(table, 0), (table, 1)]              # each candidate's own data version
    # Nothing moved: arr keeps its failed challenger (for inspection) and both champions stay at v1.
    assert fake.registry[arr]["aliases"] == {config.CHALLENGER_ALIAS: "2", config.CHAMPION_ALIAS: "1"}
    assert fake.registry[afib]["aliases"] == {config.CHAMPION_ALIAS: "1"}
    # arr v2 met its own thresholds (valid) but the gate as a whole failed; afib v1 did not (invalid).
    assert {"validation_status": "valid", "gate_result": "failed"}.items() <= fake.registry[arr]["versions"]["2"]["tags"].items()
    assert {"validation_status": "invalid", "gate_result": "failed"}.items() <= fake.registry[afib]["versions"]["1"]["tags"].items()
    assert runs_by_name(fake)["validate_and_promote"]["tags"]["validation_status"] == "invalid"


def test_fake_registry_refuses_tag_values_the_tag_policies_do_not_allow():
    """The stand-in refuses what Unity Catalog refuses, so the SETUP flow test above fails if any
    notebook writes a governed tag (GOVERNED_TAG_VALUES) with a disallowed value, as a run tag or a
    model-version tag, in either gate outcome."""
    fake = FakeMlflow(cloudpickle=None)
    fake.registry["m"] = {"versions": {"1": {"tags": {}}}, "aliases": {}}
    client = fake.client()
    for value in ("pending", "passed", "failed"):               # the values the notebooks used to write
        with pytest.raises(AssertionError, match="tag policy"):
            client.set_model_version_tag("m", "1", "validation_status", value)
        with fake.start_run(run_name="r"), pytest.raises(AssertionError, match="tag policy"):
            fake.set_tag("validation_status", value)
    for key, allowed in GOVERNED_TAG_VALUES.items():
        for value in sorted(allowed):
            client.set_model_version_tag("m", "1", key, value)
            assert fake.registry["m"]["versions"]["1"]["tags"][key] == value
    client.set_model_version_tag("m", "1", "gate_result", "passed")          # no policy: any value


def runs_by_name(fake):
    """Latest run per run name."""
    return {run["name"]: run for run in fake.runs.values()}


def runs_id_by_name(fake):
    """Run id of the latest run per run name."""
    return {run["name"]: run_id for run_id, run in fake.runs.items()}
