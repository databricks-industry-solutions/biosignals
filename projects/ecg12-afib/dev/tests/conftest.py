"""Shared test setup.

* Puts projects/ecg12-afib on sys.path so tests import the same `common` package the notebooks use.
* Selects matplotlib's non-interactive Agg backend, so the notebooks' plots render off-screen.
* ``spark``: a local Spark session for the tests that need one; skipped when only
  Databricks Connect (no local Spark) or no Java runtime is installed.
* ``GOVERNED_TAG_VALUES``: the only values the notebooks may write for tag keys that a Unity
  Catalog tag policy may restrict.
* ``TEST_CATALOG``: the catalog the tests pass as the (required) ``catalog_name`` parameter.
* ``FakeTracking``: MLflow experiments + workspace folders as Databricks treats them, for
  ``common/tracking.py`` and the SETUP flow (``SP_EXPERIMENT``: an identity with no home folder).

Python 3.11: the Spark tests need it (pyspark 3.5.3 pickles the functions it sends to its workers
with its bundled cloudpickle 2.2.1, which predates Python 3.12 support); see README.md (this folder)
and run_tests_python311.sh.
"""
import os
import posixpath
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

DEMO_ROOT = Path(__file__).resolve().parents[2]          # projects/ecg12-afib (this file: dev/tests/)
sys.path.insert(0, str(DEMO_ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

# Tag keys the notebooks write that a Unity Catalog tag policy may restrict, with the only values the
# notebooks may write for each. Under a tag policy Unity Catalog rejects any other value on a model
# version ("Tag value pending is not an allowed value for tag policy key validation_status. Allowed
# values: [valid, invalid]."). validation_status is commonly governed this way, so the gate writes
# only valid / invalid. If your workspace has a policy on another key the notebooks write
# (`databricks tag-policies list-tag-policies -p <profile>`), add it here: test_notebooks.py and
# test_setup_flow_local.py then fail when a notebook could write a value it does not allow.
GOVERNED_TAG_VALUES = {"validation_status": frozenset({"valid", "invalid"})}

# The catalog the tests pass as the catalog_name parameter (the notebooks' widget has no default).
TEST_CATALOG = "test_catalog"
# The mlflow_experiment parameter as the bundle default resolves for a service principal (its user
# name is its application id) that has no home folder in the workspace yet.
SP_EXPERIMENT = "/Users/11111111-2222-3333-4444-555555555555/biosignals-ecg12-afib"


class MlflowException(Exception):
    """Stands in for mlflow.exceptions.MlflowException: ``error_code`` is the error's name."""

    def __init__(self, message, error_code):
        super().__init__(message)
        self.error_code = error_code


class FakeTracking:
    """MLflow experiments (named by workspace path) and workspace folders, as on Databricks:

    * an experiment can only be created inside an existing folder ("Parent directory ... does not
      exist"), and ``mlflow.set_experiment`` creates a missing experiment but not its folder;
    * ``workspace.mkdirs`` (databricks-sdk) creates a folder and its missing parents, and does
      nothing for an existing one;
    * when two tasks create the same experiment at once, the loser's ``mlflow.set_experiment``
      returns the experiment WITHOUT making it active (MLflow returns early on
      RESOURCE_ALREADY_EXISTS): ``created_meanwhile`` names experiments another task creates between
      this task's lookup and its create.

    It starts with no home folder for anyone, as for a service principal that never had one."""

    def __init__(self):
        self.folders = {"/", "/Users", "/Shared"}
        self.experiments = {}                   # path -> experiment id
        self.created_meanwhile = set()
        self.active = None                      # the active experiment's path
        self.workspace = SimpleNamespace(mkdirs=self.mkdirs)

    def mkdirs(self, path):
        assert path.startswith("/"), path
        while path not in self.folders:
            self.folders.add(path)
            path = posixpath.dirname(path)

    def get_experiment_by_name(self, name):
        exp_id = self.experiments.get(name)
        return None if exp_id is None else SimpleNamespace(experiment_id=exp_id, name=name)

    def create_experiment(self, name):
        if name in self.created_meanwhile:
            self.created_meanwhile.discard(name)
            self.experiments[name] = str(len(self.experiments) + 1)
        if name in self.experiments:
            raise MlflowException(f"Experiment {name!r} already exists.", "RESOURCE_ALREADY_EXISTS")
        if posixpath.dirname(name) not in self.folders:
            raise MlflowException(f"Parent directory {posixpath.dirname(name)} does not exist.",
                                  "RESOURCE_DOES_NOT_EXIST")
        self.experiments[name] = str(len(self.experiments) + 1)
        return self.experiments[name]

    def set_experiment(self, experiment_name):
        if self.get_experiment_by_name(experiment_name) is None:
            try:
                self.create_experiment(experiment_name)
            except MlflowException as error:
                if error.error_code == "RESOURCE_ALREADY_EXISTS":
                    return self.get_experiment_by_name(experiment_name)     # not made active
                raise
        self.active = experiment_name
        return self.get_experiment_by_name(experiment_name)

    def install(self, monkeypatch, mlflow):
        """Adds the experiment calls to the stand-in ``mlflow`` module, plus ``mlflow.exceptions`` and a
        ``databricks.sdk`` whose ``WorkspaceClient()`` is this workspace."""
        for name in ("get_experiment_by_name", "create_experiment", "set_experiment"):
            setattr(mlflow, name, getattr(self, name))
        exceptions = types.ModuleType("mlflow.exceptions")
        exceptions.MlflowException = MlflowException
        mlflow.exceptions = exceptions
        databricks, sdk = types.ModuleType("databricks"), types.ModuleType("databricks.sdk")
        sdk.WorkspaceClient = lambda: self
        databricks.sdk = sdk
        for name, module in {"mlflow": mlflow, "mlflow.exceptions": exceptions, "databricks": databricks,
                             "databricks.sdk": sdk}.items():
            monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture(scope="session")
def spark():
    pyspark_sql = pytest.importorskip("pyspark.sql")
    # pyspark converts timestamps through the process time zone; without TZ it falls back to
    # "dateutil/:", which pandas >= 2.2 cannot localize. Databricks sets the time zone itself.
    os.environ.setdefault("TZ", "UTC")
    try:
        session = (pyspark_sql.SparkSession.builder.master("local[2]").appName("ecg-afib-tests")
                   .config("spark.ui.enabled", "false")
                   .config("spark.driver.host", "127.0.0.1")          # stay on loopback, not the LAN IP
                   .config("spark.driver.bindAddress", "127.0.0.1")
                   .getOrCreate())
    except Exception as exc:  # noqa: BLE001 (databricks-connect only, or no Java runtime: skip)
        pytest.skip(f"local Spark unavailable: {exc}")
    yield session
    session.stop()
