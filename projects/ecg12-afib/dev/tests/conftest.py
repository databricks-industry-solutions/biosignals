"""Shared test setup.

* Puts projects/ecg12-afib on sys.path so tests import the same `common` package the notebooks use.
* Selects matplotlib's non-interactive Agg backend, so the notebooks' plots render off-screen.
* ``spark``: a local Spark session for the tests that need one; skipped when only
  Databricks Connect (no local Spark) or no Java runtime is installed.
* ``pickle_by_value_for_spark_workers``: lets a test module ship its own functions to the Spark
  worker processes (UDFs), which cannot import the tests folder.
* ``GOVERNED_TAG_VALUES``: the workspace's tag policies for the tag keys the notebooks write.

Python 3.11: the Spark tests need it (pyspark 3.5.3 pickles UDFs with its bundled cloudpickle
2.2.1, which predates Python 3.12 support); see README.md (this folder) and run_tests_python311.sh.
"""
import os
import sys
from pathlib import Path

import pytest

DEMO_ROOT = Path(__file__).resolve().parents[2]          # projects/ecg12-afib (this file: dev/tests/)
sys.path.insert(0, str(DEMO_ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

# Tag policies (governed tags) of workspace fevm-mmt-aws-usw2 that cover a tag key the notebooks
# write, with the only values each allows. Source: `databricks tag-policies list-tag-policies
# -p fevm-mmt-aws-usw2`, checked 2026-10-30 against every key 02_ / 03_ / 04_ write: only
# validation_status has a policy. Unity Catalog rejects any other value on a model version
# ("Tag value pending is not an allowed value for tag policy key validation_status. Allowed values:
# [valid, invalid]."), which failed the first live SETUP run. Re-check the list when adding a tag key.
GOVERNED_TAG_VALUES = {"validation_status": frozenset({"valid", "invalid"})}


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
    except Exception as exc:  # databricks-connect only, or no Java runtime
        pytest.skip(f"local Spark unavailable: {exc}")
    yield session
    session.stop()


def pickle_by_value_for_spark_workers(module_name):
    """Make cloudpickle pickle the functions and classes of test module ``module_name`` by value,
    both pyspark's bundled copy (UDFs) and the standalone one MLflow uses to save a pyfunc model,
    when installed. Spark's Python workers cannot import the tests folder, so a pickled reference
    to the test module would fail to load there."""
    from pyspark import cloudpickle as pyspark_cloudpickle

    module = sys.modules[module_name]
    pyspark_cloudpickle.register_pickle_by_value(module)
    try:
        import cloudpickle
    except ImportError:
        return
    cloudpickle.register_pickle_by_value(module)
