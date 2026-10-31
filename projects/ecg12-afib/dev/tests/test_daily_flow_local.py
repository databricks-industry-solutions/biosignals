"""Runs the DAILY notebooks 06 -> 07 -> 08 on a local Spark session, twice for the same date.

Stand-ins: notebook globals (dbutils, display, spark), Unity Catalog tables (an in-memory dict;
Delta-specific writes are covered by test_scoring_and_io.py), and MLflow: a fake registry, and a
fake ``mlflow.pyfunc.spark_udf`` that builds what the real one builds for these models (an
iterator pandas UDF that reshapes the flattened 12-lead array to (n, 12, 2500) and returns the
requested result type), around models that return the preserved output schemas. So 07_ / 08_
really score on the local Spark workers. Checks the table contents each notebook writes and that
re-running a date does not duplicate rows. Skipped without a local pyspark + Java + pyarrow
(pandas UDFs)."""
import datetime as dt
import os
import sys
import types
from types import SimpleNamespace
from typing import Iterator

import numpy as np
import pandas as pd
import pytest

from common import config, scoring, uc_io
from conftest import DEMO_ROOT

RUN_DATE = "2026-10-29"


class LocalUC:
    """Spark session whose `table()` reads an in-memory dict of three-level table names."""

    def __init__(self, session):
        self._session = session
        self.tables = {}
        self.catalog = SimpleNamespace(tableExists=lambda name: name in self.tables)

    def table(self, name):
        return self.tables[name]

    def sql(self, statement):
        assert statement.startswith("OPTIMIZE "), statement

    def __getattr__(self, name):
        return getattr(self._session, name)


def _fake_writes(monkeypatch):
    def overwrite_table(spark, df, table, optimize=True):
        spark.tables[table] = spark.createDataFrame(df.toPandas(), df.schema)   # materialize

    def write_date_slice(spark, df, table, run_date, date_col=config.BATCH_DATE_COL):
        new = spark.createDataFrame(df.toPandas(), df.schema)
        if table not in spark.tables:
            spark.tables[table] = new
            return "create"
        kept = spark.tables[table].filter(f"{date_col} != DATE'{run_date.isoformat()}'")
        spark.tables[table] = kept.unionByName(new, allowMissingColumns=True)
        return "replace"

    monkeypatch.setattr(uc_io, "overwrite_table", overwrite_table)
    monkeypatch.setattr(uc_io, "write_date_slice", write_date_slice)


def _fake_mlflow(monkeypatch):
    # Defined inside this function, so cloudpickle ships them to the Spark workers by value.
    class ArrModel:
        def predict(self, X):
            spread = X[:, 6, :].std(axis=1)
            prob = (spread > np.median(spread)).astype("float32") * 0.8 + 0.1
            return pd.DataFrame({"arr_prob": prob, "arr_flag": (prob >= 0.5).astype("float32")})

    class AfibModel:
        def predict(self, X):
            p = np.clip(X[:, 0, :].std(axis=1) / 2.0, 0.05, 0.95).astype("float32")
            return np.stack([1 - p, p], axis=1)

    udfs = []

    def spark_udf(spark, model_uri, result_type=None, env_manager=None):
        # The extra index the environment build would read: set by scoring.torch_cpu_wheels().
        udfs.append((model_uri, result_type, env_manager, os.environ.get("UV_EXTRA_INDEX_URL")))
        model = ArrModel() if config.ARR_MODEL_NAME in model_uri else AfibModel()
        n_leads, n_samples = config.N_LEADS, 2500

        def tensor(flat):
            return np.vstack(flat.tolist()).reshape(-1, n_leads, n_samples).astype("float32")

        if result_type.startswith("array<"):
            def predict_array(batches: Iterator[pd.Series]) -> Iterator[pd.Series]:
                for flat in batches:
                    yield pd.Series(np.asarray(model.predict(tensor(flat)), dtype="float64").tolist())
            return pandas_udf(predict_array, result_type)

        def predict_struct(batches: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
            for flat in batches:
                yield model.predict(tensor(flat)).astype("float64")
        return pandas_udf(predict_struct, result_type)

    def get_model_version_by_alias(name, alias):
        assert alias == config.CHAMPION_ALIAS, alias            # daily scoring uses champion only
        return SimpleNamespace(version=3)

    def load_model(uri):
        raise AssertionError("the batch notebooks score with spark_udf, not a driver-side load_model")

    from pyspark.sql.functions import pandas_udf

    mlflow = types.ModuleType("mlflow")
    mlflow.set_registry_uri = lambda uri: None
    mlflow.pyfunc = SimpleNamespace(spark_udf=spark_udf, load_model=load_model)
    mlflow.MlflowClient = lambda registry_uri=None: SimpleNamespace(
        get_model_version_by_alias=get_model_version_by_alias)
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    return udfs


def _run_notebook(name, uc, widgets):
    values = {}

    def text(key, default, label=None):
        values.setdefault(key, widgets.get(key, default))

    dbutils = SimpleNamespace(widgets=SimpleNamespace(text=text, get=lambda key: values[key]),
                              library=SimpleNamespace(restartPython=lambda: None))
    source = "\n".join("" if line.startswith("# MAGIC") else line
                       for line in (DEMO_ROOT / "workflow" / name).read_text().splitlines())
    exec(compile(source, name, "exec"),
         {"__name__": "__main__", "spark": uc, "dbutils": dbutils, "display": lambda df: df.collect()})


def test_daily_notebooks_end_to_end_and_rerun_safe(spark, monkeypatch):
    pytest.importorskip("pyarrow")
    monkeypatch.chdir(DEMO_ROOT / "workflow")
    monkeypatch.delenv("UV_EXTRA_INDEX_URL", raising=False)
    monkeypatch.setattr(scoring.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")   # uv "on PATH"
    _fake_writes(monkeypatch)
    udfs = _fake_mlflow(monkeypatch)
    uc = LocalUC(spark)
    widgets = {config.RUN_DATE_PARAM: RUN_DATE}
    day = dt.date.fromisoformat(RUN_DATE)
    batch_t = config.daily_table(config.BATCH_TABLE_PREFIX, day)
    triaged_t = config.daily_table(config.TRIAGED_TABLE_PREFIX, day)
    results_t = config.uc_name(config.INFERENCE_TABLE)
    spark.conf.unset(scoring.ARROW_BATCH_ROWS_CONF)

    for attempt in range(2):                                   # second pass = re-run of the date
        for name in ("06_generate_batch_data.py", "07_arr_triage.py", "08_afib_classify.py"):
            _run_notebook(name, uc, widgets)

        batch = uc.tables[batch_t].toPandas()
        assert 85 <= len(batch) <= 115 and batch[config.ID_COL].is_unique
        assert [c for c in batch.columns if c.startswith("ecg")] == config.ecg_columns(12)
        assert {"pmh", "anti_htn", "white", "height", "weight"}.isdisjoint(batch.columns)
        assert len(batch["ecg12"].iloc[0]) == 2500
        assert (batch["n_leads"] == 12).all() and (batch["batch_date"] == day).all()
        assert (batch["bmi"] == (batch["weight_kg"] / (batch["height_cm"] / 100) ** 2).round(1)).all()
        acquired = pd.to_datetime(batch["acquisition_datetime_utc"])          # session time zone = UTC
        assert (acquired.dt.date == day).all() and acquired.dt.hour.between(7, 18).all()
        assert acquired.dt.hour.nunique() >= 8                             # spread over the day, not one instant
        assert (acquired.dt.strftime("%d-%m-%Y %H:%M:%S") == batch["batch_inference_datetime_utc"]).all()

        triaged = uc.tables[triaged_t].toPandas()
        assert len(triaged) == len(batch) and set(triaged["arr_flag"]) == {0.0, 1.0}
        # Same columns and order as the driver-side join this replaced: the id, the batch's other
        # columns, then the triage result.
        assert list(triaged.columns) == [config.ID_COL, *[c for c in batch.columns if c != config.ID_COL],
                                         "arr_prob", "arr_flag", "arr_model_info"]
        assert dict(uc.tables[triaged_t].dtypes)["arr_prob"] == dict(uc.tables[triaged_t].dtypes)["arr_flag"] == "double"
        assert set(triaged["arr_prob"].round(6)) == {0.1, 0.9}             # the stand-in model's two values
        assert set(triaged["arr_model_info"]) == {f"models:/{config.uc_name(config.ARR_MODEL_NAME)}/3 (@champion)"}

        results = uc.tables[results_t].toPandas()
        assert len(results) == int((triaged["arr_flag"] == 1.0).sum())    # one row per flagged ECG
        assert results[config.ID_COL].is_unique
        assert not any(c.startswith("ecg") for c in results.columns)
        assert {"afib_predictions_0", "afib_predictions_1", "mrn", "age_years", "heart_rate_bpm"} <= set(results.columns)
        assert list(results.columns)[-3:] == ["afib_predictions_0", "afib_predictions_1", "afib_model_info"]
        np.testing.assert_allclose(results["afib_predictions_0"] + results["afib_predictions_1"], 1.0, rtol=1e-6)
        assert set(results["afib_model_info"]) == {f"models:/{config.uc_name(config.AFIB_MODEL_NAME)}/3 (@champion)"}

    # champion was resolved to version 3, and that exact version (never "@champion") was wrapped as
    # a spark_udf with the result type each model's output needs, in the model's own environment
    # rebuilt by uv with the CPU torch index; the index is unset again after each call.
    cpu = scoring.TORCH_CPU_INDEX_URL
    assert set(udfs) == {
        (f"models:/{config.uc_name(config.ARR_MODEL_NAME)}/3", scoring.ARR_UDF_RESULT_TYPE, "uv", cpu),
        (f"models:/{config.uc_name(config.AFIB_MODEL_NAME)}/3", scoring.AFIB_UDF_RESULT_TYPE, "uv", cpu),
    }
    assert "UV_EXTRA_INDEX_URL" not in os.environ
    # The scoring tasks capped the UDF's Arrow batches (= ECGs per model call) on their session.
    assert spark.conf.get(scoring.ARROW_BATCH_ROWS_CONF) == str(config.BATCH_UDF_ROWS_PER_PARTITION)
    spark.conf.unset(scoring.ARROW_BATCH_ROWS_CONF)
