"""Runs the DAILY notebooks 06 -> 07 -> 08 on a local Spark session: one date twice (re-run) and a
second date once, then every dataset query of the committed dashboard against the tables they wrote.

Stand-ins: notebook globals (dbutils, display, spark), Unity Catalog tables (an in-memory dict;
Delta-specific writes are covered by test_scoring_and_io.py), and MLflow: a fake registry and a
fake ``mlflow.pyfunc.load_model`` returning models with the real output schemas, so 07_ / 08_ really
score on the driver with scoring.score_driver_side. Checks the table contents each notebook writes,
that re-running a date does not duplicate rows, and that the dashboard's SQL runs (ANSI mode, as on
a SQL warehouse), gives the latest day's funnel and leaves the app's warm-up pings out of the live
tiles. Skipped without a local pyspark + Java."""
import datetime as dt
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd

from common import config, uc_io
from conftest import DEMO_ROOT, TEST_CATALOG

RUN_DATES = ("2026-10-29", "2026-10-29", "2026-10-30")      # a date, its re-run, the next date
DASHBOARD = DEMO_ROOT / "dashboards" / "ecg12_afib.lvdash.json"


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
    class ArrModel:
        def predict(self, X):
            spread = X[:, 6, :].std(axis=1)
            prob = (spread > np.median(spread)).astype("float32") * 0.8 + 0.1
            return pd.DataFrame({"arr_prob": prob, "arr_flag": (prob >= 0.5).astype("float32")})

    class AfibModel:
        def predict(self, X):
            p = np.clip(X[:, 0, :].std(axis=1) / 2.0, 0.05, 0.95).astype("float32")
            return np.stack([1 - p, p], axis=1)

    loads = []

    def load_model(uri):
        loads.append(uri)
        return ArrModel() if config.ARR_MODEL_NAME in uri else AfibModel()

    def spark_udf(*args, **kwargs):
        raise AssertionError("the batch notebooks score on the driver, not with spark_udf")

    def get_model_version_by_alias(name, alias):
        assert alias == config.CHAMPION_ALIAS, alias            # daily scoring uses champion only
        return SimpleNamespace(version=3)

    mlflow = types.ModuleType("mlflow")
    mlflow.set_registry_uri = lambda uri: None
    mlflow.pyfunc = SimpleNamespace(spark_udf=spark_udf, load_model=load_model)
    mlflow.MlflowClient = lambda registry_uri=None: SimpleNamespace(
        get_model_version_by_alias=get_model_version_by_alias)
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    return loads


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


def _check_day(uc, day):
    """Checks the tables 06_ / 07_ / 08_ wrote for ``day``; returns that day's triaged rows."""
    batch_t = config.daily_table(config.BATCH_TABLE_PREFIX, day, TEST_CATALOG)
    triaged_t = config.daily_table(config.TRIAGED_TABLE_PREFIX, day, TEST_CATALOG)
    history_t = config.uc_name(config.TRIAGE_HISTORY_TABLE, TEST_CATALOG)
    results_t = config.uc_name(config.INFERENCE_TABLE, TEST_CATALOG)

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
    # score_driver_side's join: the id, the batch's other columns, then the triage result.
    assert list(triaged.columns) == [config.ID_COL, *[c for c in batch.columns if c != config.ID_COL],
                                     "arr_prob", "arr_flag", "arr_model_info"]
    assert dict(uc.tables[triaged_t].dtypes)["arr_prob"] == dict(uc.tables[triaged_t].dtypes)["arr_flag"] == "double"
    assert set(triaged["arr_prob"].round(6)) == {0.1, 0.9}             # the stand-in model's two values
    arr_info = f"models:/{config.uc_name(config.ARR_MODEL_NAME, TEST_CATALOG)}/3 (@champion)"
    assert set(triaged["arr_model_info"]) == {arr_info}

    # The history slice: exactly that day's triage rows, without the ECG arrays.
    history = uc.tables[history_t].toPandas()
    day_rows = history[history["batch_date"] == day]
    assert len(day_rows) == len(triaged) and day_rows[config.ID_COL].is_unique
    assert list(history.columns) == [c for c in triaged.columns if c not in config.ecg_columns(12)]

    results = uc.tables[results_t].toPandas()
    day_results = results[results["batch_date"] == day]
    assert len(day_results) == int((triaged["arr_flag"] == 1.0).sum())    # one row per flagged ECG
    assert day_results[config.ID_COL].is_unique
    assert not any(c.startswith("ecg") for c in results.columns)
    assert {"afib_predictions_0", "afib_predictions_1", "mrn", "age_years", "heart_rate_bpm"} <= set(results.columns)
    assert list(results.columns)[-3:] == ["afib_predictions_0", "afib_predictions_1", "afib_model_info"]
    np.testing.assert_allclose(results["afib_predictions_0"] + results["afib_predictions_1"], 1.0, rtol=1e-6)
    afib_info = f"models:/{config.uc_name(config.AFIB_MODEL_NAME, TEST_CATALOG)}/3 (@champion)"
    assert set(results["afib_model_info"]) == {afib_info}
    return triaged


def _payload_tables(spark):
    """Stand-ins for the two AI Gateway inference tables: the logged columns, one scored request
    (no client_request_id, as from /api/analyze) and one app warm-up ping each, with the response
    JSON in each endpoint's format."""
    import pyspark.sql.types as T

    schema = T.StructType([
        T.StructField("databricks_request_id", T.StringType()), T.StructField("client_request_id", T.StringType()),
        T.StructField("request_time", T.TimestampType()),
        T.StructField("status_code", T.IntegerType()), T.StructField("execution_duration_ms", T.LongType()),
        T.StructField("request", T.StringType()), T.StructField("response", T.StringType()),
        T.StructField("logging_error_codes", T.ArrayType(T.StringType()))])
    when = dt.datetime(2026, 10, 30, 9, 15)
    request = json.dumps({"inputs": [[[0.0] * 4] * 12]})
    responses = {config.ARR_ENDPOINT: {"predictions": [{"arr_prob": 0.9, "arr_flag": 1.0}]},
                 config.AFIB_ENDPOINT: {"predictions": [[0.2, 0.8]]}}
    warmup = {config.ARR_ENDPOINT: {"predictions": [{"arr_prob": 0.1, "arr_flag": 0.0}]},
              config.AFIB_ENDPOINT: {"predictions": [[0.6, 0.4]]}}
    return {config.payload_table_name(endpoint): spark.createDataFrame(
                [(f"req-{endpoint}", None, when, 200, 40, request, json.dumps(response), None),
                 (f"ping-{endpoint}", config.WARMUP_REQUEST_ID, when, 200, 35, request,
                  json.dumps(warmup[endpoint]), None)], schema)
            for endpoint, response in responses.items()}


def test_daily_notebooks_end_to_end_rerun_safe_and_the_dashboard_queries_run(spark, monkeypatch):
    monkeypatch.chdir(DEMO_ROOT / "workflow")
    _fake_writes(monkeypatch)
    loads = _fake_mlflow(monkeypatch)
    uc = LocalUC(spark)

    triaged = {}
    for run_date in RUN_DATES:
        widgets = {config.RUN_DATE_PARAM: run_date, "catalog_name": TEST_CATALOG}
        for name in ("06_generate_batch_data.py", "07_arr_triage.py", "08_afib_classify.py"):
            _run_notebook(name, uc, widgets)
        day = dt.date.fromisoformat(run_date)
        triaged[day] = _check_day(uc, day)                     # also after the re-run: no duplicates

    # champion was resolved to version 3, and that exact version (never "@champion") was loaded.
    assert set(loads) == {f"models:/{config.uc_name(name, TEST_CATALOG)}/3"
                          for name in (config.ARR_MODEL_NAME, config.AFIB_MODEL_NAME)}
    history = uc.tables[config.uc_name(config.TRIAGE_HISTORY_TABLE, TEST_CATALOG)].toPandas()
    assert sorted(set(history["batch_date"])) == sorted(triaged)            # two dates, one copy each
    assert len(history) == sum(len(t) for t in triaged.values())

    # Every dashboard query, on the tables under the bare names the dashboard uses.
    views = {config.TRIAGE_HISTORY_TABLE: config.uc_name(config.TRIAGE_HISTORY_TABLE, TEST_CATALOG),
             config.INFERENCE_TABLE: config.uc_name(config.INFERENCE_TABLE, TEST_CATALOG)}
    for bare, full in views.items():
        uc.tables[full].createOrReplaceTempView(bare)
    for bare, df in _payload_tables(spark).items():
        df.createOrReplaceTempView(bare)
    datasets = {d["name"]: "".join(d["queryLines"]) for d in json.loads(DASHBOARD.read_text())["datasets"]}
    spark.conf.set("spark.sql.ansi.enabled", "true")            # SQL warehouses run in ANSI mode
    try:
        out = {name: spark.sql(query).toPandas() for name, query in datasets.items()}
    finally:
        spark.conf.unset("spark.sql.ansi.enabled")

    latest = max(triaged)
    day = triaged[latest]
    results = uc.tables[views[config.INFERENCE_TABLE]].toPandas()
    day_results = results[results["batch_date"] == latest]
    funnel = out["funnel"].iloc[0]
    assert funnel["total_signals"] == len(day)                                   # the latest day only
    assert funnel["flagged_abnormal"] == int((day["arr_flag"] == 1.0).sum())
    assert funnel["afib_positive"] == int((day_results["afib_predictions_1"] > config.AFIB_THRESHOLD).sum())
    assert out["arrivals"]["n"].sum() == out["triage_accuracy"]["n"].sum() == len(day)
    assert out["prob_hist"]["n"].sum() == len(day_results)
    assert out["afib_by_age"]["screened"].sum() == len(day)
    assert sorted(out["trend"]["date"]) == sorted(triaged)                        # every day
    assert set(out["worklist"]["batch_date"]) == {latest} and len(out["worklist"]) == min(25, len(day_results))
    arr_label, afib_label = f"{config.ARR_ENDPOINT} (triage)", f"{config.AFIB_ENDPOINT} (classify)"
    # The warm-up pings are left out of the live tiles and only counted.
    assert sorted(out["live_requests"]["endpoint"]) == sorted([arr_label, afib_label])
    assert set(out["live_requests"]["databricks_request_id"]) == {f"req-{config.ARR_ENDPOINT}",
                                                                  f"req-{config.AFIB_ENDPOINT}"}
    assert set(out["live_requests"]["not_logged"]) == {0}
    assert len(out["live_predictions"]) == 2
    scores = dict(zip(out["live_predictions"]["endpoint"], out["live_predictions"]["score"]))
    assert scores == {arr_label: 0.9, afib_label: 0.8}
    assert sorted(out["live_warmups"]["databricks_request_id"]) == sorted(
        [f"ping-{config.ARR_ENDPOINT}", f"ping-{config.AFIB_ENDPOINT}"])
