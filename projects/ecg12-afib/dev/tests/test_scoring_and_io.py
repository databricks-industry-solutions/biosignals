"""Tests for config.py, scoring.py and uc_io.py (Spark and MLflow replaced by small fakes, plus
one check of the Spark-worker path against the real mlflow.pyfunc.spark_udf when mlflow is installed;
no job uses that path yet, see README.md "Future work")."""
import datetime as dt
import os
from types import SimpleNamespace
from typing import Iterator

import numpy as np
import pandas as pd
import pytest

from common import config, scoring, uc_io
from conftest import pickle_by_value_for_spark_workers


# --- config ------------------------------------------------------------------------------

def test_names_and_run_date():
    # No workspace in config: no catalog default, host, workspace id or dashboard id. The catalog is
    # a required parameter (bundle variable -> job parameter / app environment).
    assert config.SCHEMA == "biosignals"
    assert not {"CATALOG", "WORKSPACE_HOST", "WORKSPACE_ID", "DASHBOARD_ID"} & set(vars(config))
    assert config.uc_location("c", "s") == ("c", "s")
    assert config.uc_location(" c ", "") == config.uc_location("c", None) == ("c", config.SCHEMA)
    for empty in ("", "  ", None):
        with pytest.raises(ValueError, match="catalog_name is empty"):
            config.uc_location(empty, "s")
    # Fixed model names (scoped by catalog.schema); workspace-wide endpoint names from a prefix,
    # by default the model names.
    assert (config.ARR_MODEL_NAME, config.AFIB_MODEL_NAME) == ("bio_ecg12_arr", "bio_ecg12_afib")
    assert not {"ARR_ENDPOINT", "AFIB_ENDPOINT"} & set(vars(config))         # no fixed endpoint name
    assert config.endpoint_names() == config.endpoint_names("") == config.endpoint_names(None) == (
        config.ARR_MODEL_NAME, config.AFIB_MODEL_NAME)
    assert config.endpoint_names(" acme-ecg_1 ") == ("acme-ecg_1_arr", "acme-ecg_1_afib")
    for bad in ("a b", "a.b", "a/b", "ä"):
        with pytest.raises(ValueError, match="endpoint_prefix"):
            config.endpoint_names(bad)
    # The accumulating tables, one slice per batch_date.
    assert config.INFERENCE_TABLE == "dailyOrweekly_arr_afib_inference_12l"
    assert config.TRIAGE_HISTORY_TABLE == "arr_triage_history_12l"
    assert config.uc_name("x", "c") == f"c.{config.SCHEMA}.x"
    d = dt.date(2026, 10, 29)
    assert config.resolve_run_date("2026-10-29") == d
    assert config.resolve_run_date("") == dt.datetime.now(dt.timezone.utc).date()
    assert config.resolve_run_date(None) == dt.datetime.now(dt.timezone.utc).date()
    with pytest.raises(ValueError, match="run_date"):
        config.resolve_run_date("29-10-2026")
    assert config.daily_table(config.BATCH_TABLE_PREFIX, d, "c") == f"c.{config.SCHEMA}.batch_data_12l_20261029"
    # Payload tables are named by model, whatever the endpoint prefix.
    assert config.payload_table_name(config.ARR_MODEL_NAME) == "bio_ecg12_arr_payload"
    assert config.payload_table(config.ARR_MODEL_NAME, "c", "s") == "c.s.bio_ecg12_arr_payload"
    assert config.payload_table(config.AFIB_MODEL_NAME, "c", "s") == "c.s.bio_ecg12_afib_payload"
    assert config.ecg_columns() == [f"ecg{i}" for i in range(1, 13)]
    # The app's live calls: the bound on one endpoint call, the keep-alive interval, the ping tag.
    assert (config.SERVING_TIMEOUT_S, config.WARMUP_INTERVAL_S, config.WARMUP_REQUEST_ID) == (30, 600, "app-warmup")


def test_model_lifecycle_and_training_settings():
    # Lowercase aliases: Unity Catalog stores them lowercase.
    assert (config.CHALLENGER_ALIAS, config.CHAMPION_ALIAS) == ("challenger", "champion")
    # The training table is its own: not a daily table, not the results table.
    t = config.TRAINING_TABLE
    assert t == "training_data_12l" and t != config.INFERENCE_TABLE
    assert not t.startswith((config.BATCH_TABLE_PREFIX, config.TRIAGED_TABLE_PREFIX))
    splits = config.TRAINING_SPLITS
    assert list(splits) == [config.TRAIN_SPLIT, config.VAL_SPLIT, config.HOLDOUT_SPLIT] == ["train", "val", "holdout"]
    assert len({seed for _, seed in splits.values()}) == 3                    # independent draws
    assert splits[config.HOLDOUT_SPLIT][0] >= config.VALIDATION_MIN_ROWS
    assert 0 < config.TRAINING_FRAC_AFIB + config.TRAINING_FRAC_OTHER < 1
    assert set(config.TRAIN_PARAMS) == {"learning_rate", "epochs", "batch_size", "seed"}
    # One gate entry per model; every threshold is min_/max_ + a metric of common/evaluation.py.
    assert set(config.VALIDATION_THRESHOLDS) == {config.ARR_MODEL_NAME, config.AFIB_MODEL_NAME}
    perfect = {"n": 200.0, "n_positive": 100.0, "accuracy": 1.0, "roc_auc": 1.0, "sensitivity": 1.0,
               "specificity": 1.0, "called_positive_rate_nsr": 0.0, "called_positive_rate_afib": 1.0,
               "called_positive_rate_other": 0.0}
    from common import evaluation
    for model, thresholds in config.VALIDATION_THRESHOLDS.items():
        assert all(0.0 <= v <= 1.0 for v in thresholds.values()), model
        assert evaluation.check_thresholds(perfect, thresholds, config.VALIDATION_MIN_ROWS) == [], model


# --- scoring -----------------------------------------------------------------------------

def test_resolve_alias_pins_the_concrete_version():
    calls = []

    class FakeClient:
        def get_model_version_by_alias(self, name, alias):
            calls.append((name, alias))
            return SimpleNamespace(version=7)

    m = scoring.resolve_alias(FakeClient(), "c.s.bio_ecg12_arr")
    assert calls == [("c.s.bio_ecg12_arr", "champion")]
    assert m.uri == "models:/c.s.bio_ecg12_arr/7"            # loads that version, not @champion
    assert m.info == "models:/c.s.bio_ecg12_arr/7 (@champion)"
    challenger = scoring.resolve_alias(FakeClient(), "c.s.bio_ecg12_arr", config.CHALLENGER_ALIAS)
    assert calls[-1] == ("c.s.bio_ecg12_arr", "challenger") and challenger.alias == "challenger"


def test_ecg_tensor_shapes():
    rows = [[np.full(5, i, dtype=np.float32) for _ in range(12)] for i in range(3)]
    t = scoring.ecg_tensor(rows, n_samples=5)
    assert t.shape == (3, 12, 5) and t.dtype == np.float32
    with pytest.raises(ValueError):
        scoring.ecg_tensor(rows, n_samples=6)
    with pytest.raises(ValueError):
        scoring.ecg_tensor([], n_samples=5)


def test_output_schema_readers():
    arr = scoring.arr_columns(pd.DataFrame({"arr_prob": [0.2, 0.9], "arr_flag": [0.0, 1.0]}))
    assert list(arr) == list(scoring.ARR_OUTPUT_COLS)
    np.testing.assert_allclose(arr["arr_flag"], [0.0, 1.0])
    with pytest.raises(ValueError):
        scoring.arr_columns(pd.DataFrame({"prob": [0.1]}))
    afib = scoring.afib_columns(np.array([[0.8, 0.2], [0.1, 0.9]]))
    assert list(afib) == list(scoring.AFIB_OUTPUT_COLS)
    np.testing.assert_allclose(afib["afib_predictions_1"], [0.2, 0.9])
    with pytest.raises(ValueError):
        scoring.afib_columns(np.array([0.1, 0.9]))


def test_torch_cpu_wheels_sets_the_cpu_index_only_for_the_environment_build(monkeypatch):
    """spark_udf rebuilds the model's environment with uv (the serverless UDF workers lack the task
    environment's torch); torch_cpu_wheels points that build at PyTorch's CPU wheels and then
    restores the variable, and fails clearly without a uv command."""
    assert scoring.UDF_ENV_MANAGER == "uv"
    assert scoring.TORCH_CPU_INDEX_URL == "https://download.pytorch.org/whl/cpu"
    monkeypatch.setattr(scoring.shutil, "which", lambda cmd: f"/usr/bin/{cmd}")
    monkeypatch.delenv("UV_EXTRA_INDEX_URL", raising=False)
    with scoring.torch_cpu_wheels():
        assert os.environ["UV_EXTRA_INDEX_URL"] == scoring.TORCH_CPU_INDEX_URL
    assert "UV_EXTRA_INDEX_URL" not in os.environ
    monkeypatch.setenv("UV_EXTRA_INDEX_URL", "https://example.invalid/simple")    # a caller's own index
    with scoring.torch_cpu_wheels():
        assert os.environ["UV_EXTRA_INDEX_URL"] == f"{scoring.TORCH_CPU_INDEX_URL} https://example.invalid/simple"
    assert os.environ["UV_EXTRA_INDEX_URL"] == "https://example.invalid/simple"
    monkeypatch.setattr(scoring.shutil, "which", lambda cmd: None)
    with pytest.raises(RuntimeError, match="uv command"), scoring.torch_cpu_wheels():
        pass


# --- uc_io -------------------------------------------------------------------------------

class FakeWriter:
    def __init__(self, log):
        self.log = log

    def format(self, fmt):
        self.log.append(("format", fmt))
        return self

    def mode(self, mode):
        self.log.append(("mode", mode))
        return self

    def option(self, key, value):
        self.log.append(("option", key, value))
        return self

    def saveAsTable(self, table):
        self.log.append(("saveAsTable", table))


class FakeSpark:
    def __init__(self, existing_columns=None):
        self.existing_columns = existing_columns
        self.sql_log = []
        self.catalog = SimpleNamespace(tableExists=lambda t: existing_columns is not None)

    def table(self, _):
        return SimpleNamespace(columns=self.existing_columns)

    def sql(self, stmt):
        self.sql_log.append(stmt)


def _df(log):
    return SimpleNamespace(write=FakeWriter(log))


def test_overwrite_table_replaces_data_and_schema():
    log, spark = [], FakeSpark()
    uc_io.overwrite_table(spark, _df(log), "c.s.t")
    assert ("mode", "overwrite") in log and ("option", "overwriteSchema", "true") in log
    assert spark.sql_log == ["OPTIMIZE c.s.t"]


@pytest.mark.parametrize("existing, expected_mode, expected_options", [
    (None, "create", []),                                                        # new table
    (["pseudo_patient_id", "batch_date"], "replace",
     [("option", "replaceWhere", "batch_date = DATE'2026-10-29'")]),           # re-run of a date
])
def test_write_date_slice_never_duplicates_a_date(existing, expected_mode, expected_options):
    log = []
    mode = uc_io.write_date_slice(FakeSpark(existing), _df(log), "c.s.inf", dt.date(2026, 10, 29))
    assert mode == expected_mode
    assert [entry for entry in log if entry[0] == "option"] == expected_options
    assert ("mode", "overwrite") in log
    assert log[-1] == ("saveAsTable", "c.s.inf")


def test_table_version_and_time_travel_read():
    class HistorySpark(FakeSpark):
        def sql(self, stmt):
            self.sql_log.append(stmt)
            return SimpleNamespace(collect=lambda: [{"version": 4}]) if stmt.startswith("DESCRIBE") else stmt

    spark = HistorySpark()
    assert uc_io.table_version(spark, "c.s.training_data_12l") == 4
    assert uc_io.read_table_version(spark, "c.s.training_data_12l", 4) == "SELECT * FROM c.s.training_data_12l VERSION AS OF 4"
    assert spark.sql_log[0] == "DESCRIBE HISTORY c.s.training_data_12l LIMIT 1"
    spark.sql = lambda stmt: SimpleNamespace(collect=lambda: [])
    with pytest.raises(ValueError, match="no Delta history"):
        uc_io.table_version(spark, "c.s.empty")


def test_write_date_slice_refuses_a_table_without_batch_date():
    """A table this function did not write (no batch_date column): nothing is written to it."""
    log = []
    with pytest.raises(ValueError, match="no batch_date column"):
        uc_io.write_date_slice(FakeSpark(["pseudo_patient_id", "afib_predictions_1"]), _df(log),
                               "c.s.some_other_results_table", dt.date(2026, 10, 29))
    assert not any(entry[0] == "saveAsTable" for entry in log)


# --- score_driver_side on a local Spark session (conftest `spark`; skipped without one) -----

def _batch_sdf(spark, n_rows, n_samples=5):
    import pyspark.sql.types as T
    ecg_cols = config.ecg_columns()
    schema = T.StructType(
        [T.StructField("pseudo_patient_id", T.StringType()), T.StructField("age_years", T.IntegerType())]
        + [T.StructField(c, T.ArrayType(T.FloatType())) for c in ecg_cols])
    # Lead j, sample k of row i = i + j + k / 100: every lead and sample differs, so a scoring path
    # that mixes up lead or sample order gives different predictions.
    rows = [(f"SYNPT-{i:06d}", 60 + i, *[[i + j + k / 100 for k in range(n_samples)] for j in range(len(ecg_cols))])
            for i in range(n_rows)]
    return spark.createDataFrame(rows, schema), ecg_cols


def test_score_driver_side_joins_predictions_by_id(spark):
    sdf, ecg_cols = _batch_sdf(spark, 3)
    seen = {}

    def predict(X):
        seen["shape"] = X.shape
        prob = X[:, 0, 0] / 10.0                       # row i -> lead I value i -> prob i/10
        return scoring.arr_columns(pd.DataFrame({"arr_prob": prob, "arr_flag": (prob >= 0.15).astype("float32")}))

    out = scoring.score_driver_side(spark, sdf, predict, output_cols=scoring.ARR_OUTPUT_COLS,
                                    ecg_cols=ecg_cols, n_samples=5,
                                    model_info_col="arr_model_info", model_info="models:/m/3 (@champion)")
    assert seen["shape"] == (3, 12, 5)
    got = {r["pseudo_patient_id"]: r for r in out.collect()}
    assert sorted(got) == ["SYNPT-000000", "SYNPT-000001", "SYNPT-000002"]
    assert got["SYNPT-000002"]["arr_prob"] == pytest.approx(0.2) and got["SYNPT-000002"]["arr_flag"] == 1.0
    assert got["SYNPT-000000"]["arr_flag"] == 0.0 and got["SYNPT-000001"]["age_years"] == 61
    assert {r["arr_model_info"] for r in got.values()} == {"models:/m/3 (@champion)"}
    assert dict(out.dtypes)["arr_prob"] == "double"


def test_score_driver_side_empty_input_skips_the_model(spark):
    sdf, ecg_cols = _batch_sdf(spark, 0)

    def predict(X):
        raise AssertionError("model must not be called for an empty batch")

    out = scoring.score_driver_side(spark, sdf, predict, output_cols=scoring.AFIB_OUTPUT_COLS,
                                    ecg_cols=ecg_cols, n_samples=5,
                                    model_info_col="afib_model_info", model_info="x")
    assert out.count() == 0
    assert {"afib_predictions_0", "afib_predictions_1", "afib_model_info"} <= set(out.columns)


# --- score_spark_udf (Spark workers, unused by the jobs) vs score_driver_side, same outputs ---

def _arr_predict(X):
    """Stand-in ARR model output: depends on lead order and sample order, so a wrong reshape shows."""
    prob = (X[:, 0, 0] + 2 * X[:, 11, -1]) / 100.0
    return pd.DataFrame({"arr_prob": prob.astype("float32"), "arr_flag": (prob >= 0.3).astype("float32")})


def _afib_predict(X):
    p = np.clip(X[:, 6, 1] / 50.0, 0.0, 1.0)
    return np.stack([1 - p, p], axis=1).astype("float32")


def _udf_like_mlflow(predict, result_type, n_leads=12, n_samples=5):
    """What mlflow.pyfunc.spark_udf builds for a (-1, n_leads, n_samples) tensor signature: an
    iterator pandas UDF that reshapes each flattened array (C order) and returns ``result_type``."""
    from pyspark.sql.functions import pandas_udf

    def batches(flat):
        return np.vstack(flat.tolist()).reshape(-1, n_leads, n_samples).astype("float32")

    if result_type.startswith("array<"):
        def predict_array(it: Iterator[pd.Series]) -> Iterator[pd.Series]:
            for flat in it:
                yield pd.Series(np.asarray(predict(batches(flat)), dtype="float64").tolist())
        return pandas_udf(predict_array, result_type)

    def predict_struct(it: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
        for flat in it:
            yield predict(batches(flat)).astype("float64")
    return pandas_udf(predict_struct, result_type)


SCORERS = {
    "arr": (_arr_predict, scoring.arr_columns, scoring.arr_udf_columns, scoring.ARR_UDF_RESULT_TYPE,
            scoring.ARR_OUTPUT_COLS),
    "afib": (_afib_predict, scoring.afib_columns, scoring.afib_udf_columns, scoring.AFIB_UDF_RESULT_TYPE,
             scoring.AFIB_OUTPUT_COLS),
}


@pytest.mark.parametrize("model", sorted(SCORERS))
def test_score_spark_udf_gives_the_same_table_as_driver_side(spark, model):
    pickle_by_value_for_spark_workers(__name__)
    predict, columns, udf_columns, result_type, output_cols = SCORERS[model]
    sdf, ecg_cols = _batch_sdf(spark, 7)
    common_args = dict(output_cols=output_cols, ecg_cols=ecg_cols, model_info_col="model_info",
                       model_info="models:/m/3 (@champion)")
    driver = scoring.score_driver_side(spark, sdf, lambda X: columns(predict(X)), n_samples=5, **common_args)
    udf = scoring.score_spark_udf(sdf, _udf_like_mlflow(predict, result_type), udf_columns,
                                  rows_per_partition=3, **common_args)
    assert udf.dtypes == driver.dtypes                                   # same columns, order, types
    assert [c for c, _ in udf.dtypes] == ["pseudo_patient_id", "age_years", *ecg_cols, *output_cols, "model_info"]
    by_id = lambda df: df.orderBy("pseudo_patient_id").toPandas()        # noqa: E731
    pd.testing.assert_frame_equal(by_id(udf).drop(columns=ecg_cols), by_id(driver).drop(columns=ecg_cols))
    assert udf.rdd.getNumPartitions() == 3                               # 7 rows, at most 3 per partition


def test_score_spark_udf_empty_input_and_wrong_columns(spark):
    pickle_by_value_for_spark_workers(__name__)
    sdf, ecg_cols = _batch_sdf(spark, 0)
    out = scoring.score_spark_udf(sdf, _udf_like_mlflow(_afib_predict, scoring.AFIB_UDF_RESULT_TYPE),
                                  scoring.afib_udf_columns, output_cols=scoring.AFIB_OUTPUT_COLS,
                                  ecg_cols=ecg_cols, model_info_col="afib_model_info", model_info="x")
    assert out.count() == 0 and out.columns[-3:] == ["afib_predictions_0", "afib_predictions_1", "afib_model_info"]
    with pytest.raises(ValueError, match="expected"):
        scoring.score_spark_udf(sdf, lambda col: col, scoring.afib_udf_columns, output_cols=scoring.ARR_OUTPUT_COLS,
                                ecg_cols=ecg_cols, model_info_col="m", model_info="x")


# --- Arrow batch cap: at most rows_per_partition ECGs per model call ----------------------

@pytest.fixture(autouse=True)
def _unset_arrow_batch_cap(request):
    """score_spark_udf sets the cap on the session; each test starts the shared session without it."""
    yield
    if "spark" in request.fixturenames:
        request.getfixturevalue("spark").conf.unset(scoring.ARROW_BATCH_ROWS_CONF)


def test_arrow_batch_cap_limits_the_rows_of_each_udf_call(spark):
    """One partition of 10 rows: once capped at 3 rows, an iterator pandas UDF (spark_udf calls the
    model once per such batch) never gets more than 3 rows, whatever the partition size."""
    pytest.importorskip("pyarrow")
    from pyspark.sql.functions import pandas_udf

    pickle_by_value_for_spark_workers(__name__)

    def batch_rows(it: Iterator[pd.Series]) -> Iterator[pd.Series]:
        for ids in it:
            yield pd.Series([len(ids)] * len(ids))

    sdf = spark.range(10).coalesce(1)

    def largest_batch():
        return sdf.select(pandas_udf(batch_rows, "long")("id").alias("n")).agg({"n": "max"}).collect()[0][0]

    assert largest_batch() == 10                                         # uncapped: one batch
    assert scoring.cap_arrow_batch_rows(spark, 3) is True
    assert spark.conf.get(scoring.ARROW_BATCH_ROWS_CONF) == "3"
    assert largest_batch() == 3


def test_arrow_batch_cap_refused_by_the_compute_is_reported_not_raised():
    """Serverless lets jobs set only a few listed Spark properties: a setting that raises or does
    not take effect gives False, and scoring falls back to the partition bound."""
    errors = pytest.importorskip("pyspark.errors")

    class Conf:
        def __init__(self, behaviour):
            self.behaviour, self.values = behaviour, {}

        def set(self, key, value):
            if self.behaviour == "raise":
                raise errors.AnalysisException("not settable on this compute")
            if self.behaviour == "apply":
                self.values[key] = value

        def get(self, key):
            return self.values.get(key, "10000")

    for behaviour, applied in (("apply", True), ("raise", False), ("ignore", False)):
        assert scoring.cap_arrow_batch_rows(SimpleNamespace(conf=Conf(behaviour)), 256) is applied, behaviour


def test_score_spark_udf_caps_arrow_batches_and_reuses_a_known_count(spark, monkeypatch):
    pickle_by_value_for_spark_workers(__name__)
    sdf, ecg_cols = _batch_sdf(spark, 7)
    udf = _udf_like_mlflow(_afib_predict, scoring.AFIB_UDF_RESULT_TYPE)
    args = dict(output_cols=scoring.AFIB_OUTPUT_COLS, ecg_cols=ecg_cols, model_info_col="m", model_info="x",
                rows_per_partition=3)
    by_id = lambda df: df.orderBy("pseudo_patient_id").drop(*ecg_cols).toPandas()   # noqa: E731

    out = scoring.score_spark_udf(sdf, udf, scoring.afib_udf_columns, **args)
    assert spark.conf.get(scoring.ARROW_BATCH_ROWS_CONF) == "3"         # the cap = rows_per_partition
    assert out.rdd.getNumPartitions() == 3                              # 7 rows counted, 3 per partition
    # A row count the caller already has is used as given (n_rows=1 -> one partition), not recounted.
    known = scoring.score_spark_udf(sdf, udf, scoring.afib_udf_columns, n_rows=1, **args)
    assert known.rdd.getNumPartitions() == 1
    pd.testing.assert_frame_equal(by_id(known), by_id(out))
    # Where the compute refuses the cap, the same table comes back on the partition bound, with a warning.
    monkeypatch.setattr(scoring, "cap_arrow_batch_rows", lambda spark, rows: False)
    with pytest.warns(UserWarning, match="maxRecordsPerBatch cannot be set on this compute"):
        refused = scoring.score_spark_udf(sdf, udf, scoring.afib_udf_columns, **args)
    pd.testing.assert_frame_equal(by_id(refused), by_id(out))


@pytest.mark.parametrize("model", sorted(SCORERS))
def test_real_mlflow_spark_udf_matches_driver_side(spark, model, tmp_path):
    """The real mlflow.pyfunc.spark_udf on a pyfunc with the notebooks' signature shape
    (-1, 12, n_samples) float32: it accepts the flattened lead-order array, reshapes it the way
    scoring.ecg_tensor does, and the result types give the driver-side table. Skipped without mlflow."""
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("pyarrow")
    from mlflow.models import infer_signature

    pickle_by_value_for_spark_workers(__name__)

    predict, columns, udf_columns, result_type, output_cols = SCORERS[model]

    class Model(mlflow.pyfunc.PythonModel):           # defined here, so cloudpickle stores it by value
        def predict(self, context, model_input):
            return predict(model_input)

    example = np.arange(12 * 5, dtype="float32").reshape(1, 12, 5)
    path = str(tmp_path / model)
    mlflow.pyfunc.save_model(path, python_model=Model(), signature=infer_signature(example, predict(example)),
                             input_example=example)
    sdf, ecg_cols = _batch_sdf(spark, 5)
    common_args = dict(output_cols=output_cols, ecg_cols=ecg_cols, model_info_col="model_info", model_info="v1")
    loaded = mlflow.pyfunc.load_model(path)
    driver = scoring.score_driver_side(spark, sdf, lambda X: columns(loaded.predict(X)), n_samples=5, **common_args)
    # env_manager="local": this checks the real spark_udf's input reshaping and result types; the
    # notebooks' "uv" (scoring.UDF_ENV_MANAGER) only changes which environment runs the model, and
    # rebuilding one here would download the model's requirements.
    real_udf = mlflow.pyfunc.spark_udf(spark, path, result_type=result_type, env_manager="local")
    udf = scoring.score_spark_udf(sdf, real_udf, udf_columns, **common_args)
    assert udf.dtypes == driver.dtypes
    by_id = lambda df: df.orderBy("pseudo_patient_id").drop(*ecg_cols).toPandas()   # noqa: E731
    pd.testing.assert_frame_equal(by_id(udf), by_id(driver))
