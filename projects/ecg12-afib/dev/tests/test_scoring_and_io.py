"""Tests for config.py, scoring.py and uc_io.py (Spark and MLflow replaced by small fakes;
score_driver_side also on a local Spark session)."""
import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from common import config, scoring, uc_io

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
    assert config.resolve_run_date("") == dt.datetime.now(dt.UTC).date()
    assert config.resolve_run_date(None) == dt.datetime.now(dt.UTC).date()
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


# Names that must never reach SQL unquoted: they could end or extend the statement they are put into.
BAD_NAMES = ("a-b", "a.b", "a b", "a`b", "a'b", "a;DROP TABLE x", "ä", "x" * 256)


def test_catalog_schema_and_object_names_are_plain_identifiers():
    """Every name is checked before it is put into SQL or a model name: letters, digits and
    underscores, at most 255 characters (Unity Catalog's limit)."""
    assert config.check_identifier("Main_1", "catalog") == "Main_1"
    assert config.uc_name(config.ARR_MODEL_NAME, "c_1", "s_2") == "c_1.s_2.bio_ecg12_arr"
    assert config.check_uc_name("c.s.t") == "c.s.t"
    config.check_identifier("x" * 255, "catalog")
    for bad in BAD_NAMES:
        with pytest.raises(ValueError, match="catalog_name"):
            config.uc_location(bad, "s")
        with pytest.raises(ValueError, match="schema_name"):
            config.uc_location("c", bad)
        for parts in ((bad, "c", "s"), ("t", bad, "s"), ("t", "c", bad)):
            with pytest.raises(ValueError, match="letters, digits and underscores"):
                config.uc_name(*parts)
        with pytest.raises(ValueError):
            config.check_uc_name(f"c.s.{bad}")
    for not_three_levels in ("t", "s.t", "a.b.c.d", ""):
        with pytest.raises(ValueError, match="three-level"):
            config.check_uc_name(not_three_levels)
    with pytest.raises(ValueError, match="catalog"):
        config.daily_table(config.BATCH_TABLE_PREFIX, dt.date(2026, 10, 29), "c-1")
    # Every name the notebooks build from config passes.
    d = dt.date(2026, 10, 29)
    for name in (config.TRAINING_TABLE, config.TRIAGE_HISTORY_TABLE, config.INFERENCE_TABLE, config.DEPLOYMENT_TABLE,
                 config.ARR_MODEL_NAME, config.AFIB_MODEL_NAME, config.payload_table_name(config.ARR_MODEL_NAME)):
        config.uc_name(name, "c")
    config.daily_table(config.TRIAGED_TABLE_PREFIX, d, "c", "s")


def test_deployment_id_is_required_and_resolved():
    """The jobs pass <workspace id>:<bundle root path>; empty or an unresolved reference is refused."""
    assert (config.DEPLOYMENT_ID_PARAM, config.DEPLOYMENT_TABLE) == ("deployment_id", "ecg12_afib_deployment")
    value = "1234:/Workspace/Users/tester/.bundle/biosignals-ecg12-afib/no_app"
    assert config.resolve_deployment_id(f"  {value} ") == value
    for bad in ("", "  ", None, "{{workspace.id}}:/Workspace/x", "1234:${workspace.root_path}"):
        with pytest.raises(ValueError, match="deployment_id"):
            config.resolve_deployment_id(bad)


def test_mlflow_experiment_is_required_and_resolved():
    """The SETUP job passes an absolute workspace path inside a folder; empty, relative, not in a
    folder, a folder (trailing /) or an unresolved reference is refused, so no run falls back to the
    notebook's own experiment."""
    assert config.MLFLOW_EXPERIMENT_PARAM == "mlflow_experiment"
    value = "/Users/tester/biosignals-ecg12-afib"
    assert config.resolve_mlflow_experiment(f"  {value} ") == value
    assert config.resolve_mlflow_experiment("/Shared/ecg12") == "/Shared/ecg12"
    for bad in ("", "  ", None, "biosignals-ecg12-afib", "/biosignals-ecg12-afib", "/Users/tester/", "{{job.id}}",
                "/Users/${workspace.current_user.userName}/biosignals-ecg12-afib"):
        with pytest.raises(ValueError, match="mlflow_experiment"):
            config.resolve_mlflow_experiment(bad)


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


# --- uc_io -------------------------------------------------------------------------------

# The save modes the Spark Connect client (serverless) can send for ``DataFrameWriter.mode``. It
# refuses any other spelling, classic Spark's synonyms included ("errorifexists", "default"), with
# UNSUPPORTED_OPERATION (pyspark/sql/connect/plan.py, WriteOperation), so the fake writers do too.
CONNECT_SAVE_MODES = ("append", "overwrite", "error", "ignore")


def _connect_save_mode(mode):
    if mode.lower() not in CONNECT_SAVE_MODES:
        raise ValueError(f"[UNSUPPORTED_OPERATION] {mode} is not supported.")
    return mode.lower()


class FakeWriter:
    def __init__(self, log):
        self.log = log

    def format(self, fmt):
        self.log.append(("format", fmt))
        return self

    def mode(self, mode):
        self.log.append(("mode", _connect_save_mode(mode)))
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


def test_create_table_creates_only_with_a_save_mode_spark_connect_accepts():
    """ErrorIfExists, spelled "error": serverless (Spark Connect) refuses "errorifexists"."""
    log = []
    uc_io.create_table(FakeSpark(), _df(log), "c.s.t")
    assert log == [("format", "delta"), ("mode", "error"), ("saveAsTable", "c.s.t")]
    with pytest.raises(ValueError, match="UNSUPPORTED_OPERATION"):
        FakeWriter([]).mode("errorifexists")


def test_table_version_and_time_travel_read():
    class HistorySpark(FakeSpark):
        def sql(self, stmt):
            self.sql_log.append(stmt)
            return SimpleNamespace(collect=lambda: [{"version": 4}]) if stmt.startswith("DESCRIBE") else stmt

    spark = HistorySpark()
    assert uc_io.table_version(spark, "c.s.training_data_12l") == 4
    assert uc_io.read_table_version(spark, "c.s.training_data_12l", 4) == "SELECT * FROM c.s.training_data_12l VERSION AS OF 4"
    assert spark.sql_log[0] == "DESCRIBE HISTORY c.s.training_data_12l LIMIT 1"
    spark.sql = lambda stmt: SimpleNamespace(collect=list)
    with pytest.raises(ValueError, match="no Delta history"):
        uc_io.table_version(spark, "c.s.empty")


def test_sql_helpers_refuse_names_that_are_not_plain_identifiers():
    """uc_io puts table names into SQL: each is checked first, so nothing is sent for a bad name."""
    spark = FakeSpark(["batch_date"])
    for bad in ("c.s.t;DROP TABLE x", "c.s.t VERSION AS OF 1 --", "c.s", "`c`.s.t"):
        with pytest.raises(ValueError):
            uc_io.overwrite_table(spark, _df([]), bad)
        with pytest.raises(ValueError):
            uc_io.create_table(spark, _df([]), bad)
        with pytest.raises(ValueError):
            uc_io.table_version(spark, bad)
        with pytest.raises(ValueError):
            uc_io.read_table_version(spark, bad, 1)
        with pytest.raises(ValueError):
            uc_io.write_date_slice(spark, _df([]), bad, dt.date(2026, 10, 29))
        with pytest.raises(ValueError):
            uc_io.check_schema_owner(spark, bad, "d")
    assert spark.sql_log == []


ME = "1234:/Workspace/Users/tester/.bundle/biosignals-ecg12-afib/no_app"
OTHER = "1234:/Workspace/Users/someone/.bundle/biosignals-ecg12-afib/full"
OWNER_TABLE = f"c.s.{config.DEPLOYMENT_TABLE}"


class OwnerSpark:
    """Fake Spark for the ownership record: ``rows`` None = no table yet. ``race`` = another
    deployment's record lands between the existence check and this deployment's create. The real
    ``uc_io.create_table`` writes through ``OwnerWriter``; ``created`` lists every write that
    changed the record."""

    def __init__(self, rows=None, columns=(config.DEPLOYMENT_ID_COL, "claimed_at"), race=None, fail=None):
        self.rows, self.columns, self.race, self.fail = rows, list(columns), race, fail
        self.created = []
        self.catalog = SimpleNamespace(tableExists=lambda table: self.rows is not None)

    def table(self, name):
        assert name == OWNER_TABLE, name
        distinct = sorted({(row[0],) for row in self.rows})
        return SimpleNamespace(columns=self.columns, select=lambda col: SimpleNamespace(
            distinct=lambda: SimpleNamespace(collect=lambda: distinct)))

    def createDataFrame(self, data, schema):
        return SimpleNamespace(data=data, schema=schema, write=OwnerWriter(self, data, schema))

    def sql(self, statement):
        raise AssertionError(f"the ownership check sends no SQL: {statement}")


class OwnerWriter:
    """``df.write`` of an OwnerSpark DataFrame: ``saveAsTable`` with Spark's semantics for each
    save mode Spark Connect accepts, so only ErrorIfExists keeps another deployment's record."""

    def __init__(self, spark, data, schema):
        self.spark, self.data, self.schema = spark, list(data), schema
        self.save_mode = "error"                       # Spark's default: ErrorIfExists

    def format(self, fmt):
        assert fmt == "delta", fmt
        return self

    def mode(self, mode):
        self.save_mode = _connect_save_mode(mode)
        return self

    def saveAsTable(self, table):
        spark = self.spark
        if spark.fail is not None:
            raise spark.fail
        if spark.race is not None:                     # the other deployment's create came first
            spark.rows = [(spark.race, None)]
        if spark.rows is not None:
            if self.save_mode == "error":
                raise RuntimeError(f"[TABLE_OR_VIEW_ALREADY_EXISTS] {table}")
            if self.save_mode == "ignore":
                return
        spark.rows = (spark.rows or []) + self.data if self.save_mode == "append" else self.data
        spark.created.append((table, self.save_mode, self.data, [f.name for f in self.schema.fields]))


def test_claim_schema_records_the_first_deployment_and_refuses_others():
    pytest.importorskip("pyspark.sql.types")
    spark = OwnerSpark()
    assert uc_io.claim_schema(spark, OWNER_TABLE, ME) is True                    # first SETUP: recorded
    ((table, mode, data, columns),) = spark.created
    assert table == OWNER_TABLE and columns == [config.DEPLOYMENT_ID_COL, "claimed_at"] and data[0][0] == ME
    assert mode == "error"                                                        # create only, never replace
    assert data[0][1].tzinfo is not None                                          # a UTC instant
    assert uc_io.claim_schema(spark, OWNER_TABLE, ME) is False                   # re-run: already its own
    uc_io.check_schema_owner(spark, OWNER_TABLE, ME)
    assert len(spark.created) == 1
    for call in (lambda: uc_io.claim_schema(spark, OWNER_TABLE, OTHER),
                 lambda: uc_io.check_schema_owner(spark, OWNER_TABLE, OTHER)):
        with pytest.raises(uc_io.SchemaOwnershipError, match="belongs to another deployment") as error:
            call()
        assert ME in str(error.value) and "one schema per deployment" in str(error.value)
    assert spark.rows == [(ME, data[0][1])] and len(spark.created) == 1          # never replaced


def test_claim_schema_loses_a_race_to_another_deployment_and_stops():
    pytest.importorskip("pyspark.sql.types")
    spark = OwnerSpark(race=OTHER)
    with pytest.raises(uc_io.SchemaOwnershipError, match="belongs to another deployment"):
        uc_io.claim_schema(spark, OWNER_TABLE, ME)
    assert spark.rows == [(OTHER, None)] and spark.created == []                  # the winner's record kept
    # A create that fails for another reason (no table appeared) is not hidden.
    with pytest.raises(PermissionError):
        uc_io.claim_schema(OwnerSpark(fail=PermissionError("no CREATE TABLE")), OWNER_TABLE, ME)


@pytest.mark.parametrize("spark, match", [
    (OwnerSpark(), "no deployment owns it yet"),                                      # DAILY before SETUP
    (OwnerSpark(rows=[]), "records no deployment"),
    (OwnerSpark(rows=[(ME,), (OTHER,)]), "records several deployments"),
    (OwnerSpark(rows=[("x",)], columns=("something_else",)), "not written by this demo"),
])
def test_check_schema_owner_refuses_every_schema_that_is_not_only_this_deployments(spark, match):
    with pytest.raises(uc_io.SchemaOwnershipError, match=match):
        uc_io.check_schema_owner(spark, OWNER_TABLE, ME)


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
