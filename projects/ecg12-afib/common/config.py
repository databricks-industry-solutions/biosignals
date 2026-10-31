"""Single source of truth for the 12-lead ECG/AFib demo.

Every notebook in ``workflow/``, the dashboard builder and the app backend read names from
here. The app keeps a byte-identical copy in ``app/src/ecg_demo_react/backend/common/``
(``build_from_lockfiles.sh`` fails if it drifts).

Nothing here names a workspace. The Unity Catalog location comes from the bundle variables
``catalog`` (required) and ``schema`` (default ``SCHEMA``): the jobs pass them to every notebook
as the ``catalog_name`` / ``schema_name`` parameters (widgets), and the app reads them from its
environment (``databricks.yml``). The daily notebooks also take ``run_date``.

The demo is a reference pattern, not a fixed pipeline: the catalog/schema, lead count, model
and table names, training split, training settings and validation thresholds below are the
settings to change when adapting it.
"""

from __future__ import annotations

import datetime as dt

# --- Unity Catalog (UC) location ----------------------------------------------------------
# Every table, model and inference table of this demo lives in <catalog>.<schema>. There is no
# default catalog (each workspace has its own); SCHEMA is the default schema. The SETUP job's
# first task creates the schema when it does not exist yet.
SCHEMA = "biosignals"

# --- Model-input contract -----------------------------------------------------------------
# Leads per ECG, for generation, storage (ecg1..ecgN columns) and the models' input
# (-1, N_LEADS, 2500). Change it only together with a retrain of both models.
N_LEADS = 12

# --- Models (UC registered models) and their serving endpoints ----------------------------
# Each endpoint has the same name as the model it serves. Endpoint names are workspace-wide, so
# one workspace holds one deployment of this project; for a second copy, change these names.
ARR_MODEL_NAME = "bio_ecg12_arr"    # triage: abnormal rhythm -> arr_prob, arr_flag
AFIB_MODEL_NAME = "bio_ecg12_afib"  # classify (flagged only): [p_not_afib, p_afib]
ARR_ENDPOINT = ARR_MODEL_NAME
AFIB_ENDPOINT = AFIB_MODEL_NAME
# Their AI Gateway inference tables (enabled by 05_, read by the dashboard's live section) are
# payload_table_name(endpoint) below: bio_ecg12_arr_payload and bio_ecg12_afib_payload.
# Model-version aliases, lowercase because Unity Catalog stores aliases lowercase.
# 02_/03_ register every new version as CHALLENGER; only 04_validate_and_promote moves
# CHAMPION, and only when the challenger passes VALIDATION_THRESHOLDS. Serving (05_) and batch
# scoring (07_, 08_) use CHAMPION.
CHALLENGER_ALIAS = "challenger"
CHAMPION_ALIAS = "champion"
ARR_FLAG_THRESHOLD = 0.5            # arr_flag = 1.0 when arr_prob >= this
AFIB_THRESHOLD = 0.5                # "AFib" when p_afib > this

# --- Spark-worker scoring (scoring.score_spark_udf; not used by the notebooks yet) -----------
# The daily notebooks score on the driver. For the Spark-worker path (README, "Future work"): on
# serverless compute a Python UDF gets 1 GB of memory, and one ECG is 12 x 2500 float32 values
# (120 KB), so each model call gets at most this many ECGs (~30 MB).
BATCH_UDF_ROWS_PER_PARTITION = 256

# --- Training dataset (01_generate_training_data) -----------------------------------------
# One persisted table for both models, one row per ECG, with a split column. Its own name: it
# does not start with BATCH_TABLE_PREFIX / TRIAGED_TABLE_PREFIX and is not INFERENCE_TABLE.
TRAINING_TABLE = "training_data_12l"
SPLIT_COL = "split"
SAMPLE_ID_COL = "sample_id"
TRAIN_SPLIT, VAL_SPLIT, HOLDOUT_SPLIT = "train", "val", "holdout"
# split -> (ECGs, generator seed). Each split is its own seeded draw, so every split has the
# exact class mix below. "train" fits the models, "val" gives the per-epoch validation metrics
# logged during training, and "holdout" is read only by the validation gate (04_), so the
# metrics that decide promotion come from ECGs the training notebooks never see.
TRAINING_SPLITS: dict[str, tuple[int, int]] = {
    TRAIN_SPLIT: (600, 11),
    VAL_SPLIT: (200, 99),
    HOLDOUT_SPLIT: (200, 314),
}
TRAINING_FRAC_AFIB = 0.34           # class mix of every split; the rest after AFib + Other is NSR
TRAINING_FRAC_OTHER = 0.33
TRAINING_NOISE_LEVEL = 1.0          # same acquisition noise as the daily batch and the app

# --- Training settings (02_train_arr_model, 03_train_afib_model) --------------------------
# One fixed configuration for both models, no hyperparameter search: on this synthetic data
# it already gives ~1.0 validation accuracy (see 00_overview, "Why the metrics sit near 1.0").
TRAIN_PARAMS: dict[str, int | float] = {
    "learning_rate": 1e-3,          # Adam
    "epochs": 20,
    "batch_size": 64,
    "seed": 0,                      # weight init + batch order
}
# AFib model only: label smoothing during training plus a softmax temperature at inference
# spread its probabilities out instead of snapping to ~0/1 (the 0.5 decision is unchanged).
AFIB_LABEL_SMOOTHING = 0.1
AFIB_TEMPERATURE = 2.0

# --- Validation gate (04_validate_and_promote) --------------------------------------------
# Hold-out metrics each challenger must reach before it becomes champion. "min_<metric>" is a
# floor, "max_<metric>" a ceiling; metric names are those of common/evaluation.py. Both models
# must pass, or neither is promoted. These are engineering floors for this synthetic data (a
# healthy model scores ~1.0; one that predicts a single class for everything scores ~0.67
# accuracy and 0.5 AUC), not clinical acceptance criteria.
VALIDATION_THRESHOLDS: dict[str, dict[str, float]] = {
    ARR_MODEL_NAME: {
        "min_accuracy": 0.95,
        "min_roc_auc": 0.98,
        "min_sensitivity": 0.97,    # a missed abnormal ECG never reaches the AFib model
        "min_specificity": 0.90,
    },
    AFIB_MODEL_NAME: {
        "min_accuracy": 0.95,
        "min_roc_auc": 0.98,
        "min_sensitivity": 0.95,
        "max_called_positive_rate_other": 0.05,   # sinus tachycardia must not be called AFib
    },
}
VALIDATION_MIN_ROWS = 100           # fewer hold-out rows than this fails the gate

# --- Daily batch tables ---------------------------------------------------------------------
# Per-date tables, overwritten when a date is re-run (the 12l suffix marks the 12-lead tables).
BATCH_TABLE_PREFIX = "batch_data_12l_"       # 06_generate_batch_data -> batch_data_12l_<YYYYMMDD>
TRIAGED_TABLE_PREFIX = "arr_triaged_12l_"    # 07_arr_triage          -> arr_triaged_12l_<YYYYMMDD>
# Accumulating results table (08_afib_classify), read by the dashboard and the app; re-running a
# date replaces only that date's rows (see uc_io.write_date_slice).
INFERENCE_TABLE = "dailyOrweekly_arr_afib_inference_12l"
BATCH_DATE_COL = "batch_date"
ID_COL = "pseudo_patient_id"

# --- Run date: ONE parameter shared by every daily task -----------------------------------
# The daily job passes {{job.start_time.iso_date}} (UTC) to all three tasks, so a run that
# crosses midnight still reads and writes a single date.
RUN_DATE_PARAM = "run_date"


def uc_location(catalog: str | None, schema: str | None) -> tuple[str, str]:
    """The ``catalog_name`` / ``schema_name`` parameters, checked: the catalog is required (no
    default), an empty schema means ``SCHEMA``."""
    catalog, schema = (catalog or "").strip(), (schema or "").strip() or SCHEMA
    if not catalog:
        raise ValueError("catalog_name is empty: the jobs pass the bundle's required `catalog` "
                         "variable; when running a notebook by hand, type the catalog into the widget")
    return catalog, schema


def uc_name(name: str, catalog: str, schema: str = SCHEMA) -> str:
    """Three-level UC name, e.g. ``uc_name(ARR_MODEL_NAME, catalog)``."""
    return f"{catalog}.{schema}.{name}"


def resolve_run_date(value: str | None) -> dt.date:
    """Parse the shared ``run_date`` parameter: ISO ``YYYY-MM-DD``; empty means today (UTC)."""
    value = (value or "").strip()
    if not value:
        return dt.datetime.now(dt.timezone.utc).date()
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{RUN_DATE_PARAM}={value!r} is not an ISO date (YYYY-MM-DD)") from exc


def date_tag(run_date: dt.date) -> str:
    """``YYYYMMDD`` suffix used by the per-date tables."""
    return run_date.strftime("%Y%m%d")


def daily_table(prefix: str, run_date: dt.date, catalog: str, schema: str = SCHEMA) -> str:
    """Per-date table name, e.g. ``daily_table(BATCH_TABLE_PREFIX, d, catalog)``."""
    return uc_name(f"{prefix}{date_tag(run_date)}", catalog, schema)


def payload_table_name(endpoint: str) -> str:
    """Table name of the AI Gateway inference table that ``05_serve_endpoints`` enables for
    ``endpoint``: AI Gateway names it ``<table_name_prefix>_payload`` and 05_ passes the endpoint
    name as the prefix, so ``payload_table_name(ARR_ENDPOINT)`` is ``bio_ecg12_arr_payload``."""
    return f"{endpoint}_payload"


def payload_table(endpoint: str, catalog: str, schema: str = SCHEMA) -> str:
    """Three-level name of ``payload_table_name(endpoint)``."""
    return uc_name(payload_table_name(endpoint), catalog, schema)


def ecg_columns(n_leads: int = N_LEADS) -> list[str]:
    """Stored per-lead columns ``ecg1``..``ecgN`` (order = ``ecg_signal_gen.lead_names_for``)."""
    return [f"ecg{i + 1}" for i in range(n_leads)]
