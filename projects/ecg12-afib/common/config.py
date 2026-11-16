"""Single source of truth for the 12-lead ECG/AFib demo.

Every notebook in ``workflow/``, the dashboard builder and the app backend read names from
here. The app keeps a byte-identical copy in ``app/src/ecg_demo_react/backend/common/``
(``build_from_lockfiles.sh`` fails if it drifts).

Nothing here names a workspace. The Unity Catalog location comes from the bundle variables
``catalog`` (required) and ``schema`` (default ``SCHEMA``): the jobs pass them to every notebook
as the ``catalog_name`` / ``schema_name`` parameters (widgets), and the app reads them from its
environment (``databricks.yml``). The daily notebooks also take ``run_date``, the first task of
each job takes ``deployment_id`` (see "One schema per deployment" below), and the notebooks that log
MLflow runs (02_, 03_, 04_) take ``mlflow_experiment`` (see "MLflow experiment" below).

The demo is a reference pattern, not a fixed pipeline: the catalog/schema, lead count, model
and table names, training split, training settings and validation thresholds below are the
settings to change when adapting it.
"""

from __future__ import annotations

import datetime as dt
import re

# --- Unity Catalog (UC) location ----------------------------------------------------------
# Every table, model and inference table of this demo lives in <catalog>.<schema>. There is no
# default catalog (each workspace has its own); SCHEMA is the default schema. The SETUP job's
# first task creates the schema when it does not exist yet.
SCHEMA = "biosignals"
# Catalog, schema and object names are put into SQL and model names without quoting, so each part
# must be a plain identifier: letters, digits and underscores (Unity Catalog accepts more characters,
# such as hyphens, but only inside backticks). Checked by uc_location / uc_name before any use.
_UC_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9_]+")
_UC_IDENTIFIER_MAX = 255                # Unity Catalog's limit on an object name

# --- One schema per deployment ------------------------------------------------------------
# The model and table names below are fixed, so two deployments of this bundle in one
# <catalog>.<schema> would use the same objects: one SETUP would overwrite the other's training
# data and move its @champion, and either teardown would delete what the other serves. To make
# that impossible to do by accident, the SETUP job's first task records which deployment owns
# the schema in DEPLOYMENT_TABLE (uc_io.claim_schema), and the first task of SETUP and DAILY stops
# when the schema belongs to another deployment (uc_io.check_schema_owner). The identity is the
# DEPLOYMENT_ID_PARAM job parameter, "<workspace id>:<bundle root path>" (resources/*.job.yml):
# unique per bundle deployment (user, target and workspace) and the same on every run of it.
DEPLOYMENT_TABLE = "ecg12_afib_deployment"
DEPLOYMENT_ID_PARAM = "deployment_id"
DEPLOYMENT_ID_COL = "deployment_id"

# --- MLflow experiment (tracking, not the model registry) ---------------------------------
# Every MLflow run of the SETUP job (02_, 03_ train; 04_ gate) is logged to ONE experiment, the
# MLFLOW_EXPERIMENT_PARAM job parameter from the bundle's `mlflow_experiment` variable (default
# /Users/<deployer>/biosignals-ecg12-afib). It is a plain workspace path, not a bundle resource, so
# it outlives `bundle destroy` and run history accrues across redeploys. tracking.use_experiment
# creates it on first use, and its parent folder first: an identity may have no home folder yet
# (e.g. a service principal), and MLflow does not create a missing parent folder.
# Without it each notebook would log to its own notebook experiment inside the bundle's folder,
# which `bundle destroy` deletes. The DAILY job starts no runs, so it does not take the parameter.
MLFLOW_EXPERIMENT_PARAM = "mlflow_experiment"

# --- Model-input contract -----------------------------------------------------------------
# Leads per ECG, for generation, storage (ecg1..ecgN columns) and the models' input
# (-1, N_LEADS, 2500). Change it only together with a retrain of both models.
N_LEADS = 12

# --- Models (UC registered models) ---------------------------------------------------------
# Fixed names in <catalog>.<schema>: a deployment's models are its own only because its schema
# is its own (DEPLOYMENT_TABLE above).
ARR_MODEL_NAME = "bio_ecg12_arr"    # triage: abnormal rhythm -> arr_prob, arr_flag
AFIB_MODEL_NAME = "bio_ecg12_afib"  # classify (flagged only): [p_not_afib, p_afib]
# Their AI Gateway inference tables (enabled by 05_, read by the dashboard's live section) are
# payload_table_name(model) below: bio_ecg12_arr_payload and bio_ecg12_afib_payload, also in
# <catalog>.<schema> and also fixed (the dashboard names them as bare tables).

# --- Serving endpoints --------------------------------------------------------------------
# Endpoint names are workspace-wide, unlike the models and tables, so they are configurable:
# endpoint_names(prefix) gives <prefix>_arr and <prefix>_afib, where prefix is the bundle's
# `endpoint_prefix` variable (SETUP job parameter, app environment), default ENDPOINT_PREFIX. Each
# deployment in a workspace needs its own prefix; 05_serve_endpoints refuses to update an existing
# endpoint that serves another deployment's model. With the default prefix the endpoint names
# equal the model names.
ENDPOINT_PREFIX = "bio_ecg12"
_ENDPOINT_PREFIX_PATTERN = re.compile(r"[A-Za-z0-9_-]+")   # endpoint names: alphanumerics, '-', '_'
# Model-version aliases, lowercase because Unity Catalog stores aliases lowercase.
# 02_/03_ register every new version as CHALLENGER; only 04_validate_and_promote moves
# CHAMPION, and only when the challenger passes VALIDATION_THRESHOLDS. Serving (05_) and batch
# scoring (07_, 08_) use CHAMPION.
CHALLENGER_ALIAS = "challenger"
CHAMPION_ALIAS = "champion"
ARR_FLAG_THRESHOLD = 0.5            # arr_flag = 1.0 when arr_prob >= this
AFIB_THRESHOLD = 0.5                # "AFib" when p_afib > this

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
# Accumulating tables, one slice per batch_date (re-running a date replaces only that date's rows,
# see uc_io.write_date_slice). The dashboard reads these two under fixed names:
# every screened ECG of every day with its triage call, without the ECG arrays (07_arr_triage) ...
TRIAGE_HISTORY_TABLE = "arr_triage_history_12l"
# ... and every day's flagged ECGs with the AFib call (08_afib_classify; the app links to it).
INFERENCE_TABLE = "dailyOrweekly_arr_afib_inference_12l"
BATCH_DATE_COL = "batch_date"
ID_COL = "pseudo_patient_id"

# --- Run date: ONE parameter shared by every daily task -----------------------------------
# The daily job passes {{job.start_time.iso_date}} (UTC) to all three tasks, so a run that
# crosses midnight still reads and writes a single date.
RUN_DATE_PARAM = "run_date"

# --- App: live endpoint calls ---------------------------------------------------------------
# Both endpoints scale to zero when idle (after 30 min without requests). Waking one took 89 s and
# 130 s, request to answer, when this demo was tested, so every live call from the app is bounded:
# SERVING_TIMEOUT_S is both the databricks-sdk http_timeout_seconds (one HTTP attempt) and
# retry_timeout_seconds (no new retry after this long) of the app's endpoint client. A call to a
# cold endpoint therefore gives up after about this long and the app answers 503 "waking up, try
# again" instead of waiting.
SERVING_TIMEOUT_S = 30
# While the app's Live tab is visible it pings both endpoints on load and then every
# WARMUP_INTERVAL_S, well inside the 30 min idle window, so they stay awake during a demo.
WARMUP_INTERVAL_S = 600
# client_request_id of those pings, recorded in the payload tables; the dashboard's live section
# leaves these rows out of its request tiles and counts them separately. One home for the tag:
# the app (backend/router.py) and the dashboard builder both read it here.
WARMUP_REQUEST_ID = "app-warmup"


def check_identifier(value: str, what: str) -> str:
    """``value`` if it is a plain Unity Catalog identifier (letters, digits, underscores, at most
    255 characters), else ValueError naming ``what``. Every catalog, schema and object name passes
    here before it reaches SQL, so a name can never change the statement it is put into."""
    if not (_UC_IDENTIFIER_PATTERN.fullmatch(value) and len(value) <= _UC_IDENTIFIER_MAX):
        raise ValueError(f"{what}={value!r}: this demo needs a name of letters, digits and underscores "
                         f"only (at most {_UC_IDENTIFIER_MAX} characters)")
    return value


def uc_location(catalog: str | None, schema: str | None) -> tuple[str, str]:
    """The ``catalog_name`` / ``schema_name`` parameters, checked: the catalog is required (no
    default), an empty schema means ``SCHEMA``, and both must be plain identifiers."""
    catalog, schema = (catalog or "").strip(), (schema or "").strip() or SCHEMA
    if not catalog:
        raise ValueError("catalog_name is empty: the jobs pass the bundle's required `catalog` "
                         "variable; when running a notebook by hand, type the catalog into the widget")
    return check_identifier(catalog, "catalog_name"), check_identifier(schema, "schema_name")


def uc_name(name: str, catalog: str, schema: str = SCHEMA) -> str:
    """Three-level UC name, e.g. ``uc_name(ARR_MODEL_NAME, catalog)``; every part checked by
    ``check_identifier``."""
    return ".".join(check_identifier(part, what) for part, what in
                    ((catalog, "catalog"), (schema, "schema"), (name, "name")))


def check_uc_name(full_name: str) -> str:
    """``full_name`` if it is ``<catalog>.<schema>.<name>`` of plain identifiers (as ``uc_name``
    builds it), else ValueError. For names that arrive from elsewhere, e.g. a model-version tag."""
    parts = full_name.split(".")
    if len(parts) != 3:
        raise ValueError(f"{full_name!r} is not a three-level <catalog>.<schema>.<name>")
    return uc_name(parts[2], parts[0], parts[1])


def resolve_deployment_id(value: str | None) -> str:
    """The ``deployment_id`` parameter, checked: the jobs pass ``<workspace id>:<bundle root
    path>``; empty or unresolved (a ``{{...}}`` / ``${...}`` reference) is refused, because the
    schema-ownership check needs this deployment's real identity."""
    value = (value or "").strip()
    if not value or "{{" in value or "${" in value:
        raise ValueError(f"{DEPLOYMENT_ID_PARAM}={value!r}: the jobs pass this deployment's identity "
                         "(<workspace id>:<bundle root path>); when running a notebook by hand, copy the "
                         f"value from {DEPLOYMENT_TABLE} in your schema or from a past job run's parameters")
    return value


def resolve_mlflow_experiment(value: str | None) -> str:
    """The ``mlflow_experiment`` parameter, checked: an absolute workspace path inside a folder, such
    as ``/Users/<user name>/biosignals-ecg12-afib`` (Databricks names experiments by path; the folder
    is created when missing); empty, not in a folder, or unresolved (a ``{{...}}`` / ``${...}``
    reference) is refused, so a run never falls back to the notebook's own experiment, which
    ``bundle destroy`` deletes."""
    value = (value or "").strip()
    folder, name = value.rpartition("/")[::2]
    if not (value.startswith("/") and folder and name) or "{{" in value or "${" in value:
        raise ValueError(f"{MLFLOW_EXPERIMENT_PARAM}={value!r}: the SETUP job passes the bundle's "
                         "`mlflow_experiment` variable, an absolute workspace path; when running a notebook "
                         "by hand, type e.g. /Users/<your user name>/biosignals-ecg12-afib into the widget")
    return value


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


def endpoint_names(prefix: str | None = None) -> tuple[str, str]:
    """This deployment's (triage, classify) serving endpoint names, ``<prefix>_arr`` and
    ``<prefix>_afib``; an empty prefix means ``ENDPOINT_PREFIX``."""
    prefix = (prefix or "").strip() or ENDPOINT_PREFIX
    if not _ENDPOINT_PREFIX_PATTERN.fullmatch(prefix):
        raise ValueError(f"endpoint_prefix={prefix!r}: an endpoint name can consist of alphanumeric "
                         "characters, dashes and underscores only")
    return f"{prefix}_arr", f"{prefix}_afib"


def payload_table_name(model: str) -> str:
    """Table name of the AI Gateway inference table that ``05_serve_endpoints`` enables for the
    endpoint serving ``model``: AI Gateway names it ``<table_name_prefix>_payload`` and 05_ passes
    the model name as the prefix (not the endpoint name, which varies per deployment), so
    ``payload_table_name(ARR_MODEL_NAME)`` is ``bio_ecg12_arr_payload``."""
    return f"{model}_payload"


def payload_table(model: str, catalog: str, schema: str = SCHEMA) -> str:
    """Three-level name of ``payload_table_name(model)``."""
    return uc_name(payload_table_name(model), catalog, schema)


def ecg_columns(n_leads: int = N_LEADS) -> list[str]:
    """Stored per-lead columns ``ecg1``..``ecgN`` (order = ``ecg_signal_gen.lead_names_for``)."""
    return [f"ecg{i + 1}" for i in range(n_leads)]
