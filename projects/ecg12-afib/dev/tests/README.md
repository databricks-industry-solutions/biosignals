# dev/tests/: optional local checks (not deployed)

**Optional pre-deployment checks.** These `pytest` tests (pytest is a Python test runner) run on
your laptop only (Python 3.11; nothing runs them automatically and they are not deployed: bundle
sync excludes `dev/**`). They check code logic: the 12-lead formulas, the patient record,
training-data splits, the validation gate promoting both models or neither, batch-scoring output,
the dashboard's queries on local Spark, the bundle configuration, and that names come from config.

They run nothing in a workspace, so they cannot catch workspace-specific problems, such as a Unity
Catalog tag policy on a key the notebooks write (`GOVERNED_TAG_VALUES` in `conftest.py` lists the
values the notebooks may write; add your workspace's policies there) or what the serverless
environments install. **The real pre-deployment check is a live run:** the SETUP job, then one
DAILY run, before unpausing the DAILY schedule (see [`../../docs/deploy.md`](../../docs/deploy.md)).

## What each file checks

| File | Checks |
|---|---|
| `test_ecg_signal_gen.py` | The shared 12-lead generator (`common/ecg_signal_gen.py`), including the derived-lead formulas and the class counts of a batch |
| `test_patient_info.py` | The synthetic, non-PII patient record (`common/patient_info.py`) |
| `test_training.py` | Training data, training loop and layer table, sample-signal plots (`common/training_data.py`, `training.py`, `ecg_plots.py`) |
| `test_evaluation.py` | Metrics, the gate's threshold check and the evaluation tables, curves and figures (`common/evaluation.py`, `eval_plots.py`) |
| `test_scoring_and_io.py` | `common/config.py` (no workspace values, the required catalog, catalog / schema / table names that are plain identifiers before any SQL, the required `deployment_id` and `mlflow_experiment`), `scoring.py` and `uc_io.py` against small fakes (including the schema-ownership record: first deployment recorded, any other refused, a lost race stops), and driver-side scoring on a local Spark session |
| `test_tracking.py` | `common/tracking.py`: the SETUP job's MLflow experiment becomes active for a service principal with no home folder (the folder is created first), in a new `/Shared` folder, and when `02_` and `03_` create it at once; other errors (folder or experiment creation refused) are raised, not swallowed |
| `test_notebooks.py` | The workflow notebooks' code, read but not run (required catalog parameter, every MLflow run in the `mlflow_experiment` experiment, driver-side batch scoring, the triage history slice, tag values, 05_'s single AI Gateway call with inference tables and usage tracking), plus a pickle-and-load check of the training notebooks' inline model code |
| `test_setup_flow_local.py` | A local run of the SETUP notebooks `01_` → `02_` + `03_` → `04_` with stand-ins for Unity Catalog and MLflow (every run in the one `mlflow_experiment`, deployed as a service principal with no home folder), including a second deployment's `01_` in the same schema stopping before it writes, and a failed gate leaving the re-checked champion's tags unchanged |
| `test_daily_flow_local.py` | A local run of the DAILY notebooks `06_` → `07_` → `08_` (one date twice, then a second date; before SETUP has recorded an owner, and as another deployment, `06_` stops before writing) with the same kind of stand-ins, then every query of the committed dashboard on the tables they wrote and on stand-in payload tables with one scored request and one app warm-up ping per endpoint (ANSI mode, as on a SQL warehouse; the pings stay out of the live tiles) |
| `test_repo_consistency.py` | Single source of truth and portability: the app's copy of `common/`, the bundle's variables (required catalog and warehouse, `mlflow_experiment`, `app_name`, `embed_credentials`), targets (`no_app` without the app, `full` with it and its environment), the dashboard resource, `deploy.sh`, the app's bounded endpoint calls, warm-up route, UI time limit and not-for-clinical-use notices (read, not run: the test environment has no FastAPI), job task graphs and compute, both jobs' `deployment_id` and the ownership check before each job's first write, SETUP's `mlflow_experiment` (no bundle-managed experiment), the `remove_after` expiry tag (jobs via `presets.tags`, endpoints via SETUP's `05_`, no date committed), schema and model / endpoint names only in config, no workspace-specific values in any project file, and the committed dashboard (what the builder generates, bare table names, two sections) |
| `test_app_backend.py` | The app's backend: the rate-limit counts and warm-up cache its two workers share in one SQLite file (`backend/shared_state.py`; these run anywhere), and, in the app's own environment (below), the routes against a stand-in serving client: a malformed or non-finite ECG is a short 422 before any endpoint call, an endpoint error or unreadable answer is a 502 that keeps the endpoint's text out of the reply, the per-user and all-users limits (429 with `Retry-After`), the warm-up cache, and the not-for-clinical-use notice on every result |

## Running them

**Use Python 3.11.** The suite is verified on Python 3.11 only. pyspark 3.5.3 pickles the Python
functions it sends to its workers with its own bundled copy of cloudpickle (2.2.1), which predates
cloudpickle's support for Python 3.12 (added in cloudpickle 3.0.0); on Python 3.14 the local Spark
tests fail with a `RecursionError`. Python 3.12+ is not supported for this suite.
`run_tests_python311.sh` encodes the pin: it runs pytest in a throwaway `uv` environment
(`uv run --isolated`, nothing installed into a project or the system Python) on Python 3.11 with
every dependency pinned, and passes its arguments to pytest (default `dev/tests -q`; paths are
relative to `projects/ecg12-afib`). It needs `uv` and a Java runtime.

```bash
cd projects/ecg12-afib
dev/tests/run_tests_python311.sh            # = the command below
dev/tests/run_tests_python311.sh dev/tests/test_scoring_and_io.py -v
```

```bash
uv run --no-project --isolated --python 3.11 \
  --with numpy==1.26.4 --with scipy==1.11.4 --with pandas==2.2.3 \
  --with neurokit2==0.2.12 --with PyWavelets==1.4.1 --with matplotlib==3.10.0 --with pyyaml==6.0.3 \
  --with pytest==9.1.1 --with cloudpickle==3.0.0 --with torch==2.9.0 \
  --with pyspark==3.5.3 \
  python -B -m pytest -p no:cacheprovider dev/tests -q
```

The route tests of `test_app_backend.py` need the app's dependencies (FastAPI, databricks-sdk),
which this environment does not have, and the `_metadata.py` / `_version.py` that apx writes
(`app/build_from_lockfiles.sh` creates them); they skip without them. Run them in the app's locked
environment, with pytest and FastAPI's test client (`httpx`) added for the run only:

```bash
cd projects/ecg12-afib
uv run --project app --frozen --with pytest==9.1.1 --with httpx==0.28.1 \
  python -B -m pytest -p no:cacheprovider dev/tests/test_app_backend.py -q
```

Without the script, `python -m pytest dev/tests` (from `projects/ecg12-afib`; `conftest.py` adds
that folder to `sys.path`) needs numpy, scipy, pandas, neurokit2 and pyyaml, and skips the tests
whose extras are missing. `torch` + `cloudpickle`: the training loop and layer table, and a check
that pickles each training notebook's inline model the way MLflow does and loads it in a process
where `common` cannot be imported, as on a serving endpoint. A local (non-Databricks-Connect)
`pyspark` + Java: driver-side scoring, reading a training split, a run of the SETUP notebooks `01_`
→ `02_` + `03_` → `04_` (tiny splits, two epochs on CPU; the gate passes and promotes both models
and logs its evaluation artifacts, then fails and promotes neither; the stand-in registry refuses
tag values `GOVERNED_TAG_VALUES` does not allow, as Unity Catalog does under a tag policy) and a run
of the DAILY notebooks `06_` → `07_` → `08_` (one date twice, then the next date; driver-side
scoring) followed by every dashboard query, all with stand-ins for Unity Catalog and MLflow. No
notebook is executed on Databricks by the tests, and `05_` (endpoints) is not run.
