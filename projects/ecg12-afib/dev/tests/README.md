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
| `test_notebooks.py` | The workflow notebooks' code, read but not run (required catalog parameter, every MLflow run in the `mlflow_experiment` experiment, driver-side batch scoring, the triage history slice, tag values, 05_'s single AI Gateway call with inference tables and usage tracking, 05_'s endpoint updates (run with a stand-in for Model Serving: no update when the endpoint already serves the champion, a failed earlier update settles at once and is sent again, an update that fails stops the task with Model Serving's message, an update that keeps the config version number settles on its own result, one that fails before any poll sees it in progress stops at once when its pending config differs in anything from the one left before the update (a new `start_time`, Model Serving's timestamp, compared only with the earlier one, never with the notebook's clock; or no pending config before), without reading the events; a failure identical to the earlier one (no or the same `start_time`) stops after `UNCLEAR_LIMIT_S` with a message pointing at the events, never claimed as a failure and never the 3000 s timeout; a poll showing the new pending config next to the earlier UPDATE_FAILED is not a failure, and the update completes; an update that keeps showing progress (in progress, its served model being created, or that mixed state) for far longer than `UNCLEAR_LIMIT_S` is never stopped early, nor one whose served model is recovering or ready with the traffic not switched yet; another writer's pending config (a different model version) stops the task at once saying another update is in progress or replaced this one, never blamed on this update, a completed config that serves this update's model version with another size, scale-to-zero or traffic share stops it naming what differs and that it may have been changed in the Serving UI, not that the update did not complete (with its served entity's name, model or version left out it is unclear instead: reported as changed once a later poll is whole, could not tell when none is), while one whose served entity leaves out its name, model or version is inconclusive (waited for while in progress, unclear otherwise), never taken for another writer's, and, for every field the wait reads (config version, start time, served-entity name, model and version of the pending config left before the update, at the read and in later polls; the current config's version when deploy() reads it; the same plus the served model's state for the update's pending config; config version, served-entity name, model, version, size, scale-to-zero, traffic config, route name and traffic share of the newer current config), leaving it out never makes a failure, another writer's update or did not complete: the update completes once a later poll is whole, or the task stops as could not tell; a poll with no endpoint state counts as unclear, not progress; both endpoints are polled under one deadline, so a failure of the second shows at the first round while the first is still in progress, and two updates take as long as the slower one; a first poll that comes before a create or an update shows in progress keeps waiting), 05_'s ownership guard of an existing endpoint, fail safe, in its own cell and in deploy() on the snapshot the update is sent from, which narrows but does not close the window before the update (another model: another deployment's; an unknown current config, or a pending config whose models are left out or empty, even next to this deployment's model: could not tell, never an update), 05_'s deployment-path report and `deployment` tag before the smoke queries (run with stand-ins: a path other than express warns and never fails the task, an endpoint whose served entity is left out is unknown, and so does a `deployment` or `removeAfter` tag call that fails), every notebook's exact `%pip` lines (`PIP_LINES` in `conftest.py`) before any code runs, and no Python restart in a notebook that installs nothing, both models logged with the exact `MODEL_PIP_REQUIREMENTS`), plus a pickle-and-load check of the training notebooks' inline model code (its weights loaded with `weights_only=True`, which refuses a weights file holding anything but tensors) |
| `test_setup_flow_local.py` | A local run of the SETUP notebooks `01_` → `02_` + `03_` → `04_` with stand-ins for Unity Catalog and MLflow (every run in the one `mlflow_experiment`, deployed as a service principal with no home folder), including a second deployment's `01_` in the same schema stopping before it writes, and a failed gate leaving the re-checked champion's tags unchanged |
| `test_daily_flow_local.py` | A local run of the DAILY notebooks `06_` → `07_` → `08_` (one date twice, then a second date; before SETUP has recorded an owner, and as another deployment, `06_` stops before writing) with the same kind of stand-ins, then every query of the committed dashboard on the tables they wrote and on stand-in payload tables with one scored request and one app warm-up ping per endpoint (ANSI mode, as on a SQL warehouse; the pings stay out of the live tiles) |
| `test_deployment_path.py` | `common/deployment_path.py`: express, standard (container image build) or unknown from a served model's build log and the endpoint's events, with the messages Model Serving returned for this demo's endpoints and made-up timestamps (express with a completed preparation, express reusing an environment with only its build log's "prepared ahead of time" line, a finished container image build also after express started, unknown for a preparation or an image build that started and has not finished (also after an earlier attempt's finished one: a completion counts only at or after the latest start; done-only events still count) and for a build log that says express is still preparing with no build in the events, and for events holding a build that names no served entity, neither, and only the entity's latest preparation or image build by config version: an image build from an earlier attempt does not make a later express preparation standard, and, with no config version on any event, the latest build by start time; unknown when the events cannot say which build came last: a reused config version, a config version on some events only, or without config versions a missing timestamp or two kinds of build starting at the same time) |
| `test_repo_consistency.py` | Single source of truth and portability: the app's copy of `common/`, the bundle's variables (required catalog and warehouse, `mlflow_experiment`, `app_name`, `embed_credentials`), targets (`no_app` without the app, `full` with it and its environment), the `variable-overrides.example.json` template (declared variables only, both required ones, `<...>` placeholders only, no variable with a default), the dashboard resource, `deploy.sh` (run on a copy with stand-ins for the CLI: a CLI older than 0.286.0 stops it in step 1; a profile `workspace_id` of another workspace stops it in step 1, before the app build, while a matching one or none goes on and another failure of that check only warns; a failed plan prints the fresh-clone hint and the step's summary and stops before the deploy; the script's own checks and lookups use the bundle's resolved values, so a value only in the overrides file is the one it uses, a `BUNDLE_VAR_<name>` wins over the file and a flag over both, and a validate warning on stderr stays visible without changing the values read; with `--run-daily` step 6 republishes the dashboard once, right after DAILY and before the app's steps (an app step failing afterwards leaves it republished), with the resolved warehouse and the resolved `embed_credentials` in one explicit JSON body, and that body, run by the installed CLI against a local stand-in workspace, carries `embed_credentials` true and false as given, a failed republish, or a `bundle summary` that fails or names no dashboard id, only warns with the command to run by hand (the summary command with the script's own bundle arguments, `--var` included), and without `--run-daily` it only prints the command; no catalog, or a schema the jobs would refuse, stops it before the plan; a flag that takes a value given none stops it with the flag's name, and a bad `--catalog`, `--schema`, `--warehouse-id`, `--endpoint-prefix` or `--mlflow-experiment` stops it with the same check before any CLI call and before the app build), the app's bounded endpoint calls, warm-up route, UI time limit, not-for-clinical-use notices and the title's link to the source code, one constant, its arrow hidden from screen readers (read, not run: the test environment has no FastAPI), job task graphs and compute (every task names its serverless environment, none takes the moving default; no `%pip` line installs a version below what that environment ships, `PREINSTALLED` in `conftest.py`, from the release notes; every notebook that uses MLflow runs the models' MLflow version, the training notebooks exactly; every notebook that trains or loads the models runs their torch version; a `%pip` line's mlflow-skinny accepts the databricks-sdk, protobuf and cloudpickle the environment ships, `MLFLOW_SKINNY_REQUIRES` in `conftest.py`, from its package metadata), both jobs' `deployment_id` and the ownership check before each job's first write, SETUP's `mlflow_experiment` (no bundle-managed experiment), the `remove_after` expiry tag (jobs via `presets.tags`, endpoints via SETUP's `05_`, no date committed), schema and model / endpoint names only in config, no workspace-specific values in any project file, the committed dashboard (what the builder generates, bare table names, two sections), and the test runner (`run_tests_python311.sh` and its command below run the models' torch and cloudpickle versions) |
| `test_app_backend.py` | The app's backend: the rate-limit counts and warm-up cache its two workers share in one SQLite file (`backend/shared_state.py`; these run anywhere), and, in the app's own environment (below), the routes against a stand-in serving client: a malformed or non-finite ECG is a short 422 before any endpoint call, an endpoint error or unreadable answer is a 502 that keeps the endpoint's text out of the reply, the per-user and all-users limits of `/api/analyze`, `/api/warmup` and `/api/generate` (429 with `Retry-After`), the warm-up cache, and the not-for-clinical-use notice on every result |

## Running them

**Use Python 3.11.** The suite is verified on Python 3.11 only. pyspark 3.5.3 pickles the Python
functions it sends to its workers with its own bundled copy of cloudpickle (2.2.1), which predates
cloudpickle's support for Python 3.12 (added in cloudpickle 3.0.0); on Python 3.14 the local Spark
tests fail with a `RecursionError`. Python 3.12+ is not supported for this suite.
`run_tests_python311.sh` encodes the pin: it runs pytest in a throwaway `uv` environment
(`uv run --isolated`, nothing installed into a project or the system Python) on Python 3.11 with
every dependency pinned, and passes its arguments to pytest (default `dev/tests -q`; paths are
relative to `projects/ecg12-afib`). It needs `uv` and a Java runtime. It runs torch 2.11.0 and
cloudpickle 3.1.1, the versions the models are pickled and loaded with (`MODEL_PIP_REQUIREMENTS` in
`common/config.py`, as AI Runtime environment 6 ships them), so the local pickle-and-load check
loads them as the jobs and serving do. numpy 1.26.4 (as the app pins it) and pandas 2.2.3 are the
suite's own; the models name numpy 2.3.4 and pandas 2.3.3. With these versions it runs with no
warnings: the notebooks' `plt.show()` draws off-screen (`notebook_display.py`, the backend
`conftest.py` selects), and the deprecation warnings pyspark 3.5.3 and matplotlib 3.10.0 raise from
their own code are filtered by exact message and module (`THIRD_PARTY_WARNINGS` in `conftest.py`).

```bash
cd projects/ecg12-afib
dev/tests/run_tests_python311.sh            # = the command below
dev/tests/run_tests_python311.sh dev/tests/test_scoring_and_io.py -v
```

```bash
uv run --no-project --isolated --python 3.11 \
  --with numpy==1.26.4 --with scipy==1.11.4 --with pandas==2.2.3 \
  --with neurokit2==0.2.12 --with PyWavelets==1.4.1 --with matplotlib==3.10.0 --with pyyaml==6.0.3 \
  --with pytest==9.1.1 --with cloudpickle==3.1.1 --with torch==2.11.0 \
  --with pyspark==3.5.3 \
  python -B -m pytest -p no:cacheprovider dev/tests -q
```

The route tests of `test_app_backend.py` need the app's dependencies (FastAPI, databricks-sdk),
which this environment does not have, and the `_metadata.py` / `_version.py` that apx writes
(`app/build_from_lockfiles.sh` creates them); they skip without them. Run them in the app's locked
environment, with pytest and `httpx2` (the HTTP client of FastAPI's test client) added for the run
only:

```bash
cd projects/ecg12-afib
uv run --project app --frozen --with pytest==9.1.1 --with httpx2==2.16.2 \
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
