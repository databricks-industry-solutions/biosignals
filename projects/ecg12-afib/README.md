# ECG/AFib demo (12-lead): Databricks App + SETUP job + DAILY job + AI/BI dashboard

A two-model ECG cascade on a synthetic **clinical 12-lead ECG** (I, II, III, aVR, aVL, aVF,
V1–V6). A cheap **arrhythmia (ARR) triage** model flags abnormal ECGs, and an **atrial
fibrillation (AFib)** classifier scores only the flagged ones. Both are small custom PyTorch
1D-CNNs (one-dimensional convolutional neural networks) with 12 input channels. The demo runs the
full model cycle: **generate training data → train → validate on a hold-out split → promote →
serve → score daily → monitor**. The same Unity Catalog (UC) models run two ways:

- **Live:** the app generates a 12-lead ECG plus a synthetic patient record and sends all 12
  leads to two Model Serving endpoints, `bio_ecg12_arr` (triage), then `bio_ecg12_afib` (classify).
  Each endpoint has the same name as the model it serves, and each call is logged to the
  endpoint's AI Gateway inference table.
- **Batch:** a daily Lakeflow job loads the same champion models from UC and scores a synthetic
  batch on the job's driver into Delta tables (it does not call the endpoints).

An AI/BI dashboard shows both, in two separate sections (batch monitoring, live serving), and the
app embeds it.

Nothing in this folder names a workspace: the catalog, the SQL warehouse and your Databricks CLI
profile are inputs (see [Configuration](#configuration)). Every name the code uses (models,
endpoints, tables, thresholds) lives in [`common/config.py`](common/config.py).

> **A reference pattern, not a fixed pipeline.** The tasks, models, catalog/schema, rhythm classes
> and lead count are all swappable: names, splits, training settings and validation thresholds live
> in `common/config.py`, the data in `common/ecg_signal_gen.py`, the model code in the training
> notebooks, and the task graph in `resources/*.job.yml`. Swap in a real ECG source or another model
> and the generate → train → gate → promote → serve → monitor cycle stays the same.

## Layout

| Path | What it is |
|---|---|
| `databricks.yml` | Bundle `biosignals-ecg12-afib`: variables, targets `no_app` (jobs + dashboard) and `full` (+ the app) |
| `deploy.sh` | Full install from a laptop (target `full`): build, validate, deploy, SETUP run, permissions, app start, embedding allowlist |
| `resources/` | `ecg12_afib_setup.job.yml`, `ecg12_afib_daily.job.yml` (the jobs), `ecg12_afib.dashboard.yml` (the dashboard) |
| `workflow/` | Notebooks `00_`–`08_` (below). Each puts `..` on `sys.path` and imports `common` |
| `common/` | Shared code, single source of truth: `config.py` (names, `N_LEADS = 12`, aliases, training split + settings, validation thresholds, run date, the catalog/schema parameter check), `ecg_signal_gen.py` (12-lead generator), `training_data.py` (the persisted training dataset), `training.py` (PyTorch training loop, layer table), `evaluation.py` (metrics, the gate's threshold check, confusion counts, ROC / precision-recall curves, per-class table), `eval_plots.py` (the gate's evaluation figures), `ecg_plots.py` (sample-signal plots), `patient_info.py` (synthetic patient record), `scoring.py` (alias → version, driver-side scoring), `uc_io.py` (re-run-safe Delta writes, Delta versions) |
| `dashboards/` | `ecg12_afib.lvdash.json` (the dashboard the bundle deploys) and `build_ecg_demo_dashboard.py`, which generates it |
| `app/` | APX app (React UI + FastAPI backend). `backend/common/` is a byte-identical copy of `common/{__init__,config,ecg_signal_gen,patient_info}.py`; the build fails if it drifts. Build with `app/build_from_lockfiles.sh`, see [`app/README.md`](app/README.md) |
| `dev/` | Developer tools, not deployed (excluded from bundle sync, outside the app wheel): `signal_variety/` checks how varied and how separable the generator's ECGs are; `tests/` holds optional local checks (see [Tests](#tests)); see [`dev/README.md`](dev/README.md) |

## Notebooks and jobs

| Notebook | Job → task | Writes | Compute |
|---|---|---|---|
| `00_overview` | — | Docs: the whole flow on one page, the 12-lead legend | — |
| `01_generate_training_data` | SETUP → `generate_training_data` | the schema if it does not exist; `training_data_12l` (`train` 600 / `val` 200 / `holdout` 200 ECGs by default) + sample plots | standard serverless |
| `02_train_arr_model` | SETUP → `train_arr_model` (after 01) | MLflow run (incl. the layer table); new version of `bio_ecg12_arr` → `@challenger` | serverless GPU (A10, AI v5) |
| `03_train_afib_model` | SETUP → `train_afib_model` (after 01, parallel with 02) | MLflow run (incl. the layer table); new version of `bio_ecg12_afib` → `@challenger` | serverless GPU (A10, AI v5) |
| `04_validate_and_promote` | SETUP → `validate_and_promote` (after 02 + 03) | gate run with hold-out metrics + evaluation artifacts; both pass → `@champion` moves to the challengers, else the task fails | serverless GPU (A10, AI v5) |
| `05_serve_endpoints` | SETUP → `serve_endpoints` (after 04) | endpoints `bio_ecg12_arr`, `bio_ecg12_afib` at `@champion` + inference tables `<endpoint>_payload` | standard serverless |
| `06_generate_batch_data` | DAILY → `simulate_incoming_batch` | `batch_data_12l_<YYYYMMDD>`: simulated incoming data, not real ingestion (acquisition times spread over the clinic day) + sample plot | standard serverless |
| `07_arr_triage` | DAILY → `arr_triage` (after 06) | `arr_triaged_12l_<YYYYMMDD>` and that date's rows of `arr_triage_history_12l` (scored on the driver) | standard serverless, ML v5 |
| `08_afib_classify` | DAILY → `afib_classify` (after 07) | that date's rows of `dailyOrweekly_arr_afib_inference_12l` (scored on the driver) | standard serverless, ML v5 |

Every notebook takes `catalog_name` (required: the jobs pass the bundle's `catalog` variable; when
you run a notebook by hand, type it into the widget) and `schema_name` (default `biosignals`). The
DAILY notebooks also take `run_date`.

Run SETUP first (and again to retrain); DAILY needs its champion models. The DAILY job has a paused
06:00 UTC schedule and one `run_date` job parameter (default `{{job.start_time.iso_date}}`) that
every task reads; set `run_date=YYYY-MM-DD` to backfill a day. Re-running a date gives the same
rows and never duplicates them: per-date tables are overwritten, and the accumulating tables
(`arr_triage_history_12l`, `dailyOrweekly_arr_afib_inference_12l`) get that date's rows replaced
in one Delta transaction; `uc_io.write_date_slice` refuses any existing table without
`batch_date`.

## Model lifecycle: challenger → validation gate → champion

The training notebooks read the `train` and `val` splits of `training_data_12l` (persisted by `01_`)
at one Delta version, train once with the fixed `TRAIN_PARAMS` (no hyperparameter search), log
everything to MLflow (all settings and the dataset identity as params; `train_loss`, `val_loss`,
`val_accuracy` per epoch; `final_val_*` metrics; tags; the pyfunc model with signature and input
example), and register a new UC model version with the alias **`challenger`**. They never touch
`champion`.

`04_validate_and_promote` scores each challenger on the **`holdout`** split, which the training
notebooks never read, at the Delta version the challenger was trained beside, and checks
`VALIDATION_THRESHOLDS` (`common/config.py`):

| Model | Hold-out thresholds (default) |
|---|---|
| `bio_ecg12_arr` | accuracy ≥ 0.95, ROC AUC (area under the receiver operating characteristic curve) ≥ 0.98, sensitivity ≥ 0.97 (a missed abnormal never reaches the AFib model), specificity ≥ 0.90 |
| `bio_ecg12_afib` | accuracy ≥ 0.95, ROC AUC ≥ 0.98, sensitivity ≥ 0.95, share of Other (sinus tachycardia) called AFib ≤ 0.05 |

Both must pass or neither is promoted: on a pass `champion` moves to both challengers (if the second
move fails, the first is put back) and then `challenger` is cleared, so after a pass each model shows
only `@champion`. On a failure nothing moves, `challenger` stays on the failed version (tagged
`validation_status=invalid`) for inspection, the task fails with the reasons and `05_` never runs. So
`challenger` = a candidate awaiting (or that failed) the gate, `champion` = what is served and
batch-scored. A model that was not retrained is re-checked at its `champion` beside the retrained
one; with no `challenger` at all (re-running `04_` alone after a pass) `04_` exits cleanly with a
note. The outcome is recorded in an MLflow run and as model-version tags (`validation_status`,
`gate_result`, `holdout_*`). Aliases are lowercase (`champion`, `challenger`), as Unity Catalog
stores them.

**Tag values and tag policies.** A Unity Catalog tag policy can restrict the values a tag key may
take, and Unity Catalog then rejects any other value on a model version. The gate writes
`validation_status` only as `valid` / `invalid` (this model against its own thresholds), the values
such a policy on that key typically allows, and `gate_result` as `passed` / `failed` (both models,
the result that decides promotion). The training notebooks write no `validation_status`: until the
gate runs, the `challenger` alias is what marks a version as awaiting validation. If your workspace
has a tag policy on another key the notebooks write (check with
`databricks tag-policies list-tag-policies -p <profile>`), add it to `GOVERNED_TAG_VALUES` in
`dev/tests/conftest.py`: the local tests then fail when a notebook could write a value it does not
allow.

**Evaluation artifacts.** From the same hold-out scores (no second scoring pass), the gate run logs
per model under `holdout_eval/<model>/`: `confusion_matrix.png` + `confusion_matrix.json` (true class
x the model's call, raw counts, and the call per true rhythm), `roc_curve.png` (receiver operating
characteristic: true-positive vs false-positive rate over every threshold, with the area under it,
ROC AUC), `pr_curve.png` (precision vs recall, with the average precision), each with the operating
point of the production rule, `per_class_metrics.json` (precision, recall, F1, support per class)
and a `README.md`; each model version gets a `holdout_eval_artifacts` tag pointing there. Each
training run also logs the model's layer table, `model_architecture.md` (output shape and learned
parameters per layer; 10,514 in total), and `n_parameters`. Plots: `common/eval_plots.py`, in the
same white style as `common/ecg_plots.py`.

**Why the metrics sit near 1.0.** The synthetic rhythms are trivially separable by design: heart
rate alone separates triage (NSR is drawn at 60–100 bpm, AFib and Other at 100–160 bpm), and
beat-to-beat irregularity separates AFib (RR intervals vary ~20 % in AFib, ~3 % in sinus rhythm).
Validation and hold-out metrics therefore sit near ceiling (~1.0). This is a clean synthetic
showcase of the MLOps (machine-learning operations) cycle, not a hard ML problem, and the
thresholds are engineering floors for this data, not clinical criteria. For the same reason the
evaluation artifacts look near-perfect (an almost diagonal confusion matrix, ROC and
precision-recall curves in the top corners): they show the evaluation-artifact workflow on clean
synthetic data, not skill on a hard problem, and every figure says so in its caption.

## Batch path vs live path

- **Batch** (DAILY job): `07_` / `08_` resolve `champion` to a concrete version, load exactly that
  version with `mlflow.pyfunc.load_model` and score the day's ~100 ECGs on the job's driver
  (`scoring.score_driver_side`, as the gate does), recording the version in `arr_model_info` /
  `afib_model_info`. The endpoints are not called. Results: `arr_triaged_12l_<date>`,
  `arr_triage_history_12l`, `dailyOrweekly_arr_afib_inference_12l`.
- **Live** (the app): each *Triage → Classify* sends one ECG to `bio_ecg12_arr` and, when flagged,
  to `bio_ecg12_afib`; AI Gateway logs every request/response to `bio_ecg12_arr_payload` /
  `bio_ecg12_afib_payload`. One 12-lead ECG is ~650 KB of JSON, inside the inference table's 1 MiB
  logging limit; a request above 1 MiB is served but logged with `MAX_REQUEST_SIZE_EXCEEDED` and a
  null payload, so `05_`'s smoke test also sends one ECG per request.

## Dashboard

`resources/ecg12_afib.dashboard.yml` deploys `dashboards/ecg12_afib.lvdash.json` with both targets.
Its queries name bare tables; the bundle points them at your catalog and schema
(`dataset_catalog` / `dataset_schema`) and runs them on your `warehouse_id`. Deploying publishes
it, with `embed_credentials: true`: the published dashboard runs its queries with the deploying
identity's credentials, so viewers (and the app's embed) need only `CAN_READ` on the dashboard,
which the bundle grants to `viewer_group`. The data is synthetic; set `embed_credentials: false` to
make each viewer's own Unity Catalog grants apply instead.

- **Batch monitoring** reads `arr_triage_history_12l` and `dailyOrweekly_arr_afib_inference_12l`:
  the funnel (landed → flagged → AFib-positive), arrivals per hour, triage by rhythm, the
  AFib-probability distribution, AFib by age band and the worklist, all for the **latest
  `batch_date`** (one coherent funnel), plus the daily trend over every date.
- **Live serving** reads the two payload tables: requests, model execution time and payloads not
  logged per endpoint, the returned probabilities, the latest requests. It parses only the small
  response JSON; `execution_duration_ms` is the model's inference time, not end-to-end latency.

**Until there is data, tiles show errors, not empty charts.** The batch tiles fail until the DAILY
job has run once (their tables do not exist before). The live tiles fail with `UNRESOLVED_COLUMN`
until AI Gateway has delivered the first inference-table logs: AI Gateway creates each payload
table with a placeholder schema (`databricks_request_id` only) when `05_` enables it, and adds the
real columns with the first delivered batch of logs, best effort, typically within about an hour of
the first app request. Nothing needs to be rebuilt or redeployed; the tiles work once the data is
there.

To change the dashboard, edit `dashboards/build_ecg_demo_dashboard.py`, run
`python dashboards/build_ecg_demo_dashboard.py` and commit the regenerated JSON (`--check` reports
whether the committed file is current). A dashboard edited in the workspace UI differs from the
file, and the next `bundle deploy` stops with "has been modified remotely" unless run with
`--force`; `databricks bundle generate dashboard --resource ecg12_afib_dashboard --force` pulls the
UI version back into the file.

The app reads the dashboard id from `ECG_DEMO_REACT_DASHBOARD_ID`, which the `full` target sets to
the deployed dashboard's id (`${resources.dashboards.ecg12_afib_dashboard.id}`, resolved when the
app is deployed), so pointing the app at the dashboard needs no rebuild. Without an id the app's
Batch tab shows a "Dashboard not configured" note. Embedding also needs the app's domain on the
workspace's dashboard-embedding allowlist (a workspace admin setting; `deploy.sh` adds it).

## The 12-lead signal and patient record

- **Leads.** A standard ECG gives 12 views of the heart's electrical activity: six in the frontal
  (vertical) plane from the limb leads I, II, III and the augmented limb leads aVR, aVL, aVF, and six
  in the horizontal plane from the chest (precordial) leads V1–V6 (Merck Manual Professional,
  *Electrocardiography*). Each lead sees the same beat from a different direction. Only eight are
  independent: I and II (frontal-plane projections of per-wave cardiac vectors) and V1–V6 (normal
  R-wave progression) are synthesized; III, aVR, aVL, aVF are derived from I and II (Einthoven's
  law, Goldberger's leads): `III = II − I`, `aVR = −(I + II)/2`, `aVL = I − II/2`, `aVF = II − I/2`.
  Values are in mV, 250 Hz, 10 s. Stored as `ecg1`..`ecg12` in that lead order (column comments in
  UC say which is which). `00_overview` and the app (under the ECG chart) carry the same legend.
- **Rhythms.** NSR (normal sinus rhythm, 60–100 bpm), AFib (irregular RR, no P waves, f-waves,
  100–160 bpm), Other = sinus tachycardia (110–160 bpm). All classes share one beat model, so they
  differ only in rhythm.
- **Patient record** (synthetic, non-PII): `pseudo_patient_id`, `mrn`, `age_years`, `sex`,
  `height_cm`, `weight_kg`, `bmi` (from height and weight), `bp_systolic_mmhg`,
  `bp_diastolic_mmhg`, `acquisition_datetime_utc`, `heart_rate_bpm` (measured from the ECG),
  `sampling_rate_hz`, `n_leads`, `duration_s`. Demographics are fixed per `pseudo_patient_id`;
  AFib is assigned more often to older patients. Acquisition times fall in clinic hours
  (07:00–19:00 UTC, a busier morning), so a day's batch reads like an incoming stream.

## Configuration

Bundle variables (`databricks.yml`). Pass them as `--var name=value` on every bundle command (or
export `BUNDLE_VAR_<name>`); `bundle validate` refuses to run without the required ones.

| Variable | Required | Default | Used for |
|---|---|---|---|
| `catalog` | **yes** | — | Unity Catalog catalog of every table, model and inference table (jobs: `catalog_name`; dashboard: `dataset_catalog`; app: `ECG_DEMO_REACT_CATALOG`) |
| `warehouse_id` | **yes** | — | SQL warehouse the dashboard runs on |
| `schema` | no | `biosignals` | Unity Catalog schema (jobs: `schema_name`; dashboard: `dataset_schema`; app: `ECG_DEMO_REACT_UC_SCHEMA`); `01_` creates it if needed |
| `app_name` | no | `ecg12-demo` | Name of the Databricks App (`full` target). Lowercase letters, digits and hyphens, at most 30 characters. Override it if that name is already taken in your workspace (app names are workspace-wide) |
| `viewer_group` | no | `users` | Group granted `CAN_READ` on the dashboard and `CAN_USE` on the app |

The workspace itself comes from your Databricks CLI profile (`-p <profile>`); no host or profile
is committed. Model and endpoint names (`bio_ecg12_arr`, `bio_ecg12_afib`) are set in
`common/config.py`. Endpoint names are workspace-wide and `05_` updates an existing endpoint of the
same name, so one workspace holds one deployment of this project unless you change those names.

## Deploy

Two targets, which are **separate deployments** (each has its own folder and state under
`/Workspace/Users/<you>/.bundle/biosignals-ecg12-afib/<target>/`). Pick one per workspace; to
switch, `bundle destroy` the first. Both use `mode: development` (job and dashboard names get a
`[dev <user>]` prefix, schedules deploy paused; the app name is not prefixed).

**What the deploying identity needs:** `USE CATALOG` and `CREATE SCHEMA` on the catalog (or, if the
schema exists, `USE SCHEMA`, `CREATE TABLE` and `CREATE MODEL` on it); serverless jobs, including
serverless GPU; permission to create Model Serving endpoints; `CAN_USE` on the warehouse; for
`full`, permission to create apps. Adding the embedding allowlist entry needs a workspace admin.

### Path 1: jobs + dashboard from the workspace (target `no_app`, no local build)

1. Clone this repository into the workspace as a Git folder.
2. Open `projects/ecg12-afib/databricks.yml` in the bundle editor and set the required variables
   for the `no_app` target, for example in your copy:
   ```yaml
   targets:
     no_app:
       variables:
         catalog: <catalog>
         warehouse_id: <warehouse-id>
   ```
3. Deploy (target `no_app`, the default). Then run `ecg12_afib_setup`, then `ecg12_afib_daily`
   from the Deployments tab.

The same from a laptop:

```bash
cd projects/ecg12-afib
databricks bundle validate --strict -p <profile> --var catalog=<catalog> --var warehouse_id=<id>
databricks bundle deploy -p <profile> --var catalog=<catalog> --var warehouse_id=<id>
databricks bundle run ecg12_afib_setup -p <profile> --var catalog=<catalog> --var warehouse_id=<id>
databricks bundle run ecg12_afib_daily -p <profile> --var catalog=<catalog> --var warehouse_id=<id>
```

No app is deployed on this path; the dashboard and the serving endpoints still are.

### Path 2: everything, including the app (target `full`, `deploy.sh`)

The app's wheel and compiled UI are not committed: they are built locally from the lockfiles.
**Prerequisites:** [apx](https://github.com/databricks-solutions/apx) **0.3.8** exactly (it brings
its own bun, which builds the UI), [uv](https://docs.astral.sh/uv/), the Databricks CLI 0.281.0 or
later, `jq`, `python3`, and a CLI profile for your workspace (`databricks auth login -p <profile>`).

```bash
cd projects/ecg12-afib
./deploy.sh -p <profile> --catalog <catalog> --warehouse-id <id> [--run-daily]
```

`deploy.sh` stops with an error at the first step that fails:

1. checks the prerequisites, the required arguments and that the profile authenticates;
2. builds the app (`app/build_from_lockfiles.sh`, see [`app/README.md`](app/README.md));
3. `databricks bundle validate --strict -t full`;
4. `databricks bundle deploy -t full`: the jobs, the dashboard and the app, with `viewer_group`
   granted `CAN_READ` on the dashboard and `CAN_USE` on the app;
5. runs the SETUP job and waits for it (`--run-daily` then also runs the DAILY job once);
6. grants the app's service principal `CAN_QUERY` on both endpoints (they exist only after the
   SETUP job, so they cannot be app resources of the first deploy);
7. starts the app (`databricks bundle run ecg12_afib_app`), which deploys the built source with
   its environment, including the dashboard id;
8. adds the app's domain to the dashboard-embedding allowlist (needs a workspace admin; fails with
   the domain to add, or `--skip-embed-allowlist` skips it and prints that domain).

Other options: `--schema`, `--app-name`, `--viewer-group` (the bundle variables above). Re-running
is safe: every step updates in place, and the SETUP job retrains and re-gates both models.

The notebooks also run one at a time from a Git folder (they find `common/` one folder up); type
the catalog into each notebook's `catalog_name` widget.

## Cost

These are billable resources in your workspace: serverless job runs (SETUP trains and gates on
serverless GPU, A10; DAILY runs on standard serverless), two Model Serving endpoints (workload size
Small with scale-to-zero, so they cost only while they serve, and the first request after idling
waits for them to start), the SQL warehouse while the dashboard runs queries, and the app while it
is running (`databricks apps stop <app_name> -p <profile>` stops it). The DAILY schedule is paused.

## Teardown

```bash
cd projects/ecg12-afib
databricks bundle destroy -t <no_app|full> -p <profile> --var catalog=<catalog> --var warehouse_id=<id>
```

This removes what the bundle created: the jobs, the dashboard and (`full`) the app. It does not
remove what the jobs created; delete those yourself:

```bash
databricks serving-endpoints delete bio_ecg12_arr -p <profile>
databricks serving-endpoints delete bio_ecg12_afib -p <profile>
databricks registered-models delete <catalog>.<schema>.bio_ecg12_arr -p <profile>
databricks registered-models delete <catalog>.<schema>.bio_ecg12_afib -p <profile>
```

then drop the tables (`training_data_12l`, `batch_data_12l_*`, `arr_triaged_12l_*`,
`arr_triage_history_12l`, `dailyOrweekly_arr_afib_inference_12l`, `bio_ecg12_*_payload`) or the
whole schema if it holds nothing else (`DROP SCHEMA <catalog>.<schema> CASCADE`), and remove the
app's domain from the embedding allowlist.

## Tests

**Optional local checks.** The `pytest` tests in [`dev/tests/`](dev/tests/README.md) run on your
laptop only (Python 3.11; nothing runs them automatically and they are not deployed). They check
code logic: the 12-lead formulas, the patient record, training-data splits, the validation gate
promoting both models or neither, batch-scoring output, the dashboard's queries on local Spark, the
bundle configuration, and that names come from config. They run nothing in a workspace. **The
real pre-deployment check is a live run:** the SETUP job, then one DAILY run, before unpausing the
schedule.

```bash
cd projects/ecg12-afib
dev/tests/run_tests_python311.sh                                       # all of them, on Python 3.11
dev/tests/run_tests_python311.sh dev/tests/test_scoring_and_io.py -v   # one file
```

## Future work

**Bulk inference.** The DAILY job scores one day's ~100 ECGs on the driver. Larger backlogs need
the scoring spread over workers, for example with Spark (`mlflow.pyfunc.spark_udf`) or Ray; both
are open to explore. A Spark starting point exists: `score_spark_udf` in `common/scoring.py`,
tested locally only and used by no job. Its one try on standard serverless, where the workers need
the model's own environment (`env_manager="uv"`) with PyTorch's CPU wheels, failed on uv's index
strategy (uv took `setuptools` from the CPU torch index, which lacks the pinned version); that
looks fixable, but the path is untested on Databricks.

**Other ECG classifiers.** Models that classify other ECG patterns (other rhythms, conduction
abnormalities or other waveform signatures) belong in their own projects under `projects/`, each
with its own bundle, README and tests, rather than in this one. This project keeps to the
two-model triage → AFib cascade.
