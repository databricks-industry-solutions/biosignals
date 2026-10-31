# ECG/AFib demo (12-lead): Databricks App + SETUP job + DAILY job + AI/BI dashboard

A two-model ECG cascade on a synthetic **clinical 12-lead ECG** (I, II, III, aVR, aVL, aVF,
V1–V6). A cheap **arrhythmia (ARR) triage** model flags abnormal ECGs, and an **atrial
fibrillation (AFib)** classifier scores only the flagged ones. Both are small custom PyTorch
1D-CNNs (one-dimensional convolutional neural networks) with 12 input channels. The demo runs the
full model cycle: **generate training data → train → validate on a hold-out split → promote →
serve → score daily → monitor**. The same Unity Catalog (UC) models run two ways:

- **Live:** the app generates a 12-lead ECG plus a synthetic patient record and sends all 12
  leads to two Model Serving endpoints, `<endpoint_prefix>_arr` (triage, serving the model
  `bio_ecg12_arr`), then `<endpoint_prefix>_afib` (classify, serving `bio_ecg12_afib`). The prefix is
  the bundle variable `endpoint_prefix` (default `bio_ecg12`, so by default each endpoint has the
  same name as its model), and each call is logged to the endpoint's AI Gateway inference table.
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
| `05_serve_endpoints` | SETUP → `serve_endpoints` (after 04) | endpoints `<endpoint_prefix>_arr`, `<endpoint_prefix>_afib` at `@champion` (stops if one already serves another deployment's model) + inference tables `<model>_payload` | standard serverless |
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
- **Live** (the app): each *Triage → Classify* sends one ECG to the triage endpoint
  (`<endpoint_prefix>_arr`) and, when flagged, to the classify endpoint (`<endpoint_prefix>_afib`);
  AI Gateway logs every request/response to `bio_ecg12_arr_payload` /
  `bio_ecg12_afib_payload`. One 12-lead ECG is ~650 KB of JSON, inside the inference table's 1 MiB
  logging limit; a request above 1 MiB is served but logged with `MAX_REQUEST_SIZE_EXCEEDED` and a
  null payload, so `05_`'s smoke test and the app's warm-up pings also send one ECG per request.

**Scale-to-zero and the app.** Both endpoints scale to zero when idle, and waking one took 89 s and
130 s when this demo was tested. So every endpoint call from the app is bounded by
`SERVING_TIMEOUT_S` (30 s, `common/config.py`; the databricks-sdk `http_timeout_seconds` and
`retry_timeout_seconds` of the app's serving client): a call that hits it returns HTTP 503 with a
"waking up from scale-to-zero, try again" message instead of hanging; a rate-limited call returns
429 and any other endpoint error 502, each with the endpoint and a message the UI shows. The UI gives
*Triage → Classify* its own limit of 80 s (two bounded calls plus a margin), a Cancel button and
Retry. While the Live tab is visible it calls `POST /api/warmup` on load, again right away while an
endpoint is still cold, then every `WARMUP_INTERVAL_S` (600 s, inside the endpoints' idle window),
stops while the page is hidden, and shows a ready / waking / cold / error chip per endpoint. Each
warm-up ping is one single-ECG request per endpoint with `client_request_id = app-warmup`
(`WARMUP_REQUEST_ID`), so it is one row in that endpoint's payload table, which the dashboard leaves
out of its live tiles and counts separately. The pings use only the app's `CAN_QUERY` grant.

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
  The app's warm-up pings (`client_request_id = 'app-warmup'`) are left out of all of these and
  counted on their own ("App warm-up pings").

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
Batch tab shows a "Dashboard not configured" note. The header's "Lakeflow" link works the same way:
`ECG_DEMO_REACT_DAILY_JOB_ID` = `${resources.jobs.ecg12_afib_daily.id}` opens the DAILY job, and
without it the link opens the workspace's jobs list. Embedding also needs the app's domain on the
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
| `endpoint_prefix` | no | `bio_ecg12` | Prefix of the two serving endpoints, `<endpoint_prefix>_arr` and `<endpoint_prefix>_afib` (letters, digits, `_`, `-`; SETUP job: `endpoint_prefix`; app: `ECG_DEMO_REACT_ENDPOINT_PREFIX`). Endpoint names are workspace-wide: use a value no other deployment in your workspace uses |
| `app_name` | no | `ecg12-demo` | Name of the Databricks App (`full` target). Lowercase letters, digits and hyphens, at most 30 characters. Override it if that name is already taken in your workspace (app names are workspace-wide) |
| `viewer_group` | no | `users` | Group granted `CAN_READ` on the dashboard and `CAN_USE` on the app |

The workspace itself comes from your Databricks CLI profile (`-p <profile>`); no host or profile
is committed. The model names (`bio_ecg12_arr`, `bio_ecg12_afib`) and their inference tables
(`bio_ecg12_arr_payload`, `bio_ecg12_afib_payload`) are fixed in `common/config.py`: they live in
your catalog and schema, so they are this deployment's own. The serving endpoints are the one
workspace-wide name, so they come from `endpoint_prefix`. **`05_` never takes over another
deployment's endpoint:** before updating an existing endpoint of the same name it checks what the
endpoint serves, and if that is anything but this deployment's `<catalog>.<schema>.<model>` the
SETUP job stops with "Serving endpoint … already exists and serves …; set the bundle variable
endpoint_prefix to a unique value", having changed neither endpoint.

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
   its environment, including the dashboard and DAILY job ids;
8. adds the app's domain to the dashboard-embedding allowlist (needs a workspace admin; fails with
   the domain to add, or `--skip-embed-allowlist` skips it and prints that domain).

Other options: `--schema`, `--endpoint-prefix`, `--app-name`, `--viewer-group` (the bundle
variables above). Re-running
is safe: every step updates in place, and the SETUP job retrains and re-gates both models.

The notebooks also run one at a time from a Git folder (they find `common/` one folder up); type
the catalog into each notebook's `catalog_name` widget.

## Cost

These are billable resources in your workspace: serverless job runs (SETUP trains and gates on
serverless GPU, A10; DAILY runs on standard serverless), two Model Serving endpoints (workload size
Small with scale-to-zero, so they cost only while they serve, and the first request after idling
waits for them to start), the SQL warehouse while the dashboard runs queries, and the app while it
is running (`databricks apps stop <app_name> -p <profile>` stops it). While someone has the app's
Live tab open, its warm-up pings keep both endpoints awake (one request each every 10 minutes);
they scale to zero again once no page is open. The DAILY schedule is paused.

## Teardown

```bash
cd projects/ecg12-afib
databricks bundle destroy -t <no_app|full> -p <profile> --var catalog=<catalog> --var warehouse_id=<id>
```

This removes what the bundle created: the jobs, the dashboard and (`full`) the app. It does not
remove what the jobs created; delete those yourself. Delete only this deployment's endpoints, by
the names your `endpoint_prefix` gives (`<endpoint_prefix>_arr`, `<endpoint_prefix>_afib`), and
first check that each one serves your own `<catalog>.<schema>` models (endpoint names are
workspace-wide, so another deployment's endpoint can have a similar name):

```bash
P=<profile>; PREFIX=<endpoint_prefix>; UC=<catalog>.<schema>
# Every model the endpoint serves or is about to serve (current + pending config, incl. the
# legacy served_models field): delete it only if that is exactly this deployment's model.
SERVED='[(.config, .pending_config) | select(. != null)
         | ((.served_entities // [])[].entity_name, (.served_models // [])[].model_name)]
        | map(select(. != null)) | unique | join(",")'
for pair in arr:bio_ecg12_arr afib:bio_ecg12_afib; do
  e="${PREFIX}_${pair%%:*}"; want="$UC.${pair#*:}"
  got=$(databricks serving-endpoints get "$e" -p "$P" -o json | jq -r "$SERVED")
  if [ "$got" = "$want" ]; then databricks serving-endpoints delete "$e" -p "$P"
  else echo "Not deleting $e: it serves '$got', not '$want'"; fi
done
databricks registered-models delete "$UC.bio_ecg12_arr" -p "$P"
databricks registered-models delete "$UC.bio_ecg12_afib" -p "$P"
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

## Third-party packages and licenses

The source in this project is provided subject to the Databricks License (see
[LICENSE.md](../../LICENSE.md)). The packages below keep their own licenses. Versions come from the
lockfiles (`app/uv.lock`, `app/bun.lock`), the notebooks' `%pip install` lines and
`dev/tests/run_tests_python311.sh`; "preinstalled" means the package comes with the Databricks
serverless environment the job runs in (AI v5 for training, ML v5 for batch scoring) and the
project does not pin it. Licenses were read from each package's own metadata or license file.

### Shipped: packages the project's code uses directly

| Package | Version | Purpose (where used) | License | Copyright | Source |
|---|---|---|---|---|---|
| `@radix-ui/react-slot` | 1.2.4 | Button composition (`ui/components/ui/button.tsx`) | MIT | Copyright (c) 2022 WorkOS | [repo](https://github.com/radix-ui/primitives) |
| `@tanstack/react-query` | 5.90.16 | API request caching (`ui/lib/api.ts`, `ui/main.tsx`, `ui/routes/`) | MIT | Copyright (c) 2021-present Tanner Linsley | [repo](https://github.com/TanStack/query) |
| `@tanstack/react-router` | 1.145.7 | Client-side routing (`ui/main.tsx`, `ui/routes/`) | MIT | Copyright (c) 2021-present Tanner Linsley | [repo](https://github.com/TanStack/router) |
| `class-variance-authority` | 0.7.1 | Button variant classes (`ui/components/ui/button.tsx`) | Apache-2.0 | Copyright 2022 Joe Bell | [repo](https://github.com/joe-bell/cva) |
| `clsx` | 2.1.1 | Joining CSS class names (`ui/lib/utils.ts`) | MIT | Copyright (c) Luke Edwards | [repo](https://github.com/lukeed/clsx) |
| `databricks-sdk` | 0.117.0 (app); 0.67.0 (`05_serve_endpoints`) | Workspace, serving-endpoint and identity APIs (`backend/router.py`, `backend/core/_defaults.py`, `workflow/05_serve_endpoints.py`) | Apache-2.0 | Copyright 2023 Databricks, Inc. | [repo](https://github.com/databricks/databricks-sdk-py) |
| `fastapi` | 0.136.3 | App HTTP API (`app/src/ecg_demo_react/backend/`) | MIT | Copyright (c) 2018 Sebastián Ramírez | [repo](https://github.com/fastapi/fastapi) |
| `matplotlib` | preinstalled | ECG and evaluation plots (`common/ecg_plots.py`, `common/eval_plots.py`, notebooks 01, 04, 06) | Matplotlib License (PSF-based) | Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved | [repo](https://github.com/matplotlib/matplotlib) |
| `mlflow-skinny` | 3.8.1 (`05_serve_endpoints`); otherwise preinstalled | Experiment tracking, model registry, model loading (notebooks 02-05, 07, 08) | Apache-2.0 | Copyright 2018 Databricks, Inc. | [repo](https://github.com/mlflow/mlflow) |
| `neurokit2` | 0.2.12 | Recording noise for the synthetic ECGs (`common/ecg_signal_gen.py` and its app copy) | MIT | Copyright (c) 2020, Dominique Makowski | [repo](https://github.com/neuropsychology/NeuroKit) |
| `numpy` | 1.26.4 (app); preinstalled (notebooks) | ECG arrays, training, scoring, patient records (`common/`, app backend, notebooks) | BSD-3-Clause | Copyright (c) 2005-2023, NumPy Developers | [repo](https://github.com/numpy/numpy) |
| `pandas` | preinstalled | Training and scoring tables (`common/scoring.py`, `common/training_data.py`, notebooks 02, 03, 04, 06) | BSD-3-Clause | Copyright (c) 2008-2011, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team; Copyright (c) 2011-2023, Open source contributors | [repo](https://github.com/pandas-dev/pandas) |
| `pydantic` | 2.13.4 | API request/response models (`backend/models.py`, `backend/core/`) | MIT | Copyright (c) 2017 to present Pydantic Services Inc. and individual contributors | [repo](https://github.com/pydantic/pydantic) |
| `pydantic-settings` | 2.14.1 | App configuration from environment variables (`backend/core/_config.py`) | MIT | Copyright (c) 2022 Samuel Colvin and other contributors | [repo](https://github.com/pydantic/pydantic-settings) |
| `pyspark` | preinstalled | Spark tables and transforms (`common/scoring.py`, `common/training_data.py`, notebooks 04, 06, 08) | Apache-2.0 | Copyright 2014 and onwards The Apache Software Foundation | [repo](https://github.com/apache/spark) |
| `python-dotenv` | 1.2.2 | Loading a local `.env` file (`backend/core/_config.py`) | BSD-3-Clause | Copyright (c) 2014, Saurabh Kumar (python-dotenv), 2013, Ted Tieken (django-dotenv-rw), 2013, Jacob Kaplan-Moss (django-dotenv) | [repo](https://github.com/theskumar/python-dotenv) |
| `PyWavelets` | 1.4.1 | Wavelet support needed by neurokit2 (`app/pyproject.toml`, notebooks 01, 05, 06) | MIT | Copyright (c) 2006-2012 Filip Wasilewski; Copyright (c) 2012-2020 The PyWavelets Developers | [repo](https://github.com/PyWavelets/pywt) |
| `react`, `react-dom` | 19.2.3 | App user interface (`app/src/ecg_demo_react/ui/`) | MIT | Copyright (c) Meta Platforms, Inc. and affiliates | [repo](https://github.com/facebook/react) |
| `recharts` | 3.8.1 | The 12-lead ECG chart (`ui/components/ecg-chart.tsx`); its dependency victory-vendor bundles d3 modules under ISC | MIT | Copyright (c) 2015-present recharts | [repo](https://github.com/recharts/recharts) |
| `scipy` | 1.13.1 (app); preinstalled (notebooks) | Signal processing needed by neurokit2 (`app/pyproject.toml`) | BSD-3-Clause | Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers | [repo](https://github.com/scipy/scipy) |
| `sonner` | 2.0.7 | Pop-up notifications (`ui/routes/__root.tsx`) | MIT | Copyright (c) 2023 Emil Kowalski | [repo](https://github.com/emilkowalski/sonner) |
| `starlette` | 1.2.1 | Serving the built UI files (`backend/core/_static.py`) | BSD-3-Clause | Copyright © 2018, Encode OSS Ltd | [repo](https://github.com/Kludex/starlette) |
| `tailwind-merge` | 3.4.0 | Merging CSS classes (`ui/lib/utils.ts`) | MIT | Copyright (c) 2021 Dany Castillo | [repo](https://github.com/dcastil/tailwind-merge) |
| `tailwindcss` | 0.0.0-insiders.1bf4291 | UI styling (`ui/styles/globals.css`) | MIT | Copyright (c) Tailwind Labs, Inc. | [repo](https://github.com/tailwindlabs/tailwindcss) |
| `torch` (PyTorch) | preinstalled | The two neural-network models and their training (`common/training.py`, notebooks 02, 03) | BSD-3-Clause | Copyright (c) 2016- Facebook, Inc. (Adam Paszke) and the other contributors listed in its LICENSE | [repo](https://github.com/pytorch/pytorch) |
| `tw-animate-css` | 1.4.0 | UI animation classes (`ui/styles/globals.css`) | MIT | Copyright (c) 2025 Wombosvideo | [repo](https://github.com/Wombosvideo/tw-animate-css) |
| `uvicorn` | 0.49.0 | Runs the app's web server (`app/app.yml`) | BSD-3-Clause | Copyright © 2017-present, Encode OSS Ltd | [repo](https://github.com/Kludex/uvicorn) |

### Build and development tools (not shipped)

| Package | Version | Purpose (where used) | License | Source |
|---|---|---|---|---|
| `apx` | 0.3.8 (UI build entrypoint from commit `50073e5`) | Builds the app's UI and Python wheel (`app/build_from_lockfiles.sh`, `deploy.sh`) | Databricks License | [repo](https://github.com/databricks-solutions/apx) |
| `bun` | managed by apx, not pinned | Installs the locked UI packages and runs the UI build | MIT; the binary statically links LGPL-2 JavaScriptCore and LGPL-2.1 tinycc | [repo](https://github.com/oven-sh/bun) |
| `uv` | not pinned | Locked Python installs and the test environment (build and test scripts) | MIT OR Apache-2.0 | [repo](https://github.com/astral-sh/uv) |
| `hatchling` | not pinned | Python wheel build backend (`app/pyproject.toml`) | MIT | [repo](https://github.com/pypa/hatch) |
| `uv-dynamic-versioning` | >=0.7.0, not locked | Wheel version from git (`app/pyproject.toml`) | MIT | [repo](https://github.com/ninoseki/uv-dynamic-versioning) |
| `vite` | 8.0.16 | Bundles the React UI | MIT | [repo](https://github.com/vitejs/vite) |
| `@vitejs/plugin-react` | 6.0.2 | React support for the Vite build | MIT | [repo](https://github.com/vitejs/vite-plugin-react) |
| `@tailwindcss/vite` | 0.0.0-insiders.1bf4291 | Compiles Tailwind CSS during the build | MIT | [repo](https://github.com/tailwindlabs/tailwindcss) |
| `@tanstack/router-plugin`, `@tanstack/router-generator` | 1.168.18, 1.167.17 | Generates the typed route definitions | MIT | [repo](https://github.com/TanStack/router) |
| `@opentelemetry/api-logs`, `exporter-logs-otlp-http`, `sdk-logs`, `resources` | 0.218.0, 0.218.0, 0.218.0, 2.7.1 | Build logging in the apx UI build entrypoint | Apache-2.0 | [repo](https://github.com/open-telemetry/opentelemetry-js) |
| `smol-toml` | 1.6.0 | Reads `pyproject.toml` in the apx UI build | BSD-3-Clause | [repo](https://github.com/squirrelchat/smol-toml) |
| `typescript` | 5.9.3 | UI compilation and type checking (`app/tsconfig.json`) | Apache-2.0 | [repo](https://github.com/microsoft/TypeScript) |
| `@types/node`, `@types/react`, `@types/react-dom` | 25.0.3, 19.2.7, 19.2.3 | TypeScript type definitions | MIT | [repo](https://github.com/DefinitelyTyped/DefinitelyTyped) |
| `ty` | 0.0.47 | Python type checking (dev dependency) | MIT | [repo](https://github.com/astral-sh/ty) |
| `pytest` | 9.1.1 | Optional local checks (`dev/tests/`) | MIT | [repo](https://github.com/pytest-dev/pytest) |
| `cloudpickle` | 3.0.0 | Optional local checks: model serialization (`dev/tests/`) | BSD-3-Clause | [repo](https://github.com/cloudpipe/cloudpickle) |
| `pyarrow` | 21.0.0 | Optional local checks: local Spark (`dev/tests/`) | Apache-2.0 | [repo](https://github.com/apache/arrow) |
| `PyYAML` | 6.0.3 | Optional local checks: reading the bundle files (`dev/tests/`) | MIT | [repo](https://github.com/yaml/pyyaml) |

### Everything the deployed app installs

The app's build pins every runtime Python package from `app/uv.lock` (the output of
`uv export --frozen --no-dev --no-emit-project`: 50 lines for 49 packages, because `pandas` has
one version per Python range). Packages that pull a dependency in are named in the Purpose column.

| Package | Version | Purpose | License | Source |
|---|---|---|---|---|
| `annotated-doc` | 0.0.4 | via fastapi | MIT | [repo](https://github.com/fastapi/annotated-doc) |
| `annotated-types` | 0.7.0 | via pydantic | MIT | [repo](https://github.com/annotated-types/annotated-types) |
| `anyio` | 4.13.0 | via starlette | MIT | [repo](https://github.com/agronholm/anyio) |
| `certifi` | 2026.5.20 | via requests (TLS certificates) | MPL-2.0 | [repo](https://github.com/certifi/python-certifi) |
| `cffi` | 2.0.0 | via cryptography | MIT | [repo](https://github.com/python-cffi/cffi) |
| `charset-normalizer` | 3.4.7 | via requests | MIT | [repo](https://github.com/jawah/charset_normalizer) |
| `click` | 8.4.1 | via uvicorn | BSD-3-Clause | [repo](https://github.com/pallets/click) |
| `colorama` | 0.4.6 (Windows only) | via click | BSD-3-Clause | [repo](https://github.com/tartley/colorama) |
| `contourpy` | 1.3.3 | via matplotlib | BSD-3-Clause | [repo](https://github.com/contourpy/contourpy) |
| `cryptography` | 48.0.1 | via google-auth | Apache-2.0 OR BSD-3-Clause | [repo](https://github.com/pyca/cryptography) |
| `cycler` | 0.12.1 | via matplotlib | BSD-3-Clause | [repo](https://github.com/matplotlib/cycler) |
| `databricks-sdk` | 0.117.0 | direct | Apache-2.0 | [repo](https://github.com/databricks/databricks-sdk-py) |
| `fastapi` | 0.136.3 | direct | MIT | [repo](https://github.com/fastapi/fastapi) |
| `fonttools` | 4.63.0 | via matplotlib | MIT | [repo](https://github.com/fonttools/fonttools) |
| `google-auth` | 2.53.0 | via databricks-sdk | Apache-2.0 | [repo](https://github.com/googleapis/google-auth-library-python) |
| `h11` | 0.16.0 | via uvicorn | MIT | [repo](https://github.com/python-hyper/h11) |
| `idna` | 3.18 | via anyio, requests | BSD-3-Clause | [repo](https://github.com/kjd/idna) |
| `joblib` | 1.5.3 | via scikit-learn | BSD-3-Clause | [repo](https://github.com/joblib/joblib) |
| `kiwisolver` | 1.5.0 | via matplotlib | BSD-3-Clause | [repo](https://github.com/nucleic/kiwi) |
| `matplotlib` | 3.10.9 | via neurokit2 | Matplotlib License (PSF-based) | [repo](https://github.com/matplotlib/matplotlib) |
| `narwhals` | 2.22.1 | via scikit-learn | MIT | [repo](https://github.com/narwhals-dev/narwhals) |
| `neurokit2` | 0.2.12 | direct | MIT | [repo](https://github.com/neuropsychology/NeuroKit) |
| `numpy` | 1.26.4 | direct | BSD-3-Clause | [repo](https://github.com/numpy/numpy) |
| `packaging` | 26.2 | via matplotlib | Apache-2.0 OR BSD-2-Clause | [repo](https://github.com/pypa/packaging) |
| `pandas` | 3.0.3 (Python <3.14); 2.3.3 (Python >=3.14) | via neurokit2 | BSD-3-Clause | [repo](https://github.com/pandas-dev/pandas) |
| `pillow` | 12.2.0 | via matplotlib | MIT-CMU | [repo](https://github.com/python-pillow/Pillow) |
| `protobuf` | 6.33.6 | via databricks-sdk | BSD-3-Clause | [repo](https://github.com/protocolbuffers/protobuf) |
| `pyasn1` | 0.6.3 | via pyasn1-modules | BSD-2-Clause | [repo](https://github.com/pyasn1/pyasn1) |
| `pyasn1-modules` | 0.4.2 | via google-auth | BSD-2-Clause | [repo](https://github.com/pyasn1/pyasn1-modules) |
| `pycparser` | 3.0 | via cffi | BSD-3-Clause | [repo](https://github.com/eliben/pycparser) |
| `pydantic` | 2.13.4 | direct | MIT | [repo](https://github.com/pydantic/pydantic) |
| `pydantic-core` | 2.46.4 | via pydantic | MIT | [repo](https://github.com/pydantic/pydantic) |
| `pydantic-settings` | 2.14.1 | direct | MIT | [repo](https://github.com/pydantic/pydantic-settings) |
| `pyparsing` | 3.3.2 | via matplotlib | MIT | [repo](https://github.com/pyparsing/pyparsing) |
| `python-dateutil` | 2.9.0.post0 | via matplotlib, pandas | BSD-3-Clause OR Apache-2.0 | [repo](https://github.com/dateutil/dateutil) |
| `python-dotenv` | 1.2.2 | direct | BSD-3-Clause | [repo](https://github.com/theskumar/python-dotenv) |
| `pytz` | 2026.2 (Python >=3.14) | via pandas | MIT | [repo](https://github.com/stub42/pytz) |
| `PyWavelets` | 1.4.1 | direct | MIT | [repo](https://github.com/PyWavelets/pywt) |
| `requests` | 2.34.2 | via databricks-sdk, neurokit2 | Apache-2.0 | [repo](https://github.com/psf/requests) |
| `scikit-learn` | 1.9.0 | via neurokit2 | BSD-3-Clause | [repo](https://github.com/scikit-learn/scikit-learn) |
| `scipy` | 1.13.1 | direct | BSD-3-Clause | [repo](https://github.com/scipy/scipy) |
| `six` | 1.17.0 | via python-dateutil | MIT | [repo](https://github.com/benjaminp/six) |
| `starlette` | 1.2.1 | direct (also via fastapi) | BSD-3-Clause | [repo](https://github.com/Kludex/starlette) |
| `threadpoolctl` | 3.6.0 | via scikit-learn | BSD-3-Clause | [repo](https://github.com/joblib/threadpoolctl) |
| `typing-extensions` | 4.15.0 | via pydantic and others | PSF-2.0 | [repo](https://github.com/python/typing_extensions) |
| `typing-inspection` | 0.4.2 | via pydantic | MIT | [repo](https://github.com/pydantic/typing-inspection) |
| `tzdata` | 2026.2 (Python >=3.14, Windows or Emscripten) | via pandas | Apache-2.0 | [repo](https://github.com/python/tzdata) |
| `urllib3` | 2.7.0 | via databricks-sdk, requests | MIT | [repo](https://github.com/urllib3/urllib3) |
| `uvicorn` | 0.49.0 | direct | BSD-3-Clause | [repo](https://github.com/Kludex/uvicorn) |
