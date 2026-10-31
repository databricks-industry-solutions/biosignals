# ECG/AFib demo (12-lead): Databricks App + SETUP job + DAILY job

A two-model ECG cascade on a synthetic **clinical 12-lead ECG** (I, II, III, aVR, aVL, aVF,
V1–V6). A cheap **arrhythmia (ARR) triage** model flags abnormal ECGs, and an **atrial
fibrillation (AFib)** classifier scores only the flagged ones. Both are small custom PyTorch
1D-CNNs (the same design as the earlier 8-lead models, with 12 input channels). The demo runs the
full model cycle: **generate training data → train → validate on a hold-out split → promote →
serve → score daily → monitor**. The same Unity Catalog (UC) models run two ways:

- **Live:** the app generates a 12-lead ECG plus a synthetic patient record and sends all 12
  leads to two Model Serving endpoints, `bio_ecg12_arr` (triage), then `bio_ecg12_afib` (classify).
  Each endpoint has the same name as the model it serves, and each call is logged to the
  endpoint's AI Gateway inference table.
- **Batch:** a daily Lakeflow job loads the same champion models from UC as Spark UDFs
  (`mlflow.pyfunc.spark_udf`) and scores a synthetic batch on the Spark workers into Delta tables
  (it does not call the endpoints).

An embedded AI/BI dashboard shows both, in two separate sections (batch monitoring, live serving).

Everything targets workspace `fevm-mmt-aws-usw2` (id `7474658466980277`), catalog
`mmt_aws_usw2_catalog`, schema `biosignals` (the `CATALOG` / `SCHEMA` defaults in
[`common/config.py`](common/config.py), where every name lives; jobs can override them with the
`catalog_name` / `schema_name` parameters). The live 8-lead demo (app `ekg-demo`, models/endpoints
`arr_model_mmt` / `afib_demo_mmt`) is separate and is never written to by this code. Neither are the
first 12-lead run's models and endpoints `arr_model_12l` / `afib_demo_12l` (schema `ekg_afib`):
endpoint names are workspace-wide, so this demo uses its own names and never updates those in place.

> **A reference pattern, not a fixed pipeline.** The tasks, models, catalog/schema, rhythm classes
> and lead count are all swappable: names, splits, training settings and validation thresholds live
> in `common/config.py`, the data in `common/ecg_signal_gen.py`, the model code in the training
> notebooks, and the task graph in `resources/*.job.yml`. Swap in a real ECG source or another model
> and the generate → train → gate → promote → serve → monitor cycle stays the same.

## Layout

| Path | What it is |
|---|---|
| `common/` | Shared code, single source of truth: `config.py` (names, `N_LEADS = 12`, aliases, training split + settings, validation thresholds, batch-scoring partition size, run date), `ecg_signal_gen.py` (12-lead generator), `training_data.py` (the persisted training dataset), `training.py` (PyTorch training loop, layer table), `evaluation.py` (metrics, the gate's threshold check, confusion counts, ROC / precision-recall curves, per-class table), `eval_plots.py` (the gate's evaluation figures), `ecg_plots.py` (sample-signal plots), `patient_info.py` (synthetic patient record), `scoring.py` (alias → version, Spark-UDF batch scoring, driver-side scoring for the gate), `uc_io.py` (re-run-safe Delta writes, Delta versions) |
| `workflow/` | Notebooks `00_`–`08_` (below). Each puts `..` on `sys.path` and imports `common` |
| `app/` | APX app (React UI + FastAPI backend). `backend/common/` is a byte-identical copy of `common/{__init__,config,ecg_signal_gen,patient_info}.py`; the build fails if it drifts. Build with `app/build_from_lockfiles.sh`, see [`app/README.md`](app/README.md) |
| `dashboards/` | `build_ecg_demo_dashboard.py`: writes the AI/BI dashboard request body (local script); batch-monitoring and live-serving sections |
| `resources/` | `ecg12_afib_setup.job.yml`, `ecg12_afib_daily.job.yml` |
| `dev/` | Developer tools, not deployed (excluded from bundle sync, outside the app wheel): `signal_variety/` checks how varied and how separable the generator's ECGs are; `tests/` holds optional pre-deployment checks, `pytest` tests that run on your laptop only (see [Tests](#tests)); see [`dev/README.md`](dev/README.md) |
| `databricks.yml` | Bundle `biosignals-ecg12-afib`: the app, both jobs, `dev` target |

## Notebooks and jobs

| Notebook | Job → task | Writes | Compute |
|---|---|---|---|
| `00_overview` | — | Docs: the whole flow on one page, the 12-lead legend | — |
| `01_generate_training_data` | SETUP → `generate_training_data` | `training_data_12l` (`train` 600 / `val` 200 / `holdout` 200 ECGs by default) + sample plots | standard serverless |
| `02_train_arr_model` | SETUP → `train_arr_model` (after 01) | MLflow run (incl. the layer table); new version of `bio_ecg12_arr` → `@challenger` | serverless GPU (A10, AI v5) |
| `03_train_afib_model` | SETUP → `train_afib_model` (after 01, parallel with 02) | MLflow run (incl. the layer table); new version of `bio_ecg12_afib` → `@challenger` | serverless GPU (A10, AI v5) |
| `04_validate_and_promote` | SETUP → `validate_and_promote` (after 02 + 03) | gate run with hold-out metrics + evaluation artifacts; both pass → `@champion` moves to the challengers, else the task fails | serverless GPU (A10, AI v5) |
| `05_serve_endpoints` | SETUP → `serve_endpoints` (after 04) | endpoints `bio_ecg12_arr`, `bio_ecg12_afib` at `@champion` + inference tables `<endpoint>_payload` | standard serverless |
| `06_generate_batch_data` | DAILY → `simulate_incoming_batch` | `batch_data_12l_<YYYYMMDD>`: simulated incoming data, not real ingestion (acquisition times spread over the clinic day) + sample plot | standard serverless |
| `07_arr_triage` | DAILY → `arr_triage` (after 06) | `arr_triaged_12l_<YYYYMMDD>` (scored with a Spark UDF) | standard serverless, ML v5 |
| `08_afib_classify` | DAILY → `afib_classify` (after 07) | that date's rows of `dailyOrweekly_arr_afib_inference_12l` (scored with a Spark UDF) | standard serverless, ML v5 |

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
| `bio_ecg12_arr` | accuracy ≥ 0.95, ROC AUC ≥ 0.98, sensitivity ≥ 0.97 (a missed abnormal never reaches the AFib model), specificity ≥ 0.90 |
| `bio_ecg12_afib` | accuracy ≥ 0.95, ROC AUC ≥ 0.98, sensitivity ≥ 0.95, share of Other (sinus tachycardia) called AFib ≤ 0.05 |

Both must pass or neither is promoted: on a pass `champion` moves to both challengers (if the second
move fails, the first is put back) and then `challenger` is cleared, so after a pass each model shows
only `@champion`. On a failure nothing moves, `challenger` stays on the failed version (tagged
`validation_status=invalid`) for inspection, the task fails with the reasons and `05_` never runs. So
`challenger` = a candidate awaiting (or that failed) the gate, `champion` = what is served and
batch-scored. A model that was not retrained is re-checked at its `champion` beside the retrained
one; with no `challenger` at all (re-running `04_` alone after a pass) `04_` exits cleanly with a note. The outcome is recorded in an
MLflow run and as model-version tags (`validation_status`, `gate_result`, `holdout_*`). Aliases are
lowercase (`champion`, `challenger`), as Unity Catalog stores them.

**Tag values follow the workspace's tag policies.** The workspace has a tag policy for the key
`validation_status` that allows only `valid` and `invalid`, and Unity Catalog rejects any other value
on a model version. So the gate writes `validation_status` = `valid` / `invalid` (this model against
its own thresholds) and `gate_result` = `passed` / `failed` (both models, the result that decides
promotion; no policy covers that key). The training notebooks write no `validation_status`: until
the gate runs, the `challenger` alias is what marks a version as awaiting validation. No other tag
key this demo writes has a policy (checked 2026-10-30 with
`databricks tag-policies list-tag-policies`). The optional local tests `dev/tests/test_notebooks.py`
(on the notebook code) and `dev/tests/test_setup_flow_local.py` (on a local run of the SETUP
notebooks) fail if a notebook writes a value a policy does not allow; the policies they check are
`GOVERNED_TAG_VALUES` in `dev/tests/conftest.py`, a hand-made copy of the workspace's policies
(re-check it when adding a tag key).

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
Validation and hold-out metrics therefore sit near ceiling (~1.0; the first 12-lead run measured
validation accuracy 1.0 and 0.995). This is a clean synthetic showcase of the MLOps cycle, not a
hard ML problem, and the thresholds are engineering floors for this data, not clinical criteria.
For the same reason the evaluation artifacts look near-perfect (an almost diagonal confusion
matrix, ROC and precision-recall curves in the top corners): they show the evaluation-artifact
workflow on clean synthetic data, not skill on a hard problem, and every figure says so in its
caption.

## Batch path vs live path

- **Batch** (DAILY job), **as deployed: scored on the driver.** `07_` / `08_` load the champion versions and score
  with `scoring.score_driver_side` (as the gate does). The Spark-worker path described next is an untested
  follow-up: its one serverless try (`env_manager="uv"` + the CPU torch index) failed on a fixable install
  setting (uv's default index strategy took `setuptools` from the torch index, which lacks `78.1.1`), not on torch.
  The Spark-worker option: `07_` / `08_` would wrap the champion versions from UC as
  `mlflow.pyfunc.spark_udf`s and score the Spark DataFrame on the Spark workers (`score_spark_udf`
  in `common/scoring.py`), partitions of at most `BATCH_UDF_ROWS_PER_PARTITION` ECGs each (a Python
  UDF on serverless gets 1 GB of memory), so the same notebooks scale past one day's ~100 ECGs.
  `score_spark_udf` also caps each model call at that many ECGs by setting
  `spark.sql.execution.arrow.maxRecordsPerBatch` (spark_udf calls the model once per Arrow batch),
  a cap that holds even if Spark merges partitions. Serverless lets jobs set only a few listed Spark
  properties and this one is not listed, so there the setting may be refused: the task then logs a
  warning and the partition bound applies.
  Each ECG's 12 lead columns reach the model as one flattened array that spark_udf reshapes to the
  `(-1, 12, 2500)` signature; the output columns and types are those of the driver-side scoring
  they replaced. The workers need torch. Both tasks run on standard serverless with the ML v5 base
  environment, but its torch reaches only the driver, not the Spark UDF workers (the first live
  run, with `env_manager="local"`, failed on the workers with "No module named 'torch'"). So
  spark_udf runs with `env_manager="uv"` (`scoring.UDF_ENV_MANAGER`): mlflow builds the model's own
  logged environment (Python 3.12, `mlflow==3.8.1`, `cloudpickle==3.0.0`, `torch==2.9.0`, pandas,
  ...) once on the driver and ships it to the workers, and `scoring.torch_cpu_wheels()` points that
  build at PyTorch's CPU wheel index, so the workers get the CPU build of torch (a Spark UDF gets no
  GPU) instead of PyPI's much larger CUDA build. The endpoints are not called. Results:
  `arr_triaged_12l_<date>`, `dailyOrweekly_arr_afib_inference_12l`. The gate (`04_`) still scores
  its 200 hold-out ECGs on the driver (`score_driver_side`): it runs on serverless GPU, whose
  Spark workers have no torch.
- **Live** (the app): each *Triage → Classify* sends one ECG to `bio_ecg12_arr` and, when flagged,
  to `bio_ecg12_afib`; AI Gateway logs every request/response to `bio_ecg12_arr_payload` /
  `bio_ecg12_afib_payload`. One 12-lead ECG is ~650 KB of JSON (measured on the first 12-lead run's
  logged rows), inside the inference table's 1 MiB logging limit; a request above 1 MiB is served
  but logged with `MAX_REQUEST_SIZE_EXCEEDED` and a null payload, so `05_`'s smoke test also sends
  one ECG per request.

The dashboard builder keeps them apart: **Batch monitoring** (funnel, arrivals per hour, triage by
rhythm, AFib-probability distribution, AFib by age band, daily trend, worklist) reads the batch
tables; **Live serving** (requests, model execution time and payloads not logged per endpoint, the
returned probabilities, the latest requests) reads the payload tables. It parses only the small
response JSON; `execution_duration_ms` is the model's inference time, not end-to-end latency.

Run SETUP first (and again to retrain); DAILY needs its champion models. The DAILY job has a paused
06:00 UTC schedule and one `run_date` job parameter (default `{{job.start_time.iso_date}}`) that
every task reads; set `run_date=YYYY-MM-DD` to backfill a day. Re-running a date gives the same
rows and never duplicates them: per-date tables are overwritten, and the results table replaces
that date's 12-lead rows in one Delta transaction.

The results table `dailyOrweekly_arr_afib_inference_12l` belongs to this demo alone: the live
8-lead demo's `dailyOrweekly_arr_afib_inference` (no `batch_date` column) is never read or
written, and `uc_io.write_date_slice` refuses any existing table without `batch_date`.

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
- **Rhythms.** NSR (60–100 bpm), AFib (irregular RR, no P waves, f-waves, 100–160 bpm), Other =
  sinus tachycardia (110–160 bpm). All classes share one beat model, so they differ only in rhythm.
- **Patient record** (synthetic, non-PII): `pseudo_patient_id`, `mrn`, `age_years`, `sex`,
  `height_cm`, `weight_kg`, `bmi` (from height and weight), `bp_systolic_mmhg`,
  `bp_diastolic_mmhg`, `acquisition_datetime_utc`, `heart_rate_bpm` (measured from the ECG),
  `sampling_rate_hz`, `n_leads`, `duration_s`. Demographics are fixed per `pseudo_patient_id`;
  AFib is assigned more often to older patients. Acquisition times fall in clinic hours
  (07:00–19:00 UTC, a busier morning), so a day's batch reads like an incoming stream.

## Build, validate, deploy

App names are workspace-global, so the bundle never takes a full app name from you. You supply
only its end, the `app_suffix` variable (`<suffix>` below: required, no default; lowercase
letters, digits, hyphens, at most 19 characters, since app names max out at 30 and the prefix
takes 11), and the app is always named `ecg12-demo-<suffix>`. Because of that fixed prefix the
name can never be `ekg-demo` or any other app whose name does not start with `ecg12-demo-`
(`dev/tests/test_repo_consistency.py` checks this). Only another deploy of this bundle uses the
prefix, so check `databricks apps list -p fevm-mmt-aws-usw2` that `ecg12-demo-<suffix>` is not
already in use, then pass the suffix on every bundle command (or
`export BUNDLE_VAR_app_suffix=<suffix>`):

```bash
cd projects/ecg12-afib
(cd app && ./build_from_lockfiles.sh)                                  # app/.build (wheel)
databricks bundle validate --strict -p fevm-mmt-aws-usw2 --var app_suffix=<suffix>
databricks bundle deploy -t dev -p fevm-mmt-aws-usw2 --var app_suffix=<suffix>
databricks bundle run ecg12_afib_setup -t dev -p fevm-mmt-aws-usw2 --var app_suffix=<suffix>
databricks bundle run ecg12_afib_daily -t dev -p fevm-mmt-aws-usw2 --var app_suffix=<suffix>
databricks bundle run ecg12_afib_app -t dev -p fevm-mmt-aws-usw2 --var app_suffix=<suffix>
```

Files land in `/Workspace/Users/<user>/.bundle/biosignals-ecg12-afib/dev/files/`; the app runs
from `…/files/app/.build`, never the live app's `/Workspace/Users/<user>/apps/ekg_demo_react_build`.
Development mode prefixes the job names (`[dev <user>] biosignals_ecg12_afib_setup`, `…_daily`)
but not the app name, which stays `ecg12-demo-<suffix>`.

The notebooks also run in place from a Databricks Git folder of this repo (they find `common/`
one folder up). The app cannot be deployed from a Git folder: the compiled UI and wheel
(`app/.build/`) are not committed and need `apx`, `bun` and `uv` locally.

## Live steps still to do (not done here: each one changes the workspace)

0. Create the schema `mmt_aws_usw2_catalog.biosignals` (it does not exist yet). The first 12-lead
   run's models, tables and payload tables stay in `ekg_afib`; cleaning them up is a separate decision.
1. Run the SETUP job: writes `training_data_12l`, trains and registers `bio_ecg12_arr` and
   `bio_ecg12_afib` (12 channels, GPU) as `challenger`, gates them on the hold-out split (and logs
   the evaluation artifacts), moves `champion`, then creates the two NEW serving endpoints
   `bio_ecg12_arr` / `bio_ecg12_afib` with inference tables `bio_ecg12_arr_payload` /
   `bio_ecg12_afib_payload` in `biosignals`. The first run's `arr_model_12l` / `afib_demo_12l`
   endpoints (serving the `ekg_afib` v1 models) are left as they are; retiring them is a separate
   decision.
2. Run the DAILY job at least once (creates `batch_data_12l_<date>`, `arr_triaged_12l_<date>`
   and `dailyOrweekly_arr_afib_inference_12l`), ideally over several `run_date`s so the trend has
   history, then unpause its schedule. This is also the first live run of the Spark-UDF batch
   scoring on standard serverless with the ML v5 environment (`07_` / `08_`): check that both
   tasks succeed before relying on it (the local tests run the same code on a local Spark session,
   with the real `mlflow.pyfunc.spark_udf` checked on stand-in models).
3. Create the 12-lead dashboard (`dashboards/build_ecg_demo_dashboard.py`, then
   `databricks lakeview create`), set `DASHBOARD_ID` in `common/config.py` to its id, copy
   `common/config.py` into the app (`app/src/ecg_demo_react/backend/common/`), then rebuild and
   redeploy the app. `DASHBOARD_ID` is unset (`None`) until then, and the app's Batch tab shows a
   "12-lead dashboard not configured yet" note instead of an embed; it never shows the live 8-lead
   demo's dashboard in its place.
4. Grant the app's service principal **CAN_QUERY** on the new endpoints `bio_ecg12_arr` and
   `bio_ecg12_afib` (or, once they exist, declare them as app `resources` in `databricks.yml`). An
   app deployed from an earlier build still calls `arr_model_12l` / `afib_demo_12l` until it is
   rebuilt and redeployed with this `config.py`. The earlier manual deploy of this demo,
   `ekg12-demo-mmt` (old app-name prefix), is such an app: a bundle deploy creates the new app
   `ecg12-demo-<suffix>` beside it and never updates it; retiring it is a separate decision.
5. Add `ecg12-demo-<suffix>-7474658466980277.aws.databricksapps.com` to the workspace's dashboard
   embedding allowlist (workspace admin setting).
6. Grant users **CAN_USE** on the new app.

## Tests

**Optional pre-deployment checks.** The `pytest` tests (pytest is a Python test runner) in
[`dev/tests/`](dev/tests/README.md) run on your laptop only (Python 3.11; nothing runs them
automatically and they are not deployed). They check code logic: the 12-lead formulas, the patient
record, training-data splits, the validation gate promoting both models or neither, batch-scoring
output, and that names come from config. They did **not** catch the two problems the first live
deploy hit (a workspace tag policy rejecting a tag value, and serverless Spark workers without
torch); those only showed up in live runs. **The real pre-deployment check is a live run:** the
SETUP job, then one DAILY run, before unpausing the schedule (steps 1 and 2 under "Live steps still
to do" above).

```bash
cd projects/ecg12-afib
dev/tests/run_tests_python311.sh                                       # all of them, on Python 3.11
dev/tests/run_tests_python311.sh dev/tests/test_scoring_and_io.py -v   # one file
```

Why Python 3.11, the full pinned `uv` command, what each file checks and which tests are known to
be out of date: [`dev/tests/README.md`](dev/tests/README.md).
