# Architecture: folders, notebooks, jobs, batch vs live

[← ecg12-afib README](../README.md)

## Layout

| Path | What it is |
|---|---|
| `databricks.yml` | Bundle `biosignals-ecg12-afib`: variables, targets `no_app` (jobs + dashboard) and `full` (+ the app) |
| `deploy.sh` | Full install from a laptop (target `full`): build, validate, deploy, SETUP run, permissions, app start, embedding allowlist |
| `resources/` | `ecg12_afib_setup.job.yml`, `ecg12_afib_daily.job.yml` (the jobs), `ecg12_afib.dashboard.yml` (the dashboard) |
| `workflow/` | Notebooks `00_`–`08_` (below). Each puts `..` on `sys.path` and imports `common` |
| `common/` | Shared code, single source of truth: `config.py` (names, `N_LEADS = 12`, aliases, training split + settings, validation thresholds, run date, the catalog/schema parameter check), `ecg_signal_gen.py` (12-lead generator), `training_data.py` (the persisted training dataset), `training.py` (PyTorch training loop, layer table), `evaluation.py` (metrics, the gate's threshold check, confusion counts, ROC / precision-recall curves, per-class table), `eval_plots.py` (the gate's evaluation figures), `ecg_plots.py` (sample-signal plots), `patient_info.py` (synthetic patient record), `scoring.py` (alias → version, driver-side scoring), `uc_io.py` (re-run-safe Delta writes, Delta versions) |
| `dashboards/` | `ecg12_afib.lvdash.json` (the dashboard the bundle deploys) and `build_ecg_demo_dashboard.py`, which generates it |
| `app/` | APX app (React UI + FastAPI backend). `backend/common/` is a byte-identical copy of `common/{__init__,config,ecg_signal_gen,patient_info}.py`; the build fails if it drifts. Build with `app/build_from_lockfiles.sh`, see [`app/README.md`](../app/README.md) |
| `dev/` | Developer tools, not deployed (excluded from bundle sync, outside the app wheel): `signal_variety/` checks how varied and how separable the generator's ECGs are; `tests/` holds optional local checks (see [Tests](../dev/tests/README.md)); see [`dev/README.md`](../dev/README.md) |

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
