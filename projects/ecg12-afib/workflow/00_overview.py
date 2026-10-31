# Databricks notebook source
# MAGIC %md
# MAGIC ### OVERVIEW of the 12-lead ECG/AFib demo
# MAGIC
# MAGIC A two-model **triage → classify** cascade on a synthetic **clinical 12-lead ECG**: an arrhythmia
# MAGIC (ARR) triage model flags abnormal ECGs, then an AFib classifier scores only the flagged ones.
# MAGIC Both are small custom PyTorch 1D-CNNs with 12 input channels. The demo runs the full model cycle
# MAGIC on Databricks: **generate data → train → validate → promote → serve → score daily → monitor**.
# MAGIC Batch (the DAILY notebooks) and live (the app, through two serving endpoints) use the same Unity
# MAGIC Catalog (UC) models. Tasks communicate only through UC Delta tables and model aliases.
# MAGIC
# MAGIC ```
# MAGIC SETUP  01_generate_training_data ─┬─> 02_train_arr_model ──┐
# MAGIC        training_data_12l          └─> 03_train_afib_model ─┴─> 04_validate_and_promote ──> 05_serve_endpoints ──> app
# MAGIC        (train / val / holdout)         register @challenger     hold-out gate -> @champion    endpoints + *_payload tables
# MAGIC DAILY  06_generate_batch_data ──> 07_arr_triage ──> 08_afib_classify ──> dashboard
# MAGIC        batch_data_12l_<d>         arr_triaged_12l_<d>   dailyOrweekly_arr_afib_inference_12l
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC > **A reference pattern, not a fixed pipeline.** The tasks, the models, the Unity Catalog
# MAGIC > catalog/schema, the rhythm classes and the lead count are all swappable: names, splits,
# MAGIC > training settings and validation thresholds live in `common/config.py`, the data in
# MAGIC > `common/ecg_signal_gen.py`, the model code in `02_` / `03_`, and the task graph in
# MAGIC > `resources/*.job.yml`. Replace the synthetic generator with a real ECG source, or the CNNs with
# MAGIC > another model, and the generate → train → gate → promote → serve → monitor cycle stays the same.

# COMMAND ----------

# MAGIC %md
# MAGIC **Notebook map** (the jobs are defined in `resources/ecg12_afib_setup.job.yml` and
# MAGIC `resources/ecg12_afib_daily.job.yml`):
# MAGIC
# MAGIC | Notebook | Role | Runtime | Job → task |
# MAGIC |---|---|---|---|
# MAGIC | `00_overview` | This page | — | — |
# MAGIC | `01_generate_training_data` | Synthetic labeled 12-lead ECGs from the shared generator → `training_data_12l` with a `train` / `val` / `holdout` split; sample plots of the signals | standard serverless | SETUP → `generate_training_data` |
# MAGIC | `02_train_arr_model` | Train the triage model on `train` (+ `val` metrics), log to MLflow (incl. the layer table), register `bio_ecg12_arr` → `@challenger` | serverless GPU (A10, AI v5) | SETUP → `train_arr_model` (after 01) |
# MAGIC | `03_train_afib_model` | Train the AFib classifier the same way → `bio_ecg12_afib` `@challenger` | serverless GPU (A10, AI v5) | SETUP → `train_afib_model` (after 01, parallel with 02) |
# MAGIC | `04_validate_and_promote` | **Gate:** score both challengers on `holdout`, check the thresholds, log the evaluation artifacts; both pass → `@champion` moves to them, else the job stops | serverless GPU (A10, AI v5) | SETUP → `validate_and_promote` (after 02 + 03) |
# MAGIC | `05_serve_endpoints` | Create / update endpoints `bio_ecg12_arr` + `bio_ecg12_afib` at the champion versions, AI Gateway inference tables, smoke query | standard serverless | SETUP → `serve_endpoints` (after 04) |
# MAGIC | `06_generate_batch_data` | One day's new synthetic 12-lead ECGs + synthetic patient records, acquisition times spread over the day → `batch_data_12l_<YYYYMMDD>` | standard serverless | DAILY → `simulate_incoming_batch` (simulated incoming data, not real ingestion) |
# MAGIC | `07_arr_triage` | Score all 12 leads with `bio_ecg12_arr@champion` as a Spark UDF → `arr_prob`, `arr_flag` → `arr_triaged_12l_<YYYYMMDD>` | standard serverless (ML v5) | DAILY → `arr_triage` (after 06) |
# MAGIC | `08_afib_classify` | Score the flagged rows with `bio_ecg12_afib@champion` as a Spark UDF → that date's rows of `dailyOrweekly_arr_afib_inference_12l` | standard serverless (ML v5) | DAILY → `afib_classify` (after 07) |
# MAGIC
# MAGIC Run the SETUP job first (and again to retrain): the DAILY job scores with the champion models.

# COMMAND ----------

# MAGIC %md
# MAGIC **Model lifecycle: challenger → validation gate → champion.** `02_` / `03_` register every new
# MAGIC version under the alias `challenger`; nothing they do changes what is served. `04_` scores each
# MAGIC challenger on the `holdout` split, which the training notebooks never read (they use `train` and
# MAGIC `val`), at the Delta version of `training_data_12l` the challenger was trained beside. Only when
# MAGIC **both** models pass `VALIDATION_THRESHOLDS` (`common/config.py`) does `champion` move to them,
# MAGIC and `challenger` is then cleared: `challenger` marks a candidate awaiting (or that failed) the gate,
# MAGIC `champion` what is served. Otherwise nothing moves (`challenger` stays on the failed version) and
# MAGIC the SETUP job stops before `05_` serves anything. Serving (`05_`) and
# MAGIC daily scoring (`07_`, `08_`) always use `champion`. Aliases are lowercase because Unity Catalog
# MAGIC stores them lowercase.
# MAGIC
# MAGIC | Model | Hold-out thresholds (default) |
# MAGIC |---|---|
# MAGIC | `bio_ecg12_arr` | accuracy ≥ 0.95, ROC AUC ≥ 0.98, sensitivity ≥ 0.97 (a missed abnormal never reaches the AFib model), specificity ≥ 0.90 |
# MAGIC | `bio_ecg12_afib` | accuracy ≥ 0.95, ROC AUC ≥ 0.98, sensitivity ≥ 0.95, share of Other (sinus tachycardia) called AFib ≤ 0.05 |
# MAGIC
# MAGIC MLflow records each step: the training runs (params, per-epoch and final validation metrics,
# MAGIC tags, the layer table `model_architecture.md`, the model with signature + input example), the
# MAGIC gate run (thresholds, hold-out metrics, and per model under `holdout_eval/<model>/` a confusion
# MAGIC matrix with raw counts, ROC and precision-recall curves, a per-class precision / recall / F1
# MAGIC table and a README), and tags on each model version (training data version,
# MAGIC `validation_status`, `gate_result`, `holdout_*`, `holdout_eval_artifacts`). `validation_status`
# MAGIC is set by the gate only, to `valid` or `invalid`: the only values the workspace's tag policy for
# MAGIC that key allows. Each endpoint has the same name as
# MAGIC its model. (The first 12-lead run, in schema `ekg_afib`, used `arr_model_12l` / `afib_demo_12l`;
# MAGIC those models and endpoints are left as they are.)
# MAGIC
# MAGIC **Why the metrics sit near 1.0.** The synthetic rhythms are trivially separable by design: heart
# MAGIC rate alone separates triage (the generator draws NSR at 60–100 bpm, AFib and Other at 100–160
# MAGIC bpm), and beat-to-beat irregularity separates AFib from Other (RR intervals vary ~20 % in AFib,
# MAGIC ~3 % in sinus rhythm). So validation and hold-out metrics sit near ceiling (~1.0), and each model
# MAGIC trains once with one fixed configuration (`TRAIN_PARAMS`), no hyperparameter search. This is a
# MAGIC clean, synthetic **showcase of the MLOps cycle**, not a hard ML problem; the thresholds are
# MAGIC engineering floors for this data, not clinical acceptance criteria.
# MAGIC
# MAGIC **The same goes for the evaluation artifacts.** The gate's confusion matrices, ROC and
# MAGIC precision-recall curves come out near-perfect (an almost diagonal matrix, curves in the
# MAGIC top corners) because the data is separable by design. They are a clean synthetic showcase of
# MAGIC the evaluation-artifact workflow (what gets logged, where, and how a reviewer reads it), not
# MAGIC evidence that the models would do well on a hard, real clinical problem; every figure says so
# MAGIC in its caption.

# COMMAND ----------

# MAGIC %md
# MAGIC **How the 1D CNN reads the 12 leads.** Both models are the same small 1D convolutional neural
# MAGIC network (CNN), 10,514 learned parameters (weights), defined in `02_` / `03_` (`ArrCNN`,
# MAGIC `AfibCNN`). Each training run logs the layer-by-layer table as `model_architecture.md`.
# MAGIC
# MAGIC 1. **Input:** one ECG as a grid of 12 rows (the leads) x 2500 columns (time: 10 s at 250 Hz).
# MAGIC 2. **Convolution layers, sliding along time.** A convolution filter is a small learned pattern
# MAGIC    detector that slides along the time axis and scores how well each stretch of signal matches it.
# MAGIC    The first layer has `in_channels = N_LEADS = 12`: each of its 16 filters spans all 12 leads at
# MAGIC    once and 7 samples (28 ms) of time, so it can combine what different leads see at the same
# MAGIC    moment. Three such blocks (16 → 32 → 64 filters, each followed by batch normalization and a
# MAGIC    ReLU, the rectified linear unit that zeroes negative values) find patterns of patterns,
# MAGIC    over ever longer stretches of the beat sequence.
# MAGIC 3. **Pooling, shrinking time.** Each convolution steps 2 samples at a time, and max pooling after
# MAGIC    the first two blocks keeps the strongest response of each neighbouring pair, so 2500 samples
# MAGIC    shrink to 78 time steps; global average pooling then averages each of the 64 filters over the
# MAGIC    whole 10 s.
# MAGIC 4. **Dense layer → output.** One dense (fully connected) layer turns those 64 numbers into 2 class
# MAGIC    scores, which softmax turns into probabilities: `arr_prob` (abnormal) for triage,
# MAGIC    `[p_not_afib, p_afib]` for the AFib model.

# COMMAND ----------

# MAGIC %md
# MAGIC **The 12 leads.** A standard ECG gives *"12 different vector views of the heart's electrical
# MAGIC activity"*, measured between electrodes on the limbs and the chest wall: *"Six of these views are
# MAGIC in a vertical (coronal) plane using frontal leads I, II, and III and limb leads aVR, aVL, and aVF,
# MAGIC and 6 are in a horizontal (axial) plane using precordial leads V1, V2, V3, V4, V5, and V6."*
# MAGIC (Merck Manual Professional, *Electrocardiography*, reviewed June 2026.)
# MAGIC
# MAGIC | Group | Leads | Plane | In this demo |
# MAGIC |---|---|---|---|
# MAGIC | Limb (frontal) leads | I, II, III | frontal (vertical, coronal) | I and II recorded; III = II − I (Einthoven's law) |
# MAGIC | Augmented limb leads | aVR, aVL, aVF | frontal (vertical, coronal) | derived from I and II: aVR = −(I + II)/2, aVL = I − II/2, aVF = II − I/2 (Goldberger's leads) |
# MAGIC | Precordial (chest) leads | V1–V6 | horizontal (axial) | recorded |
# MAGIC
# MAGIC **Why 12?** Every lead looks at the same heartbeat from a different direction, so the same beat
# MAGIC has a different shape in each lead (in the generator: each wave is a cardiac vector projected
# MAGIC onto the lead's angle, and the R wave grows from V1 to V5). **Recorded vs derived:** only eight
# MAGIC leads are independent. The generator synthesizes I, II and V1–V6 and computes III, aVR, aVL and
# MAGIC aVF from I and II with the formulas above (`common/ecg_signal_gen.py`, `derive_limb_leads`), so
# MAGIC the formulas hold exactly on every stored ECG. Both models receive all 12 leads, the clinical
# MAGIC format, as a `(-1, 12, 2500)` tensor (mV, 250 Hz, 10 s).

# COMMAND ----------

# MAGIC %md
# MAGIC **Two data paths, two monitoring views.**
# MAGIC
# MAGIC * **Batch path** (DAILY job): `07_` and `08_` wrap the champion versions from Unity Catalog as
# MAGIC   `mlflow.pyfunc.spark_udf`s and score on the Spark workers (`score_spark_udf` in
# MAGIC   `common/scoring.py`), so scoring is distributed instead of collected to the driver; they never
# MAGIC   call the endpoints. They run on standard serverless with the ML v5 environment; the workers
# MAGIC   need torch, so spark_udf gives them the model's own logged environment, rebuilt with uv
# MAGIC   (CPU build of torch). Results land in `arr_triaged_12l_<date>` and
# MAGIC   `dailyOrweekly_arr_afib_inference_12l`.
# MAGIC * **Live path** (the app): every *Triage → Classify* click calls the serving endpoints, which log
# MAGIC   each request and response to their AI Gateway inference tables `bio_ecg12_arr_payload` and
# MAGIC   `bio_ecg12_afib_payload` (one 12-lead ECG is ~650 KB of JSON, inside the 1 MiB logging limit).
# MAGIC
# MAGIC The 12-lead dashboard (`dashboards/build_ecg_demo_dashboard.py`) keeps them apart: a
# MAGIC *Batch monitoring* section reads the batch tables, a *Live serving* section reads the payload
# MAGIC tables.

# COMMAND ----------

# MAGIC %md
# MAGIC **Shared code — `common/`** (one folder up; each notebook adds it to `sys.path`):
# MAGIC
# MAGIC | Module | What it holds |
# MAGIC |---|---|
# MAGIC | `config.py` | Every name and setting: catalog/schema, `N_LEADS = 12`, model + endpoint names, aliases, table names, training split, `TRAIN_PARAMS`, `VALIDATION_THRESHOLDS`, the batch-scoring partition size, the `run_date` parameter |
# MAGIC | `ecg_signal_gen.py` | The one ECG generator: leads I, II, V1–V6 synthesized; III, aVR, aVL, aVF derived. NSR / AFib / Other (sinus tachycardia), mV, 250 Hz, 10 s |
# MAGIC | `training_data.py` | The training dataset: generate the splits, its schema, read one split at a Delta version |
# MAGIC | `training.py` | The PyTorch training loop with per-epoch validation metrics, and the layer table of a model (neither is part of the logged models) |
# MAGIC | `evaluation.py` | Accuracy, ROC AUC, average precision, sensitivity, specificity, per-rhythm call rates; the threshold check of the gate; confusion counts, ROC / precision-recall curves and the per-class table for the gate's artifacts |
# MAGIC | `eval_plots.py` | The gate's evaluation figures: confusion matrix, ROC curve, precision-recall curve |
# MAGIC | `ecg_plots.py` | Sample-signal plots (rhythm strips, 12-lead layout) for `01_` and `06_` |
# MAGIC | `patient_info.py` | Synthetic, non-PII patient + acquisition record, fixed per `pseudo_patient_id`; heart rate measured from the ECG; acquisition times over the clinic day |
# MAGIC | `scoring.py` | Resolve an alias to a concrete version (that version scores and is recorded); Spark-UDF scoring for the daily batch; driver-side scoring + join for the gate |
# MAGIC | `uc_io.py` | Re-run-safe Delta writes; Delta version + time-travel reads |
# MAGIC
# MAGIC The model classes (`ArrCNN`, `AfibCNN`, their pyfunc wrappers and input reshape) stay **inside**
# MAGIC `02_` / `03_`, so MLflow pickles them by value and the serving endpoints load them without `common`.
# MAGIC The app keeps a byte-identical copy of `config`, `ecg_signal_gen` and `patient_info`.
# MAGIC Developer analysis scripts (e.g. the generator's signal-variety check) and the optional local
# MAGIC tests (`dev/tests/`) live in `dev/`, which is not deployed.

# COMMAND ----------

# MAGIC %md
# MAGIC **Parameters.** Notebooks read [dbutils.widgets](https://docs.databricks.com/aws/en/dev-tools/databricks-utils)
# MAGIC set by the jobs' parameters: `catalog_name`, `schema_name` (defaults from `common/config.py`) and,
# MAGIC for `06_`–`08_`, `run_date` (`YYYY-MM-DD`; empty = today UTC). The DAILY job passes
# MAGIC `{{job.start_time.iso_date}}` to all three tasks so they agree on one date; run it with
# MAGIC `run_date=YYYY-MM-DD` to backfill. Re-running a date gives the same rows and never duplicates them.
# MAGIC
# MAGIC **Deploy / run (bundle `biosignals-ecg12-afib`, profile `fevm-mmt-aws-usw2`):**
# MAGIC `databricks bundle deploy -t dev --var app_suffix=<suffix>`, then
# MAGIC `databricks bundle run ecg12_afib_setup -t dev` and `databricks bundle run ecg12_afib_daily -t dev`
# MAGIC (same `--var`). `app_suffix` is only the end of the app's name, which is always
# MAGIC `ecg12-demo-<suffix>`, so it can never be `ekg-demo`; it has no default. See the demo
# MAGIC `README.md` for the remaining live steps (schema creation, dashboard creation and embedding,
# MAGIC endpoint permissions for the app, app access).
