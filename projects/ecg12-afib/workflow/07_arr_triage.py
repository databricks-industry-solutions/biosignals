# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — ARR (triage) batch inference, 12-lead  [standard serverless, ML environment / driver-side scoring]
# MAGIC
# MAGIC **DAILY job, task 2 of 3** (after `06_generate_batch_data`). Scores the day's batch
# MAGIC (`batch_data_12l_<date>`) with the triage model **`bio_ecg12_arr@champion`** (the version the
# MAGIC SETUP job's validation gate promoted), sending all 12 leads as a `(-1, 12, 2500)` tensor, and
# MAGIC writes **`arr_triaged_12l_<date>`**: the batch plus `arr_prob`, `arr_flag` (1 = abnormal, goes on
# MAGIC to the AFib model in `08_`) and `arr_model_info`.
# MAGIC
# MAGIC It also writes that date's slice of **`arr_triage_history_12l`**: the same rows without the ECG
# MAGIC arrays, every day under one fixed table name, which the dashboard reads.
# MAGIC
# MAGIC **Scored on the driver** (`scoring.score_driver_side`, as in the gate): the champion version is
# MAGIC loaded with `mlflow.pyfunc.load_model`, the day's ~100 ECGs are collected to the driver as one
# MAGIC `(n, 12, 2500)` tensor, and the predictions are joined back by `pseudo_patient_id`.
# MAGIC
# MAGIC **Batch path, not the endpoint:** the serving endpoints are not called, so these rows never
# MAGIC appear in the endpoints' `*_payload` inference tables. `champion` is resolved to a concrete version
# MAGIC first; that exact version scores and is recorded in `arr_model_info`. Re-running a date overwrites
# MAGIC `arr_triaged_12l_<date>` and replaces that date's rows of `arr_triage_history_12l`.
# MAGIC
# MAGIC **Compute:** standard serverless (CPU), base environment **ML v5** (`databricks_ml_v5`, which
# MAGIC ships torch; no %pip). Parameters: `run_date`, `catalog_name` (required), `schema_name`.

# COMMAND ----------

# DBTITLE 1,Shared config + scoring helpers (common/ is one folder up from workflow/)
import os
import sys

sys.path.insert(0, os.path.abspath(".."))   # notebook working dir = workflow/

from common import config, scoring, uc_io
from common.ecg_signal_gen import N_SAMPLES

# COMMAND ----------

# DBTITLE 1,Parameters (usable as a job task; one shared run_date)
dbutils.widgets.text(config.RUN_DATE_PARAM, "", label="run_date (YYYY-MM-DD, empty = today UTC)")
dbutils.widgets.text("catalog_name", "", label="UC_catalog_name (required)")
dbutils.widgets.text("schema_name", config.SCHEMA, label="UC_schema_name")
dbutils.widgets.text("arr_model_name", config.ARR_MODEL_NAME, label="arr_model_name")

run_date = config.resolve_run_date(dbutils.widgets.get(config.RUN_DATE_PARAM))
catalog_name, schema_name = config.uc_location(dbutils.widgets.get("catalog_name"),
                                               dbutils.widgets.get("schema_name"))
read_table = config.daily_table(config.BATCH_TABLE_PREFIX, run_date, catalog_name, schema_name)
write_table = config.daily_table(config.TRIAGED_TABLE_PREFIX, run_date, catalog_name, schema_name)
history_table = config.uc_name(config.TRIAGE_HISTORY_TABLE, catalog_name, schema_name)
arr_model_full = config.uc_name(dbutils.widgets.get("arr_model_name"), catalog_name, schema_name)
print(f"run_date: {run_date}\nread : {read_table}\nwrite: {write_table}, {history_table}\n"
      f"model: {arr_model_full}@{config.CHAMPION_ALIAS}")

# COMMAND ----------

# DBTITLE 1,Resolve champion -> version, load exactly that version
import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
arr = scoring.resolve_alias(MlflowClient(registry_uri="databricks-uc"), arr_model_full)
arr_model = mlflow.pyfunc.load_model(arr.uri)
print(f"scoring with {arr.info} (driver-side)")

# COMMAND ----------

# DBTITLE 1,Read the batch and score all 12 leads on the driver
batch_SDF = spark.table(read_table)
ecg_cols = config.ecg_columns(config.N_LEADS)
print(f"batch rows: {batch_SDF.count()} | ecg columns: {ecg_cols}")

triaged_SDF = scoring.score_driver_side(
    spark,
    batch_SDF,
    lambda X: scoring.arr_columns(arr_model.predict(X)),   # DataFrame[arr_prob, arr_flag]
    output_cols=scoring.ARR_OUTPUT_COLS,
    ecg_cols=ecg_cols,
    n_samples=N_SAMPLES,
    model_info_col="arr_model_info",
    model_info=arr.info,
)

# COMMAND ----------

# DBTITLE 1,Write arr_triaged_12l_<date> (overwrite = re-run safe)
uc_io.overwrite_table(spark, triaged_SDF, write_table)
print(f"wrote {write_table}")

# COMMAND ----------

# DBTITLE 1,Write this date's slice of arr_triage_history_12l without the ECG arrays (re-run safe)
# Every day's triage results under one fixed name, for the dashboard (a per-date table name cannot
# be a dashboard dataset).
mode = uc_io.write_date_slice(spark, spark.table(write_table).drop(*ecg_cols), history_table, run_date)
print(f"{mode}: {history_table} ({config.BATCH_DATE_COL} = {run_date})")

# COMMAND ----------

# DBTITLE 1,Quick check — flagged vs not by true rhythm (flagged rows proceed to the AFib model)
display(spark.table(write_table).groupBy("rhythm_label", "arr_flag").count().orderBy("rhythm_label", "arr_flag"))
