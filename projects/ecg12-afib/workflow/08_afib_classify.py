# Databricks notebook source
# MAGIC %md
# MAGIC # 08 — AFib batch classification of the flagged ECGs, 12-lead  [standard serverless, ML environment / driver-side scoring]
# MAGIC
# MAGIC **DAILY job, task 3 of 3** (after `07_arr_triage`). Reads `arr_triaged_12l_<date>`, keeps the
# MAGIC rows the triage model flagged (`arr_flag == 1`), scores them with **`bio_ecg12_afib@champion`**
# MAGIC (all 12 leads, `(-1, 12, 2500)`), and writes that date's slice of the accumulating results
# MAGIC table **`dailyOrweekly_arr_afib_inference_12l`** (read by the 12-lead dashboard and the app):
# MAGIC
# MAGIC * `afib_predictions_0` / `afib_predictions_1` = `[p_not_afib, p_afib]`, plus `afib_model_info`;
# MAGIC * the patient record, `rhythm_label`, the triage columns, and `batch_date`; the ECG arrays are
# MAGIC   dropped to keep the table small.
# MAGIC
# MAGIC **Re-run safe:** the date's rows are replaced in one Delta transaction (`replaceWhere` on
# MAGIC `batch_date`), so re-running a date never duplicates rows. The live 8-lead demo's
# MAGIC `dailyOrweekly_arr_afib_inference` is a different table and is never written to.
# MAGIC
# MAGIC **Distributed batch scoring, like `07_`:** the champion version is wrapped as an
# MAGIC `mlflow.pyfunc.spark_udf` and the Spark workers score the flagged ECGs (`score_spark_udf` in
# MAGIC `common/scoring.py`); the result array `[p_not_afib, p_afib]` becomes `afib_predictions_0` /
# MAGIC `afib_predictions_1`. `champion` is resolved to a concrete version first; that exact version
# MAGIC scores and is recorded in `afib_model_info`. **Batch path, not the endpoint:** the serving
# MAGIC endpoints and their `*_payload` inference tables are not involved.
# MAGIC
# MAGIC **Scored on the driver** (`scoring.score_driver_side`, as in the gate). Spark-worker scoring
# MAGIC (`scoring.score_spark_udf`) is an untested follow-up: its one serverless try (`env_manager="uv"` with
# MAGIC the CPU torch index) failed on a fixable install setting (uv's index strategy took setuptools from the
# MAGIC torch index), not on torch itself.
# MAGIC
# MAGIC **Compute:** standard serverless (CPU), base environment **ML v5** (same as `07_`; no %pip). The
# MAGIC workers need torch, so, as in `07_`, spark_udf rebuilds the model's own environment with
# MAGIC `env_manager="uv"` inside `scoring.torch_cpu_wheels()` (CPU build of torch).

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
dbutils.widgets.text("afib_model_name", config.AFIB_MODEL_NAME, label="afib_model_name")

run_date = config.resolve_run_date(dbutils.widgets.get(config.RUN_DATE_PARAM))
catalog_name, schema_name = config.uc_location(dbutils.widgets.get("catalog_name"),
                                               dbutils.widgets.get("schema_name"))
read_table = config.daily_table(config.TRIAGED_TABLE_PREFIX, run_date, catalog_name, schema_name)
results_table = config.uc_name(config.INFERENCE_TABLE, catalog_name, schema_name)
afib_model_full = config.uc_name(dbutils.widgets.get("afib_model_name"), catalog_name, schema_name)
print(f"run_date: {run_date}\nread : {read_table}\nwrite: {results_table} ({config.BATCH_DATE_COL} = {run_date})\n"
      f"model: {afib_model_full}@{config.CHAMPION_ALIAS}")

# COMMAND ----------

# DBTITLE 1,Resolve champion -> version, wrap exactly that version as a Spark UDF
import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
afib = scoring.resolve_alias(MlflowClient(registry_uri="databricks-uc"), afib_model_full)
afib_model = mlflow.pyfunc.load_model(afib.uri)
print(f"scoring with {afib.info} (driver-side)")

# COMMAND ----------

# DBTITLE 1,Keep the flagged rows (arr_flag == 1, set by 07_'s triage model)
import pyspark.sql.functions as F

triaged_SDF = spark.table(read_table)
display(triaged_SDF.groupBy("arr_flag").count())
flagged_SDF = triaged_SDF.filter(F.col("arr_flag") == 1.0)

# COMMAND ----------

# DBTITLE 1,Score the flagged rows on the driver (all 12 leads) and drop the ECG arrays
ecg_cols = config.ecg_columns(config.N_LEADS)
classified_SDF = scoring.score_driver_side(
    spark,
    flagged_SDF,
    lambda X: scoring.afib_columns(afib_model.predict(X)),  # [p_not_afib, p_afib] -> afib_predictions_0 / _1
    output_cols=scoring.AFIB_OUTPUT_COLS,
    ecg_cols=ecg_cols,
    n_samples=N_SAMPLES,
    model_info_col="afib_model_info",
    model_info=afib.info,
).drop(*ecg_cols)

# COMMAND ----------

# DBTITLE 1,Write this date's slice of dailyOrweekly_arr_afib_inference_12l (re-run safe)
mode = uc_io.write_date_slice(spark, classified_SDF, results_table, run_date)
print(f"{mode}: {results_table} ({config.BATCH_DATE_COL} = {run_date})")
display(spark.table(results_table).filter(F.col(config.BATCH_DATE_COL) == F.lit(run_date)).limit(5))

# COMMAND ----------

# DBTITLE 1,Check — one copy of the date, and the daily trend
results_SDF = spark.table(results_table)
n_dupes = (results_SDF.filter(F.col(config.BATCH_DATE_COL) == F.lit(run_date))
           .groupBy(config.ID_COL).count().filter("count > 1").count())
assert n_dupes == 0, f"{n_dupes} patients appear more than once for {run_date}"

display(
    results_SDF.groupBy(config.BATCH_DATE_COL)
    .agg(F.count(F.when(F.col("afib_predictions_1") > config.AFIB_THRESHOLD, True)).alias("afib_positive"),
         F.count("*").alias("classified"))
    .withColumn("afib_share", F.col("afib_positive") / F.col("classified"))
    .orderBy(config.BATCH_DATE_COL)
)
