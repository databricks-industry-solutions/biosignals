# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Generate the daily 12-lead batch + synthetic patient records (SIMULATED incoming data)
# MAGIC
# MAGIC **DAILY job, task 1 of 3 (`simulate_incoming_batch`).** Simulated incoming data, not real
# MAGIC ingestion: it mocks the end of a data-landing pipeline with one day's synthetic
# MAGIC 12-lead ECGs, each with a synthetic, non-PII patient + acquisition record, written to the UC
# MAGIC Delta table **`batch_data_12l_<YYYYMMDD>`** for `07_arr_triage`. These are new ECGs to score,
# MAGIC separate from the training dataset of `01_`.
# MAGIC
# MAGIC **Data collection over time.** Each ECG carries its own acquisition time within the day's
# MAGIC clinic hours (07:00–19:00 UTC, a busier morning and a smaller afternoon peak; see
# MAGIC `common/patient_info.py`), so a day's batch reads like a stream of incoming ECGs rather than
# MAGIC one instant. It is still a batch job (no Structured Streaming): each run lands one day, and a
# MAGIC backfill over several `run_date`s builds the multi-day history the dashboard trends over.
# MAGIC
# MAGIC | Columns | Content |
# MAGIC |---|---|
# MAGIC | `batch_date`, `batch_inference_datetime_utc` | run date (DATE) + acquisition time as `dd-MM-yyyy HH:mm:ss` text (kept from the 8-lead tables) |
# MAGIC | `pseudo_patient_id`, `rhythm_label` | synthetic id + ground-truth rhythm (0 NSR, 1 AFib, 2 Other) |
# MAGIC | `ecg1`..`ecg12` | one ARRAY<FLOAT> per lead, mV, 2500 samples at 250 Hz, in the order I, II, III, aVR, aVL, aVF, V1..V6 |
# MAGIC | patient record | `mrn`, `age_years`, `sex`, `height_cm`, `weight_kg`, `bmi`, `bp_*_mmhg`, `acquisition_datetime_utc`, `heart_rate_bpm`, `sampling_rate_hz`, `n_leads`, `duration_s` |
# MAGIC
# MAGIC Re-running a date reproduces the same rows (everything is seeded by the date) and overwrites
# MAGIC the table. Column comments (units, which leads are derived) are stored in Unity Catalog.
# MAGIC
# MAGIC Parameters: `run_date` (`YYYY-MM-DD`, empty = today UTC; the daily job passes its start
# MAGIC date), `catalog_name`, `schema_name`.

# COMMAND ----------

# DBTITLE 1,Install generator deps (standard serverless; --no-deps keeps base numpy intact)
# MAGIC %pip install --no-deps "neurokit2==0.2.12" "PyWavelets==1.4.1"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Shared config + generators (common/ is one folder up from workflow/)
import os
import sys

sys.path.insert(0, os.path.abspath(".."))   # notebook working dir = workflow/

from common import config, ecg_plots, patient_info, uc_io
from common.ecg_signal_gen import describe_lead, generate_ecg_batch

# COMMAND ----------

# DBTITLE 1,Parameters (usable as a job task; one shared run_date)
dbutils.widgets.text(config.RUN_DATE_PARAM, "", label="run_date (YYYY-MM-DD, empty = today UTC)")
dbutils.widgets.text("catalog_name", config.CATALOG, label="UC_catalog_name")
dbutils.widgets.text("schema_name", config.SCHEMA, label="UC_schema_name")

run_date = config.resolve_run_date(dbutils.widgets.get(config.RUN_DATE_PARAM))
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
write_table = config.daily_table(config.BATCH_TABLE_PREFIX, run_date, catalog_name, schema_name)
print(f"run_date: {run_date}\nwrite   : {write_table}")

# COMMAND ----------

# DBTITLE 1,Generate the day's ECGs + patient records (deterministic per date)
import numpy as np
import pandas as pd

# Per-date jitter so each daily batch differs (a lifelike trend, not a flat line), seeded by
# YYYYMMDD so a re-run of the same date reproduces it exactly.
date_seed = int(config.date_tag(run_date))
jitter = np.random.default_rng(date_seed)
n_ecgs = int(jitter.integers(85, 116))                                     # daily volume ~85-115
frac_afib = float(np.clip(0.30 + jitter.normal(0, 0.06), 0.12, 0.45))     # AFib share varies by day
frac_other = float(np.clip(0.20 + jitter.normal(0, 0.05), 0.08, 0.35))    # Other-abnormal share varies
print(f"{run_date}: n={n_ecgs} frac_afib={frac_afib:.3f} frac_other={frac_other:.3f}")

batch = generate_ecg_batch(n_ecgs, frac_afib=frac_afib, frac_other=frac_other,
                           noise_level=1.0, seed=date_seed, n_leads=config.N_LEADS)
patient_ids = patient_info.assign_patients(batch.labels, np.random.default_rng(date_seed + 1))
records = [
    patient_info.patient_record(
        pid,
        heart_rate_bpm=int(hr),
        acquisition_datetime_utc=patient_info.acquisition_time(pid, run_date),
        ecg=ecg,
    ).to_row()
    for pid, hr, ecg in zip(patient_ids, batch.heart_rate_bpm, batch.ecg)
]
print(f"ecg {batch.ecg.shape} leads {batch.lead_names}; {len(records)} patient records")

# COMMAND ----------

# DBTITLE 1,Assemble the table (schema + UC column comments from common/)
import pyspark.sql.types as T

ecg_cols = config.ecg_columns(len(batch.lead_names))
SPARK_TYPES = {"STRING": T.StringType(), "INT": T.IntegerType(), "DOUBLE": T.DoubleType(),
               "TIMESTAMP": T.TimestampType()}

schema = T.StructType(
    [
        T.StructField(config.BATCH_DATE_COL, T.DateType(), True, {"comment": "Run date of the daily batch"}),
        T.StructField("batch_inference_datetime_utc", T.StringType(), True,
                      {"comment": "Acquisition time as dd-MM-yyyy HH:mm:ss text (UTC); the 8-lead tables' format"}),
        T.StructField(config.ID_COL, T.StringType(), True,
                      {"comment": patient_info.FIELD_SPECS[config.ID_COL][1]}),
        # DOUBLE (not INT): the type of rhythm_label in the live 8-lead demo's tables too.
        T.StructField("rhythm_label", T.DoubleType(), True,
                      {"comment": "Ground-truth rhythm: 0 NSR, 1 AFib, 2 Other (sinus tachycardia)"}),
    ]
    + [T.StructField(col, T.ArrayType(T.FloatType()), True, {"comment": describe_lead(name)})
       for col, name in zip(ecg_cols, batch.lead_names)]
    + [T.StructField(name, SPARK_TYPES[sql_type], True, {"comment": comment})
       for name, (sql_type, comment) in patient_info.FIELD_SPECS.items() if name != config.ID_COL]
)

rows = pd.DataFrame(records)
rows.insert(0, config.BATCH_DATE_COL, run_date)
rows.insert(1, "batch_inference_datetime_utc",
            [r["acquisition_datetime_utc"].strftime("%d-%m-%Y %H:%M:%S") for r in records])
rows.insert(3, "rhythm_label", batch.labels.astype("float64"))
# Timestamps go to Spark as naive UTC values read in a UTC session (tz-aware pandas columns fail
# in some pandas -> Spark conversions), so the stored instant is the UTC acquisition time.
spark.conf.set("spark.sql.session.timeZone", "UTC")
rows["acquisition_datetime_utc"] = pd.to_datetime(rows["acquisition_datetime_utc"], utc=True).dt.tz_localize(None)
for j, col in enumerate(ecg_cols):
    rows[col] = [lead.tolist() for lead in batch.ecg[:, j, :]]
batch_SDF = spark.createDataFrame(rows[[field.name for field in schema.fields]], schema=schema)
display(batch_SDF.drop(*ecg_cols).limit(5))

# COMMAND ----------

# DBTITLE 1,Save to UC as a Delta table (overwrite = re-run safe)
uc_io.overwrite_table(spark, batch_SDF, write_table)
print(f"wrote {spark.table(write_table).count()} rows -> {write_table}")

# COMMAND ----------

# DBTITLE 1,Quick check — rhythm mix and patient context by rhythm
display(
    spark.table(write_table)
    .groupBy("rhythm_label")
    .agg({"*": "count", "age_years": "avg", "heart_rate_bpm": "avg"})
    .orderBy("rhythm_label")
)

# COMMAND ----------

# DBTITLE 1,Arrivals over the day — ECGs acquired per hour (UTC), by true rhythm
import pyspark.sql.functions as F

display(
    spark.table(write_table)
    .groupBy(F.hour("acquisition_datetime_utc").alias("hour_utc"), "rhythm_label")
    .count()
    .orderBy("hour_utc", "rhythm_label")
)

# COMMAND ----------

# DBTITLE 1,Showcase — today's incoming ECGs: lead II, one per rhythm class
import matplotlib.pyplot as plt

ecg_plots.rhythm_strips(batch.ecg, batch.labels, batch.lead_names, batch.heart_rate_bpm,
                        title=f"Daily batch {run_date}: lead II, one ECG per true rhythm (same mV scale)")
plt.show()
