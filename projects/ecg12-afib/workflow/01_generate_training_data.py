# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Generate + persist the 12-lead training dataset (train / val / hold-out)
# MAGIC
# MAGIC **SETUP job, task 1 of 5.** The first step of the model cycle is the data: this notebook draws
# MAGIC labeled synthetic 12-lead ECGs from the shared generator `common/ecg_signal_gen.py` and writes
# MAGIC them to the Unity Catalog Delta table **`training_data_12l`**, one row per ECG, with a `split`
# MAGIC column:
# MAGIC
# MAGIC | split | ECGs (default) | read by | used for |
# MAGIC |---|---|---|---|
# MAGIC | `train` | 600 | `02_`, `03_` | fitting both models |
# MAGIC | `val` | 200 | `02_`, `03_` | per-epoch validation metrics logged to MLflow |
# MAGIC | `holdout` | 200 | `04_validate_and_promote` only | the metrics that decide promotion to `champion` |
# MAGIC
# MAGIC Sizes, seeds and the class mix (default 34 % AFib, 33 % Other, 33 % NSR in every split) come from
# MAGIC `common/config.py` (`TRAINING_SPLITS`, `TRAINING_FRAC_*`). Each split is its own seeded draw, so
# MAGIC re-running rewrites the same rows (as a new Delta version); the training notebooks record the
# MAGIC version they read and the gate evaluates the hold-out split at that same version.
# MAGIC
# MAGIC The table is this demo's own: its name is not a daily-batch name (`batch_data_12l_<date>`,
# MAGIC `arr_triaged_12l_<date>`) or the results table, and nothing of the live 8-lead demo is touched.
# MAGIC
# MAGIC Columns: `sample_id`, `split`, `rhythm_label` (0 NSR, 1 AFib, 2 Other), `rhythm_name`,
# MAGIC `heart_rate_bpm`, and `ecg1`..`ecg12` (one ARRAY<FLOAT> per lead, mV, 2500 samples at 250 Hz, in
# MAGIC the order I, II, III, aVR, aVL, aVF, V1..V6), the same per-lead layout as the daily batch.
# MAGIC
# MAGIC **Compute:** standard serverless. Parameters: `catalog_name`, `schema_name`.

# COMMAND ----------

# DBTITLE 1,Install generator deps (standard serverless; --no-deps keeps base numpy intact)
# MAGIC %pip install --no-deps "neurokit2==0.2.12" "PyWavelets==1.4.1"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Shared config + dataset helpers (common/ is one folder up from workflow/)
import os
import sys

sys.path.insert(0, os.path.abspath(".."))   # notebook working dir = workflow/

from common import config, ecg_plots, training_data, uc_io
from common.ecg_signal_gen import LABEL_NAMES

# COMMAND ----------

# DBTITLE 1,Parameters (UC location from the job; defaults from common/config.py)
dbutils.widgets.text("catalog_name", config.CATALOG, label="UC_catalog_name")
dbutils.widgets.text("schema_name", config.SCHEMA, label="UC_schema_name")

train_table = config.uc_name(config.TRAINING_TABLE, dbutils.widgets.get("catalog_name"),
                             dbutils.widgets.get("schema_name"))
print(f"write: {train_table}\nsplits (ECGs, seed): {config.TRAINING_SPLITS}")

# COMMAND ----------

# DBTITLE 1,Generate every split from the shared generator (deterministic per split seed)
batches = training_data.generate_splits(
    config.TRAINING_SPLITS,
    frac_afib=config.TRAINING_FRAC_AFIB,
    frac_other=config.TRAINING_FRAC_OTHER,
    noise_level=config.TRAINING_NOISE_LEVEL,
    n_leads=config.N_LEADS,
)
for split, batch in batches.items():
    counts = {name: int((batch.labels == label).sum()) for label, name in LABEL_NAMES.items()}
    print(f"{split:8s} {batch.ecg.shape}  {counts}")

# COMMAND ----------

# DBTITLE 1,Save to UC as a Delta table (overwrite = re-run safe) and report its version
from functools import reduce

lead_names = next(iter(batches.values())).lead_names
schema = training_data.spark_schema(lead_names)
rows = training_data.to_pandas(batches)
# Hand the rows to Spark 200 ECGs at a time (~24 MB of float32 each), the size class of the daily
# batch in 06_, rather than all 1000 (~120 MB) in one local-data transfer; one write, one version.
CHUNK = 200
train_SDF = reduce(lambda a, b: a.unionByName(b),
                   [spark.createDataFrame(rows.iloc[i:i + CHUNK], schema=schema) for i in range(0, len(rows), CHUNK)])
uc_io.overwrite_table(spark, train_SDF, train_table)
print(f"wrote {spark.table(train_table).count()} rows -> {train_table} "
      f"(Delta version {uc_io.table_version(spark, train_table)})")

# COMMAND ----------

# DBTITLE 1,Quick check — rows per split and rhythm
display(spark.table(train_table).groupBy(config.SPLIT_COL, "rhythm_name").count()
        .orderBy(config.SPLIT_COL, "rhythm_name"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### What the generated signals look like
# MAGIC
# MAGIC First, lead II of one training ECG per rhythm class: this is what separates the classes. NSR is
# MAGIC regular at 60–100 bpm with a P wave before every QRS; AFib is irregularly irregular with no P
# MAGIC waves and a fibrillating baseline; Other (sinus tachycardia) is regular, with P waves, but fast.
# MAGIC Then all 12 leads of the same ECGs in the clinical layout (limb leads left, chest leads right):
# MAGIC the beat looks different in every lead because each lead views the heart's electrical activity
# MAGIC from a different direction (see the lead legend in `00_overview`).

# COMMAND ----------

# DBTITLE 1,Showcase — lead II rhythm strip per class (training split)
import matplotlib.pyplot as plt

train = batches[config.TRAIN_SPLIT]
ecg_plots.rhythm_strips(train.ecg, train.labels, train.lead_names, train.heart_rate_bpm,
                        title="Training data: lead II, one ECG per rhythm class (same mV scale)")
plt.show()

# COMMAND ----------

# DBTITLE 1,Showcase — all 12 leads, one ECG per class
for label in sorted(set(train.labels.tolist())):
    i = int((train.labels == label).argmax())
    ecg_plots.twelve_lead(train.ecg[i], train.lead_names, label)
    plt.show()
