# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Train the ARR (triage) model, 12-lead, register it as `challenger`  [serverless GPU / PyTorch]
# MAGIC
# MAGIC **SETUP job, task 2 of 5** (after `01_`, in parallel with `03_`). Trains a small **PyTorch 1D-CNN**
# MAGIC that flags *abnormal* rhythms (binary: NSR = 0 vs AFib or Other = 1) from a **12-lead** ECG and
# MAGIC registers it to Unity Catalog as a new version of **`bio_ecg12_arr`** with the alias
# MAGIC **`challenger`**. It does **not** become `champion` here: `04_validate_and_promote` scores the
# MAGIC challenger on the hold-out split and promotes it only if it passes the thresholds in
# MAGIC `common/config.py`. This is the cheap triage stage; `03_` trains the AFib classifier.
# MAGIC
# MAGIC **Data:** the `train` and `val` splits of `training_data_12l`, persisted by `01_`, read at one
# MAGIC Delta version that is recorded with the model (the `holdout` split is left for the gate).
# MAGIC
# MAGIC **Training:** one fixed configuration, `TRAIN_PARAMS` in `common/config.py` (default: Adam,
# MAGIC learning rate 1e-3, 20 epochs, batch 64, seed 0), no hyperparameter search. **Why the metrics
# MAGIC sit near 1.0:** the synthetic rhythms are trivially separable by design. Heart rate alone
# MAGIC separates triage (the generator draws NSR at 60–100 bpm and both abnormal classes at 100–160
# MAGIC bpm), and beat-to-beat irregularity separates AFib (RR intervals vary ~20 % in AFib, ~3 % in
# MAGIC sinus rhythm). So validation accuracy is ~1.0: this notebook is a clean showcase of the train →
# MAGIC track → register step of the model cycle, not a hard ML problem.
# MAGIC
# MAGIC **MLflow** (the notebook's experiment, one run): every training setting and the dataset identity
# MAGIC (table, Delta version, split sizes and seeds) as params; `train_loss`, `val_loss` and
# MAGIC `val_accuracy` per epoch; the final validation accuracy, ROC AUC, average precision,
# MAGIC sensitivity and specificity (`final_val_*`); the layer table `model_architecture.md`; tags for
# MAGIC the model, its challenger version and the data version; the pyfunc model with signature + input
# MAGIC example. (The hold-out confusion matrix, ROC and precision-recall curves are logged by `04_`.)
# MAGIC PyTorch autologging covers PyTorch Lightning only, so this plain training loop logs explicitly.
# MAGIC
# MAGIC **Architecture:** a small 1D convolutional neural network (CNN). Input: one ECG as 12 leads x
# MAGIC 2500 samples. The first convolution layer has `in_channels = N_LEADS = 12`, so each of its
# MAGIC filters sees all 12 leads at once as it slides along time. Three convolution blocks (16 → 32 → 64
# MAGIC channels; the time axis shrinks 2500 → 78 steps) → global average pooling → one dense (fully
# MAGIC connected) layer → 2 class scores; 10,514 learned parameters. The run logs the layer table
# MAGIC (output shape and parameters per layer) as `model_architecture.md` and the total as the
# MAGIC `n_parameters` param; `00_overview` walks through each step.
# MAGIC
# MAGIC **Serving constraint:** the model classes (`ArrCNN`, `ArrModelPyfunc`) and the input reshape
# MAGIC (`_coerce_ecg`) are defined **in this notebook** so MLflow pickles them by value. They must only
# MAGIC use torch / numpy / pandas and plain notebook constants (`N_LEADS`, `N_SAMPLES`,
# MAGIC `ARR_FLAG_THRESHOLD`), never the `common` package, which is not installed on the serving
# MAGIC endpoint. (The training loop in `common/training.py` is not part of the logged model.)
# MAGIC
# MAGIC Contract: input `(-1, 12, 2500)` float32 (a DataFrame `ecg` column of flattened 30000-value
# MAGIC arrays also works); output DataFrame `arr_prob` (abnormal probability) + `arr_flag`
# MAGIC (1.0 when `arr_prob >= 0.5`, else 0.0).
# MAGIC
# MAGIC **Compute:** serverless GPU, **A10**, base environment **AI v5** (torch, CUDA, mlflow; no %pip).

# COMMAND ----------

# DBTITLE 1,Shared config + training helpers (common/ is one folder up from workflow/)
import os
import sys

sys.path.insert(0, os.path.abspath(".."))   # notebook working dir = workflow/

from common import config, evaluation, training, training_data, uc_io
from common.ecg_signal_gen import N_SAMPLES, derive_arr_flag

# Plain constants for the inline model code below (pickled by value with the model).
N_LEADS = int(config.N_LEADS)
ARR_FLAG_THRESHOLD = float(config.ARR_FLAG_THRESHOLD)

# COMMAND ----------

# DBTITLE 1,Parameters (UC location from the job; defaults from common/config.py)
dbutils.widgets.text("catalog_name", config.CATALOG, label="UC_catalog_name")
dbutils.widgets.text("schema_name", config.SCHEMA, label="UC_schema_name")
dbutils.widgets.text("model_name", config.ARR_MODEL_NAME, label="arr_model_name")

catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")
full_name = config.uc_name(dbutils.widgets.get("model_name"), catalog_name, schema_name)
train_table = config.uc_name(config.TRAINING_TABLE, catalog_name, schema_name)
print(f"model: {full_name} (registers @{config.CHALLENGER_ALIAS}) | data: {train_table} | "
      f"n_leads={N_LEADS} | n_samples={N_SAMPLES}")

# COMMAND ----------

# DBTITLE 1,Imports + confirm GPU
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import mlflow

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | cuda={torch.cuda.is_available()} | device={DEVICE}")

# COMMAND ----------

# DBTITLE 1,Model + inference wrapper (inline so MLflow pickles them by value)
class ArrCNN(nn.Module):
    """Small 1D-CNN over (batch, 12 leads, 2500 samples) -> 2-class logits."""
    def __init__(self, n_leads=N_LEADS, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_leads, 16, kernel_size=7, stride=2, padding=3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def _coerce_ecg(model_input):
    """Coerce a DataFrame ('ecg' col of array-likes) or ndarray to (n, 12, 2500) float32."""
    if isinstance(model_input, pd.DataFrame):
        col = "ecg" if "ecg" in model_input.columns else model_input.columns[0]
        arr = np.stack([np.asarray(v, dtype=np.float32).ravel() for v in model_input[col]])
    else:
        arr = np.asarray(model_input, dtype=np.float32)
    return arr.reshape(arr.shape[0], N_LEADS, N_SAMPLES).astype(np.float32)


class ArrModelPyfunc(mlflow.pyfunc.PythonModel):
    """Loads ArrCNN weights from the logged state_dict; predicts abnormal prob/flag."""
    def load_context(self, context):
        self._model = ArrCNN()
        self._model.load_state_dict(
            torch.load(context.artifacts["state_dict"], map_location="cpu", weights_only=True)
        )
        self._model.eval()

    def predict(self, context, model_input):
        X = torch.from_numpy(_coerce_ecg(model_input))
        with torch.no_grad():
            prob = torch.softmax(self._model(X), dim=1)[:, 1].numpy()
        return pd.DataFrame({"arr_prob": prob, "arr_flag": (prob >= ARR_FLAG_THRESHOLD).astype("float32")})

# COMMAND ----------

# DBTITLE 1,Read the persisted train + val splits at one Delta version (binary: normal vs abnormal)
data_version = uc_io.table_version(spark, train_table)
train = training_data.load_split(spark, train_table, data_version, config.TRAIN_SPLIT, N_LEADS)
val = training_data.load_split(spark, train_table, data_version, config.VAL_SPLIT, N_LEADS)
y_tr = derive_arr_flag(train.labels).astype("int64")   # abnormal (AFib or Other) = 1
y_va = derive_arr_flag(val.labels).astype("int64")
assert train.x.shape[1:] == (N_LEADS, N_SAMPLES), train.x.shape
print(f"{train_table} @ Delta version {data_version}: train {train.x.shape} abnormal {int(y_tr.sum())} | "
      f"val {val.x.shape} abnormal {int(y_va.sum())}")

# COMMAND ----------

# DBTITLE 1,Train with the fixed config; log params, per-epoch and final validation metrics to MLflow
P = config.TRAIN_PARAMS
mlflow.set_registry_uri("databricks-uc")
data_tags = {"training_table": train_table, "training_table_version": str(data_version)}


def log_epoch(m):
    mlflow.log_metrics({"train_loss": m.train_loss, "val_loss": m.val_loss, "val_accuracy": m.val_accuracy},
                       step=m.epoch)
    if m.epoch % 5 == 0 or m.epoch == int(P["epochs"]) - 1:
        print(f"epoch {m.epoch:2d}  train_loss {m.train_loss:.4f}  val_loss {m.val_loss:.4f}  "
              f"val_acc {m.val_accuracy:.4f}")


with mlflow.start_run(run_name=f"{config.ARR_MODEL_NAME}_train") as run:
    mlflow.set_tags({"model_name": full_name, "pipeline_task": "02_train_arr_model",
                     "registers_alias": config.CHALLENGER_ALIAS, **data_tags})
    convs = [m for m in ArrCNN().features if isinstance(m, nn.Conv1d)]
    arch_md, n_params = training.architecture_summary(ArrCNN(), (1, N_LEADS, N_SAMPLES))
    mlflow.log_text(arch_md, "model_architecture.md")     # layer table: output shape + parameters
    mlflow.log_params({
        **P, "optimizer": "Adam", "loss": "cross_entropy", "label_smoothing": 0.0,
        "arch": "ArrCNN_1dcnn", "task": "abnormal_vs_nsr", "n_classes": 2,
        "conv_channels": ",".join(str(c.out_channels) for c in convs),
        "conv_kernel_sizes": ",".join(str(c.kernel_size[0]) for c in convs), "n_parameters": n_params,
        "n_leads": N_LEADS, "n_samples": N_SAMPLES, "decision_threshold": ARR_FLAG_THRESHOLD,
        **data_tags, "train_split": config.TRAIN_SPLIT, "val_split": config.VAL_SPLIT,
        "n_train": len(y_tr), "n_val": len(y_va), "n_train_positive": int(y_tr.sum()),
        "n_val_positive": int(y_va.sum()),
        "train_seed": config.TRAINING_SPLITS[config.TRAIN_SPLIT][1],
        "val_seed": config.TRAINING_SPLITS[config.VAL_SPLIT][1],
        "frac_afib": config.TRAINING_FRAC_AFIB, "frac_other": config.TRAINING_FRAC_OTHER,
        "noise_level": config.TRAINING_NOISE_LEVEL,
    })
    model, history = training.fit(
        ArrCNN, train.x, y_tr, val.x, y_va,
        learning_rate=float(P["learning_rate"]), epochs=int(P["epochs"]), batch_size=int(P["batch_size"]),
        seed=int(P["seed"]), device=DEVICE, on_epoch=log_epoch,
    )

    # Final validation metrics, with the rule the model applies: arr_flag = arr_prob >= threshold.
    p_va = training.softmax(training.predict_logits(model, val.x, device=DEVICE))[:, 1]
    val_metrics = evaluation.evaluate(val.labels, y_va, p_va, p_va >= ARR_FLAG_THRESHOLD)
    mlflow.log_metrics({f"final_val_{k}": v for k, v in val_metrics.items() if k not in ("n", "n_positive")})
    mlflow.log_metrics({"final_train_loss": history[-1].train_loss, "final_val_loss": history[-1].val_loss})
print({k: round(v, 4) for k, v in val_metrics.items()})

# COMMAND ----------

# DBTITLE 1,Log as MLflow pyfunc + register to Unity Catalog as @challenger (NOT champion)
import tempfile
from mlflow import MlflowClient
from mlflow.models import infer_signature

# Persist weights as a logged artifact (the class definitions are pickled by value).
tmp_dir = tempfile.mkdtemp()
sd_path = os.path.join(tmp_dir, "state_dict.pt")
model = model.cpu().eval()
torch.save(model.state_dict(), sd_path)

# UC needs a signature (input + output). Input: (-1, 12, 2500) float32 tensor; output mirrors the
# pyfunc's arr_prob/arr_flag frame. A SINGLE-sample input_example keeps the Serving UI's test tab
# usable (a multi-sample example is too large for it).
ex_in = val.x[:1].astype("float32")                        # (1, 12, 2500)
with torch.no_grad():
    ex_p = torch.softmax(model(torch.from_numpy(ex_in)), dim=1)[:, 1].numpy()
ex_out = pd.DataFrame({"arr_prob": ex_p, "arr_flag": (ex_p >= ARR_FLAG_THRESHOLD).astype("float32")})
signature = infer_signature(ex_in, ex_out)

with mlflow.start_run(run_id=run.info.run_id):            # same run as the training above
    model_info = mlflow.pyfunc.log_model(
        name="arr_model",
        python_model=ArrModelPyfunc(),
        artifacts={"state_dict": sd_path},
        signature=signature,
        input_example=ex_in,
        registered_model_name=full_name,
    )
    version = str(model_info.registered_model_version)
    mlflow.set_tag("challenger_version", version)

# The new version is the challenger; only 04_validate_and_promote can make it champion.
# No validation_status tag here: @challenger already marks the version as awaiting the gate, and
# 04_ sets validation_status once it has scored the hold-out (no tag = not validated yet).
client = MlflowClient(registry_uri="databricks-uc")
client.set_registered_model_alias(full_name, config.CHALLENGER_ALIAS, version)
for key, value in {**data_tags, "training_run_id": run.info.run_id,
                   "val_accuracy": f"{val_metrics['accuracy']:.4f}"}.items():
    client.set_model_version_tag(full_name, version, key, value)
print(f"registered {full_name} v{version} @{config.CHALLENGER_ALIAS} (run {run.info.run_id})")

# COMMAND ----------

# DBTITLE 1,Sanity: load the registered version back and predict on a few validation rows
loaded = mlflow.pyfunc.load_model(f"models:/{full_name}/{version}")
out = loaded.predict(val.x[:5].astype("float32"))    # (5, 12, 2500) -> matches the tensor signature
out["true_abnormal"] = y_va[:5]
print(out)
