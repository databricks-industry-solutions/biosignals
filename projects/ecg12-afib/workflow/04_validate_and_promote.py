# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Validation gate: score each `challenger` on the hold-out split, promote to `champion` only if it passes
# MAGIC
# MAGIC **SETUP job, task 4 of 5** (after `02_` and `03_`; `05_serve_endpoints` runs only if this
# MAGIC passes). For both models:
# MAGIC
# MAGIC 1. resolve `@challenger` (set by `02_` / `03_`) to its concrete version; a model with no
# MAGIC    challenger (not retrained since its last promotion) is re-checked at its `@champion`, so the
# MAGIC    pair is always gated together. With no challenger on either model (e.g. re-running `04_` alone
# MAGIC    after a promotion) there is nothing to validate, and the notebook exits cleanly with a note;
# MAGIC 2. read the `holdout` split of `training_data_12l` **at the Delta version that challenger was
# MAGIC    trained beside** (its `training_table_version` tag), so the gate never scores on rows the
# MAGIC    training notebooks saw (they read only `train` + `val`) or on a regenerated table;
# MAGIC 3. score it with exactly that version (driver-side, `common/scoring.py`) and compute accuracy,
# MAGIC    ROC AUC, sensitivity, specificity and the share of each true rhythm called positive
# MAGIC    (`common/evaluation.py`);
# MAGIC 4. compare with `VALIDATION_THRESHOLDS` in `common/config.py`.
# MAGIC
# MAGIC **Both models must pass, or neither is promoted:** the two serve as one cascade, so the gate
# MAGIC never moves one alias and leaves the other behind. On a pass, `@champion` moves to each
# MAGIC challenger version (if the second move fails, the first is put back) and then `@challenger` is
# MAGIC cleared, so after a pass only `@champion` remains: **`challenger` = a candidate awaiting (or that
# MAGIC failed) the gate; `champion` = what is served and batch-scored.** On a failure nothing moves,
# MAGIC `@challenger` stays on the failed version for inspection, the task fails with the reasons, and the
# MAGIC SETUP job stops before serving. Every outcome is recorded: an MLflow run with the thresholds and
# MAGIC hold-out metrics, and tags on each model version (`validation_status`, `gate_result`,
# MAGIC `holdout_*`). `validation_status` is `valid` or `invalid` (this model against its own
# MAGIC thresholds): the workspace has a tag policy for that key that allows only those two values.
# MAGIC `gate_result` is `passed` or `failed` (both models, the result that decides promotion). The
# MAGIC training notebooks set no `validation_status`: until this gate runs, the `challenger` alias is
# MAGIC what marks a version as awaiting validation.
# MAGIC
# MAGIC | Model | Thresholds (default) | Why |
# MAGIC |---|---|---|
# MAGIC | `bio_ecg12_arr` (triage) | accuracy ≥ 0.95, ROC AUC ≥ 0.98, sensitivity ≥ 0.97, specificity ≥ 0.90 | a missed abnormal ECG never reaches the AFib model, so sensitivity has the highest floor |
# MAGIC | `bio_ecg12_afib` (classify) | accuracy ≥ 0.95, ROC AUC ≥ 0.98, sensitivity ≥ 0.95, Other called AFib ≤ 0.05 | the classifier exists to separate AFib from the other fast rhythm, sinus tachycardia |
# MAGIC
# MAGIC These are engineering floors for synthetic data, where a healthy model scores ~1.0 (the rhythms
# MAGIC are trivially separable by design, see `00_overview`) and a model that predicts one class for
# MAGIC everything scores ~0.67 accuracy and 0.5 ROC AUC. They are not clinical acceptance criteria.
# MAGIC Aliases are lowercase (`champion`, `challenger`) because Unity Catalog stores them lowercase.
# MAGIC
# MAGIC **Evaluation artifacts.** From the same hold-out scores (no second scoring pass), the gate run
# MAGIC also logs, per model, under `holdout_eval/<model>/`: a **confusion matrix** (true class x the
# MAGIC model's call; figure + raw counts, including the call per true rhythm), the **ROC curve**
# MAGIC (receiver operating characteristic: true-positive vs false-positive rate over every threshold)
# MAGIC and the **precision-recall curve**, each with the operating point the production rule picks, a
# MAGIC **per-class table** (precision, recall, F1, support), and a README. Each model version gets a
# MAGIC `holdout_eval_artifacts` tag pointing at its folder.
# MAGIC
# MAGIC > **Read them as a workflow showcase, not as evidence of skill.** The synthetic classes are
# MAGIC > separable by design, so these artifacts look near-perfect: an almost diagonal confusion matrix
# MAGIC > and curves that hug the top-left (ROC) and top-right (precision-recall) corners. That shows the
# MAGIC > evaluation workflow working end to end; it says nothing about how the models would do on a hard,
# MAGIC > real clinical problem. Every figure carries this caption.
# MAGIC
# MAGIC **Compute:** serverless GPU, AI v5 (torch + mlflow present; no %pip).

# COMMAND ----------

# DBTITLE 1,Shared config, scoring + evaluation helpers (common/ is one folder up from workflow/)
import os
import sys

sys.path.insert(0, os.path.abspath(".."))   # notebook working dir = workflow/

from common import config, eval_plots, evaluation, scoring, uc_io
from common.ecg_signal_gen import N_SAMPLES, derive_afib_truth, derive_arr_flag

# COMMAND ----------

# DBTITLE 1,Parameters (UC location from the job; defaults from common/config.py)
dbutils.widgets.text("catalog_name", config.CATALOG, label="UC_catalog_name")
dbutils.widgets.text("schema_name", config.SCHEMA, label="UC_schema_name")
catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")

# Per model: binary truth from the rhythm label, model output -> result columns, the
# positive-class score, the call each model makes in production (07_ / 08_ / the app) and its
# wording, and the (negative, positive) class names for the evaluation artifacts.
GATED = {
    config.ARR_MODEL_NAME: {
        "truth": derive_arr_flag, "columns": scoring.arr_columns, "output_cols": scoring.ARR_OUTPUT_COLS,
        "score": "arr_prob", "call": lambda p: p >= config.ARR_FLAG_THRESHOLD,
        "rule": f"arr_prob >= {config.ARR_FLAG_THRESHOLD}", "classes": ("normal (NSR)", "abnormal (AFib or Other)"),
    },
    config.AFIB_MODEL_NAME: {
        "truth": derive_afib_truth, "columns": scoring.afib_columns, "output_cols": scoring.AFIB_OUTPUT_COLS,
        "score": "afib_predictions_1", "call": lambda p: p > config.AFIB_THRESHOLD,
        "rule": f"p_afib > {config.AFIB_THRESHOLD}", "classes": ("not AFib (NSR or Other)", "AFib"),
    },
}
print(f"{catalog_name}.{schema_name}: gating {list(GATED)} @{config.CHALLENGER_ALIAS} -> @{config.CHAMPION_ALIAS}")

# COMMAND ----------

# DBTITLE 1,Resolve each challenger and the training-data version it was trained beside
import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient(registry_uri="databricks-uc")

# The gate checks the two models as one cascade. Each model's candidate is its @challenger; a model
# with no challenger (not retrained since its last promotion) is re-checked at its @champion, so a
# retrained model is always gated beside the model it will serve with. With no challenger at all
# (e.g. re-running 04_ alone after a promotion) there is nothing new to validate: stop cleanly.
aliases = {name: client.get_registered_model(config.uc_name(name, catalog_name, schema_name)).aliases
           for name in GATED}
if not any(config.CHALLENGER_ALIAS in a for a in aliases.values()):
    message = (f"Nothing to validate: no @{config.CHALLENGER_ALIAS} on {list(GATED)} (the last gate already "
               f"promoted them to @{config.CHAMPION_ALIAS}). Train with 02_ / 03_ first.")
    print(message)
    dbutils.notebook.exit(message)
for name, a in aliases.items():
    if config.CHALLENGER_ALIAS not in a and config.CHAMPION_ALIAS not in a:
        raise RuntimeError(f"{name} has neither @{config.CHALLENGER_ALIAS} nor @{config.CHAMPION_ALIAS}: "
                           "train it with 02_ / 03_ first")

challengers = {}
for model_name in GATED:
    full_name = config.uc_name(model_name, catalog_name, schema_name)
    candidate = config.CHALLENGER_ALIAS if config.CHALLENGER_ALIAS in aliases[model_name] else config.CHAMPION_ALIAS
    resolved = scoring.resolve_alias(client, full_name, candidate)
    tags = client.get_model_version(full_name, resolved.version).tags
    missing = {"training_table", "training_table_version"} - set(tags)
    if missing:
        raise RuntimeError(f"{resolved.info} has no {sorted(missing)} tag(s), so its hold-out data is unknown; "
                           "register it with 02_/03_ before gating")
    champion = aliases[model_name].get(config.CHAMPION_ALIAS)          # recorded before anything moves
    challengers[model_name] = (resolved, tags["training_table"], int(tags["training_table_version"]), champion)
    print(f"{resolved.info}: trained beside {tags['training_table']} @ Delta version "
          f"{tags['training_table_version']} | current @{config.CHAMPION_ALIAS}: "
          f"{f'v{champion}' if champion else 'none yet'}")

# COMMAND ----------

# DBTITLE 1,Score each challenger on the hold-out split and check it against the thresholds
import pyspark.sql.functions as F

ecg_cols = config.ecg_columns(config.N_LEADS)
results, holdout_scores = {}, {}
for model_name, spec in GATED.items():
    resolved, table, version, _ = challengers[model_name]
    holdout_SDF = (uc_io.read_table_version(spark, table, version)
                   .filter(F.col(config.SPLIT_COL) == config.HOLDOUT_SPLIT))
    model = mlflow.pyfunc.load_model(resolved.uri)
    scored = scoring.score_driver_side(
        spark,
        holdout_SDF,
        lambda X, model=model, spec=spec: spec["columns"](model.predict(X)),
        output_cols=spec["output_cols"],
        ecg_cols=ecg_cols,
        n_samples=N_SAMPLES,
        model_info_col="model_info",
        model_info=resolved.info,
        id_col=config.SAMPLE_ID_COL,
    ).select("rhythm_label", spec["score"]).toPandas()
    score = scored[spec["score"]].to_numpy()
    labels = scored["rhythm_label"].to_numpy()
    truth, called = spec["truth"](labels).astype(bool), spec["call"](score)
    metrics = evaluation.evaluate(labels, truth, score, called)
    failures = evaluation.check_thresholds(metrics, config.VALIDATION_THRESHOLDS[model_name],
                                           min_rows=config.VALIDATION_MIN_ROWS)
    results[model_name] = (metrics, failures)
    holdout_scores[model_name] = (labels, truth, score, called)    # reused for the evaluation artifacts
    verdict = "PASS" if not failures else "FAIL: " + "; ".join(failures)
    print(f"{resolved.info} on {config.HOLDOUT_SPLIT} ({metrics['n']:.0f} rows): "
          f"{ {k: round(v, 4) for k, v in metrics.items()} }\n  -> {verdict}")

# COMMAND ----------

# DBTITLE 1,Hold-out evaluation artifacts per model (logged in the gate run below)
import matplotlib.pyplot as plt
import pandas as pd

# On every figure and in each model's README: the synthetic classes are separable by design.
EVAL_CAPTION = ("Synthetic hold-out data, separable by design (heart rate and RR irregularity split the classes), "
                "so near-perfect results are expected: this shows the evaluation workflow, not skill on a hard "
                "clinical problem.")
EVAL_FILES = {
    "confusion_matrix.png / confusion_matrix.json": "true class x the model's call: counts (raw in the JSON, "
                                                     "with the call per true rhythm)",
    "roc_curve.png": "true-positive vs false-positive rate over every threshold; ROC AUC; the operating point",
    "pr_curve.png": "precision vs recall over every threshold; average precision; the operating point",
    "per_class_metrics.json": "precision, recall, F1 and support per class, plus macro and weighted averages",
}


def log_holdout_eval(model_name, spec, resolved, holdout_table, metrics, labels, truth, score, called):
    """Log the confusion matrix, ROC and precision-recall curves, per-class table and a README for
    one model under holdout_eval/<model>/ of the active run; returns that folder."""
    folder = f"holdout_eval/{model_name}"
    counts = evaluation.confusion_counts(truth, called)
    called_positive = counts["tp"] + counts["fp"]
    precision_at_rule = counts["tp"] / called_positive if called_positive else float("nan")
    mlflow.log_dict({"rows": "true class", "columns": "model call", "classes": list(spec["classes"]),
                     "matrix": [[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]], **counts,
                     "by_true_rhythm": evaluation.rhythm_call_counts(labels, called),
                     "decision_rule": spec["rule"], "n": int(len(truth)), "note": EVAL_CAPTION},
                    f"{folder}/confusion_matrix.json")
    mlflow.log_table(pd.DataFrame(evaluation.per_class_report(truth, called, spec["classes"])),
                     artifact_file=f"{folder}/per_class_metrics.json")
    fpr, tpr, _ = evaluation.roc_curve(truth, score)
    precision, recall, _ = evaluation.precision_recall_curve(truth, score)
    title = f"{model_name} v{resolved.version}, hold-out split (n = {len(truth)})"
    rule = f"operating point: {spec['rule']}"
    figures = {
        "confusion_matrix.png": eval_plots.confusion_matrix(counts, spec["classes"], f"{title}\nconfusion matrix",
                                                            EVAL_CAPTION),
        "roc_curve.png": eval_plots.roc_curve(fpr, tpr, metrics["roc_auc"],
                                              (1 - metrics["specificity"], metrics["sensitivity"]), rule,
                                              f"{title}\nROC curve", EVAL_CAPTION),
        "pr_curve.png": eval_plots.precision_recall_curve(precision, recall, metrics["average_precision"],
                                                          float(truth.mean()), (metrics["sensitivity"], precision_at_rule),
                                                          rule, f"{title}\nprecision-recall curve", EVAL_CAPTION),
    }
    for name, fig in figures.items():
        mlflow.log_figure(fig, f"{folder}/{name}")
    plt.show()
    plt.close("all")
    files = "\n".join(f"| `{name}` | {what} |" for name, what in EVAL_FILES.items())
    mlflow.log_text(f"# Hold-out evaluation: {model_name} v{resolved.version}\n\n{EVAL_CAPTION}\n\n"
                    f"Scored by 04_validate_and_promote on the `{config.HOLDOUT_SPLIT}` split of {holdout_table} "
                    f"({len(truth)} ECGs the training notebooks never read), with the rule the model applies in "
                    f"production: `{spec['rule']}`.\n\n| File | What it shows |\n|---|---|\n{files}\n",
                    f"{folder}/README.md")
    return folder

# COMMAND ----------

# DBTITLE 1,Record the outcome (MLflow run, evaluation artifacts, model-version tags), then promote both or neither
passed = all(not failures for _, failures in results.values())
status = "passed" if passed else "failed"


def validation_status(ok):
    """``validation_status`` tag value. The workspace's tag policy for this key allows only
    ``valid`` / ``invalid``, and Unity Catalog rejects any other value on a model version."""
    return "valid" if ok else "invalid"


with mlflow.start_run(run_name="validate_and_promote") as gate_run:
    mlflow.set_tags({"pipeline_task": "04_validate_and_promote", "validation_status": validation_status(passed)})
    for model_name, (metrics, failures) in results.items():
        resolved, table, version, champion = challengers[model_name]
        mlflow.log_params({f"{model_name}.challenger_version": resolved.version,
                           f"{model_name}.previous_champion_version": champion or "none",
                           f"{model_name}.holdout_table": f"{table}@v{version}",
                           **{f"{model_name}.{k}": v for k, v in config.VALIDATION_THRESHOLDS[model_name].items()}})
        mlflow.log_metrics({f"{model_name}.holdout_{k}": v for k, v in metrics.items()})
        eval_folder = log_holdout_eval(model_name, GATED[model_name], resolved, f"{table}@v{version}", metrics,
                                       *holdout_scores[model_name])
        # validation_status = this model vs its own thresholds (valid / invalid); gate_result = both
        # models (passed / failed), which is what promotes.
        version_tags = {"validation_status": validation_status(not failures), "gate_result": status,
                        "validation_run_id": gate_run.info.run_id,
                        "holdout_eval_artifacts": f"runs:/{gate_run.info.run_id}/{eval_folder}"}
        version_tags |= {f"holdout_{k}": f"{metrics[k]:.4f}" for k in ("accuracy", "roc_auc", "sensitivity", "specificity")}
        for key, value in version_tags.items():
            client.set_model_version_tag(resolved.full_name, resolved.version, key, value)

if not passed:
    reasons = {name: failures for name, (_, failures) in results.items() if failures}
    raise RuntimeError(f"Validation gate FAILED, nothing promoted (@{config.CHAMPION_ALIAS} unchanged): {reasons}")

# Both or neither: move @champion model by model; if a move fails, put back the champions already
# moved (to the version recorded above, or no alias if there was none) and re-raise.
moved = []
try:
    for model_name, (resolved, *_rest) in challengers.items():
        client.set_registered_model_alias(resolved.full_name, config.CHAMPION_ALIAS, resolved.version)
        moved.append(model_name)
except Exception:
    for model_name in moved:
        resolved, _table, _version, previous = challengers[model_name]
        if previous:
            client.set_registered_model_alias(resolved.full_name, config.CHAMPION_ALIAS, previous)
        else:
            client.delete_registered_model_alias(resolved.full_name, config.CHAMPION_ALIAS)
    raise

# Promoted: @challenger now marks nothing awaiting the gate, so it is cleared; only @champion
# (what is served and batch-scored) remains. On a failure (above) @challenger stays on the failed
# version, tagged validation_status=invalid, for inspection.
for model_name, (resolved, *_rest) in challengers.items():
    if resolved.alias == config.CHALLENGER_ALIAS:
        client.delete_registered_model_alias(resolved.full_name, config.CHALLENGER_ALIAS)
    print(f"promoted {resolved.full_name} v{resolved.version}: @{resolved.alias} -> @{config.CHAMPION_ALIAS}")
