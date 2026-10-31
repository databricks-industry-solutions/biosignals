# Model lifecycle

[← ecg12-afib README](../README.md)

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
