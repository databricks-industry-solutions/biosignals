"""Classification metrics, the promotion check, and the tables and curves behind the gate's
evaluation artifacts.

The training notebooks (02_, 03_) compute the metrics on the validation split and log them to
MLflow; the validation gate (04_validate_and_promote) computes them on the hold-out split and
promotes a challenger only when ``check_thresholds`` finds no failures against
``config.VALIDATION_THRESHOLDS``. The gate also logs, per model, a confusion matrix, ROC and
precision-recall curves and a per-class table from ``confusion_counts``, ``rhythm_call_counts``,
``roc_curve``, ``precision_recall_curve`` and ``per_class_report`` (plots: ``eval_plots.py``).
Numpy only, so the notebooks and the local tests run the same code.

Metric names (the ``<metric>`` of a ``min_<metric>`` / ``max_<metric>`` threshold):

  n, n_positive                     rows scored, rows whose truth is positive
  accuracy                          share of rows called correctly
  roc_auc                           area under the ROC curve (receiver operating
                                    characteristic) of the positive-class score
  average_precision                 area under the precision-recall curve (step-wise sum)
  sensitivity                       true-positive rate (recall): share of positives called positive
  specificity                       true-negative rate: share of negatives called negative
  called_positive_rate_<rhythm>     share of the ECGs of one true rhythm (nsr, afib, other)
                                    that the model called positive
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .ecg_signal_gen import AFIB, NSR, OTHER

# Rhythm class -> suffix of its called_positive_rate_<rhythm> metric.
RHYTHM_KEYS = {NSR: "nsr", AFIB: "afib", OTHER: "other"}


def roc_auc(y_true: Any, y_score: Any) -> float:
    """Area under the ROC curve: the chance a random positive scores above a random negative
    (rank formula; tied scores share their average rank). NaN unless both classes are present."""
    positive = np.asarray(y_true).astype(bool)
    score = np.asarray(y_score, dtype=float)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, tie_group, tie_count = np.unique(score, return_inverse=True, return_counts=True)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(score.shape[0])
    ranks[order] = np.arange(1, score.shape[0] + 1)
    ranks = (np.bincount(tie_group, weights=ranks) / tie_count)[tie_group]
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _distinct_score_counts(y_true: Any, y_score: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Scores sorted high to low, cut after each distinct value: (thresholds, true positives,
    false positives) when everything scoring >= threshold is called positive, and the class sizes."""
    positive = np.asarray(y_true).astype(bool)
    score = np.asarray(y_score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    score, positive = score[order], positive[order]
    last_of_value = np.r_[np.flatnonzero(np.diff(score)), score.size - 1] if score.size else np.array([], int)
    tps = np.cumsum(positive)[last_of_value].astype(float)
    fps = (last_of_value + 1) - tps
    return score[last_of_value], tps, fps, int(positive.sum()), int((~positive).sum())


def roc_curve(y_true: Any, y_score: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROC curve ``(false-positive rate, true-positive rate, thresholds)``: one point per distinct
    score, highest threshold first, from (0, 0) (threshold +inf: nothing called positive) to
    (1, 1). Its trapezoid area equals ``roc_auc``. Empty arrays unless both classes are present."""
    thresholds, tps, fps, n_pos, n_neg = _distinct_score_counts(y_true, y_score)
    if n_pos == 0 or n_neg == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    return np.r_[0.0, fps / n_neg], np.r_[0.0, tps / n_pos], np.r_[np.inf, thresholds]


def precision_recall_curve(y_true: Any, y_score: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision-recall curve ``(precision, recall, thresholds)``: one point per distinct score,
    highest threshold first, up to the first threshold that reaches full recall (lower thresholds
    only add false positives; scikit-learn stops there too). Empty arrays when there is no
    positive row."""
    thresholds, tps, fps, n_pos, _ = _distinct_score_counts(y_true, y_score)
    if n_pos == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    stop = int(np.searchsorted(tps, n_pos)) + 1
    return (tps / (tps + fps))[:stop], (tps / n_pos)[:stop], thresholds[:stop]


def average_precision(y_true: Any, y_score: Any) -> float:
    """Area under the precision-recall curve as the step-wise sum of precision x recall gain
    (no interpolation). NaN when there is no positive row."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    if recall.size == 0:
        return float("nan")
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def confusion_counts(y_true: Any, y_pred: Any) -> dict[str, int]:
    """Binary confusion counts (1 = positive): true/false negatives and positives."""
    truth = np.asarray(y_true).astype(bool)
    called = np.asarray(y_pred).astype(bool)
    return {"tn": int((~truth & ~called).sum()), "fp": int((~truth & called).sum()),
            "fn": int((truth & ~called).sum()), "tp": int((truth & called).sum())}


def rhythm_call_counts(labels: Any, y_pred: Any) -> dict[str, dict[str, int]]:
    """Per true rhythm present (keys of ``RHYTHM_KEYS``): ECGs called negative / positive."""
    labels = np.asarray(labels).astype(int)
    called = np.asarray(y_pred).astype(bool)
    return {key: {"called_negative": int((~called[labels == label]).sum()),
                  "called_positive": int(called[labels == label].sum())}
            for label, key in RHYTHM_KEYS.items() if (labels == label).any()}


def per_class_report(y_true: Any, y_pred: Any, class_names: tuple[str, str]) -> list[dict[str, Any]]:
    """Precision, recall, F1 and support for the negative and the positive class (named by
    ``class_names``), then their macro (unweighted) and support-weighted averages."""
    counts = confusion_counts(y_true, y_pred)
    rows = []
    for name, (hit, missed, wrongly_called) in zip(
            class_names, ((counts["tn"], counts["fp"], counts["fn"]), (counts["tp"], counts["fn"], counts["fp"]))):
        precision = hit / (hit + wrongly_called) if hit + wrongly_called else float("nan")
        recall = hit / (hit + missed) if hit + missed else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else float("nan")
        rows.append({"class": name, "precision": precision, "recall": recall, "f1": f1, "support": hit + missed})
    support = np.array([r["support"] for r in rows], dtype=float)
    for label, weights in (("macro avg", np.ones(2)), ("weighted avg", support)):
        rows.append({"class": label, **{k: float(np.average([r[k] for r in rows[:2]], weights=weights))
                                        for k in ("precision", "recall", "f1")},
                     "support": int(support.sum())})
    return rows


def _share(mask: np.ndarray) -> float:
    return float(mask.mean()) if mask.size else float("nan")


def evaluate(labels: Any, y_true: Any, y_score: Any, y_pred: Any) -> dict[str, float]:
    """Every metric above for one binary model.

    ``labels``: true rhythm class per row (``ecg_signal_gen`` labels); ``y_true``: the model's
    binary truth (1 = positive); ``y_score``: positive-class probability; ``y_pred``: the model's
    call (1 = positive), from the same rule the model applies in production."""
    labels = np.asarray(labels).astype(int)
    truth = np.asarray(y_true).astype(bool)
    called = np.asarray(y_pred).astype(bool)
    if not labels.shape == truth.shape == called.shape == np.shape(y_score):
        raise ValueError("labels, y_true, y_score and y_pred must have one value per row")
    metrics = {
        "n": float(truth.size),
        "n_positive": float(truth.sum()),
        "accuracy": _share(called == truth),
        "roc_auc": roc_auc(truth, y_score),
        "average_precision": average_precision(truth, y_score),
        "sensitivity": _share(called[truth]),
        "specificity": _share(~called[~truth]),
    }
    for label, key in RHYTHM_KEYS.items():
        rows = labels == label
        if rows.any():
            metrics[f"called_positive_rate_{key}"] = _share(called[rows])
    return metrics


def check_thresholds(
    metrics: Mapping[str, float], thresholds: Mapping[str, float], min_rows: int = 0
) -> list[str]:
    """Why ``metrics`` fail ``thresholds`` (``{"min_<metric>": floor, "max_<metric>": ceiling}``)
    and ``min_rows``; an empty list means pass. A metric that is NaN (not computable) fails.

    Raises ``ValueError`` for a threshold that is not ``min_``/``max_`` + a metric in
    ``metrics``, so a typo in the config can never pass silently."""
    failures = []
    if metrics.get("n", 0.0) < min_rows:
        failures.append(f"n = {metrics.get('n', 0.0):.0f} rows < {min_rows} required")
    for key, bound in thresholds.items():
        kind, _, name = key.partition("_")
        if kind not in ("min", "max") or name not in metrics:
            raise ValueError(f"threshold {key!r} is not min_/max_ + one of {sorted(metrics)}")
        value = float(metrics[name])
        if np.isnan(value):
            failures.append(f"{name} is not computable on this data")
        elif kind == "min" and value < bound:
            failures.append(f"{name} = {value:.4f} < {bound} (floor)")
        elif kind == "max" and value > bound:
            failures.append(f"{name} = {value:.4f} > {bound} (ceiling)")
    return failures
