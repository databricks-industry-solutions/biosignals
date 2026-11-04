"""Tests for common/evaluation.py (the metrics the training notebooks log and the gate checks, and
the tables and curves of the gate's evaluation artifacts) and common/eval_plots.py."""
import math

import numpy as np
import pytest

from common import eval_plots, evaluation
from common.ecg_signal_gen import AFIB, NSR, OTHER


def test_roc_auc_matches_known_values():
    assert evaluation.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert evaluation.roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert evaluation.roc_auc([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5           # ties share ranks
    assert evaluation.roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75
    assert evaluation.roc_auc([0, 1, 1], [0.3, 0.3, 0.9]) == 0.75                   # one tied pair = 1/2
    assert math.isnan(evaluation.roc_auc([1, 1], [0.2, 0.9]))                       # one class only


def test_evaluate_binary_and_per_rhythm_rates():
    labels = np.array([NSR, NSR, AFIB, AFIB, OTHER, OTHER])
    truth = labels == AFIB
    score = np.array([0.1, 0.2, 0.9, 0.4, 0.7, 0.3])
    called = score > 0.5
    m = evaluation.evaluate(labels, truth, score, called)
    assert m["n"] == 6 and m["n_positive"] == 2
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["sensitivity"] == 0.5 and m["specificity"] == 0.75
    assert m["called_positive_rate_nsr"] == 0.0 and m["called_positive_rate_afib"] == 0.5
    assert m["called_positive_rate_other"] == 0.5
    assert m["roc_auc"] == pytest.approx(0.875)
    assert m["average_precision"] == pytest.approx(5 / 6)   # AFib at score ranks 1 and 3: 0.5 x 1 + 0.5 x 2/3
    only_nsr = evaluation.evaluate([NSR, NSR], [0, 0], [0.1, 0.2], [0, 0])
    assert "called_positive_rate_afib" not in only_nsr and math.isnan(only_nsr["sensitivity"])
    with pytest.raises(ValueError, match="one value per row"):
        evaluation.evaluate([NSR], [0, 1], [0.1, 0.2], [0, 0])


def test_check_thresholds_pass_fail_and_typos():
    m = {"n": 200.0, "accuracy": 0.99, "roc_auc": 1.0, "sensitivity": 0.9, "called_positive_rate_other": 0.1}
    assert evaluation.check_thresholds(m, {"min_accuracy": 0.95, "min_roc_auc": 0.98}, min_rows=100) == []
    failures = evaluation.check_thresholds(m, {"min_sensitivity": 0.95, "max_called_positive_rate_other": 0.05})
    assert failures == ["sensitivity = 0.9000 < 0.95 (floor)", "called_positive_rate_other = 0.1000 > 0.05 (ceiling)"]
    assert evaluation.check_thresholds(m, {}, min_rows=500) == ["n = 200 rows < 500 required"]
    assert evaluation.check_thresholds({**m, "roc_auc": float("nan")}, {"min_roc_auc": 0.5}) == [
        "roc_auc is not computable on this data"]
    for typo in ("min_acuracy", "accuracy", "above_accuracy"):                      # never passes silently
        with pytest.raises(ValueError, match="threshold"):
            evaluation.check_thresholds(m, {typo: 0.5})


def test_curves_match_known_values_and_the_rank_auc():
    y, s = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
    fpr, tpr, thresholds = evaluation.roc_curve(y, s)
    np.testing.assert_allclose(fpr, [0, 0, 0.5, 0.5, 1])
    np.testing.assert_allclose(tpr, [0, 0.5, 0.5, 1, 1])
    assert thresholds[0] == np.inf and list(thresholds[1:]) == [0.8, 0.4, 0.35, 0.1]
    precision, recall, _ = evaluation.precision_recall_curve(y, s)
    np.testing.assert_allclose(precision, [1, 0.5, 2 / 3])                 # stops at full recall
    np.testing.assert_allclose(recall, [0.5, 0.5, 1])
    assert evaluation.average_precision(y, s) == pytest.approx(5 / 6)      # scikit-learn's value for this case
    # With ties and at scale, the trapezoid area under roc_curve equals the rank-based roc_auc.
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.random(300) + 0.3 * y, 1)                              # rounded: many tied scores
    fpr, tpr, _ = evaluation.roc_curve(y, s)
    assert np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2) == pytest.approx(evaluation.roc_auc(y, s))
    assert evaluation.roc_curve([1, 1], [0.2, 0.9])[0].size == 0            # one class: no curve
    assert math.isnan(evaluation.average_precision([0, 0], [0.2, 0.9]))


def test_confusion_counts_per_rhythm_and_per_class_report():
    labels = np.array([NSR, NSR, AFIB, AFIB, OTHER, OTHER])
    truth, called = labels == AFIB, np.array([0, 0, 1, 0, 1, 0], dtype=bool)
    assert evaluation.confusion_counts(truth, called) == {"tn": 3, "fp": 1, "fn": 1, "tp": 1}
    assert evaluation.rhythm_call_counts(labels, called) == {
        "nsr": {"called_negative": 2, "called_positive": 0}, "afib": {"called_negative": 1, "called_positive": 1},
        "other": {"called_negative": 1, "called_positive": 1}}
    rows = {r["class"]: r for r in evaluation.per_class_report(truth, called, ("not AFib", "AFib"))}
    assert list(rows) == ["not AFib", "AFib", "macro avg", "weighted avg"]
    assert rows["AFib"] == {"class": "AFib", "precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2}
    assert rows["not AFib"]["precision"] == 0.75 and rows["not AFib"]["recall"] == 0.75 and rows["not AFib"]["support"] == 4
    assert rows["macro avg"]["f1"] == pytest.approx(0.625) and rows["weighted avg"]["recall"] == pytest.approx(4 / 6)
    assert rows["weighted avg"]["support"] == 6


def test_eval_figures_render_with_caption_and_operating_point():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.6, 0.4, 0.8, 0.9])
    called = s >= 0.5
    counts = evaluation.confusion_counts(y, called)
    caption = "synthetic data, separable by design"
    fpr, tpr, _ = evaluation.roc_curve(y, s)
    precision, recall, _ = evaluation.precision_recall_curve(y, s)
    figures = [
        eval_plots.confusion_matrix(counts, ("normal", "abnormal"), "m v1\nconfusion matrix", caption),
        eval_plots.roc_curve(fpr, tpr, evaluation.roc_auc(y, s), (1 / 3, 2 / 3), "rule", "m v1\nROC curve", caption),
        eval_plots.precision_recall_curve(precision, recall, evaluation.average_precision(y, s), 0.5, (2 / 3, 2 / 3),
                                          "rule", "m v1\nprecision-recall curve", caption),
    ]
    try:
        for fig in figures:
            assert [t.get_text() for t in fig.texts] == [caption]                 # caption on every figure
            assert fig.axes[0].get_title(loc="left").startswith("m v1")
        cells = [t.get_text() for t in figures[0].axes[0].texts]
        assert cells == ["2\n67% of row", "1\n33% of row", "1\n33% of row", "2\n67% of row"]
        assert "ROC AUC = 0.889" in [t.get_text() for t in figures[1].axes[0].texts]
    finally:
        for fig in figures:
            plt.close(fig)
