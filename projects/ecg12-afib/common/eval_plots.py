"""Hold-out evaluation figures (matplotlib) that 04_validate_and_promote logs to MLflow per model.

* ``confusion_matrix``: true class x the model's call, each cell with its count and its share of
  the true class, shaded on a light-to-dark blue scale by that share.
* ``roc_curve``: true-positive rate against false-positive rate over every threshold, with the
  ROC AUC (area under the curve) and the operating point (the threshold the model applies).
* ``precision_recall_curve``: precision against recall over every threshold, with the average
  precision, the share of positives (what a random score would reach) and the operating point.

The curves come from ``evaluation.py``. Same white style as ``ecg_plots.py``; every figure prints
``caption`` along its bottom edge (04_ passes the note that the synthetic data is separable by
design), so the figure carries its own context wherever it is viewed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .ecg_plots import INK, INK2, STYLE

LINE = "#2a78d6"       # categorical slot 1 (blue), the one series on each curve plot
MUTED = "#898781"      # reference lines: chance diagonal, share of positives
# Sequential blue, light -> dark (steps 100..700 of the validated ramp), for the confusion cells.
BLUES = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")


def _finish(fig: Any, caption: str | None) -> Any:
    if caption:
        fig.text(0.01, 0.01, caption, ha="left", va="bottom", fontsize=7.5, color=INK2, wrap=True)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
    else:
        fig.tight_layout()
    return fig


def confusion_matrix(counts: dict[str, int], class_names: tuple[str, str], title: str,
                     caption: str | None = None) -> Any:
    """Figure: the 2 x 2 confusion matrix of ``counts`` (``evaluation.confusion_counts``); rows =
    true class, columns = the model's call, ``class_names`` = (negative, positive)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cells = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]], dtype=float)
    totals = cells.sum(axis=1, keepdims=True)
    share = np.divide(cells, totals, out=np.zeros_like(cells), where=totals > 0)
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(5.6, 4.8))
        ax.pcolormesh(share, cmap=LinearSegmentedColormap.from_list("blues", BLUES), vmin=0, vmax=1,
                      edgecolors="white", linewidth=2)
        for row in range(2):
            for col in range(2):
                ax.text(col + 0.5, row + 0.5, f"{int(cells[row, col])}\n{share[row, col]:.0%} of row",
                        ha="center", va="center", fontsize=11,
                        color="white" if share[row, col] > 0.5 else INK)
        ax.set_xticks([0.5, 1.5], [f"called {n}" for n in class_names])
        ax.set_yticks([0.5, 1.5], [f"true {n}" for n in class_names])
        ax.invert_yaxis()
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, loc="left", fontsize=10)
    return _finish(fig, caption)


def _curve_axes(ax: Any, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10)


def _operating_point(ax: Any, x: float, y: float, label: str) -> None:
    ax.plot([x], [y], "o", ms=8, color=LINE, mec="white", mew=2, zorder=3)
    right_half = bool(x > 0.5)              # label on the inner side of the point, inside the axes
    ax.annotate(label, (x, y), textcoords="offset points", xytext=(-10 if right_half else 10, -14),
                ha="right" if right_half else "left", fontsize=8, color=INK)


def roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float, operating_point: tuple[float, float],
              operating_label: str, title: str, caption: str | None = None) -> Any:
    """Figure: the ROC curve (``evaluation.roc_curve``), its AUC, and the operating point
    ``(false-positive rate, true-positive rate)`` of the model's own call."""
    import matplotlib.pyplot as plt

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED)
        ax.annotate("chance", (0.62, 0.56), fontsize=8, color=MUTED, rotation=45)
        ax.plot(fpr, tpr, lw=2, color=LINE)
        ax.text(0.98, 0.04, f"ROC AUC = {auc:.3f}", ha="right", fontsize=9, color=INK)
        _operating_point(ax, *operating_point, operating_label)
        _curve_axes(ax, "false-positive rate (1 - specificity)", "true-positive rate (sensitivity)", title)
    return _finish(fig, caption)


def precision_recall_curve(precision: np.ndarray, recall: np.ndarray, average_precision: float,
                           positive_share: float, operating_point: tuple[float, float],
                           operating_label: str, title: str, caption: str | None = None) -> Any:
    """Figure: the precision-recall curve (``evaluation.precision_recall_curve``), its average
    precision, the share of positives (a random score's precision) and the operating point
    ``(recall, precision)`` of the model's own call."""
    import matplotlib.pyplot as plt

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        ax.axhline(positive_share, ls="--", lw=1, color=MUTED)
        ax.annotate(f"share of positives = {positive_share:.2f}", (0.02, positive_share),
                    textcoords="offset points", xytext=(0, 4), fontsize=8, color=MUTED)
        ax.plot(np.r_[0.0, recall], np.r_[precision[0], precision], lw=2, color=LINE, drawstyle="steps-pre")
        ax.text(0.02, 0.04, f"average precision = {average_precision:.3f}", fontsize=9, color=INK)
        _operating_point(ax, *operating_point, operating_label)
        _curve_axes(ax, "recall (sensitivity)", "precision (positive predictive value)", title)
    return _finish(fig, caption)
