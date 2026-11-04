"""Showcase plots of the generated ECGs (matplotlib), for 01_generate_training_data and
06_generate_batch_data.

* ``rhythm_strips``: one lead (II by default), 10 s, for one ECG of each rhythm class, stacked
  on one millivolt scale. This is the difference the models learn: regular and 60-100 bpm
  (NSR), irregularly irregular with no P waves (AFib), regular but fast (Other).
* ``twelve_lead``: one ECG in the standard 6 x 2 layout (limb leads left, chest leads right),
  every lead on the same millivolt scale; derived leads are marked "(derived)".

Colors: the first three slots of the validated categorical palette, one per rhythm class
(blue NSR, orange AFib, aqua Other), always next to a text label, never color alone.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .ecg_signal_gen import AFIB, DERIVATION, LABEL_NAMES, NSR, OTHER, SAMPLING_RATE

CLASS_COLORS = {NSR: "#2a78d6", AFIB: "#eb6834", OTHER: "#1baf7a"}
LIMB_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF")
CHEST_LEADS = ("V1", "V2", "V3", "V4", "V5", "V6")
# White figure style, also used by eval_plots.py: primary / secondary ink, hairline grid.
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {"figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRID,
         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 9,
         "axes.spines.top": False, "axes.spines.right": False}


def _seconds(n_samples: int) -> np.ndarray:
    return np.arange(n_samples) / SAMPLING_RATE


def rhythm_strips(
    ecg: np.ndarray,
    labels: np.ndarray,
    lead_names: tuple[str, ...],
    heart_rate_bpm: np.ndarray | None = None,
    lead: str = "II",
    title: str | None = None,
) -> Any:
    """Figure: lead ``lead`` of the first ECG of each rhythm class in ``labels``, one row each.

    ``ecg``: (n, n_leads, n_samples) mV; ``labels``: rhythm class per ECG."""
    import matplotlib.pyplot as plt

    labels = np.asarray(labels).astype(int)
    classes = [c for c in LABEL_NAMES if np.any(labels == c)]
    if not classes:
        raise ValueError("no ECGs to plot")
    j = lead_names.index(lead)
    picks = [int(np.flatnonzero(labels == c)[0]) for c in classes]
    t = _seconds(ecg.shape[2])
    lo, hi = float(ecg[picks, j].min()), float(ecg[picks, j].max())
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(classes), 1, figsize=(12, 1.9 * len(classes)), sharex=True,
                                 sharey=True, squeeze=False)
        for ax, cls, i in zip(axes[:, 0], classes, picks):
            ax.plot(t, ecg[i, j], color=CLASS_COLORS[cls], lw=0.9)
            rate = f", {int(heart_rate_bpm[i])} bpm" if heart_rate_bpm is not None else ""
            ax.set_title(f"{LABEL_NAMES[cls]}{rate}", loc="left", fontsize=10)
            ax.set_ylim(lo - 0.1, hi + 0.1)
            ax.set_ylabel(f"lead {lead} (mV)")
        axes[-1, 0].set_xlabel("time (s)")
        fig.suptitle(title or f"Lead {lead}, one ECG per rhythm class (same mV scale)",
                     x=0.01, ha="left", fontsize=11)
        fig.tight_layout()
    return fig


def twelve_lead(
    ecg_one: np.ndarray,
    lead_names: tuple[str, ...],
    label: int,
    seconds: float = 5.0,
    title: str | None = None,
) -> Any:
    """Figure: one ECG (n_leads, n_samples) in the clinical 6 x 2 layout, first ``seconds``."""
    import matplotlib.pyplot as plt

    columns = [[n for n in LIMB_LEADS if n in lead_names], [n for n in CHEST_LEADS if n in lead_names]]
    n_rows = max(len(c) for c in columns)
    t = _seconds(ecg_one.shape[1])
    window = t < seconds
    lo, hi = float(ecg_one[:, window].min()), float(ecg_one[:, window].max())
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 1.4 * n_rows + 0.6), sharex=True,
                                 sharey=True, squeeze=False)
        for col, names in enumerate(columns):
            for row in range(n_rows):
                ax = axes[row, col]
                if row >= len(names):
                    ax.set_visible(False)
                    continue
                name = names[row]
                ax.plot(t[window], ecg_one[lead_names.index(name), window], color=CLASS_COLORS[label], lw=0.9)
                tag = f"{name} (derived)" if name in DERIVATION else name
                ax.text(0.01, 0.92, tag, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
                ax.set_ylim(lo - 0.1, hi + 0.1)
        for ax in axes[-1]:
            ax.set_xlabel("time (s)")
        for ax in axes[:, 0]:
            ax.set_ylabel("mV")
        fig.suptitle(title or f"{LABEL_NAMES[label]}: all {len(lead_names)} leads, first {seconds:g} s, "
                     "same mV scale on every lead", x=0.01, ha="left", fontsize=11)
        fig.tight_layout()
    return fig
