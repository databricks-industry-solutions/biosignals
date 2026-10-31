"""Read-only variety analysis of the synthetic 12-lead ECG generator (developer utility, see
../README.md; not part of the app, the jobs or the bundle).

Imports common/ecg_signal_gen.py of this demo (run with `python -B` so no bytecode is written into
common/) and writes stats + plots to output/ next to this file (git-ignored), or to the folder
given as the first argument:

    python -B dev/signal_variety/analyze_signal_variety.py [OUT_DIR]

Each patient gets its own seed. The same seed is run at noise_level=1.0 (as every caller ships)
and 0.0, so noisy - clean isolates the acquisition noise exactly. The true R-peak times come from
g._r_peak_times(label, default_rng(seed)): it is the first rng consumer in _synthesize_one, so it
reproduces that patient's beat times (checked against heart_rate_bpm below).
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

DEMO_ROOT = Path(__file__).resolve().parents[2]       # projects/ecg12-afib (holds common/)
sys.path.insert(0, str(DEMO_ROOT))
from common import ecg_signal_gen as g  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)
FS = g.SAMPLING_RATE
N_PER_CLASS = 40
N_LARGE = 20_000                     # per class, rhythm-only (no waveform) separability check
CLASSES = [g.NSR, g.AFIB, g.OTHER]
CLS_NAME = {g.NSR: "NSR", g.AFIB: "AFib", g.OTHER: "Other (sinus tachy)"}
LEADS = list(g.LEADS_12)
LI = {name: i for i, name in enumerate(LEADS)}
RECORDED = [LI[n] for n in g.INDEPENDENT_LEADS]
TPL_PRE, TPL_POST = int(0.22 * FS), int(0.30 * FS)   # beat window around R: P wave .. T wave
BASE = TPL_PRE - int(0.07 * FS)                        # PR segment = isoelectric reference
QRS = slice(TPL_PRE - int(0.05 * FS), TPL_PRE + int(0.05 * FS))
COLORS = {g.NSR: "#2a78d6", g.AFIB: "#eb6834", g.OTHER: "#1baf7a"}  # validated slots 1-3


def one(label: int, seed: int, noise: float) -> tuple[np.ndarray, int]:
    b = g.synthesize_ecgs(np.array([label]), np.random.default_rng(seed), noise_level=noise)
    return b.ecg[0].astype(np.float64), int(b.heart_rate_bpm[0])


def in_window(r_times: np.ndarray) -> np.ndarray:
    return r_times[(r_times >= 0.0) & (r_times < g.DURATION)]


def detect_r(ecg: np.ndarray) -> np.ndarray:
    """R-peak sample indices detected on the NOISY signal (5-25 Hz QRS energy, 8 recorded leads)."""
    bb, aa = butter(3, [5 / (FS / 2), 25 / (FS / 2)], btype="band")
    x = filtfilt(bb, aa, ecg[RECORDED], axis=1)
    env = (np.diff(x, axis=1, prepend=x[:, :1]) ** 2).sum(axis=0)
    env = np.convolve(env, np.ones(int(0.04 * FS)) / int(0.04 * FS), mode="same")
    pk, _ = find_peaks(env, distance=int(0.22 * FS), height=0.25 * np.percentile(env, 99.5))
    return pk


def templates(ecg: np.ndarray, r_idx: np.ndarray) -> np.ndarray:
    """(12, W) median beat per lead, aligned on the true R sample, PR-segment baseline = 0."""
    ok = [p for p in r_idx if p - TPL_PRE >= 0 and p + TPL_POST <= ecg.shape[1]]
    tpl = np.median(np.stack([ecg[:, p - TPL_PRE:p + TPL_POST] for p in ok]), axis=0)
    return tpl - tpl[:, BASE:BASE + 1]


def upper(c: np.ndarray) -> np.ndarray:
    return c[np.triu_indices_from(c, k=1)]


def q(a) -> dict:
    a = np.asarray(a, dtype=float)
    return {"min": round(float(a.min()), 4), "p5": round(float(np.percentile(a, 5)), 4),
            "median": round(float(np.median(a)), 4), "mean": round(float(a.mean()), 4),
            "p95": round(float(np.percentile(a, 95)), 4), "max": round(float(a.max()), 4),
            "std": round(float(a.std(ddof=1)), 4), "n": int(a.size)}


def best_threshold_acc(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Best single-threshold accuracy of feature x for binary y (either direction)."""
    order = np.argsort(x)
    xs, ys = x[order], y[order].astype(int)
    # accuracy of "predict 1 when x > xs[i]" for every split i, plus the mirrored rule
    ones_left = np.concatenate([[0], np.cumsum(ys)])
    zeros_left = np.arange(len(xs) + 1) - ones_left
    acc_hi = (zeros_left + (ys.sum() - ones_left)) / len(xs)
    acc_lo = 1 - acc_hi
    i = int(np.argmax(np.maximum(acc_hi, acc_lo)))
    thr = float(xs[min(i, len(xs) - 1)])
    return round(float(max(acc_hi[i], acc_lo[i])), 5), round(thr, 4)


# --- 1. generate ------------------------------------------------------------------------------
recs = []
for k in CLASSES:
    for j in range(N_PER_CLASS):
        seed = 1_000_000 * (k + 1) + j
        noisy, hr_true = one(k, seed, 1.0)
        clean, hr0 = one(k, seed, 0.0)
        r_times = g._r_peak_times(k, np.random.default_rng(seed))
        w = in_window(r_times)
        assert hr_true == hr0 == int(round(60 / np.mean(np.diff(w)))), seed
        recs.append({"label": k, "seed": seed, "noisy": noisy, "clean": clean, "hr_true": hr_true,
                     "r_times": r_times, "r_idx": np.round(w * FS).astype(int)})

for r in recs:
    pk = detect_r(r["noisy"])
    r["rr"] = np.diff(pk) / FS                          # measured on the noisy signal
    r["n_beats_detected_vs_true"] = (len(pk), len(r["r_idx"]))
    r["hr_meas"] = 60.0 / r["rr"].mean()
    r["rr_cv"] = r["rr"].std(ddof=1) / r["rr"].mean()
    r["rmssd_ms"] = 1000 * np.sqrt(np.mean(np.diff(r["rr"]) ** 2))
    r["tpl_clean"] = templates(r["clean"], r["r_idx"])
    r["tpl_noisy"] = templates(r["noisy"], r["r_idx"])
    r["noise"] = ((r["noisy"] - r["noisy"].mean(1, keepdims=True))
                  - (r["clean"] - r["clean"].mean(1, keepdims=True)))
    t = r["tpl_clean"]
    r["R_mV"], r["S_mV"] = t[:, QRS].max(axis=1), t[:, QRS].min(axis=1)
    r["qrs_axis_deg"] = float(np.degrees(np.arctan2(t[LI["aVF"], QRS].sum(), t[LI["I"], QRS].sum())))
    v5 = t[LI["V5"], QRS]
    above = np.where(v5 >= 0.5 * v5.max())[0]
    r["R_fwhm_ms_V5"] = 1000 * (above.max() - above.min() + 1) / FS

# --- 1. rhythm: between-class separation, within-class spread ---------------------------------
summary = {"n_per_class": N_PER_CLASS, "per_class": {}}
for k in CLASSES:
    cr = [r for r in recs if r["label"] == k]
    lo, hi = g.HEART_RATE_RANGE_BPM[k]
    summary["per_class"][CLS_NAME[k]] = {
        "hr_generator_truth_bpm": q([r["hr_true"] for r in cr]),
        "hr_measured_from_noisy_signal_bpm": q([r["hr_meas"] for r in cr]),
        "rr_cv": q([r["rr_cv"] for r in cr]),
        "rmssd_ms": q([r["rmssd_ms"] for r in cr]),
        "hr_outside_class_range": int(sum(not (lo <= r["hr_true"] <= hi) for r in cr)),
    }
summary["detector_check"] = {
    "abs_hr_err_bpm": q([abs(r["hr_meas"] - r["hr_true"]) for r in recs]),
    "records_with_beat_count_mismatch": int(sum(a != b for a, b in (r["n_beats_detected_vs_true"] for r in recs))),
}
lab = np.array([r["label"] for r in recs])
hr = np.array([r["hr_meas"] for r in recs])
cv = np.array([r["rr_cv"] for r in recs])
summary["single_feature_separability_120_records"] = {
    "NSR_vs_abnormal_by_HR": best_threshold_acc(hr, lab != g.NSR),
    "AFib_vs_rest_by_RR_CV": best_threshold_acc(cv, lab == g.AFIB),
    "AFib_vs_Other_by_HR": best_threshold_acc(hr[lab != g.NSR], lab[lab != g.NSR] == g.AFIB),
    "RR_CV_min_AFib__max_nonAFib": [round(float(cv[lab == g.AFIB].min()), 4), round(float(cv[lab != g.AFIB].max()), 4)],
    "HR_max_NSR__min_abnormal": [round(float(hr[lab == g.NSR].max()), 1), round(float(hr[lab != g.NSR].min()), 1)],
}

# Rhythm-only at scale: the beat times alone (what the class labels are built from).
big_lab, big_hr, big_cv = [], [], []
for k in CLASSES:
    for j in range(N_LARGE):
        w = in_window(g._r_peak_times(k, np.random.default_rng(50_000_000 * (k + 1) + j)))
        rr = np.diff(w)
        big_lab.append(k)
        big_hr.append(60 / rr.mean())
        big_cv.append(rr.std(ddof=1) / rr.mean())
big_lab, big_hr, big_cv = map(np.array, (big_lab, big_hr, big_cv))
nb = big_lab != g.NSR
summary[f"single_feature_separability_{N_LARGE}_per_class_beat_times_only"] = {
    "NSR_vs_abnormal_by_HR": best_threshold_acc(big_hr, nb),
    "AFib_vs_rest_by_RR_CV": best_threshold_acc(big_cv, big_lab == g.AFIB),
    "AFib_vs_Other_by_RR_CV": best_threshold_acc(big_cv[nb], big_lab[nb] == g.AFIB),
    "AFib_vs_Other_by_HR": best_threshold_acc(big_hr[nb], big_lab[nb] == g.AFIB),
    "RR_CV_min_AFib__max_nonAFib": [round(float(big_cv[big_lab == g.AFIB].min()), 4),
                                    round(float(big_cv[big_lab != g.AFIB].max()), 4)],
    "HR_max_NSR__min_Other__min_AFib": [round(float(big_hr[big_lab == g.NSR].max()), 1),
                                        round(float(big_hr[big_lab == g.OTHER].min()), 1),
                                        round(float(big_hr[big_lab == g.AFIB].min()), 1)],
    "HR_by_class_p1_p99": {CLS_NAME[k]: [round(float(np.percentile(big_hr[big_lab == k], p)), 1) for p in (1, 99)]
                           for k in CLASSES},
    "RR_CV_by_class_p1_p99": {CLS_NAME[k]: [round(float(np.percentile(big_cv[big_lab == k], p)), 4) for p in (1, 99)]
                              for k in CLASSES},
}

# --- 2. within-class patient-to-patient variation ---------------------------------------------
within = {}
for k in CLASSES:
    cr = [r for r in recs if r["label"] == k]
    concat = np.corrcoef(np.stack([r["tpl_clean"].ravel() for r in cr]))
    tplII = upper(np.corrcoef(np.stack([r["tpl_clean"][LI["II"]] for r in cr])))
    nn = concat.copy()
    np.fill_diagonal(nn, np.nan)
    within[CLS_NAME[k]] = {
        "raw_10s_leadII_noisy_corr": q(upper(np.corrcoef(np.stack([r["noisy"][LI["II"]] for r in cr])))),
        "beat_shape_leadII_clean_corr": q(tplII),
        "beat_shape_leadII_noisy_corr": q(upper(np.corrcoef(np.stack([r["tpl_noisy"][LI["II"]] for r in cr])))),
        "beat_shape_12lead_clean_corr": q(upper(concat)),
        "frac_pairs_12lead_corr_gt_0.99": round(float(np.mean(upper(concat) > 0.99)), 4),
        "frac_pairs_12lead_corr_gt_0.95": round(float(np.mean(upper(concat) > 0.95)), 4),
        "nearest_neighbour_12lead_corr": q(np.nanmax(nn, axis=1)),
        "R_mV_leadII": q([r["R_mV"][LI["II"]] for r in cr]),
        "R_mV_V5": q([r["R_mV"][LI["V5"]] for r in cr]),
        "S_mV_V2": q([r["S_mV"][LI["V2"]] for r in cr]),
        "p2p_mV_leadII_noisy_10s": q([np.ptp(r["noisy"][LI["II"]]) for r in cr]),
        "qrs_axis_deg": q([r["qrs_axis_deg"] for r in cr]),
        "R_fwhm_ms_V5": q([r["R_fwhm_ms_V5"] for r in cr]),
    }
summary["within_class"] = within
summary["between_class_beat_shape_leadII_clean_corr"] = {
    f"{CLS_NAME[a]} vs {CLS_NAME[b]}": q(np.corrcoef(np.vstack(
        [np.stack([r["tpl_clean"][LI["II"]] for r in recs if r["label"] == c]) for c in (a, b)]))
        [:N_PER_CLASS, N_PER_CLASS:].ravel())
    for a, b in combinations(CLASSES, 2)}

# Timing invariants: T-peak latency after R in V3 vs preceding RR, sinus classes, every beat.
tl, rrp, pid = [], [], []
for i, r in enumerate(recs):
    if r["label"] == g.AFIB:
        continue
    rt = r["r_times"]
    for b in range(1, len(rt) - 1):
        p = int(round(rt[b] * FS))
        lo_, hi_ = p + int(0.06 * FS), p + int(0.6 * (rt[b + 1] - rt[b]) * FS)
        if p < 0 or hi_ >= g.N_SAMPLES:
            continue
        tl.append((lo_ + int(np.argmax(r["clean"][LI["V3"], lo_:hi_])) - p) / FS)
        rrp.append(rt[b] - rt[b - 1])
        pid.append(i)
tl, rrp, pid = map(np.array, (tl, rrp, pid))
coef = np.polyfit(np.sqrt(rrp), tl, 1)
resid = tl - np.polyval(coef, np.sqrt(rrp))
summary["timing_invariants_sinus_classes"] = {
    "Tpeak_vs_sqrtRR_fit_slope_intercept_s": [round(float(c), 4) for c in coef],
    "fit_residual_sd_ms": round(1000 * float(resid.std()), 2),
    "per_patient_mean_residual_sd_ms": round(1000 * float(np.std([resid[pid == i].mean() for i in np.unique(pid)])), 2),
    "n_beats": int(len(tl)),
}

# --- 3. across-lead variation within a patient ------------------------------------------------
xl = {"deriv_err": [], "offdiag": [], "gt095": [], "pair": [], "avr_qrs_neg": [], "avr_t_neg": [],
      "avr_p_neg": [], "rprog_135": [], "rprog_1234": [], "s_v2": [], "trans": [], "svd3": [], "svd5": []}
iu = np.triu_indices(12, k=1)
for r in recs:
    e = r["noisy"].astype(np.float32)          # the generator's float32 output
    I, II = e[LI["I"]], e[LI["II"]]
    xl["deriv_err"].append(max(np.abs(e[LI["III"]] - (II - I)).max(), np.abs(e[LI["aVR"]] + (I + II) / 2).max(),
                               np.abs(e[LI["aVL"]] - (I - II / 2)).max(), np.abs(e[LI["aVF"]] - (II - I / 2)).max()))
    t = r["tpl_clean"]
    c = np.corrcoef(t)[iu]
    xl["offdiag"].append(np.abs(c).mean())
    xl["gt095"].append(int((c > 0.95).sum()))
    xl["pair"].append((f"{LEADS[iu[0][np.argmax(c)]]}~{LEADS[iu[1][np.argmax(c)]]}", float(c.max())))
    xl["avr_qrs_neg"].append(abs(r["S_mV"][LI["aVR"]]) > r["R_mV"][LI["aVR"]])
    rr_mean = np.mean(np.diff(r["r_times"]))
    t_idx = TPL_PRE + int(round((0.40 * np.sqrt(rr_mean) - 0.12) * FS))   # generator's T-peak centre
    xl["avr_t_neg"].append(t[LI["aVR"], min(t_idx, t.shape[1] - 1)] < 0)
    if r["label"] != g.AFIB:
        p_idx = TPL_PRE - int(round((0.12 + 0.04 * min(rr_mean, 1.2)) * FS))
        xl["avr_p_neg"].append(t[LI["aVR"], p_idx] < 0)
    R = r["R_mV"][[LI[f"V{i}"] for i in range(1, 7)]]
    S = r["S_mV"][[LI[f"V{i}"] for i in range(1, 7)]]
    xl["rprog_135"].append(R[0] < R[2] < R[4])
    xl["rprog_1234"].append(bool(np.all(np.diff(R[:4]) > 0)))
    xl["s_v2"].append(int(np.argmin(S)) == 1)
    ratio = R / np.abs(S)
    xl["trans"].append(f"V{int(np.argmax(ratio >= 1.0)) + 1}" if np.any(ratio >= 1.0) else "none")
    sv = np.linalg.svd(r["clean"] - r["clean"].mean(1, keepdims=True), compute_uv=False) ** 2
    xl["svd3"].append(sv[:3].sum() / sv.sum())
    xl["svd5"].append(sv[:5].sum() / sv.sum())
summary["across_leads"] = {
    "derivation_max_abs_err_mV": float(np.max(xl["deriv_err"])),
    "mean_abs_lead_to_lead_beat_shape_corr": q(xl["offdiag"]),
    "lead_pairs_corr_gt_0.95_of_66": q(xl["gt095"]),
    "most_similar_lead_pair_counts": Counter(p for p, _ in xl["pair"]).most_common(5),
    "most_similar_pair_corr": q([s for _, s in xl["pair"]]),
    "aVR_QRS_predominantly_negative_frac": float(np.mean(xl["avr_qrs_neg"])),
    "aVR_T_negative_frac": float(np.mean(xl["avr_t_neg"])),
    "aVR_P_negative_frac_sinus": float(np.mean(xl["avr_p_neg"])),
    "R_V1<R_V3<R_V5_frac": float(np.mean(xl["rprog_135"])),
    "R_strictly_rising_V1..V4_frac": float(np.mean(xl["rprog_1234"])),
    "S_deepest_in_V2_frac": float(np.mean(xl["s_v2"])),
    "transition_lead_counts": sorted(Counter(xl["trans"]).items()),
    "svd_energy_top3_components_clean": q(xl["svd3"]),
    "svd_energy_top5_components_clean": q(xl["svd5"]),
}
summary["per_lead_mean_R_S_mV_all_patients"] = {
    n: [round(float(np.mean([r["R_mV"][LI[n]] for r in recs])), 3),
        round(float(np.mean([r["S_mV"][LI[n]] for r in recs])), 3)] for n in LEADS}

# --- 4. noise / degeneracy --------------------------------------------------------------------
hashes = {hashlib.sha256(r["noisy"].astype(np.float32).tobytes()).hexdigest() for r in recs}
noise_rms = np.array([np.sqrt((r["noise"][RECORDED] ** 2).mean(axis=1)) for r in recs]).ravel()
sig_rms = np.array([np.sqrt(((r["clean"] - r["clean"].mean(1, keepdims=True))[RECORDED] ** 2).mean(axis=1))
                    for r in recs]).ravel()
allx = np.stack([r["noisy"] for r in recs])
same_a, same_b = one(g.AFIB, 12345, 1.0)[0], one(g.AFIB, 12345, 1.0)[0]
summary["noise_and_degeneracy"] = {
    "unique_records_of_total": [len(hashes), len(recs)],
    "same_seed_reproduces_exactly": bool(np.array_equal(same_a, same_b)),
    "noise_rms_mV_per_recorded_lead": q(noise_rms),
    "signal_rms_mV_per_recorded_lead": q(sig_rms),
    "snr_db_per_recorded_lead": q(20 * np.log10(sig_rms / noise_rms)),
    "noisy_vs_clean_leadII_corr": q([np.corrcoef(r["noisy"][LI["II"]], r["clean"][LI["II"]])[0, 1] for r in recs]),
    "nan_or_inf_samples": int((~np.isfinite(allx)).sum()),
    "global_min_max_mV": [round(float(allx.min()), 3), round(float(allx.max()), 3)],
    "leads_with_std_lt_0.01mV": int(sum(r["noisy"][l].std() < 0.01 for r in recs for l in range(12))),
    "leads_with_max_repeated_ge5_samples": int(sum(np.sum(np.isclose(r["noisy"][l], r["noisy"][l].max(), atol=1e-6)) >= 5
                                                   for r in recs for l in range(12))),
    "identical_lead_pairs_within_a_record": int(sum(np.allclose(r["noisy"][a], r["noisy"][b])
                                                    for r in recs for a, b in combinations(range(12), 2))),
}

# --- cross-check: the production path (one seed per batch, sequential rng draws) -------------
b = g.generate_ecg_batch(120, frac_afib=1 / 3, frac_other=1 / 3, seed=20261029, noise_level=1.0)
summary["batch_path_crosscheck_seed_20261029"] = {
    CLS_NAME[k]: {"hr_truth_bpm": q(b.heart_rate_bpm[b.labels == k]),
                  "rr_cv": q([np.std(np.diff(detect_r(b.ecg[i].astype(float))) / FS, ddof=1)
                              / np.mean(np.diff(detect_r(b.ecg[i].astype(float))) / FS)
                              for i in np.where(b.labels == k)[0]])}
    for k in CLASSES}

(OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
with open(OUT / "per_record_stats.csv", "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["class", "seed", "hr_truth_bpm", "hr_measured_bpm", "rr_cv", "rmssd_ms", "qrs_axis_deg",
                 "R_leadII_mV", "R_V5_mV", "S_V2_mV", "R_fwhm_V5_ms", "noise_rms_leadII_mV"])
    for r in recs:
        wr.writerow([CLS_NAME[r["label"]], r["seed"], r["hr_true"], round(r["hr_meas"], 1), round(r["rr_cv"], 4),
                     round(r["rmssd_ms"], 1), round(r["qrs_axis_deg"], 1), round(r["R_mV"][LI["II"]], 3),
                     round(r["R_mV"][LI["V5"]], 3), round(r["S_mV"][LI["V2"]], 3), round(r["R_fwhm_ms_V5"], 1),
                     round(float(np.sqrt((r["noise"][LI["II"]] ** 2).mean())), 4)])

# --- plots ------------------------------------------------------------------------------------
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
tsec = np.arange(g.N_SAMPLES) / FS

# (a1) 5 different patients' lead II per class, 10 s strips stacked with an offset.
fig, axes = plt.subplots(3, 1, figsize=(12, 10.5), sharex=True)
for ax, k in zip(axes, CLASSES):
    for j, r in enumerate([r for r in recs if r["label"] == k][:5]):
        off = -2.4 * j
        ax.plot(tsec, r["noisy"][LI["II"]] + off, color=COLORS[k], lw=0.9)
        ax.text(-0.15, off, f"patient {j + 1}\n{r['hr_true']} bpm\nRR CV {r['rr_cv']:.2f}",
                ha="right", va="center", fontsize=7.5, color=INK2)
    ax.set_title(f"{CLS_NAME[k]}: lead II from 5 different patients (strips 2.4 mV apart)", loc="left", fontsize=10)
    ax.set_yticks([])
    ax.set_xlim(-1.7, 10)
axes[-1].set_xlabel("time (s)")
fig.tight_layout()
fig.savefig(OUT / "a1_leadII_5_patients_per_class.png", dpi=150)
plt.close(fig)

# (a2) R-aligned median beats, all 40 patients per class, noise-free, lead II.
tt = np.arange(-TPL_PRE, TPL_POST) / FS * 1000
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
for ax, k in zip(axes, CLASSES):
    cr = [r for r in recs if r["label"] == k]
    for r in cr:
        ax.plot(tt, r["tpl_clean"][LI["II"]], color=COLORS[k], lw=0.7, alpha=0.4)
    ax.plot(tt, np.median([r["tpl_clean"][LI["II"]] for r in cr], axis=0), color=INK, lw=1.6)
    s = within[CLS_NAME[k]]["beat_shape_leadII_clean_corr"]
    ax.set_title(f"{CLS_NAME[k]}: {len(cr)} patients\nshape corr between patients:\n"
                 f"median {s['median']:.2f}, 5th pct {s['p5']:.2f}", loc="left", fontsize=9.5)
    ax.set_xlabel("ms from R peak")
axes[0].set_ylabel("lead II (mV), noise-free")
fig.text(0.99, 0.01, "thin line = one patient's median beat   ·   black = class median",
         ha="right", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT / "a2_leadII_beat_shapes_40_patients_per_class.png", dpi=150)
plt.close(fig)

# (b) all 12 leads for one representative (most typical) patient per class, shared mV scale.
for k in CLASSES:
    cr = [r for r in recs if r["label"] == k]
    rep = cr[int(np.argmax(np.corrcoef(np.stack([r["tpl_clean"].ravel() for r in cr])).mean(axis=1)))]
    fig, axes = plt.subplots(6, 2, figsize=(12, 9), sharex=True, sharey=True)
    win = tsec < 5.0
    for col, names in enumerate((["I", "II", "III", "aVR", "aVL", "aVF"], ["V1", "V2", "V3", "V4", "V5", "V6"])):
        for row, name in enumerate(names):
            ax = axes[row, col]
            ax.plot(tsec[win], rep["noisy"][LI[name]][win], color=COLORS[k], lw=0.9)
            ax.text(0.01, 0.92, name, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("mV")
    fig.suptitle(f"{CLS_NAME[k]}: all 12 leads, one typical patient (seed {rep['seed']}, {rep['hr_true']} bpm, "
                 f"QRS axis {rep['qrs_axis_deg']:.0f}°), first 5 s, same mV scale on every lead",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / f"b_12_leads_{CLS_NAME[k].split()[0]}.png", dpi=150)
    plt.close(fig)

# (c) heart rate vs RR irregularity, the two features the classes are built from.
fig, ax = plt.subplots(figsize=(7.5, 5))
for k in CLASSES:
    m = lab == k
    ax.scatter(hr[m], cv[m], s=40, color=COLORS[k], edgecolor="white", linewidth=1.5, label=CLS_NAME[k])
    ax.annotate(CLS_NAME[k], (np.median(hr[m]), np.max(cv[m])), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=9, color=INK)
ax.set_xlabel("heart rate measured from the noisy signal (bpm)")
ax.set_ylabel("RR-interval coefficient of variation (std / mean)")
ax.set_title(f"Each dot = one synthetic patient ({N_PER_CLASS} per class)", loc="left", fontsize=10)
ax.set_ylim(top=cv.max() * 1.1)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "c_heart_rate_vs_rr_irregularity.png", dpi=150)
plt.close(fig)
print(json.dumps(summary, indent=1, default=str))
