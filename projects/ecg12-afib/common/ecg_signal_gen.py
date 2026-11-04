"""Synthetic, labeled clinical 12-lead ECG generator for the ECG/AFib demo.

Fully synthetic (no patient data). One generator feeds the persisted training dataset
(01_generate_training_data), the daily batch (06_generate_batch_data), the serving smoke test
(05_serve_endpoints) and the app's live tab (byte-identical copy in
``app/src/ecg_demo_react/backend/common/``).

Rhythm classes (``labels``):

  0  NSR    normal sinus rhythm: regular RR, P-QRS-T, 60-100 bpm             arr_flag 0
  1  AFIB   atrial fibrillation: irregularly irregular RR, no P waves,        arr_flag 1
            fibrillatory f-waves (4.5-8.5 Hz), ventricular rate 100-160 bpm
  2  OTHER  sinus tachycardia: regular RR, P waves present, 110-160 bpm        arr_flag 1

Triage flags both abnormal classes; the AFib model must then separate AFIB from OTHER.

Leads (standard order ``LEADS_12``). The eight leads an ECG machine actually records are
synthesized; the other four are computed from I and II, as ECG machines do:

  independent  I, II           frontal-plane projection of per-wave cardiac vectors
                                (P, septal q, R, terminal s, T; QRS axis -30..+90 deg)
               V1..V6          per-lead wave amplitudes with normal R-wave progression
                                (R grows V1->V5, smaller in V6; S deepest V2; R~S at V3)
  derived      III = II - I    aVR = -(I + II) / 2    aVL = I - II / 2    aVF = II - I / 2

Every class uses the same beat model (sum of Gaussian P, Q, R, S, T waves, the wave shape
of McSharry et al. 2003, "A dynamical model for generating synthetic ECG signals"), so the
classes differ only in rhythm features, not in how they were synthesized.

neurokit2 0.2.12 ``ecg_simulate(method="multileads")`` was checked and not used: it rescales
every lead to the same -0.4..1.2 mV range (no amplitude differences between leads), its
aVR and V1 come out mostly positive, its limb leads break Einthoven's law, and it has no AFib
mode. neurokit2 is still used for realistic acquisition noise (``signal_distort``).

Units: millivolts (mV), 250 Hz, 10 s. Noise is added to the eight recorded leads before the
four limb leads are derived, so the derivation formulas hold exactly on the output.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import NamedTuple

import numpy as np

from .config import N_LEADS

# --- Signal geometry (model-input contract: (-1, n_leads, N_SAMPLES)) ----------------------
N_SAMPLES = 2500
SAMPLING_RATE = 250                        # Hz
DURATION = N_SAMPLES / SAMPLING_RATE       # 10.0 s

# --- Rhythm classes -------------------------------------------------------------------------
NSR, AFIB, OTHER = 0, 1, 2
LABEL_NAMES = {NSR: "NSR", AFIB: "AFib", OTHER: "Other(tachy)"}

# Heart-rate range (bpm) per class. Rates are drawn from a clipped normal inside the range;
# the returned heart_rate_bpm is measured from the synthesized beats.
HEART_RATE_RANGE_BPM = {NSR: (60.0, 100.0), AFIB: (100.0, 160.0), OTHER: (110.0, 160.0)}
_HEART_RATE_CENTER_BPM = {NSR: (74.0, 9.0), AFIB: (124.0, 14.0), OTHER: (130.0, 12.0)}

# --- Leads ----------------------------------------------------------------------------------
LEADS_12 = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
INDEPENDENT_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
DERIVED_LEADS = ("III", "aVR", "aVL", "aVF")

# Waves of the beat model, in column order of the amplitude matrices below.
_WAVES = ("P", "Q", "R", "S", "T")

# Precordial amplitudes (mV) for a typical adult, columns = _WAVES. R grows V1 -> V5 and is
# smaller in V6, S is deepest in V2, R ~ S at V3 (transition), T is flat/inverted in V1.
_PRECORDIAL_MV = np.array([
    #  P      Q      R      S      T
    [0.06,  0.00,  0.30, -1.00, -0.02],   # V1
    [0.08,  0.00,  0.60, -1.40,  0.45],   # V2
    [0.08, -0.02,  0.95, -0.90,  0.50],   # V3
    [0.08, -0.05,  1.35, -0.50,  0.42],   # V4
    [0.08, -0.08,  1.45, -0.25,  0.32],   # V5
    [0.08, -0.10,  1.10, -0.10,  0.25],   # V6
])
# How strongly atrial f-waves show in V1..V6 relative to lead II (largest in V1).
_PRECORDIAL_F_GAIN = np.array([1.4, 0.9, 0.6, 0.4, 0.3, 0.25])

# Acquisition noise, absolute mV, scaled by noise_level.
_EMG_MV = 0.015        # broadband / muscle
_MOTION_MV = 0.05      # electrode-contact pops
_HASH_MV = 0.008       # high-frequency hash (50/60 Hz mains is not resolvable at 250 Hz)
_WANDER_MV = 0.05      # baseline wander (respiration / motion), per component


class EcgBatch(NamedTuple):
    """Output of the generator. Unpacks as ``ecg, labels, heart_rate_bpm, lead_names``."""
    ecg: np.ndarray             # float32 (n, n_leads, N_SAMPLES), mV, lead order = lead_names
    labels: np.ndarray          # int (n,), values in LABEL_NAMES
    heart_rate_bpm: np.ndarray  # int (n,), ventricular rate measured from the beats
    lead_names: tuple[str, ...]


def lead_names_for(n_leads: int = N_LEADS) -> tuple[str, ...]:
    """Lead names (and stored-column order) for ``n_leads``: ``LEADS_12``, the only lead set."""
    if n_leads != len(LEADS_12):
        raise ValueError(f"n_leads={n_leads} not supported; the generator gives all {len(LEADS_12)} leads")
    return LEADS_12


# How each derived lead is computed from I and II (as text; derive_limb_leads computes it).
DERIVATION = {"III": "II - I", "aVR": "-(I + II) / 2", "aVL": "I - II / 2", "aVF": "II - I / 2"}


def derive_limb_leads(lead_i: np.ndarray, lead_ii: np.ndarray) -> dict[str, np.ndarray]:
    """III, aVR, aVL, aVF from leads I and II (Einthoven's law + Goldberger's leads)."""
    return {
        "III": lead_ii - lead_i,
        "aVR": -(lead_i + lead_ii) / 2.0,
        "aVL": lead_i - lead_ii / 2.0,
        "aVF": lead_ii - lead_i / 2.0,
    }


def describe_lead(name: str) -> str:
    """Column comment for a stored lead, e.g. ``Lead III (derived: II - I), mV, ...``."""
    how = f"derived: {name} = {DERIVATION[name]}" if name in DERIVATION else "recorded"
    return f"Lead {name} ({how}), mV, {N_SAMPLES} samples at {SAMPLING_RATE} Hz"


def derive_arr_flag(labels: np.ndarray) -> np.ndarray:
    """Triage truth: 1.0 for any abnormal rhythm (AFib or Other), else 0.0."""
    return (np.asarray(labels) != NSR).astype("float32")


def derive_afib_truth(labels: np.ndarray) -> np.ndarray:
    """AFib truth: 1.0 for AFib, else 0.0."""
    return (np.asarray(labels) == AFIB).astype("float32")


def class_counts(n: int, frac_afib: float = 0.25, frac_other: float = 0.25) -> dict[int, int]:
    """ECGs per rhythm class (``LABEL_NAMES`` keys) for ``n`` ECGs with the given mix, the rest NSR.

    The counts sum to ``n`` and each is its share (``n * frac``; NSR: ``n * (1 - frac_afib -
    frac_other)``) rounded down or up. First the counts every persisted dataset was drawn with:
    AFib and Other each rounded to the nearest count (Other capped at what AFib leaves), NSR the
    rest. Those miss only when both abnormal shares end in one half, as Python rounds halves to
    even (n=5 at 0.5 / 0.5 gave 2 + 2 and 1 NSR, n=1 gave 1 NSR); then the counts are the
    largest-remainder allocation of the three shares (equal remainders: AFib, then Other, then NSR).

    The shares are computed exactly, from each fraction's shortest decimal form (0.58 as 29/50),
    not in floating point: there 25 x 0.58 is 14.499999999999998, 25 - 25 x 0.02 - 25 x 0.58 is
    10.000000000000002 and 5 x (1 - 0.3 - 0.3) is 1.9999999999999998, which would accept a count
    past its share (11 NSR for a share of 10, 1 for a share of 2)."""
    if frac_afib + frac_other > 1.0:
        raise ValueError(f"frac_afib + frac_other = {frac_afib + frac_other} > 1.0")
    # The old counts, from the float products exactly as the old code rounded them.
    n_afib = min(int(round(n * frac_afib)), n)
    counts = {AFIB: n_afib, OTHER: min(int(round(n * frac_other)), n - n_afib)}
    counts[NSR] = n - counts[AFIB] - counts[OTHER]
    afib, other = Fraction(str(float(frac_afib))), Fraction(str(float(frac_other)))
    shares = {AFIB: n * afib, OTHER: n * other, NSR: max(n * (1 - afib - other), Fraction(0))}
    if all(math.floor(shares[k]) <= counts[k] <= math.ceil(shares[k]) for k in shares):
        return counts
    counts = {k: math.floor(share) for k, share in shares.items()}
    by_remainder = sorted(shares, key=lambda k: counts[k] - shares[k])     # stable: AFib, Other, NSR
    for k in by_remainder[:n - sum(counts.values())]:
        counts[k] += 1
    return counts


def generate_ecg_batch(
    n: int,
    frac_afib: float = 0.25,
    frac_other: float = 0.25,
    seed: int | None = None,
    noise_level: float = 1.0,
    n_leads: int = N_LEADS,
) -> EcgBatch:
    """``n`` labeled ECGs with the given class mix (``class_counts``; the rest NSR), shuffled.

    ``noise_level`` scales acquisition noise (0 = clean, 1 = realistic). ``seed`` makes the
    batch reproducible.
    """
    counts = class_counts(n, frac_afib, frac_other)
    rng = np.random.default_rng(seed)
    labels = np.full(n, NSR, dtype=int)
    order = rng.permutation(n)
    labels[order[:counts[AFIB]]] = AFIB
    labels[order[counts[AFIB]:counts[AFIB] + counts[OTHER]]] = OTHER
    return synthesize_ecgs(labels, rng, noise_level=noise_level, n_leads=n_leads)


def synthesize_ecgs(
    labels: np.ndarray,
    rng: np.random.Generator,
    noise_level: float = 1.0,
    n_leads: int = N_LEADS,
) -> EcgBatch:
    """One ECG per entry of ``labels`` (each a key of ``LABEL_NAMES``)."""
    names = lead_names_for(n_leads)
    labels = np.asarray(labels, dtype=int)
    unknown = set(np.unique(labels).tolist()) - set(LABEL_NAMES)
    if unknown:
        raise ValueError(f"unknown rhythm labels {sorted(unknown)}; expected {sorted(LABEL_NAMES)}")
    ecg = np.empty((labels.shape[0], len(names), N_SAMPLES), dtype=np.float32)
    heart_rate = np.empty(labels.shape[0], dtype=int)
    for i, label in enumerate(labels):
        leads12, heart_rate[i] = _synthesize_one(int(label), rng, noise_level)
        ecg[i] = leads12.astype(np.float32)
    return EcgBatch(ecg=ecg, labels=labels, heart_rate_bpm=heart_rate, lead_names=names)


# --- Internals ------------------------------------------------------------------------------

def _clipped_normal(rng: np.random.Generator, mean: float, sd: float, lo: float, hi: float) -> float:
    return float(np.clip(rng.normal(mean, sd), lo, hi))


def _r_peak_times(label: int, rng: np.random.Generator) -> np.ndarray:
    """R-peak times (s), starting one beat before t=0 so the strip opens mid-rhythm."""
    lo, hi = HEART_RATE_RANGE_BPM[label]
    center, spread = _HEART_RATE_CENTER_BPM[label]
    mean_rr = 60.0 / _clipped_normal(rng, center, spread, lo, hi)
    resp_hz, resp_phase = rng.uniform(0.2, 0.33), rng.uniform(0, 2 * np.pi)
    times = [-rng.uniform(0.0, mean_rr)]
    while times[-1] < DURATION + 0.5:
        if label == AFIB:   # irregularly irregular ventricular response
            rr = max(rng.normal(mean_rr, 0.20 * mean_rr), 0.25)
        else:               # regular, with mild respiratory sinus arrhythmia
            rr = mean_rr * (1 + 0.03 * np.sin(2 * np.pi * resp_hz * times[-1] + resp_phase)
                            + rng.normal(0.0, 0.008))
        times.append(times[-1] + rr)
    return np.asarray(times)


def _frontal_amplitudes(rng: np.random.Generator, voltage: float) -> np.ndarray:
    """(2, 5) wave amplitudes (mV) in leads I and II: each wave is a frontal-plane vector
    (axis, magnitude) projected onto lead I (0 deg) and lead II (60 deg)."""
    qrs_axis = _clipped_normal(rng, 50.0, 20.0, -30.0, 90.0)   # normal axis -30..+90 deg
    axes = np.array([
        _clipped_normal(rng, 55.0, 10.0, 0.0, 75.0),              # P
        qrs_axis + 150.0 + rng.normal(0.0, 10.0),                 # septal q (rightward)
        qrs_axis,                                                 # R
        qrs_axis - 170.0 + rng.normal(0.0, 10.0),                 # terminal s (superior-right)
        _clipped_normal(rng, qrs_axis - 5.0, 10.0, qrs_axis - 40.0, qrs_axis + 40.0),  # T
    ])
    magnitude = np.array([0.14, 0.12, 1.15 * voltage, 0.30 * voltage, 0.32 * voltage])
    magnitude = magnitude * rng.normal(1.0, 0.07, 5)
    lead_angles = np.deg2rad([0.0, 60.0])[:, None]                # I, II
    return magnitude[None, :] * np.cos(np.deg2rad(axes)[None, :] - lead_angles)


def _precordial_amplitudes(rng: np.random.Generator, voltage: float) -> np.ndarray:
    """(6, 5) wave amplitudes (mV) in V1..V6, with per-patient rotation of the transition
    zone (R and S columns shifted along the leads) and amplitude jitter."""
    amps = _PRECORDIAL_MV.copy()
    shift = rng.normal(0.0, 0.35)                    # lead positions; >0 = earlier transition
    pos = np.clip(np.arange(6) + shift, 0, 5)
    for col in (2, 3):                               # R, S
        amps[:, col] = np.interp(pos, np.arange(6), _PRECORDIAL_MV[:, col])
    amps[:, 2:] *= voltage                           # QRS + T scale with body habitus
    amps[0, 4] = rng.normal(-0.02, 0.06)             # V1 T: flat, slightly up or down
    return amps * rng.normal(1.0, 0.07, amps.shape)


def _wave_trains(r_times: np.ndarray, label: int, rng: np.random.Generator) -> np.ndarray:
    """(5, N_SAMPLES) unit-amplitude P, Q, R, S, T trains placed around every R peak."""
    t = np.arange(N_SAMPLES) / SAMPLING_RATE
    rr = np.clip(np.diff(r_times, prepend=r_times[0] - np.median(np.diff(r_times))), 0.3, 1.5)
    n_beats = r_times.shape[0]
    # Center (s, relative to R) and width (s) per wave and beat. QT follows Bazett
    # (QTc 0.40 s); PR shortens slightly at fast rates.
    centers = np.stack([
        -(0.12 + 0.04 * np.minimum(rr, 1.2)),
        np.full(n_beats, -0.026),
        np.zeros(n_beats),
        np.full(n_beats, 0.026),
        -0.04 + 0.40 * np.sqrt(rr) - 0.08,
    ])
    widths = np.stack([
        np.full(n_beats, 0.020), np.full(n_beats, 0.008), np.full(n_beats, 0.010),
        np.full(n_beats, 0.010), 0.045 * np.sqrt(rr),
    ])
    # Beat-to-beat amplitude variation: respiration in sinus rhythm, more scatter in AFib.
    if label == AFIB:
        beat_gain = np.stack([np.zeros(n_beats)] + [rng.normal(1.0, 0.05, n_beats)] * 3
                             + [rng.normal(1.0, 0.10, n_beats)])
    else:
        resp = 1 + 0.04 * np.sin(2 * np.pi * rng.uniform(0.2, 0.33) * r_times + rng.uniform(0, 2 * np.pi))
        beat_gain = np.stack([resp + rng.normal(0.0, 0.015, n_beats) for _ in _WAVES])
    peaks = r_times[None, :] + centers                                      # (5, beats)
    pulses = np.exp(-((t[None, None, :] - peaks[:, :, None]) ** 2) / (2.0 * widths[:, :, None] ** 2))
    return (beat_gain[:, :, None] * pulses).sum(axis=1)


def _f_waves(rng: np.random.Generator) -> np.ndarray:
    """AFib fibrillatory baseline: a few 4.5-8.5 Hz components with slow amplitude drift,
    scaled so the RMS amplitude is 0.03-0.08 mV."""
    t = np.arange(N_SAMPLES) / SAMPLING_RATE
    f = np.zeros(N_SAMPLES)
    for _ in range(int(rng.integers(3, 6))):
        f += rng.uniform(0.5, 1.0) * np.sin(2 * np.pi * rng.uniform(4.5, 8.5) * t + rng.uniform(0, 2 * np.pi))
    f *= 1 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.1, 0.4) * t + rng.uniform(0, 2 * np.pi))
    return f / np.sqrt(np.mean(f ** 2)) * rng.uniform(0.03, 0.08)


def _acquisition_noise(rng: np.random.Generator, noise_level: float) -> np.ndarray:
    """Noise for one recorded lead (mV): EMG + motion artifacts (neurokit2), hash, wander.
    signal_distort scales amplitudes by the input's SD unless the SD is 0, so it is called on
    a flat signal to get absolute amplitudes."""
    import neurokit2 as nk   # here, so notebooks that only need N_SAMPLES skip neurokit2

    t = np.arange(N_SAMPLES) / SAMPLING_RATE
    noise = nk.signal_distort(
        np.zeros(N_SAMPLES),
        sampling_rate=SAMPLING_RATE,
        noise_amplitude=_EMG_MV * noise_level,
        noise_frequency=20,          # <= 25 Hz: neurokit skips anything above 0.1 x sampling rate
        artifacts_amplitude=_MOTION_MV * noise_level,
        artifacts_number=3,
        artifacts_frequency=20,
        linear_drift=False,          # a linear ramp is unrealistic; wander is added below
        random_state=int(rng.integers(0, 2**31 - 1)),
        silent=True,
    )
    noise = noise + rng.normal(0.0, _HASH_MV * noise_level, N_SAMPLES)
    for wander_hz in (rng.uniform(0.15, 0.3), rng.uniform(0.3, 0.5)):
        noise = noise + _WANDER_MV * noise_level * np.sin(2 * np.pi * wander_hz * t + rng.uniform(0, 2 * np.pi))
    return noise


def _synthesize_one(label: int, rng: np.random.Generator, noise_level: float) -> tuple[np.ndarray, int]:
    """All 12 leads (float64, mV, LEADS_12 order) + the measured heart rate for one ECG."""
    r_times = _r_peak_times(label, rng)
    trains = _wave_trains(r_times, label, rng)                         # (5, N)
    voltage = float(rng.lognormal(0.0, 0.15))                          # per-patient QRS voltage
    amps = np.vstack([_frontal_amplitudes(rng, voltage), _precordial_amplitudes(rng, voltage)])
    if label == AFIB:
        amps[:, 0] = 0.0                                               # no P waves
    recorded = amps @ trains                                           # (8, N): I, II, V1..V6

    if label == AFIB:
        f = _f_waves(rng)
        atrial_axis = np.deg2rad(_clipped_normal(rng, 70.0, 15.0, 30.0, 110.0))
        f_gain = np.concatenate([np.cos(atrial_axis - np.deg2rad([0.0, 60.0])), _PRECORDIAL_F_GAIN])
        recorded = recorded + f_gain[:, None] * f[None, :]

    if noise_level > 0:
        recorded = recorded + np.stack([_acquisition_noise(rng, noise_level) for _ in INDEPENDENT_LEADS])
    recorded = recorded - recorded.mean(axis=1, keepdims=True)

    by_name = dict(zip(INDEPENDENT_LEADS, recorded))
    by_name.update(derive_limb_leads(by_name["I"], by_name["II"]))

    in_window = r_times[(r_times >= 0.0) & (r_times < DURATION)]
    heart_rate = int(round(60.0 / float(np.mean(np.diff(in_window)))))
    return np.stack([by_name[name] for name in LEADS_12]), heart_rate
