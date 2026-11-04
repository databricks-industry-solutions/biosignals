"""Tests for the shared 12-lead generator (common/ecg_signal_gen.py).

    cd projects/ecg12-afib && python -m pytest dev/tests -v
"""
import numpy as np
import pytest
from scipy.signal import find_peaks

from common import config
from common.ecg_signal_gen import (
    AFIB, DERIVED_LEADS, HEART_RATE_RANGE_BPM, LABEL_NAMES, LEADS_12, N_SAMPLES,
    NSR, OTHER, SAMPLING_RATE, class_counts, derive_limb_leads, generate_ecg_batch, lead_names_for,
    synthesize_ecgs,
)


def _leads(batch):
    """name -> (n, N_SAMPLES) view of a batch."""
    return {name: batch.ecg[:, i, :] for i, name in enumerate(batch.lead_names)}


def test_default_is_12_lead_contract_from_config():
    batch = generate_ecg_batch(4, seed=1)
    assert config.N_LEADS == 12
    assert batch.ecg.shape == (4, config.N_LEADS, N_SAMPLES)
    assert batch.ecg.dtype == np.float32
    assert batch.lead_names == LEADS_12 == lead_names_for(config.N_LEADS)
    assert LEADS_12 == ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
    assert set(np.unique(batch.labels)).issubset(set(LABEL_NAMES))
    assert np.isfinite(batch.ecg).all()
    assert np.abs(batch.ecg).max() < 5.0          # millivolts, clinically plausible range


@pytest.mark.parametrize("noise_level", [0.0, 1.0])
def test_derived_limb_leads_follow_einthoven_and_goldberger(noise_level):
    """III = II - I, aVR = -(I+II)/2, aVL = I - II/2, aVF = II - I/2 hold on the output,
    with and without acquisition noise (noise goes on the recorded leads before deriving)."""
    batch = generate_ecg_batch(30, frac_afib=0.34, frac_other=0.33, seed=3, noise_level=noise_level)
    L = _leads(batch)
    atol = 1e-5   # float32 rounding of mV values
    np.testing.assert_allclose(L["III"], L["II"] - L["I"], atol=atol)
    np.testing.assert_allclose(L["aVR"], -(L["I"] + L["II"]) / 2, atol=atol)
    np.testing.assert_allclose(L["aVL"], L["I"] - L["II"] / 2, atol=atol)
    np.testing.assert_allclose(L["aVF"], L["II"] - L["I"] / 2, atol=atol)
    np.testing.assert_allclose(L["aVR"] + L["aVL"] + L["aVF"], 0.0, atol=atol)   # Goldberger identity
    np.testing.assert_allclose(L["I"] + L["III"], L["II"], atol=atol)               # Einthoven's law


def test_derive_limb_leads_formulas():
    i, ii = np.array([1.0, 0.2, -0.4]), np.array([0.5, 1.0, 0.8])
    d = derive_limb_leads(i, ii)
    assert set(d) == set(DERIVED_LEADS)
    np.testing.assert_allclose(d["III"], [-0.5, 0.8, 1.2])
    np.testing.assert_allclose(d["aVR"], [-0.75, -0.6, -0.2])
    np.testing.assert_allclose(d["aVL"], [0.75, -0.3, -0.8])
    np.testing.assert_allclose(d["aVF"], [0.0, 0.9, 1.0])


def test_unsupported_lead_count_is_rejected():
    with pytest.raises(ValueError, match="n_leads=5"):
        generate_ecg_batch(2, seed=1, n_leads=5)


def test_seed_reproducible_and_labels_follow_fractions():
    a = generate_ecg_batch(20, frac_afib=0.25, frac_other=0.25, seed=42)
    b = generate_ecg_batch(20, frac_afib=0.25, frac_other=0.25, seed=42)
    np.testing.assert_array_equal(a.ecg, b.ecg)
    assert (a.labels == AFIB).sum() == 5 and (a.labels == OTHER).sum() == 5 and (a.labels == NSR).sum() == 10
    with pytest.raises(ValueError):
        generate_ecg_batch(4, frac_afib=0.7, frac_other=0.5)
    with pytest.raises(ValueError, match="unknown rhythm"):
        synthesize_ecgs(np.array([0, 7]), np.random.default_rng(0))


def test_class_counts_keep_the_mix_for_small_n():
    """Counts sum to n and each is its share rounded down or up, also where rounding both abnormal
    shares half to even used to miss (n=5 at 0.5 / 0.5 gave 2 + 2 and 1 NSR; n=1 gave 1 NSR)."""
    assert class_counts(5, 0.5, 0.5) == {AFIB: 3, OTHER: 2, NSR: 0}
    assert class_counts(1, 0.5, 0.5) == {AFIB: 1, OTHER: 0, NSR: 0}
    assert class_counts(6, 0.25, 0.25) == {AFIB: 2, OTHER: 1, NSR: 3}
    assert class_counts(25, 0.02, 0.58) == {AFIB: 1, OTHER: 14, NSR: 10}       # NSR share exactly 10, not 11
    assert class_counts(200, 0.34, 0.33) == {AFIB: 68, OTHER: 66, NSR: 66}      # as before (val / holdout)
    for n in range(1, 13):
        for fa, fo in [(a / 4, o / 4) for a in range(5) for o in range(5 - a)]:
            counts = class_counts(n, fa, fo)
            shares = {AFIB: n * fa, OTHER: n * fo, NSR: n * (1 - fa - fo)}
            assert sum(counts.values()) == n, (n, fa, fo)
            assert all(abs(counts[k] - shares[k]) < 1 for k in shares), (n, fa, fo, counts)
    labels = generate_ecg_batch(5, frac_afib=0.5, frac_other=0.5, seed=1, noise_level=0.0).labels
    assert [int((labels == k).sum()) for k in (AFIB, OTHER, NSR)] == [3, 2, 0]


def test_normal_12_lead_morphology():
    """Clean NSR: aVR mostly negative, V1 rS (S deeper than R), R grows V1 -> V5 and is smaller in
    V6, S deepest in V1/V2 and shallowest in V6."""
    L = _leads(generate_ecg_batch(200, frac_afib=0.0, frac_other=0.0, seed=5, noise_level=0.0))
    r = {n: L[n].max(axis=1).mean() for n in LEADS_12}
    s = {n: L[n].min(axis=1).mean() for n in LEADS_12}
    assert -s["aVR"] > r["aVR"]
    assert -s["V1"] > r["V1"]
    assert r["V1"] < r["V2"] < r["V3"] < r["V4"] < r["V5"]
    assert r["V6"] < r["V5"]
    assert min(s["V1"], s["V2"]) < s["V4"] < s["V6"]
    assert r["II"] > r["aVL"]          # normal axis: II taller than aVL on average


def _r_peaks(v5):
    peaks, _ = find_peaks(v5, height=0.5 * v5.max(), distance=int(0.25 * SAMPLING_RATE))
    return peaks


@pytest.mark.parametrize("label", [NSR, AFIB, OTHER])
def test_heart_rate_is_measured_from_the_signal_and_in_class_range(label):
    batch = synthesize_ecgs(np.full(40, label), np.random.default_rng(label), noise_level=0.0)
    lo, hi = HEART_RATE_RANGE_BPM[label]
    assert ((batch.heart_rate_bpm >= lo - 10) & (batch.heart_rate_bpm <= hi + 10)).all()
    v5 = batch.ecg[:, LEADS_12.index("V5"), :]
    for hr, lead in zip(batch.heart_rate_bpm, v5):
        rr = np.diff(_r_peaks(lead)) / SAMPLING_RATE
        assert abs(60.0 / rr.mean() - hr) <= 0.1 * hr


def test_rhythm_features_match_the_label():
    """AFib: irregular RR and no P wave; NSR / Other: regular RR with a P wave before each QRS.

    The P wave is measured on the R-aligned average beat of lead II: mean level 130-190 ms before
    R (where the P wave peaks) above the PR segment 70 ms before R. Averaging keeps a P wave and
    cancels f-waves, which have no fixed timing relative to the QRS."""
    fs = SAMPLING_RATE
    start = int(0.25 * fs)                                     # average beat = 250 ms before R
    p_lo, p_hi, pr = start - int(0.19 * fs), start - int(0.13 * fs), start - int(0.07 * fs)

    def features(label):
        batch = synthesize_ecgs(np.full(30, label), np.random.default_rng(100 + label), noise_level=0.0)
        rr_cv, p_wave = [], []
        for ecg in batch.ecg:
            peaks = _r_peaks(ecg[LEADS_12.index("V5")])
            rr = np.diff(peaks)
            rr_cv.append(rr.std() / rr.mean())
            lead_ii = ecg[LEADS_12.index("II")]
            avg = np.mean([lead_ii[p - start:p] for p in peaks if p >= start], axis=0)   # 250 ms before R
            p_wave.append(avg[p_lo:p_hi].mean() - avg[pr])
        return np.median(rr_cv), np.median(p_wave)

    nsr_cv, nsr_p = features(NSR)
    other_cv, other_p = features(OTHER)
    afib_cv, afib_p = features(AFIB)
    assert nsr_cv < 0.06 and other_cv < 0.06 and afib_cv > 0.12
    assert nsr_p > 0.05 and other_p > 0.05
    assert afib_p < 0.04 and afib_p < 0.5 * min(nsr_p, other_p)
