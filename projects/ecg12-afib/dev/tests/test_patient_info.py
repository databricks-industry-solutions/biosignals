"""Tests for the synthetic, non-PII patient record (common/patient_info.py)."""
import dataclasses
import datetime as dt

import numpy as np
import pytest

from common import patient_info as pi
from common.ecg_signal_gen import AFIB, DURATION, NSR, OTHER, SAMPLING_RATE, generate_ecg_batch

RUN_DATE = dt.date(2026, 10, 29)


@pytest.fixture(scope="module")
def batch_and_records():
    batch = generate_ecg_batch(120, frac_afib=0.3, frac_other=0.2, seed=20261029)
    ids = pi.assign_patients(batch.labels, np.random.default_rng(1))
    records = [pi.patient_record(pid, hr, pi.acquisition_time(pid, RUN_DATE), ecg)
               for pid, hr, ecg in zip(ids, batch.heart_rate_bpm, batch.ecg)]
    return batch, records


def test_schema_fields_and_spark_specs_match():
    names = [f.name for f in dataclasses.fields(pi.PatientRecord)]
    assert names == list(pi.FIELD_SPECS)
    assert names == [
        "pseudo_patient_id", "mrn", "age_years", "sex", "height_cm", "weight_kg", "bmi",
        "bp_systolic_mmhg", "bp_diastolic_mmhg", "acquisition_datetime_utc", "heart_rate_bpm",
        "sampling_rate_hz", "n_leads", "duration_s",
    ]
    assert not any("name" in n for n in names)      # no person-name fields, by design


def test_consistent_per_pseudo_patient_id():
    pid = pi.patient_id(123)
    ecg = np.zeros((12, 2500), dtype=np.float32)
    a = pi.patient_record(pid, 70, pi.acquisition_time(pid, RUN_DATE), ecg)
    b = pi.patient_record(pid, 130, pi.acquisition_time(pid, RUN_DATE + dt.timedelta(days=5)), ecg)
    same = ["mrn", "age_years", "sex", "height_cm", "weight_kg", "bmi", "bp_systolic_mmhg", "bp_diastolic_mmhg"]
    assert {k: getattr(a, k) for k in same} == {k: getattr(b, k) for k in same}
    assert pi.demographics(pid) == pi.demographics(pi.patient_id(123))
    assert pi.demographics(pi.patient_id(124)) != pi.demographics(pid)


def test_values_are_synthetic_plausible_and_derived(batch_and_records):
    _, records = batch_and_records
    assert len({r.pseudo_patient_id for r in records}) == len(records)       # one patient per ECG
    for r in records:
        assert r.pseudo_patient_id.startswith(pi.ID_PREFIX) and r.mrn.startswith(pi.MRN_PREFIX)
        assert 40 <= r.age_years <= 90 and r.sex in {"F", "M"}
        assert 145 <= r.height_cm <= 200 and 30 <= r.weight_kg <= 200
        assert r.bmi == round(r.weight_kg / (r.height_cm / 100) ** 2, 1)    # derived from stored values
        assert 90 <= r.bp_systolic_mmhg <= 200 and r.bp_diastolic_mmhg <= r.bp_systolic_mmhg - 25
        assert r.acquisition_datetime_utc.tzinfo == dt.timezone.utc
        assert r.acquisition_datetime_utc.date() == RUN_DATE
        assert 7 <= r.acquisition_datetime_utc.hour <= 19


def test_acquisition_fields_come_from_the_ecg(batch_and_records):
    batch, records = batch_and_records
    for r, hr in zip(records, batch.heart_rate_bpm):
        assert r.sampling_rate_hz == SAMPLING_RATE == 250
        assert r.n_leads == batch.ecg.shape[1] == 12
        assert r.duration_s == DURATION == 10.0
        assert r.heart_rate_bpm == hr
    eight = pi.patient_record(pi.patient_id(1), 70, dt.datetime.now(dt.timezone.utc), np.zeros((8, 2500)))
    assert eight.n_leads == 8


def test_heart_rate_follows_rhythm(batch_and_records):
    batch, records = batch_and_records
    hr = np.array([r.heart_rate_bpm for r in records])
    assert 55 <= np.median(hr[batch.labels == NSR]) <= 100
    assert np.median(hr[batch.labels == AFIB]) > 100
    assert np.median(hr[batch.labels == OTHER]) > 110


def test_afib_patients_skew_older():
    rng = np.random.default_rng(0)
    labels = rng.choice([NSR, AFIB, OTHER], size=3000, p=[0.5, 0.3, 0.2])
    ids = pi.assign_patients(labels, rng)
    ages = np.array([pi.demographics(p).age_years for p in ids])
    assert len(set(ids)) == len(ids)
    assert ages[labels == AFIB].mean() > ages[labels == NSR].mean() + 10


def test_record_serializes_to_a_row():
    pid = pi.patient_id(7)
    row = pi.patient_record(pid, 72, pi.acquisition_time(pid, RUN_DATE), np.zeros((12, 2500))).to_row()
    assert list(row) == list(pi.FIELD_SPECS) and row["pseudo_patient_id"] == pid
    with pytest.raises(ValueError):
        pi.assign_patients(np.zeros(pi.POPULATION_SIZE + 1, dtype=int), np.random.default_rng(0))


def test_acquisition_times_spread_over_the_clinic_day():
    """Arrivals look like a stream: inside clinic hours, spread over the day, a busier morning, and no
    pile-up at opening or closing time."""
    times = [pi.acquisition_time(pi.patient_id(k), RUN_DATE) for k in range(2000)]
    hours = np.array([t.hour + t.minute / 60 + t.second / 3600 for t in times])
    opening, closing = pi.CLINIC_HOURS_UTC
    assert ((hours >= opening) & (hours < closing)).all()
    assert len({t.hour for t in times}) >= 10
    assert (hours < 12.5).mean() > 0.5                                   # busier morning
    assert max(np.unique(times, return_counts=True)[1]) <= 3             # no instant repeated
    assert pi.acquisition_time(pi.patient_id(5), RUN_DATE) == pi.acquisition_time(pi.patient_id(5), RUN_DATE)

