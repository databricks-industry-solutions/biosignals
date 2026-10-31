"""Synthetic, non-PII patient + ECG-acquisition record for the ECG/AFib demo.

No real people, no names: identifiers are visibly synthetic (``SYNPT-000123``,
``SYN-MRN-0123456``). Used by the daily batch (06_generate_batch_data) and the app's live tab
(byte-identical copy in ``app/src/ecg_demo_react/backend/common/``).

Consistency rules:
  * demographics (mrn, age, sex, height, weight, bmi, blood pressure) are a pure function of
    ``pseudo_patient_id`` (seeded by a SHA-256 of the id), so a patient who appears on several
    days always has the same record;
  * ``bmi`` is computed from the stored height and weight;
  * ``heart_rate_bpm`` is the rate measured from the synthesized ECG, so it follows the
    rhythm (NSR 60-100, AFib 100-160, Other/sinus tachycardia 110-160 bpm);
  * ``sampling_rate_hz``, ``n_leads`` and ``duration_s`` come from the ECG array itself;
  * ``acquisition_datetime_utc`` (daily batch) falls in clinic hours of the batch date, spread
    over a morning and an afternoon peak, so a day's ECGs read like an incoming stream;
  * AFib is assigned preferentially to older patients (AFib prevalence rises with age), so
    the patient-context views have a real pattern to show.

Generated with numpy instead of Faker: nothing here needs Faker's locale data (no names or
addresses by design), and the app's lockfiles must stay byte-identical, so no new dependency.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, NamedTuple

import numpy as np

from .ecg_signal_gen import AFIB, NSR, OTHER, SAMPLING_RATE

POPULATION_SIZE = 5000          # synthetic patients the daily batches and the app draw from
ID_PREFIX = "SYNPT-"
MRN_PREFIX = "SYN-MRN-"
# When a day's ECGs are acquired (UTC hours): clinic opening/closing and the arrival peaks as
# (center, spread, weight), a busier morning and a smaller afternoon peak.
CLINIC_HOURS_UTC = (7.0, 19.0)
ARRIVAL_PEAKS = ((10.0, 1.5, 0.6), (14.5, 1.8, 0.4))


@dataclass(frozen=True)
class PatientRecord:
    """One synthetic patient + acquisition record (one row per ECG)."""
    pseudo_patient_id: str
    mrn: str
    age_years: int
    sex: str
    height_cm: float
    weight_kg: float
    bmi: float
    bp_systolic_mmhg: int
    bp_diastolic_mmhg: int
    acquisition_datetime_utc: dt.datetime
    heart_rate_bpm: int
    sampling_rate_hz: int
    n_leads: int
    duration_s: float

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


# Spark SQL type + column comment (with units) per PatientRecord field, in field order.
# 06_generate_batch_data builds its table schema from this.
FIELD_SPECS: dict[str, tuple[str, str]] = {
    "pseudo_patient_id": ("STRING", "Synthetic patient id (SYNPT-######); not a real person"),
    "mrn": ("STRING", "Synthetic medical record number (SYN-MRN-#######); not a real MRN"),
    "age_years": ("INT", "Age at acquisition, years (40-90)"),
    "sex": ("STRING", "Sex, F or M"),
    "height_cm": ("DOUBLE", "Height, cm"),
    "weight_kg": ("DOUBLE", "Weight, kg"),
    "bmi": ("DOUBLE", "Body-mass index, kg/m^2 = weight_kg / (height_cm / 100)^2"),
    "bp_systolic_mmhg": ("INT", "Systolic blood pressure, mmHg"),
    "bp_diastolic_mmhg": ("INT", "Diastolic blood pressure, mmHg"),
    "acquisition_datetime_utc": ("TIMESTAMP", "ECG acquisition time, UTC"),
    "heart_rate_bpm": ("INT", "Ventricular rate measured from the ECG, beats/min"),
    "sampling_rate_hz": ("INT", "ECG sampling rate, Hz"),
    "n_leads": ("INT", "Number of ECG leads stored (ecg1..ecgN)"),
    "duration_s": ("DOUBLE", "ECG duration, seconds"),
}

# Relative chance that a patient of a given age is picked for a rhythm class.
_AGE_WEIGHT = {
    AFIB: lambda age: np.exp((age - 65.0) / 12.0),
    OTHER: lambda age: np.exp((age - 65.0) / 30.0),
    NSR: lambda age: np.exp(-(age - 65.0) / 20.0),
}


class _Demographics(NamedTuple):
    mrn: str
    age_years: int
    sex: str
    height_cm: float
    weight_kg: float
    bmi: float
    bp_systolic_mmhg: int
    bp_diastolic_mmhg: int


def patient_id(k: int) -> str:
    """Synthetic id of patient ``k`` (0-based) of the population."""
    return f"{ID_PREFIX}{k:06d}"


def _rng_for(*parts: object) -> np.random.Generator:
    """Generator seeded by a SHA-256 of ``parts`` (stable across processes and machines)."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


@lru_cache(maxsize=None)
def demographics(pseudo_patient_id: str) -> _Demographics:
    """Demographics for one synthetic patient; always the same for the same id."""
    rng = _rng_for("demographics", pseudo_patient_id)
    age = int(round(np.clip(rng.normal(64.0, 12.0), 40, 90)))
    sex = "M" if rng.random() < 0.52 else "F"
    height_mean, height_sd = (176.0, 7.0) if sex == "M" else (162.0, 6.5)
    height_loss = 0.1 * max(age - 50, 0)                       # cm lost with age after 50
    height_cm = round(float(np.clip(rng.normal(height_mean, height_sd) - height_loss, 145, 200)), 1)
    target_bmi = float(np.clip(rng.lognormal(np.log(27.5), 0.17), 17, 50))
    weight_kg = round(target_bmi * (height_cm / 100) ** 2, 1)
    bmi = round(weight_kg / (height_cm / 100) ** 2, 1)
    systolic = int(round(np.clip(112 + 0.55 * (age - 40) + 0.4 * (bmi - 25) + rng.normal(0, 11), 90, 200)))
    diastolic = int(round(np.clip(0.5 * systolic + 18 + rng.normal(0, 6), 50, min(115, systolic - 25))))
    mrn = f"{MRN_PREFIX}{int(rng.integers(0, 10**7)):07d}"
    return _Demographics(mrn, age, sex, height_cm, weight_kg, bmi, systolic, diastolic)


@lru_cache(maxsize=4)
def _population(size: int) -> tuple[tuple[str, ...], np.ndarray]:
    ids = tuple(patient_id(k) for k in range(size))
    return ids, np.array([demographics(pid).age_years for pid in ids], dtype=float)


def assign_patients(
    labels: np.ndarray, rng: np.random.Generator, population_size: int = POPULATION_SIZE
) -> list[str]:
    """A distinct synthetic patient per ECG, AFib rows skewed toward older patients."""
    labels = np.asarray(labels, dtype=int)
    if labels.shape[0] > population_size:
        raise ValueError(f"{labels.shape[0]} ECGs > population of {population_size} patients")
    ids, ages = _population(population_size)
    available = np.ones(population_size, dtype=bool)
    chosen_ids = [""] * labels.shape[0]
    for label in (AFIB, OTHER, NSR):                   # most age-dependent class first
        rows = np.flatnonzero(labels == label)
        if rows.size == 0:
            continue
        weight = _AGE_WEIGHT[label](ages) * available
        picks = rng.choice(population_size, size=rows.size, replace=False, p=weight / weight.sum())
        available[picks] = False
        for row, k in zip(rows, picks):
            chosen_ids[row] = ids[k]
    return chosen_ids


def acquisition_time(pseudo_patient_id: str, run_date: dt.date) -> dt.datetime:
    """Deterministic acquisition time (UTC) on ``run_date``, inside clinic hours.

    Drawn from a morning and an afternoon peak (``ARRIVAL_PEAKS``), redrawn when it falls
    outside ``CLINIC_HOURS_UTC``, so a day's ECGs arrive spread over the day like an incoming
    stream, with no pile-up at opening or closing time."""
    rng = _rng_for("acquisition", pseudo_patient_id, run_date.isoformat())
    centers, spreads, weights = (np.array(v, dtype=float) for v in zip(*ARRIVAL_PEAKS))
    opening, closing = CLINIC_HOURS_UTC
    while True:
        peak = int(rng.choice(len(centers), p=weights / weights.sum()))
        hour = rng.normal(centers[peak], spreads[peak])
        if opening <= hour < closing:
            break
    seconds = int(hour * 3600)
    return dt.datetime.combine(run_date, dt.time(), tzinfo=dt.timezone.utc) + dt.timedelta(seconds=seconds)


def patient_record(
    pseudo_patient_id: str,
    heart_rate_bpm: int,
    acquisition_datetime_utc: dt.datetime,
    ecg: np.ndarray,
) -> PatientRecord:
    """Record for one ECG. ``ecg`` is that ECG's (n_leads, n_samples) array; its shape and
    the generator's sampling rate give n_leads and duration_s."""
    n_leads, n_samples = np.asarray(ecg).shape
    return PatientRecord(
        pseudo_patient_id=pseudo_patient_id,
        **demographics(pseudo_patient_id)._asdict(),
        acquisition_datetime_utc=acquisition_datetime_utc,
        heart_rate_bpm=int(heart_rate_bpm),
        sampling_rate_hz=SAMPLING_RATE,
        n_leads=int(n_leads),
        duration_s=n_samples / SAMPLING_RATE,
    )
