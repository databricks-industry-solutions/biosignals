"""The persisted training dataset ``config.TRAINING_TABLE``.

Written once per SETUP run by 01_generate_training_data; read by 02_/03_ (train + val splits)
and by the validation gate 04_ (hold-out split). One row per ECG:

  sample_id        ``<split>-<NNNNN>``, unique in the table
  split            train / val / holdout (``config.TRAINING_SPLITS``)
  rhythm_label     0 NSR, 1 AFib, 2 Other (sinus tachycardia); rhythm_name spells it out
  heart_rate_bpm   ventricular rate measured from the synthesized beats
  ecg1..ecgN       ARRAY<FLOAT> per lead, mV, lead order = ``ecg_signal_gen.lead_names_for``

The per-lead layout is the same as the daily batch tables', so ``scoring.ecg_tensor`` and
``scoring.score_driver_side`` read both. Every split is its own seeded draw from the shared
generator, so re-running 01_ rewrites identical rows.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np

from . import config
from .ecg_signal_gen import LABEL_NAMES, N_SAMPLES, EcgBatch, describe_lead, generate_ecg_batch
from .scoring import ecg_tensor
from .uc_io import read_table_version


class SplitArrays(NamedTuple):
    """One split, ready for torch: ``x`` (n, n_leads, n_samples) float32, ``labels`` (n,) int."""
    x: np.ndarray
    labels: np.ndarray
    sample_ids: list[str]


def generate_splits(
    splits: Mapping[str, tuple[int, int]],
    *,
    frac_afib: float,
    frac_other: float,
    noise_level: float,
    n_leads: int,
) -> dict[str, EcgBatch]:
    """One seeded generator draw per split (``splits``: split -> (ECGs, seed), e.g.
    ``config.TRAINING_SPLITS``), each with the same class mix."""
    return {name: generate_ecg_batch(n, frac_afib=frac_afib, frac_other=frac_other, seed=seed,
                                     noise_level=noise_level, n_leads=n_leads)
            for name, (n, seed) in splits.items()}


def to_pandas(batches: Mapping[str, EcgBatch]) -> Any:
    """The table's rows (pandas DataFrame, columns in ``spark_schema`` order)."""
    import pandas as pd

    frames = []
    for split, batch in batches.items():
        frame = pd.DataFrame({
            config.SAMPLE_ID_COL: [f"{split}-{i:05d}" for i in range(len(batch.labels))],
            config.SPLIT_COL: split,
            "rhythm_label": batch.labels.astype("int32"),
            "rhythm_name": [LABEL_NAMES[int(label)] for label in batch.labels],
            "heart_rate_bpm": batch.heart_rate_bpm.astype("int32"),
        })
        for j, col in enumerate(config.ecg_columns(len(batch.lead_names))):
            frame[col] = [lead.tolist() for lead in batch.ecg[:, j, :]]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def spark_schema(lead_names: tuple[str, ...]) -> Any:
    """Spark schema with Unity Catalog column comments (units, which leads are derived)."""
    import pyspark.sql.types as T

    split_names = "/".join(config.TRAINING_SPLITS)
    return T.StructType(
        [
            T.StructField(config.SAMPLE_ID_COL, T.StringType(), False,
                          {"comment": "Synthetic ECG id, <split>-<NNNNN>; not a patient"}),
            T.StructField(config.SPLIT_COL, T.StringType(), False,
                          {"comment": f"Dataset split: {split_names} (holdout is read only by the validation gate)"}),
            T.StructField("rhythm_label", T.IntegerType(), False,
                          {"comment": "Ground-truth rhythm: 0 NSR, 1 AFib, 2 Other (sinus tachycardia)"}),
            T.StructField("rhythm_name", T.StringType(), False, {"comment": "rhythm_label spelled out"}),
            T.StructField("heart_rate_bpm", T.IntegerType(), False,
                          {"comment": "Ventricular rate measured from the synthesized beats, beats/min"}),
        ]
        + [T.StructField(col, T.ArrayType(T.FloatType()), False, {"comment": describe_lead(name)})
           for col, name in zip(config.ecg_columns(len(lead_names)), lead_names)]
    )


def load_split(spark: Any, table: str, version: int, split: str, n_leads: int = config.N_LEADS) -> SplitArrays:
    """``split`` of ``table`` at Delta ``version``, ordered by sample id."""
    import pyspark.sql.functions as F

    ecg_cols = config.ecg_columns(n_leads)
    pdf = (read_table_version(spark, table, version)
           .filter(F.col(config.SPLIT_COL) == split)
           .select(config.SAMPLE_ID_COL, "rhythm_label", *ecg_cols)
           .orderBy(config.SAMPLE_ID_COL)
           .toPandas())
    if pdf.empty:
        raise ValueError(f"{table} version {version} has no rows with {config.SPLIT_COL} = {split!r}")
    x = ecg_tensor(pdf[ecg_cols].to_numpy().tolist(), N_SAMPLES)
    return SplitArrays(x=x, labels=pdf["rhythm_label"].to_numpy(dtype=int),
                       sample_ids=pdf[config.SAMPLE_ID_COL].tolist())
