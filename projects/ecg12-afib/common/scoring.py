"""Scoring helpers shared by the validation gate (04_validate_and_promote) and the daily
batch scoring (07_arr_triage, 08_afib_classify).

* ``resolve_alias`` turns ``<model>@<alias>`` (``champion`` by default, ``challenger`` in the
  gate) into a concrete version. The notebooks load that exact version and record it, so the
  recorded version is always the one that scored.
* ``score_driver_side`` (the gate, 04_, and the daily batch, 07_ / 08_) predicts on the driver
  and joins the predictions back by id. The batches are small (200 hold-out ECGs, ~100 a day).
  It returns the input columns (id first), the model's result columns (DOUBLE), then the
  ``*_model_info`` column.
* ``arr_columns`` / ``afib_columns`` read each model's output schema: ARR ->
  DataFrame[arr_prob, arr_flag]; AFib -> (n, 2) [p_not_afib, p_afib]. The result column names
  are ``ARR_OUTPUT_COLS`` / ``AFIB_OUTPUT_COLS``.

Only ``score_driver_side`` touches Spark, and imports it lazily.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import CHAMPION_ALIAS, ID_COL

ARR_OUTPUT_COLS = ("arr_prob", "arr_flag")
AFIB_OUTPUT_COLS = ("afib_predictions_0", "afib_predictions_1")


@dataclass(frozen=True)
class ResolvedModel:
    """A UC model version pinned from an alias."""
    full_name: str
    alias: str
    version: str

    @property
    def uri(self) -> str:
        """Version-pinned URI to load, e.g. ``models:/cat.sch.bio_ecg12_arr/3``."""
        return f"models:/{self.full_name}/{self.version}"

    @property
    def info(self) -> str:
        """Value stored in the ``*_model_info`` column: the version that scored + its alias."""
        return f"{self.uri} (@{self.alias})"


def resolve_alias(client: Any, full_name: str, alias: str = CHAMPION_ALIAS) -> ResolvedModel:
    """Resolve ``full_name@alias`` to a version with an ``mlflow.MlflowClient``."""
    version = client.get_model_version_by_alias(full_name, alias).version
    return ResolvedModel(full_name=full_name, alias=alias, version=str(version))


def ecg_tensor(rows: Sequence[Sequence[Any]], n_samples: int) -> np.ndarray:
    """(n, n_leads, n_samples) float32 from rows of per-lead arrays (ecg1..ecgN values)."""
    if len(rows) == 0:
        raise ValueError("no ECG rows to score")
    tensor = np.asarray([[np.asarray(lead, dtype=np.float32) for lead in row] for row in rows], dtype=np.float32)
    if tensor.ndim != 3 or tensor.shape[2] != n_samples:
        raise ValueError(f"ECG rows have shape {tensor.shape}; expected (n, n_leads, {n_samples})")
    return tensor


def arr_columns(pred: Any) -> dict[str, np.ndarray]:
    """ARR model output (DataFrame with arr_prob, arr_flag) -> result columns."""
    missing = set(ARR_OUTPUT_COLS) - set(getattr(pred, "columns", []))
    if missing:
        raise ValueError(f"ARR model output lacks columns {sorted(missing)}")
    return {name: np.asarray(pred[name], dtype="float64") for name in ARR_OUTPUT_COLS}


def afib_columns(pred: Any) -> dict[str, np.ndarray]:
    """AFib model output ((n, 2) [p_not_afib, p_afib]) -> result columns."""
    probs = np.asarray(pred, dtype="float64")
    if probs.ndim != 2 or probs.shape[1] != 2:
        raise ValueError(f"AFib model output has shape {probs.shape}; expected (n, 2)")
    return dict(zip(AFIB_OUTPUT_COLS, (probs[:, 0], probs[:, 1])))


def score_driver_side(
    spark: Any,
    sdf: Any,
    predict: Callable[[np.ndarray], Mapping[str, np.ndarray]],
    *,
    output_cols: Sequence[str],
    ecg_cols: Sequence[str],
    n_samples: int,
    model_info_col: str,
    model_info: str,
    id_col: str = ID_COL,
) -> Any:
    """``sdf`` joined with the model's ``output_cols`` (DOUBLE) + ``model_info_col``.

    ``predict`` maps the (n, n_leads, n_samples) tensor to ``output_cols`` (use
    ``arr_columns`` / ``afib_columns`` on the model output). An empty ``sdf`` gives an empty
    result with the same columns, without calling the model."""
    import pandas as pd
    import pyspark.sql.functions as F
    import pyspark.sql.types as T

    pdf = sdf.select(id_col, *ecg_cols).toPandas()
    if pdf.empty:
        columns = {name: np.empty(0, dtype="float64") for name in output_cols}
    else:
        columns = dict(predict(ecg_tensor(pdf[list(ecg_cols)].to_numpy().tolist(), n_samples)))
        if list(columns) != list(output_cols):
            raise ValueError(f"predict returned columns {list(columns)}; expected {list(output_cols)}")
    schema = T.StructType(
        [T.StructField(id_col, T.StringType(), True)]
        + [T.StructField(name, T.DoubleType(), True) for name in output_cols]
    )
    pred_pdf = pd.DataFrame({id_col: pdf[id_col].to_numpy(), **columns})
    pred_sdf = spark.createDataFrame(pred_pdf, schema=schema).withColumn(model_info_col, F.lit(model_info))
    return sdf.join(pred_sdf, on=id_col, how="inner")
