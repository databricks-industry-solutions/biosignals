"""Scoring helpers shared by the validation gate (04_validate_and_promote) and the daily
batch scoring (07_arr_triage, 08_afib_classify).

* ``resolve_alias`` turns ``<model>@<alias>`` (``champion`` by default, ``challenger`` in the
  gate) into a concrete version. The notebooks load that exact version and record it, so the
  recorded version is always the one that scored.
* ``score_driver_side`` (the gate, 04_, and the daily batch, 07_ / 08_) predicts on the driver
  and joins the predictions back by id. The batches are small (200 hold-out ECGs, ~100 a day).
* ``score_spark_udf`` scores on the Spark workers with an ``mlflow.pyfunc.spark_udf`` of that
  version instead (each ECG's 12 per-lead columns as one flattened array, reshaped by spark_udf
  to the (-1, 12, 2500) signature; ``cap_arrow_batch_rows`` caps the ECGs per model call;
  ``torch_cpu_wheels`` points the workers' environment build at PyTorch's CPU wheels). No job
  uses it: it is tested locally only (docs/future-work.md).
* Both return the same columns in the same order and types: the input columns (id first), the
  model's result columns (DOUBLE), then the ``*_model_info`` column.
* ``arr_columns`` / ``afib_columns`` (driver side) and ``arr_udf_columns`` /
  ``afib_udf_columns`` (spark_udf) read each model's output schema: ARR ->
  DataFrame[arr_prob, arr_flag]; AFib -> (n, 2) [p_not_afib, p_afib]. The result column names
  are ``ARR_OUTPUT_COLS`` / ``AFIB_OUTPUT_COLS``.

Only the ``score_*`` functions, ``cap_arrow_batch_rows`` and the ``*_udf_columns`` readers touch
Spark, and import it lazily.
"""

from __future__ import annotations

import math
import os
import shutil
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import BATCH_UDF_ROWS_PER_PARTITION, CHAMPION_ALIAS, ID_COL

ARR_OUTPUT_COLS = ("arr_prob", "arr_flag")
AFIB_OUTPUT_COLS = ("afib_predictions_0", "afib_predictions_1")
# spark_udf result types (Spark DDL), explicit so both come back as DOUBLE like the driver-side
# path: ARR returns a struct of its two output columns, AFib the array [p_not_afib, p_afib].
ARR_UDF_RESULT_TYPE = "arr_prob double, arr_flag double"
AFIB_UDF_RESULT_TYPE = "array<double>"
# spark_udf env_manager: "uv" rebuilds the model's logged environment (python_env.yaml +
# requirements.txt: Python 3.12, mlflow, cloudpickle, torch, ...) with uv and runs the model in it
# on the workers ("local", the task's own environment, gives serverless UDF workers no torch).
UDF_ENV_MANAGER = "uv"
# PyTorch's CPU-only wheel index. The models log `torch==2.9.0`, which on Linux PyPI is the CUDA
# build (plus GBs of nvidia-* wheels); a Spark UDF gets no GPU. The CPU wheel `2.9.0+cpu` also
# satisfies `==2.9.0` (a local version label), so with this index ahead of PyPI uv installs it.
TORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
_UV_EXTRA_INDEX_ENV = "UV_EXTRA_INDEX_URL"   # read by the `uv pip install` that builds the environment
_PREDICTION_COL = "_prediction"     # scratch column holding the UDF result, dropped by the select
# Spark SQL setting: the most rows per Arrow batch a pandas UDF reads. mlflow's spark_udf calls the
# model once per Arrow batch, so this caps the ECGs per model call whatever the partition sizes.
ARROW_BATCH_ROWS_CONF = "spark.sql.execution.arrow.maxRecordsPerBatch"


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


def arr_udf_columns(prediction: Any) -> dict[str, Any]:
    """ARR spark_udf result (struct column ``ARR_UDF_RESULT_TYPE``) -> result columns."""
    return {name: prediction[name] for name in ARR_OUTPUT_COLS}


def afib_udf_columns(prediction: Any) -> dict[str, Any]:
    """AFib spark_udf result (array column [p_not_afib, p_afib]) -> result columns."""
    return {name: prediction[i] for i, name in enumerate(AFIB_OUTPUT_COLS)}


@contextmanager
def torch_cpu_wheels() -> Iterator[None]:
    """Around ``mlflow.pyfunc.spark_udf(..., env_manager=UDF_ENV_MANAGER)``: the model environment
    it builds with uv gets torch from ``TORCH_CPU_INDEX_URL`` (the CPU build) instead of PyPI's CUDA
    build. Restores the previous ``UV_EXTRA_INDEX_URL`` afterwards. Fails first, with a clear
    message, when no ``uv`` command is on PATH (spark_udf would fail deeper down without one)."""
    if UDF_ENV_MANAGER == "uv" and shutil.which("uv") is None:
        raise RuntimeError("spark_udf(env_manager='uv') builds the model environment with the uv command, "
                           "and there is none on PATH")
    previous = os.environ.get(_UV_EXTRA_INDEX_ENV)
    os.environ[_UV_EXTRA_INDEX_ENV] = TORCH_CPU_INDEX_URL if previous is None else f"{TORCH_CPU_INDEX_URL} {previous}"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_UV_EXTRA_INDEX_ENV, None)
        else:
            os.environ[_UV_EXTRA_INDEX_ENV] = previous


def cap_arrow_batch_rows(spark: Any, rows: int) -> bool:
    """Set ``ARROW_BATCH_ROWS_CONF`` to ``rows`` for the session; True when it took effect.

    False when the compute refuses the setting: serverless compute lets notebooks and jobs set
    only a few listed Spark properties, and this one is not listed
    (https://docs.databricks.com/aws/en/spark/conf), so it may raise or not apply there."""
    from pyspark.errors import PySparkException

    try:
        spark.conf.set(ARROW_BATCH_ROWS_CONF, str(rows))
        return spark.conf.get(ARROW_BATCH_ROWS_CONF) == str(rows)
    except PySparkException:
        return False


def score_spark_udf(
    sdf: Any,
    predict_udf: Callable[[Any], Any],
    udf_columns: Callable[[Any], Mapping[str, Any]],
    *,
    output_cols: Sequence[str],
    ecg_cols: Sequence[str],
    model_info_col: str,
    model_info: str,
    id_col: str = ID_COL,
    rows_per_partition: int = BATCH_UDF_ROWS_PER_PARTITION,
    n_rows: int | None = None,
) -> Any:
    """``sdf`` + the model's ``output_cols`` (DOUBLE) + ``model_info_col``, scored on the Spark
    workers by ``predict_udf`` (an ``mlflow.pyfunc.spark_udf``); same columns, order and types as
    ``score_driver_side``.

    ``predict_udf`` gets one flattened array per ECG (``ecg_cols`` concatenated in lead order);
    ``udf_columns`` maps its result column to ``output_cols`` (``arr_udf_columns`` /
    ``afib_udf_columns``). Two bounds keep each UDF call at most ``rows_per_partition`` ECGs:
    the rows are spread over partitions of at most that many ECGs (``n_rows``: the row count of
    ``sdf`` when the caller already has it; counted otherwise), and ``cap_arrow_batch_rows``
    caps the Arrow batches at that many rows, which holds even if Spark merges partitions. Where
    the compute refuses the cap (serverless), a warning says so and the partition bound applies."""
    import pyspark.sql.functions as F

    columns = dict(udf_columns(F.col(_PREDICTION_COL)))
    if list(columns) != list(output_cols):
        raise ValueError(f"udf_columns returned columns {list(columns)}; expected {list(output_cols)}")
    if not cap_arrow_batch_rows(sdf.sparkSession, rows_per_partition):
        warnings.warn(f"{ARROW_BATCH_ROWS_CONF} cannot be set on this compute: each UDF call is bounded "
                      f"by the partitions of at most {rows_per_partition} ECGs only", stacklevel=2)
    n_partitions = max(1, math.ceil((sdf.count() if n_rows is None else n_rows) / rows_per_partition))
    ecg = F.flatten(F.array(*[F.col(c) for c in ecg_cols]))
    scored = sdf.repartition(n_partitions).withColumn(_PREDICTION_COL, predict_udf(ecg))
    return scored.select(
        id_col,
        *[c for c in sdf.columns if c != id_col],
        *[column.cast("double").alias(name) for name, column in columns.items()],
        F.lit(model_info).alias(model_info_col),
    )
