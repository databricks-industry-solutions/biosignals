"""Delta reads and writes shared by the notebooks. Every write is safe to re-run.

* ``overwrite_table``: tables rebuilt whole on every run (the training dataset
  ``training_data_12l`` and the per-date ``batch_data_12l_<date>``, ``arr_triaged_12l_<date>``)
  are replaced, schema included, so a re-run leaves exactly one copy.
* ``write_date_slice``: the accumulating ``dailyOrweekly_arr_afib_inference_12l`` table gets
  one date replaced atomically (Delta ``replaceWhere`` on ``batch_date``), so re-running a date
  does not duplicate rows. An existing table without ``batch_date`` (such as the live 8-lead
  demo's ``dailyOrweekly_arr_afib_inference``) is refused, never written to.
* ``table_version`` / ``read_table_version``: the training notebooks record the Delta version
  of the training dataset they read, and the validation gate reads the hold-out split at that
  same version, so a challenger is always judged on the data snapshot it was trained beside.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from .config import BATCH_DATE_COL


def overwrite_table(spark: Any, df: Any, table: str, optimize: bool = True) -> None:
    """Create or fully replace ``table`` (data and schema) with ``df``."""
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
    if optimize:
        spark.sql(f"OPTIMIZE {table}")


def write_date_slice(
    spark: Any, df: Any, table: str, run_date: dt.date, date_col: str = BATCH_DATE_COL
) -> str:
    """Write ``df`` (all rows dated ``run_date``) as that date's slice of ``table``.

    Returns how it wrote: ``"create"`` (new table) or ``"replace"`` (rows with
    ``date_col = run_date`` swapped in one transaction; the schema must already match, so a
    schema change fails loudly). Raises ``ValueError`` for an existing table without
    ``date_col``: it was not written by this function, so it is left untouched.
    """
    writer = df.write.format("delta")
    if not spark.catalog.tableExists(table):
        writer.mode("overwrite").saveAsTable(table)
        return "create"
    if date_col not in spark.table(table).columns:
        raise ValueError(f"{table} has no {date_col} column, so it is not a results table of this "
                         "demo; not writing to it")
    predicate = f"{date_col} = DATE'{run_date.isoformat()}'"
    writer.mode("overwrite").option("replaceWhere", predicate).saveAsTable(table)
    return "replace"


def table_version(spark: Any, table: str) -> int:
    """Current Delta version of ``table`` (its latest commit in ``DESCRIBE HISTORY``)."""
    rows = spark.sql(f"DESCRIBE HISTORY {table} LIMIT 1").collect()
    if not rows:
        raise ValueError(f"{table} has no Delta history")
    return int(rows[0]["version"])


def read_table_version(spark: Any, table: str, version: int) -> Any:
    """``table`` as of Delta ``version`` (time travel), as a Spark DataFrame."""
    return spark.sql(f"SELECT * FROM {table} VERSION AS OF {int(version)}")
