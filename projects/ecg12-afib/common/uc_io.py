"""Delta reads and writes shared by the notebooks. Every write is safe to re-run.

* ``overwrite_table``: tables rebuilt whole on every run (the training dataset
  ``training_data_12l`` and the per-date ``batch_data_12l_<date>``, ``arr_triaged_12l_<date>``)
  are replaced, schema included, so a re-run leaves exactly one copy.
* ``write_date_slice``: the accumulating tables ``arr_triage_history_12l`` and
  ``dailyOrweekly_arr_afib_inference_12l`` get one date replaced atomically (Delta
  ``replaceWhere`` on ``batch_date``), so re-running a date does not duplicate rows. An existing
  table without ``batch_date`` was not written by this function and is refused, never written to.
* ``table_version`` / ``read_table_version``: the training notebooks record the Delta version
  of the training dataset they read, and the validation gate reads the hold-out split at that
  same version, so a challenger is always judged on the data snapshot it was trained beside.
* ``claim_schema`` / ``check_schema_owner``: one schema per deployment. The SETUP job's first task
  records this deployment as the owner of its schema in ``ecg12_afib_deployment`` (first run) and
  the first task of SETUP and DAILY stops, before writing anything, when another deployment owns
  it (the model and table names are fixed, so two deployments in one schema would overwrite each
  other; see ``config.DEPLOYMENT_TABLE``).

Every table name these functions put into SQL is checked first (``config.check_uc_name``).
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from .config import BATCH_DATE_COL, DEPLOYMENT_ID_COL, DEPLOYMENT_TABLE, check_uc_name


def overwrite_table(spark: Any, df: Any, table: str, optimize: bool = True) -> None:
    """Create or fully replace ``table`` (data and schema) with ``df``."""
    table = check_uc_name(table)
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)
    if optimize:
        spark.sql(f"OPTIMIZE {table}")


def create_table(spark: Any, df: Any, table: str) -> None:
    """Create ``table`` from ``df``; fails, writing nothing, when ``table`` already exists.

    The mode is spelled ``"error"``: Spark Connect (serverless) accepts only append, overwrite,
    error and ignore, and refuses its synonym ``"errorifexists"`` (UNSUPPORTED_OPERATION)."""
    df.write.format("delta").mode("error").saveAsTable(check_uc_name(table))


def write_date_slice(
    spark: Any, df: Any, table: str, run_date: dt.date, date_col: str = BATCH_DATE_COL
) -> str:
    """Write ``df`` (all rows dated ``run_date``) as that date's slice of ``table``.

    Returns how it wrote: ``"create"`` (new table) or ``"replace"`` (rows with
    ``date_col = run_date`` swapped in one transaction; the schema must already match, so a
    schema change fails loudly). Raises ``ValueError`` for an existing table without
    ``date_col``: it was not written by this function, so it is left untouched.
    """
    table = check_uc_name(table)
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
    rows = spark.sql(f"DESCRIBE HISTORY {check_uc_name(table)} LIMIT 1").collect()
    if not rows:
        raise ValueError(f"{table} has no Delta history")
    return int(rows[0]["version"])


def read_table_version(spark: Any, table: str, version: int) -> Any:
    """``table`` as of Delta ``version`` (time travel), as a Spark DataFrame."""
    return spark.sql(f"SELECT * FROM {check_uc_name(table)} VERSION AS OF {int(version)}")


class SchemaOwnershipError(RuntimeError):
    """The schema belongs to another deployment of this demo (or to none yet, for DAILY)."""


def _owners(spark: Any, table: str) -> list[str]:
    """The deployment ids recorded in the ownership table ``table`` (normally exactly one)."""
    df = spark.table(table)
    if DEPLOYMENT_ID_COL not in df.columns:
        raise SchemaOwnershipError(
            f"{table} has no {DEPLOYMENT_ID_COL} column, so it was not written by this demo: the schema "
            "holds something else. Set the bundle variable `schema` to a schema of its own and run again.")
    return sorted({row[0] for row in df.select(DEPLOYMENT_ID_COL).distinct().collect()})


def check_schema_owner(spark: Any, table: str, deployment_id: str) -> None:
    """Raise ``SchemaOwnershipError`` unless ``table`` (``<catalog>.<schema>.ecg12_afib_deployment``)
    records ``deployment_id``, and only it, as the owner of its schema. Reads, never writes."""
    table = check_uc_name(table)
    schema = table.rsplit(".", 1)[0]
    if not spark.catalog.tableExists(table):
        raise SchemaOwnershipError(
            f"{schema} has no {DEPLOYMENT_TABLE} table, so no deployment owns it yet: run this deployment's "
            "SETUP job first (its first task records the owner), then this job.")
    owners = _owners(spark, table)
    if owners == [deployment_id]:
        return
    if not owners:
        problem = f"{table} records no deployment (it is empty)"
    elif deployment_id in owners:
        problem = f"{table} records several deployments {owners}"
    else:
        problem = f"{schema} belongs to another deployment of this demo ({owners[0]!r})"
    raise SchemaOwnershipError(
        f"{problem}, not only this one ({deployment_id!r}). Every deployment uses the same fixed table and "
        "model names, so running here could overwrite another deployment's data and move its @champion; "
        "nothing was written. Set the bundle variable `schema` to a schema no other deployment uses (one "
        "schema per deployment) and run again. If the other deployment is gone, delete its objects first, "
        "this table last (docs/deploy.md, Teardown).")


def claim_schema(spark: Any, table: str, deployment_id: str) -> bool:
    """SETUP: record ``deployment_id`` as the owner of ``table``'s schema if no deployment owns it
    yet, then ``check_schema_owner``. Returns True when this call recorded it. The record is a
    one-row Delta table created only if it does not exist (``create_table``), so a second
    deployment cannot replace it; if two deployments create it at the same moment, the one whose
    create fails reads the other's row and stops."""
    table = check_uc_name(table)
    claimed = False
    if not spark.catalog.tableExists(table):
        import pyspark.sql.types as T

        schema = T.StructType([
            T.StructField(DEPLOYMENT_ID_COL, T.StringType(), False,
                          {"comment": "Bundle deployment that owns this schema: <workspace id>:<bundle root path>"}),
            T.StructField("claimed_at", T.TimestampType(), False,
                          {"comment": "When its SETUP job recorded it (UTC)"}),
        ])
        now = dt.datetime.now(dt.UTC)                   # timezone-aware: stored as this UTC instant
        try:
            create_table(spark, spark.createDataFrame([(deployment_id, now)], schema), table)
            claimed = True
        except Exception:
            if not spark.catalog.tableExists(table):
                raise                       # a real failure, not another deployment's record landing first
    check_schema_owner(spark, table, deployment_id)
    return claimed
