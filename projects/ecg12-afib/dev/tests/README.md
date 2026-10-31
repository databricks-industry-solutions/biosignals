# dev/tests/: optional pre-deployment checks (local only, not deployed)

**Optional pre-deployment checks.** These `pytest` tests (pytest is a Python test runner) run on
your laptop only (Python 3.11; nothing runs them automatically and they are not deployed: bundle
sync excludes `dev/**`). They check code logic: the 12-lead formulas, the patient record,
training-data splits, the validation gate promoting both models or neither, batch-scoring output,
and that names come from config.

They did **not** catch the two problems the first live deploy hit: a workspace tag policy
rejecting a tag value, and serverless Spark workers without torch (PyTorch). Those only showed up
in live runs. (A local copy of the tag policies, `GOVERNED_TAG_VALUES` in `conftest.py`, was added
afterwards; it only knows the policies copied into it.) **The real pre-deployment check is a live
run:** the SETUP job, then one DAILY run, before unpausing the DAILY schedule (steps 1 and 2 under
"Live steps still to do" in [`../../README.md`](../../README.md)).

**Known out of date** (found by reading the code, not by running the suite): the DAILY notebooks
`07_` / `08_` now score on the driver (`scoring.score_driver_side`), and `config.DASHBOARD_ID` is
set. `test_notebooks.py` (`test_batch_scoring_is_distributed_with_spark_udf`) and
`test_daily_flow_local.py` still expect Spark-UDF scoring, and `test_scoring_and_io.py` still
expects `DASHBOARD_ID` to be `None`, so expect those to fail until they are updated.

## What each file checks

| File | Checks |
|---|---|
| `test_ecg_signal_gen.py` | The shared 12-lead generator (`common/ecg_signal_gen.py`), including the derived-lead formulas |
| `test_patient_info.py` | The synthetic, non-PII patient record (`common/patient_info.py`) |
| `test_training.py` | Training data, training loop and layer table, sample-signal plots (`common/training_data.py`, `training.py`, `ecg_plots.py`) |
| `test_evaluation.py` | Metrics, the gate's threshold check and the evaluation tables, curves and figures (`common/evaluation.py`, `eval_plots.py`) |
| `test_scoring_and_io.py` | `common/config.py`, `scoring.py` and `uc_io.py` against small fakes, plus the batch path against the real `mlflow.pyfunc.spark_udf` |
| `test_notebooks.py` | The workflow notebooks' code, read but not run, plus a pickle-and-load check of the training notebooks' inline model code |
| `test_setup_flow_local.py` | A local run of the SETUP notebooks `01_` → `02_` + `03_` → `04_` with stand-ins for Unity Catalog and MLflow |
| `test_daily_flow_local.py` | A local run of the DAILY notebooks `06_` → `07_` → `08_`, twice for one date, with the same kind of stand-ins |
| `test_repo_consistency.py` | Single source of truth: the app's copy of `common/`, bundle defaults and sync rules, job task graphs and compute, schema and model / endpoint names only in config, the dashboard builder's two sections |

## Running them

**Use Python 3.11.** The suite is verified on Python 3.11 only. pyspark 3.5.3 pickles Spark UDFs
with its own bundled copy of cloudpickle (2.2.1), which predates cloudpickle's support for Python
3.12 (added in cloudpickle 3.0.0); on Python 3.14 the local Spark tests fail with a
`RecursionError`. Python 3.12+ is not supported for this suite. `run_tests_python311.sh` encodes
the pin: it runs pytest in a throwaway `uv` environment (`uv run --isolated`, nothing installed into
a project or the system Python) on Python 3.11 with every dependency pinned, and passes its
arguments to pytest (default `dev/tests -q`; paths are relative to `projects/ecg12-afib`). It needs
`uv` and a Java runtime.

```bash
cd projects/ecg12-afib
dev/tests/run_tests_python311.sh            # = the command below
dev/tests/run_tests_python311.sh dev/tests/test_scoring_and_io.py -v
```

```bash
uv run --no-project --isolated --python 3.11 \
  --with numpy==1.26.4 --with scipy==1.11.4 --with pandas==2.2.3 \
  --with neurokit2==0.2.12 --with PyWavelets==1.4.1 --with matplotlib==3.10.0 --with pyyaml==6.0.3 \
  --with pytest==9.1.1 --with cloudpickle==3.0.0 --with torch==2.9.0 \
  --with pyspark==3.5.3 --with pyarrow==21.0.0 --with mlflow-skinny==3.8.1 \
  python -B -m pytest -p no:cacheprovider dev/tests -q
```

Without the script, `python -m pytest dev/tests` (from `projects/ecg12-afib`; `conftest.py` adds
that folder to `sys.path`) needs numpy, scipy, pandas, neurokit2 and pyyaml, and skips the tests
whose extras are missing. `torch` + `cloudpickle`: the training loop and layer table, and a check
that pickles each training notebook's inline model the way MLflow does and loads it in a process
where `common` cannot be imported, as on a serving endpoint. A local (non-Databricks-Connect)
`pyspark` + Java (+ `pyarrow` for the pandas UDFs): driver-side and Spark-UDF scoring (same table
from both), reading a training split, a run of the SETUP notebooks `01_` → `02_` + `03_` → `04_`
(tiny splits, two epochs on CPU; the gate passes and promotes both models and logs its evaluation
artifacts, then fails and promotes neither; the stand-in registry refuses tag values the
workspace's tag policies do not allow, as Unity Catalog does) and a run of the DAILY notebooks
`06_` → `07_` → `08_` twice for one date (written for Spark-UDF scoring on the local Spark
workers; see "Known out of date" above), both with stand-ins for Unity Catalog and MLflow.
`mlflow-skinny`: the batch path against the real `mlflow.pyfunc.spark_udf` on small stand-in
models with the notebooks' tensor signature. No notebook is executed on Databricks by the tests,
and `05_` (endpoints) is not run.
