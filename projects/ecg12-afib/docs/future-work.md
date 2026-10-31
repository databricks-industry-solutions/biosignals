# Future work

[← ecg12-afib README](../README.md)

## Future work

**Bulk inference.** The DAILY job scores one day's ~100 ECGs on the driver. Larger backlogs need
the scoring spread over workers, for example with Spark (`mlflow.pyfunc.spark_udf`) or Ray; both
are open to explore. A Spark starting point exists: `score_spark_udf` in `common/scoring.py`,
tested locally only and used by no job. Its one try on standard serverless, where the workers need
the model's own environment (`env_manager="uv"`) with PyTorch's CPU wheels, failed on uv's index
strategy (uv took `setuptools` from the CPU torch index, which lacks the pinned version); that
looks fixable, but the path is untested on Databricks.

**Other ECG classifiers.** Models that classify other ECG patterns (other rhythms, conduction
abnormalities or other waveform signatures) belong in their own projects under `projects/`, each
with its own bundle, README and tests, rather than in this one. This project keeps to the
two-model triage → AFib cascade.
