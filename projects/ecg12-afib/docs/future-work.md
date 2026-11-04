# Future work

[← ecg12-afib README](../README.md)

## Future work

**Bulk inference.** The DAILY job scores one day's ~100 ECGs on the driver; larger backlogs would
need the scoring spread over workers, for example with Spark or Ray.

**Other ECG classifiers.** Models that classify other ECG patterns (other rhythms, conduction
abnormalities or other waveform signatures) belong in their own projects under `projects/`, each
with its own bundle, README and tests, rather than in this one. This project keeps to the
two-model triage → AFib cascade.
