# dev/: developer utilities and optional local tests (not deployed)

Scripts and tests that help develop and check the demo. They are not part of the app, the jobs or
the bundle: `databricks.yml` excludes `dev/**` from bundle sync, the app wheel is built from `app/`
only, and nothing in `workflow/`, `common/` or `app/` imports from here.

## `tests/`: optional pre-deployment checks

`pytest` tests (pytest is a Python test runner) that run on your laptop only (Python 3.11; nothing
runs them automatically and they are not deployed). They check code logic: the 12-lead formulas,
the patient record, training-data splits, the validation gate promoting both models or neither,
batch-scoring output, and that names come from config. They did **not** catch the two problems the
first live deploy hit (a workspace tag policy rejecting a tag value, and serverless Spark workers
without torch); those only showed up in live runs. The real pre-deployment check is a live run: the
SETUP job, then one DAILY run, before unpausing the schedule. How to run them, why Python 3.11, and
which tests are known to be out of date: [`tests/README.md`](tests/README.md).

## `signal_variety/analyze_signal_variety.py`

A read-only look at how varied the synthetic 12-lead ECGs of `common/ecg_signal_gen.py` are, and
how easily the three rhythm classes (NSR = normal sinus rhythm, AFib = atrial fibrillation,
Other = sinus tachycardia) can be told apart. It draws 40 synthetic patients per class, each with
its own seed and both with and without acquisition noise, plus the beat times alone of 20,000
patients per class, and reports:

1. **Rhythm.** Heart rate and the irregularity of the RR intervals (the time between consecutive
   beats: coefficient of variation and RMSSD, the root mean square of successive differences) per
   class, and how accurately one threshold on a single feature separates the classes. This is why
   the models' metrics sit near 1.0.
2. **Patient-to-patient variation within a class.** How alike the beat shapes of different
   patients are (correlation), wave amplitudes and the heart's electrical axis.
3. **Lead-to-lead variation within one patient.** The derived-lead formulas, the R-wave growth
   from V1 to V6, the polarity of aVR, and how many independent components the 12 leads carry.
4. **Noise and degenerate signals.** Signal-to-noise ratio, duplicate records, NaN values, flat leads.

Outputs go to `output/` next to the script (git-ignored) or to the folder given as the first
argument: `summary.json`, `per_record_stats.csv` and six PNG figures (lead II of 5 patients per
class, the beat shapes of 40 patients per class, all 12 leads of one typical patient per class,
and heart rate against RR irregularity). The figures are not committed (about 1.5 MB together);
regenerate them by running the script. The output is seeded: two runs give the same `summary.json`.

Run from `projects/ecg12-afib/` (needs numpy, scipy, matplotlib, neurokit2 and PyWavelets, all in
the test environment of [`tests/README.md`](tests/README.md); `-B` keeps bytecode out of `common/`):

```bash
python -B dev/signal_variety/analyze_signal_variety.py              # -> dev/signal_variety/output/
python -B dev/signal_variety/analyze_signal_variety.py /path/to/out   # or any folder
```
