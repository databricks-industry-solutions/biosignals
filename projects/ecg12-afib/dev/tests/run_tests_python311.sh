#!/usr/bin/env bash
# Runs the test suite the way it is verified: Python 3.11, in a throwaway uv environment with
# pinned versions (`uv run --isolated`: nothing is installed into a project or the system Python).
#
# Why Python 3.11: pyspark 3.5.3 pickles the Python functions it sends to its workers with its own
# bundled copy of cloudpickle (2.2.1), which predates cloudpickle's Python 3.12 support (3.0.0).
# On Python 3.14 the local Spark tests fail with a RecursionError, so 3.12+ is not supported for
# this suite.
# Versions: pyspark 3.5.3 (local Spark); numpy 1.26.4 as the app pins it; torch, cloudpickle, pandas
# and matplotlib as the AI v5 environment of the training and gate tasks ships them (the ML v5
# environment of the batch tasks has the same torch 2.9.0, CPU build).
#
# Usage, from anywhere: dev/tests/run_tests_python311.sh [pytest arguments; default: dev/tests -q]
# (pytest paths are relative to projects/ecg12-afib). See README.md in this folder.
# Needs uv and a Java runtime (for local Spark).
set -euo pipefail
cd "$(dirname "$0")/../.."         # projects/ecg12-afib
if [[ $# -eq 0 ]]; then set -- dev/tests -q; fi
exec uv run --no-project --isolated --python 3.11 \
  --with numpy==1.26.4 --with scipy==1.11.4 --with pandas==2.2.3 \
  --with neurokit2==0.2.12 --with PyWavelets==1.4.1 --with matplotlib==3.10.0 --with pyyaml==6.0.3 \
  --with pytest==9.1.1 --with cloudpickle==3.0.0 --with torch==2.9.0 \
  --with pyspark==3.5.3 \
  python -B -m pytest -p no:cacheprovider "$@"
