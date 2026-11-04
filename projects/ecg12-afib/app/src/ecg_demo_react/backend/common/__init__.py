"""Shared code for the 12-lead ECG/AFib demo (notebooks, dashboard builder, app backend).

Submodules are imported explicitly (``from common import config``) so that importing one does
not pull in another's dependencies (neurokit2, mlflow, pyspark).
"""
