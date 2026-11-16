"""The SETUP job's one MLflow experiment (config.MLFLOW_EXPERIMENT_PARAM): made active before any run.

``mlflow.set_experiment`` creates a missing experiment, but not a missing parent folder: on Databricks
the create fails with "Parent directory does not exist". The default path is in the deploying
identity's home folder, and an identity can have none yet (a service principal that never had a
workspace folder, or an override such as ``/Shared/<team>/<name>``), so ``use_experiment`` creates the
parent folder first. It also creates the experiment itself rather than leaving that to
``set_experiment``: 02_ and 03_ start in parallel, and when both create it at once, the one that loses
the race gets the experiment back from ``set_experiment`` without it becoming active (MLflow returns
early on RESOURCE_ALREADY_EXISTS), so that notebook's run would land in its own notebook experiment.
"""

from __future__ import annotations

import posixpath
from typing import Any

from .config import resolve_mlflow_experiment


def use_experiment(workspace: Any, path: str | None) -> str:
    """Make ``path`` (the ``mlflow_experiment`` parameter, checked by ``resolve_mlflow_experiment``) the
    active MLflow experiment, creating its parent folder (``workspace``: a databricks-sdk
    ``WorkspaceClient``; mkdirs creates missing parents and is a no-op on an existing folder) and the
    experiment when missing. Returns the experiment id. A deleted experiment of that name is not
    revived: ``set_experiment`` refuses it."""
    import mlflow
    from mlflow.exceptions import MlflowException

    path = resolve_mlflow_experiment(path)
    workspace.workspace.mkdirs(posixpath.dirname(path))
    if mlflow.get_experiment_by_name(path) is None:
        try:
            mlflow.create_experiment(path)
        except MlflowException as error:
            if error.error_code != "RESOURCE_ALREADY_EXISTS":     # created meanwhile by the other task
                raise
    return mlflow.set_experiment(path).experiment_id
