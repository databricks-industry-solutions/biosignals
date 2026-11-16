"""common/tracking.py: the SETUP job's MLflow experiment becomes active for any deploying identity,
including a service principal with no home folder, and also when 02_ and 03_ create it at once
(stand-ins for MLflow and the workspace: conftest.FakeTracking)."""
import types

import pytest
from conftest import SP_EXPERIMENT, FakeTracking, MlflowException

from common import tracking


@pytest.fixture
def fake(monkeypatch):
    fake = FakeTracking()
    fake.install(monkeypatch, types.ModuleType("mlflow"))
    return fake


def test_service_principal_without_a_home_folder(fake):
    """The bundle default for a service principal: its /Users/<application id> folder does not exist,
    so MLflow alone cannot create the experiment; use_experiment creates the folder first."""
    home = SP_EXPERIMENT.rpartition("/")[0]
    assert home not in fake.folders
    with pytest.raises(MlflowException, match="Parent directory"):     # what set_experiment alone does
        FakeTracking().set_experiment(SP_EXPERIMENT)
    experiment_id = tracking.use_experiment(fake, SP_EXPERIMENT)
    assert home in fake.folders and fake.experiments == {SP_EXPERIMENT: experiment_id}
    assert fake.active == SP_EXPERIMENT
    # The next run (a redeploy, the other training task, the gate) reuses it: same id, nothing new.
    assert tracking.use_experiment(fake, f" {SP_EXPERIMENT} ") == experiment_id and len(fake.experiments) == 1


def test_override_in_a_new_shared_folder(fake):
    path = "/Shared/cardiology/ecg/biosignals-ecg12-afib"
    tracking.use_experiment(fake, path)
    assert {"/Shared/cardiology", "/Shared/cardiology/ecg"} <= fake.folders and fake.active == path


def test_parallel_creation_still_activates_the_experiment(fake):
    """02_ and 03_ start together: the other task creates the experiment between this task's lookup
    and its create. MLflow's set_experiment would then return without activating it; use_experiment
    still makes it active."""
    fake.mkdirs(SP_EXPERIMENT.rpartition("/")[0])
    fake.created_meanwhile.add(SP_EXPERIMENT)
    bare = FakeTracking()
    bare.folders, bare.created_meanwhile = set(fake.folders), {SP_EXPERIMENT}
    bare.set_experiment(SP_EXPERIMENT)
    assert bare.active is None                                         # the MLflow behaviour modelled
    tracking.use_experiment(fake, SP_EXPERIMENT)
    assert fake.active == SP_EXPERIMENT and len(fake.experiments) == 1


def test_errors_are_not_swallowed(fake):
    def refuse(path):
        raise PermissionError(f"cannot create {path}")

    fake.workspace.mkdirs = refuse                                    # e.g. no permission on the folder
    with pytest.raises(PermissionError):
        tracking.use_experiment(fake, SP_EXPERIMENT)
    assert fake.active is None and not fake.experiments
    for bad in ("", "/biosignals-ecg12-afib", "/Users/${workspace.current_user.userName}/biosignals-ecg12-afib"):
        with pytest.raises(ValueError, match="mlflow_experiment"):
            tracking.use_experiment(fake, bad)


def test_other_mlflow_errors_from_the_create_are_raised(monkeypatch):
    """Only RESOURCE_ALREADY_EXISTS (the parallel-create race) is tolerated; any other MLflow error
    from creating the experiment is raised right there, before set_experiment is reached (which would
    otherwise retry the create and could leave no experiment active)."""
    fake = FakeTracking()
    set_calls = []

    def refuse_create(name):
        raise MlflowException(f"no permission to create {name}", "PERMISSION_DENIED")

    fake.create_experiment = refuse_create                 # installed below as mlflow.create_experiment
    fake.set_experiment = lambda experiment_name: set_calls.append(experiment_name)
    fake.install(monkeypatch, types.ModuleType("mlflow"))
    with pytest.raises(MlflowException, match="no permission") as refused:
        tracking.use_experiment(fake, SP_EXPERIMENT)
    assert refused.value.error_code == "PERMISSION_DENIED"
    assert set_calls == [] and fake.active is None and not fake.experiments
