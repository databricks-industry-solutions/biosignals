from typing import Literal

from pydantic import BaseModel, Field
from .. import __version__
from .common.patient_info import PatientRecord


class VersionOut(BaseModel):
    version: str

    @classmethod
    def from_metadata(cls):
        return cls(version=__version__)


# ---- ECG/AFib demo models ---------------------------------------------------

class GenerateRequest(BaseModel):
    """Which rhythm to simulate. 'Random' draws a class uniformly server-side."""
    rhythm: str = Field(default="Random")


class GenerateResponse(BaseModel):
    ecg: list[list[float]]     # n_leads x n_samples, mV; all leads are displayed AND scored
    lead_names: list[str]      # clinical lead name per row of `ecg` (I, II, III, aVR, ...)
    true_label: int            # 0=NSR, 1=AFib, 2=Other
    label_name: str
    sampling_rate: int
    n_samples: int
    patient: PatientRecord     # synthetic, non-PII patient + acquisition record


class AnalyzeRequest(BaseModel):
    """The ECG to score: n_leads x n_samples (the models' input contract)."""
    ecg: list[list[float]]


class AnalyzeResponse(BaseModel):
    arr_flag: float            # 1.0 = flagged abnormal (triage)
    arr_prob: float
    flagged: bool
    p_afib: float | None = None        # only set when flagged (afib model run)
    afib_label: str | None = None      # "AFib" / "not AFib"
    arr_model: str             # endpoint name (served-model lineage)
    afib_model: str | None = None


class ServingError(BaseModel):
    """Why a live endpoint call failed: the ``detail`` of the app's 429 / 502 / 503 replies."""
    endpoint: str
    # waking: no answer within SERVING_TIMEOUT_S (or a 504), most likely scaling up from zero;
    # rate_limited: the endpoint answered 429; error: any other endpoint error.
    reason: Literal["waking", "rate_limited", "error"]
    message: str               # shown to the user as is


class ServingErrorOut(BaseModel):
    detail: ServingError


class EndpointStatus(BaseModel):
    """One endpoint's answer to a warm-up ping."""
    endpoint: str
    status: Literal["ready", "cold", "error"]   # cold = no answer yet: it is waking from scale-to-zero
    seconds: float             # how long the ping took
    message: str | None = None   # why it is cold / in error


class WarmupResponse(BaseModel):
    endpoints: list[EndpointStatus]


class AppConfigOut(BaseModel):
    """Names the UI needs for deep links (see router.app_config for where each comes from)."""
    workspace_host: str
    workspace_id: str
    catalog: str
    uc_schema: str
    arr_endpoint: str
    afib_endpoint: str
    arr_model: str
    afib_model: str
    inference_table: str
    dashboard_id: str | None   # None when no dashboard id is configured; the UI then shows no embed
    n_leads: int
    champion_alias: str        # the model alias the endpoints serve (lowercase, as Unity Catalog stores it)
    daily_job_id: str | None   # the DAILY job the header's Lakeflow link opens; None = the jobs list
    serving_timeout_s: int     # the backend's bound on one endpoint call (the UI's time limits build on it)
    warmup_interval_s: int     # keep-alive ping interval while the Live tab is visible
