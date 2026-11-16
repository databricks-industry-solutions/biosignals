from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator

from .common.config import N_LEADS
from .common.ecg_signal_gen import N_SAMPLES
from .common.patient_info import PatientRecord

# One ECG value: a finite number a float32 can hold (the models take a float32 tensor, so a larger
# value would become infinite there).
_FLOAT32_MAX = float(np.finfo(np.float32).max)
EcgValue = Annotated[float, Field(allow_inf_nan=False, ge=-_FLOAT32_MAX, le=_FLOAT32_MAX)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]   # NaN fails both bounds


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
    """The ECG to score: N_LEADS x N_SAMPLES finite numbers (the models' input contract). Any other
    shape, a missing value or NaN / infinity is a 422."""
    ecg: list[list[EcgValue]]

    @field_validator("ecg")
    @classmethod
    def _is_n_leads_by_n_samples(cls, ecg: list[list[float]]) -> list[list[float]]:
        lengths = sorted({len(lead) for lead in ecg})
        if len(ecg) != N_LEADS or lengths != [N_SAMPLES]:
            raise ValueError(f"ecg must be {N_LEADS} leads x {N_SAMPLES} samples, got {len(ecg)} leads "
                             f"of {lengths} samples")
        return ecg


class ArrPrediction(BaseModel):
    """One prediction of the triage endpoint (workflow/02_train_arr_model.py): the abnormal-rhythm
    probability and the flag the model derives from it (1.0 = flagged). Anything else is a 502."""
    arr_prob: Probability
    arr_flag: float

    @field_validator("arr_flag")
    @classmethod
    def _is_zero_or_one(cls, flag: float) -> float:
        if flag not in (0.0, 1.0):
            raise ValueError(f"arr_flag must be 0.0 or 1.0, got {flag}")
        return flag


# One prediction of the classify endpoint (workflow/03_train_afib_model.py): [p_not_afib, p_afib].
AfibPrediction = tuple[Probability, Probability]


# Sent with every result and shown next to its recommended action (routes/index.tsx), worded as the
# root README's disclaimer.
NOT_FOR_CLINICAL_USE = ("Not for clinical use: a demonstration on synthetic data, not a medical device. "
                        "Its results and recommended actions are not a diagnosis or medical advice.")


class AnalyzeResponse(BaseModel):
    arr_flag: float            # 1.0 = flagged abnormal (triage)
    arr_prob: float
    flagged: bool
    p_afib: float | None = None        # only set when flagged (afib model run)
    afib_label: str | None = None      # "AFib" / "not AFib"
    arr_model: str             # endpoint name (served-model lineage)
    afib_model: str | None = None
    notice: str = NOT_FOR_CLINICAL_USE


class ServingError(BaseModel):
    """Why a live endpoint call failed: the ``detail`` of the app's 429 / 502 / 503 replies."""
    endpoint: str
    # waking: no answer within SERVING_TIMEOUT_S (or a 504), most likely scaling up from zero;
    # rate_limited: the endpoint answered 429; error: any other endpoint error, or an answer the
    # app cannot read.
    reason: Literal["waking", "rate_limited", "error"]
    message: str               # shown to the user as is; never the endpoint's own error text (logged)


class ServingErrorOut(BaseModel):
    detail: ServingError


class RateLimited(BaseModel):
    """Why the app refused a call before querying any endpoint: the ``detail`` of its own 429 replies
    (an endpoint's 429 has a ServingError instead)."""
    scope: Literal["caller", "total"]   # caller: this user's limit; total: all users' together
    limit: int                 # calls allowed in any window_s seconds
    window_s: int
    retry_after_s: int         # also the reply's Retry-After header
    message: str               # shown to the user as is


class RateLimitedOut(BaseModel):
    detail: RateLimited


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
