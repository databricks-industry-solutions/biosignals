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


class AppConfigOut(BaseModel):
    """Names the UI needs for deep links, from common/config.py (single source of truth)."""
    workspace_host: str
    workspace_id: str
    catalog: str
    uc_schema: str
    arr_endpoint: str
    afib_endpoint: str
    arr_model: str
    afib_model: str
    inference_table: str
    dashboard_id: str | None   # None until the 12-lead dashboard exists; the UI then shows no embed
    n_leads: int
    champion_alias: str        # the model alias the endpoints serve (lowercase, as Unity Catalog stores it)
