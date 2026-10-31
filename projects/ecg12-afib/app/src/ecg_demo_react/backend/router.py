import datetime as dt

import numpy as np
from databricks.sdk.service.iam import User as UserOut
from fastapi import HTTPException

from .core import Dependencies, create_router
from .models import (
    VersionOut,
    GenerateRequest, GenerateResponse,
    AnalyzeRequest, AnalyzeResponse,
    AppConfigOut,
)
from .common import config, patient_info
from .common.ecg_signal_gen import AFIB, LABEL_NAMES, NSR, N_SAMPLES, OTHER, SAMPLING_RATE, synthesize_ecgs

router = create_router()

# UI rhythm choice -> generator class label.
RHYTHM_LABELS = {
    "NSR (normal)": NSR,
    "AFib": AFIB,
    "Other (tachycardia)": OTHER,
}


@router.get("/version", response_model=VersionOut, operation_id="version")
async def version():
    return VersionOut.from_metadata()


@router.get("/current-user", response_model=UserOut, operation_id="currentUser")
def me(user_ws: Dependencies.UserClient):
    return user_ws.current_user.me()


@router.get("/app-config", response_model=AppConfigOut, operation_id="appConfig")
def app_config(settings: Dependencies.Config, ws: Dependencies.Client):
    """Deep-link names for the UI: the workspace from the app's own credentials (the Apps runtime
    sets DATABRICKS_HOST / DATABRICKS_WORKSPACE_ID), the Unity Catalog location and dashboard from
    the app's environment (AppConfig), the model and endpoint names from common/config.py."""
    return AppConfigOut(
        workspace_host=(ws.config.host or "").rstrip("/"),
        workspace_id=str(ws.get_workspace_id()),
        catalog=settings.catalog,
        uc_schema=settings.uc_schema,
        arr_endpoint=config.ARR_ENDPOINT,
        afib_endpoint=config.AFIB_ENDPOINT,
        arr_model=config.ARR_MODEL_NAME,
        afib_model=config.AFIB_MODEL_NAME,
        inference_table=config.INFERENCE_TABLE.lower(),
        dashboard_id=settings.dashboard_id or None,
        n_leads=config.N_LEADS,
        champion_alias=config.CHAMPION_ALIAS,
    )


@router.post("/generate", response_model=GenerateResponse, operation_id="generate")
def generate(req: GenerateRequest):
    """One synthetic 12-lead ECG for the chosen rhythm + a synthetic patient record."""
    rng = np.random.default_rng()
    if req.rhythm == "Random":
        label = int(rng.choice(list(LABEL_NAMES)))
    elif req.rhythm in RHYTHM_LABELS:
        label = RHYTHM_LABELS[req.rhythm]
    else:
        raise HTTPException(status_code=422, detail=f"unknown rhythm {req.rhythm!r}")
    batch = synthesize_ecgs(np.array([label]), rng, noise_level=1.0, n_leads=config.N_LEADS)
    pseudo_id = patient_info.assign_patients(batch.labels, rng)[0]
    record = patient_info.patient_record(
        pseudo_id,
        heart_rate_bpm=int(batch.heart_rate_bpm[0]),
        acquisition_datetime_utc=dt.datetime.now(dt.timezone.utc).replace(microsecond=0),
        ecg=batch.ecg[0],
    )
    return GenerateResponse(
        ecg=batch.ecg[0].astype(float).tolist(),
        lead_names=list(batch.lead_names),
        true_label=label,
        label_name=LABEL_NAMES[label],
        sampling_rate=SAMPLING_RATE,
        n_samples=N_SAMPLES,
        patient=record,
    )


@router.post("/analyze", response_model=AnalyzeResponse, operation_id="analyze")
def analyze(req: AnalyzeRequest, ws: Dependencies.Client):
    """Triage (arr) then, if flagged, classify (afib) via the live serving endpoints.

    All N_LEADS leads go to both models, as a (1, N_LEADS, N_SAMPLES) tensor."""
    ecg = np.asarray(req.ecg, dtype=np.float32)
    if ecg.shape != (config.N_LEADS, N_SAMPLES):
        raise HTTPException(
            status_code=422,
            detail=f"ecg must be {config.N_LEADS} leads x {N_SAMPLES} samples, got {list(ecg.shape)}",
        )
    inputs = ecg[None, :, :].tolist()
    arr = (ws.serving_endpoints.query(name=config.ARR_ENDPOINT, inputs=inputs).predictions or [])[0]
    arr_flag = float(arr["arr_flag"])
    flagged = arr_flag == 1.0
    out = AnalyzeResponse(
        arr_flag=arr_flag, arr_prob=float(arr["arr_prob"]), flagged=flagged,
        arr_model=config.ARR_ENDPOINT,
    )
    if flagged:
        # The AFib model takes the same ecg tensor and returns [p_not_afib, p_afib].
        af = (ws.serving_endpoints.query(name=config.AFIB_ENDPOINT, inputs=inputs).predictions or [])[0]
        p_afib = float(af[1])
        out.p_afib = p_afib
        out.afib_label = "AFib" if p_afib > config.AFIB_THRESHOLD else "not AFib"
        out.afib_model = config.AFIB_ENDPOINT
    return out
