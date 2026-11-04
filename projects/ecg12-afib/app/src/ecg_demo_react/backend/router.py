import datetime as dt
import functools
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Annotated, Any, AsyncGenerator

import numpy as np
from databricks.sdk import WorkspaceClient
from databricks.sdk.config import Config
from databricks.sdk.errors import DatabricksError, DeadlineExceeded, TooManyRequests
from fastapi import FastAPI, HTTPException, Request

from .core import Dependencies, create_router
from .core._base import LifespanDependency
from .models import (
    GenerateRequest, GenerateResponse,
    AnalyzeRequest, AnalyzeResponse,
    AppConfigOut,
    EndpointStatus, ServingError, ServingErrorOut, WarmupResponse,
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

# The replies an endpoint call can end in besides 200, each with a ServingError as `detail`.
ENDPOINT_ERRORS: dict[int | str, dict[str, Any]] = {
    429: {"model": ServingErrorOut, "description": "An endpoint is rate-limiting requests"},
    502: {"model": ServingErrorOut, "description": "An endpoint returned an error"},
    503: {"model": ServingErrorOut, "description": "An endpoint did not answer in time (waking from scale-to-zero)"},
}


class _ServingClientDependency(LifespanDependency):
    """WorkspaceClient for the live endpoint calls: the app's own identity (the same credentials
    `WorkspaceClient()` finds) with every call bounded by `config.SERVING_TIMEOUT_S`.

    Without that bound databricks-sdk 0.117 (app/uv.lock) gives each HTTP attempt 60 s and retries
    timeouts, connection errors and 503/429 replies for 300 s (databricks/sdk/_base_client.py,
    retries.py), so a call to a scaled-to-zero endpoint could hold a request for over 5 minutes.
    With both limits at SERVING_TIMEOUT_S, a cold endpoint makes the call give up about
    SERVING_TIMEOUT_S + 2 s after it starts (the held attempt times out, or 503 replies are retried
    until then, plus one back-off); at most ~2 x SERVING_TIMEOUT_S + 11 s, when a retry starts just
    before the retry window closes (longer only if an endpoint sends a Retry-After above 10 s)."""

    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncGenerator[None, None]:
        app.state.serving_client = WorkspaceClient(config=Config(
            http_timeout_seconds=config.SERVING_TIMEOUT_S,
            retry_timeout_seconds=config.SERVING_TIMEOUT_S,
        ))
        yield

    @staticmethod
    def __call__(request: Request) -> WorkspaceClient:
        return request.app.state.serving_client


ServingClient = Annotated[WorkspaceClient, _ServingClientDependency.depends()]


def _endpoint_error(status_code: int, error: ServingError) -> HTTPException:
    return HTTPException(status_code=status_code, detail=error.model_dump())


def _predict(serving: WorkspaceClient, endpoint: str, inputs: Any, client_request_id: str | None = None) -> Any:
    """First prediction of one endpoint call. Every failure becomes a 429 / 502 / 503 whose detail
    is a ServingError the UI shows as is, never a hang or a bare 500."""
    waking = ServingError(
        endpoint=endpoint, reason="waking",
        message=(f"The model endpoint {endpoint} did not answer within {config.SERVING_TIMEOUT_S} s. "
                 "It is most likely waking up from scale-to-zero: try again in a minute or two."),
    )
    try:
        res = serving.serving_endpoints.query(name=endpoint, inputs=inputs, client_request_id=client_request_id)
    except TimeoutError as err:
        # The SDK stopped retrying after SERVING_TIMEOUT_S; the last failed attempt is the cause.
        if isinstance(err.__cause__, TooManyRequests):
            raise _endpoint_error(429, ServingError(
                endpoint=endpoint, reason="rate_limited",
                message=f"The model endpoint {endpoint} is rate-limiting requests. Try again shortly.",
            )) from err
        raise _endpoint_error(503, waking) from err
    except DeadlineExceeded as err:   # 504: the serving gateway itself gave up waiting
        raise _endpoint_error(503, waking) from err
    except DatabricksError as err:
        raise _endpoint_error(502, ServingError(
            endpoint=endpoint, reason="error", message=f"The model endpoint {endpoint} returned an error: {err}",
        )) from err
    predictions = res.predictions or []
    if not predictions:
        raise _endpoint_error(502, ServingError(
            endpoint=endpoint, reason="error", message=f"The model endpoint {endpoint} returned no predictions.",
        ))
    return predictions[0]


@router.get("/app-config", response_model=AppConfigOut, operation_id="appConfig")
def app_config(settings: Dependencies.Config, ws: Dependencies.Client):
    """Deep-link names for the UI: the workspace from the app's own credentials (the Apps runtime
    sets DATABRICKS_HOST / DATABRICKS_WORKSPACE_ID), the Unity Catalog location and dashboard from
    the app's environment (AppConfig), the endpoint names from its endpoint prefix
    (AppConfig.endpoints) and the model names from common/config.py."""
    return AppConfigOut(
        workspace_host=(ws.config.host or "").rstrip("/"),
        workspace_id=str(ws.get_workspace_id()),
        catalog=settings.catalog,
        uc_schema=settings.uc_schema,
        arr_endpoint=settings.endpoints[0],
        afib_endpoint=settings.endpoints[1],
        arr_model=config.ARR_MODEL_NAME,
        afib_model=config.AFIB_MODEL_NAME,
        inference_table=config.INFERENCE_TABLE.lower(),
        dashboard_id=settings.dashboard_id or None,
        n_leads=config.N_LEADS,
        champion_alias=config.CHAMPION_ALIAS,
        daily_job_id=settings.daily_job_id or None,
        serving_timeout_s=config.SERVING_TIMEOUT_S,
        warmup_interval_s=config.WARMUP_INTERVAL_S,
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


@router.post("/analyze", response_model=AnalyzeResponse, operation_id="analyze", responses=ENDPOINT_ERRORS)
def analyze(req: AnalyzeRequest, serving: ServingClient, settings: Dependencies.Config):
    """Triage (arr) then, if flagged, classify (afib) via the live serving endpoints.

    All N_LEADS leads go to both models, as a (1, N_LEADS, N_SAMPLES) tensor. Each call is bounded
    by SERVING_TIMEOUT_S (see _ServingClientDependency), so this request makes at most two bounded
    calls."""
    arr_endpoint, afib_endpoint = settings.endpoints    # this deployment's names (endpoint_prefix)
    ecg = np.asarray(req.ecg, dtype=np.float32)
    if ecg.shape != (config.N_LEADS, N_SAMPLES):
        raise HTTPException(
            status_code=422,
            detail=f"ecg must be {config.N_LEADS} leads x {N_SAMPLES} samples, got {list(ecg.shape)}",
        )
    inputs = ecg[None, :, :].tolist()
    arr = _predict(serving, arr_endpoint, inputs)
    arr_flag = float(arr["arr_flag"])
    flagged = arr_flag == 1.0
    out = AnalyzeResponse(
        arr_flag=arr_flag, arr_prob=float(arr["arr_prob"]), flagged=flagged,
        arr_model=arr_endpoint,
    )
    if flagged:
        # The AFib model takes the same ecg tensor and returns [p_not_afib, p_afib].
        af = _predict(serving, afib_endpoint, inputs)
        p_afib = float(af[1])
        out.p_afib = p_afib
        out.afib_label = "AFib" if p_afib > config.AFIB_THRESHOLD else "not AFib"
        out.afib_model = afib_endpoint
    return out


@functools.cache
def _warmup_inputs() -> list:
    """One fixed synthetic NSR ECG as a (1, N_LEADS, N_SAMPLES) tensor. One ECG per ping: a request
    of two ECGs exceeds the 1 MiB cap above which AI Gateway does not log the payload."""
    batch = synthesize_ecgs(np.array([NSR]), np.random.default_rng(0), noise_level=1.0, n_leads=config.N_LEADS)
    return batch.ecg[:1].astype(np.float32).tolist()


def _ping(serving: WorkspaceClient, endpoint: str, inputs: list) -> EndpointStatus:
    start = time.monotonic()
    try:
        _predict(serving, endpoint, inputs, client_request_id=config.WARMUP_REQUEST_ID)
    except HTTPException as err:
        error = ServingError.model_validate(err.detail)
        seconds = round(time.monotonic() - start, 1)
        if error.reason == "waking":
            return EndpointStatus(endpoint=endpoint, status="cold", seconds=seconds, message=error.message)
        return EndpointStatus(endpoint=endpoint, status="error", seconds=seconds, message=error.message)
    return EndpointStatus(endpoint=endpoint, status="ready", seconds=round(time.monotonic() - start, 1))


@router.post("/warmup", response_model=WarmupResponse, operation_id="warmup")
def warmup(serving: ServingClient, settings: Dependencies.Config):
    """Wake both endpoints, or keep them awake: ONE single-ECG request to each, in parallel, each
    bounded by SERVING_TIMEOUT_S like /analyze. Always 200, with each endpoint's status.

    Each ping is a real query with the app's existing CAN_QUERY grant (no other permission), so AI
    Gateway logs it as one row in that endpoint's payload table, with client_request_id =
    config.WARMUP_REQUEST_ID to tell it apart from a scored ECG; the dashboard's live-serving
    section leaves those rows out and counts them separately."""
    inputs = _warmup_inputs()   # built once per worker (the first build takes seconds), never timed
    endpoints = list(settings.endpoints)
    with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
        statuses = list(pool.map(lambda endpoint: _ping(serving, endpoint, inputs), endpoints))
    return WarmupResponse(endpoints=statuses)
