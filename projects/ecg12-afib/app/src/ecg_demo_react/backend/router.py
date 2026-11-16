import datetime as dt
import functools
import math
import tempfile
import time
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TypeVar

import numpy as np
from databricks.sdk import WorkspaceClient
from databricks.sdk.config import Config
from databricks.sdk.errors import DatabricksError, DeadlineExceeded, TooManyRequests
from fastapi import FastAPI, HTTPException, Request
from pydantic import TypeAdapter, ValidationError

from .._metadata import app_slug
from .common import config, patient_info
from .common.ecg_signal_gen import AFIB, LABEL_NAMES, N_SAMPLES, NSR, OTHER, SAMPLING_RATE, synthesize_ecgs
from .core import Dependencies, create_router, logger
from .core._base import LifespanDependency
from .models import (
    AfibPrediction,
    AnalyzeRequest,
    AnalyzeResponse,
    AppConfigOut,
    ArrPrediction,
    EndpointStatus,
    GenerateRequest,
    GenerateResponse,
    RateLimited,
    RateLimitedOut,
    ServingError,
    ServingErrorOut,
    WarmupResponse,
)
from .shared_state import SharedState

router = create_router()

# UI rhythm choice -> generator class label.
RHYTHM_LABELS = {
    "NSR (normal)": NSR,
    "AFib": AFIB,
    "Other (tachycardia)": OTHER,
}

# The replies /api/analyze can end in besides 200 and 422: the app's own rate limit (429, a
# RateLimited `detail`) or a failed endpoint call (429 / 502 / 503, a ServingError `detail`).
ANALYZE_ERRORS: dict[int | str, dict[str, Any]] = {
    429: {"model": RateLimitedOut | ServingErrorOut,
          "description": "Too many requests to this app (RateLimited), or an endpoint is rate-limiting requests"},
    502: {"model": ServingErrorOut, "description": "An endpoint returned an error, or an answer the app cannot read"},
    503: {"model": ServingErrorOut, "description": "An endpoint did not answer in time (waking from scale-to-zero)"},
}
WARMUP_ERRORS: dict[int | str, dict[str, Any]] = {
    429: {"model": RateLimitedOut, "description": "Too many warm-up pings to this app"},
}


@dataclass(frozen=True)
class _Limit:
    """At most ``per_caller`` calls by one user and ``total`` by all users in any ``window_s`` seconds."""
    route: str          # the counter's name (shared_state.py)
    what: str           # what the user is told was limited
    per_caller: int
    total: int
    window_s: int


# Each /api/analyze makes up to two endpoint calls; one person clicking Triage -> Classify (the UI
# sends one at a time) stays well below 10 a minute. /api/warmup counts only the pings that query
# the endpoints, not the cached answers (WARMUP_CACHE_S): one Live tab sends at most 1 +
# MAX_COLD_PINGS (routes/index.tsx) = 9 while the endpoints wake, then one every WARMUP_INTERVAL_S.
ANALYZE_LIMIT = _Limit("analyze", "Triage -> Classify requests", per_caller=10, total=60, window_s=60)
WARMUP_LIMIT = _Limit("warmup", "warm-up pings", per_caller=12, total=60, window_s=config.WARMUP_INTERVAL_S)
# When both endpoints answered a warm-up ping less than this long ago, /api/warmup answers from that
# ping without querying them, so any number of open tabs query each endpoint about once per
# WARMUP_INTERVAL_S. A gap between two queries is then at most WARMUP_CACHE_S + WARMUP_INTERVAL_S
# (15 min), inside the endpoints' 30 min idle window.
WARMUP_CACHE_S = config.WARMUP_INTERVAL_S // 2


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


class _SharedStateDependency(LifespanDependency):
    """The rate-limit counts and warm-up cache that the app's uvicorn workers share (app.yml runs
    two; see shared_state.py), in one SQLite file in the system temp folder. Each worker opens it at
    startup."""

    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncGenerator[None, None]:
        app.state.shared_state = SharedState(Path(tempfile.gettempdir()) / f"{app_slug}_rate_limits_and_warmup.sqlite3")
        yield

    @staticmethod
    def __call__(request: Request) -> SharedState:
        return request.app.state.shared_state


SharedStateDep = Annotated[SharedState, _SharedStateDependency.depends()]


def _caller(request: Request) -> str:
    """Who is calling: the user id the Databricks Apps proxy sets as X-Forwarded-User on every
    request it lets through, else (local development) the client address."""
    return request.headers.get("X-Forwarded-User") or (request.client.host if request.client else "unknown")


def _admit(state: SharedState, request: Request, limit: _Limit) -> None:
    """Count this call against ``limit``, or refuse it: a 429 with a RateLimited detail and a
    Retry-After header."""
    refusal = state.admit(limit.route, _caller(request), per_caller=limit.per_caller, total=limit.total,
                          window_s=limit.window_s)
    if refusal is None:
        return
    retry_after_s = max(1, math.ceil(refusal.retry_after_s))
    whose = "per user" if refusal.scope == "caller" else "from all users together"
    raise HTTPException(status_code=429, headers={"Retry-After": str(retry_after_s)}, detail=RateLimited(
        scope=refusal.scope, limit=refusal.limit, window_s=limit.window_s, retry_after_s=retry_after_s,
        message=(f"This app allows {refusal.limit} {limit.what} {whose} in {limit.window_s} s. "
                 f"Try again in {retry_after_s} s."),
    ).model_dump())


def _endpoint_error(status_code: int, error: ServingError) -> HTTPException:
    return HTTPException(status_code=status_code, detail=error.model_dump())


# What each endpoint's first prediction must look like (models.py); _predict checks it.
_ARR_OUTPUT = TypeAdapter(ArrPrediction)
_AFIB_OUTPUT = TypeAdapter(AfibPrediction)
_P = TypeVar("_P")


def _predict(serving: WorkspaceClient, endpoint: str, inputs: Any, output: TypeAdapter[_P],
             client_request_id: str | None = None) -> _P:
    """First prediction of one endpoint call, checked against ``output``. Every failure becomes a
    429 / 502 / 503 whose detail is a ServingError the UI shows as is, never a hang or a bare 500.
    The endpoint's own error text and any unreadable answer go to the app's log, not to the user."""
    waking = ServingError(
        endpoint=endpoint, reason="waking",
        message=(f"The model endpoint {endpoint} did not answer within {config.SERVING_TIMEOUT_S} s. "
                 "It is most likely waking up from scale-to-zero: try again in a minute or two."),
    )
    see_log = "Try again; if it keeps failing, the app's logs have the details."
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
        logger.warning("model endpoint %s returned an error: %s", endpoint, err)
        raise _endpoint_error(502, ServingError(
            endpoint=endpoint, reason="error", message=f"The model endpoint {endpoint} returned an error. {see_log}",
        )) from err
    except Exception as err:   # e.g. a connection or credential error the SDK does not retry
        logger.exception("call to model endpoint %s failed", endpoint)
        raise _endpoint_error(502, ServingError(
            endpoint=endpoint, reason="error",
            message=f"The app could not get an answer from the model endpoint {endpoint}. {see_log}",
        )) from err
    predictions = res.predictions
    try:
        if not isinstance(predictions, list) or not predictions:
            raise ValueError(f"predictions is {predictions!r:.200}, not a non-empty list")
        return output.validate_python(predictions[0])
    except (ValueError, ValidationError) as err:   # ValidationError is a ValueError; named for the reader
        logger.warning("model endpoint %s answered in an unexpected format: %s", endpoint, err)
        raise _endpoint_error(502, ServingError(
            endpoint=endpoint, reason="error",
            message=f"The model endpoint {endpoint} answered in a format the app cannot read. {see_log}",
        )) from err


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
        acquisition_datetime_utc=dt.datetime.now(dt.UTC).replace(microsecond=0),
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


@router.post("/analyze", response_model=AnalyzeResponse, operation_id="analyze", responses=ANALYZE_ERRORS)
def analyze(req: AnalyzeRequest, request: Request, serving: ServingClient, state: SharedStateDep,
            settings: Dependencies.Config):
    """Triage (arr) then, if flagged, classify (afib) via the live serving endpoints.

    All N_LEADS leads go to both models, as a (1, N_LEADS, N_SAMPLES) tensor. Each call is bounded
    by SERVING_TIMEOUT_S (see _ServingClientDependency), so this request makes at most two bounded
    calls. ANALYZE_LIMIT refuses it (429) before any call when a user, or all users together, sent
    too many."""
    _admit(state, request, ANALYZE_LIMIT)
    arr_endpoint, afib_endpoint = settings.endpoints    # this deployment's names (endpoint_prefix)
    # AnalyzeRequest has checked the shape and that every value is a finite float32 (else 422).
    inputs = np.asarray(req.ecg, dtype=np.float32)[None, :, :].tolist()
    arr = _predict(serving, arr_endpoint, inputs, _ARR_OUTPUT)
    flagged = arr.arr_flag == 1.0
    out = AnalyzeResponse(
        arr_flag=arr.arr_flag, arr_prob=arr.arr_prob, flagged=flagged,
        arr_model=arr_endpoint,
    )
    if flagged:
        # The AFib model takes the same ecg tensor and returns [p_not_afib, p_afib].
        _, p_afib = _predict(serving, afib_endpoint, inputs, _AFIB_OUTPUT)
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


def _ping(serving: WorkspaceClient, endpoint: str, output: TypeAdapter[Any], inputs: list) -> EndpointStatus:
    """One endpoint's warm-up status; never raises."""
    start = time.monotonic()
    try:
        _predict(serving, endpoint, inputs, output, client_request_id=config.WARMUP_REQUEST_ID)
    except HTTPException as err:
        error = ServingError.model_validate(err.detail)
        seconds = round(time.monotonic() - start, 1)
        if error.reason == "waking":
            return EndpointStatus(endpoint=endpoint, status="cold", seconds=seconds, message=error.message)
        return EndpointStatus(endpoint=endpoint, status="error", seconds=seconds, message=error.message)
    return EndpointStatus(endpoint=endpoint, status="ready", seconds=round(time.monotonic() - start, 1))


@router.post("/warmup", response_model=WarmupResponse, operation_id="warmup", responses=WARMUP_ERRORS)
def warmup(request: Request, serving: ServingClient, state: SharedStateDep, settings: Dependencies.Config):
    """Wake both endpoints, or keep them awake: ONE single-ECG request to each, in parallel, each
    bounded by SERVING_TIMEOUT_S like /analyze. 200 with each endpoint's status (an endpoint that
    fails in any way is "error", with a message and the details in the app's log), or 429 when
    WARMUP_LIMIT refuses the ping. When both endpoints answered a ping (from any tab) less than
    WARMUP_CACHE_S ago, the reply is that ping's "ready", with no new query and not counted.

    Each ping is a real query with the app's existing CAN_QUERY grant (no other permission), so AI
    Gateway logs it as one row in that endpoint's payload table, with client_request_id =
    config.WARMUP_REQUEST_ID to tell it apart from a scored ECG; the dashboard's live-serving
    section leaves those rows out and counts them separately."""
    endpoints = list(settings.endpoints)
    answers = state.ready_within(endpoints, WARMUP_CACHE_S)
    if answers is not None:
        return WarmupResponse(endpoints=[EndpointStatus(
            endpoint=endpoint, status="ready", seconds=answer.seconds,
            message=f"answered a warm-up ping {round(answer.age_s)} s ago",
        ) for endpoint, answer in zip(endpoints, answers)])
    _admit(state, request, WARMUP_LIMIT)
    try:
        inputs = _warmup_inputs()   # built once per worker (the first build takes seconds), never timed
    except Exception:
        logger.exception("could not build the warm-up ECG")
        return WarmupResponse(endpoints=[EndpointStatus(
            endpoint=endpoint, status="error", seconds=0.0,
            message="The app could not build its warm-up request; its logs have the details.",
        ) for endpoint in endpoints])
    with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
        statuses = list(pool.map(lambda endpoint, output: _ping(serving, endpoint, output, inputs),
                                 endpoints, (_ARR_OUTPUT, _AFIB_OUTPUT)))
    for status in statuses:
        if status.status == "ready":
            state.mark_ready(status.endpoint, status.seconds)
    return WarmupResponse(endpoints=statuses)
