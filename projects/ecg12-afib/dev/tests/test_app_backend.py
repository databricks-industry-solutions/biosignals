"""The app's backend (app/src/ecg_demo_react/backend/): the rate-limit counts and warm-up cache its
workers share (shared_state.py), and the routes' replies to a bad ECG, a rate limit and a failed or
unreadable endpoint call (router.py, models.py), against a stand-in serving client.

shared_state.py is standard library only, so its tests run in any environment, a clean checkout
included: they load the file itself, not through the ecg_demo_react package, whose __init__.py
imports the _version.py that apx writes. The route tests need the app's own dependencies (FastAPI,
databricks-sdk), which the environment of run_tests_python311.sh does not have, and the
_metadata.py / _version.py that apx writes (app/build_from_lockfiles.sh); they skip without them.
README.md (this folder) has the command that runs them in the app's locked environment.
"""
import importlib.util
import json
import logging
import sqlite3
import sys
from types import SimpleNamespace

import pytest
from conftest import DEMO_ROOT, TEST_CATALOG

APP_SRC = DEMO_ROOT / "app" / "src"
sys.path.insert(0, str(APP_SRC))


def _load_shared_state():
    """shared_state.py on its own (standard library only), without the package's generated files."""
    spec = importlib.util.spec_from_file_location(
        "app_shared_state", APP_SRC / "ecg_demo_react" / "backend" / "shared_state.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module                   # @dataclass looks its module up here
    spec.loader.exec_module(module)
    return module


SharedState = _load_shared_state().SharedState

N_LEADS, N_SAMPLES = 12, 2500


class Clock:
    """A settable wall clock for SharedState."""

    def __init__(self, now: float = 1_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def state(tmp_path, clock):
    return SharedState(tmp_path / "state.sqlite3", clock=clock)


# --- shared_state.py ---

def test_admit_counts_each_caller_and_all_callers(state, clock):
    limits = {"per_caller": 2, "total": 3, "window_s": 60}
    assert state.admit("analyze", "alice", **limits) is None
    clock.now += 10
    assert state.admit("analyze", "alice", **limits) is None
    refusal = state.admit("analyze", "alice", **limits)
    assert (refusal.scope, refusal.limit, refusal.retry_after_s) == ("caller", 2, 50.0)   # her first call leaves at +60
    assert state.admit("analyze", "bob", **limits) is None
    refusal = state.admit("analyze", "carol", **limits)
    assert (refusal.scope, refusal.limit, refusal.retry_after_s) == ("total", 3, 50.0)
    clock.now += 50                                    # alice's first call has left the window
    assert state.admit("analyze", "carol", **limits) is None


def test_a_refused_call_is_not_counted(state, clock):
    limits = {"per_caller": 1, "total": 10, "window_s": 60}
    assert state.admit("analyze", "alice", **limits) is None
    for _ in range(5):
        assert state.admit("analyze", "alice", **limits) is not None
    clock.now += 60
    assert state.admit("analyze", "alice", **limits) is None   # the refusals did not extend the wait


def test_routes_count_separately_each_over_its_own_window(state, clock):
    """Pruning one route's expired calls must not drop another route's, counted over a longer window."""
    assert state.admit("warmup", "alice", per_caller=1, total=10, window_s=600) is None
    clock.now += 120
    assert state.admit("analyze", "alice", per_caller=1, total=10, window_s=60) is None
    assert state.admit("warmup", "alice", per_caller=1, total=10, window_s=600) is not None


def test_every_worker_process_sees_the_same_counts(tmp_path, clock):
    """app.yml runs two uvicorn workers; each opens the same file."""
    path = tmp_path / "state.sqlite3"
    worker_1, worker_2 = SharedState(path, clock=clock), SharedState(path, clock=clock)
    assert worker_1.admit("analyze", "alice", per_caller=1, total=10, window_s=60) is None
    assert worker_2.admit("analyze", "alice", per_caller=1, total=10, window_s=60).scope == "caller"


def test_ready_within_needs_a_recent_answer_from_every_endpoint(state, clock):
    assert state.ready_within(["arr", "afib"], 300) is None
    state.mark_ready("arr", 1.5)
    assert state.ready_within(["arr", "afib"], 300) is None      # afib never answered
    clock.now += 100
    state.mark_ready("afib", 2.0)
    answers = state.ready_within(["arr", "afib"], 300)
    assert [(a.age_s, a.seconds) for a in answers] == [(100.0, 1.5), (0.0, 2.0)]
    clock.now += 200                                              # arr's answer is now 300 s old
    assert state.ready_within(["arr", "afib"], 300) is None


def test_a_broken_state_file_lets_calls_through_and_is_logged(tmp_path, clock, caplog):
    path = tmp_path / "state.sqlite3"
    state = SharedState(path, clock=clock)
    for leftover in tmp_path.glob("state.sqlite3*"):
        leftover.unlink()
    path.write_bytes(b"not a database" * 100)
    with caplog.at_level(logging.ERROR):
        assert state.admit("analyze", "alice", per_caller=0, total=0, window_s=60) is None
        state.mark_ready("arr", 1.0)
        assert state.ready_within(["arr"], 300) is None
    assert len(caplog.records) == 3 and all(r.exc_info and issubclass(r.exc_info[0], sqlite3.Error)
                                            for r in caplog.records)


# --- routes (router.py, models.py) ---

class StandInServing:
    """Stands in for the serving WorkspaceClient: ``serving_endpoints.query`` answers from
    ``answers`` (endpoint name -> the response's ``predictions``, or an exception to raise) and
    records each call."""

    def __init__(self):
        self.answers = {}
        self.calls = []
        self.serving_endpoints = self

    def query(self, name, inputs, client_request_id=None):
        self.calls.append((name, client_request_id))
        answer = self.answers[name]
        if isinstance(answer, BaseException):
            raise answer
        return SimpleNamespace(predictions=answer)


@pytest.fixture
def backend(tmp_path, clock):
    """The app with a stand-in serving client and its own state file, plus the router module."""
    pytest.importorskip("fastapi")
    # FastAPI's TestClient is starlette's, which uses httpx2, or httpx with a deprecation warning.
    if not any(importlib.util.find_spec(client) for client in ("httpx2", "httpx")):
        pytest.skip("needs httpx2, the HTTP client of starlette's TestClient")
    missing = [name for name in ("_metadata.py", "_version.py") if not (APP_SRC / "ecg_demo_react" / name).is_file()]
    if missing:
        pytest.skip(f"app/src/ecg_demo_react/{' and '.join(missing)} not generated: run app/build_from_lockfiles.sh")
    from ecg_demo_react.backend import router
    from ecg_demo_react.backend.core import create_app
    from ecg_demo_react.backend.core._config import AppConfig
    from ecg_demo_react.backend.core._static import add_not_found_handler
    from ecg_demo_react.backend.shared_state import SharedState  # the class the routes annotate
    from fastapi.testclient import TestClient

    app = create_app(routers=[router.router])
    add_not_found_handler(app)            # the error handler the deployed app has (it serves the UI)
    # What the lifespans would set, without the workspace client the app-config route needs.
    app.state.config = AppConfig(catalog=TEST_CATALOG)
    app.state.serving_client = serving = StandInServing()
    app.state.shared_state = SharedState(tmp_path / "state.sqlite3", clock=clock)
    arr, afib = app.state.config.endpoints
    serving.answers = {arr: [{"arr_prob": 0.9, "arr_flag": 1.0}], afib: [[0.2, 0.8]]}
    return SimpleNamespace(client=TestClient(app), serving=serving, router=router, arr=arr, afib=afib)


def _ecg(value=0.0):
    return [[value] * N_SAMPLES for _ in range(N_LEADS)]


def test_analyze_scores_and_sends_the_not_for_clinical_use_notice(backend):
    from ecg_demo_react.backend.models import NOT_FOR_CLINICAL_USE

    reply = backend.client.post("/api/analyze", json={"ecg": _ecg()})
    assert reply.status_code == 200, reply.text
    body = reply.json()
    assert (body["flagged"], body["p_afib"], body["afib_label"]) == (True, 0.8, "AFib")
    assert body["notice"] == NOT_FOR_CLINICAL_USE and NOT_FOR_CLINICAL_USE.startswith("Not for clinical use")
    assert [name for name, _ in backend.serving.calls] == [backend.arr, backend.afib]


@pytest.mark.parametrize("ecg", [
    _ecg()[:-1],                                            # 11 leads
    _ecg()[:-1] + [[0.0] * (N_SAMPLES - 1)],                # ragged: one lead a sample short
    [], [[]],
    _ecg(float("nan")), _ecg(float("inf")),                # json.dumps writes NaN / Infinity
    _ecg(1e39),                                             # finite, but infinite as a float32
    [["x"] * N_SAMPLES for _ in range(N_LEADS)],
], ids=["11-leads", "ragged", "empty", "empty-lead", "nan", "inf", "beyond-float32", "text"])
def test_analyze_refuses_a_malformed_ecg_with_422_before_any_endpoint_call(backend, ecg):
    reply = backend.client.post("/api/analyze", content=json.dumps({"ecg": ecg}),
                                headers={"Content-Type": "application/json"})
    assert reply.status_code == 422, reply.text
    assert backend.serving.calls == []
    # core/_factory.py: at most MAX_VALIDATION_ERRORS errors (+ one "and N more"), none echoing the input.
    errors = reply.json()["detail"]
    assert 1 <= len(errors) <= 11 and not any("input" in error for error in errors)
    assert len(reply.content) < 5_000


@pytest.mark.parametrize("arr_answer, message", [
    (None, "format the app cannot read"),
    ([], "format the app cannot read"),
    ([{"arr_prob": 0.5}], "format the app cannot read"),                    # no arr_flag
    ([{"arr_prob": 1.5, "arr_flag": 1.0}], "format the app cannot read"),   # not a probability
    ([{"arr_prob": 0.5, "arr_flag": 0.5}], "format the app cannot read"),   # flag not 0 / 1
    ("not a list", "format the app cannot read"),
    ("databricks-error", "returned an error"),
    (ConnectionError("connection refused by host-internal-name"), "could not get an answer"),
])
def test_analyze_endpoint_failures_are_502s_without_the_endpoint_text(backend, caplog, arr_answer, message):
    from databricks.sdk.errors import DatabricksError

    if arr_answer == "databricks-error":
        arr_answer = DatabricksError("INTERNAL_ERROR: stack trace from host-internal-name")
    backend.serving.answers[backend.arr] = arr_answer
    with caplog.at_level(logging.WARNING):
        reply = backend.client.post("/api/analyze", json={"ecg": _ecg()})
    assert reply.status_code == 502, reply.text
    detail = reply.json()["detail"]
    assert (detail["endpoint"], detail["reason"]) == (backend.arr, "error")
    assert message in detail["message"] and "logs" in detail["message"]
    assert "host-internal-name" not in reply.text                 # the endpoint's text stays in the log
    assert any(backend.arr in r.getMessage() for r in caplog.records)


def test_analyze_unreadable_afib_answer_is_a_502(backend):
    backend.serving.answers[backend.afib] = [["x", "y"]]
    reply = backend.client.post("/api/analyze", json={"ecg": _ecg()})
    assert reply.status_code == 502 and reply.json()["detail"]["endpoint"] == backend.afib


def test_analyze_rate_limit_per_user_and_for_all_users(backend, monkeypatch, clock):
    limit = backend.router._Limit("analyze", "Triage -> Classify requests", per_caller=2, total=3, window_s=60)
    monkeypatch.setattr(backend.router, "ANALYZE_LIMIT", limit)

    def analyze(user):
        return backend.client.post("/api/analyze", json={"ecg": _ecg()}, headers={"X-Forwarded-User": user})

    assert [analyze("alice").status_code for _ in range(2)] == [200, 200]
    clock.now += 15
    refused = analyze("alice")
    assert refused.status_code == 429 and refused.headers["Retry-After"] == "45"
    detail = refused.json()["detail"]
    assert {k: detail[k] for k in ("scope", "limit", "window_s", "retry_after_s")} == {
        "scope": "caller", "limit": 2, "window_s": 60, "retry_after_s": 45}
    assert "Try again in 45 s" in detail["message"]
    assert analyze("bob").status_code == 200
    assert analyze("carol").json()["detail"]["scope"] == "total"
    assert len(backend.serving.calls) == 3 * 2                     # refused calls query nothing
    # A malformed ECG is a 422 whatever the counts (FastAPI checks the body before the route runs).
    assert backend.client.post("/api/analyze", json={"ecg": []},
                               headers={"X-Forwarded-User": "alice"}).status_code == 422


def test_generate_rate_limit_per_user_and_for_all_users(backend, monkeypatch, clock):
    """/api/generate is limited like /api/analyze and /api/warmup (the same _admit and SharedState
    counts): a 429 with a RateLimited detail and Retry-After once a user, or all users together,
    asked for too many readings in the window; another user still gets one until the total is hit."""
    assert backend.router.GENERATE_LIMIT.route == "generate"        # its own counter, not analyze's
    limit = backend.router._Limit("generate", "new readings", per_caller=2, total=3, window_s=60)
    monkeypatch.setattr(backend.router, "GENERATE_LIMIT", limit)

    def generate(user):
        return backend.client.post("/api/generate", json={"rhythm": "AFib"}, headers={"X-Forwarded-User": user})

    assert [generate("alice").status_code for _ in range(2)] == [200, 200]
    clock.now += 15
    refused = generate("alice")
    assert refused.status_code == 429 and refused.headers["Retry-After"] == "45"
    detail = refused.json()["detail"]
    assert {k: detail[k] for k in ("scope", "limit", "window_s", "retry_after_s")} == {
        "scope": "caller", "limit": 2, "window_s": 60, "retry_after_s": 45}
    assert detail["message"] == "This app allows 2 new readings per user in 60 s. Try again in 45 s."
    assert generate("bob").status_code == 200
    assert generate("carol").json()["detail"]["scope"] == "total"
    assert backend.client.post("/api/analyze", json={"ecg": _ecg()},
                               headers={"X-Forwarded-User": "alice"}).status_code == 200   # counted apart
    clock.now += 60
    assert generate("alice").status_code == 200                    # the window has passed


def test_warmup_answers_from_a_recent_ping_without_querying_or_counting(backend, monkeypatch, clock):
    from ecg_demo_react.backend.common import config

    limit = backend.router._Limit("warmup", "warm-up pings", per_caller=1, total=10, window_s=600)
    monkeypatch.setattr(backend.router, "WARMUP_LIMIT", limit)
    first = backend.client.post("/api/warmup")
    assert first.status_code == 200 and [e["status"] for e in first.json()["endpoints"]] == ["ready", "ready"]
    assert sorted(backend.serving.calls) == sorted([(backend.arr, config.WARMUP_REQUEST_ID),
                                                    (backend.afib, config.WARMUP_REQUEST_ID)])
    clock.now += backend.router.WARMUP_CACHE_S - 1
    cached = backend.client.post("/api/warmup").json()["endpoints"]
    assert [e["status"] for e in cached] == ["ready", "ready"] and len(backend.serving.calls) == 2
    assert cached[0]["message"] == f"answered a warm-up ping {backend.router.WARMUP_CACHE_S - 1} s ago"
    clock.now += 1                                                 # the ping is too old: query again, counted
    refused = backend.client.post("/api/warmup")
    assert refused.status_code == 429 and refused.json()["detail"]["scope"] == "caller"
    assert int(refused.headers["Retry-After"]) == 600 - backend.router.WARMUP_CACHE_S


def test_warmup_reports_cold_and_error_and_caches_neither(backend):
    from databricks.sdk.errors import DatabricksError

    backend.serving.answers = {backend.arr: TimeoutError("retries exhausted"),
                               backend.afib: DatabricksError("internal detail")}
    for _ in range(2):
        endpoints = backend.client.post("/api/warmup").json()["endpoints"]
        assert [(e["endpoint"], e["status"]) for e in endpoints] == [(backend.arr, "cold"), (backend.afib, "error")]
        assert "internal detail" not in endpoints[1]["message"]
    assert len(backend.serving.calls) == 4                         # no answer was cached
