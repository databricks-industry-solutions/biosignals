"""Which Model Serving deployment path a served entity took: express, or standard (container image build).

Express deployment has no switch: Model Serving uses it for an eligible model (02_ / 03_ log the
models with ``config.MODEL_PIP_REQUIREMENTS``, exact plain PyPI pins) and builds a container image
otherwise, or when express does not apply. Two signals tell the paths apart, as Model Serving
reports them:

* the served entity's build log (databricks-sdk ``serving_endpoints.build_logs``). Express starts it
  with "Express deployment is preparing this model's environment and artifacts. ..." while it
  prepares, and "This environment was prepared ahead of time by express deployment and is shared by
  models with identical requirements." once it is ready; a container image build logs the image
  build instead.
* the endpoint's events (``GET /api/2.0/serving-endpoints/<name>/events``): per served entity,
  ``SERVED_ENTITY_CONTAINER_EVENT`` "Model artifact preparation started" / "... completed
  successfully" on the express path, "Container image creation initiated" / "... finished
  successfully" on the standard one.

An update that keeps a served entity (same name, new config version) reuses its environment and adds
no container events, so the build log is the first express signal; the events add how long the
preparation or the image build took. Both are needed: a container image build (express that fell
back) shows only in the events, so when either could not be read, the path is unknown, never
express from the build log alone.

The same events also tell 05_'s wait whether the update it just sent failed (``update_failure``):
Model Serving writes ``ENDPOINT_UPDATE_EVENT`` "Endpoint update failed for endpoint `<name>`, config
version <n>." (after "Served entity creation aborted for served entity `<entity>`, config version
<n>, since the update timed out." or another reason). The events carry no id, and their timestamps
are Model Serving's clock, not the notebook's, so an event is told apart as new by comparing with the
events read just before the update was sent (``fingerprints``, counted, so an event identical to
an earlier one is still new), never by time. The events API returns an endpoint's events in no
particular order and, in the responses seen, with no page token. Pure: 05_ fetches the log and the
events, this module only reads them.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

EXPRESS, STANDARD, UNKNOWN = "express", "standard", "unknown"
# The endpoint tag 05_ sets to the path of the endpoint's current served entity.
DEPLOYMENT_TAG = "deployment"

CONTAINER_EVENT = "SERVED_ENTITY_CONTAINER_EVENT"
EXPRESS_LOG_MARKER = "express deployment"                  # in the build log, any case
EXPRESS_STARTED = "Model artifact preparation started"
EXPRESS_DONE = "Model artifact preparation completed successfully"
IMAGE_STARTED = "Container image creation initiated"
IMAGE_DONE = "Container image creation finished successfully"
UPDATE_EVENT = "ENDPOINT_UPDATE_EVENT"
UPDATE_FAILED = "Endpoint update failed for endpoint `{endpoint}`"   # then ", config version <n>."


class DeploymentPath(NamedTuple):
    path: str            # EXPRESS, STANDARD or UNKNOWN
    line: str            # one line for the notebook output
    unread: str = ""     # for UNKNOWN: which read failed ("" when both were read but named neither path)


def _is_events(events: Any) -> bool:
    return isinstance(events, list) and all(isinstance(e, Mapping) for e in events)


def events_of(response: Any) -> list[Mapping[str, Any]] | None:
    """The events of an events API response, ``{"events": [<event object>, ...]}``; None for anything
    else (not a mapping, no ``events`` list, an item that is not an object), which classify() treats
    as events that could not be read. A missing ``events`` key is not taken as "no events": the
    endpoints 05_ just created or updated always have some."""
    events = response.get("events") if isinstance(response, Mapping) else None
    return events if _is_events(events) else None


def _minutes(seconds: float) -> str:
    seconds = round(seconds)
    return f"{seconds // 60}m{seconds % 60:02d}s"


def _took(events: Sequence[Mapping[str, Any]], started: str, done: str) -> float | None:
    """Seconds from the last ``started`` event to the first ``done`` event after it, or None."""
    def times(message):
        return [e["timestamp"] for e in events if e.get("message") == message
                and isinstance(e.get("timestamp"), (int, float))]

    starts = times(started)
    if not starts:
        return None
    ends = [t for t in times(done) if t >= max(starts)]
    return (min(ends) - max(starts)) / 1000 if ends else None


def classify(served_entity: str, build_log: str | None,
             events: Sequence[Mapping[str, Any]] | None) -> DeploymentPath:
    """The deployment path of ``served_entity`` from its build log (None, or anything but text, when
    it could not be read) and the endpoint's events (``events_of``'s list, any order; None when they
    could not be read). Unknown when either could not be read. Otherwise a container image build
    wins over the express signals: express that falls back still writes its first log line, and then
    builds an image."""
    unread = " and ".join(name for name, ok in (("its build log", isinstance(build_log, str)),
                                                ("the endpoint's events", _is_events(events))) if not ok)
    if unread:
        return DeploymentPath(UNKNOWN, f"{served_entity}: UNKNOWN deployment path (could not read {unread})",
                              unread)
    mine = [e for e in events if e.get("event_type") == CONTAINER_EVENT
            and e.get("served_entity_name") == served_entity]
    if any(e.get("message") in (IMAGE_STARTED, IMAGE_DONE) for e in mine):
        took = _took(mine, IMAGE_STARTED, IMAGE_DONE)
        detail = f"container image built in {_minutes(took)}" if took is not None else "container image build"
        return DeploymentPath(STANDARD, f"{served_entity}: STANDARD deployment ({detail}; not express)")
    express_events = any(e.get("message") in (EXPRESS_STARTED, EXPRESS_DONE) for e in mine)
    if express_events or EXPRESS_LOG_MARKER in build_log.lower():
        took = _took(mine, EXPRESS_STARTED, EXPRESS_DONE)
        if took is not None:
            detail = f"environment prepared in {_minutes(took)}"
        elif any(e.get("message") == EXPRESS_STARTED for e in mine):
            detail = "environment preparation not finished yet"
        else:
            detail = "environment prepared earlier and reused"
        return DeploymentPath(EXPRESS, f"{served_entity}: EXPRESS deployment ({detail}; no container image build)")
    return DeploymentPath(UNKNOWN, f"{served_entity}: UNKNOWN deployment path (neither its build log nor the "
                                   "endpoint's events name express deployment or a container image build)")


def _fingerprint(event: Mapping[str, Any]) -> tuple:
    """What tells two events apart (the events API gives no id): type, time, message, config version."""
    return tuple(str(event.get(key)) for key in ("event_type", "timestamp", "message", "config_version"))


def fingerprints(events: Sequence[Mapping[str, Any]] | None) -> Counter[tuple] | None:
    """The events read just before 05_ sends an update, as the baseline ``update_failure`` compares
    with: how many events of each fingerprint there were; None when they could not be read
    (``events_of`` gave None)."""
    return Counter(_fingerprint(e) for e in events) if _is_events(events) else None


def update_failure(events: Sequence[Mapping[str, Any]] | None, endpoint: str, before: Counter[tuple]) -> list[str]:
    """Model Serving's messages about a failed update of ``endpoint`` among the events that are new
    since ``before`` (``fingerprints`` of the events read just before the update was sent; of the
    events with one fingerprint, those beyond the baseline's count are new): when the new events
    include an ``ENDPOINT_UPDATE_EVENT`` "Endpoint update failed for endpoint `<endpoint>`", every
    failure or abort message among them, oldest first; otherwise (no such event, only ones that were
    there before, another endpoint's, or events that could not be read: None) an empty list.

    Accepted gap: an earlier failure event that Model Serving delivers late, missing from the
    baseline read but present in a read after the send, counts as new. Accepted because it needs an
    update sent again right after a failure, while that failure's events are still arriving, and
    because the result is only a SETUP task that fails with Model Serving's message and is run again:
    the false failure changes nothing the endpoint serves (the update already sent goes on)."""
    if not _is_events(events):
        return []
    seen: Counter[tuple] = Counter()
    new = []
    for event in events:
        key = _fingerprint(event)
        seen[key] += 1
        if seen[key] > before[key] and isinstance(event.get("message"), str):
            new.append(event)
    trigger = UPDATE_FAILED.format(endpoint=endpoint)
    if not any(e.get("event_type") == UPDATE_EVENT and e["message"].startswith(trigger) for e in new):
        return []
    new.sort(key=lambda e: e["timestamp"] if isinstance(e.get("timestamp"), (int, float)) else 0)
    return [e["message"] for e in new if any(word in e["message"].lower() for word in ("failed", "aborted"))]
