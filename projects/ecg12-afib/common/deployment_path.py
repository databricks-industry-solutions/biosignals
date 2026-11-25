"""Which Model Serving deployment path a served entity took: express, or standard (container image build).

Express deployment has no switch: Model Serving uses it for an eligible model (02_ / 03_ log the
models with ``config.MODEL_PIP_REQUIREMENTS``, exact plain PyPI pins) and builds a container image
otherwise, or when express does not apply. Two signals tell the paths apart, as Model Serving
reports them:

* the served entity's build log (databricks-sdk ``serving_endpoints.build_logs``; the API promises
  only "The logs associated with building the served entity's environment"). In our tests express
  started it with "Express deployment is preparing this model's environment and artifacts. ..."
  while it prepared, and with "This environment was prepared ahead of time by express deployment
  and is shared by models with identical requirements." once it was ready; a container image build
  logged the image build instead.
* the endpoint's events (``GET /api/2.0/serving-endpoints/<name>/events``): per served entity,
  ``SERVED_ENTITY_CONTAINER_EVENT`` "Model artifact preparation started" / "... completed
  successfully" on the express path, "Container image creation initiated" / "... finished
  successfully" on the standard one.

Only a finished build names the path: "Container image creation finished successfully" for
standard, "Model artifact preparation completed successfully" for express. A build that started and
has not finished (still running, or it stopped) is unknown. An update that keeps a served entity
(same name, new config version) reuses its environment and adds no container events; with no build
in the events, the "prepared ahead of time by express deployment" line of the build log is express
(the environment was prepared earlier), while "is preparing" is not finished, so unknown. Each
container event carries the config version of its update, and only the entity's latest preparation
or image build counts, so an image built by an earlier, failed attempt does not hide an express
preparation after it. Both are needed. We assume (no Databricks page says so) that express which
falls back to an image build may still have written an express line to the build log first, so a
finished image build in the events wins over the build log, and when either could not be read the
path is unknown, never express from the build log alone.

The events API returns an endpoint's events in no particular order and, in the responses seen, with
no page token. Pure: 05_ fetches the log and the events, this module only reads them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

EXPRESS, STANDARD, UNKNOWN = "express", "standard", "unknown"
# The endpoint tag 05_ sets to the path of the endpoint's current served entity.
DEPLOYMENT_TAG = "deployment"

CONTAINER_EVENT = "SERVED_ENTITY_CONTAINER_EVENT"
EXPRESS_PREPARED_LOG = "prepared ahead of time by express deployment"   # in the build log, any case
EXPRESS_PREPARING_LOG = "express deployment is preparing"                # in the build log, any case
EXPRESS_STARTED = "Model artifact preparation started"
EXPRESS_DONE = "Model artifact preparation completed successfully"
IMAGE_STARTED = "Container image creation initiated"
IMAGE_DONE = "Container image creation finished successfully"


class DeploymentPath(NamedTuple):
    path: str            # EXPRESS, STANDARD or UNKNOWN
    line: str            # one line for the notebook output
    reason: str = ""     # for UNKNOWN: why, in words for a warning ("could not read ..." or what was unclear)


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


def _is_image(event: Mapping[str, Any]) -> bool:
    return event.get("message") in (IMAGE_STARTED, IMAGE_DONE)


def _is_express(event: Mapping[str, Any]) -> bool:
    return event.get("message") in (EXPRESS_STARTED, EXPRESS_DONE)


def _latest_build(events: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]] | None:
    """Of one served entity's container events, those of its latest preparation or image build, or
    None when the events cannot say which one came last.

    Each event carries the config version of the update it belongs to (seen live), and a later build
    replaces an earlier one: an image built by an earlier, failed attempt does not describe an express
    preparation after it. So the events of the latest config version that has any count; when no
    event carries a config version, all of them. Express and image events can still meet in the
    events that count: one attempt where express falls back (its express events all before its image
    build, by Model Serving's timestamps) is a build of the image; anything else (an express event
    after an image event, missing timestamps, or both kinds where only some events carry a config
    version, so an unversioned one cannot be placed) is two attempts in an order the events do not
    establish: None."""
    versions = [e["config_version"] for e in events if isinstance(e.get("config_version"), int)]
    if versions:
        unplaced = [e for e in events if not isinstance(e.get("config_version"), int)]
        latest = [e for e in events if e.get("config_version") == max(versions)]
        if unplaced and (any(map(_is_image, events)) and any(map(_is_express, events))):
            return None
    else:
        latest = list(events)
    image, express = [e for e in latest if _is_image(e)], [e for e in latest if _is_express(e)]
    if image and express:
        times = [e.get("timestamp") for e in image + express]
        if not all(isinstance(t, (int, float)) for t in times) or (
                max(e["timestamp"] for e in express) >= min(e["timestamp"] for e in image)):
            return None
    return latest


def classify(served_entity: str, build_log: str | None,
             events: Sequence[Mapping[str, Any]] | None) -> DeploymentPath:
    """The deployment path of ``served_entity`` from its build log (None, or anything but text, when
    it could not be read) and the endpoint's events (``events_of``'s list, any order; None when they
    could not be read). Unknown when either could not be read, and when the events cannot say which
    of the entity's builds came last (``_latest_build``): a wrong label is worse than unknown. Only
    the latest build counts, and only a finished one names the path: a finished container image build
    is standard (it wins over the express signals, on the assumption that express which falls back
    may already have written an express build-log line), a completed preparation is express, one
    that started and has not finished is unknown. With no build in the events, the build log's
    "prepared ahead of time by express deployment" line is express (an environment prepared earlier
    and reused); "is preparing" is not finished, so unknown."""
    def unknown(reason, line=None):
        return DeploymentPath(UNKNOWN, f"{served_entity}: UNKNOWN deployment path ({line or reason})", reason)

    unread = " and ".join(name for name, ok in (("its build log", isinstance(build_log, str)),
                                                ("the endpoint's events", _is_events(events))) if not ok)
    if unread:
        return unknown(f"could not read {unread}")
    container = [e for e in events if e.get("event_type") == CONTAINER_EVENT]
    # A build event that names no served entity (the API left it out) could be this entity's: unknown,
    # never a path from the events that do name it.
    if any(e.get("served_entity_name") is None and (_is_image(e) or _is_express(e)) for e in container):
        return unknown("its events hold a build that names no served entity")
    mine = _latest_build([e for e in container if e.get("served_entity_name") == served_entity])
    if mine is None:
        return unknown("its events hold an express preparation and a container image build whose order is unclear")
    messages = {e.get("message") for e in mine}
    if IMAGE_DONE in messages:
        took = _took(mine, IMAGE_STARTED, IMAGE_DONE)
        detail = f"container image built in {_minutes(took)}" if took is not None else "container image built"
        return DeploymentPath(STANDARD, f"{served_entity}: STANDARD deployment ({detail}; not express)")
    if IMAGE_STARTED in messages:
        return unknown("a container image build started and has not finished (still running, or it failed)")
    if EXPRESS_DONE in messages:
        took = _took(mine, EXPRESS_STARTED, EXPRESS_DONE)
        detail = f"environment prepared in {_minutes(took)}" if took is not None else "environment prepared"
        return DeploymentPath(EXPRESS, f"{served_entity}: EXPRESS deployment ({detail}; no container image build)")
    if EXPRESS_STARTED in messages:
        return unknown("its express preparation started and has not finished (still running, or it stopped)")
    log = build_log.lower()
    if EXPRESS_PREPARED_LOG in log:
        return DeploymentPath(EXPRESS, f"{served_entity}: EXPRESS deployment (environment prepared earlier and "
                                       "reused; no container image build)")
    if EXPRESS_PREPARING_LOG in log:
        return unknown("its build log says express deployment is still preparing, and its events hold no build")
    return unknown("its build log and events named neither path",
                   "neither its build log nor the endpoint's events name express deployment or a container image build")
