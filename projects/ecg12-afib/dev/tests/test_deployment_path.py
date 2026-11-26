"""common/deployment_path.py: which Model Serving path (express or container image build) a served entity
took, read from its build log and the endpoint's events. The messages are the ones Model Serving
returned for this demo's endpoints (build logs and the events API), cut to the lines that matter; the
timestamps are made up, with the durations and the order of the recorded ones."""
from datetime import datetime

from common import deployment_path as dp

# Its first two sentences; the live log line goes on ("If it fall...") past what was recorded here.
PREPARING_LOG = ("Express deployment is preparing this model's environment and artifacts. See the endpoint "
                 "events for progress.")
# Its first line, which classify() reads; the live log goes on with build timings.
PREPARED_LOG = ("This environment was prepared ahead of time by express deployment and is shared by models with "
                "identical requirements.\n")
IMAGE_LOG = ("#18 0.382 Applying Databricks-managed pip constraints:\n"
             "#18 0.384 channels:\n#18 0.384 - conda-forge\n#18 0.384 dependencies:\n#18 0.384 - python=3.12.3\n")


def ms(when):
    """Epoch milliseconds, as the events API gives them, of a UTC time 'YYYY-MM-DD HH:MM:SS'."""
    return int(datetime.fromisoformat(when + "+00:00").timestamp() * 1000)


def _event(served, message, timestamp, event_type=dp.CONTAINER_EVENT, config_version=None):
    event = {"endpoint_name": "ep", "event_type": event_type, "message": message, "timestamp": timestamp,
             "served_entity_name": served, "served_model_name": served}
    return event if config_version is None else {**event, "config_version": config_version}


# The events API returns them in no particular order; the other entity's and other types' events
# must not count.
EXPRESS_EVENTS = [
    _event("ep-2", dp.EXPRESS_DONE, ms("2026-01-01 00:04:00")),
    {"endpoint_name": "ep", "event_type": "ENDPOINT_UPDATE_EVENT", "timestamp": ms("2025-12-31 23:59:45"),
     "message": "Endpoint updated by someone."},
    _event("ep-2", "Served entity created for served entity `ep-2`, config version 3.", ms("2026-01-01 00:04:01"),
           "SERVED_ENTITY_CREATION_EVENT"),
    _event("ep-1", dp.EXPRESS_STARTED, ms("2025-12-29 09:00:00")),
    _event("ep-2", dp.EXPRESS_STARTED, ms("2026-01-01 00:00:00")),
    _event("ep-1", dp.EXPRESS_DONE, ms("2025-12-29 09:02:42")),
]
IMAGE_EVENTS = [
    _event("ep-4", dp.IMAGE_DONE, ms("2025-12-01 01:15:27")),
    _event("ep-3", dp.IMAGE_STARTED, ms("2025-12-01 00:00:00")),
    _event("ep-4", dp.IMAGE_STARTED, ms("2025-12-01 01:00:00")),
]


def test_express_from_its_build_log_and_preparation_events():
    found = dp.classify("ep-2", PREPARED_LOG, EXPRESS_EVENTS)
    assert found == (dp.EXPRESS, "ep-2: EXPRESS deployment (environment prepared in 4m00s; no container image build)", "")
    assert dp.classify("ep-1", PREPARED_LOG, EXPRESS_EVENTS).line.endswith("(environment prepared in 2m42s; "
                                                                           "no container image build)")


def test_express_from_the_build_log_alone_when_the_environment_was_reused():
    """A config update that keeps the served entity reuses its environment: the events were read but
    hold none for it (or none left), and the build log still says express."""
    found = dp.classify("ep-2", PREPARED_LOG, [])
    assert found == (dp.EXPRESS, ("ep-2: EXPRESS deployment (environment prepared earlier and reused; "
                                  "no container image build)"), "")
    assert dp.classify("ep-2", PREPARED_LOG, IMAGE_EVENTS).path == dp.EXPRESS     # other entities' events


def test_express_from_the_completed_preparation_events():
    assert dp.classify("ep-2", "", EXPRESS_EVENTS).path == dp.EXPRESS          # an empty build log
    assert dp.classify("ep-2", PREPARING_LOG, EXPRESS_EVENTS).path == dp.EXPRESS


NOT_FINISHED = {
    "express": "its express preparation started and has not finished (still running, or it stopped)",
    "image": "a container image build started and has not finished (still running, or it failed)",
    "preparing": "its build log says express deployment is still preparing, and its events hold no build",
}


def test_a_build_that_started_and_has_not_finished_is_unknown():
    """Only a finished build names the path: an express preparation or a container image build that
    started and has not finished (still running, or it stopped) is unknown, with the reason, whatever
    the build log says; so is a build log that says express is still preparing with no build in the
    events. Before, the first counted as express and the second as standard."""
    started = [e for e in EXPRESS_EVENTS if e["message"] != dp.EXPRESS_DONE]
    for log in (PREPARING_LOG, PREPARED_LOG, ""):
        found = dp.classify("ep-2", log, started)
        assert found == (dp.UNKNOWN, f"ep-2: UNKNOWN deployment path ({NOT_FINISHED['express']})",
                         NOT_FINISHED["express"]), log
        assert dp.classify("ep-3", log, IMAGE_EVENTS).reason == NOT_FINISHED["image"], log
    # One attempt that started express, fell back and has not finished its image build yet.
    fallback = started + [_event("ep-2", dp.IMAGE_STARTED, ms("2026-01-01 00:10:00"))]
    assert dp.classify("ep-2", PREPARING_LOG, fallback).reason == NOT_FINISHED["image"]
    assert dp.classify("ep-2", PREPARING_LOG, []) == (
        dp.UNKNOWN, f"ep-2: UNKNOWN deployment path ({NOT_FINISHED['preparing']})", NOT_FINISHED["preparing"])


def test_standard_from_a_finished_container_image_build_whatever_the_build_log_says():
    """A finished container image build means the standard path, also with an express build-log line
    (assumed possible when express falls back: not documented, so the events decide)."""
    assert dp.classify("ep-4", IMAGE_LOG, IMAGE_EVENTS) == (
        dp.STANDARD, "ep-4: STANDARD deployment (container image built in 15m27s; not express)", "")
    assert dp.classify("ep-4", PREPARING_LOG, IMAGE_EVENTS).path == dp.STANDARD
    done_only = [e for e in IMAGE_EVENTS if e["message"] == dp.IMAGE_DONE]
    assert dp.classify("ep-4", "", done_only).line == "ep-4: STANDARD deployment (container image built; not express)"


def test_unknown_when_either_read_failed_even_with_an_express_build_log():
    """Without the events a container image build (express that fell back) could hide behind the
    express build-log line, and without the build log the evidence is incomplete: unknown, naming
    what could not be read."""
    for log, events, unread in (
            (PREPARED_LOG, None, "the endpoint's events"),
            (PREPARING_LOG, None, "the endpoint's events"),
            (None, EXPRESS_EVENTS, "its build log"),
            (None, IMAGE_EVENTS, "its build log"),
            (None, None, "its build log and the endpoint's events"),
            (42, [], "its build log"),
            (PREPARED_LOG, ["junk"], "the endpoint's events"),
            (PREPARED_LOG, {"events": []}, "the endpoint's events")):
        assert dp.classify("ep-2", log, events) == (
            dp.UNKNOWN, f"ep-2: UNKNOWN deployment path (could not read {unread})", f"could not read {unread}"), (
            log, events)


def test_unknown_when_both_were_read_but_name_neither_path():
    for log, events in ((IMAGE_LOG, []), ("", EXPRESS_EVENTS)):
        found = dp.classify("ep-9", log, events)
        assert found.path == dp.UNKNOWN and found.reason == "its build log and events named neither path", (log, events)
        assert found.line.startswith("ep-9: UNKNOWN deployment path (neither its build log"), (log, events)


def test_events_of_accepts_only_a_list_of_event_objects():
    """A valid response with no events is an empty list; anything else is None (could not be read),
    including a missing ``events`` key and a list with an item that is not an object."""
    event = _event("ep-2", dp.EXPRESS_DONE, 1)
    assert dp.events_of({"events": [event]}) == [event]
    assert dp.events_of({"events": []}) == []
    for response in ({}, {"events": None}, {"events": "x"}, {"events": [event, "not an event"]}, "<html>", None,
                     [event]):
        assert dp.events_of(response) is None, response


def test_events_without_a_usable_timestamp_still_count():
    """An event without a usable timestamp still counts as a signal, just without a time: a completed
    preparation is express, with no duration."""
    no_time = [{"event_type": dp.CONTAINER_EVENT, "served_entity_name": "ep-2", "message": dp.EXPRESS_STARTED},
               {"event_type": dp.CONTAINER_EVENT, "served_entity_name": "ep-2", "message": dp.EXPRESS_DONE,
                "timestamp": "soon"}]
    assert dp.classify("ep-2", "", no_time).line == "ep-2: EXPRESS deployment (environment prepared; no container image build)"

def test_only_the_latest_preparation_of_the_entity_counts():
    """Model Serving gives each container event the config version of the update it belongs to (seen
    live). An image build left over from an earlier attempt at the same served entity (a failed
    update, say) does not make a later express preparation STANDARD: only the container events of
    the latest config version that has any for the entity count. An update that adds none reuses the
    environment of the latest one that did."""
    stale_image = [_event("ep-2", dp.IMAGE_STARTED, ms("2026-01-01 00:00:00"), config_version=5),
                   _event("ep-2", dp.IMAGE_DONE, ms("2026-01-01 00:14:00"), config_version=5)]
    express = [_event("ep-2", dp.EXPRESS_STARTED, ms("2026-01-01 01:00:00"), config_version=6),
               _event("ep-2", dp.EXPRESS_DONE, ms("2026-01-01 01:04:00"), config_version=6)]
    assert dp.classify("ep-2", PREPARED_LOG, express + stale_image) == (
        dp.EXPRESS, "ep-2: EXPRESS deployment (environment prepared in 4m00s; no container image build)", "")
    # Newest first is the same: the events API gives them in no particular order.
    assert dp.classify("ep-2", PREPARING_LOG, stale_image[::-1] + express[::-1]).path == dp.EXPRESS
    # The other way round, an image build after an express preparation is STANDARD.
    later_image = [{**e, "config_version": 7} for e in stale_image]
    assert dp.classify("ep-2", PREPARED_LOG, express + later_image).path == dp.STANDARD
    # An update that keeps the entity and adds no container events: the latest build still decides.
    assert dp.classify("ep-2", IMAGE_LOG, stale_image).path == dp.STANDARD
    assert dp.classify("ep-2", PREPARED_LOG, express).line.endswith("(environment prepared in 4m00s; "
                                                                    "no container image build)")


def test_unknown_when_the_order_of_attempts_cannot_be_established():
    """When the events cannot say which attempt came last, the path is unknown, not a confident
    wrong label: a failed attempt's image build and a later express preparation that share one
    config version (Model Serving reusing it), or carry a config version on some events only. One
    attempt that starts express and then builds an image (express falling back) is still standard,
    since its express events all come before its image build."""
    image = [_event("ep-2", dp.IMAGE_STARTED, ms("2026-01-01 00:00:00"), config_version=5),
             _event("ep-2", dp.IMAGE_DONE, ms("2026-01-01 00:14:00"), config_version=5)]
    later_express = [_event("ep-2", dp.EXPRESS_STARTED, ms("2026-01-01 01:00:00"), config_version=5),
                     _event("ep-2", dp.EXPRESS_DONE, ms("2026-01-01 01:04:00"), config_version=5)]
    reason = "its events hold an express preparation and a container image build whose order is unclear"
    for events in (image + later_express,                                              # reused config version
                   image + [{k: v for k, v in e.items() if k != "config_version"} for e in later_express],
                   [{k: v for k, v in e.items() if k != "timestamp"} for e in image + later_express]):
        found = dp.classify("ep-2", PREPARED_LOG, events)
        assert found.path == dp.UNKNOWN, events
        assert found == (dp.UNKNOWN, f"ep-2: UNKNOWN deployment path ({reason})", reason), events
    fallback = [{**e, "timestamp": e["timestamp"] - 2 * 3600 * 1000} for e in later_express] + image
    assert dp.classify("ep-2", PREPARING_LOG, fallback).path == dp.STANDARD


def test_a_build_event_that_names_no_served_entity_makes_the_path_unknown():
    """A container event of a build that names no served entity (left out by the API) could be this
    entity's: the path is unknown, not express or standard from the events that do name it."""
    unnamed = {"event_type": dp.CONTAINER_EVENT, "message": dp.IMAGE_DONE, "timestamp": ms("2026-01-01 00:20:00")}
    reason = "its events hold a build that names no served entity"
    for events in (EXPRESS_EVENTS + [unnamed], [unnamed], [{**unnamed, "served_entity_name": None}]):
        assert dp.classify("ep-2", PREPARED_LOG, events) == (dp.UNKNOWN, f"ep-2: UNKNOWN deployment path ({reason})",
                                                             reason), events
    # Events of other types that name no served entity do not count.
    other = {"event_type": "ENDPOINT_UPDATE_EVENT", "message": "Endpoint updated.", "timestamp": 1}
    assert dp.classify("ep-2", PREPARED_LOG, EXPRESS_EVENTS + [other]).path == dp.EXPRESS


def test_without_config_versions_the_latest_attempt_by_start_time_counts():
    """When no container event carries a config version, the timestamps order the attempts: the
    latest start and the events from it on count. An older finished image build and a newer
    finished express preparation (all unversioned) is express (before, unknown although the
    timestamps make the order clear); a newer express start that has not finished is unknown, not
    finished. Missing timestamps, or two kinds of build starting at the same time, leave the order
    unclear: unknown."""
    image = [_event("ep-2", dp.IMAGE_STARTED, ms("2026-01-01 00:00:00")),
             _event("ep-2", dp.IMAGE_DONE, ms("2026-01-01 00:14:00"))]
    express = [_event("ep-2", dp.EXPRESS_STARTED, ms("2026-01-01 01:00:00")),
               _event("ep-2", dp.EXPRESS_DONE, ms("2026-01-01 01:04:00"))]
    assert dp.classify("ep-2", PREPARED_LOG, image + express) == (
        dp.EXPRESS, "ep-2: EXPRESS deployment (environment prepared in 4m00s; no container image build)", "")
    assert dp.classify("ep-2", PREPARING_LOG, express[::-1] + image[::-1]).path == dp.EXPRESS     # any order
    assert dp.classify("ep-2", PREPARED_LOG, image + express[:1]).reason == NOT_FINISHED["express"]
    # The other way round, an image build after an express preparation is standard.
    later_image = [{**e, "timestamp": e["timestamp"] + 2 * 3600 * 1000} for e in image]
    assert dp.classify("ep-2", PREPARED_LOG, express + later_image).path == dp.STANDARD
    reason = "its events hold an express preparation and a container image build whose order is unclear"
    for events in (image + [{k: v for k, v in e.items() if k != "timestamp"} for e in express],
                   image + [{**express[0], "timestamp": image[0]["timestamp"]}, express[1]]):     # same start
        assert dp.classify("ep-2", PREPARED_LOG, events) == (
            dp.UNKNOWN, f"ep-2: UNKNOWN deployment path ({reason})", reason), events


def test_a_completion_counts_only_at_or_after_the_latest_start():
    """A build's completion counts only when it comes at or after the latest start of that build: an
    older completed express preparation and a newer start that has not finished is that newer
    attempt, not finished (before, EXPRESS), with or without config versions; the same for a
    container image build (before, STANDARD). A retry that finished is timed from its own start.
    Done-only events (no start reported) still name the path, and so does one start with a
    completion that has no usable timestamp; several starts with one are unclear: unknown."""
    old = [_event("ep-2", dp.EXPRESS_STARTED, ms("2026-01-01 00:00:00")),
           _event("ep-2", dp.EXPRESS_DONE, ms("2026-01-01 00:04:00"))]
    newer = _event("ep-2", dp.EXPRESS_STARTED, ms("2026-01-01 01:00:00"))
    for version in (None, 6):
        events = [{**e, "config_version": version} if version else e for e in [*old, newer]]
        assert dp.classify("ep-2", PREPARED_LOG, events).reason == NOT_FINISHED["express"], version
    retry = [*old, newer, _event("ep-2", dp.EXPRESS_DONE, ms("2026-01-01 01:03:00"))]
    assert dp.classify("ep-2", PREPARED_LOG, retry).line.endswith("(environment prepared in 3m00s; "
                                                                  "no container image build)")
    image_again = [*IMAGE_EVENTS, _event("ep-4", dp.IMAGE_STARTED, ms("2025-12-02 00:00:00"))]
    assert dp.classify("ep-4", IMAGE_LOG, image_again).reason == NOT_FINISHED["image"]
    assert dp.classify("ep-2", "", old[1:]).line == ("ep-2: EXPRESS deployment (environment prepared; "
                                                     "no container image build)")          # done only
    untimed_done = {k: v for k, v in old[1].items() if k != "timestamp"}
    assert dp.classify("ep-2", "", [old[0], untimed_done]).path == dp.EXPRESS
    reason = "its events hold several starts of one build and a completion whose order is unclear"
    assert dp.classify("ep-2", "", [old[0], newer, untimed_done]) == (
        dp.UNKNOWN, f"ep-2: UNKNOWN deployment path ({reason})", reason)
