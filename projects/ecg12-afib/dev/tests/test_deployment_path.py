"""common/deployment_path.py: which Model Serving path (express or container image build) a served entity
took, read from its build log and the endpoint's events. The messages are the ones Model Serving
returned for this demo's endpoints (build logs and the events API), cut to the lines that matter; the
timestamps are made up, with the durations and the order of the recorded ones."""
from collections import Counter
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


def _event(served, message, timestamp, event_type=dp.CONTAINER_EVENT):
    return {"endpoint_name": "ep", "event_type": event_type, "message": message, "timestamp": timestamp,
            "served_entity_name": served, "served_model_name": served}


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


def test_express_from_the_events_and_while_preparing():
    assert dp.classify("ep-2", "", EXPRESS_EVENTS).path == dp.EXPRESS          # an empty build log
    started = [e for e in EXPRESS_EVENTS if e["message"] != dp.EXPRESS_DONE]
    assert dp.classify("ep-2", PREPARING_LOG, started).line == (
        "ep-2: EXPRESS deployment (environment preparation not finished yet; no container image build)")


def test_standard_from_container_image_events_even_when_express_started():
    """A container image build means the standard path, also when express wrote its first log line
    and then fell back."""
    assert dp.classify("ep-4", IMAGE_LOG, IMAGE_EVENTS) == (
        dp.STANDARD, "ep-4: STANDARD deployment (container image built in 15m27s; not express)", "")
    assert dp.classify("ep-4", PREPARING_LOG, IMAGE_EVENTS).path == dp.STANDARD
    assert dp.classify("ep-3", "", IMAGE_EVENTS).line == "ep-3: STANDARD deployment (container image build; not express)"


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
            dp.UNKNOWN, f"ep-2: UNKNOWN deployment path (could not read {unread})", unread), (log, events)


def test_unknown_when_both_were_read_but_name_neither_path():
    for log, events in ((IMAGE_LOG, []), ("", EXPRESS_EVENTS)):
        found = dp.classify("ep-9", log, events)
        assert found.path == dp.UNKNOWN and found.unread == "", (log, events)
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
    """An event without a usable timestamp still counts as a signal, just without a time."""
    no_time = [{"event_type": dp.CONTAINER_EVENT, "served_entity_name": "ep-2", "message": dp.EXPRESS_STARTED},
               {"event_type": dp.CONTAINER_EVENT, "served_entity_name": "ep-2", "message": dp.EXPRESS_DONE,
                "timestamp": "soon"}]
    assert dp.classify("ep-2", "", no_time).line == (
        "ep-2: EXPRESS deployment (environment preparation not finished yet; no container image build)")


# A failed update, as Model Serving recorded it for a demo endpoint (messages and event types from the
# events API; the endpoint renamed, the timestamps made up, 53 ms apart as recorded).
ABORTED_EVENT = _event("ep-2", "Served entity creation aborted for served entity `ep-2`, config version 5, since "
                               "the update timed out.", ms("2026-11-21 10:00:00"), "SERVED_ENTITY_SERVICE_EVENT")
FAILED_EVENT = _event(None, "Endpoint update failed for endpoint `ep`, config version 5.",
                      ms("2026-11-21 10:00:00") + 53, "ENDPOINT_UPDATE_EVENT")


def test_update_failure_reads_only_failure_events_new_since_the_baseline():
    """The failure and abort messages among the events not in the baseline (read before the update was
    sent), oldest first, when they include the endpoint's ENDPOINT_UPDATE_EVENT "Endpoint update
    failed". Not for: a failure already in the baseline, whatever its time; another endpoint's (`ep`
    is not `ep_2`); the trigger text under another event type; an update still going; events that
    could not be read. No clock is compared: timestamps only order the messages."""
    progress = _event("ep-2", "Served entity `ep-2` entered DEPLOYMENT_CREATING state: Deploying",
                      ms("2026-11-21 09:55:01"), "SERVED_ENTITY_SERVICE_EVENT")
    before = dp.fingerprints([progress])
    events = [FAILED_EVENT, progress, ABORTED_EVENT]
    assert dp.update_failure(events, "ep", before) == [ABORTED_EVENT["message"], FAILED_EVENT["message"]]
    assert dp.update_failure(events, "ep", dp.fingerprints(events)) == []                 # all there before
    assert dp.update_failure(events, "ep", dp.fingerprints([FAILED_EVENT])) == []         # the failure was there
    future = {**FAILED_EVENT, "timestamp": FAILED_EVENT["timestamp"] + 10 ** 12}
    assert dp.update_failure([future], "ep", dp.fingerprints([future])) == []             # however late its time
    other = {**FAILED_EVENT, "message": "Endpoint update failed for endpoint `ep_2`, config version 5."}
    assert dp.update_failure([other, ABORTED_EVENT], "ep", before) == []
    assert dp.update_failure([other], "ep_2", before) == [other["message"]]
    wrong_type = {**FAILED_EVENT, "event_type": "SERVED_ENTITY_SERVICE_EVENT"}
    assert dp.update_failure([wrong_type, ABORTED_EVENT], "ep", before) == []
    assert dp.update_failure([progress, ABORTED_EVENT], "ep", before) == []               # no "update failed" yet
    assert dp.update_failure(None, "ep", before) == []
    assert dp.fingerprints(None) is None and dp.fingerprints([]) == Counter()
    assert dp.update_failure([{**FAILED_EVENT, "timestamp": "soon"}], "ep", before) == [FAILED_EVENT["message"]]


def test_update_failure_counts_events_with_the_same_fingerprint():
    """The baseline keeps how many events had each fingerprint: a new failure identical in every field
    to one already there (same type, time, message and config version) is still new, and as many
    copies as the baseline had are not."""
    before = dp.fingerprints([FAILED_EVENT])
    assert before[dp._fingerprint(FAILED_EVENT)] == 1
    assert dp.update_failure([FAILED_EVENT], "ep", before) == []
    assert dp.update_failure([FAILED_EVENT, dict(FAILED_EVENT)], "ep", before) == [FAILED_EVENT["message"]]
    assert dp.update_failure([FAILED_EVENT, dict(FAILED_EVENT)], "ep", dp.fingerprints([FAILED_EVENT] * 2)) == []
