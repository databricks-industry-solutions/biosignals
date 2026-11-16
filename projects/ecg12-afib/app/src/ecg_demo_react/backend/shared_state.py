"""State that every worker process of the app shares: the rate-limit counts of /api/analyze and
/api/warmup, and when each serving endpoint last answered (the warm-up cache).

app.yml runs uvicorn with --workers 2, and each worker is its own process, so in-memory counters
would let a caller through twice as often and miss the other worker's warm-up answers. Both live
in one SQLite file instead, which every worker opens (one connection per call, as the routes run
in a thread pool). BEGIN IMMEDIATE makes each count-and-record step atomic across processes.

Standard library only. The file is created at startup (an error there stops the app); if it
cannot be read or written later, the error is logged and the call is let through (no limit, no
cache hit): a broken counter file never blocks the demo.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (route TEXT NOT NULL, caller TEXT NOT NULL, at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS calls_by_route ON calls (route, at);
CREATE INDEX IF NOT EXISTS calls_by_caller ON calls (route, caller, at);
CREATE TABLE IF NOT EXISTS ready (endpoint TEXT PRIMARY KEY, at REAL NOT NULL, seconds REAL NOT NULL);
"""


@dataclass(frozen=True)
class Refusal:
    """Why a call was refused: the limit it hit, and how long until the oldest counted call
    leaves the window (then one more call fits)."""
    scope: Literal["caller", "total"]
    limit: int
    retry_after_s: float


@dataclass(frozen=True)
class ReadyAnswer:
    """An endpoint's last successful answer: how long ago, and how long that call took."""
    age_s: float
    seconds: float


class SharedState:
    def __init__(self, path: Path, clock: Callable[[], float] = time.time):
        self._path = path
        self._clock = clock     # wall-clock seconds: every process reads the same clock
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")     # readers never wait for the writer
            db.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self._path, timeout=5.0, isolation_level=None)   # autocommit; explicit BEGIN
        try:
            yield db
        finally:
            db.close()

    def admit(self, route: str, caller: str, *, per_caller: int, total: int, window_s: float) -> Refusal | None:
        """Count one call to ``route`` by ``caller`` if, over the last ``window_s`` seconds, the
        caller has made fewer than ``per_caller`` such calls and all callers fewer than ``total``;
        otherwise count nothing and return why (the caller's limit is checked first)."""
        try:
            with self._connect() as db:
                now = self._clock()
                db.execute("BEGIN IMMEDIATE")
                try:
                    # This route's calls only: another route may count over a longer window.
                    db.execute("DELETE FROM calls WHERE route = ? AND at <= ?", (route, now - window_s))
                    refusal = None
                    for scope, limit, sql, args in (
                        ("caller", per_caller, "SELECT count(*), min(at) FROM calls WHERE route = ? AND caller = ?",
                         (route, caller)),
                        ("total", total, "SELECT count(*), min(at) FROM calls WHERE route = ?", (route,)),
                    ):
                        count, oldest = db.execute(sql, args).fetchone()
                        if count >= limit:
                            refusal = Refusal(scope, limit, max(0.0, oldest + window_s - now))
                            break
                    if refusal is None:
                        db.execute("INSERT INTO calls VALUES (?, ?, ?)", (route, caller, now))
                    db.execute("COMMIT")
                except BaseException:
                    db.execute("ROLLBACK")
                    raise
                return refusal
        except sqlite3.Error:
            logger.exception("rate-limit store %s failed; letting the %s call through", self._path, route)
            return None

    def mark_ready(self, endpoint: str, seconds: float) -> None:
        """Record that ``endpoint`` just answered successfully, in ``seconds``."""
        try:
            with self._connect() as db:
                db.execute("INSERT INTO ready VALUES (?, ?, ?) ON CONFLICT (endpoint) DO UPDATE "
                           "SET at = excluded.at, seconds = excluded.seconds", (endpoint, self._clock(), seconds))
        except sqlite3.Error:
            logger.exception("warm-up cache %s failed; %s not recorded as ready", self._path, endpoint)

    def ready_within(self, endpoints: list[str], window_s: float) -> list[ReadyAnswer] | None:
        """Each endpoint's last successful answer, in order, if every one answered within the last
        ``window_s`` seconds; otherwise None."""
        try:
            with self._connect() as db:
                placeholders = ", ".join("?" * len(endpoints))
                rows = {endpoint: (at, seconds) for endpoint, at, seconds in db.execute(
                    f"SELECT endpoint, at, seconds FROM ready WHERE endpoint IN ({placeholders})", endpoints)}
        except sqlite3.Error:
            logger.exception("warm-up cache %s failed; pinging the endpoints", self._path)
            return None
        now = self._clock()
        answers = []
        for endpoint in endpoints:
            if endpoint not in rows or now - rows[endpoint][0] >= window_s:
                return None
            at, seconds = rows[endpoint]
            answers.append(ReadyAnswer(now - at, seconds))
        return answers
