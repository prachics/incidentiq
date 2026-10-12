"""Runs investigations off the request path, one at a time.

At most one job per investigation is queued or running; callers claim the slot
with `try_reserve` before writing anything. A resume requested by a decision
while the investigation's job is still finishing is kept as a follow-up and
runs when that job ends. Both guarantees are per process (docs/DESIGN.md).
A crashed job leaves the investigation `failed` with the reason; its
checkpoints remain, and POST /investigations/{id}/resume continues it.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait

import psycopg
from opentelemetry import context as otel_context

from incidentiq.agent.graph import InvestigationRunner
from incidentiq.config import get_settings
from incidentiq.telemetry import INVESTIGATIONS_SETTLED, link_to, span, traceparent

log = logging.getLogger(__name__)


class Worker:
    def __init__(self, runner_factory: Callable[[], InvestigationRunner] | None = None):
        self._factory = runner_factory or InvestigationRunner
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="investigation")
        self._futures: list[Future] = []
        self._active: set[str] = set()
        self._follow_ups: dict[str, tuple[Callable[[InvestigationRunner], object],
                                          otel_context.Context]] = {}
        self._lock = threading.Lock()

    def is_active(self, investigation_id: str) -> bool:
        with self._lock:
            return investigation_id in self._active

    def start(self, investigation_id: str, query: str, *, reserved: bool = False) -> Future | None:
        return self._submit(investigation_id, lambda r: r.start(investigation_id, query),
                            reserved=reserved)

    def resume(self, investigation_id: str, *, reserved: bool = False) -> Future | None:
        return self._submit(investigation_id, lambda r: r.resume(investigation_id),
                            reserved=reserved)

    def try_reserve(self, investigation_id: str) -> bool:
        """Claim the investigation's slot before changing anything in the database,
        so two callers cannot both decide to run it. Pass `reserved=True` to
        `start`/`resume` to use the claim, or `release` it if nothing will run."""
        with self._lock:
            if investigation_id in self._active:
                return False
            self._active.add(investigation_id)
            return True

    def release(self, investigation_id: str) -> None:
        with self._lock:
            self._active.discard(investigation_id)

    def resume_after_decision(self, investigation_id: str, approval_id: str) -> None:
        """Resume for this decision now, or right after the job that is still finishing."""
        def job(runner: InvestigationRunner) -> object:
            return runner.resume_for_decision(investigation_id, approval_id)

        with self._lock:
            if investigation_id in self._active:
                # Keep the decision request's trace for when the follow-up runs.
                self._follow_ups[investigation_id] = (job, otel_context.get_current())
                return
        self._submit(investigation_id, job)

    def drain(self, timeout: float | None = None) -> None:
        """Block until the worker is idle, follow-ups included. For tests and shutdown."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            wait(list(self._futures),
                 timeout=None if deadline is None else max(0.0, deadline - time.monotonic()))
            with self._lock:
                idle = not self._active and not self._follow_ups
            if idle or (deadline is not None and time.monotonic() >= deadline):
                return
            time.sleep(0.01)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _submit(self, investigation_id: str, job: Callable[[InvestigationRunner], object],
                *, reserved: bool = False,
                parent: otel_context.Context | None = None) -> Future | None:
        """Queue a job, or return None if this investigation already has one."""
        if not reserved and not self.try_reserve(investigation_id):
            return None

        # Thread pools do not carry the submitter's context, so the trace of
        # the request that queued this job is captured here and attached there.
        parent = parent if parent is not None else otel_context.get_current()

        def run() -> None:
            token = otel_context.attach(parent)
            try:
                # Each job is part of the trace of the request that queued it
                # (a start, a decision, a resume) and links to the investigation's
                # first job, so the segments of one investigation can be followed.
                first = _trace_context(investigation_id)
                with span("investigation job", links=link_to(first),
                          **{"incidentiq.investigation_id": investigation_id}) as current:
                    if first is None:
                        _save_trace_context(investigation_id, traceparent(current))
                    with self._factory() as runner:
                        job(runner)
            except Exception as exc:  # noqa: BLE001 - a crashed job must leave a reason
                log.exception("investigation %s crashed", investigation_id)
                INVESTIGATIONS_SETTLED.labels("failed").inc()
                _mark_failed(investigation_id, f"{type(exc).__name__}: {exc}")
            finally:
                otel_context.detach(token)
                with self._lock:
                    self._active.discard(investigation_id)
                    follow_up = self._follow_ups.pop(investigation_id, None)
                if follow_up is not None:
                    follow_job, follow_context = follow_up
                    self._submit(investigation_id, follow_job, parent=follow_context)

        future = self._pool.submit(run)
        self._futures = [f for f in self._futures if not f.done()] + [future]
        return future


def _mark_failed(investigation_id: str, reason: str) -> None:
    try:
        with psycopg.connect(get_settings().database_url, autocommit=True) as conn:
            conn.execute(
                "UPDATE investigations SET status = 'failed', failure_reason = %s, "
                "updated_at = now() WHERE id = %s",
                (reason[:2000], investigation_id),
            )
    except psycopg.Error:
        log.exception("could not record the failure of %s", investigation_id)


def _trace_context(investigation_id: str) -> str | None:
    try:
        with psycopg.connect(get_settings().database_url, autocommit=True) as conn:
            row = conn.execute("SELECT trace_context FROM investigations WHERE id = %s",
                               (investigation_id,)).fetchone()
    except psycopg.Error:
        return None
    return row[0] if row else None


def _save_trace_context(investigation_id: str, value: str) -> None:
    try:
        with psycopg.connect(get_settings().database_url, autocommit=True) as conn:
            conn.execute("UPDATE investigations SET trace_context = %s "
                         "WHERE id = %s AND trace_context IS NULL", (value, investigation_id))
    except psycopg.Error:
        log.warning("could not store the trace context of %s", investigation_id)
