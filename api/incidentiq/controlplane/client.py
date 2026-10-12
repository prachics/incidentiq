"""Sends an approved action to the control plane, with retries.

Every attempt carries the same Idempotency-Key, which is what makes retrying
safe. Retried: connection errors, timeouts, 5xx and 429. Not retried: any other
4xx, since the request itself is wrong.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import psycopg

from incidentiq.config import Settings, get_settings
from incidentiq.controlplane.service import ControlPlane, FaultPlan, LostResponse

log = logging.getLogger(__name__)


class ControlPlaneError(Exception):
    def __init__(self, message: str, *, attempts: int, retryable: bool,
                 errors: list[str]):
        super().__init__(message)
        self.attempts = attempts
        self.retryable = retryable
        self.errors = errors


@dataclass
class Submission:
    body: dict[str, Any]
    attempts: int
    replayed: bool
    errors: list[str] = field(default_factory=list)   # one per failed attempt


def _retryable_status(status: int) -> bool:
    return status == 429 or status >= 500


class ControlPlaneClient:
    def __init__(self, http: httpx.Client, *, max_attempts: int = 4,
                 backoff_s: float = 0.25, sleep: Callable[[float], None] = time.sleep):
        self.http = http
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        self.sleep = sleep

    def close(self) -> None:
        self.http.close()

    @classmethod
    def from_settings(cls, conn: psycopg.Connection,
                      settings: Settings | None = None,
                      faults: FaultPlan | None = None) -> ControlPlaneClient:
        s = settings or get_settings()
        if s.control_plane_url:
            http = httpx.Client(base_url=s.control_plane_url,
                                timeout=s.control_plane_timeout_s)
        else:
            plane = ControlPlane(
                conn, faults or FaultPlan.seeded(s.control_plane_fault_rate,
                                                 s.failure_injection_seed))
            http = in_process_http(plane)
        return cls(http, max_attempts=s.action_max_attempts, backoff_s=s.action_backoff_s)

    def submit(self, operation: str, params: dict[str, Any], *,
               idempotency_key: str) -> Submission:
        errors: list[str] = []
        for attempt in range(1, self.max_attempts + 1):
            try:
                r = self.http.post(
                    "/v1/operations",
                    json={"operation": operation, "params": params},
                    headers={"Idempotency-Key": idempotency_key},
                )
            except httpx.TransportError as exc:
                errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            else:
                if r.status_code < 300:
                    return Submission(
                        body=r.json(), attempts=attempt, errors=errors,
                        replayed=r.headers.get("Idempotent-Replay") == "true",
                    )
                detail = _error_detail(r)
                errors.append(f"attempt {attempt}: HTTP {r.status_code}: {detail}")
                if not _retryable_status(r.status_code):
                    raise ControlPlaneError(
                        f"control plane refused {operation}: HTTP {r.status_code} {detail}",
                        attempts=attempt, retryable=False, errors=errors)

            if attempt < self.max_attempts:
                delay = self.backoff_s * 2 ** (attempt - 1)
                log.warning("control plane attempt %d for %s failed; retrying in %.2fs",
                            attempt, idempotency_key, delay)
                self.sleep(delay)

        raise ControlPlaneError(
            f"control plane did not accept {operation} after {self.max_attempts} attempts",
            attempts=self.max_attempts, retryable=True, errors=errors)


def _error_detail(r: httpx.Response) -> str:
    try:
        return str(r.json().get("error", r.text))
    except (json.JSONDecodeError, AttributeError):
        return r.text[:200]


def in_process_http(plane: ControlPlane) -> httpx.Client:
    """An httpx client whose requests are answered by `plane` directly.

    The same client code path as the networked service, minus the network. A
    lost response becomes a read timeout, which is what the caller would see.
    """
    def handle(request: httpx.Request) -> httpx.Response:
        if request.method != "POST" or request.url.path != "/v1/operations":
            return httpx.Response(404, json={"error": "not found"})
        try:
            payload = json.loads(request.content or b"{}")
        except json.JSONDecodeError:
            return httpx.Response(400, json={"error": "body is not JSON"})
        try:
            reply = plane.submit(request.headers.get("Idempotency-Key"),
                                 payload.get("operation", ""), payload.get("params") or {})
        except LostResponse as exc:
            raise httpx.ReadTimeout("no response from control plane",
                                    request=request) from exc
        headers = {"Idempotent-Replay": "true"} if reply.replayed else {}
        return httpx.Response(reply.status, json=reply.body, headers=headers)

    return httpx.Client(transport=httpx.MockTransport(handle),
                        base_url="http://control-plane.local")
