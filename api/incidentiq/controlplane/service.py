"""A mock of the infrastructure API that approved actions are sent to.

It records operations and changes nothing real. What it reproduces is how a
remote call fails: refused before applying, or applied with the response lost.

Contract, as in Stripe-style mutating APIs: every request carries an
Idempotency-Key. The first request per key is applied and its response stored;
later requests with that key get the stored response; the same key with a
different body is a 422.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import psycopg

OPERATIONS = frozenset({"restart_service", "scale_service", "rollback_deploy"})


class Fault(StrEnum):
    UNAVAILABLE = "unavailable"        # refused before applying: safe to retry
    LOST_RESPONSE = "lost_response"    # applied, then the response never arrived


class LostResponse(Exception):
    """The operation was applied but the caller will not see the response."""

    def __init__(self, response: dict[str, Any]):
        super().__init__("response lost after the operation was applied")
        self.response = response


@dataclass
class Reply:
    status: int
    body: dict[str, Any]
    replayed: bool = False


@dataclass
class FaultPlan:
    """Which request fails, and how.

    Decided per (idempotency key, request number), so a retry is a different
    draw and can succeed, and a given key fails the same way on every run.
    The request counter lives here rather than in the database because it
    describes the network between caller and control plane, not the ledger.
    """

    decide: Callable[[str, int], Fault | None] = lambda _key, _n: None
    _seen: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def next(self, key: str) -> Fault | None:
        with self._lock:
            self._seen[key] += 1
            n = self._seen[key]
        return self.decide(key, n)

    @classmethod
    def seeded(cls, rate: float, seed: int) -> FaultPlan:
        def decide(key: str, n: int) -> Fault | None:
            if rate <= 0:
                return None
            roll = _roll(f"{seed}:{key}:{n}")
            if roll >= rate:
                return None
            # Half of the faults are the dangerous kind on purpose: an
            # unavailable control plane is easy, a lost response is the case
            # that double-applies without idempotency.
            return Fault.LOST_RESPONSE if _roll(f"{seed}:{key}:{n}#mode") < 0.5 \
                else Fault.UNAVAILABLE
        return cls(decide=decide)

    @classmethod
    def scripted(cls, faults: dict[str, list[Fault | None]]) -> FaultPlan:
        """Exact faults per key, in request order. Used by tests."""
        def decide(key: str, n: int) -> Fault | None:
            script = faults.get(key) or []
            return script[n - 1] if n <= len(script) else None
        return cls(decide=decide)


def _roll(key: str) -> float:
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def request_hash(operation: str, params: dict[str, Any]) -> str:
    canonical = json.dumps({"operation": operation, "params": params},
                           sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


class ControlPlane:
    """Applies operations at most once per idempotency key."""

    def __init__(self, conn: psycopg.Connection, faults: FaultPlan | None = None):
        self.conn = conn
        self.faults = faults or FaultPlan()

    def submit(self, idempotency_key: str | None, operation: str,
               params: dict[str, Any]) -> Reply:
        if not idempotency_key:
            return Reply(400, {"error": "Idempotency-Key header is required"})
        if operation not in OPERATIONS:
            return Reply(422, {"error": f"unknown operation {operation!r}"})

        fault = self.faults.next(idempotency_key)
        if fault is Fault.UNAVAILABLE:
            return Reply(503, {"error": "control plane unavailable"})

        digest = request_hash(operation, params)
        stored = self._stored(idempotency_key)
        if stored is not None:
            return self._replay(stored, digest)

        response = {
            "operation_id": f"op-{uuid.uuid4().hex[:12]}",
            "operation": operation,
            "params": params,
            "status": "applied",
            "note": "Mock control plane. No live system was modified.",
        }
        inserted = self.conn.execute(
            "INSERT INTO control_plane_operations "
            "(idempotency_key, operation, request_hash, params, response) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (idempotency_key) DO NOTHING RETURNING id",
            (idempotency_key, operation, digest, json.dumps(params, default=str),
             json.dumps(response)),
        ).fetchone()
        if inserted is None:
            # Two requests with the same key raced and the other one won. Its
            # response is the one that was applied, so that is the one to return.
            return self._replay(self._stored(idempotency_key), digest)

        if fault is Fault.LOST_RESPONSE:
            raise LostResponse(response)
        return Reply(201, response)

    def _stored(self, key: str) -> tuple[str, dict[str, Any]] | None:
        return self.conn.execute(
            "SELECT request_hash, response FROM control_plane_operations "
            "WHERE idempotency_key = %s",
            (key,),
        ).fetchone()

    @staticmethod
    def _replay(stored: tuple[str, dict[str, Any]], digest: str) -> Reply:
        stored_hash, response = stored
        if stored_hash != digest:
            return Reply(422, {"error": "Idempotency-Key was already used for a "
                                        "different request"})
        return Reply(200, response, replayed=True)
