"""Investigation endpoints: start, inspect, and resume after a crash."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

import psycopg
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from incidentiq.api.auth import Approver, Viewer
from incidentiq.api.deps import DB, Jobs

router = APIRouter(prefix="/investigations", tags=["investigations"])

Status = Literal["queued", "running", "awaiting_approval", "complete", "failed"]


class StartIn(BaseModel):
    query: str = Field(min_length=10, max_length=4000,
                       description="The on-call engineer's description of the problem.")
    incident_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9-]{2,63}$")


@router.post("", status_code=status.HTTP_202_ACCEPTED)
def start(body: StartIn, conn: DB, who: Viewer, jobs: Jobs) -> dict[str, Any]:
    iid = body.incident_id or f"IQ-{datetime.now(UTC):%Y%m%d}-{uuid.uuid4().hex[:6]}"
    # Claim the slot before the row is visible, so a resume cannot start this
    # investigation between the commit and the submit below.
    if not jobs.try_reserve(iid):
        raise HTTPException(status.HTTP_409_CONFLICT, f"investigation {iid} already exists")
    try:
        with conn.transaction():
            created = conn.execute(
                "INSERT INTO investigations (id, user_query, status) VALUES (%s, %s, 'queued') "
                "ON CONFLICT (id) DO NOTHING RETURNING id",
                (iid, body.query),
            ).fetchone()
            if created is None:
                raise HTTPException(status.HTTP_409_CONFLICT,
                                    f"investigation {iid} already exists")
            conn.execute(
                "INSERT INTO audit_log (investigation_id, event_type, actor, payload) "
                "VALUES (%s, 'investigation_requested', %s, %s)",
                (iid, who.user, json.dumps({"query": body.query})),
            )
    except BaseException:
        jobs.release(iid)
        raise
    jobs.start(iid, body.query, reserved=True)
    return {"id": iid, "status": "queued", "links": {"self": f"/investigations/{iid}"}}


@router.get("")
def list_investigations(
    conn: DB,
    _: Viewer,
    state: Annotated[Status | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[dict[str, Any]]:
    return conn.execute(
        "SELECT id, user_query, status, iteration, failure_reason, created_at, updated_at "
        "FROM investigations WHERE (%(s)s::text IS NULL OR status = %(s)s) "
        "ORDER BY created_at DESC LIMIT %(n)s",
        {"s": state, "n": limit},
    ).fetchall()


@router.get("/{investigation_id}")
def get_investigation(investigation_id: str, conn: DB, _: Viewer) -> dict[str, Any]:
    inv = _require(conn, investigation_id)
    inv["approvals"] = conn.execute(
        "SELECT id, proposed_tool, proposed_arguments, decision, decided_by, decided_at, "
        "rejection_reason, modified_arguments, created_at FROM approvals "
        "WHERE investigation_id = %s ORDER BY created_at",
        (investigation_id,),
    ).fetchall()
    inv["actions"] = conn.execute(
        "SELECT approval_id, tool_name, arguments, outcome, attempts, result, error, "
        "executed_at FROM actions WHERE investigation_id = %s ORDER BY id",
        (investigation_id,),
    ).fetchall()
    inv["tool_call_count"] = conn.execute(
        "SELECT count(*) AS n FROM tool_calls WHERE investigation_id = %s",
        (investigation_id,),
    ).fetchone()["n"]
    return inv


@router.get("/{investigation_id}/tool-calls")
def tool_calls(investigation_id: str, conn: DB, _: Viewer) -> list[dict[str, Any]]:
    _require(conn, investigation_id)
    return conn.execute(
        "SELECT tool_name, arguments, attempt, status, result, error, latency_ms, "
        "injected_failure, started_at FROM tool_calls WHERE investigation_id = %s ORDER BY id",
        (investigation_id,),
    ).fetchall()


@router.get("/{investigation_id}/audit")
def audit(investigation_id: str, conn: DB, _: Viewer) -> list[dict[str, Any]]:
    _require(conn, investigation_id)
    return conn.execute(
        "SELECT event_type, actor, payload, occurred_at FROM audit_log "
        "WHERE investigation_id = %s ORDER BY id",
        (investigation_id,),
    ).fetchall()


@router.post("/{investigation_id}/resume", status_code=status.HTTP_202_ACCEPTED)
def resume(investigation_id: str, conn: DB, who: Approver, jobs: Jobs) -> dict[str, Any]:
    """Continue an investigation whose worker died, or retry an approved action
    that failed. Both are the ordinary resume from the last checkpoint."""
    inv = _require(conn, investigation_id)
    if inv["status"] == "complete":
        raise HTTPException(status.HTTP_409_CONFLICT, f"{investigation_id} is complete")
    pending = conn.execute(
        "SELECT id FROM approvals WHERE investigation_id = %s AND decision IS NULL",
        (investigation_id,),
    ).fetchone()
    if pending and inv["status"] == "awaiting_approval":
        # The run is paused on this approval; the decision endpoint resumes it.
        # (After a crash the status is not awaiting_approval, and resuming is
        # how the run gets back to the same approval.)
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"approval {pending['id']} is waiting for a decision")
    if not jobs.try_reserve(investigation_id):
        # A second job would run against state the current one is still writing.
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"{investigation_id} is already queued or running")
    try:
        with conn.transaction():
            conn.execute(
                "INSERT INTO audit_log (investigation_id, event_type, actor, payload) "
                "VALUES (%s, 'investigation_resumed', %s, '{}')",
                (investigation_id, who.user),
            )
            conn.execute("UPDATE investigations SET status = 'queued', failure_reason = NULL, "
                         "updated_at = now() WHERE id = %s", (investigation_id,))
    except BaseException:
        jobs.release(investigation_id)
        raise
    if _has_checkpoints(conn, investigation_id):
        jobs.resume(investigation_id, reserved=True)
    else:
        # Queued but never picked up: there is nothing to resume, so start it.
        jobs.start(investigation_id, inv["user_query"], reserved=True)
    return {"id": investigation_id, "status": "queued"}


def _require(conn: psycopg.Connection, investigation_id: str) -> dict[str, Any]:
    row = conn.execute(
        "SELECT id, user_query, status, entities, iteration, final_summary, failure_reason, "
        "created_at, updated_at FROM investigations WHERE id = %s",
        (investigation_id,),
    ).fetchone()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no investigation {investigation_id!r}")
    return row


def _has_checkpoints(conn: psycopg.Connection, investigation_id: str) -> bool:
    try:
        row = conn.execute("SELECT count(*) AS n FROM checkpoints WHERE thread_id = %s",
                           (investigation_id,)).fetchone()
    except psycopg.errors.UndefinedTable:
        return False
    return row["n"] > 0
