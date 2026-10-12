"""Approval endpoints: list, inspect, decide.

A decision is recorded exactly once: the UPDATE matches only a row whose
decision is still NULL, so concurrent approvers get one success and 409s. The
decision, its audit event and the status change commit together, and the run
is resumed only after that commit. Edited arguments are validated against the
tool's schema before they are stored, and a rejection must carry the reason the
agent will revise from.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field, ValidationError, model_validator

from incidentiq.api.auth import Approver, Viewer
from incidentiq.api.deps import DB, Jobs
from incidentiq.telemetry import DECISIONS
from incidentiq.tools import registry

router = APIRouter(prefix="/approvals", tags=["approvals"])

SUMMARY_COLUMNS = ("id, investigation_id, proposed_tool, proposed_arguments, decision, "
                   "decided_by, decided_at, rejection_reason, modified_arguments, created_at")


class DecisionIn(BaseModel):
    decision: Literal["approve", "reject"]
    reason: str | None = Field(default=None, max_length=2000)
    # The full replacement arguments, when the approver edits them.
    arguments: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _consistent(self) -> DecisionIn:
        if self.decision == "reject":
            if not self.reason or len(self.reason.strip()) < 5:
                raise ValueError("a rejection needs a reason (5+ characters): the agent "
                                 "reads it to revise its diagnosis")
            if self.arguments is not None:
                raise ValueError("arguments can only be edited on an approval")
        return self


@router.get("")
def list_approvals(
    conn: DB,
    _: Viewer,
    state: Annotated[Literal["pending", "decided", "all"], Query(alias="status")] = "pending",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[dict[str, Any]]:
    where = {"pending": "WHERE decision IS NULL", "decided": "WHERE decision IS NOT NULL",
             "all": ""}[state]
    return conn.execute(
        f"SELECT {SUMMARY_COLUMNS} FROM approvals {where} ORDER BY created_at DESC LIMIT %s",
        (limit,),
    ).fetchall()


@router.get("/{approval_id}")
def get_approval(approval_id: str, conn: DB, _: Viewer) -> dict[str, Any]:
    row = conn.execute(
        f"SELECT {SUMMARY_COLUMNS}, reasoning, evidence FROM approvals WHERE id = %s",
        (approval_id,),
    ).fetchone()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no approval {approval_id!r}")
    tool = registry.get(row["proposed_tool"])
    # The schema lets a client render an edit form that the server will accept.
    row["arguments_schema"] = tool.args_model.model_json_schema() if tool else None
    return row


@router.post("/{approval_id}/decision", status_code=status.HTTP_202_ACCEPTED)
def decide(approval_id: str, body: DecisionIn, conn: DB, who: Approver,
           jobs: Jobs) -> dict[str, Any]:
    row = conn.execute(
        "SELECT investigation_id, proposed_tool, proposed_arguments, decision, decided_by, "
        "decided_at FROM approvals WHERE id = %s",
        (approval_id,),
    ).fetchone()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no approval {approval_id!r}")
    if row["decision"] is not None:
        raise _already_decided(approval_id, row)

    decision, modified = _resolve(body, row)
    # One transaction: a decision without its audit event would be permanent
    # (a retry gets 409) and unexplained.
    with conn.transaction():
        updated = conn.execute(
            "UPDATE approvals SET decision = %s, decided_by = %s, decided_at = now(), "
            "rejection_reason = %s, modified_arguments = %s "
            "WHERE id = %s AND decision IS NULL RETURNING decided_at",
            (decision, who.user, body.reason if decision == "rejected" else None,
             json.dumps(modified) if modified is not None else None, approval_id),
        ).fetchone()
        if updated is None:
            # Lost the race between the read above and this write.
            latest = conn.execute(
                "SELECT decision, decided_by, decided_at FROM approvals WHERE id = %s",
                (approval_id,),
            ).fetchone()
            raise _already_decided(approval_id, latest)
        conn.execute(
            "INSERT INTO audit_log (investigation_id, event_type, actor, payload) "
            "VALUES (%s, 'approval_decided', %s, %s)",
            (row["investigation_id"], who.user, json.dumps({
                "approval_id": approval_id, "decision": decision, "reason": body.reason,
                "modified_arguments": modified,
            })),
        )
        # Leave `awaiting_approval` now rather than when the worker starts: a
        # client polling for the result would otherwise stop watching.
        conn.execute("UPDATE investigations SET status = 'queued', updated_at = now() "
                     "WHERE id = %s", (row["investigation_id"],))
    DECISIONS.labels(decision).inc()
    jobs.resume_after_decision(row["investigation_id"], approval_id)
    return {
        "approval_id": approval_id,
        "investigation_id": row["investigation_id"],
        "decision": decision,
        "decided_by": who.user,
        "decided_at": updated["decided_at"],
        "status": "queued",
    }


def _resolve(body: DecisionIn, row: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    if body.decision == "reject":
        return "rejected", None
    if body.arguments is None or body.arguments == row["proposed_arguments"]:
        return "approved", None
    tool = registry.get(row["proposed_tool"])
    if tool is None:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"tool {row['proposed_tool']!r} is no longer registered")
    try:
        tool.args_model.model_validate(body.arguments)
    except ValidationError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            {"message": f"edited arguments are not valid for {tool.name}",
             "errors": [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]},
        ) from exc
    return "modified", body.arguments


def _already_decided(approval_id: str, row: dict[str, Any]) -> HTTPException:
    return HTTPException(
        status.HTTP_409_CONFLICT,
        f"approval {approval_id} was already {row['decision']} by {row['decided_by']} "
        f"at {row['decided_at']}",
    )
