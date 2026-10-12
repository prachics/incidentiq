"""Write tools. Every one of these is approval-gated.

Each tool validates its request against the mock infrastructure; `execute_action`
sends it to the mock control plane (`incidentiq.controlplane`) at most once per
approval. Nothing here changes a real system.

The gate is structural: `requires_approval` makes the graph interrupt, and
`execute_action` refuses without an approval row for this investigation and
tool whose decision is 'approved' or 'modified'.
"""

from __future__ import annotations

import json
from typing import Any

import psycopg
from pydantic import BaseModel, Field, field_validator

from incidentiq.controlplane import ControlPlaneClient, ControlPlaneError
from incidentiq.telemetry import ACTION_ATTEMPTS, ACTIONS
from incidentiq.tools.base import Tool, ToolError, ToolStatus, registry

APPROVED_DECISIONS = ("approved", "modified")


class RestartServiceArgs(BaseModel):
    service: str = Field(description="Exact service name.")
    instance_id: str | None = Field(
        default=None,
        description=(
            "A specific instance to restart, or null for a rolling restart of all "
            "instances. Prefer a single instance when the problem is isolated."
        ),
    )
    reason: str = Field(
        min_length=10,
        description="Why this restart is expected to help. Shown to the approver.",
    )


class RestartService(Tool):
    name = "restart_service"
    description = (
        "Restart service instances. Disruptive: drops in-flight requests. Appropriate "
        "for a saturated pool or a deadlock that a restart clears; NOT appropriate when "
        "the service is merely waiting on a broken dependency, since restarting a "
        "victim does not fix the cause. Requires human approval."
    )
    args_model = RestartServiceArgs
    requires_approval = True

    def run(self, conn: psycopg.Connection, args: RestartServiceArgs) -> dict[str, Any]:
        row = conn.execute(
            "SELECT replica_count FROM services WHERE name = %s", (args.service,)
        ).fetchone()
        if not row:
            raise ToolError(f"unknown service {args.service!r}", status=ToolStatus.MALFORMED)

        if args.instance_id:
            inst = conn.execute(
                "SELECT id FROM service_instances WHERE id = %s AND service = %s",
                (args.instance_id, args.service),
            ).fetchone()
            if not inst:
                raise ToolError(
                    f"instance {args.instance_id!r} does not belong to {args.service}",
                    status=ToolStatus.MALFORMED,
                )
            targets = [args.instance_id]
        else:
            targets = [
                r[0] for r in conn.execute(
                    "SELECT id FROM service_instances WHERE service = %s", (args.service,)
                ).fetchall()
            ]

        return {
            "action": "restart_service",
            "service": args.service,
            "instances_targeted": targets,
            "instance_count": len(targets),
            "rolling": args.instance_id is None,
            "reason": args.reason,
            "note": "Validated against the mock infrastructure. No live system is modified.",
        }


class ScaleServiceArgs(BaseModel):
    service: str = Field(description="Exact service name.")
    replica_count: int = Field(
        ge=1, le=50, description="Target replica count. Must be between 1 and 50."
    )
    reason: str = Field(min_length=10, description="Why scaling addresses the problem.")


class ScaleService(Tool):
    name = "scale_service"
    description = (
        "Change a service's replica count. Appropriate when the bottleneck is genuinely "
        "capacity. Confirm the service is not simply waiting on a downstream dependency "
        "first - scaling a victim increases load on whatever is actually broken and "
        "makes the incident worse. Requires human approval."
    )
    args_model = ScaleServiceArgs
    requires_approval = True

    def run(self, conn: psycopg.Connection, args: ScaleServiceArgs) -> dict[str, Any]:
        row = conn.execute(
            "SELECT replica_count FROM services WHERE name = %s", (args.service,)
        ).fetchone()
        if not row:
            raise ToolError(f"unknown service {args.service!r}", status=ToolStatus.MALFORMED)
        current = row[0]
        return {
            "action": "scale_service",
            "service": args.service,
            "current_replicas": current,
            "target_replicas": args.replica_count,
            "delta": args.replica_count - current,
            "direction": "up" if args.replica_count > current else
                         ("down" if args.replica_count < current else "no change"),
            "reason": args.reason,
            "note": "Validated against the mock infrastructure. No live system is modified.",
        }


class RollbackDeployArgs(BaseModel):
    service: str = Field(description="Exact service name.")
    target_version: str = Field(
        description=(
            "Version to roll back to. Must be a version that appears in "
            "get_recent_deploys output for this service - call it first if you "
            "have not. Do not invent or guess a version string."
        )
    )
    reason: str = Field(min_length=10, description="Why rollback is the right action.")

    @field_validator("target_version")
    @classmethod
    def _looks_like_a_version(cls, v: str) -> str:
        if not v.startswith("v"):
            # Deliberately no example version here. The previous message named
            # 'v1.2.3', that message reached the model through the scratchpad
            # after a failed attempt, and the model used it as the next
            # attempt's value - on a service whose deploys were all v2.27.x. A
            # concrete fake value in an error message is a value the model can
            # copy.
            raise ValueError(
                f"target_version must start with 'v' and name a version from this "
                f"service's deploy history; got {v!r}. Call get_recent_deploys to "
                f"see the real versions."
            )
        return v


class RollbackDeploy(Tool):
    name = "rollback_deploy"
    description = (
        "Roll a service back to a previous version. Appropriate when symptoms began "
        "sharply and coincide with a release. Verify the target version predates the "
        "onset of symptoms - rolling back to a version deployed after onset does "
        "nothing. Requires human approval."
    )
    args_model = RollbackDeployArgs
    requires_approval = True

    def run(self, conn: psycopg.Connection, args: RollbackDeployArgs) -> dict[str, Any]:
        current = conn.execute(
            "SELECT version, deployed_at FROM deploys WHERE service = %s "
            "ORDER BY deployed_at DESC LIMIT 1",
            (args.service,),
        ).fetchone()
        if not current:
            raise ToolError(
                f"no deploy history for {args.service!r}", status=ToolStatus.FAILED
            )

        target = conn.execute(
            "SELECT id, deployed_at FROM deploys WHERE service = %s AND version = %s "
            "ORDER BY deployed_at DESC LIMIT 1",
            (args.service, args.target_version),
        ).fetchone()
        if not target:
            known = [
                r[0] for r in conn.execute(
                    "SELECT DISTINCT version FROM deploys WHERE service = %s "
                    "ORDER BY version DESC LIMIT 5",
                    (args.service,),
                ).fetchall()
            ]
            raise ToolError(
                f"{args.target_version!r} was never deployed to {args.service}. "
                f"Known versions: {', '.join(known)}",
                status=ToolStatus.MALFORMED,
            )

        return {
            "action": "rollback_deploy",
            "service": args.service,
            "current_version": current[0],
            "target_version": args.target_version,
            "target_deployed_at": target[1].isoformat(),
            "target_deploy_id": target[0],
            "reason": args.reason,
            "note": "Validated against the mock infrastructure. No live system is modified.",
        }


def execute_action(
    conn: psycopg.Connection,
    tool: Tool,
    raw_args: dict[str, Any],
    *,
    investigation_id: str,
    approval_id: str,
    control_plane: ControlPlaneClient | None = None,
) -> dict[str, Any]:
    """Run an approved write tool, at most once per approval, and record it.

    Refuses unless the named approval exists, belongs to this investigation, and
    carries an approving decision. This is the structural half of the gate: the
    graph interrupts before acting, and this refuses to act without the row
    that interrupt produced. Neither alone is sufficient.

    The approval id is the idempotency key. One approval authorises one
    execution, so a key per approval says exactly that: however many times this
    function runs for it - a retry after a lost response, or `act` re-running
    after a crash that happened before its checkpoint was written - the control
    plane applies the action once.
    """
    row = conn.execute(
        "SELECT decision, proposed_tool, proposed_arguments, modified_arguments FROM approvals "
        "WHERE id = %s AND investigation_id = %s",
        (approval_id, investigation_id),
    ).fetchone()

    if row is None:
        raise ToolError(
            f"no approval {approval_id!r} for investigation {investigation_id!r}",
            status=ToolStatus.FAILED,
        )
    decision, proposed_tool, proposed, modified = row
    if decision not in APPROVED_DECISIONS:
        raise ToolError(
            f"approval {approval_id} is {decision or 'still pending'}, not approved",
            status=ToolStatus.FAILED,
        )
    if proposed_tool != tool.name:
        raise ToolError(
            f"approval {approval_id} authorises {proposed_tool!r}, not {tool.name!r}",
            status=ToolStatus.FAILED,
        )

    # What runs is what was approved: the approver's edit if there is one,
    # otherwise the arguments shown on the approval. Using the agent's original
    # would silently discard an edit; accepting different arguments from the
    # caller would run something nobody approved.
    if raw_args and raw_args != proposed:
        raise ToolError(
            f"arguments for {approval_id} differ from the ones that were approved",
            status=ToolStatus.FAILED,
        )
    effective = modified if modified else proposed

    done = conn.execute(
        "SELECT result FROM actions WHERE idempotency_key = %s AND outcome = 'executed'",
        (approval_id,),
    ).fetchone()
    if done:
        ACTIONS.labels(tool.name, "already_executed").inc()
        return {**done[0], "execution": {**done[0]["execution"], "already_executed": True}}

    try:
        args = tool.validate_args(effective)
        request = tool.run(conn, args)
    except ToolError:
        # Arguments that fail validation, or name something that does not
        # exist, never reach the control plane; counted so they are visible.
        ACTIONS.labels(tool.name, "invalid_arguments").inc()
        raise

    client = control_plane or ControlPlaneClient.from_settings(conn)
    try:
        sent = client.submit(tool.name, args.model_dump(mode="json"),
                             idempotency_key=approval_id)
    except ControlPlaneError as exc:
        _record(conn, investigation_id, approval_id, tool.name, effective,
                outcome="failed", attempts=exc.attempts, error="; ".join(exc.errors))
        ACTIONS.labels(tool.name, "failed").inc()
        ACTION_ATTEMPTS.labels(tool.name).observe(exc.attempts)
        raise ToolError(str(exc), status=ToolStatus.FAILED) from exc
    finally:
        if control_plane is None:
            client.close()
    ACTIONS.labels(tool.name, "executed").inc()
    ACTION_ATTEMPTS.labels(tool.name).observe(sent.attempts)

    result = {
        **request,
        "operation_id": sent.body.get("operation_id"),
        "execution": {
            "attempts": sent.attempts,
            "replayed": sent.replayed,
            "already_executed": False,
            "failed_attempts": sent.errors,
        },
    }
    _record(conn, investigation_id, approval_id, tool.name, effective,
            outcome="executed", attempts=sent.attempts, result=result)
    return result


def _record(conn: psycopg.Connection, investigation_id: str, approval_id: str,
            tool_name: str, arguments: dict[str, Any], *, outcome: str, attempts: int,
            result: dict[str, Any] | None = None, error: str | None = None) -> None:
    """One row per approval. A later run updates it, adds its attempts and appends
    its failure details, so the row keeps the whole history of what it took."""
    conn.execute(
        "INSERT INTO actions (investigation_id, approval_id, tool_name, arguments, "
        "outcome, idempotency_key, attempts, result, error) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (idempotency_key) WHERE idempotency_key IS NOT NULL DO UPDATE SET "
        "outcome = EXCLUDED.outcome, attempts = actions.attempts + EXCLUDED.attempts, "
        "result = EXCLUDED.result, "
        "error = CASE WHEN EXCLUDED.error IS NULL THEN actions.error "
        "ELSE concat_ws(' | ', actions.error, EXCLUDED.error) END, executed_at = now()",
        (investigation_id, approval_id, tool_name, json.dumps(arguments, default=str),
         outcome, approval_id, attempts,
         json.dumps(result, default=str) if result is not None else None, error),
    )


for _tool in (RestartService(), ScaleService(), RollbackDeploy()):
    registry.register(_tool)
