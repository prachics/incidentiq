"""Write tools. Every one of these is approval-gated.

Nothing here touches a real system - each records intent to the `actions` table.
That is deliberate and is the honest design for a portfolio system: the
interesting engineering is the approval gate, the audit trail, and the graph
interrupt, none of which are made more interesting by actually restarting
something.

The gate is enforced structurally rather than by convention. `requires_approval`
is a class attribute the graph reads to decide whether to interrupt, and
`execute_action` refuses to run without an approval row whose decision is
'approved'. A tool cannot be made to run by prompting the model differently.
"""

from __future__ import annotations

from typing import Any

import psycopg
from pydantic import BaseModel, Field, field_validator

from incidentiq.tools.base import Tool, ToolError, ToolStatus, registry


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
            "note": "Intent recorded. No live system was modified.",
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
            "note": "Intent recorded. No live system was modified.",
        }


class RollbackDeployArgs(BaseModel):
    service: str = Field(description="Exact service name.")
    target_version: str = Field(
        description="Version to roll back to, e.g. 'v2.14.1'. Must be a known deploy."
    )
    reason: str = Field(min_length=10, description="Why rollback is the right action.")

    @field_validator("target_version")
    @classmethod
    def _looks_like_a_version(cls, v: str) -> str:
        if not v.startswith("v"):
            raise ValueError(f"target_version should look like 'v1.2.3', got {v!r}")
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
            "note": "Intent recorded. No live system was modified.",
        }


def execute_action(
    conn: psycopg.Connection,
    tool: Tool,
    raw_args: dict[str, Any],
    *,
    investigation_id: str,
    approval_id: str,
) -> dict[str, Any]:
    """Run an approved write tool and record it.

    Refuses unless the named approval exists, belongs to this investigation, and
    carries decision='approved'. This is the structural half of the gate: the
    graph interrupts before proposing, and this refuses to act without the row
    that interrupt produced. Neither alone is sufficient.
    """
    row = conn.execute(
        "SELECT decision, proposed_tool, modified_arguments FROM approvals "
        "WHERE id = %s AND investigation_id = %s",
        (approval_id, investigation_id),
    ).fetchone()

    if row is None:
        raise ToolError(
            f"no approval {approval_id!r} for investigation {investigation_id!r}",
            status=ToolStatus.FAILED,
        )
    decision, proposed_tool, modified = row
    if decision != "approved":
        raise ToolError(
            f"approval {approval_id} is {decision or 'still pending'}, not approved",
            status=ToolStatus.FAILED,
        )
    if proposed_tool != tool.name:
        raise ToolError(
            f"approval {approval_id} authorises {proposed_tool!r}, not {tool.name!r}",
            status=ToolStatus.FAILED,
        )

    # An approver who edited the arguments approved *their* version, not the
    # agent's. Using the original would silently discard the correction.
    effective = modified if modified else raw_args

    args = tool.validate_args(effective)
    result = tool.run(conn, args)

    conn.execute(
        "INSERT INTO actions (investigation_id, approval_id, tool_name, arguments, outcome) "
        "VALUES (%s, %s, %s, %s, 'recorded')",
        (investigation_id, approval_id, tool.name, __import__("json").dumps(effective)),
    )
    return result


for _tool in (RestartService(), ScaleService(), RollbackDeploy()):
    registry.register(_tool)
