"""The investigation graph and the runner that drives it.

`build_graph` wires the nine nodes; the graph interrupts before `act`, so every
write action waits for a decision with a checkpoint written. `InvestigationRunner`
starts and resumes runs and mirrors the checkpoint into IncidentIQ's own tables
after every node. Why a graph rather than a loop: docs/DESIGN.md.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph

from incidentiq.agent import nodes
from incidentiq.agent.nodes import NodeContext
from incidentiq.agent.state import InvestigationState
from incidentiq.config import Settings, get_settings

log = logging.getLogger(__name__)


# ── Conditional edges ───────────────────────────────────────
def route_after_reflect(state: InvestigationState) -> str:
    """plan → execute_tool → reflect is a loop with a hard cap.

    The cap is checked here as well as inside `reflect` because it is a safety
    property, not a reasoning one: a model that always answers "keep going"
    must still terminate.
    """
    if state.get("iteration", 0) >= state.get("max_iterations", 8):
        return "propose"
    return "plan" if state.get("should_continue", False) else "propose"


def route_after_propose(state: InvestigationState) -> str:
    """Only a write action goes to the approval gate."""
    proposal = state.get("proposal") or {}
    if proposal.get("requires_approval") and proposal.get("remediation_tool"):
        return "await_approval"
    return "summarize"


def route_after_act(state: InvestigationState) -> str:
    """A rejection re-enters planning; a failed action waits to be retried;
    anything else closes out.

    Rejection is a correction rather than a dead end - the reason the human
    gave is already in the scratchpad, so the next `plan` sees it.

    An approved action that could not be executed routes back to `act`. The
    graph interrupts before `act`, so the run pauses with the investigation
    failed and its reason recorded, and a resume retries the same approval,
    with the same idempotency key.
    """
    pending = state.get("pending_approval") or {}
    if pending.get("decision") == "rejected" and \
            state.get("iteration", 0) < state.get("max_iterations", 8):
        return "plan"
    if pending.get("decision") in ("approved", "modified") and \
            state.get("status") == "failed" and not state.get("action_result"):
        return "act"
    return "summarize"


def build_graph(ctx: NodeContext, checkpointer: Any | None = None):
    """Wire the nine nodes together.

    `interrupt_before=["act"]` is the human-in-the-loop mechanism: the run stops
    with a checkpoint written and returns control to the caller. Resuming with
    the same thread_id continues from exactly there.
    """
    g = StateGraph(InvestigationState)

    g.add_node("intake", partial(nodes.intake, ctx=ctx))
    g.add_node("retrieve", partial(nodes.retrieve, ctx=ctx))
    g.add_node("plan", partial(nodes.plan, ctx=ctx))
    g.add_node("execute_tool", partial(nodes.execute_tool_node, ctx=ctx))
    g.add_node("reflect", partial(nodes.reflect, ctx=ctx))
    g.add_node("propose", partial(nodes.propose, ctx=ctx))
    g.add_node("await_approval", partial(nodes.await_approval, ctx=ctx))
    g.add_node("act", partial(nodes.act, ctx=ctx))
    g.add_node("summarize", partial(nodes.summarize, ctx=ctx))

    g.add_edge(START, "intake")
    g.add_edge("intake", "retrieve")
    g.add_edge("retrieve", "plan")
    g.add_edge("plan", "execute_tool")
    # Re-retrieve after each tool call: Phase 2 measured retrieval improving
    # from 0.461 to 0.800 once observed error signatures are in the query.
    g.add_edge("execute_tool", "retrieve_again")
    g.add_node("retrieve_again", partial(nodes.retrieve, ctx=ctx))
    g.add_edge("retrieve_again", "reflect")

    g.add_conditional_edges("reflect", route_after_reflect,
                            {"plan": "plan", "propose": "propose"})
    g.add_conditional_edges("propose", route_after_propose,
                            {"await_approval": "await_approval", "summarize": "summarize"})
    g.add_edge("await_approval", "act")
    g.add_conditional_edges("act", route_after_act,
                            {"plan": "plan", "act": "act", "summarize": "summarize"})
    g.add_edge("summarize", END)

    return g.compile(checkpointer=checkpointer, interrupt_before=["act"])


class InvestigationRunner:
    """Owns the checkpointer connection and the compiled graph.

    The checkpointer needs its own connection with autocommit on - LangGraph
    manages its own transactions, and sharing the node connection would mean a
    node's failed transaction could roll back a checkpoint.
    """

    def __init__(self, settings: Settings | None = None, llm=None, conn=None,
                 control_plane=None):
        from incidentiq.llm import get_llm
        from incidentiq.tools.base import FailureInjector

        self.settings = settings or get_settings()
        self._own_conn = conn is None
        self.conn = conn or psycopg.connect(self.settings.database_url, autocommit=True)
        try:
            from pgvector.psycopg import register_vector
            register_vector(self.conn)
        except Exception:  # noqa: BLE001 - only needed for retrieval
            log.debug("pgvector not registered on the node connection")

        self.ctx = NodeContext(
            conn=self.conn,
            llm=llm or get_llm(self.settings),
            injector=FailureInjector(self.settings),
            tool_max_retries=self.settings.tool_max_retries,
            control_plane=control_plane,
        )
        self._cp_conn = psycopg.connect(self.settings.database_url, autocommit=True)
        self.checkpointer = PostgresSaver(self._cp_conn)
        self.checkpointer.setup()
        self.graph = build_graph(self.ctx, self.checkpointer)

    def close(self) -> None:
        self._cp_conn.close()
        if self._own_conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    # ── Running ─────────────────────────────────────────────
    def start(self, incident_id: str, user_query: str,
              *, reset: bool = False) -> InvestigationState:
        """Run until completion or until the approval interrupt.

        Raises if `incident_id` already has checkpoints, unless `reset=True`.
        LangGraph keys checkpoints by thread_id, so re-running an existing id
        silently *resumes* it - which is exactly the desired behaviour for
        recovery and exactly the wrong one for a fresh investigation. Left
        implicit it produces confusing results: an eval re-run would continue
        the previous attempt and report the combined trace as one run.
        """
        from incidentiq.agent.state import initial_state

        existing = self._cp_conn.execute(
            "SELECT count(*) FROM checkpoints WHERE thread_id = %s", (incident_id,)
        ).fetchone()[0]
        if existing and not reset:
            raise ValueError(
                f"investigation {incident_id!r} already has {existing} checkpoints. "
                "Use resume() to continue it, or start(..., reset=True) to discard "
                "and start over."
            )
        if existing and reset:
            self.reset(incident_id)

        state = initial_state(incident_id, user_query, self.settings.max_iterations)
        self._persist_start(state)
        config = {"configurable": {"thread_id": incident_id}}
        return self._settle(incident_id, self._advance(config, state))

    def reset(self, incident_id: str) -> None:
        """Discard all checkpoints and derived rows for an investigation."""
        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
            self._cp_conn.execute(f"DELETE FROM {table} WHERE thread_id = %s", (incident_id,))
        self.conn.execute("DELETE FROM tool_calls WHERE investigation_id = %s", (incident_id,))
        self.conn.execute("DELETE FROM investigations WHERE id = %s", (incident_id,))

    def resume(self, incident_id: str) -> InvestigationState:
        """Continue from the last checkpoint.

        Used both after an approval decision and after a crash. That these are
        the same call is the point: recovery is not a special path bolted on, it
        is the ordinary resume path.
        """
        config = {"configurable": {"thread_id": incident_id}}
        snapshot = self.graph.get_state(config)
        if snapshot.next == ("act",) and not self._decided(snapshot.values):
            # Never run `act` for an undecided approval: it would find nothing
            # to do and close the investigation without its action. Mirroring
            # the checkpoint puts the status back to awaiting_approval.
            log.warning("resume of %s skipped: its approval has no decision", incident_id)
            self._persist(snapshot.values)
            return snapshot.values

        self.conn.execute(
            "UPDATE investigations SET status = 'running', updated_at = now() WHERE id = %s",
            (incident_id,),
        )
        return self._settle(incident_id, self._advance(config, None))

    def _advance(self, config: dict[str, Any],
                 payload: InvestigationState | None) -> InvestigationState:
        """Run until completion or the next interrupt, mirroring after every node.

        Streaming node by node instead of a single invoke is what lets the API
        and the UI show an investigation while it runs. The end state is the
        same: `stream` stops at the same interrupt `invoke` does.
        """
        for _update in self.graph.stream(payload, config, stream_mode="updates"):
            self._persist(self.graph.get_state(config).values)
        return self.graph.get_state(config).values

    def resume_for_decision(self, incident_id: str, approval_id: str) -> InvestigationState:
        """Resume because `approval_id` was decided - unless the run already
        moved past it. A run that was still finishing when the decision came
        consumes the decision itself (`_settle`); resuming again would act a
        second time, for example retrying a failed action nobody asked to retry.
        """
        config = {"configurable": {"thread_id": incident_id}}
        snapshot = self.graph.get_state(config)
        values = snapshot.values or {}
        # Consumed: the run finished, or acted on it and is paused on a failure.
        # A run stopped earlier (a crash before await_approval was checkpointed)
        # has not reached the approval yet and is resumed to it.
        consumed = not snapshot.next or (snapshot.next == ("act",)
                                         and values.get("status") != "awaiting_approval")
        if consumed:
            log.info("decision on %s already consumed by the run of %s", approval_id, incident_id)
            if values:
                self._persist(values)   # the API set `queued`; put the real status back
            return values
        return self.resume(incident_id)

    def _settle(self, incident_id: str, final: InvestigationState) -> InvestigationState:
        """Mirror the result, and go past an approval that is already decided.

        A run can reach the interrupt before `act` with its approval decided:
        the decision came while the run was still finishing, or the approval
        survived a crash and was decided before recovery. Stopping there would
        wait for a decision that already exists. A failed action also pauses
        before `act`, but its status is `failed` and it waits for a resume.
        """
        config = {"configurable": {"thread_id": incident_id}}
        while final.get("status") == "awaiting_approval" and \
                self.next_node(incident_id) == ("act",) and self._decided(final):
            self._persist(final)
            before = self._checkpoint_id(config)
            final = self._advance(config, None)
            if self._checkpoint_id(config) == before:
                # The graph stopped at the same interrupt without running `act`
                # (for example a checkpoint written by an incompatible version).
                # Looping would spin forever; fail visibly instead.
                raise RuntimeError("resume made no progress past the approval interrupt")
        self._persist(final)
        return final

    def _checkpoint_id(self, config: dict[str, Any]) -> str | None:
        snapshot = self.graph.get_state(config)
        return (snapshot.config or {}).get("configurable", {}).get("checkpoint_id")

    def _decided(self, state: InvestigationState) -> bool:
        approval_id = (state.get("pending_approval") or {}).get("approval_id")
        row = self.conn.execute("SELECT decision FROM approvals WHERE id = %s",
                                (approval_id,)).fetchone() if approval_id else None
        return bool(row and row[0])

    def get_state(self, incident_id: str) -> InvestigationState | None:
        config = {"configurable": {"thread_id": incident_id}}
        snapshot = self.graph.get_state(config)
        return snapshot.values if snapshot and snapshot.values else None

    def next_node(self, incident_id: str) -> tuple[str, ...]:
        """Which node the graph would run next. Empty when finished."""
        config = {"configurable": {"thread_id": incident_id}}
        snapshot = self.graph.get_state(config)
        return snapshot.next if snapshot else ()

    # ── Mirror into the human-readable tables ───────────────
    def _persist_start(self, state: InvestigationState) -> None:
        self.conn.execute(
            # The API inserts the row as 'queued' before the worker picks it up.
            "INSERT INTO investigations (id, user_query, status) VALUES (%s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET user_query = EXCLUDED.user_query, "
            "status = EXCLUDED.status, updated_at = now()",
            (state["incident_id"], state["user_query"], "running"),
        )

    def _persist(self, state: InvestigationState) -> None:
        """Mirror the checkpoint into IncidentIQ's own tables.

        LangGraph's checkpoint is the source of truth for resuming; these tables
        are what the API, the frontend, and SQL-based analysis read. Keeping
        both is deliberate - the checkpoint format is LangGraph's to change.
        """
        import json

        self.conn.execute(
            "UPDATE investigations SET status = %s, entities = %s, scratchpad = %s, "
            "iteration = %s, retrieved_docs = %s, final_summary = %s, failure_reason = %s, "
            "events = %s, proposal = %s, updated_at = now() WHERE id = %s",
            (
                state.get("status", "running"),
                json.dumps(state.get("entities") or {}, default=str),
                json.dumps(state.get("scratchpad") or [], default=str),
                state.get("iteration", 0),
                json.dumps(state.get("retrieved_docs") or [], default=str),
                state.get("final_summary") or None,
                state.get("failure_reason"),
                json.dumps(state.get("events") or [], default=str),
                json.dumps(state.get("proposal"), default=str) if state.get("proposal") else None,
                state["incident_id"],
            ),
        )

        existing = self.conn.execute(
            "SELECT count(*) FROM tool_calls WHERE investigation_id = %s",
            (state["incident_id"],),
        ).fetchone()[0]
        for call in (state.get("tool_calls") or [])[existing:]:
            self.conn.execute(
                "INSERT INTO tool_calls (investigation_id, tool_name, arguments, attempt, "
                "status, result, error, latency_ms, injected_failure) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (state["incident_id"], call["tool_name"],
                 json.dumps(call["arguments"], default=str), call["attempt"],
                 call["status"], 
                 json.dumps(call["result"], default=str) if call["result"] else None,
                 call.get("error"), call.get("latency_ms", 0),
                 call.get("injected_failure", False)),
            )
