"""Assembling the LangGraph.

Why a graph rather than a while loop
------------------------------------
The loop this agent runs is genuinely expressible as
`while not done: plan(); call_tool(); reflect()`. Three things make that
insufficient here, and none of them are about elegance.

1. **Checkpointing between steps.** LangGraph persists state after every node
   transition. In a while loop the equivalent is a manual save after each step,
   in every branch, including the error paths - and the one branch you forget is
   the one that loses an investigation.

2. **Pausing for a human and resuming later, in a different process.** The graph
   interrupts before `act` and the run ends. Hours later a different process
   resumes from the checkpoint with the approval decision. A while loop cannot
   do this without being rewritten as a state machine that saves and restores
   its own position - which is what LangGraph already is.

3. **The topology is inspectable.** Nodes and edges are data, so the trace view
   can show which node is active and the eval harness can assert on the path
   taken. A loop's control flow only exists while it is running.

The honest counter-argument: for a fixed three-step pipeline with no human in
the loop, a while loop is simpler and the dependency is not worth it. The
justification here rests on the interrupt and the durability, not on the loop.
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
    """A rejection re-enters planning; anything else closes out.

    This is what makes rejection a correction rather than a dead end - the
    reason the human gave is already in the scratchpad, so the next `plan` sees
    it.
    """
    pending = state.get("pending_approval") or {}
    if pending.get("decision") == "rejected" and \
            state.get("iteration", 0) < state.get("max_iterations", 8):
        return "plan"
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
                            {"plan": "plan", "summarize": "summarize"})
    g.add_edge("summarize", END)

    return g.compile(checkpointer=checkpointer, interrupt_before=["act"])


class InvestigationRunner:
    """Owns the checkpointer connection and the compiled graph.

    The checkpointer needs its own connection with autocommit on - LangGraph
    manages its own transactions, and sharing the node connection would mean a
    node's failed transaction could roll back a checkpoint.
    """

    def __init__(self, settings: Settings | None = None, llm=None, conn=None):
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
        final = self.graph.invoke(state, config)
        self._persist(final)
        return final

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
        final = self.graph.invoke(None, config)
        self._persist(final)
        return final

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
            "INSERT INTO investigations (id, user_query, status) VALUES (%s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET user_query = EXCLUDED.user_query",
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
            "iteration = %s, retrieved_docs = %s, final_summary = %s, failure_reason = %s "
            "WHERE id = %s",
            (
                state.get("status", "running"),
                json.dumps(state.get("entities") or {}, default=str),
                json.dumps(state.get("scratchpad") or [], default=str),
                state.get("iteration", 0),
                json.dumps(state.get("retrieved_docs") or [], default=str),
                state.get("final_summary") or None,
                state.get("failure_reason"),
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
