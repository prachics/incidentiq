"""The investigation state object.

This is the single most important design decision in the project, because it is
what makes "state survives failures" true rather than aspirational. Everything
the agent knows lives here, it is JSON-serialisable, and LangGraph checkpoints
it to Postgres after every node transition.

Two consequences follow, and they are the reason for the shape below.

**Everything must be serialisable.** No open connections, no model handles, no
Pydantic objects with non-JSON types. Tool results are plain dicts; retrieved
documents are flattened to their fields. A live object in state would checkpoint
successfully and then fail to restore, which is a failure mode that only appears
in the situation you built the checkpointing for.

**Lists are append-only, via reducers.** `scratchpad`, `tool_calls`, and
`events` are annotated with `operator.add`, so a node returns only what it wants
to *add* and LangGraph concatenates. A node that returned the whole list would
silently clobber concurrent updates, and would make every node responsible for
preserving history it did not create.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

InvestigationStatus = Literal["running", "awaiting_approval", "complete", "failed"]


class Entities(TypedDict, total=False):
    """What `intake` extracted from the user's report.

    All optional: a vague report ("something's wrong") legitimately yields
    almost nothing, and the agent must cope rather than fail. `confidence`
    records how much of this was inferred versus stated.
    """
    service: str | None
    related_services: list[str]
    error_signature: str | None
    time_window: str
    severity: str | None
    symptoms: list[str]
    confidence: float


class RetrievedDocRecord(TypedDict):
    """A retrieved chunk, flattened for serialisation.

    Carries the ranks from both retrievers, not just the fused score, because
    the evidence panel shows *why* a document surfaced and the groundedness
    judge needs the text that was actually in context.
    """
    chunk_id: int
    doc_type: str
    parent_doc_id: str
    title: str | None
    service: str | None
    excerpt: str
    rrf_score: float
    vector_rank: int | None
    keyword_rank: int | None


class ToolCallRecord(TypedDict):
    """One tool *attempt*. Retries appear as separate records.

    Collapsing retries into a single logical call would make the tool
    success-rate metric flattering and wrong: a tool that fails twice then
    succeeds is one success out of three attempts, and the eval reports it
    that way.
    """
    tool_name: str
    arguments: dict[str, Any]
    attempt: int
    status: str
    result: dict[str, Any] | None
    error: str | None
    latency_ms: int
    injected_failure: bool
    truncated: bool


class ApprovalRequest(TypedDict):
    """A pending human decision. Present in state only while the graph is
    interrupted at `await_approval`."""
    approval_id: str
    tool_name: str
    arguments: dict[str, Any]
    reasoning: str
    evidence: list[RetrievedDocRecord]
    supporting_tool_calls: list[ToolCallRecord]
    decision: str | None
    rejection_reason: str | None
    modified_arguments: dict[str, Any] | None


class Proposal(TypedDict):
    """The agent's conclusion."""
    root_cause: str
    confidence: float
    evidence_citations: list[str]     # parent_doc_ids backing the claim
    remediation: str
    remediation_tool: str | None
    remediation_arguments: dict[str, Any] | None
    requires_approval: bool
    abstained: bool                   # true when the agent found nothing usable
    abstention_reason: str | None


class InvestigationState(TypedDict, total=False):
    """Checkpointed to Postgres after every node transition."""

    # ── Identity and input ──────────────────────────────────
    incident_id: str
    user_query: str

    # ── Accumulated understanding ───────────────────────────
    entities: Entities
    retrieved_docs: list[RetrievedDocRecord]
    tool_calls: Annotated[list[ToolCallRecord], operator.add]
    scratchpad: Annotated[list[str], operator.add]

    # ── Loop control ────────────────────────────────────────
    iteration: int
    max_iterations: int
    next_tool: dict[str, Any] | None      # what `plan` decided to call
    plan_rationale: str
    should_continue: bool                 # what `reflect` decided

    # ── Output ──────────────────────────────────────────────
    proposal: Proposal | None
    pending_approval: ApprovalRequest | None
    action_result: dict[str, Any] | None
    final_summary: str

    # ── Bookkeeping ─────────────────────────────────────────
    status: InvestigationStatus
    failure_reason: str | None
    events: Annotated[list[dict[str, Any]], operator.add]   # node timings, for the trace view
    total_input_tokens: int
    total_output_tokens: int


def initial_state(
    incident_id: str, user_query: str, max_iterations: int = 8
) -> InvestigationState:
    """A fresh investigation.

    Every key is populated, including the empty ones. LangGraph merges partial
    updates into this dict, and a key that is absent rather than empty produces
    a KeyError in whichever node first reads it - at which point the graph has
    already checkpointed a state that cannot be resumed.
    """
    return {
        "incident_id": incident_id,
        "user_query": user_query,
        "entities": {},
        "retrieved_docs": [],
        "tool_calls": [],
        "scratchpad": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "next_tool": None,
        "plan_rationale": "",
        "should_continue": True,
        "proposal": None,
        "pending_approval": None,
        "action_result": None,
        "final_summary": "",
        "status": "running",
        "failure_reason": None,
        "events": [],
        "total_input_tokens": 0,
        "total_output_tokens": 0,
    }


def tool_success_rate(state: InvestigationState) -> float | None:
    """Successes over attempts. None when nothing was attempted.

    Counts PARTIAL as a success: the call returned usable data. Whether the
    agent then *noticed* the truncation is a groundedness question, measured
    separately - conflating the two would hide which of the two failed.
    """
    calls = state.get("tool_calls") or []
    if not calls:
        return None
    ok = sum(1 for c in calls if c["status"] in ("success", "partial"))
    return ok / len(calls)


def cited_doc_ids(state: InvestigationState) -> set[str]:
    """Documents actually retrieved into context. The groundedness judge scores
    claims against these and nothing else."""
    return {d["parent_doc_id"] for d in state.get("retrieved_docs") or []}
