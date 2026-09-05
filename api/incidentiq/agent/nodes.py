"""The nine graph nodes.

Each is a pure-ish function: state in, partial state out. LangGraph merges the
partial into the checkpoint. Nodes never mutate the state they are given -
returning only the delta is what lets the `operator.add` reducers work and what
keeps every node's contribution auditable in the trace.

A node must never raise. An unhandled exception leaves the graph with no
checkpoint for that transition, which is precisely the case the persistence is
supposed to survive. Failures become `status: failed` with a reason, or a
recorded tool failure the agent can reason about.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

import psycopg

from incidentiq.agent import prompts
from incidentiq.agent.schemas import (
    IntakeOut,
    PlanOut,
    ProposeOut,
    ReflectOut,
    SummarizeOut,
    schema_of,
)
from incidentiq.agent.state import InvestigationState
from incidentiq.llm.base import LLMProvider, LLMResponse, extract_json
from incidentiq.tools import ToolStatus, execute_tool, registry
from incidentiq.tools.base import FailureInjector

log = logging.getLogger(__name__)


class NodeContext:
    """Everything the nodes need that is not state.

    Held outside the state object because none of it is serialisable - a live
    psycopg connection in a checkpoint would serialise and then fail to restore.
    """

    def __init__(
        self,
        conn: psycopg.Connection,
        llm: LLMProvider,
        injector: FailureInjector | None = None,
        tool_max_retries: int = 3,
    ):
        self.conn = conn
        self.llm = llm
        self.injector = injector or FailureInjector()
        self.tool_max_retries = tool_max_retries


def _event(node: str, started: float, **extra: Any) -> dict[str, Any]:
    return {
        "node": node,
        "duration_ms": int((time.perf_counter() - started) * 1000),
        "at": time.time(),
        **extra,
    }


def _ask_json(
    ctx: NodeContext, system: str, user: str, *, max_tokens: int = 1024,
    model: type | None = None,
) -> tuple[dict | None, LLMResponse]:
    """One LLM call that is expected to return JSON.

    Tolerant parsing rather than strict, because smaller local models wrap JSON
    in prose or fences regardless of instructions. Being strict here would make
    the local-model path fail for formatting reasons rather than reasoning ones,
    which would make the local-vs-frontier comparison measure the wrong thing.
    """
    resp = ctx.llm.complete_json(
        system, [{"role": "user", "content": user}],
        max_tokens=max_tokens,
        schema=schema_of(model) if model else None,
    )
    parsed = extract_json(resp.text)
    if parsed is None or model is None:
        return parsed, resp
    # Validate the shape even when it was constrained: a provider that ignores
    # `format` must not silently produce a differently-shaped result.
    try:
        return model.model_validate(parsed).model_dump(), resp
    except Exception:  # noqa: BLE001 - a bad shape is a parse failure
        expected = set(model.model_fields)
        if not expected & set(parsed):
            log.warning("response had none of the expected fields %s; got %s",
                        sorted(expected)[:4], sorted(parsed)[:4])
            return None, resp
        return parsed, resp


def _tokens(state: InvestigationState, resp: LLMResponse) -> dict[str, int]:
    return {
        "total_input_tokens": state.get("total_input_tokens", 0) + resp.input_tokens,
        "total_output_tokens": state.get("total_output_tokens", 0) + resp.output_tokens,
    }


# ── 1. intake ───────────────────────────────────────────────
def intake(state: InvestigationState, ctx: NodeContext) -> dict[str, Any]:
    """Parse the report into entities: service, error signature, window, severity."""
    started = time.perf_counter()
    user = f"On-call engineer's report:\n\n{state['user_query']}"
    parsed, resp = _ask_json(ctx, prompts.INTAKE, user, max_tokens=600, model=IntakeOut)

    if not parsed:
        # Not fatal. A failed extraction means retrieval works from the raw
        # query and the tools have to find the service - degraded, not broken.
        return {
            "entities": {"confidence": 0.0, "time_window": "6h"},
            "scratchpad": ["intake: could not extract structured entities; "
                           "proceeding from the raw report"],
            "events": [_event("intake", started, ok=False)],
            **_tokens(state, resp),
        }

    entities = {
        # Empty string is the schema's representation of "not present".
        "service": parsed.get("service") or None,
        "related_services": parsed.get("related_services") or [],
        "error_signature": parsed.get("error_signature") or None,
        "time_window": parsed.get("time_window") or "6h",
        "severity": parsed.get("severity") or None,
        "symptoms": parsed.get("symptoms") or [],
        "confidence": float(parsed.get("confidence") or 0.0),
    }
    note = (f"intake: service={entities['service']}, "
            f"error={entities['error_signature']}, "
            f"window={entities['time_window']}, "
            f"confidence={entities['confidence']:.2f}")
    return {
        "entities": entities,
        "scratchpad": [note],
        "events": [_event("intake", started, ok=True,
                          service=entities["service"])],
        **_tokens(state, resp),
    }


# ── 2. retrieve ─────────────────────────────────────────────
def retrieve(state: InvestigationState, ctx: NodeContext) -> dict[str, Any]:
    """Hybrid search over incidents, runbooks, and service docs.

    Runs on every loop iteration, not once. Phase 2 measured why: retrieval on
    the bare report scores Recall@5 0.461, and on a query enriched with observed
    error signatures it scores 0.800. After a tool call the agent has those
    signatures, so re-retrieving is worth an iteration.
    """
    started = time.perf_counter()
    from incidentiq.rag.retrieval import Filters, hydrate, search

    entities = state.get("entities") or {}
    parts = [state["user_query"]]
    if entities.get("error_signature"):
        parts.append(f"Error: {entities['error_signature']}")
    if entities.get("symptoms"):
        parts.append("Observed: " + "; ".join(entities["symptoms"]))

    # Signatures seen in tool results so far - the enrichment that Phase 2
    # measured as worth +0.34 Recall@5.
    for call in (state.get("tool_calls") or [])[-4:]:
        if call["status"] not in ("success", "partial") or not call["result"]:
            continue
        for sig in (call["result"].get("distinct_signatures") or [])[:3]:
            parts.append(sig["message"])

    query = " ".join(parts)[:2000]

    service = entities.get("service")
    services = [service, *entities.get("related_services", [])] if service else None
    try:
        hits = hydrate(ctx.conn, search(
            ctx.conn, query,
            filters=Filters(service=services) if services else None,
            limit=6, mode="hybrid",
        ))
    except Exception as exc:  # noqa: BLE001
        log.exception("retrieval failed")
        return {
            "scratchpad": [f"retrieve: search failed ({exc}); continuing without documents"],
            "events": [_event("retrieve", started, ok=False)],
        }

    docs = [
        {
            "chunk_id": h.chunk_id, "doc_type": h.doc_type,
            "parent_doc_id": h.parent_doc_id, "title": h.parent_title,
            "service": h.service, "excerpt": h.content[:900],
            "rrf_score": round(h.rrf_score, 6),
            "vector_rank": h.vector_rank, "keyword_rank": h.keyword_rank,
        }
        for h in hits
    ]
    # Union with what is already held, best-scoring copy of each chunk kept.
    existing = {d["chunk_id"]: d for d in (state.get("retrieved_docs") or [])}
    for d in docs:
        if d["chunk_id"] not in existing or d["rrf_score"] > existing[d["chunk_id"]]["rrf_score"]:
            existing[d["chunk_id"]] = d
    merged = sorted(existing.values(), key=lambda d: d["rrf_score"], reverse=True)[:12]

    note = (f"retrieve: {len(docs)} documents "
            f"({', '.join(d['parent_doc_id'] for d in docs[:5]) or 'none'})")
    return {
        "retrieved_docs": merged,
        "scratchpad": [note],
        "events": [_event("retrieve", started, ok=True, count=len(docs))],
    }


# ── 3. plan ─────────────────────────────────────────────────
def plan(state: InvestigationState, ctx: NodeContext) -> dict[str, Any]:
    """Choose the next diagnostic tool."""
    started = time.perf_counter()
    tools = registry.read_only()
    catalog = "\n".join(
        f"- {t.name}: {t.description}\n  schema: "
        f"{json.dumps(t.args_model.model_json_schema().get('properties', {}))}"
        for t in tools
    )
    user = f"""\
Entities extracted from the report:
{prompts.render_entities(state.get('entities') or {})}

Original report: {state['user_query']}

Retrieved documents:
{prompts.render_retrieved(state.get('retrieved_docs') or [])}

Tools called so far:
{prompts.render_tool_calls(state.get('tool_calls') or [])}

Reasoning so far:
{prompts.render_scratchpad(state.get('scratchpad') or [])}

Available tools:
{catalog}

This is iteration {state.get('iteration', 0) + 1} of {state.get('max_iterations', 8)}.
"""
    parsed, resp = _ask_json(ctx, prompts.PLAN, user, max_tokens=700, model=PlanOut)

    valid = {t.name for t in tools}
    if not parsed or parsed.get("tool_name") not in valid:
        # Fall back to the standard opening move rather than failing: the
        # error signature identifies the failure mode more often than not.
        # Always build the fallback from known entities. Reusing the model's
        # arguments here was a bug: when it named an invalid tool its arguments
        # were usually empty too, so the fallback called get_service_logs with
        # service="" - which fails schema validation on every iteration and
        # burned the whole budget without ever calling a tool successfully.
        entities = state.get("entities") or {}
        service = entities.get("service") or ""
        args = {"time_window": entities.get("time_window") or "6h"}
        if service:
            args["service"] = service
        # Prefer the model's arguments only when they name a real service.
        if parsed and isinstance(parsed.get("arguments"), dict):
            proposed = parsed["arguments"]
            if proposed.get("service"):
                args = {**args, **proposed}
        chosen = {"tool_name": "get_service_logs", "arguments": args}
        return {
            "next_tool": chosen,
            "plan_rationale": "fallback: model did not select a valid tool",
            "iteration": state.get("iteration", 0) + 1,
            "scratchpad": [
                f"plan: invalid selection "
                f"{parsed.get('tool_name') if parsed else None!r}; "
                "defaulting to get_service_logs"
            ],
            "events": [_event("plan", started, ok=False)],
            **_tokens(state, resp),
        }

    chosen = {"tool_name": parsed["tool_name"], "arguments": parsed.get("arguments") or {}}
    rationale = parsed.get("rationale", "")
    return {
        "next_tool": chosen,
        "plan_rationale": rationale,
        "iteration": state.get("iteration", 0) + 1,
        "scratchpad": [f"plan: {chosen['tool_name']} - {rationale}"],
        "events": [_event("plan", started, ok=True, tool=chosen["tool_name"])],
        **_tokens(state, resp),
    }


# ── 4. execute_tool ─────────────────────────────────────────
def execute_tool_node(state: InvestigationState, ctx: NodeContext) -> dict[str, Any]:
    """Run the planned tool, retrying transient failures.

    Retry policy lives here rather than inside `execute_tool` so that it is
    visible, testable, and so every attempt is recorded separately. TIMEOUT and
    FAILED are retried; MALFORMED is not - a schema violation is unlikely to
    fix itself, and the error text goes back to the model so `plan` can correct
    the arguments next iteration.
    """
    started = time.perf_counter()
    planned = state.get("next_tool")
    if not planned:
        return {
            "scratchpad": ["execute_tool: nothing planned; skipping"],
            "events": [_event("execute_tool", started, ok=False)],
        }

    tool = registry.get(planned["tool_name"])
    if tool is None:
        return {
            "tool_calls": [{
                "tool_name": planned["tool_name"], "arguments": planned["arguments"],
                "attempt": 1, "status": "failed", "result": None,
                "error": f"no such tool {planned['tool_name']!r}",
                "latency_ms": 0, "injected_failure": False, "truncated": False,
            }],
            "scratchpad": [f"execute_tool: unknown tool {planned['tool_name']!r}"],
            "events": [_event("execute_tool", started, ok=False)],
        }

    records, last = [], None
    for attempt in range(1, ctx.tool_max_retries + 1):
        result = execute_tool(
            ctx.conn, tool, planned["arguments"],
            investigation_id=state["incident_id"], attempt=attempt, injector=ctx.injector,
        )
        records.append({
            "tool_name": result.tool_name, "arguments": result.arguments,
            "attempt": result.attempt, "status": result.status.value,
            "result": result.data, "error": result.error,
            "latency_ms": result.latency_ms,
            "injected_failure": result.injected_failure, "truncated": result.truncated,
        })
        last = result
        if result.ok:
            break
        if result.status is ToolStatus.MALFORMED:
            break   # arguments are wrong; retrying identical arguments cannot help

    if last is not None and last.ok:
        note = f"execute_tool: {last.tool_name} ok"
        if len(records) > 1:
            note += f" after {len(records)} attempts"
        if last.truncated:
            note += " (PARTIAL - result was incomplete)"
    else:
        note = (f"execute_tool: {planned['tool_name']} failed after {len(records)} "
                f"attempt(s): {last.error if last else 'unknown'}")

    return {
        "tool_calls": records,
        "scratchpad": [note],
        "events": [_event("execute_tool", started, ok=bool(last and last.ok),
                          tool=planned["tool_name"], attempts=len(records))],
    }


# ── 5. reflect ──────────────────────────────────────────────
def reflect(state: InvestigationState, ctx: NodeContext) -> dict[str, Any]:
    """Decide whether to keep gathering evidence or move to a proposal."""
    started = time.perf_counter()
    iteration = state.get("iteration", 0)
    cap = state.get("max_iterations", 8)

    if iteration >= cap:
        return {
            "should_continue": False,
            "scratchpad": [f"reflect: iteration cap ({cap}) reached; proceeding to propose"],
            "events": [_event("reflect", started, ok=True, capped=True)],
        }

    user = f"""\
Original report: {state['user_query']}

Entities:
{prompts.render_entities(state.get('entities') or {})}

Retrieved documents:
{prompts.render_retrieved(state.get('retrieved_docs') or [], limit=4)}

Tools called:
{prompts.render_tool_calls(state.get('tool_calls') or [])}

Reasoning:
{prompts.render_scratchpad(state.get('scratchpad') or [])}

Iteration {iteration} of {cap}.
"""
    parsed, resp = _ask_json(ctx, prompts.REFLECT, user, max_tokens=500, model=ReflectOut)

    if not parsed:
        # Undecidable: continue while budget remains, so an unparseable
        # reflection costs an iteration rather than the whole investigation.
        keep_going = iteration < cap
        return {
            "should_continue": keep_going,
            "scratchpad": ["reflect: could not parse; continuing while budget remains"],
            "events": [_event("reflect", started, ok=False)],
            **_tokens(state, resp),
        }

    sufficient = bool(parsed.get("sufficient"))
    reasoning = parsed.get("reasoning", "")
    missing = parsed.get("missing") or []
    note = f"reflect: {'sufficient' if sufficient else 'need more'} - {reasoning}"
    if missing:
        note += f" (missing: {', '.join(map(str, missing))})"

    return {
        "should_continue": not sufficient,
        "scratchpad": [note],
        "events": [_event("reflect", started, ok=True, sufficient=sufficient)],
        **_tokens(state, resp),
    }


# ── 6. propose ──────────────────────────────────────────────
def propose(state: InvestigationState, ctx: NodeContext) -> dict[str, Any]:
    """State the root cause and a remediation, or abstain."""
    started = time.perf_counter()
    user = f"""\
Original report: {state['user_query']}

Entities:
{prompts.render_entities(state.get('entities') or {})}

Retrieved documents (cite these by id):
{prompts.render_retrieved(state.get('retrieved_docs') or [])}

Tool results:
{prompts.render_tool_calls(state.get('tool_calls') or [])}

Reasoning:
{prompts.render_scratchpad(state.get('scratchpad') or [])}
"""
    parsed, resp = _ask_json(ctx, prompts.PROPOSE, user, max_tokens=1200, model=ProposeOut)

    if not parsed:
        # Cannot parse a proposal: abstain rather than emit an empty one. An
        # unparseable response is not evidence of a cause.
        proposal = {
            "root_cause": "", "confidence": 0.0, "evidence_citations": [],
            "remediation": "", "remediation_tool": None, "remediation_arguments": None,
            "requires_approval": False, "abstained": True,
            "abstention_reason": "the model did not return a parseable proposal",
        }
        return {
            "proposal": proposal,
            "scratchpad": ["propose: unparseable response; abstaining"],
            "events": [_event("propose", started, ok=False)],
            **_tokens(state, resp),
        }

    tool_name = parsed.get("remediation_tool") or None
    tool = registry.get(tool_name) if tool_name else None
    # A remediation must be a WRITE tool. The model reliably picks a diagnostic
    # tool here when it is uncertain - `get_service_logs` as a "remediation" -
    # which would produce an approval request for an action that changes
    # nothing. Registered-but-read-only is dropped as firmly as hallucinated.
    if tool_name and (tool is None or not tool.requires_approval):
        tool_name, tool = None, None

    # Confidence is specified as 0..1 and comes back as 5, 90, or 0.9 depending
    # on how the model read the field. Normalise rather than propagate: an
    # out-of-range confidence shown to an approver is worse than a rough one.
    raw_confidence = float(parsed.get("confidence") or 0.0)
    if raw_confidence > 1.0:
        raw_confidence = raw_confidence / 100.0 if raw_confidence > 10 else raw_confidence / 10.0
    confidence = max(0.0, min(1.0, raw_confidence))

    # Keep only citations that name a document actually in context. A citation
    # of something never retrieved is a fabrication, and passing it through
    # would let it reach the approver as though it were evidence.
    available = {d["parent_doc_id"] for d in (state.get("retrieved_docs") or [])}
    claimed = [str(c) for c in (parsed.get("evidence_citations") or [])]
    citations = [c for c in claimed if c in available]
    invented = [c for c in claimed if c not in available]

    proposal = {
        "root_cause": parsed.get("root_cause") or "",
        "confidence": round(confidence, 3),
        "evidence_citations": citations,
        "remediation": parsed.get("remediation") or "",
        "remediation_tool": tool_name,
        "remediation_arguments": parsed.get("remediation_arguments") or None,
        "requires_approval": bool(tool and tool.requires_approval),
        "abstained": bool(parsed.get("abstained")),
        "abstention_reason": parsed.get("abstention_reason") or None,
    }

    notes = [f"propose: {'ABSTAINED' if proposal['abstained'] else proposal['root_cause'][:120]}"]
    if invented:
        notes.append(f"propose: dropped {len(invented)} citation(s) to documents that "
                     f"were never retrieved: {', '.join(invented)}")

    return {
        "proposal": proposal,
        "scratchpad": notes,
        "events": [_event("propose", started, ok=True,
                          abstained=proposal["abstained"],
                          dropped_citations=len(invented))],
        **_tokens(state, resp),
    }


# ── 7. await_approval ───────────────────────────────────────
def await_approval(state: InvestigationState, ctx: NodeContext) -> dict[str, Any]:
    """Create the approval record and mark the investigation as waiting.

    The interrupt itself is configured on the graph (`interrupt_before`), not
    performed here. This node's job is to make the pending decision durable so
    it survives the process that created it - an approval may sit for an hour,
    far longer than any HTTP request.
    """
    started = time.perf_counter()
    proposal = state.get("proposal") or {}
    approval_id = f"APR-{uuid.uuid4().hex[:12]}"

    evidence = [
        d for d in (state.get("retrieved_docs") or [])
        if d["parent_doc_id"] in set(proposal.get("evidence_citations") or [])
    ] or (state.get("retrieved_docs") or [])[:5]

    supporting = [c for c in (state.get("tool_calls") or []) if c["status"] in
                  ("success", "partial")][-6:]

    ctx.conn.execute(
        "INSERT INTO approvals (id, investigation_id, proposed_tool, proposed_arguments, "
        "reasoning, evidence) VALUES (%s, %s, %s, %s, %s, %s)",
        (approval_id, state["incident_id"], proposal.get("remediation_tool"),
         json.dumps(proposal.get("remediation_arguments") or {}),
         proposal.get("root_cause", "") + "\n\n" + proposal.get("remediation", ""),
         json.dumps({"documents": evidence, "tool_calls": supporting}, default=str)),
    )
    ctx.conn.execute(
        "INSERT INTO audit_log (investigation_id, event_type, actor, payload) "
        "VALUES (%s, 'approval_requested', 'agent', %s)",
        (state["incident_id"], json.dumps({
            "approval_id": approval_id,
            "tool": proposal.get("remediation_tool"),
            "arguments": proposal.get("remediation_arguments"),
        }, default=str)),
    )

    request = {
        "approval_id": approval_id,
        "tool_name": proposal.get("remediation_tool") or "",
        "arguments": proposal.get("remediation_arguments") or {},
        "reasoning": proposal.get("root_cause", ""),
        "evidence": evidence,
        "supporting_tool_calls": supporting,
        "decision": None,
        "rejection_reason": None,
        "modified_arguments": None,
    }
    return {
        "pending_approval": request,
        "status": "awaiting_approval",
        "scratchpad": [f"await_approval: {approval_id} created for "
                       f"{proposal.get('remediation_tool')}"],
        "events": [_event("await_approval", started, ok=True, approval_id=approval_id)],
    }


# ── 8. act ──────────────────────────────────────────────────
def act(state: InvestigationState, ctx: NodeContext) -> dict[str, Any]:
    """Execute an approved action, or route a rejection back into planning."""
    started = time.perf_counter()
    from incidentiq.tools.actions import execute_action

    pending = state.get("pending_approval") or {}
    approval_id = pending.get("approval_id")
    if not approval_id:
        return {
            "scratchpad": ["act: no pending approval; nothing to do"],
            "events": [_event("act", started, ok=False)],
        }

    row = ctx.conn.execute(
        "SELECT decision, rejection_reason, modified_arguments FROM approvals WHERE id = %s",
        (approval_id,),
    ).fetchone()
    decision, rejection_reason, modified = row if row else (None, None, None)

    if decision == "rejected":
        # A rejection is information, not a stop. It goes into the scratchpad so
        # the next `plan` sees why the human disagreed and can revise, rather
        # than the investigation simply ending.
        ctx.conn.execute(
            "INSERT INTO audit_log (investigation_id, event_type, actor, payload) "
            "VALUES (%s, 'approval_rejected', 'human', %s)",
            (state["incident_id"], json.dumps({"approval_id": approval_id,
                                               "reason": rejection_reason}, default=str)),
        )
        return {
            "pending_approval": {**pending, "decision": "rejected",
                                 "rejection_reason": rejection_reason},
            "status": "running",
            "should_continue": True,
            "scratchpad": [
                f"act: the human REJECTED {pending.get('tool_name')}. "
                f"Reason: {rejection_reason or 'not given'}. "
                "Revise the diagnosis - do not propose this action again."
            ],
            "events": [_event("act", started, ok=True, decision="rejected")],
        }

    if decision not in ("approved", "modified"):
        return {
            "status": "awaiting_approval",
            "scratchpad": [f"act: {approval_id} is still pending"],
            "events": [_event("act", started, ok=False, decision=decision)],
        }

    tool = registry.get(pending.get("tool_name") or "")
    if tool is None:
        return {
            "status": "failed",
            "failure_reason": f"approved tool {pending.get('tool_name')!r} is not registered",
            "events": [_event("act", started, ok=False)],
        }

    try:
        result = execute_action(
            ctx.conn, tool, pending.get("arguments") or {},
            investigation_id=state["incident_id"], approval_id=approval_id,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("approved action failed")
        return {
            "status": "failed",
            "failure_reason": f"approved action failed: {exc}",
            "scratchpad": [f"act: execution failed - {exc}"],
            "events": [_event("act", started, ok=False)],
        }

    ctx.conn.execute(
        "INSERT INTO audit_log (investigation_id, event_type, actor, payload) "
        "VALUES (%s, 'action_executed', 'agent', %s)",
        (state["incident_id"], json.dumps({"approval_id": approval_id,
                                           "result": result}, default=str)),
    )
    return {
        "action_result": result,
        "pending_approval": {**pending, "decision": decision,
                             "modified_arguments": modified},
        "status": "running",
        "scratchpad": [f"act: executed {tool.name} under approval {approval_id}"],
        "events": [_event("act", started, ok=True, decision=decision)],
    }


# ── 9. summarize ────────────────────────────────────────────
def summarize(state: InvestigationState, ctx: NodeContext) -> dict[str, Any]:
    """Write the closing summary and mark the investigation complete."""
    started = time.perf_counter()
    proposal = state.get("proposal") or {}
    user = f"""\
Original report: {state['user_query']}

Conclusion: {json.dumps(proposal, default=str)}

Action taken: {json.dumps(state.get('action_result'), default=str)}

Reasoning trace:
{prompts.render_scratchpad(state.get('scratchpad') or [], limit=20)}
"""
    parsed, resp = _ask_json(ctx, prompts.SUMMARIZE, user, max_tokens=800, model=SummarizeOut)

    if parsed and parsed.get("summary"):
        summary = parsed["summary"]
        uncertainties = parsed.get("uncertainties") or []
        if uncertainties:
            summary += "\n\nStill uncertain: " + "; ".join(map(str, uncertainties))
    else:
        # Deterministic fallback so an investigation always closes with
        # something readable, even when the model fails at the last step.
        summary = (
            f"Investigation {state['incident_id']} closed. "
            + (f"Root cause: {proposal.get('root_cause')}. " if proposal.get("root_cause")
               else "No root cause was established. ")
            + (f"Remediation: {proposal.get('remediation')}. "
               if proposal.get("remediation") else "")
            + f"{len(state.get('tool_calls') or [])} tool call(s), "
              f"{len(state.get('retrieved_docs') or [])} document(s) retrieved."
        )

    return {
        "final_summary": summary,
        "status": "complete",
        "scratchpad": ["summarize: investigation closed"],
        "events": [_event("summarize", started, ok=True)],
        **_tokens(state, resp),
    }
