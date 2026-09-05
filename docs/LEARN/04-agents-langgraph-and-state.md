# 04 — Agent loops, LangGraph, and why state lives in Postgres

What separates an "agent" from a single LLM call, and what the machinery is
actually for.

---

## A single call versus an agent

A single LLM call answers a question you already knew how to ask. An **agent**
decides *which* questions to ask, gathers information, looks at what came back,
and decides what to do next.

IncidentIQ's loop:

```
plan  →  execute_tool  →  reflect  →  plan  →  …  →  propose
```

The engineer's report — *"order-service looks unhealthy but I can't tell what
changed"* — does not contain enough information to answer. The agent has to go
and get it.

---

## Tool calling: the model never executes anything

You describe available functions to the model. It responds by *asking* for one
to be called with specific arguments. **Your code executes it** and hands back
the result.

That separation is the entire safety story. The model cannot restart a service;
it can only produce a message saying it would like to. Whether that happens is
your code's decision.

Each tool here declares a Pydantic model that does three jobs at once:

```python
class ServiceLogsArgs(_WindowMixin):
    service: str = Field(min_length=1, description="Exact service name")
    level: Literal["DEBUG","INFO","WARN","ERROR","FATAL"] = "ERROR"
    limit: int = Field(default=50, ge=1, le=200)
```

It generates the JSON Schema the model sees, it validates what comes back, and
it documents the tool. Keeping them as one artifact means the schema shown and
the validation performed cannot drift apart — a drift whose only symptom is
"the model is bad at tool calling".

---

## Why a graph instead of a while loop

This is the question an interviewer will ask, and "LangGraph is the standard
choice" is not an answer.

The loop genuinely is expressible as:

```python
while not done:
    tool = plan()
    result = call(tool)
    done = reflect(result)
```

Three things make that insufficient here:

**1. Checkpointing between steps.** LangGraph persists state after every node
transition. The while-loop equivalent is a manual save after each step, in every
branch, including error paths — and the branch you forget is the one that loses
an investigation.

**2. Pausing for a human, resuming in a different process.** The graph
interrupts before `act` and the run *ends*. Hours later a different process
resumes from the checkpoint with the decision. A while loop cannot do this
without being rewritten as a state machine that saves and restores its own
position — which is what LangGraph already is.

**3. The topology is data.** Nodes and edges can be inspected, so the trace view
shows which node is active and the eval harness can assert on the path taken. A
loop's control flow exists only while it is running.

**The honest counter-argument**, which is worth being able to give: for a fixed
three-step pipeline with no human in the loop, a while loop is simpler and the
dependency is not worth it. The justification rests on the interrupt and the
durability, not on the loop.

---

## State: the design decision everything rests on

```python
class InvestigationState(TypedDict, total=False):
    incident_id: str
    entities: Entities
    retrieved_docs: list[RetrievedDocRecord]
    tool_calls: Annotated[list[ToolCallRecord], operator.add]
    scratchpad: Annotated[list[str], operator.add]
    iteration: int
    pending_approval: ApprovalRequest | None
    status: Literal["running","awaiting_approval","complete","failed"]
```

Two properties of that shape matter.

**Everything must be JSON-serialisable.** No open connections, no model handles.
A live object would checkpoint successfully and then fail to restore — a failure
that only appears in the situation checkpointing exists for. The database
connection lives in a `NodeContext` passed alongside the state, never inside it.

**Lists are append-only via reducers.** `Annotated[list, operator.add]` means a
node returns only what it *adds*, and LangGraph concatenates. A node returning
the whole list would clobber concurrent updates and make every node responsible
for preserving history it did not create.

### One row per attempt, not per call

`tool_calls` records every *attempt*, including failures. A tool that fails twice
then succeeds is **one success out of three attempts**, and the eval reports it
that way. Collapsing retries into one logical call would make the tool
success-rate metric flattering and wrong.

---

## Failure injection as a feature

A config flag makes tools fail at a set rate, in three modes — because they
demand different handling:

| Mode | What the agent sees | Correct response |
|---|---|---|
| `timeout` | No result | Retry |
| `malformed` | Unparseable | Do not retry; fix the arguments |
| `partial` | **Valid but incomplete** | Notice, and qualify the conclusion |

`partial` is the dangerous one. Nothing raises. An agent that does not notice
reasons confidently from evidence it never saw. So it announces itself three
ways: a flag on the result, a field in the payload, and a sentence in the text
the model reads.

Injection is **seeded on (investigation, tool, attempt)**. A given investigation
fails identically on every run — otherwise a scenario "with injected failures"
scores differently each time and the number means nothing. But `attempt` is part
of the key, so a retry is a genuinely different draw and retry logic is testable.

---

## Human-in-the-loop

The gate is **structural**, not a prompt instruction:

- `requires_approval` is a class attribute the graph reads to decide whether to
  interrupt
- `execute_action` refuses without an approval row naming this investigation,
  this tool, and `decision='approved'`
- A rejection re-enters `plan` with the reason in the scratchpad, so the agent
  revises rather than stopping
- If a human *edits* the arguments, the edited version is what runs — using the
  agent's original would silently discard the correction

No amount of prompting makes a write tool execute. That is the point.

---

## What the first real run taught me

Running the graph against a local 14B model surfaced six bugs. Three are worth
remembering because they generalise.

**Retry classification matters more than retry count.** "Unknown service" was
classified as a transient failure, so the loop retried identical bad arguments
three times per iteration and burned the entire budget without a single
successful call. Argument errors must break the loop and send a *specific*
correction back — *"Did you mean: checkout-service?"* gets a right answer next
turn.

**JSON mode guarantees syntax, not schema.** Constrained to valid JSON, the model
was still free to invent field names — and past ~3000 tokens of prompt it did:

```
expected: {"root_cause": ..., "evidence_citations": [...], "remediation_tool": ...}
got:      {"incident_updates": ...}
```

Parsing succeeded. Every field read back as `None`. The agent produced an empty
proposal **having reasoned its way to the correct answer** — the reflect step
had already said "thread pool saturation and potential deadlocks". A silent
failure at the very last step.

The fix is passing an actual JSON Schema to the sampler so the wrong shape is
unrepresentable. And then a second lesson: constraining field *names* was not
enough, because with no `required` array the grammar still let the model skip
`root_cause` entirely. Every field had to be required — which meant replacing
nullable types with empty-string sentinels, since a field can only be required
if it has a representable way to say "nothing here".

**Validate the tool's *category*, not just its existence.** The model proposed
`get_service_logs` as a remediation. It is a real registered tool, so an
existence check passed — and it would have raised an approval request for an
action that changes nothing.

---

## What to be able to explain

- Why a graph rather than a while loop — **and** when a while loop is right
- Why state cannot hold a database connection
- Why `tool_calls` records attempts rather than calls
- What happens when a tool times out mid-investigation
- Why `partial` is more dangerous than `timeout`
- Why the approval gate is structural rather than a prompt instruction
- The difference between constraining JSON syntax and constraining JSON schema

---

**Next:** 05 — Evaluating an agent: completion, groundedness, and the limits of LLM-as-judge
