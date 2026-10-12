# Design

> Written incrementally as each phase lands. Sections marked *pending* are
> deliberately empty rather than speculative — they get written from what the
> implementation actually does, not from what it was planned to do.

## Contents

1. [State persistence and recovery](#state-persistence-and-recovery)
2. [Why LangGraph rather than a while loop](#why-langgraph-rather-than-a-while-loop)
3. [Retrieval design](#retrieval-design)
4. [Tool design and failure injection](#tool-design-and-failure-injection)
5. [Human-in-the-loop](#human-in-the-loop)
6. [Known failure modes](#known-failure-modes)
7. [What breaks at 100x](#what-breaks-at-100x)

---

## State persistence and recovery

An investigation is not a request/response interaction. It is a process that may
run for minutes, call several tools, and stop dead in the middle waiting for a
human to approve something. Three things follow from that.

**State cannot live in process memory.** The API can restart, a worker can be
rescheduled, a tool can hang past any reasonable timeout. If the investigation's
working set — extracted entities, retrieved documents, tool results, reasoning
scratchpad, iteration count — exists only as a Python object, all of it is lost.
IncidentIQ writes state to Postgres after every node transition.

**Approval outlives the request.** The graph interrupts before any write action
and waits. That wait can be an hour. There is no HTTP request that stays open
for an hour, so the pause has to be a durable state in a database, not a blocked
coroutine.

**The audit trail is a consequence, not an addition.** Because every transition
is already written down, "what did the agent know when it proposed that?" is a
query rather than a reconstruction. `audit_log` is enforced append-only by a
Postgres trigger — `UPDATE` and `DELETE` both raise. That is a database-level
guarantee, not an application convention:

```
=> UPDATE audit_log SET actor='attacker' WHERE investigation_id='IQ-TEST';
ERROR:  audit_log is append-only; UPDATE is not permitted
```

### Schema shape

| Table | Holds |
|---|---|
| `investigations` | One row per investigation: status, entities, scratchpad, iteration |
| `tool_calls` | One row per **attempt**, including failures — retries are visible, not collapsed |
| `approvals` | Proposed action, reasoning, cited evidence, decision |
| `actions` | One row per approved action: outcome, attempts, result. Unique on the idempotency key |
| `control_plane_operations` | The mock control plane's ledger: what it applied, once per key |
| `audit_log` | Append-only decision record |

`tool_calls` storing one row per attempt rather than per logical call is what
makes the tool success-rate metric honest. A tool that fails twice and succeeds
on the third try is 1 success out of 3 attempts, and the eval reports it that
way.

### Recovery, demonstrated

The claim "state survives failures" is only worth making if it can be shown.
Phase 3 adds a test that kills the process mid-investigation and asserts the
resumed run continues from the last completed node rather than restarting.
Until that test exists, the claim stays in this document rather than the README.


---

## Why LangGraph rather than a while loop

The loop is genuinely expressible as a `while`:

```python
while not done:
    tool = plan(state)
    result = call(tool)
    done = reflect(state, result)
```

Three requirements make that insufficient.

**Checkpointing between steps.** LangGraph persists state after every node
transition. The while-loop equivalent is a manual save after each step in every
branch, error paths included, and the branch you forget is the one that loses an
investigation.

**Pausing for a human and resuming in a different process.** The graph
interrupts before `act` and the run *ends*. Hours later a different process
resumes from the checkpoint with the decision. A while loop cannot do this
without being rewritten as a state machine that saves and restores its own
position — which is what LangGraph already is.

**Inspectable topology.** Nodes and edges are data, so the trace view can show
which node is active and the eval harness can assert on the path taken. A loop's
control flow exists only while it runs.

**The counter-argument**, which is real: for a fixed pipeline with no human in
the loop, a while loop is simpler and the dependency is not worth it. The
justification here rests on the interrupt and the durability, not on the loop.

### The graph

```
intake → retrieve → plan → execute_tool → retrieve → reflect ─┬→ plan  (loop)
                                                              └→ propose
propose ─┬→ await_approval → [INTERRUPT] → act ─┬→ plan  (rejected)
         └→ summarize                            └→ summarize
```

`retrieve` runs after every tool call rather than once. Phase 2 measured why:
retrieval on the bare report scores Recall@5 0.461; on a query enriched with
observed error signatures it scores 0.800. After a tool call the agent has those
signatures.

Two independent termination guarantees: `reflect` checks the iteration cap, and
`route_after_reflect` checks it again. A model that always answers "keep going"
must still terminate, so the cap is enforced in control flow, not in a prompt.

---

## Retrieval design

Covered in full in [EVALS.md](EVALS.md#phase-2-results--retrieval). In brief:

- Documents in three typed tables; chunks in one indexed table
- Chunk 320 / overlap 64, chosen from a measured plateau
- Hybrid: pgvector cosine + Postgres FTS, fused with weighted RRF
  (`k=60`, `keyword_weight=0.5`)
- Metadata denormalised onto chunks so filtering happens before ranking

The result worth knowing: with a service pre-filter, hybrid and vector-only tie
at 0.800; without it, vector drops to 0.728 and hybrid holds. Hybrid's value is
insensitivity to whether the filter is available — the condition on turn one.

---

## Tool design and failure injection

Each tool declares one Pydantic model that is simultaneously the JSON Schema the
LLM sees, the validator, and the documentation. Keeping them as one artifact
means the schema shown and the validation performed cannot drift — a drift whose
only symptom is "the model is bad at tool calling".

### Retry policy

Retry classification turned out to matter more than retry count.

| Status | Retried? | Why |
|---|---|---|
| `timeout` | yes | Transient |
| `failed` | yes | May be transient |
| `malformed` | **no** | The arguments are wrong; retrying them cannot help |
| `partial` | no | It succeeded, just incompletely |

Argument errors — unknown service, unknown metric, a version never deployed —
are classified `malformed`, so the loop breaks and a *specific* correction goes
back to the model. Unknown service names use trigram similarity to suggest a
real one: *"Did you mean: checkout-service?"*

Before this, an unknown service was `failed`, so the loop retried identical bad
arguments three times per iteration and burned the entire budget without one
successful call.

### Failure injection

| Mode | What the agent sees | Correct response |
|---|---|---|
| `timeout` | No result | Retry |
| `malformed` | Unparseable | Fix the arguments |
| `partial` | **Valid but incomplete** | Notice; qualify the conclusion |

`partial` is the dangerous one — nothing raises, so an agent that misses it
reasons confidently from evidence it never saw. It announces itself three ways:
a flag on the result object, a field in the payload, and a sentence in the text
the model reads.

Injection is seeded on `(investigation_id, tool_name, attempt)`. A given
investigation fails identically on every run, so an eval scenario is
reproducible — but `attempt` is in the key, so a retry is a genuinely different
draw and retry logic is testable.

---

## Human-in-the-loop

The gate is structural, in three independent places:

1. `requires_approval` is a class attribute the graph reads to decide whether to
   interrupt
2. `interrupt_before=["act"]` halts the graph with a checkpoint written
3. `execute_action` refuses without an approval row naming this investigation,
   this tool, and an approving decision (`approved`, or `modified` with the approver's arguments)

No prompt phrasing makes a write tool execute.

A rejection is a correction rather than a dead end: the reason enters the
scratchpad and `route_after_act` returns to `plan`, so the agent revises. If the
human *edits* the arguments, the edited version runs — using the agent's
original would silently discard the correction.

`audit_log` is append-only, enforced by a Postgres trigger. The honest limit: a
superuser can drop the trigger. This is tamper-evident against application
mistakes, not tamper-proof against a DBA. Real immutability needs an external
append-only store.

---

### Executing an approved action

Approval answers *whether* an action may run. It does not answer how many times
it runs, and that becomes a real question as soon as the action leaves the
process. Phase 4 sends it to a control plane over HTTP (a mock: it records what
it would apply, and touches nothing real), which brings two ways to apply an
action twice:

1. **The response is lost.** The control plane applied a rollback, the reply
   never arrived, and the client retries. Without protection the rollback runs
   again.
2. **The process dies between acting and checkpointing.** LangGraph checkpoints
   after a node completes. If the API dies after `act` ran the action but before
   that checkpoint, resuming re-runs `act`.

Both are handled with one key. The approval id is sent as the `Idempotency-Key`
on every attempt. The control plane stores the first response per key and
returns it, marked as a replay, to any later request with that key; the same key
with a different body is a 422, because it means the caller lost track of what
it was retrying. `execute_action` also checks its own `actions` table first, so
a re-run `act` returns the recorded result without calling out at all.

Retries are limited to what can succeed on a second try: connection errors,
timeouts, 5xx and 429. Any other 4xx means the request is wrong, and is not repeated. A
control plane that stays down for every attempt leaves the action recorded as
`failed` with its attempt count, and the investigation failed with the reason;
deciding to try again later reuses the same key, so it is still safe.

What the key does not cover: a control plane that loses its own ledger. Real
systems expire idempotency keys after a retention window for the same reason,
and a retry after that window is a new request.

### Deciding over HTTP

`POST /approvals/{id}/decision` is the one write a human makes. It is a
conditional `UPDATE ... WHERE decision IS NULL`, so two approvers clicking at
once produce one decision and one 409, rather than whichever write landed last.
The approver's identity comes from the API key, not the body. Edited arguments
are validated against the tool's schema before they are stored, so a bad edit
is a 422 now rather than a failed action after the agent resumes.

The run itself happens in one background worker thread, with at most one job
per investigation. A single local model serves one request at a time, so more
threads would only move the queue into the model server. If the worker dies,
nothing is lost: checkpoints are in Postgres and `POST /investigations/{id}/resume`
continues from the last one. The resume refuses to run `act` for an approval
that has no decision, and an approved action that failed pauses before `act`
again, so the same resume retries it under the same idempotency key.

The approval row is written by the job that then pauses, so a fast approver can
decide before that job has returned. The decision's resume is held as a
follow-up and runs as soon as the job ends, rather than being dropped.

An approval's id is derived from the investigation, the run, the iteration and
the proposed action, not generated at random. The run id is created on every
fresh start, so re-running an investigation after a reset gets new keys.

Checkpoints belong to the graph version that wrote them. A run paused by one
version and resumed by a version with different state fields can stop at the
same interrupt without running `act`; the runner detects a resume that makes no
progress and fails the investigation with that reason instead of retrying it in
a loop. Upgrades should drain paused investigations or migrate their state. If the process dies after the row is
written but before `await_approval` is checkpointed, the node runs again on
resume and finds the same approval, decided or not, instead of creating a
second one that the original decision would never reach.

These worker guarantees - one job per investigation, follow-ups after a
decision - hold within one API process. Running several API processes would
need a per-investigation lease in Postgres (for example `SELECT … FOR UPDATE
SKIP LOCKED` on a jobs table) in place of the in-memory set.

## Known failure modes

Found by running the system, not by inspection.

**JSON mode guarantees syntax, not schema.** Constrained to valid JSON, a local
14B model past ~3000 prompt tokens returned `{"incident_updates": ...}` where
`root_cause` was expected. Parsing succeeded, every field read back `None`, and
the agent emitted an empty proposal *having reasoned its way to the correct
answer*. Fixed by passing a JSON Schema to the sampler for grammar-constrained
decoding — and then again, because with no `required` array the grammar still
let the model skip `root_cause`. Every field is now required, which meant
replacing nullable types with empty-string sentinels.

**Existence checks are not category checks.** The model proposed
`get_service_logs` as a *remediation*. It is a registered tool, so an existence
check passed — and it would have raised an approval request for an action that
changes nothing. Remediations are now validated as write tools.

**Silent thread resumption.** LangGraph keys checkpoints by `thread_id`, so
re-running an existing id *resumes* it. Correct for recovery, wrong for a fresh
run — a re-run reported two runs as one trace. `start()` now raises unless
`reset=True`.

**Automatic retrieval makes the explicit search tool redundant.** Across every
investigation run so far, `search_similar_incidents` was called **zero** times:

| tool | calls | success |
|---|---|---|
| `get_service_logs` | 17 | 82.4% |
| `get_service_dependencies` | 3 | 100% |
| `get_metrics` | 3 | 100% |
| `get_recent_deploys` | 1 | 100% |
| `search_similar_incidents` | **0** | — |

The `retrieve` node runs after every tool call and puts documents into context,
so by the time the model plans its next step the evidence it would have searched
for is already there. The tool is kept — it can search *unfiltered* across
services, which automatic retrieval does not, since that filters by the
extracted entities — but scenario expectations no longer list it, because
expecting a tool the design makes unnecessary would be scoring the agent against
a worse investigation than the one it actually performs.

The general point: automatic context injection and agentic tool use overlap, and
the overlap is worth noticing rather than shipping both and assuming both get
used.

**Local inference speed is a design constraint.** qwen2.5:14b generates at
~24 tok/s on an M4 Pro. An investigation is 10–12 LLM calls, so ~2 minutes; a
100-scenario suite is therefore hours, not minutes. That changes how the project
is developed — a stratified `--sample` subset exists because of it.

---

## What breaks at 100x

Honest answers for a corpus 100x larger (~68,000 chunks, ~50,000 incidents).

**Retrieval quality degrades before retrieval speed does.** HNSW stays fast, but
its approximation starts to matter — the index was verified deterministic at
this size (stdev 0.0000 across four rebuilds) and that check would need
repeating. The bigger problem is that near-duplicates multiply: 100x more
incidents of the same archetype means the top 5 fill with variations of one
event, and recall against *distinct* relevant documents falls. The fix is
diversity-aware reranking, which does not exist here.

**The metadata pre-filter becomes load-bearing rather than a nice win.** At this
size it buys 0.072 recall. At 100x, an unfiltered search over 68k chunks is
mostly noise, and filtering stops being an optimisation.

**Embedding the corpus stops being free.** 680 chunks take 2 seconds locally.
68,000 take ~3 minutes — still fine, but re-embedding on every chunk-size
experiment stops being interactive, and the sweep in EVALS.md would become an
overnight job.

**Context budget becomes the binding constraint, not retrieval.** Six documents
at 900 characters is already ~1,400 tokens per prompt. More documents do not
fit, so the ceiling shifts from "can we find it" to "can we fit it", and the
answer becomes reranking and summarisation rather than better search.

**`tool_calls` and `checkpoints` grow without bound.** Neither is partitioned or
pruned. A checkpoint per node transition at ~15 transitions per investigation is
fine at demo scale and needs a retention policy in production.
