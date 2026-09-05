# Design

> Written incrementally as each phase lands. Sections marked *pending* are
> deliberately empty rather than speculative — they get written from what the
> implementation actually does, not from what it was planned to do.

## Contents

1. [State persistence and recovery](#state-persistence-and-recovery)
2. Why LangGraph rather than a while loop — *pending, Phase 3*
3. Retrieval design: chunking, hybrid search, metadata filtering — *pending, Phase 2*
4. Tool design and failure injection — *pending, Phase 3*
5. Human-in-the-loop and the audit trail — *pending, Phase 4*
6. Known failure modes — *pending, Phase 3*
7. What breaks at 100x corpus size — *pending, Phase 2*

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
| `actions` | Write-tool intent. Records what *would* have happened; touches nothing real |
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
