# IncidentIQ

An agentic AI platform for production incident support.

An on-call engineer describes a problem in plain language. The agent retrieves
relevant incident history and runbooks, calls read-only diagnostic tools against
a target system, reasons across several steps, and proposes a remediation.
Anything that would change production state stops and waits for human approval.

What distinguishes it from a demo is the parts demos skip: **investigation state
is persisted and survives process death**, **write actions are gated
structurally rather than by prompt**, and **there is an evaluation harness that
produces real numbers, including bad ones.**

**Status:** Phases 1–5 complete and measured: the agent, its evaluation, an approval API and a web UI.
Every number below comes from the harness in this repository and can be
reproduced with the commands given.

---

## Quickstart

Requires Docker and Python 3.11 or newer. No API key needed — the default configuration
uses local embeddings and a local model.

```bash
git clone https://github.com/prachics/incidentiq.git
cd incidentiq

python3 -m venv .venv && source .venv/bin/activate   # Python 3.11+
pip install -e ".[dev]"            # pulls torch for local embeddings; ~2 min
cp .env.example .env

docker compose up -d postgres      # Postgres 16 + pgvector
python scripts/migrate.py          # apply schema
python seeds/generate.py           # 500 incidents, 71 runbooks, 38 service docs, mock infra
python -m incidentiq.rag.indexer   # chunk + embed → 680 searchable chunks
```

That gives a fully seeded, working system. To reproduce the retrieval numbers:

```bash
python -m evals.retrieval_eval --ablation
```

To run the agent you also need a model. The default is local and free:

```bash
brew install ollama && ollama serve &
ollama pull qwen2.5:14b            # ~9GB

python -m evals.run --sample 2 --max-iterations 3   # ~18 min, all 5 scenario kinds
python -m evals.run                                 # the full 100-scenario suite
```

The full suite is a multi-hour job on local inference, so results are written as
each scenario finishes and `--resume` continues an interrupted run:

```bash
./scripts/run_eval.sh              # starts or resumes, whichever is right
python -m evals.run --resume       # the same thing, explicitly
```

An interrupted run loses at most the scenario that was in flight. The progress
file is keyed by provider, model, iteration cap and judge setting, so a resume
cannot silently mix results from two different configurations.

Set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY` in `.env` to run against
Claude instead; nothing else changes. `python -m evals.compare A.json B.json`
then reports which failure *modes* moved between two runs, not just the headline
number.

Optional services:

```bash
docker compose up -d langfuse      # tracing UI at :3001 (schema ready, not yet instrumented)
docker compose --profile app up -d # API at :8000, mock control plane at :8100
```

---

## Results

### Retrieval (Phase 2)

| Metric | Result |
|---|---|
| **Recall@5** | **0.800** |
| Hit@5 | 0.917 |
| MRR | 0.917 |
| Median latency | 28.7 ms |

Hybrid retrieval — pgvector cosine similarity plus Postgres full-text search,
fused with weighted reciprocal rank fusion — over `BAAI/bge-small-en-v1.5`
embeddings, 320-token chunks with 64-token overlap, service pre-filter applied
before ranking. Measured across 12 labelled situations.

### Agent (Phase 3)

| Metric | Target | Result |
|---|---|---|
| Task completion | ≥ 90% | **50.0%** |
| Abstention accuracy | — | **100%** |
| Tool execution success | ≥ 95% | 81.2% (32 attempts) |
| Recovery after injected tool failure | — | 50% |
| Median / p95 latency | — | 111.8s / 183.4s |
| Tokens per investigation | — | 16.0k in / 0.7k out |
| Groundedness | ≥ 90% | not yet scored |

| Scenario kind | n | Completion |
|---|---|---|
| Nothing retrievable (must abstain) | 2 | **100%** |
| Injected tool failures | 2 | **100%** |
| Cascading / multi-service | 2 | 50% |
| Single service | 2 | 0% |
| Approval-gated action | 2 | 0% |

**Read these numbers carefully.** They come from a 10-scenario stratified sample
against a local 14-billion-parameter model at a 3-iteration cap — not the full
100-scenario suite at full budget. Task completion is therefore a lower bound.
The comparison against a frontier model is the reason the provider interface
exists and has not been run yet. Groundedness is unscored because a groundedness
figure without a calibration kappa beside it is not interpretable, and that
calibration has not been done.

**Where the failures are is more informative than the score.** Of the five
failures, two were under-commitment rather than wrong answers: the agent
identified the correct failure mode and then declined to commit to it. In one
case it wrote
*"the recommendation-service is experiencing connection pool exhaustion"* — the
correct diagnosis — and abstained because it could not tell whether the cause was
a code defect or a downstream dependency.

That is the direct cost of a deliberate choice. The prompts push hard toward
abstaining rather than fabricating, which is why every abstention scenario
passed. The same bias makes the agent under-commit when evidence is present. A
system tuned the other way would score higher here and would fabricate on the
fifteen scenarios where fabricating is the worst possible behaviour. The
tradeoff is now measured rather than assumed.

Two more were hallucinated tool arguments — a version string that had never been
deployed, and a malformed one — both refused by the tool's own schema validation
before execution, after a human had already approved the action. The fifth named
the right service but not the failure mode.

Note the distinction between the two injected-failure figures: both scenarios in
that category reached a correct conclusion (100% completion), while the recovery
rate is 50% because only one of them had a tool that failed and then succeeded
on a retry.

Full methodology, ablations, and limitations: [`docs/EVALS.md`](docs/EVALS.md).

---

## Architecture

```mermaid
graph TB
    subgraph agent["Agent — LangGraph"]
        direction TB
        N1[intake] --> N2[retrieve]
        N2 --> N3[plan]
        N3 --> N4[execute_tool]
        N4 --> N5[reflect]
        N5 -->|needs more evidence| N3
        N5 -->|sufficient| N6[propose]
        N6 --> N7{{await_approval}}
        N7 -->|approved| N8[act]
        N7 -->|rejected| N3
        N8 --> N9[summarize]
    end

    subgraph tools["Tools"]
        RO[Read-only, auto-executed<br/>logs · metrics · deploys · dependencies]
        WR[Write, approval-gated<br/>restart · scale · rollback]
    end

    subgraph data["PostgreSQL 16 + pgvector"]
        CORP[(Corpora<br/>incidents · runbooks · docs)]
        CHUNK[(chunks<br/>HNSW + GIN indexes)]
        STATE[(Agent state<br/>checkpoints · approvals · audit)]
        MOCK[(Mock infrastructure<br/>logs · metrics · deploys)]
    end

    API[API — FastAPI<br/>API keys · viewer / approver]
    UI[Web UI — React + TypeScript] -->|REST + server-sent events| API
    HUMAN([Approver]) -->|decides in| UI
    CP[Mock control plane<br/>Idempotency-Key ledger]

    API -->|start / resume| N1
    N2 --> CHUNK
    CHUNK -.cites.-> CORP
    N4 --> RO --> MOCK
    N8 --> WR -->|HTTP, retried,<br/>key = approval id| CP
    agent <-->|checkpoint after<br/>every node transition| STATE

    classDef store fill:#1f2937,stroke:#4b5563,color:#e5e7eb
    classDef gate fill:#7c2d12,stroke:#ea580c,color:#fed7aa
    class CORP,CHUNK,STATE,MOCK store
    class N7,WR,CP gate
```

Three properties are worth calling out.

**`plan → execute_tool → reflect` is a loop with a hard cap**, enforced in
control flow rather than in a prompt. A model that always answers "keep going"
still terminates.

**`retrieve` runs after every tool call, not once.** Retrieval on the bare
on-call report scores Recall@5 0.461; on a query enriched with observed error
signatures it scores 0.800. After a tool call the agent has those signatures, so
re-retrieving is worth an iteration.

**The approval gate is structural, in three independent places:** a class
attribute the graph reads to decide whether to interrupt, LangGraph's
`interrupt_before` halting the run with a checkpoint written, and a runtime
check that refuses to execute without an approval row naming that investigation,
that tool, and an approving decision. No phrasing of a prompt makes a write tool
execute.

**An approved action executes once, however many times it is sent.** Actions
go to a control plane over HTTP, which can refuse a request, or apply it and
lose the response. The client retries with the approval id as the
`Idempotency-Key`; the control plane stores the first response per key and
replays it to every retry. If the process dies after the action ran but before
the checkpoint was written, `act` runs again on resume and finds the action
already recorded. An action that fails on every attempt leaves the
investigation `failed`, paused before `act`; `resume` retries it under the same
key. All of these are tested against injected faults.

---

## Using the API

Every investigation and approval endpoint needs an API key (`API_KEYS` in
`.env`, `key:user:role`). A `viewer` can start and read investigations; an
`approver` can also decide. Who approved is taken from the key, never from the
request body.

```bash
V="Authorization: Bearer dev-viewer-key"; A="Authorization: Bearer dev-approver-key"

# Start: returns 202 at once; the agent runs in a background worker
curl -X POST localhost:8000/investigations -H "$V" -H 'content-type: application/json' \
  -d '{"query": "getting paged for web-frontend, latency is way up. what is going on?"}'

curl localhost:8000/approvals?status=pending -H "$V"     # what is waiting for a human
curl localhost:8000/approvals/APR-… -H "$V"              # reasoning, evidence, argument schema

# Approve as proposed, approve with edited arguments, or reject with a reason
curl -X POST localhost:8000/approvals/APR-…/decision -H "$A" -H 'content-type: application/json' \
  -d '{"decision": "approve"}'
curl -X POST localhost:8000/approvals/APR-…/decision -H "$A" -H 'content-type: application/json' \
  -d '{"decision": "reject", "reason": "web-frontend is a victim; look at its dependencies"}'

curl localhost:8000/investigations/IQ-… -H "$V"          # status, approvals, executed actions
curl localhost:8000/investigations/IQ-…/audit -H "$V"    # append-only decision record
```

| Endpoint | Role | Notes |
|---|---|---|
| `POST /investigations` | viewer | 202; `incident_id` optional; 409 if it exists |
| `GET /investigations`, `/{id}`, `/{id}/tool-calls`, `/{id}/audit` | viewer | filter the list with `?status=` |
| `GET /investigations/{id}/stream` | viewer | server-sent events: the investigation is read once a second and sent as a `snapshot` when it changed; `end` when it settles |
| `GET /me` | viewer | the user and role behind the key |
| `POST /investigations/{id}/resume` | approver | recovery after a crash, or retry of a failed action; 409 while an approval is pending or a job is running |
| `GET /approvals`, `/approvals/{id}` | viewer | `?status=pending\|decided\|all` |
| `POST /approvals/{id}/decision` | approver | exactly one decision: concurrent attempts get 409; edited arguments validated (422); a rejection needs a reason |

Interactive docs at `localhost:8000/docs`.

---

## Web UI

React 18 + TypeScript (strict), TanStack Query and Tailwind, in `web/`. The API
serves the built app at `/app`; the Docker image builds it in a Node stage.

```bash
cd web && npm ci
npm run dev        # http://localhost:5173/app/ - proxies API calls to localhost:8000
npm run build      # then the API serves it at http://localhost:8000/app/
npm test           # Vitest + Testing Library
```

- **Investigations:** start one from a report, see each run's status at a glance.
- **Timeline:** every node the agent ran, in order, with durations, retries and failures, updated live.
- **Approval panel:** approve, approve with edited arguments (a form generated from the tool's JSON
  Schema, validated before it is sent), or reject with the reason the agent will revise from. Viewers
  see the request but cannot decide; a decision another approver made first shows who and when.
- **Evidence:** the proposal's root cause and remediation, each tool call with its status, latency
  and retries, and each retrieved document with its rank in vector and keyword search.
- **Audit log:** the append-only record of who did what.

Live updates come from `GET /investigations/{id}/stream`, which samples the investigation once a
second. Browsers' `EventSource` cannot send an `Authorization` header, so the UI reads the stream with
`fetch` and a small SSE parser, and polls instead if the stream fails or closes before its `end` event.
While an approval waits, the page polls slowly so a decision made in another browser shows up. The API
key lives in `sessionStorage`, not `localStorage`.

---

## Seeded data

The corpus is synthetic but not random. Every incident is one of **16 failure
archetypes** applied to one of **33 services** in a hand-authored dependency
graph, which makes it internally consistent in the ways the evaluation depends
on:

- An incident that blames a dependency blames a **real** dependency of that
  service. Verified by query: 90 root causes name another service, and **zero**
  name a service that is not an actual dependency of it.
- Cascading failures follow real edges — `payment-db` degrading surfaces as
  symptoms in `payment-service → checkout-service → api-gateway`.
- The same archetype on a *different* service produces a document that is
  topically similar but factually wrong for a given incident, which is what
  makes retrieval scores meaningful rather than trivial.
- Generation is seeded, so two runs produce identical data and eval numbers stay
  comparable.

**Caveat worth stating plainly:** a synthetic corpus is more internally
consistent than a real company's incident history, so retrieval scores here are
optimistic relative to production. What the numbers legitimately demonstrate is
that the measurement apparatus exists and works.

---

## Engineering findings

Results that were not what was expected going in. Each was found by reading one
failing case end to end, never from an aggregate.

**Hybrid retrieval buys exactly what a metadata filter would have bought.** With
a service pre-filter, hybrid and vector-only tie at Recall@5 0.800. Without it,
vector-only drops to 0.728 while hybrid holds at 0.800. Hybrid's value here is
not beating dense retrieval — it is being insensitive to whether the filter is
available, which is the condition on the first turn, before the agent knows
which service is at fault.

**Hybrid only started helping once the corpus was realistic.** Initially it
scored at or below vector-only everywhere. The cause was a corpus defect, not a
fusion defect: `HikariPool`, `OutOfMemoryError` and `x509` appeared in the logs
and **zero** times in the incident write-ups, so the keyword ranker had no
identifiers to match. A migration added quoted error lines to incidents, as a
real postmortem would. The general lesson: hybrid retrieval beats dense
retrieval when documents contain tokens embeddings cannot represent, and a
corpus of pure prose gains little from it.

**JSON mode guarantees syntax, not schema.** Constrained to valid JSON, the
local model past roughly 3,000 prompt tokens returned `{"incident_updates": ...}`
where `root_cause` was expected. Parsing succeeded, every field read back as
`None`, and the agent emitted an empty proposal *having already reasoned its way
to the correct answer*. Fixed by passing a JSON Schema to the sampler for
grammar-constrained decoding — and then again, because with no `required` array
the grammar still let the model omit `root_cause` entirely.

**Retry classification matters more than retry count.** An unknown service name
was classified as a transient failure, so the retry loop tried identical bad
arguments three times per iteration and burned the entire budget without a
single successful call. Argument errors now break the loop and return a specific
correction — *"Did you mean: checkout-service?"*, via trigram similarity.

**An example value in an error message is a value the model will copy.** A
validator rejected a malformed version with the message *"should look like
'v1.2.3'"*. That message reached the model through its scratchpad, and its next
attempt proposed rolling back to `v1.2.3` — on a service whose deploys were all
`v2.27.x`. Error text that feeds back into an agent's context is part of the
prompt.

**A passing test can pin the wrong behaviour.** The schema allowed an approval
decision of `modified`, and `act` routed it to execution, but `execute_action`
accepted only `approved`. Every approval with edited arguments would have
failed. The existing test set `decision='approved'` together with edited
arguments, so it passed. Building the API that writes `modified` exposed it.

**When a capable model fails in a way that looks careless, read the scenario
before believing the score.** Five times an eval expectation was wrong rather
than the agent — including one case where the agent read logs describing two
concurrent incidents and correctly declined to name a single cause, and was
marked down for it. Models do fail carelessly, but a failure that looks *stupid*
is more often a broken expectation than a broken model.

---

## What is built, and what is not

| Phase | Scope | Status |
|---|---|---|
| 1 | Docker Compose, schema, migrations, seeded corpora and mock infrastructure | **complete** |
| 2 | Chunking, embeddings, hybrid search, metadata filtering, Recall@5 | **complete** |
| 3 | LangGraph agent, state persistence, tools, retry logic, evaluation harness | **complete** |
| 4 | Approval API, audit trail, API keys, actions executed exactly once via a mock control plane | **complete** |
| 5 | Web UI: investigations, live timeline, evidence, approvals, audit log | **complete** |
| 6 | MCP server exposing the diagnostic tools | not started |

Langfuse is provisioned in Docker Compose and configured in settings, but the
agent is **not yet instrumented** to emit traces to it. Per-node timings, token
counts and retry counts are recorded in the investigation state and in the
`tool_calls` table today.

---

## Testing

300 backend tests and 22 frontend tests. Beyond the usual unit coverage, four areas are tested specifically:

- **State recovery.** A test kills a run mid-investigation, constructs a
  completely fresh runner sharing nothing but the database, resumes it, and
  asserts the pre-crash reasoning is intact and the run continues rather than
  restarting.
- **The approval gate.** Unapproved, rejected, mismatched-tool and nonexistent
  approvals are each refused; an approver's edited arguments override the
  agent's. Five approvers deciding at once produce exactly one decision.
- **Executing an action once.** A response lost after the control plane applied
  the action, `act` re-running after a crash before its checkpoint, a crash
  between applying and recording, and a control plane that is down for every
  attempt and later recovers through `resume`: in each case the action is
  applied at most once, and the outcome is stored with its attempt count. A
  manual resume is refused while the investigation already has a job, never runs
  `act` for an undecided approval, and a decision made while the job that
  requested it is still finishing is resumed right after that job. A crash
  right after an approval is created recovers to that same approval, whether
  it is decided first or the run is resumed first.
- **The evaluation harness itself.** An eval that grades incorrectly is worse
  than no eval, because it produces numbers people believe. The grading rules
  are pinned by test, including that the grader survives arbitrary model output
  rather than crashing on it.

```bash
pytest -q              # backend
cd web && npm test     # frontend
```

Tests that need Postgres skip cleanly when it is unavailable.

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | Why LangGraph over a plain loop (with the counter-argument), retry classification, failure injection, known failure modes, what breaks at 100× scale |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Running log of every non-obvious choice, what it cost, and how to reverse it |
| [`docs/EVALS.md`](docs/EVALS.md) | Evaluation methodology, ablations, per-run results, and stated limitations |

---

## Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph 1.2 with Postgres checkpointing |
| Backend | Python 3.12, FastAPI |
| Vector store | PostgreSQL 16 + pgvector (HNSW, cosine) |
| Keyword search | PostgreSQL full-text search (GIN over a generated `tsvector`) |
| Embeddings | `BAAI/bge-small-en-v1.5`, 384-dimensional, local |
| LLM | Swappable: Ollama (local, default), Anthropic, deterministic stub |
| Packaging | Docker Compose |
| Tracing | Langfuse (provisioned; instrumentation pending) |
| Frontend | React 18, TypeScript (strict), Vite, TanStack Query, Tailwind, Vitest |

---

## License

MIT
