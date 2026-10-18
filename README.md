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

**Status:** Phases 1–7 complete and measured: the agent, its evaluation, an approval API, a web UI,
tracing (Jaeger or Langfuse), and an MCP server for the read-only tools.
Every number below comes from the harness in this repository and can be
reproduced with the commands given. CI runs the backend tests against Postgres with
pgvector, and the web app's typecheck, tests and build, on every push.

![An investigation in the web UI on a local 14B model, sped up: payment-service hangs;
the agent reads payment-service's and payment-db's logs, concludes payment-db is out of
disk space, and stops at the approval panel with its proposed action](docs/demo.gif)

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
docker compose up -d jaeger        # traces: set OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318, open :16686
docker compose --profile app up -d # API at :8000, mock control plane at :8100
```

---

## Results

### Retrieval (Phase 2)

| Metric | Result |
|---|---|
| **Recall@5** (capped) | **0.800** |
| Recall@5 (uncapped) | 0.608 |
| Hit@5 | 0.917 |
| MRR | 0.917 |
| Median latency | 28.7 ms |

Hybrid retrieval — pgvector cosine similarity plus Postgres full-text search,
fused with weighted reciprocal rank fusion — over `BAAI/bge-small-en-v1.5`
embeddings, 320-token chunks with 64-token overlap, service pre-filter applied
before ranking. Measured across 12 labelled situations with 2–15 relevant
documents each, searching the incident corpus with the query the agent holds
after its first log read (the report plus the observed error signatures).
*Capped* recall divides by min(relevant, 5), the most a top-5 list can find;
*uncapped* divides by every relevant document. Recall figures elsewhere in this
README are capped. [`docs/EVALS.md`](docs/EVALS.md) has the full ablation.

### Agent

| Metric | Target | Result |
|---|---|---|
| Task completion | ≥ 90% | **70.0%** |
| Abstention accuracy | — | **100%** |
| Tool execution success | ≥ 95% | 80.7% (31 attempts) |
| Recovery after injected tool failure | — | 50% |
| Median / p95 latency | — | 108.3s / 157.4s |
| Tokens per investigation | — | 15.9k in / 0.8k out |
| Groundedness | ≥ 90% | not yet scored |

| Scenario kind | n | Completion |
|---|---|---|
| Nothing retrievable (must abstain) | 2 | **100%** |
| Approval-gated action | 2 | **100%** |
| Cascading / multi-service | 2 | **100%** |
| Single service | 2 | 50% |
| Injected tool failures | 2 | 0% |

**Read these numbers carefully.** They come from a 10-scenario stratified sample
against a local 14-billion-parameter model at a 3-iteration cap and a fixed
sampling seed — not the full 100-scenario suite at full budget. They measure the
capped agent, which often stops at a diagnosis before proposing an action, and
one scenario is ten points. Groundedness is
unscored because a groundedness figure without a calibration kappa beside it is
not interpretable, and that calibration has not been done.

**The measurement was fixed before the agent was changed.** The harness had been
approving arguments the approval API refuses, grading against one narrative per
failure mode word for word (and later, grading advice as if it were the
diagnosis), planting metric anomalies that hourly averages hid, and letting
sampling alone move the score ten points. With those fixed, the Phase 3 agent
scores 70%. Three agent changes, measured one at a time on the same harness and
regraded together, scored 70%, 80% and 70%, each moving individual scenarios both
ways. The third costs a scenario on purpose: the agent may no longer propose an
action on a service whose own logs or metrics, read in this investigation, show
nothing abnormal (no error lines, no metric moved 30% or more either way), which once
led it to apply a past incident's fix to a service with nothing wrong. When an
injected tool failure hides that evidence, it abstains instead. The tables above
are the agent as it stands, after review fixes, which scored the same scenario for
scenario as the third.

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
        N4 --> N4b[retrieve_again]
        N4b --> N5[reflect]
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
    MCPC([MCP client]) -->|stdio| MCPS[MCP server<br/>read-only tools only] --> RO
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

## Observability

**One request, one trace; one investigation, linked traces.** Starting an
investigation produces one trace: the request, the worker job, every graph node in
order, and every tool attempt and LLM call under its node. The worker carries the
context across its thread, and the control plane continues it from the
`traceparent` header, so an approved action's HTTP call is in the trace too. A
decision or a resume later runs in the trace of *that* request, and its job span
links back to the investigation's first job. LLM spans follow the OpenTelemetry
GenAI conventions (`gen_ai.request.model`, `gen_ai.usage.*_tokens`); each retry is
its own span; a node that handled a failure is still an error span, with the reason.

```bash
docker compose up -d jaeger
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 uvicorn incidentiq.api.main:app
open http://localhost:16686        # service: incidentiq-api
```

Retrieval is a span of its own under the node that searched (`retrieval hybrid`,
with the candidates each ranker returned and the result count), and the job span
carries `session.id` set to the investigation id.

**Viewing traces in Langfuse.** The same traces go to Langfuse, with no Langfuse
SDK: Langfuse accepts OTLP and reads the GenAI attributes. Each LLM call shows as a
generation with its model, `max_tokens`, input and output tokens and latency;
retrieval and each tool attempt are spans under their node; and the Sessions view
groups an investigation's traces (the start, then each decision or resume) by
`session.id`. Use Langfuse Cloud or a
[self-hosted Langfuse](https://langfuse.com/self-hosting/docker-compose), create a
project, and point the exporter at it with the project's keys:

```bash
AUTH=$(printf '%s' "pk-lf-...:sk-lf-..." | base64 | tr -d '\n')
export OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel   # self-hosted: http://localhost:3000/api/public/otel
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic%20$AUTH"                  # %20 is the space
uvicorn incidentiq.api.main:app
```

Langfuse's own Compose file maps its Postgres to `127.0.0.1:5432`, the port
IncidentIQ's Postgres uses: change one of the two before starting both. Checked
against a self-hosted Langfuse 4: one investigation on `qwen2.5:7b` arrived as one
trace with its 6 LLM calls as generations, token counts included, 3 retrieval spans
and 2 tool spans, in the session named after the investigation.

**Metrics** at `GET /metrics` (Prometheus): runs started and settled by status
(crashed runs included), time per node and LLM latency by outcome (failures are
timed too), tool attempts by tool and status, tokens per model, approved actions
by outcome (executed, already executed, failed, invalid arguments) with the
attempts each took, and approval decisions.

## MCP server

The five read-only diagnostic tools - `get_service_logs`, `get_metrics`,
`get_recent_deploys`, `get_service_dependencies` and `search_similar_incidents` -
are also an MCP server, so any MCP client can investigate with them against the
same seeded data. The write tools (`restart_service`, `scale_service`,
`rollback_deploy`) are not exposed: they run only after a person approves them
through the API. Two guards keep them out: the server lists and accepts only tools
without `requires_approval`, and every call runs on a read-only database session,
so a tool that tried to write would be refused by Postgres. Failure injection is
off for MCP calls.

```bash
python -m incidentiq.mcp_server    # stdio; an MCP client starts it, you don't run it by hand
```

Claude Code, from the repository root:

```bash
claude mcp add incidentiq -e DATABASE_URL=postgresql://incidentiq:incidentiq_dev_password@localhost:5432/incidentiq \
  -- "$PWD/.venv/bin/python" -m incidentiq.mcp_server
```

Claude Desktop or any client configured with JSON (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "incidentiq": {
      "command": "/absolute/path/to/incidentiq/.venv/bin/python",
      "args": ["-m", "incidentiq.mcp_server"],
      "env": {
        "DATABASE_URL": "postgresql://incidentiq:incidentiq_dev_password@localhost:5432/incidentiq"
      }
    }
  }
}
```

The client starts the server in its own working directory, where `.env` is not
found, so `DATABASE_URL` is passed explicitly. If the corpus was indexed with a
non-default embedding model, pass `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL` and
`EMBEDDING_DIM` too: the search embeds queries with whatever model is configured. To try the tools without a client,
the MCP Inspector opens a browser UI:

```bash
npx @modelcontextprotocol/inspector .venv/bin/python -m incidentiq.mcp_server
```

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
| 6 | Tracing and metrics: OpenTelemetry traces across threads and HTTP, linked per investigation, viewable in Jaeger or Langfuse; Prometheus | **complete** |
| 7 | MCP server exposing the read-only diagnostic tools (write tools stay behind approval) | **complete** |

---

## Testing

695 backend tests (the API, the agent, the MCP server and the evaluation harness) and 23 frontend tests.
Beyond the usual unit coverage, four areas are tested specifically:

- **State recovery.** A test crashes a run mid-investigation with an injected
  failure, constructs a fresh runner sharing nothing but the database, resumes it, and
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

Tests that need Postgres skip cleanly when it is unavailable; `INCIDENTIQ_REQUIRE_DB=1`
makes that a failure. CI (`.github/workflows/ci.yml`) sets it, migrates, seeds and
indexes a fresh database (CPU-only torch for the embeddings), then runs ruff and the
backend tests; a second job typechecks, tests and builds the web app.

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
| Observability | OpenTelemetry (OTLP → Jaeger or Langfuse), Prometheus metrics |
| Tool access | MCP server (official Python SDK, stdio) |
| Frontend | React 18, TypeScript (strict), Vite, TanStack Query, Tailwind, Vitest |

---

## License

MIT, see [`LICENSE`](LICENSE).
