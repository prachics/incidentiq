# IncidentIQ

An agentic AI production-support platform. An on-call engineer describes a
problem in natural language; the agent retrieves relevant incident history and
runbooks, runs read-only diagnostic tools against a target system, reasons
across multiple steps, and proposes a remediation. Anything that would change
production state requires explicit human approval.

The distinguishing features are the ones most agent demos skip: **state survives
tool failures**, **every action is traced**, and **there is a real evaluation
harness with numbers**.

> **Build status:** Phase 1 (foundation) complete. Phases 2–6 in progress.
> The evaluation table below is empty on purpose — it gets filled from actual
> harness output in Phase 3, not from estimates.

---

## Architecture

```mermaid
graph TB
    subgraph client["Frontend — React + TypeScript"]
        UI[Investigation view]
        EV[Evidence panel]
        AP[Approval modal]
        TR[Trace view]
    end

    subgraph api["API — FastAPI"]
        REST[REST endpoints]
        SSE[SSE stream]
    end

    subgraph graph["Agent — LangGraph"]
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
        RO[Read-only<br/>logs · metrics · deploys · deps]
        WR[Write — approval-gated<br/>restart · scale · rollback]
    end

    subgraph data["PostgreSQL 16 + pgvector"]
        CORP[(Corpora<br/>incidents · runbooks · docs)]
        CHUNK[(chunks<br/>HNSW + GIN)]
        STATE[(Agent state<br/>checkpoints · approvals · audit)]
        MOCK[(Mock infra<br/>logs · metrics · deploys)]
    end

    LF[Langfuse<br/>traces · token counts · latency]

    UI --> REST
    SSE --> UI
    REST --> N1
    N2 --> CHUNK
    CHUNK -.cites.-> CORP
    N4 --> RO --> MOCK
    N8 --> WR --> STATE
    graph <-->|checkpoint after<br/>every node| STATE
    graph -.traces.-> LF

    classDef store fill:#1f2937,stroke:#4b5563,color:#e5e7eb
    classDef gate fill:#7c2d12,stroke:#ea580c,color:#fed7aa
    class CORP,CHUNK,STATE,MOCK store
    class N7,WR gate
```

**The loop that matters:** `plan → execute_tool → reflect` runs until the agent
has enough evidence or hits the iteration cap. `propose → await_approval` uses
LangGraph's interrupt mechanism, so the graph genuinely pauses — the process can
restart while an approval sits pending, and the investigation resumes where it
stopped.

---

## Quickstart

```bash
git clone https://github.com/prachics/incidentiq.git
cd incidentiq
cp .env.example .env

docker compose up -d postgres        # Postgres 16 + pgvector
python scripts/migrate.py            # apply schema
python seeds/generate.py             # 500 incidents, 71 runbooks, 38 docs, mock infra
python -m incidentiq.rag.indexer     # chunk + embed -> 680 searchable chunks
python -m evals.retrieval_eval       # reproduce the Recall@5 number above
```

To run the agent you also need an LLM. The default is local and free:

```bash
brew install ollama && ollama serve &
ollama pull qwen2.5:14b              # ~9GB

python -m evals.run --sample 3       # stratified subset, all 5 scenario kinds
python -m evals.run                  # the full 100-scenario suite
```

Set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY` in `.env` to run against
Claude instead. The provider interface is the only thing that changes.

To compare two runs — a local model against a frontier one, or before and after
a change:

```bash
python -m evals.run --provider ollama    --model qwen2.5:14b   # writes run_A.json
python -m evals.run --provider anthropic --model claude-sonnet-5

python -m evals.compare evals/results/run_A.json evals/results/run_B.json \
    --label-a "qwen2.5:14b" --label-b "Claude"
```

`compare` reports which failure *modes* moved, not just the headline number — a
change that trades one failure mode for another is not an improvement — and
refuses to compare quietly across different scenario counts or iteration caps.

That is a working, fully-seeded system with no API key required — the default
configuration uses local embeddings and a local LLM.

Optional:

```bash
docker compose up -d langfuse        # tracing UI at localhost:3001
docker compose --profile app up -d   # API at localhost:8000 (health, stats, tool catalogue;
                                     #   investigation + approval endpoints land in Phase 4)
```

**Requirements:** Docker, Python 3.11+. No API key needed for the default
(free, fully local) configuration.

---

## Evaluation results

Run with `python -m evals.run`. Results are written to a timestamped JSON file
and a markdown summary in `evals/results/`.

| Metric | Target | Measured | Status |
|---|---|---|---|
| **Retrieval Recall@5** | measured | **0.800** | ✅ Phase 2 |
| Retrieval Hit@5 | — | 0.917 | ✅ Phase 2 |
| Retrieval MRR | — | 0.917 | ✅ Phase 2 |
| Retrieval median latency | — | 28.7 ms | ✅ Phase 2 |
| Task completion | ≥ 90% | **40.0%** | ⚠️ Phase 3 |
| Abstention accuracy | — | **100%** | ✅ Phase 3 |
| Tool execution success | ≥ 95% | 80.0% | ⚠️ Phase 3 |
| Grounded response rate | ≥ 90% | not scored | Phase 3 |
| Cost per investigation | reported | 15.8k in / 0.7k out | ✅ Phase 3 |

Agent figures are from a **10-scenario stratified sample against a local 14B
model (`qwen2.5:14b`) at a 3-iteration cap**, not the full suite at full budget.
They are reported as measured rather than withheld until they look better.

**Four of the six failures are under-commitment, not wrong answers** — the agent
had the evidence and declined to act on it. That is the direct cost of prompts
tuned to abstain rather than fabricate, which is also why abstention scored
100%. The tradeoff is now measured instead of assumed. Full failure taxonomy in
[`docs/EVALS.md`](docs/EVALS.md).

Retrieval configuration: hybrid (pgvector cosine + Postgres FTS, weighted RRF),
service pre-filter on, `BAAI/bge-small-en-v1.5` local embeddings, chunk 320 /
overlap 64. Measured over 12 labelled situations; full ablation, fusion sweep,
chunk sweep, and stated limitations in [`docs/EVALS.md`](docs/EVALS.md).

**Scenario suite — 100 total:**

| Count | Kind |
|---|---|
| 40 | Straightforward single-service incidents |
| 25 | Multi-service / cascading failures |
| 15 | Retrieval returns nothing useful — the agent must say so rather than fabricate |
| 10 | Requiring an approval-gated action |
| 10 | With injected tool failures requiring recovery |

Methodology and per-run history: [`docs/EVALS.md`](docs/EVALS.md).

---

## What is built, and what is not

| Phase | Scope | Status |
|---|---|---|
| 1 | Docker Compose, schema, migrations, seeded corpora + mock infra | **done** |
| 2 | Chunking, embedding, hybrid search, metadata filtering, Recall@5 | **done** |
| 3 | LangGraph agent, state persistence, tools, retries — then the eval harness | **done** |
| 4 | Approval interrupts, approval API, immutable audit log | next |
| 5 | Frontend: investigation, evidence, approval, trace | |
| 6 | MCP server exposing the diagnostic tools | |

---

## Seeded data

The corpus is synthetic but not random. Every incident is an instance of one of
14 **failure archetypes** applied to one of 33 services in a hand-authored
dependency graph, so the data is internally consistent in ways the evaluation
depends on:

- An incident that blames a dependency blames a **real** dependency of that
  service. Verified by query: 103 root causes name another service, and **zero**
  name a service that is not an actual dependency.
- Cascading failures follow real edges — `payment-db` degrading surfaces as
  symptoms in `payment-service → checkout-service → api-gateway`.
- The same archetype on a different service produces a document that is
  topically similar but factually wrong for a given incident, which is what
  makes retrieval scores meaningful rather than trivial.
- Generation is seeded (`RANDOM_SEED = 20260101`), so two runs produce identical
  data and eval numbers stay comparable across runs.

**Honest caveat:** a synthetic corpus is more internally consistent than a real
company's incident history, so retrieval scores here are optimistic relative to
production. What the numbers legitimately demonstrate is that the measurement
apparatus exists and works.

---

## Findings worth reading

Three results from Phase 2 that were not what I expected going in.

**Query enrichment matters more than any retrieval technique — Recall@5 0.461 → 0.800.**
The first version of this eval scored retrieval on the bare on-call report
(*"something's wrong with auth-service, customers are complaining, help?"*)
against labels defined by failure archetype. That query carries no information
about the failure mode, so the measurement was impossible by construction. The
agent retrieves *after* `intake` extracts entities and after tools return log
signatures. Both numbers are reported — the gap between them is the argument for
the agent loop existing at all.

**Hybrid retrieval buys exactly what the metadata filter would have bought.**
With a service pre-filter, hybrid and vector-only tie at 0.800. Without it,
vector drops to 0.728 while hybrid stays at 0.800. Hybrid's value here is not
beating dense retrieval — it is being insensitive to whether the filter is
available, which is the condition on the first turn, when the agent does not yet
know which service is at fault.

**Hybrid only started helping once the corpus was realistic.** Initially it
scored at or below vector everywhere. The cause was a corpus defect, not a
fusion defect: `HikariPool`, `OutOfMemoryError`, and `x509` appeared in
`log_entries` and **zero** times in the incident write-ups, so the keyword
ranker had no identifiers to match. Migration `005` adds quoted error lines to
incidents, as a real postmortem would. The general lesson: hybrid retrieval
beats dense retrieval when documents contain tokens embeddings cannot represent,
and a corpus of pure prose gains little.

Full working — including a chunk-size sweep that came out a plateau rather than
a peak, and a reproducibility check on the approximate index — in
[`docs/EVALS.md`](docs/EVALS.md).

### From Phase 3 — the agent

**JSON mode guarantees syntax, not schema.** Constrained to valid JSON, a local
14B model past ~3000 prompt tokens returned `{"incident_updates": ...}` where
`root_cause` was expected. Parsing succeeded, every field read back `None`, and
the agent emitted an empty proposal **having reasoned its way to the correct
answer** — `reflect` had already said *"thread pool saturation and potential
deadlocks"*. Fixed by passing a JSON Schema to the sampler for
grammar-constrained decoding. Then again, because with no `required` array the
grammar still let the model skip `root_cause` — which meant replacing nullable
types with empty-string sentinels, since a field can only be required if it has
a representable way to say "nothing here".

**Retry classification matters more than retry count.** An unknown service name
was classified as a transient failure, so the loop retried identical bad
arguments three times per iteration and burned the whole budget without one
successful call. Argument errors now break the loop and send back a specific
correction — *"Did you mean: checkout-service?"* via trigram similarity.

**Existence checks are not category checks.** The model proposed
`get_service_logs` as a *remediation*. It is a registered tool, so an existence
check passed — and it would have opened an approval request for an action that
changes nothing.

## Documentation

| Document | Contents |
|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | Why LangGraph over a plain loop, why hybrid retrieval, how state recovery works, known failure modes |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Running log of every non-obvious choice and its reasoning |
| [`docs/EVALS.md`](docs/EVALS.md) | Evaluation methodology, scenario breakdown, results over time |
| [`docs/LEARN/`](docs/LEARN/) | Plain-English explainers for every concept the project introduces |

---

## Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| Backend | Python 3.12, FastAPI |
| Vector store | PostgreSQL 16 + pgvector (HNSW) |
| Tracing / evals | Langfuse (self-hosted) |
| Frontend | React + TypeScript + Vite |
| Packaging | Docker Compose |
| LLM | Swappable: Ollama (local, default), Anthropic, deterministic stub |
| Embeddings | `BAAI/bge-small-en-v1.5`, 384-dim, local |

---

## License

MIT
