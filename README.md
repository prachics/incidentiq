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
python seeds/generate.py             # 500 incidents, 66 runbooks, 38 docs, mock infra
```

That is a working, fully-seeded system with no API key required — the default
configuration uses local embeddings and a local LLM.

Optional:

```bash
docker compose up -d langfuse        # tracing UI at localhost:3001
docker compose --profile app up -d   # containerised API at localhost:8000
```

**Requirements:** Docker, Python 3.11+. No API key needed for the default
(free, fully local) configuration.

---

## Evaluation results

Run with `python -m evals.run`. Results are written to a timestamped JSON file
and a markdown summary in `evals/results/`.

| Metric | Target | Local (Qwen 2.5 14B) | Claude | Status |
|---|---|---|---|---|
| Task completion | ≥ 90% | — | — | Phase 3 |
| Grounded response rate | ≥ 90% | — | — | Phase 3 |
| Tool execution success | ≥ 95% | — | — | Phase 3 |
| Retrieval Recall@5 | measured | — | n/a | Phase 2 |
| Median latency | reported | — | — | Phase 3 |
| p95 latency | reported | — | — | Phase 3 |
| Cost per investigation | reported | $0.00 | — | Phase 3 |

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
| 2 | Chunking, embedding, hybrid search, metadata filtering, Recall@5 | next |
| 3 | LangGraph agent, state persistence, tools, retries — then the eval harness | |
| 4 | Approval interrupts, approval API, immutable audit log | |
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
