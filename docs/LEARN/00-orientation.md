# 00 — Orientation: what this system actually is

Read this first. It defines every term the rest of the project assumes, using
IncidentIQ's own parts as the examples. No prior AI background assumed.

---

## The one-paragraph version

An on-call engineer types *"checkout-service is throwing 502s since about
14:00"*. IncidentIQ finds similar past incidents and relevant runbooks, calls
read-only diagnostic tools to gather evidence, reasons across several steps
about what the evidence means, and proposes a fix — citing what it based the
proposal on. If the fix would change production, a human approves it first.

Four capabilities are stacked there, and they're worth separating because they
are four different bodies of knowledge.

---

## 1. LLM — the reasoning engine

A **large language model** takes text and produces text. That's the whole
interface. Everything else is scaffolding around that one operation.

What matters for building on one:

- **Tokens.** Models don't see characters or words, they see *tokens* — chunks
  of roughly 4 characters. "checkout-service" is maybe 4 tokens. You are billed
  per token, input and output separately, and every model has a **context
  window**: a hard limit on how many tokens fit in one request. This is why we
  retrieve a handful of relevant documents instead of pasting in all 500.
- **Stateless.** The model remembers nothing between calls. Any "memory" is you
  resending prior text on the next request. IncidentIQ's `scratchpad` exists
  precisely because the model won't remember step 3 when it runs step 4.
- **It generates plausible text, not true text.** This is the central problem
  the whole architecture is built around. A model asked about an incident it
  knows nothing about will produce a confident, fluent, entirely invented
  answer. That failure has a name — **hallucination** — and the two defences
  against it are *retrieval* (give it real documents) and *groundedness
  measurement* (check afterwards that its claims trace back to those documents).
  Both are below.

## 2. RAG — giving the model real facts

**Retrieval-Augmented Generation.** Before asking the model anything, go find
relevant documents and put them in the prompt.

Why not just fine-tune a model on your incident history? Because retrieval is
cheaper, updatable the instant a new incident is written, and — decisively —
*attributable*. You know exactly which documents produced the answer, so you can
show them to the user and check the answer against them. A fine-tuned model
gives you no such trace.

The mechanism:

- **Embedding.** A model that converts text into a list of numbers — a
  **vector**. Ours produces 384 of them per chunk. The useful property is that
  texts with similar *meaning* land near each other in that 384-dimensional
  space, even with no words in common. "the database is refusing connections"
  and "postgres connection pool exhausted" score as close; keyword search would
  score them as unrelated.
- **Vector database.** Somewhere to store those vectors and answer *"which 5
  are closest to this one?"* quickly. We use **pgvector**, a Postgres
  extension — so vectors live in the same database as everything else, and a
  retrieval query can join against ordinary SQL columns. No second datastore.
- **Chunking.** Documents get split into pieces before embedding, because one
  vector cannot faithfully represent 2000 words. Chunk too small and you lose
  context; too large and the vector blurs into an average of several topics.
  Finding the size that works is empirical, and Phase 2 measures it.
- **Hybrid retrieval.** Vector search is strong on meaning and weak on exact
  strings — error codes, service names, stack-trace fragments. Keyword search
  is the reverse. Run both, then merge the two rankings. Phase 2 covers how.

## 3. Agent — deciding what to do, in a loop

A single LLM call answers a question. An **agent** decides *which questions to
ask*, gathers information, looks at what came back, and decides what to do next.

- **Tool / function calling.** You describe available functions to the model;
  it responds by asking for one to be called with specific arguments. It does
  not execute anything — your code does, then hands the result back. Every real
  action stays under your control. IncidentIQ's tools are things like
  `get_service_logs(service, time_window, level)`.
- **The agent loop.** plan → call a tool → look at the result → decide whether
  that answered the question → plan again. Continue until done or until an
  iteration cap stops it. The cap matters: without it, a confused agent bills
  you forever.
- **Orchestration (LangGraph).** The loop expressed as an explicit graph of
  **nodes** (steps) and **edges** (transitions). We could write it as a `while`
  loop instead — Phase 3 explains at length why we don't, but the short version
  is that a graph gives you checkpointing between nodes and the ability to
  *pause* at a node and resume later, which is exactly what human approval
  needs.
- **Human-in-the-loop.** Read-only tools run automatically. Anything that
  changes production stops and waits for a person. The agent proposes; the human
  decides.

## 4. Evaluation — knowing whether it works

This is the part most projects skip, and the reason this one is worth building.

"It seemed to work when I tried it" is not a claim you can defend. So there are
100 scenarios with known correct answers, and a harness that runs all of them
and reports:

- **Task completion** — did it reach a correct remediation inside the cap?
- **Groundedness** — is every factual claim traceable to a retrieved document,
  or did it invent something? Measured by having a second LLM check each claim
  against the sources (**LLM-as-judge** — imperfect, and Phase 3 discusses the
  limits honestly).
- **Tool success rate** — including recovery from deliberately injected failures.
- **Recall@5** — of the documents that *should* have been retrieved, what
  fraction appeared in the top 5? This is the number that tells you whether the
  retrieval layer works, independent of the model.

**Tracing (Langfuse)** is the companion to this: a record of every LLM call,
retrieval, and tool execution, so when a scenario fails you can see *where*
rather than guess.

---

## How the pieces map to folders

| Folder | What lives there | Phase |
|---|---|---|
| `db/migrations/` | Schema: corpora, mock infra, agent state | 1 |
| `seeds/` | Synthetic incident/runbook/doc generation | 1 |
| `api/incidentiq/rag/` | Chunking, embedding, hybrid search | 2 |
| `api/incidentiq/agent/` | LangGraph nodes, state, checkpointing | 3 |
| `api/incidentiq/tools/` | Diagnostic tools + failure injection | 3 |
| `evals/` | 100 scenarios + scoring harness | 3 |
| `api/incidentiq/api/` | FastAPI routes, SSE streaming, approvals | 4 |
| `frontend/` | The four views | 5 |

---

## The honest caveat

Every number this project reports is measured against a **synthetic** corpus of
generated incidents. That's a real limitation and worth stating out loud rather
than being caught on: a synthetic corpus is more internally consistent than a
real one, so retrieval scores here are optimistic relative to a real company's
messy, contradictory, half-written incident history.

What the numbers *do* legitimately demonstrate is that the measurement
apparatus exists and works — which is the harder thing to build, and the thing
most portfolio projects don't have at all.

---

**Next:** [01 — Postgres, pgvector, and why the schema looks like that](01-postgres-and-pgvector.md)
