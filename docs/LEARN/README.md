# Learning notes

Plain-English explainers for every concept IncidentIQ introduces, written as the
project is built. No prior AI background assumed.

These exist to be reread. The goal is being able to explain the system, not just
having built it.

| # | Note | Phase |
|---|---|---|
| 00 | [Orientation — what this system actually is](00-orientation.md) | 1 |
| 01 | [Postgres, pgvector, and why the schema looks like that](01-postgres-and-pgvector.md) | 1 |
| 02 | [Chunking, embeddings, and hybrid retrieval](02-chunking-embeddings-hybrid-retrieval.md) | 2 |
| 03 | Measuring retrieval: Recall@5 and what it does not tell you | 2 |
| 04 | [Agent loops, LangGraph, and why state lives in Postgres](04-agents-langgraph-and-state.md) | 3 |
| 05a | Tool calling, schemas, and failure injection — folded into note 04 | 3 |
| 05 | [Evaluating an agent: completion, groundedness, LLM-as-judge](05-evaluating-an-agent.md) | 3 |
| 07 | Human-in-the-loop: interrupts, approvals, audit | 4 |
| 08 | Streaming a graph to a browser with SSE | 5 |
| 09 | MCP — exposing tools to any client | 6 |
