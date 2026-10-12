# Eval run 20261012T084155Z

**Model:** `qwen2.5:14b` via `ollama`  
**Scenarios:** 10

| Metric | Target | Result |
|---|---|---|
| Task completion | ≥ 90% | 70.0% |
| Tool execution success | ≥ 95% | 83.9% (31 attempts) |
| Abstention accuracy | — | 50.0% |
| Approval flow exercised | — | 100.0% |
| Recovery under injection | — | 50.0% |
| Median latency | — | 109.5s |
| p95 latency | — | 128.7s |
| Mean tokens / investigation | — | 16,424 in / 695 out |

## By scenario kind

| Kind | n | Task completion | Median latency |
|---|---|---|---|
| approval_required | 2 | 100.0% | 98.3s |
| cascading | 2 | 50.0% | 116.8s |
| no_retrieval | 2 | 50.0% | 65.4s |
| single_service | 2 | 50.0% | 109.8s |
| tool_failure | 2 | 100.0% | 128.6s |

## What the failing scenarios have in common

| Failure mode | Count |
|---|---|
| named the service but not the failure mode | 2 |
| fabricated an answer for a query with no relevant evidence - the worst failure mode in the suite | 1 |

> **Reduced iteration budget.** Every scenario was capped at 3 iterations rather than its own budget of 4-8. Task completion measures the capped agent, and this run is not comparable with an uncapped one.
