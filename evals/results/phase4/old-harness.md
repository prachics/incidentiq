# Eval run 20261012T061059Z

**Model:** `qwen2.5:14b` via `ollama`  
**Scenarios:** 10

| Metric | Target | Result |
|---|---|---|
| Task completion | ≥ 90% | 60.0% |
| Tool execution success | ≥ 95% | 78.1% (32 attempts) |
| Abstention accuracy | — | 100.0% |
| Approval flow exercised | — | 100.0% |
| Recovery under injection | — | 100.0% |
| Median latency | — | 119.5s |
| p95 latency | — | 148.8s |
| Mean tokens / investigation | — | 16,597 in / 683 out |

## By scenario kind

| Kind | n | Task completion | Median latency |
|---|---|---|---|
| approval_required | 2 | 50.0% | 104.9s |
| cascading | 2 | 0.0% | 123.0s |
| no_retrieval | 2 | 100.0% | 65.0s |
| single_service | 2 | 50.0% | 126.7s |
| tool_failure | 2 | 100.0% | 141.8s |

## What the failing scenarios have in common

| Failure mode | Count |
|---|---|
| named the service but not the failure mode | 4 |

> **Reduced iteration budget.** Every scenario was capped at 3 iterations rather than its own budget of 4-8. Task completion measures the capped agent, and this run is not comparable with an uncapped one.
