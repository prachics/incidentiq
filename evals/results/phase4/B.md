# Eval run 20261012T090030Z

**Model:** `qwen2.5:14b` via `ollama`  
**Scenarios:** 10

| Metric | Target | Result |
|---|---|---|
| Task completion | ≥ 90% | 70.0% |
| Tool execution success | ≥ 95% | 83.9% (31 attempts) |
| Abstention accuracy | — | 100.0% |
| Approval flow exercised | — | 100.0% |
| Recovery under injection | — | 50.0% |
| Median latency | — | 117.1s |
| p95 latency | — | 158.2s |
| Mean tokens / investigation | — | 16,618 in / 736 out |

## By scenario kind

| Kind | n | Task completion | Median latency |
|---|---|---|---|
| approval_required | 2 | 100.0% | 106.0s |
| cascading | 2 | 50.0% | 124.6s |
| no_retrieval | 2 | 100.0% | 63.5s |
| single_service | 2 | 0.0% | 118.2s |
| tool_failure | 2 | 100.0% | 143.8s |

## What the failing scenarios have in common

| Failure mode | Count |
|---|---|
| named the service but not the failure mode | 2 |
| the approval was refused: 422 the proposed arguments for rollback_deploy cannot be executed; approve with edited arguments instead: no deploy history for 'redis-cart' | 1 |

> **Reduced iteration budget.** Every scenario was capped at 3 iterations rather than its own budget of 4-8. Task completion measures the capped agent, and this run is not comparable with an uncapped one.
