# Eval run 20261012T091910Z

**Model:** `qwen2.5:14b` via `ollama`  
**Scenarios:** 10

| Metric | Target | Result |
|---|---|---|
| Task completion | ≥ 90% | 80.0% |
| Tool execution success | ≥ 95% | 80.7% (31 attempts) |
| Abstention accuracy | — | 100.0% |
| Approval flow exercised | — | 100.0% |
| Recovery under injection | — | 50.0% |
| Median latency | — | 116.1s |
| p95 latency | — | 160.1s |
| Mean tokens / investigation | — | 15,899 in / 754 out |

## By scenario kind

| Kind | n | Task completion | Median latency |
|---|---|---|---|
| approval_required | 2 | 100.0% | 106.7s |
| cascading | 2 | 100.0% | 124.6s |
| no_retrieval | 2 | 100.0% | 70.8s |
| single_service | 2 | 50.0% | 113.1s |
| tool_failure | 2 | 50.0% | 143.2s |

## What the failing scenarios have in common

| Failure mode | Count |
|---|---|
| the approval was refused: 422 the proposed arguments for rollback_deploy cannot be executed; approve with edited arguments instead: invalid arguments for rollback_deploy: [{'type': 'value_error', 'loc': | 1 |
| abstained on a scenario that had a discoverable cause | 1 |

> **Reduced iteration budget.** Every scenario was capped at 3 iterations rather than its own budget of 4-8. Task completion measures the capped agent, and this run is not comparable with an uncapped one.
