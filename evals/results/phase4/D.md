# Eval run 20261012T093754Z

**Model:** `qwen2.5:14b` via `ollama`  
**Scenarios:** 10

| Metric | Target | Result |
|---|---|---|
| Task completion | ≥ 90% | 70.0% |
| Tool execution success | ≥ 95% | 80.7% (31 attempts) |
| Abstention accuracy | — | 100.0% |
| Approval flow exercised | — | 100.0% |
| Recovery under injection | — | 50.0% |
| Median latency | — | 115.5s |
| p95 latency | — | 159.5s |
| Mean tokens / investigation | — | 15,881 in / 756 out |

## By scenario kind

| Kind | n | Task completion | Median latency |
|---|---|---|---|
| approval_required | 2 | 100.0% | 106.9s |
| cascading | 2 | 100.0% | 128.4s |
| no_retrieval | 2 | 100.0% | 70.3s |
| single_service | 2 | 50.0% | 115.2s |
| tool_failure | 2 | 0.0% | 139.7s |

## What the failing scenarios have in common

| Failure mode | Count |
|---|---|
| abstained on a scenario that had a discoverable cause | 2 |
| the approval was refused: 422 the proposed arguments for rollback_deploy cannot be executed; approve with edited arguments instead: invalid arguments for rollback_deploy: [{'type': 'value_error', 'loc': | 1 |

> **Reduced iteration budget.** Every scenario was capped at 3 iterations rather than its own budget of 4-8. Task completion measures the capped agent, and this run is not comparable with an uncapped one.
