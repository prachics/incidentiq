# Evaluation

> The harness lands in Phase 3. This document defines the methodology **before**
> any numbers exist, so the metrics cannot be quietly redefined to flatter the
> results later.

## Principles

1. **The harness is built before the polish.** Phase 3, immediately after the
   agent — not at the end.
2. **Numbers come from the harness or they do not appear.** No estimates in the
   README, no "roughly", no cherry-picked single runs.
3. **Failures are reported alongside successes.** A section on what the failing
   scenarios have in common is part of the output, not an appendix.

## Scenario suite — 100 total

| Count | Kind | What it tests |
|---|---|---|
| 40 | Single-service incidents | The base path: retrieve, diagnose, propose |
| 25 | Multi-service / cascading | Whether the agent distinguishes victim from cause |
| 15 | Nothing useful retrievable | **Abstention.** The agent must say it found nothing |
| 10 | Approval-gated action | Interrupt, approval, resume |
| 10 | Injected tool failures | Retry and recovery |

Each scenario is a YAML file specifying the query, seeded infrastructure state,
expected root cause, expected tool sequence (order-independent), and acceptable
remediations.

The 15 abstention scenarios exist because the most damaging failure mode of a
system like this is not being wrong — it is being confidently wrong about an
incident it has no information on. A system that scores 100% on the other 85 and
fabricates on these 15 is worse than useless in production.

## Metrics

| Metric | Definition |
|---|---|
| **Task completion** | Reached a remediation in the acceptable set, within the iteration cap |
| **Groundedness** | Share of factual claims traceable to a retrieved chunk, scored by LLM-as-judge |
| **Tool success rate** | Successful executions ÷ total attempts, **retries counted as attempts** |
| **Recall@5** | Of the labelled relevant documents, the fraction in the top 5 |
| **Latency** | Median and p95 wall-clock per investigation |
| **Cost** | Input + output tokens per investigation, priced per the configured model |

### Ground truth for Recall@5

Relevance is defined **structurally**, not by hand-labelling. An incident is
relevant to a query if it is the same failure archetype AND involves either the
same service or one in its immediate dependency neighbourhood. This is generated
alongside the corpus into `seeds/live_situations.json`.

The advantage is that nobody's judgement is in the loop, so the label set cannot
drift toward whatever the retriever happens to return. The limitation is that
structural relevance is a proxy: a genuinely useful incident from an unrelated
service would be scored as a miss. That is a real weakness of the measurement
and is stated here rather than discovered by an interviewer.

Situations with **zero** relevant documents are excluded from the Recall@5
average — recall is undefined with an empty label set — and scored on abstention
instead. They are flagged `expect_no_useful_retrieval` in the manifest.

### Limits of LLM-as-judge

Groundedness is scored by a second model checking each claim against retrieved
context. This is standard practice and it is genuinely imperfect:

- The judge shares failure modes with the model it grades, especially when both
  are the same model — a shared blind spot goes undetected.
- Judges reward fluent, confident text, which is exactly the failure mode being
  tested for.
- Scores are not stable across runs; the same claim can be graded differently.

Mitigations used here: a fixed judge prompt with explicit claim-by-claim output,
a judge model distinct from the agent model where possible, and a manually
graded subset to calibrate the judge's agreement with a human. That agreement
rate is reported alongside the groundedness number, because a groundedness score
means nothing without it.

## Results over time

Populated from `evals/results/`. Empty until Phase 3.

| Date | Commit | Model | Completion | Grounded | Tool success | Recall@5 | p95 latency | Cost/investigation |
|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | — |
