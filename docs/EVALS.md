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

---

# Phase 2 results — retrieval

Reproduce with:

```bash
python -m evals.retrieval_eval --ablation      # the table below
python -m evals.retrieval_eval --fusion-sweep  # RRF k and ranker weights
python -m evals.retrieval_eval --sweep         # chunk size / overlap
```

Labelled set: 12 situations, 2–15 relevant documents each (median 5).
Configuration: `BAAI/bge-small-en-v1.5` (384-dim, local), chunk 320 / overlap 64,
680 chunks over 609 documents.

## Headline

| Metric | Value |
|---|---|
| **Recall@5** (capped) | **0.800** |
| Recall@5 (uncapped) | 0.608 |
| Hit@5 | 0.917 |
| MRR | 0.917 |
| Median latency | 28.7 ms |

Configuration: hybrid, service pre-filter on, enriched query, incidents corpus.

## Ablation

Retrieval mode × service pre-filter × query condition.

| query | mode | filter | Recall@5 | Hit@5 | MRR | p50 ms |
|---|---|---|---|---|---|---|
| cold | vector | off | 0.300 | 0.333 | 0.292 | 17.1 |
| cold | vector | on | 0.461 | 0.583 | 0.454 | 14.9 |
| cold | keyword | off | 0.294 | 0.583 | 0.406 | 1.5 |
| cold | keyword | on | 0.378 | 0.667 | 0.478 | 0.7 |
| cold | hybrid | off | 0.300 | 0.417 | 0.294 | 29.9 |
| cold | hybrid | on | 0.461 | 0.583 | 0.450 | 26.9 |
| enriched | vector | off | 0.728 | 0.917 | 0.917 | 25.0 |
| enriched | vector | on | 0.800 | 0.917 | 0.917 | 16.7 |
| enriched | keyword | off | 0.750 | 0.833 | 0.750 | 22.9 |
| enriched | keyword | on | 0.767 | 0.917 | 0.861 | 2.5 |
| enriched | hybrid | off | 0.800 | 0.917 | 0.875 | 59.9 |
| **enriched** | **hybrid** | **on** | **0.800** | **0.917** | **0.917** | **28.7** |

### Three findings, one of which contradicts what I expected

**1. Query enrichment is worth more than anything else: 0.461 → 0.800.**

The first version of this evaluation scored retrieval on the bare on-call
report — *"something's wrong with auth-service, customers are complaining,
help?"* — against a label set defined by failure archetype. That query contains
no information about the failure mode, so the measurement was asking retrieval
to do something logically impossible, and the resulting numbers described the
eval rather than the retriever.

The agent does not retrieve from that text. Its `retrieve` node runs after
`intake` extracts entities and, on later iterations, after tools return
evidence. The enriched condition supplies the distinct error signatures a first
`get_service_logs` call returns — which is what the agent actually holds.

Both numbers are reported. Cold is the floor: what retrieval achieves before any
diagnostic work. Enriched is the operating condition. The gap between them is an
argument for the agent loop existing at all — retrieval genuinely gets better
after the first tool call, which is why `plan → execute_tool → reflect` loops
back to retrieval rather than running it once.

**2. Hybrid retrieval buys what the metadata filter would have bought.**

| | filter off | filter on |
|---|---|---|
| vector only | 0.728 | **0.800** |
| hybrid | **0.800** | 0.800 |

With the service filter on, hybrid and vector tie exactly. With it off, hybrid
recovers the entire 0.072 the filter was providing.

This matters operationally because the filter is not always available. On the
first turn the agent may not yet know which service is at fault — that is often
what it is trying to establish. Hybrid retrieval is what keeps recall up in
exactly that condition, and then costs nothing once the service is known.

Reported honestly: at the best configuration, hybrid does **not** beat vector
search on recall. Its value is being insensitive to whether the filter is
available.

**3. Hybrid only started helping once the corpus was realistic.**

The first fusion measurements had hybrid at or below vector-only in every
configuration. The cause was a corpus defect, not a fusion defect: incident
write-ups contained no error strings. `HikariPool`, `OutOfMemoryError`, `x509`,
and `NOT_ENOUGH_REPLICAS` all appeared in `log_entries` and **zero** times in the
incident corpus. With no identifiers to match, the keyword ranker had nothing
distinctive to contribute and fusion could only dilute the vector ranking.

Migration `005_incident_error_signatures.sql` adds the quoted error lines to
incidents, as a real postmortem would. After that change the keyword ranker
became competitive on its own (Recall@5 0.767) and fusion became worthwhile.

The general lesson is about technique selection: hybrid retrieval is not
universally better than dense retrieval. It is better when documents contain
tokens embeddings cannot represent. A corpus of pure prose gains little.

## Fusion parameters

Grid over RRF `k` and the keyword ranker's weight. Enriched query, filter on.

| k | keyword weight | Recall@5 | Hit@5 | MRR |
|---|---|---|---|---|
| — | vector only | 0.800 | 0.917 | 0.917 |
| — | keyword only | 0.767 | 0.917 | 0.861 |
| 5 | 0.25 | 0.800 | 0.917 | 0.917 |
| 5 | 0.50 | 0.800 | 0.917 | 0.917 |
| 5 | 1.00 | 0.783 | 0.917 | 0.917 |
| 10 | 0.50 | 0.800 | 0.917 | 0.917 |
| 20 | 0.50 | 0.800 | 0.917 | 0.917 |
| 60 | 0.50 | 0.800 | 0.917 | 0.917 |
| 60 | 1.00 | 0.783 | 0.917 | 0.917 |

`k` makes no measurable difference on this corpus — 5 and 60 score identically —
so the canonical 60 is kept. The keyword **weight** does matter, in the
direction that matters: at 1.00, the naive default, the weaker ranker drags
results down. At 0.50 it contributes without dominating. Chosen: `k=60`,
`keyword_weight=0.5`.

Plain RRF weights both rankers equally, which rewards *agreement* over
*confidence*: at k=60 a chunk ranked 10th by both scores 0.029, beating a chunk
ranked 1st by one ranker alone at 0.016. When one ranker is weaker, its mediocre
picks get promoted for being seconded. The weight is the correction.

## Chunk size and overlap

Overlap held at 20% of chunk size. Enriched query, hybrid, filter on.

| chunk tokens | overlap | chunks | mean tokens | Recall@5 | Hit@5 | MRR |
|---|---|---|---|---|---|---|
| 96 | 20 | 1967 | 79 | 0.783 | 0.917 | 0.917 |
| 128 | 24 | 1485 | 101 | 0.783 | 0.917 | 0.917 |
| 192 | 40 | 986 | 148 | 0.800 | 0.917 | 0.917 |
| 256 | 48 | 703 | 193 | 0.800 | 0.917 | 0.917 |
| **320** | **64** | **680** | **200** | **0.800** | **0.917** | **0.917** |
| 448 | 88 | 632 | 211 | 0.800 | 0.917 | 0.917 |
| 640 | 128 | 609 | 216 | 0.800 | 0.917 | 0.917 |

**The result is a plateau, not a peak.** Everything from 192 upward scores
identically. The reason is visible in the mean-tokens column: even at a 640-token
budget the mean chunk is 216 tokens, because most documents are shorter than any
of these limits and are never split at all. Only the longer runbooks split, and
they are a minority of the corpus.

Selection rule: take every setting within tolerance of the best score and choose
the **middle of the plateau**, not an edge. An edge setting is one corpus change
away from falling off; on a 12-situation labelled set that robustness is worth
more than a third decimal place. That gives **320 / 64**.

Below 192 recall degrades slightly, consistent with chunks becoming too small to
hold a complete symptom-plus-cause statement.

## Measurement reproducibility

HNSW is an approximate index and its construction involves randomisation, so
"did the number change or did the index?" is a real question. Rebuilding the
index four times at identical settings and re-measuring:

| build | chunks | vector Recall@5 | hybrid Recall@5 |
|---|---|---|---|
| 1 | 680 | 0.800 | 0.800 |
| 2 | 680 | 0.800 | 0.800 |
| 3 | 680 | 0.800 | 0.800 |
| 4 | 680 | 0.800 | 0.800 |

Standard deviation 0.0000. At this corpus size the index is effectively
deterministic, so differences in the tables above are real rather than noise.
This check should be repeated if the corpus grows by an order of magnitude,
where HNSW's approximation starts to bite.

## Limitations of these numbers

- **12 situations is a small labelled set.** One situation is worth 0.083 of the
  Recall@5 average. Differences smaller than that should not be treated as real.
- **The corpus is synthetic** and more internally consistent than a real incident
  history, so these figures are optimistic relative to production.
- **Structural relevance is a proxy.** A genuinely useful incident from an
  unrelated service is scored as a miss, making the reported recall a lower
  bound on real usefulness.
- **The enriched query uses ground-truth log data.** It is built from the actual
  error signatures present for that service, which is what a working
  `get_service_logs` returns — but it assumes the tool call succeeded. Phase 3
  measures the end-to-end path including tool failures.
- **No abstention cases in the current labelled set.** The Phase 1 dependency-graph
  fix changed the neighbourhood computation and every situation now has relevant
  history. The 15 abstention scenarios required by the suite will be
  constructed deliberately in Phase 3 rather than left to chance.

---

# Phase 3 results — the agent

Reproduce with:

```bash
python -m evals.run                       # the full 100-scenario suite
python -m evals.run --sample 2 --max-iterations 3 --no-judge   # a dev subset
```

## How to read these numbers

Four things about this run, stated before the table rather than after it.

**The iteration budget was capped.** Local inference runs at ~24 tok/s, so a
full-budget scenario is 15-20 LLM calls and several minutes; the whole suite is
an overnight job. Capped runs are reported as capped, and the harness writes the
cap into the results file so a capped number can never be mistaken for a full
one. **Task completion under a cap is a lower bound**, and not for a subtle
reason: with fewer iterations the agent frequently reaches a correct diagnosis
and stops there without proposing a remediation, which the grader counts as a
miss. That is the right grading — "did it reach a correct remediation" is the
metric — but it means the gap between capped and uncapped is larger than the
missing evidence alone would suggest.

**Groundedness was not scored in this run.** The judge roughly doubles wall
clock. A groundedness number also needs a calibration kappa beside it to be
interpretable, and that has not been run either. Reporting the number without
it would be worse than omitting it.

**The corpus was verified clean beforehand** — 9,927 log rows, 89,856 metric
rows, zero orphaned fixture rows — because earlier runs leaked planted evidence
that would have quietly inflated later scenarios.

**Every scenario got the evidence its failure would leave.** Without that, most
scenarios were unanswerable from tool output and the suite was measuring
something other than what it claimed.

## What the eval measured before it was trustworthy

Four measurement bugs were found by running the harness and reading individual
results rather than the aggregate. All four would have produced numbers that
looked plausible.

**1. Grading against one narrative variant.** Each archetype carries two
root-cause stories and the corpus contains both. A scenario drew one and graded
against that sentence alone, so an agent that diagnosed `disk_full` correctly
and described the other variant scored 0 of 5 keywords. Ground truth is now the
archetype — which is determinate — with keywords pooled across variants. The
same answer scores 6.

**2. Scenarios with no evidence to find.** The corpus plants error signals for
12 "live situations"; the generator produces 100 scenarios across every service
and archetype. **88 had no trace anywhere the diagnostic tools could look.** The
symptom was an agent doing the right thing and being marked wrong — asked about
a memory leak it found no ERROR logs, reasoned the service was probably waiting
on a dependency, and said so while noting it had no evidence. The eval was
measuring whether it could guess an archetype from retrieval alone.
`evals/fixtures.py` now plants the evidence and removes it afterwards.

**3. Fixture cleanup that did not survive an unclean exit.** Row ids lived in a
Python object, so `kill -9` orphaned them — 137 log rows, 1 deploy, ~153 metric
rows, the last with no marker distinguishing them from the base corpus. Leaked
evidence becomes background noise for every later scenario and the numbers drift
with nothing failing. Cleanup state now lives in the database and is purged at
startup, scoped so a run never deletes a concurrent run's fixtures.

**4. Grading a narrower slice than the agent's answer.** `identified_cause`
searched `root_cause` + `remediation` only. An agent that concluded "the root
cause of the catalog-db incident is disk exhaustion, confidence 0.9" — naming
the service in its summary and remediation arguments but not inside the
`root_cause` string — was graded as not having named the service. Grading now
spans the whole conclusion, with a test asserting that naming the *wrong*
service everywhere still fails.

The pattern in all four: **the measurement was narrower than the thing being
measured.** Each was found by reading a single failing scenario end to end, not
by looking at an aggregate.

---

## Results over time

Populated from `evals/results/`.

| Date | Phase | Recall@5 | Hit@5 | MRR | Completion | Grounded | Tool success |
|---|---|---|---|---|---|---|---|
| 2026-09-04 | 2 | 0.800 | 0.917 | 0.917 | — | — | — |
