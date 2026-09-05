# 05 — Evaluating an agent

How you find out whether the thing works, and why most of the difficulty is in
the measurement rather than the model.

---

## Why "it seemed to work when I tried it" is not enough

Three demos going well tells you almost nothing. You chose the demos, you
remember the good runs, and you cannot tell a 70% system from a 90% one by feel.

So: **100 scenarios with known correct answers**, run by one command, producing a
report.

The composition is deliberate:

| Count | Kind | What it tests |
|---|---|---|
| 40 | Single service | The base path |
| 25 | Cascading | Whether it distinguishes victim from cause |
| 15 | **Nothing retrievable** | **Abstention** |
| 10 | Approval-gated | Interrupt, decide, resume |
| 10 | Injected failures | Retry and recovery |

The 15 abstention scenarios matter most. The worst failure mode of a system like
this is not being wrong — it is being **confidently wrong about an incident it
has no information on**, which is exactly the situation where the engineer has
no independent way to check. A system that scores 100% on the other 85 and
fabricates on these 15 is worse than useless.

---

## Where ground truth comes from

Every incident in the corpus is a **failure archetype applied to a service**.
That is what makes a correct answer exist at all: the scenario's expected root
cause *is* the archetype's root cause. Grading is not a judgement call.

Scenarios are **generated, not hand-written**, for the same reason retrieval
labels are structural: a suite written after watching the agent drifts toward
what the agent already does.

### The grading bug worth knowing about

The first real run reported:

```
[1/5] fail  SC-AP-001  50.5s  named the service but not the mechanism (0/5 keywords)
```

Inspecting it: the scenario expected *"autovacuum could not keep up, dead tuples
filled the volume"*. The agent said *"the data volume reached 100%… clearing
archived WAL segments"*.

**The agent was right.** `disk_full` has two narratives for the same failure, the
corpus contains both, and the generator had drawn one and graded against that
sentence alone.

The fix is a principle: **grade on what is determinate.** The archetype is
determinate; which narrative gets used is not. Keywords are now pooled across
every variant, and the same answer scores 6 hits instead of 0.

And immediately, the counter-risk: a vocabulary broad enough to accept anything
is a worse bug than the one being fixed. So a test asserts that diagnosing TLS
expiry against a `disk_full` scenario still fails.

---

## The metrics

**Task completion** — did it name the right causing service, describe the right
failure mode, and propose an acceptable remediation, inside the iteration cap?
Structural, not prose comparison.

**Tool success rate** — successes ÷ **attempts**. A tool that fails twice then
succeeds is 1 of 3, not 1 of 1. Collapsing retries would make the number
flattering and wrong.

**Recovery under injection** — did a tool that failed on one attempt succeed on a
later one?

**Abstention accuracy** — on the 15 no-retrieval scenarios, did it say so?

**Latency and cost** — median and p95, tokens in and out.

**Groundedness** — the hard one, below.

---

## LLM-as-judge, and why the number needs a second number

Groundedness asks: is every factual claim traceable to evidence that was
actually in context? You cannot check that with string matching, so a second
model reads the evidence and the analysis and grades each claim.

This is standard practice and it is **genuinely unreliable**:

- **Shared blind spots.** A judge and an agent running the same model share
  failure modes. A claim both find plausible passes regardless of truth.
- **Fluency bias.** Judges reward confident, well-structured text — precisely
  the failure mode being tested for.
- **Instability.** The same claim scores differently across runs.

### Calibration, and why raw agreement lies

The mitigation is to grade a sample by hand and report how often the judge
agrees. But **raw agreement is misleading on its own**:

| Judge | Raw agreement | Cohen's kappa |
|---|---|---|
| Always answers "supported" | **90%** | **0.000** |
| Realistic | 90% | 0.444 |
| Perfect | 100% | 1.000 |

On a corpus that is 90% supported, a judge that never thinks scores 90%. Kappa
corrects for agreement expected by chance:

| kappa | Meaning |
|---|---|
| < 0.20 | Negligible — do not report groundedness as a headline metric |
| 0.21–0.40 | Fair |
| 0.41–0.60 | Moderate — usable, state the kappa alongside |
| 0.61–0.80 | Substantial |
| > 0.80 | Strong |

`python -m evals.calibration` produces a labelling sheet with the judge's
verdicts present but flagged not to read first, then scores the agreement. The
generated report says explicitly when this has not been run — a groundedness
number without a kappa is not interpretable, and the report should not pretend
otherwise.

---

## Failure analysis is an output, not an appendix

"What do the 10% of failing scenarios have in common?" is the question the suite
exists to answer, so the report always includes:

```
## What the failing scenarios have in common

| Failure mode                                              | Count |
|-----------------------------------------------------------|-------|
| blamed the victim service instead of the downstream cause |     4 |
| did not name the causing service                          |     3 |
| named the service but not the failure mode                |     2 |
```

Blaming the victim is called out **specifically** rather than folded into
"wrong answer", because it is the characteristic wrong answer for cascading
scenarios and the pattern the suite exists to detect. A generic bucket would
hide it.

---

## The constraint nobody mentions in tutorials

qwen2.5:14b generates at **~24 tokens/sec** on an M4 Pro. Prompts run ~3,000
tokens, of which **retrieved documents are 45%**. An investigation is 10–12 LLM
calls, so ~2 minutes; approval scenarios that interrupt and resume take longer.

**The full 100-scenario suite is an overnight job, not a coffee break.**

That is a design constraint, not a footnote. It is why the harness has:

- `--sample N` — stratified across scenario kinds. `--limit` was not enough: it
  takes the first N, which would run only one category.
- `--max-iterations N` — trades depth for turnaround, **and is written into the
  results file**, because a capped run is not comparable with an uncapped one
  and nobody should have to remember mentioning it.

---

## What to be able to explain

- Why 15 of 100 scenarios test that the agent says nothing
- Where ground truth comes from, and why scenarios are generated
- Why tool success counts attempts rather than calls
- How groundedness is measured, and three specific limits of LLM-as-judge
- Why a 90% agreement rate can mean the judge is worthless
- What the failing scenarios have in common — and that you looked

---

**Next:** 06 — Human-in-the-loop: interrupts, approvals, and audit *(Phase 4)*
