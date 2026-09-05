#!/usr/bin/env python3
"""The evaluation harness.

    python -m evals.run                      # all 100 scenarios
    python -m evals.run --kind cascading     # one category
    python -m evals.run --limit 10           # a dev subset
    python -m evals.run --no-judge           # skip groundedness (much faster)

Writes a timestamped JSON file and a markdown summary to evals/results/.

Grading philosophy
------------------
Every metric is computed from structured facts about the run - which tools were
called, which service was named, whether the agent abstained - rather than from
prose comparison. Free-text similarity between the agent's root cause and the
expected one would measure phrasing, and would quietly reward a model that
writes fluently about the wrong thing.

The one place prose is unavoidable is groundedness, which is why the judge's
limitations are documented in `judge.py` and reported alongside the number.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "api"))

from incidentiq.config import Settings, get_settings  # noqa: E402

from evals import fixtures  # noqa: E402
from evals.scenario import Scenario, load_scenarios  # noqa: E402

SCENARIO_DIR = REPO_ROOT / "evals" / "scenarios"
# Set from --max-iterations so it lands in the results file: a run with a
# reduced budget is not comparable with a full one, and the number must say so.
ITERATION_CAP: int | None = None
# Identifies this process's fixture rows, so a later run can tell its own
# leftovers from another's.
RUN_ID: str = "adhoc"
RESULTS_DIR = REPO_ROOT / "evals" / "results"


@dataclass
class ScenarioResult:
    scenario_id: str
    kind: str
    query: str

    completed: bool = False           # reached a conclusion inside the cap
    task_success: bool = False        # ... and the conclusion is acceptable
    identified_cause: bool = False    # named the right service
    matched_keywords: int = 0
    total_keywords: int = 0
    correct_remediation: bool = False
    abstained: bool = False
    abstention_correct: bool | None = None

    approval_requested: bool = False
    approval_resolved: bool = False

    tool_attempts: int = 0
    tool_successes: int = 0
    tools_called: list[str] = field(default_factory=list)
    expected_tools_hit: int = 0
    injected_failures: int = 0
    recovered_from_failure: bool = False

    grounded_fraction: float | None = None
    fully_grounded: bool | None = None
    n_claims: int = 0

    iterations: int = 0
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    status: str = ""
    error: str | None = None
    failure_note: str = ""            # why it failed, for the failure analysis


def _grade(scenario: Scenario, state: dict) -> ScenarioResult:
    proposal = state.get("proposal") or {}
    calls = state.get("tool_calls") or []
    abstained = bool(proposal.get("abstained"))

    r = ScenarioResult(
        scenario_id=scenario.id, kind=scenario.kind, query=scenario.query,
        abstained=abstained,
        iterations=state.get("iteration", 0),
        status=state.get("status", ""),
        input_tokens=state.get("total_input_tokens", 0),
        output_tokens=state.get("total_output_tokens", 0),
        tool_attempts=len(calls),
        tool_successes=sum(1 for c in calls if c["status"] in ("success", "partial")),
        tools_called=sorted({c["tool_name"] for c in calls}),
        injected_failures=sum(1 for c in calls if c.get("injected_failure")),
    )
    r.expected_tools_hit = sum(1 for t in scenario.expected_tools if t in r.tools_called)

    # Recovery: a tool that failed on one attempt and succeeded on a later one.
    by_tool: dict[str, list[dict]] = {}
    for c in calls:
        by_tool.setdefault(c["tool_name"], []).append(c)
    r.recovered_from_failure = any(
        any(a["status"] not in ("success", "partial") for a in attempts)
        and any(a["status"] in ("success", "partial") for a in attempts)
        for attempts in by_tool.values()
    )

    # An investigation whose approved action failed to execute has not completed
    # its task, however good the diagnosis was.
    r.completed = (state.get("status") in ("complete", "awaiting_approval")
                   and not state.get("failure_reason"))
    if state.get("failure_reason"):
        r.failure_note = f"the approved action failed to execute: {state['failure_reason']}"

    # ── Abstention scenarios ────────────────────────────────
    if scenario.should_abstain:
        r.abstention_correct = abstained
        r.task_success = abstained
        if not abstained:
            r.failure_note = ("fabricated an answer for a query with no relevant "
                              "evidence - the worst failure mode in the suite")
        return r

    if abstained:
        r.abstention_correct = False
        r.failure_note = "abstained on a scenario that had a discoverable cause"
        return r

    # ── Did it name the right service? ──────────────────────
    # Grade against the agent's WHOLE conclusion, not one field of it. Grading
    # on root_cause + remediation alone marked a correct answer wrong: the agent
    # concluded "disk exhaustion on catalog-db with confidence 0.9", named the
    # service in its summary and in the remediation arguments, and simply did
    # not repeat it inside the root_cause string. Which field a model puts the
    # service name in is not something the agent should be scored on.
    text = " ".join([
        str(proposal.get("root_cause") or ""),
        str(proposal.get("remediation") or ""),
        json.dumps(proposal.get("remediation_arguments") or {}, default=str),
        str(state.get("final_summary") or ""),
    ]).lower()
    cause_service = (scenario.true_cause_service or scenario.expected_service or "").lower()
    r.identified_cause = bool(cause_service) and cause_service in text

    # For cascading scenarios, blaming the victim is a specific, common, and
    # wrong answer. Call it out rather than scoring it as a near miss.
    if scenario.kind == "cascading" and not r.identified_cause:
        if any(v.lower() in text for v in scenario.victim_services):
            r.failure_note = "blamed the victim service instead of the downstream cause"

    # ── Did it describe the right mechanism? ────────────────
    # Keywords are pooled across every narrative variant of the archetype (see
    # evals/generate_scenarios._archetype_keywords), so the pool is larger and a
    # proportion threshold no longer means anything stable. Ask for a small
    # absolute number of hits instead: enough that the agent clearly named this
    # failure mode, few enough that it is not required to echo one variant's
    # exact phrasing.
    MIN_KEYWORD_HITS = 2
    r.total_keywords = len(scenario.root_cause_keywords)
    r.matched_keywords = sum(1 for k in scenario.root_cause_keywords if k in text)
    mechanism_ok = (
        r.total_keywords == 0 or r.matched_keywords >= min(MIN_KEYWORD_HITS, r.total_keywords)
    )

    # ── Remediation ─────────────────────────────────────────
    proposed_tool = proposal.get("remediation_tool")
    if scenario.acceptable_remediation_tools:
        r.correct_remediation = proposed_tool in scenario.acceptable_remediation_tools
    else:
        r.correct_remediation = True   # no specific action expected

    r.task_success = bool(r.completed and r.identified_cause and mechanism_ok
                          and r.correct_remediation)

    if not r.task_success and not r.failure_note:
        if not r.completed:
            r.failure_note = "did not reach a conclusion within the iteration cap"
        elif not r.identified_cause:
            r.failure_note = "did not name the causing service"
        elif not mechanism_ok:
            r.failure_note = (f"named the service but not the failure mode "
                              f"({r.matched_keywords} of {r.total_keywords} archetype "
                              f"terms matched, needed {MIN_KEYWORD_HITS})")
        elif not r.correct_remediation:
            r.failure_note = (f"proposed {proposed_tool!r}, expected one of "
                              f"{scenario.acceptable_remediation_tools}")
    return r


def run_scenario(scenario: Scenario, settings: Settings, judge_llm=None,
                 iteration_cap: int | None = None) -> ScenarioResult:
    from incidentiq.agent.graph import InvestigationRunner

    cfg = settings.model_copy(update={
        "max_iterations": min(scenario.max_iterations, iteration_cap)
                          if iteration_cap else scenario.max_iterations,
        "failure_injection_enabled": scenario.inject_failures,
        "failure_injection_rate": scenario.failure_rate,
    })
    incident_id = f"EVAL-{scenario.id}"
    started = time.perf_counter()
    seeded = None

    try:
        with InvestigationRunner(settings=cfg) as runner:
            # Plant the evidence this scenario's failure would leave, so the
            # diagnostic tools have something to find. Without it most
            # scenarios were unanswerable from evidence and the eval measured
            # whether the agent could guess the archetype from retrieval alone.
            seeded = fixtures.apply(runner.conn, scenario, run_id=RUN_ID)
            state = runner.start(incident_id, scenario.query, reset=True)

            # Approval-gated scenarios: record the human decision and resume,
            # so the interrupt/resume path is exercised rather than assumed.
            result_holder: dict[str, Any] = {}
            if state.get("status") == "awaiting_approval":
                pending = state.get("pending_approval") or {}
                decision = {"approve": "approved", "reject": "rejected",
                            "modify": "approved"}[scenario.approval_decision or "approve"]
                runner.conn.execute(
                    "UPDATE approvals SET decision = %s, decided_by = 'eval-harness', "
                    "decided_at = now(), rejection_reason = %s WHERE id = %s",
                    (decision,
                     "the eval harness rejected this action" if decision == "rejected" else None,
                     pending.get("approval_id")),
                )
                state = runner.resume(incident_id)
                result_holder["approval_resolved"] = True

            r = _grade(scenario, state)
            r.latency_s = round(time.perf_counter() - started, 2)
            r.approval_requested = bool(state.get("pending_approval"))
            r.approval_resolved = bool(result_holder.get("approval_resolved"))

            if judge_llm is not None:
                from evals.judge import score_groundedness
                g = score_groundedness(judge_llm, state)
                r.grounded_fraction = g.grounded_fraction
                r.fully_grounded = g.fully_grounded
                r.n_claims = g.n_claims
            return r

    except Exception as exc:  # noqa: BLE001
        return ScenarioResult(
            scenario_id=scenario.id, kind=scenario.kind, query=scenario.query,
            latency_s=round(time.perf_counter() - started, 2),
            error=f"{type(exc).__name__}: {exc}",
            failure_note=f"harness error: {traceback.format_exc(limit=2)[-200:]}",
        )
    finally:
        # Always clean up, including on failure. Leaked fixture rows would make
        # every later scenario for that service look like it had evidence it
        # was never given.
        if seeded is not None:
            try:
                import psycopg as _pg
                with _pg.connect(cfg.database_url, autocommit=True) as c:
                    fixtures.clear(c, seeded)
            except Exception:  # noqa: BLE001
                pass


def aggregate(results: list[ScenarioResult], settings: Settings) -> dict[str, Any]:
    n = len(results)
    if not n:
        return {}

    attempts = sum(r.tool_attempts for r in results)
    successes = sum(r.tool_successes for r in results)
    latencies = sorted(r.latency_s for r in results)
    # Two separate filters: a scenario can have a fully_grounded verdict while
    # grounded_fraction is absent (or the reverse) if the judge partially
    # failed. Averaging over the wrong set raises on None, and - worse - a
    # single broken judge call would otherwise skew the mean silently.
    judged = [r for r in results if r.fully_grounded is not None]
    with_fraction = [r for r in results if r.grounded_fraction is not None]
    approval = [r for r in results if r.kind == "approval_required"]
    failure = [r for r in results if r.kind == "tool_failure"]
    abstention = [r for r in results if r.kind == "no_retrieval"]

    by_kind: dict[str, dict] = {}
    for kind in sorted({r.kind for r in results}):
        subset = [r for r in results if r.kind == kind]
        by_kind[kind] = {
            "n": len(subset),
            "task_completion": round(sum(r.task_success for r in subset) / len(subset), 4),
            "median_latency_s": round(statistics.median(r.latency_s for r in subset), 1),
        }

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "n_scenarios": n,
        "model": settings.ollama_model if settings.llm_provider == "ollama"
                 else settings.anthropic_model,
        "provider": settings.llm_provider,
        "iteration_cap_override": ITERATION_CAP,
        "task_completion": round(sum(r.task_success for r in results) / n, 4),
        "grounded_response_rate": (
            round(sum(bool(r.fully_grounded) for r in judged) / len(judged), 4)
            if judged else None
        ),
        "mean_grounded_fraction": (
            round(statistics.mean(r.grounded_fraction for r in with_fraction), 4)
            if with_fraction else None
        ),
        "n_judged": len(judged),
        "tool_success_rate": round(successes / attempts, 4) if attempts else None,
        "tool_attempts": attempts,
        "median_latency_s": round(statistics.median(latencies), 1),
        # ceil, not int: with n=4, int(0.95*4)-1 = 2 picks the third of four
        # values as "p95", which understates the tail exactly when the sample is
        # small enough for the tail to matter most.
        "p95_latency_s": round(latencies[min(n - 1, math.ceil(0.95 * n) - 1)], 1),
        "mean_input_tokens": round(statistics.mean(r.input_tokens for r in results)),
        "mean_output_tokens": round(statistics.mean(r.output_tokens for r in results)),
        "abstention_accuracy": (
            round(sum(bool(r.abstention_correct) for r in abstention) / len(abstention), 4)
            if abstention else None
        ),
        "approval_flow_exercised": (
            round(sum(r.approval_resolved for r in approval) / len(approval), 4)
            if approval else None
        ),
        "recovery_rate_under_injection": (
            round(sum(r.recovered_from_failure for r in failure) / len(failure), 4)
            if failure else None
        ),
        "harness_errors": sum(1 for r in results if r.error),
        "by_kind": by_kind,
    }


def failure_analysis(results: list[ScenarioResult]) -> dict[str, int]:
    """What the failing scenarios have in common - a required output, not an
    appendix. 'Which 10% fail and why' is the question this suite exists to
    answer."""
    counts: dict[str, int] = {}
    for r in results:
        if r.task_success or not r.failure_note:
            continue
        key = r.failure_note.split("(")[0].strip().rstrip(".")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def write_report(summary: dict, results: list[ScenarioResult], failures: dict,
                 stamp: str) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"run_{stamp}.json"
    md_path = RESULTS_DIR / f"run_{stamp}.md"

    json_path.write_text(json.dumps(
        {"summary": summary, "failure_analysis": failures,
         "results": [asdict(r) for r in results]},
        indent=2, default=str,
    ))

    lines = [
        f"# Eval run {stamp}", "",
        f"**Model:** `{summary['model']}` via `{summary['provider']}`  ",
        f"**Scenarios:** {summary['n_scenarios']}", "",
        "| Metric | Target | Result |", "|---|---|---|",
        f"| Task completion | ≥ 90% | {summary['task_completion']:.1%} |",
    ]
    if summary.get("grounded_response_rate") is not None:
        lines.append(f"| Grounded response rate | ≥ 90% | "
                     f"{summary['grounded_response_rate']:.1%} |")
    if summary.get("tool_success_rate") is not None:
        lines.append(f"| Tool execution success | ≥ 95% | "
                     f"{summary['tool_success_rate']:.1%} "
                     f"({summary['tool_attempts']} attempts) |")
    for label, key, fmt in [
        ("Abstention accuracy", "abstention_accuracy", "{:.1%}"),
        ("Approval flow exercised", "approval_flow_exercised", "{:.1%}"),
        ("Recovery under injection", "recovery_rate_under_injection", "{:.1%}"),
        ("Median latency", "median_latency_s", "{:.1f}s"),
        ("p95 latency", "p95_latency_s", "{:.1f}s"),
    ]:
        if summary.get(key) is not None:
            lines.append(f"| {label} | — | {fmt.format(summary[key])} |")
    lines += [
        f"| Mean tokens / investigation | — | "
        f"{summary['mean_input_tokens']:,} in / {summary['mean_output_tokens']:,} out |",
        "", "## By scenario kind", "",
        "| Kind | n | Task completion | Median latency |", "|---|---|---|---|",
    ]
    for kind, d in summary["by_kind"].items():
        lines.append(f"| {kind} | {d['n']} | {d['task_completion']:.1%} | "
                     f"{d['median_latency_s']:.1f}s |")

    if failures:
        lines += ["", "## What the failing scenarios have in common", "",
                  "| Failure mode | Count |", "|---|---|"]
        for note, count in failures.items():
            lines.append(f"| {note} | {count} |")

    if summary.get("iteration_cap_override"):
        lines += ["", "> **Reduced iteration budget.** Every scenario was capped at "
                  f"{summary['iteration_cap_override']} iterations rather than its own "
                  "budget of 4-8. Task completion is therefore a lower bound, and this "
                  "run is not comparable with an uncapped one."]

    if summary.get("grounded_response_rate") is not None:
        lines += ["", "> **The groundedness figure is uncalibrated.** It is one model's "
                  "opinion of another model's output, and the two share failure modes. "
                  "Raw agreement with a human is not enough either - a judge that "
                  "answers \"supported\" unconditionally scores 90% agreement on a "
                  "corpus that is 90% supported while carrying no information. Run "
                  "`python -m evals.calibration` to produce a labelling sheet and a "
                  "Cohen's kappa, and report that alongside this number."]

    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", help="run only one scenario kind")
    parser.add_argument("--limit", type=int, help="run only the first N")
    parser.add_argument("--sample", type=int, metavar="N",
                        help="stratified sample of N per scenario kind - use this rather "
                             "than --limit for a dev subset, since --limit takes the first "
                             "N and would run only one category")
    parser.add_argument("--max-iterations", type=int, metavar="N",
                        help="cap every scenario's iteration budget at N. Local "
                             "inference at ~24 tok/s makes the full suite an overnight "
                             "job; this trades depth for turnaround during development. "
                             "Always reported in the results file, because it changes "
                             "the numbers.")
    parser.add_argument("--no-judge", action="store_true",
                        help="skip groundedness scoring (roughly halves runtime)")
    parser.add_argument("--provider", help="override LLM_PROVIDER")
    parser.add_argument("--model", help="override the model name")
    args = parser.parse_args()

    settings = get_settings()
    updates: dict[str, Any] = {}
    if args.provider:
        updates["llm_provider"] = args.provider
    if args.model:
        key = "anthropic_model" if (args.provider or settings.llm_provider) == "anthropic" \
            else "ollama_model"
        updates[key] = args.model
    if updates:
        settings = settings.model_copy(update=updates)

    global ITERATION_CAP, RUN_ID
    ITERATION_CAP = args.max_iterations
    RUN_ID = datetime.now(UTC).strftime("run-%Y%m%dT%H%M%SZ")

    # Remove anything a previous run left behind before measuring anything.
    # A killed run orphans its planted evidence, which then becomes background
    # noise for every scenario that follows - invisibly, because nothing errors.
    import psycopg as _pg
    with _pg.connect(settings.database_url, autocommit=True) as _c:
        purged = fixtures.purge_orphans(_c, except_run_id=RUN_ID)
    if purged:
        print(f"purged orphaned fixture rows from a previous run: {purged}")

    scenarios = load_scenarios(SCENARIO_DIR)
    if args.kind:
        scenarios = [s for s in scenarios if s.kind == args.kind]
    if args.sample:
        by_kind: dict[str, list[Scenario]] = {}
        for sc in scenarios:
            by_kind.setdefault(sc.kind, []).append(sc)
        scenarios = [sc for group in by_kind.values() for sc in group[: args.sample]]
    if args.limit:
        scenarios = scenarios[: args.limit]
    if not scenarios:
        print("no scenarios matched", file=sys.stderr)
        return 1

    judge_llm = None
    if not args.no_judge:
        from incidentiq.llm import get_llm
        judge_llm = get_llm(settings)

    model = (settings.ollama_model if settings.llm_provider == "ollama"
             else settings.anthropic_model)
    print(f"running {len(scenarios)} scenarios against {model} "
          f"({settings.llm_provider}), judge={'on' if judge_llm else 'off'}")

    results: list[ScenarioResult] = []
    t0 = time.perf_counter()
    for i, scenario in enumerate(scenarios, 1):
        r = run_scenario(scenario, settings, judge_llm, args.max_iterations)
        results.append(r)
        mark = "ok  " if r.task_success else ("ERR " if r.error else "fail")
        print(f"  [{i:>3}/{len(scenarios)}] {mark} {r.scenario_id:<12} "
              f"{r.latency_s:>6.1f}s  {r.failure_note[:58]}")

    summary = aggregate(results, settings)
    failures = failure_analysis(results)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path, md_path = write_report(summary, results, failures, stamp)

    print(f"\n{'='*64}")
    print(f"task completion      {summary['task_completion']:.1%}")
    if summary.get("grounded_response_rate") is not None:
        print(f"grounded responses   {summary['grounded_response_rate']:.1%}")
    if summary.get("tool_success_rate") is not None:
        print(f"tool success         {summary['tool_success_rate']:.1%} "
              f"({summary['tool_attempts']} attempts)")
    print(f"median / p95 latency {summary['median_latency_s']:.1f}s / "
          f"{summary['p95_latency_s']:.1f}s")
    print(f"total wall clock     {(time.perf_counter()-t0)/60:.1f} min")
    if failures:
        print("\nfailure modes:")
        for note, count in failures.items():
            print(f"  {count:>3}x  {note}")
    print(f"\nwrote {md_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
