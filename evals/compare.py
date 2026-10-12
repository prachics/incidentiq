#!/usr/bin/env python3
"""Compare two eval runs, scenario by scenario.

    python -m evals.compare evals/results/run_A.json evals/results/run_B.json

Reads two finished results files (before and after a change, or two models on the
same scenarios). Every difference that would make the comparison misleading
(scenarios, iteration cap, seeds, execution settings, grader) is printed as a
warning above the tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Metrics where a higher number is better. Latency and token counts are not.
HIGHER_IS_BETTER = {
    "task_completion", "grounded_response_rate", "tool_success_rate",
    "abstention_accuracy", "approval_flow_exercised", "recovery_rate_under_injection",
}

HEADLINE = [
    ("task_completion", "Task completion", "pct"),
    ("abstention_accuracy", "Abstention accuracy", "pct"),
    ("tool_success_rate", "Tool success", "pct"),
    ("grounded_response_rate", "Grounded responses", "pct"),
    ("recovery_rate_under_injection", "Recovery under injection", "pct"),
    ("median_latency_s", "Median latency", "s"),
    ("mean_input_tokens", "Input tokens", "int"),
    ("mean_output_tokens", "Output tokens", "int"),
]


def _fmt(value, kind: str) -> str:
    if value is None:
        return "—"
    if kind == "pct":
        return f"{value:.1%}"
    if kind == "s":
        return f"{value:.1f}s"
    return f"{value:,.0f}"


def _comparability(a: dict, b: dict) -> list[str]:
    """Reasons these two runs should not be compared.

    Reported rather than enforced, because a caveated comparison is sometimes
    what you want - but never a silent one.
    """
    warnings = []
    if a["n_scenarios"] != b["n_scenarios"]:
        warnings.append(
            f"different scenario counts ({a['n_scenarios']} vs {b['n_scenarios']}) - "
            "the runs did not see the same work"
        )
    if a.get("iteration_cap_override") != b.get("iteration_cap_override"):
        warnings.append(
            f"different iteration caps ({a.get('iteration_cap_override') or 'none'} vs "
            f"{b.get('iteration_cap_override') or 'none'}) - a capped run understates "
            "task completion, so this comparison measures the cap as much as the model"
        )
    if a.get("seed") != b.get("seed"):
        warnings.append(f"different sampling seeds ({a.get('seed')} vs {b.get('seed')}) - "
                        "some differences are sampling, not the change")
    a_judged = a.get("grounded_response_rate") is not None
    b_judged = b.get("grounded_response_rate") is not None
    if a_judged != b_judged:
        warnings.append("groundedness scored in only one run")
    if a.get("by_kind", {}).keys() != b.get("by_kind", {}).keys():
        warnings.append("different scenario kinds present")
    if a.get("scenarios_sha256") != b.get("scenarios_sha256"):
        warnings.append("different scenario files (or one run predates the fingerprint)")
    exec_a, exec_b = a.get("execution"), b.get("execution")
    if exec_a is None or exec_b is None:
        if exec_a is not exec_b:
            warnings.append("only one run records its execution settings")
    elif exec_a != exec_b:
        differ = sorted(k for k in exec_a.keys() | exec_b.keys() if exec_a.get(k) != exec_b.get(k))
        warnings.append(f"different execution settings ({', '.join(differ)}) - failure "
                        "injection, retries or the control plane differ")
    grader_a = a.get("regraded_with") or a.get("revision")
    grader_b = b.get("regraded_with") or b.get("revision")
    if grader_a != grader_b:
        warnings.append(f"graded by different revisions ({grader_a} vs {grader_b})")
    ids_a, ids_b = a.get("scenario_ids"), b.get("scenario_ids")
    if ids_a is not None and ids_b is not None and set(ids_a) != set(ids_b):
        only_a, only_b = sorted(set(ids_a) - set(ids_b)), sorted(set(ids_b) - set(ids_a))
        warnings.append(f"different scenarios (only in A: {', '.join(only_a) or 'none'}; "
                        f"only in B: {', '.join(only_b) or 'none'})")
    return warnings


def render(a: dict, b: dict, label_a: str, label_b: str) -> str:
    lines = [
        f"# {label_a} vs {label_b}", "",
        f"**A:** `{a['model']}` via `{a['provider']}`, {a['n_scenarios']} scenarios  ",
        f"**B:** `{b['model']}` via `{b['provider']}`, {b['n_scenarios']} scenarios", "",
    ]

    warnings = _comparability(a, b)
    if warnings:
        lines += ["> **These runs are not cleanly comparable:**", ">"]
        lines += [f"> - {w}" for w in warnings]
        lines.append("")

    lines += [f"| Metric | {label_a} | {label_b} | Δ |", "|---|---|---|---|"]
    for key, label, kind in HEADLINE:
        va, vb = a.get(key), b.get(key)
        if va is None and vb is None:
            continue
        delta = "—"
        if va is not None and vb is not None:
            diff = vb - va
            better = (diff > 0) == (key in HIGHER_IS_BETTER)
            arrow = "" if diff == 0 else ("better" if better else "worse")
            delta = f"{_fmt(abs(diff), kind)} {arrow}".strip()
        lines.append(f"| {label} | {_fmt(va, kind)} | {_fmt(vb, kind)} | {delta} |")

    kinds = sorted(set(a.get("by_kind", {})) | set(b.get("by_kind", {})))
    if kinds:
        lines += ["", "## By scenario kind", "",
                  f"| Kind | n | {label_a} | {label_b} |", "|---|---|---|---|"]
        for kind in kinds:
            ka = a.get("by_kind", {}).get(kind, {})
            kb = b.get("by_kind", {}).get(kind, {})
            n = ka.get("n") or kb.get("n") or "—"
            lines.append(
                f"| {kind} | {n} | {_fmt(ka.get('task_completion'), 'pct')} | "
                f"{_fmt(kb.get('task_completion'), 'pct')} |"
            )
    return "\n".join(lines) + "\n"


def render_failures(fa: dict, fb: dict, label_a: str, label_b: str) -> str:
    """Which failure modes moved. More informative than the headline number:
    a change that trades one failure mode for another is not an improvement
    even when the totals look better."""
    keys = sorted(set(fa) | set(fb))
    if not keys:
        return ""
    lines = ["", "## Failure modes", "",
             f"| Mode | {label_a} | {label_b} |", "|---|---|---|"]
    for k in keys:
        lines.append(f"| {k[:80]} | {fa.get(k, 0)} | {fb.get(k, 0)} |")
    return "\n".join(lines) + "\n"


def render_scenarios(ra: list[dict], rb: list[dict], label_a: str, label_b: str) -> str:
    """Scenarios in both runs whose outcome or failure changed. A replay reruns
    only a run's failures, so this is the table that says what a fix did."""
    before = {r["scenario_id"]: r for r in ra}
    rows = []
    for r in rb:
        old = before.get(r["scenario_id"])
        if old is None:
            continue
        note_a, note_b = old.get("failure_note") or "", r.get("failure_note") or ""
        if old["task_success"] != r["task_success"] or note_a != note_b:
            rows.append(f"| {r['scenario_id']} | {_outcome(old)} | {_outcome(r)} |")
    if not rows:
        return ""
    return "\n".join(["", "## Scenarios that changed", "",
                      f"| Scenario | {label_a} | {label_b} |", "|---|---|---|", *rows]) + "\n"


def _outcome(r: dict) -> str:
    if r["task_success"]:
        return "pass"
    return f"fail: {(r.get('failure_note') or '').split('(')[0].strip()[:70]}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--out", help="write the markdown here as well as printing it")
    args = parser.parse_args()

    try:
        a = json.loads(Path(args.run_a).read_text())
        b = json.loads(Path(args.run_b).read_text())
    except FileNotFoundError as exc:
        print(f"no such results file: {exc.filename}", file=sys.stderr)
        return 1

    # Which scenarios each run saw, beside its summary, so a different set is caught
    # even when the counts and kinds agree.
    summaries = [{**run["summary"], "scenario_ids": [r["scenario_id"]
                                                      for r in run.get("results", [])]}
                 for run in (a, b)]
    md = render(*summaries, args.label_a, args.label_b)
    md += render_failures(a.get("failure_analysis", {}), b.get("failure_analysis", {}),
                          args.label_a, args.label_b)
    md += render_scenarios(a.get("results", []), b.get("results", []),
                           args.label_a, args.label_b)
    print(md)
    if args.out:
        Path(args.out).write_text(md)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
