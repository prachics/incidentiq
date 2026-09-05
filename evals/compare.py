#!/usr/bin/env python3
"""Compare two eval runs.

    python -m evals.compare evals/results/run_A.json evals/results/run_B.json

The intended use is a local model against a frontier model on the same
scenarios, which is the reason the provider interface exists. It also works for
before/after comparisons of a single change, which is the cheaper and more
frequent case.

Why a separate tool rather than a `--compare` flag on the harness: running two
providers in one process means one crash loses both halves, and the results are
only comparable if the scenario set, iteration cap and corpus all match. Reading
two finished result files makes those preconditions checkable rather than
assumed - and this refuses to compare runs that differ in ways that would make
the comparison meaningless.
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
    a_judged = a.get("grounded_response_rate") is not None
    b_judged = b.get("grounded_response_rate") is not None
    if a_judged != b_judged:
        warnings.append("groundedness scored in only one run")
    if a.get("by_kind", {}).keys() != b.get("by_kind", {}).keys():
        warnings.append("different scenario kinds present")
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

    md = render(a["summary"], b["summary"], args.label_a, args.label_b)
    md += render_failures(a.get("failure_analysis", {}), b.get("failure_analysis", {}),
                          args.label_a, args.label_b)
    print(md)
    if args.out:
        Path(args.out).write_text(md)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
