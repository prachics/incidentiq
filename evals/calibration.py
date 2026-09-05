#!/usr/bin/env python3
"""Calibrate the groundedness judge against human labels.

A groundedness score is not interpretable on its own. It is one model's opinion
of another model's output, and the two share failure modes. What makes it usable
is knowing how often that opinion agrees with a person's.

    python -m evals.calibration --sample 20 --from evals/results/run_X.json
        writes a labelling sheet with the judge's verdict hidden

    python -m evals.calibration --score evals/results/calibration_X.json
        compares your labels with the judge's and reports agreement

Agreement is reported as raw agreement **and** Cohen's kappa. Raw agreement
alone is misleading when one verdict dominates: if 90% of claims are supported,
a judge that answers "supported" unconditionally scores 90% agreement while
carrying no information at all. Kappa corrects for agreement expected by chance.

Interpreting kappa, by the usual convention:

    < 0.20   negligible - the score means nothing
    0.21-0.40   fair
    0.41-0.60   moderate
    0.61-0.80   substantial - the score is usable with caveats
    > 0.80   strong

Anything below 0.4 means the groundedness number should not be reported as a
headline metric, and this file exists so that judgement is made on evidence
rather than on hope.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "evals" / "results"


def cohens_kappa(a: list[str], b: list[str]) -> float:
    """Agreement corrected for chance."""
    if not a or len(a) != len(b):
        return 0.0
    labels = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    expected = sum(
        (a.count(label) / n) * (b.count(label) / n) for label in labels
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def build_sheet(results_path: Path, n: int, seed: int = 7) -> Path:
    """Extract a random sample of judged claims, with the verdict hidden."""
    data = json.loads(results_path.read_text())
    claims: list[dict] = []
    for r in data.get("results", []):
        for c in r.get("claims", []) or []:
            claims.append({
                "scenario_id": r["scenario_id"],
                "claim": c.get("claim"),
                "judge_verdict": c.get("verdict"),
                "human_verdict": None,   # <- fill in: supported|unsupported|contradicted
            })
    if not claims:
        raise SystemExit(
            f"{results_path.name} contains no per-claim verdicts. Run the harness "
            "with the judge enabled and per-claim output retained."
        )

    random.Random(seed).shuffle(claims)
    sample = claims[:n]
    out = RESULTS_DIR / f"calibration_{results_path.stem}.json"
    out.write_text(json.dumps({
        "source": results_path.name,
        "instructions": (
            "Set human_verdict for each claim to supported, unsupported, or "
            "contradicted, judging ONLY against the evidence that was available "
            "to the agent. Do not read judge_verdict first - it is included so "
            "the file is self-contained for scoring, and reading it first "
            "defeats the purpose."
        ),
        "n_claims": len(sample),
        "claims": sample,
    }, indent=2))
    return out


def score_sheet(path: Path) -> dict:
    data = json.loads(path.read_text())
    labelled = [c for c in data["claims"] if c.get("human_verdict")]
    if not labelled:
        raise SystemExit(f"{path.name} has no human_verdict values filled in.")

    human = [c["human_verdict"] for c in labelled]
    judge = [c["judge_verdict"] for c in labelled]
    agree = sum(h == j for h, j in zip(human, judge, strict=True))
    kappa = cohens_kappa(human, judge)

    band = (
        "negligible - do not report groundedness as a headline metric" if kappa < 0.20
        else "fair - report with a prominent caveat" if kappa < 0.41
        else "moderate - usable, state the kappa alongside" if kappa < 0.61
        else "substantial - usable" if kappa < 0.81
        else "strong"
    )
    disagreements = [
        {"claim": c["claim"][:120], "judge": c["judge_verdict"], "human": c["human_verdict"]}
        for c in labelled if c["judge_verdict"] != c["human_verdict"]
    ]
    return {
        "n_labelled": len(labelled),
        "raw_agreement": round(agree / len(labelled), 4),
        "cohens_kappa": round(kappa, 4),
        "interpretation": band,
        "disagreements": disagreements[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="source", help="a run_*.json to sample from")
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--score", help="a filled-in calibration_*.json to score")
    args = parser.parse_args()

    if args.score:
        report = score_sheet(Path(args.score))
        print(f"labelled claims  {report['n_labelled']}")
        print(f"raw agreement    {report['raw_agreement']:.1%}")
        print(f"Cohen's kappa    {report['cohens_kappa']:.3f}  ({report['interpretation']})")
        if report["disagreements"]:
            print("\ndisagreements:")
            for d in report["disagreements"]:
                print(f"  judge={d['judge']:<13} human={d['human']:<13} {d['claim']}")
        return 0

    if not args.source:
        print("give either --from <run.json> or --score <calibration.json>", file=sys.stderr)
        return 1

    out = build_sheet(Path(args.source), args.sample)
    print(f"wrote {out.relative_to(REPO_ROOT)} - fill in human_verdict, then:")
    print(f"  python -m evals.calibration --score {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
