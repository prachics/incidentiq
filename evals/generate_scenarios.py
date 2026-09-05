#!/usr/bin/env python3
"""Generate the 100-scenario evaluation suite.

Scenarios are derived from the same archetype-times-service structure the corpus
is built from, which is what gives each one a known correct answer. They are
generated rather than hand-written for the same reason the retrieval labels are
structural (docs/DECISIONS.md #5): a suite written after seeing what the agent
does drifts toward what the agent already does.

Composition, per the specification:

  40  single_service      one service, one archetype, evidence available
  25  cascading           the reported service is a victim; the cause is
                          downstream. Graded on naming the cause, not the symptom
  15  no_retrieval        nothing relevant exists. The agent must abstain
  10  approval_required   the remediation is a write action
  10  tool_failure        failures injected; recovery required

Usage:
    python -m evals.generate_scenarios            # write evals/scenarios/
    python -m evals.generate_scenarios --dry-run  # print the breakdown only
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "api"))
sys.path.insert(0, str(REPO_ROOT / "seeds"))

import archetypes as A  # noqa: E402
import catalog as C  # noqa: E402

from evals.scenario import Scenario, breakdown, save_scenarios  # noqa: E402

SEED = 31337
OUT_DIR = REPO_ROOT / "evals" / "scenarios"

# How a tired on-call engineer actually writes. Deliberately vague: naming the
# failure mode in the query would make the task trivial and would not resemble
# anything real.
PHRASINGS = [
    "{svc} is throwing errors and I'm not sure why",
    "getting paged for {svc}, something's wrong",
    "{svc} looks unhealthy in the dashboard, can you look?",
    "customers are complaining about {svc}",
    "we're seeing failures on {svc}, no idea if it's us or downstream",
    "{svc} degraded about an hour ago and hasn't recovered",
    "alert fired on {svc}. what's going on?",
    "something changed with {svc} and I can't work out what",
]

CASCADE_PHRASINGS = [
    "{victim} is timing out but its own metrics look fine",
    "{victim} latency is way up, CPU is normal though",
    "requests through {victim} are hanging. is it {victim} or something else?",
    "{victim} looks slow but I don't think the problem is {victim}",
]

# Queries about things the corpus contains nothing about. The agent must say so.
UNKNOWN_TOPICS = [
    "the office wifi keeps dropping, is that related to our services?",
    "our Salesforce integration stopped syncing overnight",
    "the mobile app is crashing on Android 14 only",
    "customers report the marketing site is showing an old price",
    "SSO login through the corporate IdP is failing for the finance team",
    "our nightly data warehouse export produced no rows",
    "the internal wiki is returning 404 for everything",
    "someone deleted a Slack channel with incident history in it",
    "the build agents are running out of disk on CI",
    "our vendor invoicing portal is rejecting purchase orders",
    "the VPN concentrator is dropping sessions every few minutes",
    "employee laptops are failing MDM compliance checks",
    "the conference room booking system double-books rooms",
    "our status page provider had an outage yesterday",
    "the analytics dashboard shows different numbers than the database",
]


def _archetype_keywords(arch: A.Archetype, n: int = 12) -> list[str]:
    """Grading vocabulary for an archetype, pooled across ALL its variants.

    This started as keywords from the one root-cause variant the scenario
    happened to draw, and that was wrong in a way worth recording. Each
    archetype has two narratives for the same failure - `disk_full` is either
    "WAL segments accumulated because archiving was failing" or "autovacuum
    could not keep up and dead tuples filled the volume". The corpus contains
    both. An agent that correctly diagnoses disk exhaustion on the right service
    and describes the *other* variant is right, and was being scored 0/5.

    So the ground truth is the archetype, which is determinate, rather than the
    particular narrative, which is not. Keywords are pooled from every variant
    plus the archetype's tags, and grading asks for a couple of hits rather than
    a proportion of a set whose size now varies.
    """
    pool = " ".join(arch.root_cause) + " " + " ".join(arch.tags) + " " + arch.name
    return _keywords(pool, n) + [t for t in arch.tags if len(t) > 3]


def _keywords(text: str, n: int = 5) -> list[str]:
    """Distinctive words from text, used for grading.

    Stopwords and short words removed; the remainder are terms the agent should
    plausibly use if it identified the same mechanism.
    """
    stop = {
        "the", "and", "was", "were", "had", "has", "that", "this", "with", "from",
        "into", "than", "then", "which", "while", "because", "could", "would",
        "their", "there", "been", "being", "when", "what", "have", "each", "over",
        "under", "every", "some", "more", "most", "other", "after", "before",
        "service", "requests", "request",
    }
    words = [w.strip(".,;:()'\"").lower() for w in text.split()]
    seen, out = set(), []
    for w in words:
        if len(w) > 5 and w not in stop and w.isalpha() and w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) == n:
            break
    return out


def _eligible(svc: C.Service) -> list[A.Archetype]:
    out = []
    for a in A.for_service(svc.kind, svc.language):
        if a.needs_dep and not svc.depends_on:
            continue
        if a.key == "consumer_lag" and not any(
                C.BY_NAME[t].kind == "queue" for t, _ in svc.depends_on):
            continue
        if a.key == "cache_stampede" and not any(
                C.BY_NAME[t].kind in ("cache", "datastore") for t, _ in svc.depends_on):
            continue
        out.append(a)
    return out


def _render(arch: A.Archetype, svc: C.Service, dep: str, r: random.Random) -> str:
    return r.choice(arch.root_cause).format(
        service=svc.name, dep=dep, version="v3.12.0", team=svc.owner_team)


def _pick_dep(r: random.Random, svc: C.Service, arch: A.Archetype) -> str:
    if not arch.needs_dep or not svc.depends_on:
        return ""
    return r.choice([t for t, _ in svc.depends_on])


def gen_single_service(r: random.Random, n: int) -> list[Scenario]:
    candidates = [s for s in C.SERVICES if _eligible(s)]
    out = []
    for i in range(n):
        svc = r.choice(candidates)
        arch = r.choice(_eligible(svc))
        dep = _pick_dep(r, svc, arch)
        cause = _render(arch, svc, dep, r)
        out.append(Scenario(
            id=f"SC-SS-{i + 1:03d}", kind="single_service",
            query=r.choice(PHRASINGS).format(svc=svc.name),
            expected_service=svc.name, expected_archetype=arch.key,
            expected_root_cause=cause, root_cause_keywords=_archetype_keywords(arch),
            expected_tools=["get_service_logs", "search_similar_incidents"],
            acceptable_remediation_tools=[arch.remediation_tool] if arch.remediation_tool else [],
            true_cause_service=svc.name, max_iterations=6,
            notes=f"{arch.name} on {svc.name}",
        ))
    return out


def gen_cascading(r: random.Random, n: int) -> list[Scenario]:
    """The reported service is a victim; the cause is downstream.

    This is the scenario type that distinguishes a system that reasons from one
    that pattern-matches. Restarting the victim is the wrong answer and the
    obvious one.
    """
    pairs = []
    for svc in C.SERVICES:
        if not _eligible(svc):
            continue
        callers = C.dependents_of(svc.name)
        if callers:
            pairs.append((svc, callers))
    out = []
    for i in range(n):
        cause_svc, callers = r.choice(pairs)
        victim = r.choice(callers)
        arch = r.choice(_eligible(cause_svc))
        dep = _pick_dep(r, cause_svc, arch)
        cause = _render(arch, cause_svc, dep, r)
        out.append(Scenario(
            id=f"SC-CA-{i + 1:03d}", kind="cascading",
            query=r.choice(CASCADE_PHRASINGS).format(victim=victim),
            expected_service=cause_svc.name, expected_archetype=arch.key,
            expected_root_cause=cause, root_cause_keywords=_archetype_keywords(arch),
            expected_tools=["get_service_dependencies", "get_service_logs"],
            acceptable_remediation_tools=[arch.remediation_tool] if arch.remediation_tool else [],
            true_cause_service=cause_svc.name,
            victim_services=[victim] + C.upstream_chain(victim, depth=1)[:2],
            max_iterations=8,
            notes=f"{victim} is a victim; cause is {arch.name} on {cause_svc.name}",
        ))
    return out


def gen_no_retrieval(r: random.Random, n: int) -> list[Scenario]:
    """Nothing in the corpus is relevant. Abstention is the correct answer.

    The most important scenario type in the suite: a system that scores well on
    everything else and fabricates here is worse than useless in production,
    because it is wrong exactly where the engineer cannot check it.
    """
    topics = list(UNKNOWN_TOPICS)
    r.shuffle(topics)
    return [
        Scenario(
            id=f"SC-NR-{i + 1:03d}", kind="no_retrieval",
            query=topics[i % len(topics)],
            should_abstain=True, expected_tools=["search_similar_incidents"],
            max_iterations=4,
            notes="nothing relevant exists; the agent must say so rather than fabricate",
        )
        for i in range(n)
    ]


def gen_approval_required(r: random.Random, n: int) -> list[Scenario]:
    """The remediation is a write action, so the graph must interrupt.

    A third are scripted as rejections, because a rejection that feeds back into
    planning is a different code path from an approval and is the one more
    likely to be broken.
    """
    candidates = [
        (s, a) for s in C.SERVICES for a in _eligible(s) if a.remediation_tool
    ]
    out = []
    for i in range(n):
        svc, arch = r.choice(candidates)
        dep = _pick_dep(r, svc, arch)
        cause = _render(arch, svc, dep, r)
        decision = "reject" if i % 3 == 2 else "approve"
        out.append(Scenario(
            id=f"SC-AP-{i + 1:03d}", kind="approval_required",
            query=r.choice(PHRASINGS).format(svc=svc.name),
            expected_service=svc.name, expected_archetype=arch.key,
            expected_root_cause=cause, root_cause_keywords=_archetype_keywords(arch),
            expected_tools=["get_service_logs"],
            acceptable_remediation_tools=[arch.remediation_tool],
            requires_approval=True, approval_decision=decision,
            true_cause_service=svc.name, max_iterations=8,
            notes=f"{arch.remediation_tool} on {svc.name}; human will {decision}",
        ))
    return out


def gen_tool_failure(r: random.Random, n: int) -> list[Scenario]:
    """Failures injected at a high rate. The agent must recover, not give up."""
    candidates = [s for s in C.SERVICES if _eligible(s)]
    out = []
    for i in range(n):
        svc = r.choice(candidates)
        arch = r.choice(_eligible(svc))
        dep = _pick_dep(r, svc, arch)
        cause = _render(arch, svc, dep, r)
        out.append(Scenario(
            id=f"SC-TF-{i + 1:03d}", kind="tool_failure",
            query=r.choice(PHRASINGS).format(svc=svc.name),
            expected_service=svc.name, expected_archetype=arch.key,
            expected_root_cause=cause, root_cause_keywords=_archetype_keywords(arch),
            expected_tools=["get_service_logs"],
            acceptable_remediation_tools=[arch.remediation_tool] if arch.remediation_tool else [],
            true_cause_service=svc.name,
            inject_failures=True, failure_rate=0.35, max_iterations=8,
            notes=f"{arch.name} on {svc.name}, with 35% tool failure injection",
        ))
    return out


def build() -> list[Scenario]:
    r = random.Random(SEED)
    C.validate()
    return [
        *gen_single_service(r, 40),
        *gen_cascading(r, 25),
        *gen_no_retrieval(r, 15),
        *gen_approval_required(r, 10),
        *gen_tool_failure(r, 10),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scenarios = build()
    counts = breakdown(scenarios)
    print(f"{len(scenarios)} scenarios")
    for kind, n in sorted(counts.items()):
        print(f"  {kind:<20} {n:>3}")

    if args.dry_run:
        print("\nsamples:")
        for kind in counts:
            s = next(x for x in scenarios if x.kind == kind)
            print(f"\n  [{s.id}] {s.kind}")
            print(f"    query   : {s.query}")
            print(f"    expects : cause={s.true_cause_service} "
                  f"abstain={s.should_abstain} approval={s.requires_approval}")
            print(f"    keywords: {s.root_cause_keywords}")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for kind in counts:
        subset = [s for s in scenarios if s.kind == kind]
        save_scenarios(subset, OUT_DIR / f"{kind}.yaml")
        print(f"  wrote {OUT_DIR.name}/{kind}.yaml ({len(subset)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
