#!/usr/bin/env python3
"""The evaluation harness.

    python -m evals.run                      # all 100 scenarios
    python -m evals.run --kind cascading     # one category
    python -m evals.run --limit 10           # a dev subset
    python -m evals.run --no-judge           # skip groundedness (much faster)
    python -m evals.run --replay evals/results/run_X.json   # rerun only its failures
    python -m evals.run --regrade evals/results/run_X.json  # grade its saved answers again

Writes a timestamped JSON file and a markdown summary to evals/results/.

Grading is from facts about the run (which tools were called, which service the
agent acted on, whether it abstained) and from two readings of its prose: the
service it names, and the failure mode its stated root cause asserts, against
each archetype's vocabulary (evals/mechanism.py). Groundedness, when judged, is
the judge's (judge.py), reported with its limits.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
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
sys.path.insert(0, str(REPO_ROOT / "seeds"))

import catalog  # noqa: E402
from incidentiq.config import Settings, get_settings  # noqa: E402

from evals import fixtures, mechanism  # noqa: E402
from evals.scenario import Scenario, load_scenarios  # noqa: E402
from evals.words import keyword_hits  # noqa: E402

SCENARIO_DIR = REPO_ROOT / "evals" / "scenarios"
SERVICE_NAMES = {s.name.lower() for s in catalog.SERVICES}
# Set from --max-iterations so it lands in the results file: a run with a
# reduced budget is not comparable with a full one, and the number must say so.
ITERATION_CAP: int | None = None
# Identifies this process's fixture rows, so a later run can tell its own
# leftovers from another's.
RUN_ID: str = "adhoc"
RESULTS_DIR = REPO_ROOT / "evals" / "results"

# Results are appended one line at a time as each scenario finishes, so an
# interrupted run keeps what it had done (--resume).
PROGRESS_DIR = RESULTS_DIR / "progress"


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

    # What the agent concluded, so a failure can be read rather than guessed at;
    # `graded_text` is exactly what the grader read, so a result can be regraded.
    graded_text: str = ""
    root_cause: str = ""
    remediation_tool: str | None = None
    remediation_arguments: dict[str, Any] = field(default_factory=dict)


def _grade(scenario: Scenario, state: dict) -> ScenarioResult:
    proposal = state.get("proposal") or {}
    if not isinstance(proposal, dict):
        proposal = {}
    calls = [c for c in (state.get("tool_calls") or []) if isinstance(c, dict)]
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
    r.root_cause = str(proposal.get("root_cause") or "")
    tool = proposal.get("remediation_tool")
    r.remediation_tool = tool if isinstance(tool, str) else None
    arguments = proposal.get("remediation_arguments")
    r.remediation_arguments = arguments if isinstance(arguments, dict) else {}

    r.recovered_from_failure = recovered(calls)

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
    # Anywhere in the conclusion: which field a model names the service in (root
    # cause, summary or the action's arguments) is not what is being scored.
    text = " ".join([
        str(proposal.get("root_cause") or ""),
        str(proposal.get("remediation") or ""),
        # default=str so an unserialisable value degrades to text rather than
        # raising: the grader must survive any shape the model produces.
        json.dumps(proposal.get("remediation_arguments") or {}, default=str),
        str(state.get("final_summary") or ""),
    ]).lower()
    r.graded_text = text
    _identify(scenario, r)

    # ── Remediation ─────────────────────────────────────────
    if scenario.acceptable_remediation_tools:
        r.correct_remediation = r.remediation_tool in scenario.acceptable_remediation_tools
    else:
        r.correct_remediation = True   # no specific action expected

    _verdict(scenario, r)
    return r


def _cause(scenario: Scenario) -> str:
    return (scenario.true_cause_service or scenario.expected_service or "").lower()


def _identify(scenario: Scenario, r: ScenarioResult) -> None:
    """Whether the conclusion names the service at fault as failing, or acts on it.

    Named where a failure is asserted, not anywhere: "payment-db is healthy;
    payment-service has a full disk" mentions payment-db and clears it. On a
    cascading scenario, acting on a victim, or naming only a victim as failing, is
    the characteristic wrong answer and is named as such. Everything read here came
    from a language model, so no type is assumed."""
    cause = _cause(scenario)
    asserted = mechanism.asserted(r.graded_text)
    target = r.remediation_arguments.get("service") if isinstance(r.remediation_arguments, dict) \
        else None
    target = target.lower() if isinstance(target, str) else None
    r.identified_cause = bool(cause) and (_mentions(asserted, cause) or target == cause)
    victims = {v.lower() for v in scenario.victim_services if isinstance(v, str)}
    acted_on_victim = bool(target) and target in victims and target != cause
    if scenario.kind == "cascading" and (acted_on_victim or not r.identified_cause):
        if acted_on_victim or any(_mentions(asserted, v) for v in victims):
            r.identified_cause = False
            r.failure_note = ("acted on the victim service instead of the downstream cause"
                              if acted_on_victim else
                              "blamed the victim service instead of the downstream cause")


def _bound(scenario: Scenario, archetype: str, text: str) -> bool:
    """Whether the failure is asserted of the service at fault. Each smallest span of a
    clause that asserts it (a segment, or neighbouring segments when the phrasing runs
    across them: "retried with no backoff, tripling load") has owners: the services it
    names. The failure is bound when a span names the service at fault, or when no
    span names any service; a span that names only another service is a conflicting
    attribution, which an unattributed span elsewhere does not outweigh. A coordinated
    part, or a clause that opens without naming a service, is about the one named
    before it ("payment-service has high disk usage and disk space issues"; "The volume
    on payment-service filled. WAL accumulated."), and a service named as where connections
    go ("all user-db connections", "connections to user-db") is not their owner; one
    named as holding them ("payment-service's connections") is."""
    cause = _cause(scenario)

    def named(text: str) -> set[str]:
        return {s for s in SERVICE_NAMES if _mentions(_CONNECTED_TO.sub(" ", text), s)}

    owners_found: list[set[str]] = []
    subject: set[str] = set()
    for clause, fails in mechanism.clauses(text):
        owners = []
        for i, segment in enumerate(clause):
            services = named(segment.text or segment.written)
            # A coordinated part, or a sentence that goes on about what came before,
            # is about the service last named.
            if not services and (segment.joined or i == 0):
                services = subject
            subject = services
            owners.append(services)
        if not fails:
            continue
        spans = [(s.text, owner) for s, owner in zip(clause, owners, strict=True) if s.text]
        for start, end in _smallest_matches([t for t, _ in spans], archetype):
            owners_found.append(set().union(*(o for _, o in spans[start:end])))
    if any(cause in owner for owner in owners_found):
        return True
    return bool(owners_found) and not any(owners_found)


# Where connections go or what a cache holds, not who holds them: anything connections
# go "to" ("connection pool to catalog-db"), and a store (a database, cache, search
# cluster or queue) named as the far end ("user-db connections", "the pool on user-db",
# "the fraud-db entries"). An application named on, in, of or for a pool holds it
# ("the connection pool on payment-service"), as does a possessive. And "a cache flush
# left X empty" names the cache, not the culprit.
_STORES = "|".join(re.escape(s.name) for s in sorted(
    catalog.SERVICES, key=lambda s: -len(s.name))
    if s.kind in ("datastore", "cache", "search", "queue"))
_CONNECTED_TO = re.compile(
    r"\b(?:connection\s+)?(?:connections?|pool)\s+(?:to|into)\s+[a-z][a-z0-9-]*\b|"
    rf"\b(?:connection\s+)?(?:connections?|pool|cache|entries)\s+(?:on|for|of|in)\s+"
    rf"(?:{_STORES})(?![\w-])|"
    rf"(?<![\w-])(?:{_STORES})\s+(?:db\s+)?(?:connections?|pool|cache|entries)\b|"
    r"\bflush\w*\b.{0,40}?\bleft\s+(?:the\s+)?[a-z][a-z0-9-]*\s+(?:cache\s+)?empty\b")


def _smallest_matches(texts: list[str], archetype: str) -> list[tuple[int, int]]:
    """The spans [start, end) of neighbouring texts whose joined text asserts the
    archetype while no shorter span inside them does."""
    found: list[tuple[int, int]] = []
    for length in range(1, len(texts) + 1):
        for start in range(len(texts) - length + 1):
            end = start + length
            if any(s >= start and e <= end for s, e in found):
                continue
            if mechanism.matches(" ".join(texts[start:end]))[archetype]:
                found.append((start, end))
    return found


def _mentions(text: str, service: str) -> bool:
    """The service's name as a name of its own, not inside a longer one."""
    return re.search(rf"(?<![\w-]){re.escape(service)}(?![\w-])", text) is not None


def _mechanism(scenario: Scenario, r: ScenarioResult) -> tuple[bool, str | None]:
    """Whether the agent's stated root cause describes the scenario's failure mode
    (one of its assertion phrasings, and no other archetype's matched more: see
    evals/mechanism.py), and if not, the one it describes. The root cause alone is
    read: the remediation and the
    summary give advice ("add a circuit breaker to prevent this") in the words of
    other failure modes. Scenarios without an archetype use their listed keywords."""
    text = r.root_cause.lower()
    if scenario.expected_archetype in mechanism.VOCABULARY:
        vocabulary = mechanism.VOCABULARY[scenario.expected_archetype]
        r.total_keywords = len(vocabulary)
        r.matched_keywords = keyword_hits(vocabulary, mechanism.asserted(text))
        if mechanism.names(scenario.expected_archetype, text) \
                and _bound(scenario, scenario.expected_archetype, text):
            return True, None
        return False, mechanism.closest(text)
    r.total_keywords = len(scenario.root_cause_keywords)
    r.matched_keywords = keyword_hits(scenario.root_cause_keywords, text)
    return (r.total_keywords == 0
            or r.matched_keywords >= min(mechanism.MIN_HITS, r.total_keywords)), None


def _verdict(scenario: Scenario, r: ScenarioResult) -> None:
    """Task success, and why not, once the facts are in."""
    mechanism_ok, described = _mechanism(scenario, r)
    proposed_tool = r.remediation_tool
    r.task_success = bool(r.completed and r.identified_cause and mechanism_ok
                          and r.correct_remediation)

    if not r.task_success and not r.failure_note:
        if not r.completed:
            r.failure_note = "did not reach a conclusion within the iteration cap"
        elif not r.identified_cause:
            r.failure_note = "did not name the causing service"
        elif not mechanism_ok:
            closer = f"described {described}; " if described else ""
            r.failure_note = (f"named the service but not the failure mode ({closer}"
                              f"{r.matched_keywords} of {r.total_keywords} "
                              f"{scenario.expected_archetype or 'expected'} terms)")
        elif not r.correct_remediation:
            r.failure_note = (f"proposed {proposed_tool!r}, expected one of "
                              f"{scenario.acceptable_remediation_tools}")


# Notes the grading of a conclusion writes. A result failed for anything else
# (abstaining, an approval refused, an action that failed, a harness error) failed
# before its conclusion was graded, and a regrade leaves it as it was.
_VERDICT_NOTES = ("did not reach a conclusion", "did not name the causing service",
                  "named the service but not the failure mode", "proposed ",
                  "acted on the victim service", "blamed the victim service")


def regrade(path: Path) -> tuple[dict, dict]:
    """Grade a results file's saved conclusions again with the current grader, in
    place, and write its markdown beside it. Returns the old and new summaries."""
    data = json.loads(path.read_text())
    if any("root_cause" not in r for r in data["results"]):
        raise ValueError(f"{path} predates saved root causes, so it cannot be regraded")
    recorded = data["summary"].get("scenarios_sha256")
    if recorded is not None and recorded != provenance()["scenarios_sha256"]:
        raise ValueError(f"the scenario files have changed since {path} ran, so its answers "
                         "would be graded against other expectations")
    scenarios = {s.id: s for s in load_scenarios(SCENARIO_DIR)}
    results = [ScenarioResult(**r) for r in data["results"]]
    for r in results:
        sc = scenarios[r.scenario_id]
        if sc.should_abstain or r.abstained or r.error:
            continue
        if r.failure_note and not r.failure_note.startswith(_VERDICT_NOTES):
            continue
        r.failure_note = ""
        _identify(sc, r)
        _verdict(sc, r)
    old = data["summary"]
    new = {**old, "task_completion": round(sum(r.task_success for r in results)
                                           / len(results), 4),
           "regraded_with": provenance()["revision"]}
    for kind, row in new.get("by_kind", {}).items():
        subset = [r for r in results if r.kind == kind]
        row["task_completion"] = round(sum(r.task_success for r in subset) / len(subset), 4)
    stamp = datetime.fromisoformat(old["generated_at"]).strftime("%Y%m%dT%H%M%SZ")
    write_report(new, results, failure_analysis(results), stamp,
                 json_path=path, md_path=path.with_suffix(".md"))
    return old, new


def progress_path(settings: Settings, kind: str | None, cap: int | None,
                  judged: bool) -> Path:
    """One progress file per configuration.

    The configuration is in the filename rather than checked after loading,
    because resuming into a run with a different model, cap or judge setting
    would silently mix incomparable results - the kind of error that produces a
    number nobody can reproduce.
    """
    model = {
        "ollama": settings.ollama_model,
        "anthropic": settings.anthropic_model,
        "stub": "stub",
    }.get(settings.llm_provider, settings.llm_provider)
    model = model.replace(":", "-").replace("/", "-")
    parts = [settings.llm_provider, model, f"cap{cap or 'full'}",
             "judged" if judged else "nojudge", f"seed{settings.llm_seed}", kind or "all"]
    return PROGRESS_DIR / ("progress_" + "_".join(parts) + ".jsonl")


def load_progress(path: Path) -> dict[str, ScenarioResult]:
    """Read completed scenarios from a previous run of the same configuration."""
    if not path.exists():
        return {}
    done: dict[str, ScenarioResult] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            done[record["scenario_id"]] = ScenarioResult(**record)
        except (json.JSONDecodeError, TypeError):
            # A partially-written final line is expected after a hard kill.
            # Skip it; that scenario simply re-runs.
            continue
    return done


def stamp_path(progress: Path) -> Path:
    """Beside a progress file: the revision and scenario files it was started with."""
    return progress.with_name(progress.stem + ".provenance.json")


def progress_stamp(settings: Settings, scenarios: list[Scenario]) -> dict[str, Any]:
    """What a run's results depend on besides the options in its progress file's name:
    the code, the scenario files, the execution settings and the scenarios chosen."""
    return {**provenance(), "execution": {name: getattr(settings, name) for name in EXECUTION},
            "scenario_ids": sorted(s.id for s in scenarios)}


def resumable(progress: Path, current: dict[str, Any]) -> str | None:
    """Why this run may not continue a progress file, or None. Results made by another
    revision, on other scenarios or with other settings would be reported as this run's."""
    if current.get("execution", {}).get("control_plane_url"):
        return ("actions go to an external control plane, whose fault settings are that "
                "server's: a resumed run could not be told from a mixed one")
    stamp = stamp_path(progress)
    if not stamp.exists():
        return "it does not record the revision, settings and scenarios it was started with"
    started = json.loads(stamp.read_text())
    differ = sorted(k for k in started.keys() | current.keys() if started.get(k) != current.get(k))
    if differ:
        return f"it was started with a different {', '.join(differ)}"
    if current["revision"].endswith("-dirty"):
        return "uncommitted changes cannot be told apart from other uncommitted changes"
    return None


def append_progress(path: Path, result: ScenarioResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(asdict(result), default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())   # survive a power cut, not just a process kill


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

            # Approval-gated scenarios: decide and resume, so the interrupt/resume
            # path is exercised rather than assumed.
            result_holder: dict[str, Any] = {}
            refused = None
            edited = None
            if state.get("status") == "awaiting_approval":
                approval_id = (state.get("pending_approval") or {}).get("approval_id")
                refused, edited = _decide(cfg, approval_id, scenario)
                if refused is None:
                    state = runner.resume_for_decision(incident_id, approval_id)
                    result_holder["approval_resolved"] = True
                    if edited is not None:
                        refused = _ran_edited(cfg, approval_id, edited)

            r = _grade(scenario, state)
            if refused is not None:
                # A human could not have approved it either: the run stops here.
                r.completed = r.task_success = False
                r.failure_note = f"the approval was refused: {refused}"
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
        # every later scenario for that service look like it had evidence it was
        # never given, so a cleanup that fails stops the run (CleanupFailed); the
        # next run's purge restores the corpus from the tracking rows.
        if seeded is not None:
            import psycopg as _pg
            try:
                with _pg.connect(cfg.database_url, autocommit=True) as c:
                    fixtures.clear(c, seeded)
            except Exception as exc:
                raise CleanupFailed(scenario.id, exc) from exc


class CleanupFailed(Exception):
    """A scenario's fixtures could not be removed: no other scenario may run on the
    corpus as it was left."""

    def __init__(self, scenario_id: str, cause: Exception):
        super().__init__(f"cleaning up {scenario_id} failed: {type(cause).__name__}: {cause}")


class _NoWorker:
    """The decision endpoint queues a resume; the harness resumes in process."""

    def resume_after_decision(self, investigation_id: str, approval_id: str) -> None:
        pass


def _decide(settings: Settings, approval_id: str | None,
            scenario: Scenario) -> tuple[str | None, dict[str, Any] | None]:
    """Decide through the same code as POST /approvals/{id}/decision, so the
    harness is refused exactly where a human approver would be (arguments that
    cannot be executed, for example). Returns why it was refused (or None) and,
    for a "modify" scenario, the edited arguments it approved."""
    import psycopg as _pg
    from fastapi import HTTPException
    from incidentiq.api.approvals import DecisionIn, decide
    from incidentiq.api.auth import Principal
    from psycopg.rows import dict_row

    edited = None
    with _pg.connect(settings.database_url, autocommit=True, row_factory=dict_row) as conn:
        if scenario.approval_decision == "reject":
            body = DecisionIn(decision="reject", reason="the eval harness rejected this action")
        elif scenario.approval_decision == "modify":
            row = conn.execute("SELECT proposed_tool, proposed_arguments FROM approvals "
                               "WHERE id = %s", (approval_id,)).fetchone()
            edited = (edit_arguments(row["proposed_tool"], row["proposed_arguments"])
                      if row else None)
            body = DecisionIn(decision="approve", arguments=edited)
        else:
            body = DecisionIn(decision="approve")
        try:
            decide(approval_id or "", body, conn, Principal("eval-harness", "approver"),
                   _NoWorker())
        except HTTPException as exc:
            detail = exc.detail
            if isinstance(detail, dict):
                errors = "; ".join(e["msg"] for e in detail.get("errors", []))
                detail = f"{detail.get('message')}: {errors}" if errors else detail.get("message")
            return f"{exc.status_code} {detail}", edited
    return None, edited


def edit_arguments(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """The edit an approver makes in a "modify" scenario, the same every run: a
    reason of their own, and for scaling, one replica more."""
    edited = {**arguments, "reason": "edited by the approver: " + str(arguments.get("reason", ""))}
    if tool == "scale_service":
        edited["replica_count"] = min(50, int(arguments.get("replica_count") or 1) + 1)
    return edited


def _ran_edited(settings: Settings, approval_id: str, edited: dict[str, Any]) -> str | None:
    """None if the approval was recorded as modified and the action ran with the
    edited arguments; otherwise what went wrong."""
    import psycopg as _pg
    from psycopg.rows import dict_row

    with _pg.connect(settings.database_url, autocommit=True, row_factory=dict_row) as conn:
        decision = conn.execute("SELECT decision FROM approvals WHERE id = %s",
                                (approval_id,)).fetchone()
        action = conn.execute("SELECT arguments FROM actions WHERE approval_id = %s",
                              (approval_id,)).fetchone()
    if not decision or decision["decision"] != "modified":
        return f"the edit was recorded as {decision and decision['decision']!r}, not 'modified'"
    if not action or action["arguments"] != edited:
        return "the action did not run with the edited arguments"
    return None


def recovered(calls: list[dict[str, Any]]) -> bool:
    """An injected failure followed by a successful retry of the same call: the
    same tool with the same arguments. Another tool's success, a success before
    the failure, or a failure nobody injected is not a recovery."""
    def same(c: dict[str, Any]) -> tuple[str, str]:
        return c["tool_name"], json.dumps(c.get("arguments") or {}, sort_keys=True)

    return any(
        c.get("injected_failure") and any(
            same(d) == same(c) and d["status"] in ("success", "partial") for d in calls[i + 1:])
        for i, c in enumerate(calls))

# What a run's numbers depend on besides its settings: the agent and the harness,
# and the seeds the fixtures and the grader import (archetypes, catalog, generator).
CODE = ("api", "evals", "seeds")


def provenance(root: Path = REPO_ROOT) -> dict[str, str]:
    """What produced a run's numbers: the code's revision (marked if the agent, the
    harness or the seeds they import had uncommitted changes; results files aside)
    and a fingerprint of the scenario files."""
    import hashlib
    import subprocess

    try:
        git = ["git", "-C", str(root)]
        revision = subprocess.run([*git, "rev-parse", "--short", "HEAD"], capture_output=True,
                                  text=True, check=True).stdout.strip()
        changed = subprocess.run([*git, "status", "--porcelain", "--", *CODE,
                                  ":(exclude)evals/results"],
                                 capture_output=True, text=True, check=True).stdout.strip()
        revision += "-dirty" if changed else ""
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    digest = hashlib.sha256()
    for path in sorted((root / "evals" / "scenarios").glob("*.yaml")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return {"revision": revision, "scenarios_sha256": digest.hexdigest()[:16]}


# Settings besides the model that change what a run does, recorded so a replay can
# restore them; none is a secret.
EXECUTION = ("tool_max_retries", "failure_injection_seed", "control_plane_url",
             "control_plane_timeout_s", "control_plane_fault_rate", "action_max_attempts",
             "action_backoff_s", "embedding_provider", "embedding_model", "embedding_dim")


def aggregate(results: list[ScenarioResult], settings: Settings,
              judge: bool = False) -> dict[str, Any]:
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
        **provenance(),
        "n_scenarios": n,
        "model": settings.ollama_model if settings.llm_provider == "ollama"
                 else settings.anthropic_model,
        "provider": settings.llm_provider,
        "iteration_cap_override": ITERATION_CAP,
        "seed": settings.llm_seed,
        "judge": judge,
        "execution": {name: getattr(settings, name) for name in EXECUTION},
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
        # A host suspend inflates whichever scenario was in flight, and the
        # result looks like a slow scenario rather than a stopped clock.
        "latency_outlier_ratio": round(
            max(latencies) / statistics.median(latencies), 1
        ) if statistics.median(latencies) else 0,
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
                 stamp: str, json_path: Path | None = None,
                 md_path: Path | None = None) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = json_path or RESULTS_DIR / f"run_{stamp}.json"
    md_path = md_path or RESULTS_DIR / f"run_{stamp}.md"

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

    if summary.get("latency_outlier_ratio", 0) > 5:
        lines += ["", "> **Latency figures are unreliable for this run.** The slowest "
                  f"scenario took {summary['latency_outlier_ratio']:.0f}x the median, "
                  "which is the signature of the host suspending mid-run rather than of "
                  "a slow scenario. Task completion and tool success are unaffected; "
                  "the latency columns should not be quoted."]

    if summary.get("iteration_cap_override"):
        lines += ["", "> **Reduced iteration budget.** Every scenario was capped at "
                  f"{summary['iteration_cap_override']} iterations rather than its own "
                  "budget of 4-8. Task completion measures the capped agent, and this "
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


def run_all(todo: list[Scenario], settings: Settings, judge_llm, cap: int | None,
            progress: Path) -> list[ScenarioResult]:
    """Run the scenarios in turn, each result saved before the next starts. A cleanup
    that fails (CleanupFailed) ends the run there."""
    results = []
    for i, scenario in enumerate(todo, 1):
        r = run_scenario(scenario, settings, judge_llm, cap)
        results.append(r)
        # Write before printing: if the process dies during the next scenario,
        # this one is already safe on disk.
        append_progress(progress, r)
        mark = "ok  " if r.task_success else ("ERR " if r.error else "fail")
        print(f"  [{i:>3}/{len(todo)}] {mark} {r.scenario_id:<12} "
              f"{r.latency_s:>6.1f}s  {r.failure_note[:58]}")
    return results


def replay_plan(path: Path) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    """The scenarios that failed in a results file, the options to rerun them with
    (model, iteration cap, judge, seed) and the execution settings (failure injection,
    retries, control plane): all as that run had them, or the comparison is void."""
    source = json.loads(path.read_text())
    base = source["summary"]
    if "execution" not in base or "judge" not in base:
        raise ValueError(f"{path} does not record its execution settings, so a replay "
                         "could not reproduce them")
    if base["execution"].get("control_plane_url"):
        raise ValueError(f"{path} sent its actions to an external control plane, whose fault "
                         "settings are that server's and are not recorded, so a replay could not "
                         "reproduce them")
    if base.get("scenarios_sha256") != provenance()["scenarios_sha256"]:
        raise ValueError(f"the scenario files have changed since {path} ran (a failure rate, "
                         "a query or a fixture may differ), so a replay would not rerun them")
    failed = [r["scenario_id"] for r in source["results"] if not r["task_success"]]
    options = {"provider": base["provider"], "model": base["model"],
               "max_iterations": base.get("iteration_cap_override"),
               "no_judge": not base["judge"], "seed": base.get("seed")}
    return failed, options, dict(base["execution"])


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
    parser.add_argument("--resume", action="store_true",
                        help="continue a run of this exact configuration, skipping "
                             "scenarios already recorded. Results are appended as each "
                             "scenario finishes, so an interrupted run loses at most the "
                             "one in flight.")
    parser.add_argument("--no-judge", action="store_true",
                        help="skip groundedness scoring (roughly halves runtime)")
    parser.add_argument("--replay", metavar="RUN_JSON",
                        help="rerun only the scenarios that failed in this results file, "
                             "with its model, iteration cap and judge setting")
    parser.add_argument("--seed", type=int, default=7,
                        help="sampling seed for the local model (default 7), so a rerun "
                             "differs only where the code does")
    parser.add_argument("--provider", help="override LLM_PROVIDER")
    parser.add_argument("--model", help="override the model name")
    parser.add_argument("--regrade", metavar="RUN_JSON", nargs="+",
                        help="grade these results files' saved conclusions again with the "
                             "current grader, in place; no model is run")
    args = parser.parse_args()

    if args.regrade:
        for path in map(Path, args.regrade):
            try:
                old, new = regrade(path)
            except ValueError as exc:
                print(exc, file=sys.stderr)
                return 1
            print(f"{path}: task completion {old['task_completion']:.1%} -> "
                  f"{new['task_completion']:.1%}")
        return 0

    replay_ids: list[str] = []
    replay_of = None
    execution: dict[str, Any] = {}
    if args.replay:
        if args.kind or args.sample or args.limit:
            parser.error("--replay takes its scenarios from the results file")
        try:
            replay_ids, config, execution = replay_plan(Path(args.replay))
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
        if not replay_ids:
            print(f"nothing failed in {args.replay}")
            return 0
        replay_of = Path(args.replay).stem
        for key, value in config.items():
            setattr(args, key, value)

    settings = get_settings()
    updates: dict[str, Any] = {"llm_seed": args.seed, **execution}
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

    # This run owns the database until it exits (see fixtures.EVAL_LOCK).
    import contextlib
    holding = contextlib.ExitStack()
    try:
        holding.enter_context(fixtures.exclusive(settings.database_url))
    except RuntimeError as exc:
        print(f"not starting: {exc}", file=sys.stderr)
        return 1

    # Remove what a killed run left behind before measuring anything: its planted
    # evidence would be background noise for every scenario that follows.
    import psycopg as _pg
    with _pg.connect(settings.database_url, autocommit=True) as _c:
        purged = fixtures.purge_orphans(_c, except_run_id=RUN_ID)
    if purged:
        print(f"purged orphaned fixture rows from a previous run: {purged}")

    scenarios = load_scenarios(SCENARIO_DIR)
    if replay_ids:
        missing = sorted(set(replay_ids) - {s.id for s in scenarios})
        if missing:
            print(f"scenarios no longer in the suite: {missing}", file=sys.stderr)
            return 1
        scenarios = [s for s in scenarios if s.id in set(replay_ids)]
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

    progress = progress_path(settings, f"replay-{replay_of}" if replay_of else args.kind,
                             args.max_iterations, bool(judge_llm))
    completed: dict[str, ScenarioResult] = {}
    current = progress_stamp(settings, scenarios)
    if args.resume:
        completed = load_progress(progress)
        why = resumable(progress, current) if completed else None
        if why:
            print(f"\ncannot resume {progress.name}: {why}.\n"
                  f"  delete it to start over", file=sys.stderr)
            return 1
        if completed:
            print(f"resuming: {len(completed)} scenario(s) already done in "
                  f"{progress.name}")
        else:
            print(f"nothing to resume from {progress.name}; starting fresh")
    elif progress.exists():
        # Refuse to append this run's results onto a previous run's file. Mixing
        # them would produce an aggregate over two different executions that
        # looks like one.
        print(f"\n{progress.relative_to(REPO_ROOT)} already exists.\n"
              f"  --resume  to continue it\n"
              f"  or delete it to start over", file=sys.stderr)
        return 1

    if not completed:
        progress.parent.mkdir(parents=True, exist_ok=True)
        progress.unlink(missing_ok=True)
        stamp_path(progress).write_text(json.dumps(current) + "\n")
    todo = [sc for sc in scenarios if sc.id not in completed]
    if completed:
        print(f"  {len(todo)} remaining of {len(scenarios)}")

    results: list[ScenarioResult] = list(completed.values())
    t0 = time.perf_counter()
    try:
        results += run_all(todo, settings, judge_llm, args.max_iterations, progress)
    except CleanupFailed as exc:
        print(f"\nstopped: {exc}. Its results are not kept; the next run purges what it "
              f"left, and --resume continues from {progress.name}.", file=sys.stderr)
        return 1

    summary = aggregate(results, settings, judge=judge_llm is not None)
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
    if len(results) == len(scenarios):
        progress.unlink(missing_ok=True)   # suite finished; the resume file is spent
        stamp_path(progress).unlink(missing_ok=True)
    else:
        print(f"progress kept in {progress.relative_to(REPO_ROOT)} - "
              f"re-run with --resume to continue")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
