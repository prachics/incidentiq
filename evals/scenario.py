"""Scenario schema and loader.

A scenario is a YAML file describing one investigation with a known answer:
the query, the expected root cause, the tool calls that ought to be made
(order-independent), and the remediations that count as correct.

The grade comes from facts about the run (the action proposed, whether the agent
abstained) and from what the agent's conclusion asserts: the service at fault named
as failing, and the scenario's failure mode in its stated root cause, read by the
phrasings in evals/mechanism.py rather than by likeness to a reference answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

ScenarioKind = Literal["single_service", "cascading", "no_retrieval",
                       "approval_required", "tool_failure"]


@dataclass
class Scenario:
    id: str
    kind: ScenarioKind
    query: str

    # ── Ground truth ────────────────────────────────────────
    expected_service: str | None = None
    expected_archetype: str | None = None
    expected_root_cause: str = ""
    # For a scenario with an archetype, its words, reported beside the grade as how
    # many an answer used (the grade reads the archetype's assertion phrasings: see
    # evals/mechanism.py). For one without, the grade itself: at least two of them
    # (or all, if fewer) in the stated root cause.
    root_cause_keywords: list[str] = field(default_factory=list)
    # Tools that ought to be called. Order-independent: several diagnostic
    # sequences are legitimate and grading on order would penalise good ones.
    expected_tools: list[str] = field(default_factory=list)
    acceptable_remediation_tools: list[str] = field(default_factory=list)
    relevant_doc_ids: list[str] = field(default_factory=list)

    # ── Expected behaviour ──────────────────────────────────
    should_abstain: bool = False
    requires_approval: bool = False
    # The service that is the *cause*, when different from the one reported.
    # Cascading scenarios are graded on getting this right, not the symptom.
    true_cause_service: str | None = None
    victim_services: list[str] = field(default_factory=list)

    # ── Run configuration ───────────────────────────────────
    inject_failures: bool = False
    failure_rate: float = 0.0
    max_iterations: int = 8
    # "modify": approve with edited arguments, which are what must then run.
    approval_decision: Literal["approve", "modify", "reject"] | None = None

    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scenario:
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"scenario {data.get('id')}: unknown fields {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v not in (None, [], "", False, 0.0)}


def load_scenarios(directory: Path | str) -> list[Scenario]:
    directory = Path(directory)
    scenarios = []
    for path in sorted(directory.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text())
        if isinstance(raw, list):
            scenarios.extend(Scenario.from_dict(item) for item in raw)
        else:
            scenarios.append(Scenario.from_dict(raw))
    ids = [s.id for s in scenarios]
    if len(ids) != len(set(ids)):
        dupes = {i for i in ids if ids.count(i) > 1}
        raise ValueError(f"duplicate scenario ids: {sorted(dupes)}")
    return scenarios


def save_scenarios(scenarios: list[Scenario], path: Path | str) -> None:
    Path(path).write_text(
        yaml.safe_dump([s.to_dict() for s in scenarios], sort_keys=False, width=100)
    )


def breakdown(scenarios: list[Scenario]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in scenarios:
        out[s.kind] = out.get(s.kind, 0) + 1
    return out
