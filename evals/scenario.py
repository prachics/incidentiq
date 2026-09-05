"""Scenario schema and loader.

A scenario is a YAML file describing one investigation with a known answer:
the query, the expected root cause, the tool calls that ought to be made
(order-independent), and the remediations that count as correct.

Grading is deliberately structural rather than free-text comparison. "Did the
agent reach an acceptable remediation" is answerable from the proposal object;
"is this prose equivalent to that prose" is not, and pretending otherwise is
how eval suites end up measuring phrasing.
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
    # Keywords that must appear in the stated root cause. Cheap, robust, and
    # honest about what it measures: topical correctness, not phrasing.
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
    approval_decision: Literal["approve", "reject", "modify"] | None = None

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
