"""Seed the mock infrastructure for one scenario, then remove it again.

Why this exists
---------------
The corpus generator plants error signals for twelve "live situations". The
scenario generator produces a hundred scenarios spanning every service and
archetype. Those two sets barely overlap, so most scenarios described a failure
that left no trace anywhere the diagnostic tools could look.

The symptom was an agent doing the right thing and being marked wrong. Asked
about a memory leak on user-service, it called `get_service_logs`, found no
ERROR lines, reasoned that the service was probably waiting on a slow
dependency, and said so - while noting, correctly, that it had no ERROR-level
evidence for that conclusion. The scenario was unanswerable from evidence, so
the eval was measuring whether the agent could guess the archetype from
retrieval alone.

A scenario now carries its infrastructure state, as the specification asked:
before it runs, the archetype's log signatures and metric anomaly are written
for that service; afterwards they are removed. Inserted rows are tracked by id
rather than deleted by predicate, so cleanup cannot take the base corpus with
it.

The abstention scenarios are deliberately exempt: their whole point is that
there is nothing to find.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import psycopg

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "seeds"))
sys.path.insert(0, str(REPO_ROOT / "api"))

import archetypes as A  # noqa: E402
import catalog as C  # noqa: E402
from generate import BASELINES, NOW, ZERO_BASELINE_METRICS  # noqa: E402

from evals.scenario import Scenario  # noqa: E402

METRIC_INTERVAL_MIN = 5


@dataclass
class SeededState:
    """Row ids written for one scenario, so they can be removed exactly."""
    log_ids: list[int] = field(default_factory=list)
    metric_ids: list[int] = field(default_factory=list)
    deploy_ids: list[str] = field(default_factory=list)
    onset: datetime | None = None


def apply(conn: psycopg.Connection, scenario: Scenario, *, seed: int = 0) -> SeededState:
    """Write the evidence this scenario's failure would actually leave."""
    state = SeededState()
    if scenario.should_abstain or not scenario.expected_archetype:
        return state

    arch = A.BY_KEY.get(scenario.expected_archetype)
    service = scenario.true_cause_service or scenario.expected_service
    if arch is None or not service or service not in C.BY_NAME:
        return state

    r = random.Random(f"{scenario.id}:{seed}")
    onset = NOW - timedelta(hours=r.randint(2, 10), minutes=r.choice([0, 11, 23, 37]))
    state.onset = onset
    dep = ""
    svc = C.BY_NAME[service]
    if arch.needs_dep and svc.depends_on:
        dep = r.choice([t for t, _ in svc.depends_on])

    # ── Error signatures on the failing service ─────────────
    rows = []
    for _ in range(r.randint(40, 90)):
        ts = onset + timedelta(minutes=r.uniform(0, (NOW - onset).total_seconds() / 60))
        line = r.choice(arch.log_lines).format(service=service, dep=dep or "upstream")
        rows.append((service, ts, r.choices(["ERROR", "FATAL"], weights=[9, 1], k=1)[0],
                     line, f"evalfix-{r.randbytes(5).hex()}"))

    # ── The cascade, for scenarios that have one ────────────
    # Without this a cascading scenario gives the agent no reason to look
    # downstream: the reported service would show nothing at all.
    for victim in scenario.victim_services[:3]:
        if victim not in C.BY_NAME:
            continue
        for _ in range(r.randint(12, 30)):
            ts = onset + timedelta(minutes=r.uniform(3, (NOW - onset).total_seconds() / 60))
            rows.append((victim, ts, "ERROR",
                         f"context deadline exceeded calling {service}",
                         f"evalfix-{r.randbytes(5).hex()}"))

    with conn.cursor() as cur:
        for row in rows:
            cur.execute(
                "INSERT INTO log_entries (service, ts, level, message, trace_id) "
                "VALUES (%s,%s,%s,%s,%s) RETURNING id", row,
            )
            state.log_ids.append(cur.fetchone()[0])

        # ── The metric the archetype moves ──────────────────
        base = BASELINES.get(arch.metric, 1.0)
        points = int((NOW - onset).total_seconds() / 60 / METRIC_INTERVAL_MIN)
        for i in range(points):
            ts = onset + timedelta(minutes=i * METRIC_INTERVAL_MIN)
            ramp = min(1.0, (ts - onset).total_seconds() / 1800)
            if arch.metric in ZERO_BASELINE_METRICS:
                value = round(arch.metric_spike * ramp)
            else:
                value = base * (1 + (arch.metric_spike - 1) * ramp) * r.uniform(0.95, 1.05)
            cur.execute(
                "INSERT INTO metric_points (service, metric_name, ts, value) "
                "VALUES (%s,%s,%s,%s) RETURNING id",
                (service, arch.metric, ts, round(value, 4)),
            )
            state.metric_ids.append(cur.fetchone()[0])

        # ── A triggering deploy, where the archetype implies one ─
        if arch.key in ("bad_deploy_regression", "memory_leak_oom", "config_drift"):
            deploy_id = f"DEP-EVAL-{scenario.id}"
            cur.execute(
                "INSERT INTO deploys (id, service, version, previous_version, deployed_at, "
                "deployed_by, status, changelog) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (id) DO NOTHING",
                (deploy_id, service, "v3.12.0", "v3.11.4",
                 onset - timedelta(minutes=r.randint(4, 25)), "eval-harness", "succeeded",
                 "optimise response handling; add in-process cache for lookups"),
            )
            state.deploy_ids.append(deploy_id)

    return state


def clear(conn: psycopg.Connection, state: SeededState) -> None:
    """Remove exactly the rows `apply` wrote.

    By id, never by predicate. Deleting "all ERROR logs for this service since
    the onset" would take the base corpus with it, and the damage would only
    show up as a later scenario mysteriously having no evidence.
    """
    with conn.cursor() as cur:
        if state.log_ids:
            cur.execute("DELETE FROM log_entries WHERE id = ANY(%s)", (state.log_ids,))
        if state.metric_ids:
            cur.execute("DELETE FROM metric_points WHERE id = ANY(%s)", (state.metric_ids,))
        if state.deploy_ids:
            cur.execute("DELETE FROM deploys WHERE id = ANY(%s)", (state.deploy_ids,))
