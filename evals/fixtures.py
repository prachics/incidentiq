"""Seed the mock infrastructure for one scenario, then remove it again.

Before a scenario runs, its archetype's log signatures and metric anomaly are
written for the service at fault, so the failure leaves evidence the diagnostic
tools can find (without it most scenarios could only be guessed from retrieval).
Inserted rows are tracked by id and scaled corpus readings keep their originals,
so cleanup restores the corpus exactly. Abstention scenarios get nothing: there
must be nothing to find. One evaluation holds the database at a time.
"""

from __future__ import annotations

import random
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import psycopg

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "seeds"))
sys.path.insert(0, str(REPO_ROOT / "api"))

import archetypes as A  # noqa: E402
import catalog as C  # noqa: E402
from generate import (  # noqa: E402
    BASELINES,
    LOG_WINDOW_HOURS,
    NOW,
    ZERO_BASELINE_METRICS,
    diurnal,
)

from evals.scenario import Scenario  # noqa: E402

METRIC_INTERVAL_MIN = 5


# One evaluation at a time per database: fixtures rewrite shared corpus rows and
# restore them, and a purge takes every other run's rows, so two at once would
# save each other's anomalies as originals. The test suite holds it too, since
# its database tests empty tables an evaluation is reading.
EVAL_LOCK = 0x49515F4556414C


@contextmanager
def exclusive(url: str, key: int = EVAL_LOCK) -> Iterator[None]:
    """Hold the database for one evaluation (or one test session) until the block
    ends; refuse at once if something else holds it."""
    conn = psycopg.connect(url, autocommit=True)
    try:
        if not conn.execute("SELECT pg_try_advisory_lock(%s)", (key,)).fetchone()[0]:
            raise RuntimeError("another evaluation, or the test suite, is using this database")
        yield
    finally:
        conn.close()                         # the lock ends with the session

@dataclass
class SeededState:
    """Row ids written for one scenario, so they can be removed exactly. Mirrored
    into `eval_fixture_rows`, so cleanup survives a killed process: leaked evidence
    would become background noise for every later scenario."""
    run_id: str = ""
    scenario_id: str = ""
    log_ids: list[int] = field(default_factory=list)
    metric_ids: list[int] = field(default_factory=list)
    deploy_ids: list[str] = field(default_factory=list)
    # Corpus readings scaled for the incident, with their original values.
    scaled_metrics: list[tuple[int, float]] = field(default_factory=list)
    onset: datetime | None = None


def apply(conn: psycopg.Connection, scenario: Scenario, *, seed: int = 0,
          run_id: str = "adhoc") -> SeededState:
    """Write the evidence this scenario's failure would actually leave. The changes
    and the record of them are one transaction: a run stopped part way leaves the
    corpus as it was, never scaled readings without the originals to restore."""
    with conn.transaction():
        return _apply(conn, scenario, seed, run_id)


def _apply(conn: psycopg.Connection, scenario: Scenario, seed: int, run_id: str) -> SeededState:
    state = SeededState(run_id=run_id, scenario_id=scenario.id)
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
        # The corpus's own readings are scaled, not joined by a second series:
        # hourly averages of planted and normal points at the same timestamps
        # hid a full disk entirely (the hour's mean fell while its max was 90%).
        base = BASELINES.get(arch.metric, 1.0)
        cur.execute(
            "SELECT id, ts, value FROM metric_points WHERE service = %s AND metric_name = %s "
            "AND ts >= %s AND ts <= %s ORDER BY ts",
            (service, arch.metric, onset, NOW),
        )
        readings = cur.fetchall()
        if readings:
            for point_id, ts, original in readings:
                cur.execute("UPDATE metric_points SET value = %s WHERE id = %s",
                            (_anomalous(arch, original, ts, onset), point_id))
                state.scaled_metrics.append((point_id, original))
        else:
            # The corpus does not collect this metric for the service. One that
            # reports it during the incident reported it before, too.
            start = NOW - timedelta(hours=LOG_WINDOW_HOURS)
            for i in range(LOG_WINDOW_HOURS * 60 // METRIC_INTERVAL_MIN):
                ts = start + timedelta(minutes=i * METRIC_INTERVAL_MIN)
                normal = base * diurnal(ts) * r.uniform(0.9, 1.1)
                value = _anomalous(arch, normal, ts, onset) if ts >= onset else round(normal, 4)
                cur.execute(
                    "INSERT INTO metric_points (service, metric_name, ts, value) "
                    "VALUES (%s,%s,%s,%s) RETURNING id",
                    (service, arch.metric, ts, value),
                )
                state.metric_ids.append(cur.fetchone()[0])

        # ── A triggering deploy, where the archetype implies one ─
        if arch.key in ("bad_deploy_regression", "memory_leak_oom", "config_drift"):
            # Both the bad deploy and the version it replaced must exist, or a
            # rollback to the previous version is refused as unknown.
            bad_at = onset - timedelta(minutes=r.randint(4, 25))
            prev_id = f"DEP-EVAL-{scenario.id}-prev"
            cur.execute(
                "INSERT INTO deploys (id, service, version, previous_version, deployed_at, "
                "deployed_by, status, changelog) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (id) DO NOTHING",
                (prev_id, service, "v3.11.4", "v3.11.3",
                 bad_at - timedelta(days=r.randint(2, 9)), "eval-harness", "succeeded",
                 "routine dependency bumps"),
            )
            state.deploy_ids.append(prev_id)

            deploy_id = f"DEP-EVAL-{scenario.id}"
            cur.execute(
                "INSERT INTO deploys (id, service, version, previous_version, deployed_at, "
                "deployed_by, status, changelog) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (id) DO NOTHING",
                (deploy_id, service, "v3.12.0", "v3.11.4", bad_at, "eval-harness",
                 "succeeded",
                 "optimise response handling; add in-process cache for lookups"),
            )
            state.deploy_ids.append(deploy_id)

    _record(conn, state)
    return state


def _anomalous(arch: A.Archetype, normal: float, ts: datetime, onset: datetime) -> float:
    """The reading at `ts` once the failure has ramped in over half an hour."""
    ramp = max(0.0, min(1.0, (ts - onset).total_seconds() / 1800))
    if arch.metric in ZERO_BASELINE_METRICS:
        return float(round(arch.metric_spike * ramp))
    return round(normal * (1 + (arch.metric_spike - 1) * ramp), 4)


def _record(conn: psycopg.Connection, state: SeededState) -> None:
    """Write the row ids to the database so cleanup survives an unclean exit."""
    rows = (
        [("log_entries", str(i)) for i in state.log_ids]
        + [("metric_points", str(i)) for i in state.metric_ids]
        + [("deploys", i) for i in state.deploy_ids]
    )
    if not rows and not state.scaled_metrics:
        return
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO eval_fixture_rows (run_id, scenario_id, target_table, target_id) "
            "VALUES (%s,%s,%s,%s)",
            [(state.run_id, state.scenario_id, t, i) for t, i in rows],
        )
        cur.executemany(
            "INSERT INTO eval_fixture_rows (run_id, scenario_id, target_table, target_id, "
            "original_value) VALUES (%s,%s,'metric_points',%s,%s)",
            [(state.run_id, state.scenario_id, str(i), v) for i, v in state.scaled_metrics],
        )


def purge_orphans(conn: psycopg.Connection, *,
                  except_run_id: str | None = None,
                  only_run_id: str | None = None) -> dict[str, int]:
    """Remove fixture rows left behind by a run that is not this one, at harness
    startup: a killed run's planted evidence would otherwise be silent noise in
    every later one. A run purges only what is not its own (`except_run_id`)."""
    if except_run_id and only_run_id:
        raise ValueError("pass except_run_id or only_run_id, not both")

    counts: dict[str, int] = {}
    if only_run_id:
        scope, scope_arg = " AND run_id = %s", only_run_id
    elif except_run_id:
        scope, scope_arg = " AND run_id <> %s", except_run_id
    else:
        scope, scope_arg = "", None

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE metric_points m SET value = f.original_value FROM eval_fixture_rows f "
            "WHERE f.target_table = 'metric_points' AND f.original_value IS NOT NULL "
            f"AND m.id = f.target_id::bigint{scope}",
            [scope_arg] if scope_arg else [],
        )
        counts["metric_points_restored"] = cur.rowcount
        for table, cast in (("log_entries", "::bigint"), ("metric_points", "::bigint"),
                            ("deploys", "")):
            params = [table] + ([scope_arg] if scope_arg else [])
            cur.execute(
                f"DELETE FROM {table} WHERE id IN ("
                f"  SELECT target_id{cast} FROM eval_fixture_rows "
                f"  WHERE target_table = %s AND original_value IS NULL{scope})",
                params,
            )
            counts[table] = cur.rowcount
        if scope_arg:
            cur.execute(f"DELETE FROM eval_fixture_rows WHERE TRUE{scope}", (scope_arg,))
        else:
            cur.execute("DELETE FROM eval_fixture_rows")
    return {k: v for k, v in counts.items() if v}



def clear(conn: psycopg.Connection, state: SeededState) -> None:
    """Remove exactly the rows `apply` wrote.

    By id, never by predicate: deleting "all ERROR logs for this service since the
    onset" would take the base corpus with it. All of it or nothing: if it fails,
    the tracking rows stay, and the next run's purge restores the corpus from them.
    """
    with conn.transaction(), conn.cursor() as cur:
        if state.log_ids:
            cur.execute("DELETE FROM log_entries WHERE id = ANY(%s)", (state.log_ids,))
        if state.metric_ids:
            cur.execute("DELETE FROM metric_points WHERE id = ANY(%s)", (state.metric_ids,))
        for point_id, original in state.scaled_metrics:
            cur.execute("UPDATE metric_points SET value = %s WHERE id = %s",
                        (original, point_id))
        if state.deploy_ids:
            cur.execute("DELETE FROM deploys WHERE id = ANY(%s)", (state.deploy_ids,))
        cur.execute(
            "DELETE FROM eval_fixture_rows WHERE run_id = %s AND scenario_id = %s",
            (state.run_id, state.scenario_id),
        )
