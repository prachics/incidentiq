"""The internal-consistency guarantees the README claims.

If these pass, the claim "an incident that blames a dependency blames a real
dependency" is verified rather than asserted.
"""

import sys
from pathlib import Path

import psycopg
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT / "seeds"))

from incidentiq.config import get_settings  # noqa: E402


@pytest.fixture(scope="module")
def conn():
    try:
        c = psycopg.connect(get_settings().database_url, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable")
    if c.execute("SELECT count(*) FROM incidents").fetchone()[0] == 0:
        pytest.skip("no seed data - run: python seeds/generate.py")
    yield c
    c.close()


def test_no_orphan_incidents(conn):
    n = conn.execute(
        "SELECT count(*) FROM incidents i "
        "LEFT JOIN services s ON s.name=i.service WHERE s.name IS NULL"
    ).fetchone()[0]
    assert n == 0


def test_root_causes_only_reference_real_dependencies(conn):
    """The headline consistency claim. An incident on service X whose root cause
    names service Y must have Y as an actual dependency of X."""
    real, fake = conn.execute("""
        WITH mentioned AS (
          SELECT i.id, i.service, d.to_service
          FROM incidents i
          JOIN services s2 ON i.root_cause LIKE '%' || s2.name || '%' AND s2.name <> i.service
          LEFT JOIN service_dependencies d
            ON d.from_service = i.service AND d.to_service = s2.name
        )
        SELECT count(*) FILTER (WHERE to_service IS NOT NULL),
               count(*) FILTER (WHERE to_service IS NULL) FROM mentioned
    """).fetchone()
    assert fake == 0, f"{fake} incidents blame a service that is not a dependency"
    assert real > 0, "no incidents reference a dependency at all - templates broken?"


def test_every_live_situation_has_error_logs(conn):
    """A live situation with no log signal would be unsolvable: the agent's
    tools would return nothing to reason about."""
    import json
    manifest = json.loads((ROOT / "seeds" / "live_situations.json").read_text())
    for sit in manifest["live_situations"]:
        n = conn.execute(
            "SELECT count(*) FROM log_entries WHERE service=%s AND level IN ('ERROR','FATAL') "
            "AND ts >= %s",
            (sit["service"], sit["onset"]),
        ).fetchone()[0]
        assert n > 0, f"{sit['id']} ({sit['service']}) has no error logs after onset"


def test_every_live_situation_has_a_metric_anomaly(conn):
    """The archetype's metric must actually move after onset, or the metric tool
    tells the agent nothing."""
    import json
    manifest = json.loads((ROOT / "seeds" / "live_situations.json").read_text())
    for sit in manifest["live_situations"]:
        before, after = conn.execute(
            """
            SELECT avg(value) FILTER (WHERE ts <  %(onset)s),
                   avg(value) FILTER (WHERE ts >= %(onset)s)
            FROM metric_points WHERE service=%(svc)s AND metric_name=%(m)s
            """,
            {"onset": sit["onset"], "svc": sit["service"], "m": sit["metric"]},
        ).fetchone()
        assert before is not None and after is not None, f"{sit['id']}: missing metric data"
        moved = abs(after - before) / max(before, 1e-6)
        assert moved > 0.15, (
            f"{sit['id']} ({sit['service']}/{sit['metric']}): metric barely moved "
            f"({before:.3f} -> {after:.3f}); the anomaly is not detectable"
        )


def test_labelled_retrieval_set_is_usable(conn):
    """Most situations must have relevant documents, or Recall@5 is measured on
    almost nothing. Zero-relevance situations are legitimate but must be the
    minority and must be explicitly flagged."""
    import json
    manifest = json.loads((ROOT / "seeds" / "live_situations.json").read_text())
    labelled = manifest["labelled_retrieval"]
    zero = [r for r in labelled if r["relevant_count"] == 0]
    for r in zero:
        assert r["expect_no_useful_retrieval"] is True, (
            f"{r['situation_id']} has no relevant docs but is not flagged as an "
            "abstention case"
        )
    assert len(zero) < len(labelled) / 2, "over half the situations have no ground truth"


def test_relevant_incident_ids_all_exist(conn):
    import json
    manifest = json.loads((ROOT / "seeds" / "live_situations.json").read_text())
    all_ids = {r[0] for r in conn.execute("SELECT id FROM incidents").fetchall()}
    for r in manifest["labelled_retrieval"]:
        missing = set(r["relevant_incident_ids"]) - all_ids
        assert not missing, f"{r['situation_id']} references unknown incidents: {missing}"


def test_generation_is_deterministic():
    """Two generations from the same seed must produce identical corpora, or
    eval numbers are not comparable across runs."""
    import importlib

    import generate

    importlib.reload(generate)
    r1 = generate.rng()
    inc1 = generate.gen_incidents(r1)
    r2 = generate.rng()
    inc2 = generate.gen_incidents(r2)
    assert [i["id"] for i in inc1] == [i["id"] for i in inc2]
    assert [i["root_cause"] for i in inc1] == [i["root_cause"] for i in inc2]


class TestArchetypeAnomalyGeneration:
    """Every archetype must produce a detectable metric anomaly if it fires.

    The live-situation tests only cover archetypes the RNG happened to select -
    currently 9 of 16. The other 7, including both zero-baseline metrics, were
    never exercised end to end. That is exactly where a silent defect hides: the
    multiplicative-spike bug survived Phase 1 because `unassigned_shards` had no
    live situation, and would have surfaced only when a scenario finally used it.

    These tests drive the generator directly, one archetype at a time, so
    coverage does not depend on chance.
    """

    @staticmethod
    def _simulate(arch, service_name):
        import datetime
        import random

        import archetypes as A  # noqa: F401
        import generate as G

        onset = G.NOW - datetime.timedelta(hours=6)
        situation = [{
            "id": "SIM", "service": service_name, "archetype": arch.key,
            "dep": "", "onset": onset.isoformat(), "metric": arch.metric,
            "cascade_visible_in": [],
        }]
        rows = G.gen_metrics(random.Random(7), situation)
        before = [v for (svc, m, ts, v) in rows
                  if svc == service_name and m == arch.metric and ts < onset]
        after = [v for (svc, m, ts, v) in rows
                 if svc == service_name and m == arch.metric and ts >= onset]
        return before, after

    @pytest.mark.parametrize(
        "arch",
        __import__("archetypes").ARCHETYPES,
        ids=lambda a: a.key,
    )
    def test_archetype_produces_detectable_anomaly(self, arch):
        import catalog as C

        candidates = [
            s for s in C.SERVICES
            if s.kind in arch.applies_to
            and (not arch.languages or s.language in arch.languages)
        ]
        assert candidates, f"{arch.key} applies to no service"

        before, after = self._simulate(arch, candidates[0].name)
        assert before and after, f"{arch.key}: no metric points generated for {arch.metric}"

        mean_before = sum(before) / len(before)
        # Not every anomaly is an increase: cache_hit_rate FALLS from 0.94 to
        # 0.33 during a stampede. Taking max() would pick the value nearest
        # baseline and report no movement. The signal is the largest deviation
        # in whichever direction the metric actually moves.
        peak_after = max(after, key=lambda v: abs(v - mean_before))
        delta = abs(peak_after - mean_before)
        direction = "rises" if peak_after > mean_before else "falls"

        assert delta > 0, (
            f"{arch.key} on {candidates[0].name}: {arch.metric} does not move at all "
            f"(flat at {mean_before:.3f}). A multiplicative spike on a zero baseline "
            f"is the usual cause."
        )
        # Relative movement, guarding against a change too small for a tool to
        # surface as a signal.
        relative = delta / max(abs(mean_before), 1e-6)
        assert relative > 0.15 or delta >= 1.0, (
            f"{arch.key}: {arch.metric} only {direction} {mean_before:.3f} -> "
            f"{peak_after:.3f}; too small for a diagnostic tool to surface"
        )

    def test_zero_baseline_metrics_are_declared(self):
        """Any archetype whose metric has a zero baseline must be listed in
        ZERO_BASELINE_METRICS, or its spike silently multiplies zero by zero."""
        import archetypes as A
        import generate as G

        for arch in A.ARCHETYPES:
            baseline = G.BASELINES.get(arch.metric)
            if baseline == 0.0:
                assert arch.metric in G.ZERO_BASELINE_METRICS, (
                    f"{arch.metric} has a zero baseline but is not in "
                    "ZERO_BASELINE_METRICS - its anomaly will be 0 * spike = 0"
                )

    def test_every_archetype_metric_has_a_baseline(self):
        """A metric with no BASELINES entry silently defaults to 1.0, which
        makes the generated series meaningless rather than absent."""
        import archetypes as A
        import generate as G

        missing = [a.metric for a in A.ARCHETYPES if a.metric not in G.BASELINES]
        assert not missing, f"archetype metrics with no baseline defined: {missing}"
