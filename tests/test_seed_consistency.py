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
