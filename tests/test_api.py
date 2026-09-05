"""API tests.

Small surface for now - the investigation and approval endpoints arrive in
Phase 4. What is tested here is the liveness/readiness distinction, which is
easy to get wrong in a way that causes real outages.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))

from incidentiq.api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_liveness_does_not_depend_on_the_database(client, monkeypatch):
    """A service that is itself healthy but whose database is down must not be
    reported unhealthy - restarting it does not fix the database and drops
    in-flight work. This is the same distinction that makes `restart_service`
    the wrong remediation for a downstream failure."""
    import incidentiq.api.main as main

    def explode():
        raise RuntimeError("database is down")

    monkeypatch.setattr(main, "_connect", explode)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_readiness_fails_when_the_database_is_unreachable(client, monkeypatch):
    import incidentiq.api.main as main

    def explode():
        raise RuntimeError("database is down")

    monkeypatch.setattr(main, "_connect", explode)
    r = client.get("/readyz")
    assert r.status_code == 503
    assert "unreachable" in str(r.json()["detail"]["database"])


def test_readiness_reports_the_index_size(client):
    """An empty chunk table means retrieval silently returns nothing, so it is
    a readiness failure rather than a healthy-but-empty state."""
    r = client.get("/readyz")
    if r.status_code == 503:
        pytest.skip("database not reachable")
    assert r.json()["indexed_chunks"] > 0


def test_tools_endpoint_separates_read_from_write(client):
    r = client.get("/tools")
    assert r.status_code == 200
    body = r.json()
    assert len(body["read_only"]) == 5
    assert len(body["approval_required"]) == 3
    read_names = {t["name"] for t in body["read_only"]}
    write_names = {t["name"] for t in body["approval_required"]}
    assert not read_names & write_names, "a tool appears in both lists"
    assert write_names == {"restart_service", "scale_service", "rollback_deploy"}


def test_stats_reports_the_corpus(client):
    r = client.get("/stats")
    if r.status_code == 503:
        pytest.skip("database not reachable")
    body = r.json()
    assert body["services"] == 33
    assert body["incidents"] == 500
