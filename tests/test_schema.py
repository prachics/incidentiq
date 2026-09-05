"""Tests against a live database.

Skipped automatically when Postgres is not reachable, so the suite still runs
in an environment without Docker.
"""

import sys
from pathlib import Path

import psycopg
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "seeds"))

from incidentiq.config import get_settings  # noqa: E402


@pytest.fixture(scope="module")
def conn():
    try:
        c = psycopg.connect(get_settings().database_url, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable - run: docker compose up -d postgres")
    yield c
    c.close()


def test_pgvector_installed(conn):
    row = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()
    assert row, "pgvector extension is not installed"


def test_chunks_has_hnsw_and_gin_indexes(conn):
    defs = [
        r[0] for r in conn.execute(
            "SELECT indexdef FROM pg_indexes WHERE tablename='chunks'"
        ).fetchall()
    ]
    assert any("hnsw" in d for d in defs), "no HNSW index on chunks.embedding"
    assert any("gin" in d for d in defs), "no GIN index on chunks.content_tsv"


def test_tsvector_column_is_generated(conn):
    row = conn.execute(
        "SELECT is_generated FROM information_schema.columns "
        "WHERE table_name='chunks' AND column_name='content_tsv'"
    ).fetchone()
    assert row[0] == "ALWAYS"


def test_audit_log_rejects_update(conn):
    conn.execute(
        "INSERT INTO audit_log (investigation_id, event_type, actor, payload) "
        "VALUES ('PYTEST','test','agent','{}')"
    )
    with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
        conn.execute("UPDATE audit_log SET actor='x' WHERE investigation_id='PYTEST'")


def test_audit_log_rejects_delete(conn):
    with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
        conn.execute("DELETE FROM audit_log WHERE investigation_id='PYTEST'")


def test_severity_check_constraint(conn):
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(
            "INSERT INTO incidents (id,title,service,severity,occurred_at,"
            "symptoms,root_cause,resolution) VALUES "
            "('BAD','t','api-gateway','SEV9',now(),'s','r','x')"
        )


def test_dependency_self_reference_rejected(conn):
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(
            "INSERT INTO service_dependencies (from_service,to_service,kind) "
            "VALUES ('api-gateway','api-gateway','sync')"
        )
