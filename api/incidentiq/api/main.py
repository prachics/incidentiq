"""FastAPI application.

Deliberately small for now. The investigation, approval, and streaming
endpoints arrive in Phase 4 alongside the approval flow they exist to serve;
building them ahead of that would mean guessing at the shapes the frontend
needs.

What is here makes `docker compose --profile app up` a true statement rather
than an aspirational one: the container starts, reports whether its
dependencies are actually reachable, and answers what is loaded.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import psycopg
from fastapi import FastAPI, HTTPException

from incidentiq.config import get_settings

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Fail at startup rather than on the first request. A container that starts
    # healthy and then 500s on every call is harder to diagnose than one that
    # refuses to start.
    try:
        settings.require_llm_credentials()
    except RuntimeError as exc:
        log.warning("LLM not configured: %s", exc)
    yield


app = FastAPI(
    title="IncidentIQ",
    description="Agentic AI production-support platform.",
    version="0.1.0",
    lifespan=lifespan,
)


def _connect() -> psycopg.Connection:
    return psycopg.connect(get_settings().database_url, autocommit=True, connect_timeout=3)


@app.get("/healthz")
def liveness() -> dict[str, str]:
    """Liveness only. Deliberately does not check dependencies.

    A service that is itself healthy but whose database is down should not be
    restarted - restarting does not fix the database and drops in-flight work.
    This is the same distinction the seeded service docs describe, and the
    reason `restart_service` is the wrong remediation for a downstream failure.
    """
    return {"status": "ok"}


@app.get("/readyz")
def readiness() -> dict[str, Any]:
    """Readiness. Checks the things a request would actually need."""
    settings = get_settings()
    checks: dict[str, Any] = {}

    try:
        with _connect() as conn:
            conn.execute("SELECT 1")
            checks["database"] = "ok"
            n = conn.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")
            checks["indexed_chunks"] = n.fetchone()[0]
    except Exception as exc:  # noqa: BLE001
        checks["database"] = f"unreachable: {exc}"

    checks["llm_provider"] = settings.llm_provider
    checks["embedding_model"] = settings.embedding_model

    ready = checks.get("database") == "ok" and checks.get("indexed_chunks", 0) > 0
    if not ready:
        raise HTTPException(status_code=503, detail=checks)
    return {"status": "ready", **checks}


@app.get("/stats")
def stats() -> dict[str, Any]:
    """What is loaded. Useful for confirming a fresh `docker compose up` worked."""
    tables = ["services", "incidents", "runbooks", "service_docs", "chunks",
              "investigations", "approvals", "actions"]
    try:
        with _connect() as conn:
            return {
                t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables
            }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/tools")
def tools() -> dict[str, Any]:
    """The tool catalogue, with the read/write split made explicit.

    Exposed because "which actions require approval" is a question an operator
    should be able to answer without reading the source.
    """
    from incidentiq.tools import registry

    return {
        "read_only": [
            {"name": t.name, "description": t.description} for t in registry.read_only()
        ],
        "approval_required": [
            {"name": t.name, "description": t.description} for t in registry.write()
        ],
    }
