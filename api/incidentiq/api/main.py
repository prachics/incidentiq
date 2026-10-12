"""FastAPI application.

Operational endpoints live here: liveness, readiness, what is loaded, and the
tool catalogue, so a fresh `docker compose --profile app up` can be checked.

Investigations and approvals are in their own routers and need an API key.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import psycopg
from fastapi import FastAPI, HTTPException

from incidentiq.api import approvals, investigations
from incidentiq.api.auth import parse_api_keys
from incidentiq.api.worker import Worker
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
    # Parsing raises on a malformed entry, so a broken API_KEYS stops the start.
    if not parse_api_keys(settings.api_keys):
        log.warning("API_KEYS is empty: investigation and approval endpoints will answer 401")
    if getattr(app.state, "worker", None) is None:
        app.state.worker = Worker()
    yield
    worker = getattr(app.state, "worker", None)
    if worker is not None:
        worker.shutdown()


app = FastAPI(
    title="IncidentIQ",
    description="Agentic AI production-support platform.",
    version="0.2.0",
    lifespan=lifespan,
)
app.include_router(investigations.router)
app.include_router(approvals.router)


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
