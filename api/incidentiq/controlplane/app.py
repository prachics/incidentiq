"""The mock control plane as its own HTTP service.

`docker compose --profile app up` runs it next to the API, and setting
CONTROL_PLANE_URL points the API at it, so an approved action crosses a real
network boundary. Without that variable the API uses the same logic in-process.

A lost response cannot be faked by dropping the connection from inside FastAPI,
so here it is a 504 after the operation was applied. To the caller that is the
same situation: it does not know whether the request took effect.
"""

from __future__ import annotations

from typing import Any

import psycopg
from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from pydantic import BaseModel, Field

from incidentiq.config import get_settings
from incidentiq.controlplane.service import ControlPlane, FaultPlan, LostResponse
from incidentiq.telemetry import setup_tracing

app = FastAPI(title="IncidentIQ mock control plane", version="0.1.0")

_settings = get_settings()
setup_tracing("incidentiq-control-plane", _settings.otel_exporter_otlp_endpoint)
# Extracts `traceparent`, so each request continues the caller's trace.
FastAPIInstrumentor.instrument_app(app, excluded_urls=r"://[^/]+/healthz/?(?:\?.*)?$")
# One plan for the process, so request counts per key survive across requests.
_faults = FaultPlan.seeded(_settings.control_plane_fault_rate,
                           _settings.failure_injection_seed)


class OperationRequest(BaseModel):
    operation: str
    params: dict[str, Any] = Field(default_factory=dict)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/operations")
def submit(body: OperationRequest,
           idempotency_key: str | None = Header(default=None)) -> JSONResponse:
    with psycopg.connect(_settings.database_url, autocommit=True) as conn:
        try:
            reply = ControlPlane(conn, _faults).submit(idempotency_key, body.operation,
                                                       body.params)
        except LostResponse:
            return JSONResponse(status_code=504,
                                content={"error": "gateway timeout after apply"})
    headers = {"Idempotent-Replay": "true"} if reply.replayed else None
    return JSONResponse(status_code=reply.status, content=reply.body, headers=headers)
