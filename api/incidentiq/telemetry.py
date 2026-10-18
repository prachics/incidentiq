"""Tracing (OpenTelemetry) and metrics (Prometheus).

A run started by one request is one trace: the request, the worker job, every
graph node, every tool attempt and LLM call, and the request to the control
plane, which continues the trace from the W3C `traceparent` header. Later runs
of the same investigation (after a decision or a resume) are traces of their own
requests, and their job spans link to the investigation's first job. Spans are
exported over OTLP only when OTEL_EXPORTER_OTLP_ENDPOINT is set.

Metrics are Prometheus counters and histograms served at /metrics, recorded
whether or not tracing is exported.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from urllib.parse import unquote

from opentelemetry import propagate, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import Link, Span, SpanKind, Status, StatusCode
from prometheus_client import Counter, Histogram

log = logging.getLogger(__name__)

tracer = trace.get_tracer("incidentiq")

_configured = False


def parse_otlp_headers(value: str) -> dict[str, str]:
    """OTEL_EXPORTER_OTLP_HEADERS format: comma-separated key=value pairs, values
    URL-encoded. Split on the first '=' only: base64 credentials end in '='."""
    headers = {}
    for pair in value.split(","):
        if not pair.strip():
            continue
        key, sep, raw = pair.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"OTLP header {pair.strip()!r} is not key=value")
        headers[key.strip()] = unquote(raw.strip())
    return headers


def otlp_exporter(endpoint: str, headers: str = "") -> SpanExporter:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces",
                            headers=parse_otlp_headers(headers) or None)


def setup_tracing(service_name: str, endpoint: str = "",
                  exporter: SpanExporter | None = None, headers: str = "") -> bool:
    """Install the global tracer provider once per process. Returns whether
    spans will be exported. A second call is a no-op: OpenTelemetry allows one
    global provider, and both the API and the worker live in one process."""
    global _configured
    if _configured:
        return True
    if exporter is None:
        if not endpoint:
            return False
        exporter = otlp_exporter(endpoint, headers)
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _configured = True
    log.info("exporting traces for %s", service_name)
    return True


@contextmanager
def span(name: str, *, kind: SpanKind = SpanKind.INTERNAL,
         links: list[Link] | None = None, **attributes: Any) -> Iterator[Span]:
    """A span with attributes set from keyword arguments (None values skipped).
    An exception marks the span as an error and is re-raised."""
    with tracer.start_as_current_span(name, kind=kind, links=links, record_exception=True,
                                      set_status_on_exception=True) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        yield current


def fail(current: Span, reason: str, error_type: str = "handled") -> None:
    """Mark a span failed without an exception: a failure the code handled and
    reported, which a trace view should still show in red."""
    current.set_status(Status(StatusCode.ERROR, reason[:500]))
    current.set_attribute("error.type", error_type)


def traceparent(current: Span) -> str:
    """W3C traceparent of a span, for storing and linking to later."""
    ctx = current.get_span_context()
    return f"00-{ctx.trace_id:032x}-{ctx.span_id:016x}-{int(ctx.trace_flags):02x}"


def link_to(value: str | None) -> list[Link]:
    """A span link to a stored traceparent; empty if there is none."""
    if not value:
        return []
    remote = trace.get_current_span(propagate.extract({"traceparent": value}))
    ctx = remote.get_span_context()
    return [Link(ctx)] if ctx.is_valid else []


def inject_headers(headers: dict[str, str]) -> dict[str, str]:
    """Add `traceparent` (and `tracestate`) for the current span to outgoing headers."""
    propagate.inject(headers)
    return headers


# ── Metrics ─────────────────────────────────────────────────
# Started counts new investigations; settled counts every run that stops,
# including the runs after a decision or a resume. One investigation that waits
# for approval settles twice (awaiting_approval, then complete), so the two are
# not a started/finished pair.
INVESTIGATIONS_STARTED = Counter(
    "incidentiq_investigations_started_total", "New investigations started.")
INVESTIGATIONS_SETTLED = Counter(
    "incidentiq_investigations_settled_total",
    "Runs that stopped (first runs and resumes), by the status they stopped in.", ["status"])
NODE_SECONDS = Histogram(
    "incidentiq_node_duration_seconds", "Time spent in each graph node, by outcome.",
    ["node", "outcome"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120))
TOOL_CALLS = Counter(
    "incidentiq_tool_calls_total", "Tool attempts, by tool and status.", ["tool", "status"])
LLM_SECONDS = Histogram(
    "incidentiq_llm_request_duration_seconds", "LLM request latency, by outcome.",
    ["model", "node", "outcome"],
    buckets=(0.5, 1, 2.5, 5, 10, 20, 40, 80, 160))
LLM_TOKENS = Counter(
    "incidentiq_llm_tokens_total", "LLM tokens, by direction.", ["model", "direction"])
ACTIONS = Counter(
    "incidentiq_actions_total",
    "Executions of approved actions, by tool and outcome "
    "(executed, already_executed, failed, invalid_arguments).", ["tool", "outcome"])
ACTION_ATTEMPTS = Histogram(
    "incidentiq_action_attempts", "Control-plane requests it took to settle an action.",
    ["tool"], buckets=(1, 2, 3, 4, 6, 8))
DECISIONS = Counter(
    "incidentiq_approval_decisions_total", "Approval decisions, by decision.", ["decision"])
