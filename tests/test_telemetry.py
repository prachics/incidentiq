"""Tracing and metrics.

A run triggered by one request is one trace: the request, the worker job, every
node in order, and every tool attempt and LLM call under its node. Later runs of
the same investigation (after a decision or a resume) are traces of their own
requests and link to the first. The control plane continues a trace from the
`traceparent` header. Failures are visible as error spans and as measurements.
"""

import json
import socket
import sys
import threading
import uuid
from pathlib import Path

import httpx
import psycopg
import pytest
from conftest import SPANS, database_missing
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.trace import SpanKind, StatusCode
from prometheus_client import REGISTRY

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))

from incidentiq import telemetry  # noqa: E402
from incidentiq.agent.graph import InvestigationRunner  # noqa: E402
from incidentiq.api.main import app  # noqa: E402
from incidentiq.api.worker import Worker  # noqa: E402
from incidentiq.config import get_settings  # noqa: E402
from incidentiq.controlplane import ControlPlaneClient, ControlPlaneError  # noqa: E402
from incidentiq.llm import LLMResponse, StubProvider  # noqa: E402
from test_approval_api import APPROVER, PROPOSED, VIEWER, scripted_run  # noqa: E402

EXPORTER = SPANS   # installed by conftest before the app was imported
RUN_NODES = ["intake", "retrieve", "plan", "execute_tool", "retrieve_again", "reflect",
             "propose", "await_approval"]


def _reset():
    """Flush first: spans that ended earlier may still be in the batch queue,
    and would otherwise arrive after the clear."""
    trace.get_tracer_provider().force_flush()
    EXPORTER.clear()


@pytest.fixture
def spans():
    _reset()

    def finished():
        trace.get_tracer_provider().force_flush()
        return EXPORTER.get_finished_spans()

    return finished


@pytest.fixture(scope="module")
def conn():
    try:
        c = psycopg.connect(get_settings().database_url, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        database_missing("Postgres not reachable")
    yield c
    c.close()


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setattr(get_settings(), "api_keys",
                        "test-viewer-key:vera:viewer,test-approver-key:alice:approver")


def _named(found, prefix, kind=None):
    return [s for s in found if s.name.startswith(prefix) and (kind is None or s.kind == kind)]


def _sample(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


def _start(client, worker):
    iid = f"IQ-OTEL-{uuid.uuid4().hex[:8]}"
    client.post("/investigations", headers=VIEWER,
                json={"query": "checkout-service is timing out under load", "incident_id": iid})
    worker.drain(120)
    return iid


# ── Traces ──────────────────────────────────────────────────
def test_a_run_is_one_trace_with_every_node_in_order(conn, keys, spans):
    stub = StubProvider(script=scripted_run())
    worker = Worker(lambda: InvestigationRunner(llm=stub))
    app.state.worker = worker
    try:
        _start(TestClient(app), worker)
    finally:
        app.state.worker = None
    found = spans()

    request = _named(found, "POST /investigations", SpanKind.SERVER)[0]
    job = _named(found, "investigation job")[0]
    nodes = sorted(_named(found, "node "), key=lambda s: s.start_time)
    assert [s.name.removeprefix("node ") for s in nodes] == RUN_NODES
    assert {s.context.trace_id for s in found} == {request.context.trace_id}
    assert job.parent.span_id == request.context.span_id
    assert all(s.parent.span_id == job.context.span_id for s in nodes)

    node_ids = {s.context.span_id: s.name for s in nodes}
    tools, chats = _named(found, "tool "), _named(found, "chat ")
    assert [node_ids[t.parent.span_id] for t in tools] == ["node execute_tool"]
    assert sorted(node_ids[c.parent.span_id] for c in chats) == \
        ["node intake", "node plan", "node propose", "node reflect"]
    assert all(c.attributes["incidentiq.node"] == node_ids[c.parent.span_id].removeprefix("node ")
               for c in chats)
    for c in chats:
        assert c.attributes["gen_ai.operation.name"] == "chat"
        assert c.attributes["gen_ai.request.model"] == "stub"
        assert c.attributes["gen_ai.usage.input_tokens"] == 10
        assert c.attributes["gen_ai.usage.output_tokens"] == 10
    assert tools[0].attributes["incidentiq.tool.attempt"] == 1
    assert tools[0].attributes["incidentiq.tool.status"] == "success"


def test_a_decision_runs_in_its_own_trace_and_links_to_the_first(conn, keys, spans):
    stub = StubProvider(script=scripted_run())
    worker = Worker(lambda: InvestigationRunner(llm=stub))
    app.state.worker = worker
    try:
        client = TestClient(app)
        iid = _start(client, worker)
        first_job = _named(spans(), "investigation job")[0]
        _reset()
        aid = client.get(f"/investigations/{iid}", headers=VIEWER).json()["approvals"][0]["id"]
        client.post(f"/approvals/{aid}/decision", headers=APPROVER, json={"decision": "approve"})
        worker.drain(120)
    finally:
        app.state.worker = None
    found = spans()

    decision = _named(found, "POST /approvals/", SpanKind.SERVER)[0]
    job = _named(found, "investigation job")[0]
    act = _named(found, "node act")[0]
    plane = _named(found, "control_plane scale_service")[0]
    assert job.context.trace_id == decision.context.trace_id != first_job.context.trace_id
    assert job.parent.span_id == decision.context.span_id
    assert act.parent.span_id == job.context.span_id
    assert plane.parent.span_id == act.context.span_id
    [link] = job.links
    assert (link.context.trace_id, link.context.span_id) == \
        (first_job.context.trace_id, first_job.context.span_id)


def test_a_follow_up_resume_keeps_the_decision_trace(conn, keys, spans):
    """Decided while the first job is still finishing: the follow-up runs later,
    in the decision request's trace, not in the first job's."""
    gate, reached = threading.Event(), threading.Event()
    stub = StubProvider(script=scripted_run())

    class SlowToFinish(InvestigationRunner):
        def start(self, iid, query, **kwargs):
            state = super().start(iid, query, **kwargs)
            reached.set()
            gate.wait(30)
            return state

    worker = Worker(lambda: SlowToFinish(llm=stub))
    app.state.worker = worker
    client = TestClient(app)
    try:
        iid = f"IQ-OTEL-{uuid.uuid4().hex[:8]}"
        client.post("/investigations", headers=VIEWER,
                    json={"query": "checkout-service is timing out under load", "incident_id": iid})
        assert reached.wait(120)
        aid = next(a["id"] for a in client.get("/approvals", headers=VIEWER).json()
                   if a["investigation_id"] == iid)
        client.post(f"/approvals/{aid}/decision", headers=APPROVER, json={"decision": "approve"})
        gate.set()
        worker.drain(120)
    finally:
        gate.set()
        app.state.worker = None
    found = spans()

    decision = _named(found, "POST /approvals/", SpanKind.SERVER)[0]
    jobs = sorted(_named(found, "investigation job"), key=lambda s: s.start_time)
    assert len(jobs) == 2
    assert jobs[1].context.trace_id == decision.context.trace_id != jobs[0].context.trace_id


@pytest.fixture
def control_plane_server():
    """The control plane under a real server, in its own thread: nothing but
    the `traceparent` header can connect its spans to the caller's trace."""
    import incidentiq.controlplane.app as service
    import uvicorn

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(service.app, host="127.0.0.1", port=port,
                                           log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        threading.Event().wait(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(10)


def test_the_control_plane_continues_the_trace_from_the_header(conn, spans,
                                                               control_plane_server):
    client = ControlPlaneClient(httpx.Client(base_url=control_plane_server, timeout=10),
                                sleep=lambda _s: None)
    with telemetry.span("caller"):
        client.submit("restart_service", {"service": "x"}, idempotency_key=f"K-{uuid.uuid4().hex}")
    found = spans()

    caller = _named(found, "caller")[0]
    attempt = _named(found, "POST /v1/operations", SpanKind.CLIENT)[0]
    server = _named(found, "POST /v1/operations", SpanKind.SERVER)[0]
    assert server.context.trace_id == caller.context.trace_id
    assert server.parent.span_id == attempt.context.span_id
    assert server.parent.is_remote, "the parent must have come from the header"


def test_retries_are_separate_client_spans_and_failures_are_errors(spans):
    replies = [httpx.Response(503, json={"error": "unavailable"}),
               httpx.Response(201, json={"operation_id": "op-1"})]
    http = httpx.Client(transport=httpx.MockTransport(lambda _r: replies.pop(0)),
                        base_url="http://cp.test")
    ControlPlaneClient(http, sleep=lambda _s: None).submit("scale_service", {},
                                                            idempotency_key="K-retry")
    attempts = sorted(_named(spans(), "POST /v1/operations"),
                      key=lambda s: s.attributes["incidentiq.attempt"])
    assert [a.kind for a in attempts] == [SpanKind.CLIENT, SpanKind.CLIENT]
    assert [a.attributes["http.response.status_code"] for a in attempts] == [503, 201]
    assert attempts[0].status.status_code == StatusCode.ERROR
    assert attempts[0].attributes["error.type"] == "503"
    assert attempts[1].status.status_code != StatusCode.ERROR


def test_a_transport_failure_is_typed_by_its_exception(spans):
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    http = httpx.Client(transport=httpx.MockTransport(refuse), base_url="http://cp.test")
    with pytest.raises(ControlPlaneError):
        ControlPlaneClient(http, max_attempts=1, sleep=lambda _s: None).submit(
            "scale_service", {}, idempotency_key="K-refused")
    attempt = _named(spans(), "POST /v1/operations")[0]
    assert attempt.attributes["error.type"] == "ConnectError"


def test_an_exporting_app_import_cannot_displace_the_test_exporter(spans, monkeypatch):
    """With an endpoint configured, the app's own setup_tracing runs at import;
    it must find the test provider already installed and leave it in place."""
    assert telemetry.setup_tracing("incidentiq-api", endpoint="http://127.0.0.1:9/") is True
    with telemetry.span("still-captured"):
        pass
    assert _named(spans(), "still-captured")


def test_a_node_that_handled_a_failure_is_an_error_span(conn, spans, monkeypatch):
    """Retrieval failing is handled (the run continues without documents) but
    must still show in red, with the reason."""
    def broken(*_args, **_kwargs):
        raise RuntimeError("vector index unavailable")

    import incidentiq.rag.retrieval as retrieval

    monkeypatch.setattr(retrieval, "search", broken)
    stub = StubProvider(script=scripted_run())
    with InvestigationRunner(llm=stub) as runner:
        runner.start(f"IQ-OTEL-{uuid.uuid4().hex[:8]}", "checkout-service is timing out")
    retrieve = _named(spans(), "node retrieve")[0]
    assert retrieve.status.status_code == StatusCode.ERROR
    assert "vector index unavailable" in retrieve.status.description
    assert retrieve.attributes["error.type"] == "retrieve_failed"


# ── Metrics ─────────────────────────────────────────────────
def test_metrics_measure_a_run_and_its_action(conn, keys):
    before = {
        "started": _sample("incidentiq_investigations_started_total"),
        "awaiting": _sample("incidentiq_investigations_settled_total", status="awaiting_approval"),
        "complete": _sample("incidentiq_investigations_settled_total", status="complete"),
        "intake": _sample("incidentiq_node_duration_seconds_count", node="intake", outcome="ok"),
        "tool": _sample("incidentiq_tool_calls_total", tool="get_service_logs", status="success"),
        "tokens": _sample("incidentiq_llm_tokens_total", model="stub", direction="input"),
        "llm": _sample("incidentiq_llm_request_duration_seconds_count", model="stub",
                       node="intake", outcome="ok"),
        "executed": _sample("incidentiq_actions_total", tool="scale_service", outcome="executed"),
        "attempts": _sample("incidentiq_action_attempts_count", tool="scale_service"),
        "approved": _sample("incidentiq_approval_decisions_total", decision="approved"),
    }
    stub = StubProvider(script=scripted_run())
    worker = Worker(lambda: InvestigationRunner(llm=stub))
    app.state.worker = worker
    try:
        client = TestClient(app)
        iid = _start(client, worker)
        aid = client.get(f"/investigations/{iid}", headers=VIEWER).json()["approvals"][0]["id"]
        client.post(f"/approvals/{aid}/decision", headers=APPROVER, json={"decision": "approve"})
        worker.drain(120)
    finally:
        app.state.worker = None
    after = {
        "started": _sample("incidentiq_investigations_started_total"),
        "awaiting": _sample("incidentiq_investigations_settled_total", status="awaiting_approval"),
        "complete": _sample("incidentiq_investigations_settled_total", status="complete"),
        "intake": _sample("incidentiq_node_duration_seconds_count", node="intake", outcome="ok"),
        "tool": _sample("incidentiq_tool_calls_total", tool="get_service_logs", status="success"),
        "tokens": _sample("incidentiq_llm_tokens_total", model="stub", direction="input"),
        "llm": _sample("incidentiq_llm_request_duration_seconds_count", model="stub",
                       node="intake", outcome="ok"),
        "executed": _sample("incidentiq_actions_total", tool="scale_service", outcome="executed"),
        "attempts": _sample("incidentiq_action_attempts_count", tool="scale_service"),
        "approved": _sample("incidentiq_approval_decisions_total", decision="approved"),
    }
    delta = {k: after[k] - before[k] for k in before}
    # The scripted run makes five model calls (four before approval, one after)
    # of 10 input tokens each.
    assert delta == {"started": 1, "awaiting": 1, "complete": 1, "intake": 1, "tool": 1,
                     "tokens": 50, "llm": 1, "executed": 1, "attempts": 1, "approved": 1}
    assert "incidentiq_actions_total" in client.get("/metrics").text


def test_failures_are_measured_too(conn):
    """A model call that raises is still timed (outcome=error) and so is its node;
    an approved action with invalid arguments is counted as such."""
    class Down(StubProvider):
        def complete_json(self, *args, **kwargs):
            raise TimeoutError("ollama did not answer")

    llm_errors = _sample("incidentiq_llm_request_duration_seconds_count",
                         model="stub", node="intake", outcome="error")
    node_errors = _sample("incidentiq_node_duration_seconds_count", node="intake", outcome="error")
    with InvestigationRunner(llm=Down()) as runner, pytest.raises(TimeoutError):
        runner.start(f"IQ-OTEL-{uuid.uuid4().hex[:8]}", "checkout-service is timing out")
    assert _sample("incidentiq_llm_request_duration_seconds_count",
                   model="stub", node="intake", outcome="error") == llm_errors + 1
    assert _sample("incidentiq_node_duration_seconds_count",
                   node="intake", outcome="error") == node_errors + 1

    # The model leaves out a required argument; the approval runs anyway.
    script = scripted_run()
    proposal = json.loads(script[3].text)
    proposal["remediation_arguments"] = {k: v for k, v in PROPOSED.items() if k != "reason"}
    script[3] = LLMResponse(text=json.dumps(proposal), input_tokens=10, output_tokens=10,
                            model="stub", stop_reason="end_turn")
    invalid = _sample("incidentiq_actions_total", tool="scale_service", outcome="invalid_arguments")
    _reset()
    with InvestigationRunner(llm=StubProvider(script=script)) as runner:
        iid = f"IQ-OTEL-{uuid.uuid4().hex[:8]}"
        state = runner.start(iid, "checkout-service is timing out under load")
        conn.execute("UPDATE approvals SET decision = 'approved' WHERE id = %s",
                     (state["pending_approval"]["approval_id"],))
        runner.resume(iid)
    trace.get_tracer_provider().force_flush()
    act = _named(EXPORTER.get_finished_spans(), "node act")[0]
    assert act.status.status_code == StatusCode.ERROR
    assert "invalid arguments" in act.status.description
    assert _sample("incidentiq_actions_total", tool="scale_service",
                   outcome="invalid_arguments") == invalid + 1


def test_a_crashed_job_counts_as_a_failed_run(conn, keys):
    class Crashes:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def start(self, *_args):
            raise RuntimeError("worker died")

    failed = _sample("incidentiq_investigations_settled_total", status="failed")
    worker = Worker(Crashes)
    app.state.worker = worker
    try:
        _start(TestClient(app), worker)
    finally:
        app.state.worker = None
    assert _sample("incidentiq_investigations_settled_total", status="failed") == failed + 1



def test_only_the_operational_endpoints_are_left_untraced(spans, keys):
    """Exclusions match the exact paths, not any URL containing the word."""
    client = TestClient(app)
    client.get("/healthz")
    client.get("/metrics")
    client.get("/investigations/IQ-metrics-healthz", headers=VIEWER)   # 404, still a request
    names = [s.name for s in spans() if s.kind == SpanKind.SERVER]
    assert names == ["GET /investigations/{investigation_id}"]


def test_the_attempts_histogram_records_how_many_attempts(conn):
    """An action that needed two requests (a lost response, then the replay)
    is observed as 2: in the sum and in the buckets, as exported."""
    from incidentiq.controlplane import ControlPlane, Fault, FaultPlan, in_process_http
    from incidentiq.tools import registry
    from incidentiq.tools.actions import execute_action
    from prometheus_client.parser import text_string_to_metric_families

    def exported(name, **labels):
        text = TestClient(app).get("/metrics").text
        for family in text_string_to_metric_families(text):
            for sample in family.samples:
                matches = all(sample.labels.get(k) == v for k, v in labels.items())
                if sample.name == name and matches:
                    return sample.value
        return 0.0

    iid, aid = f"IQ-OTEL-{uuid.uuid4().hex[:8]}", f"APR-{uuid.uuid4().hex[:10]}"
    conn.execute("INSERT INTO investigations (id, user_query, status) VALUES (%s, 'q', 'running')",
                 (iid,))
    conn.execute("INSERT INTO approvals (id, investigation_id, proposed_tool, proposed_arguments, "
                 "reasoning, evidence, decision) VALUES (%s, %s, 'scale_service', %s, 'r', '{}', "
                 "'approved')", (aid, iid, json.dumps(PROPOSED)))
    before = {k: exported(f"incidentiq_action_attempts_{k}", tool="scale_service", **extra)
              for k, extra in (("sum", {}), ("count", {}))}
    le1 = exported("incidentiq_action_attempts_bucket", tool="scale_service", le="1.0")
    le2 = exported("incidentiq_action_attempts_bucket", tool="scale_service", le="2.0")

    plane = ControlPlane(conn, FaultPlan.scripted({aid: [Fault.LOST_RESPONSE]}))
    execute_action(conn, registry.get("scale_service"), PROPOSED, investigation_id=iid,
                   approval_id=aid,
                   control_plane=ControlPlaneClient(in_process_http(plane), sleep=lambda _s: None))

    assert exported("incidentiq_action_attempts_sum", tool="scale_service") == before["sum"] + 2
    assert exported("incidentiq_action_attempts_count", tool="scale_service") == before["count"] + 1
    assert exported("incidentiq_action_attempts_bucket", tool="scale_service", le="1.0") == le1
    assert exported("incidentiq_action_attempts_bucket", tool="scale_service", le="2.0") == le2 + 1



def _fresh_process(code: str, **env: str) -> str:
    """Run `code` in a new interpreter, where no tracer provider is installed yet."""
    import os
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
        cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT / "api"), **env})
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_tracing_is_off_without_an_endpoint():
    out = _fresh_process(
        "from incidentiq import telemetry\n"
        "print(telemetry.setup_tracing('x', ''))\n"
        "with telemetry.span('s') as s: print(s.is_recording())",
        OTEL_EXPORTER_OTLP_ENDPOINT="")
    assert out.splitlines() == ["False", "False"]


def test_an_endpoint_installs_an_otlp_exporter_on_import():
    """Importing the app with an endpoint configured is the production path."""
    out = _fresh_process(
        "import incidentiq.api.main\n"
        "from opentelemetry import trace\n"
        "p = trace.get_tracer_provider()\n"
        "print(type(p).__name__, p.resource.attributes['service.name'])\n"
        "procs = p._active_span_processor._span_processors\n"
        "print([type(x.span_exporter).__name__ for x in procs])",
        OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:9")
    assert out.splitlines() == ["TracerProvider incidentiq-api", "['OTLPSpanExporter']"]
