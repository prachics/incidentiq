"""The investigation and approval API, end to end against the scripted stub.

The stub makes the run deterministic: intake, one diagnostic tool call, a
sufficient reflection, and a proposal to scale checkout-service, which stops at
the approval gate. What is under test is everything around the model - auth,
the decision rules, the worker resuming the graph, and the action executing
once.
"""

import json
import sys
import threading
import uuid
from pathlib import Path

import psycopg
import pytest
from conftest import database_missing
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))

from incidentiq.agent.graph import InvestigationRunner  # noqa: E402
from incidentiq.api import deps  # noqa: E402
from incidentiq.api.auth import parse_api_keys  # noqa: E402
from incidentiq.api.main import app  # noqa: E402
from incidentiq.api.worker import Worker  # noqa: E402
from incidentiq.config import get_settings  # noqa: E402
from incidentiq.controlplane import (  # noqa: E402
    ControlPlane,
    ControlPlaneClient,
    Fault,
    FaultPlan,
    in_process_http,
)
from incidentiq.llm import LLMResponse, StubProvider  # noqa: E402
from incidentiq.tools import registry  # noqa: E402
from incidentiq.tools.actions import execute_action  # noqa: E402

VIEWER = {"Authorization": "Bearer test-viewer-key"}
APPROVER = {"Authorization": "Bearer test-approver-key"}
PROPOSED = {"service": "checkout-service", "replica_count": 6,
            "reason": "pool saturation under peak load"}


def _json(d):
    return LLMResponse(text=json.dumps(d), input_tokens=10, output_tokens=10,
                       model="stub", stop_reason="end_turn")


def scripted_run():
    """The four calls that reach the approval gate, then the summary."""
    return [
        _json({"service": "checkout-service", "related_services": [],
               "error_signature": "connection pool exhausted", "time_window": "1h",
               "severity": "high", "symptoms": ["timeouts"], "confidence": 0.8}),
        _json({"tool_name": "get_service_logs",
               "arguments": {"service": "checkout-service", "time_window": "1h"},
               "rationale": "look at the errors"}),
        _json({"sufficient": True, "reasoning": "pool exhaustion is clear", "missing": []}),
        _json({"root_cause": "connection pool exhausted under load", "confidence": 0.8,
               "evidence_citations": [], "remediation": "scale out",
               "remediation_tool": "scale_service", "remediation_arguments": PROPOSED,
               "abstained": False, "abstention_reason": ""}),
        _json({"summary": "scaled checkout-service", "uncertainties": []}),
    ]


@pytest.fixture(scope="module")
def conn():
    try:
        c = psycopg.connect(get_settings().database_url, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        database_missing("Postgres not reachable")
    if c.execute("SELECT count(*) FROM services").fetchone()[0] == 0:
        database_missing("no seed data")
    if c.execute("SELECT to_regclass('control_plane_operations')").fetchone()[0] is None:
        database_missing("migration 007 not applied")
    yield c
    c.close()


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setattr(get_settings(), "api_keys",
                        "test-viewer-key:vera:viewer,test-approver-key:alice:approver")


@pytest.fixture
def api(conn, keys):
    """A client whose worker runs the scripted stub. Max 3 iterations keeps a
    rejected run, which goes back to planning, short."""
    stub = StubProvider(script=scripted_run())
    settings = get_settings().model_copy(update={"max_iterations": 3})
    worker = Worker(lambda: InvestigationRunner(settings=settings, llm=stub))
    worker.stub = stub   # lets a test inspect what the model was asked
    app.state.worker = worker
    yield TestClient(app), worker
    worker.drain(60)
    app.state.worker = None


def _start(client, worker):
    r = client.post("/investigations", headers=VIEWER,
                    json={"query": "checkout-service is timing out under load",
                          "incident_id": f"IQ-API-{uuid.uuid4().hex[:8]}"})
    assert r.status_code == 202, r.text
    worker.drain(120)
    iid = r.json()["id"]
    inv = client.get(f"/investigations/{iid}", headers=VIEWER).json()
    assert inv["status"] == "awaiting_approval", inv
    return iid, inv["approvals"][0]["id"]


# ── Auth ────────────────────────────────────────────────────
def test_endpoints_need_a_key(conn, keys):
    client = TestClient(app)
    assert client.get("/investigations").status_code == 401
    assert client.get("/investigations",
                      headers={"Authorization": "Bearer nope"}).status_code == 401


def test_a_viewer_cannot_decide(api):
    client, worker = api
    _, aid = _start(client, worker)
    r = client.post(f"/approvals/{aid}/decision", headers=VIEWER, json={"decision": "approve"})
    assert r.status_code == 403


def test_malformed_api_keys_fail_loudly_without_echoing_the_key():
    with pytest.raises(ValueError) as exc:
        parse_api_keys("good-key:vera:viewer,secret-key-123")
    assert "entry 2" in str(exc.value) and "secret-key-123" not in str(exc.value)
    with pytest.raises(ValueError) as exc:
        parse_api_keys("secret-key-456:alice:admin")
    assert "secret-key-456" not in str(exc.value)


# ── The happy path ──────────────────────────────────────────
def test_approve_resumes_the_run_and_executes_once(api, conn):
    client, worker = api
    iid, aid = _start(client, worker)

    detail = client.get(f"/approvals/{aid}", headers=VIEWER).json()
    assert detail["proposed_tool"] == "scale_service"
    assert detail["arguments_schema"]["properties"]["replica_count"]

    r = client.post(f"/approvals/{aid}/decision", headers=APPROVER,
                    json={"decision": "approve"})
    assert r.status_code == 202, r.text
    assert r.json()["decided_by"] == "alice"
    worker.drain(120)

    inv = client.get(f"/investigations/{iid}", headers=VIEWER).json()
    assert inv["status"] == "complete"
    assert [(a["outcome"], a["attempts"]) for a in inv["actions"]] == [("executed", 1)]
    applied = conn.execute("SELECT count(*) FROM control_plane_operations "
                           "WHERE idempotency_key = %s", (aid,)).fetchone()[0]
    assert applied == 1

    events = [(e["event_type"], e["actor"])
              for e in client.get(f"/investigations/{iid}/audit", headers=VIEWER).json()]
    assert ("investigation_requested", "vera") in events
    assert ("approval_decided", "alice") in events
    assert ("action_executed", "agent") in events


def test_an_edited_approval_runs_the_edited_arguments(api):
    client, worker = api
    iid, aid = _start(client, worker)
    edited = {**PROPOSED, "replica_count": 4, "reason": "smaller step first, then watch"}

    r = client.post(f"/approvals/{aid}/decision", headers=APPROVER,
                    json={"decision": "approve", "arguments": edited})
    assert r.json()["decision"] == "modified"
    worker.drain(120)

    inv = client.get(f"/investigations/{iid}", headers=VIEWER).json()
    assert inv["status"] == "complete"
    assert inv["actions"][0]["result"]["target_replicas"] == 4


# ── Decision rules ──────────────────────────────────────────
def test_invalid_edited_arguments_are_refused_before_anything_runs(api):
    client, worker = api
    _, aid = _start(client, worker)
    r = client.post(f"/approvals/{aid}/decision", headers=APPROVER,
                    json={"decision": "approve",
                          "arguments": {**PROPOSED, "replica_count": 500}})
    assert r.status_code == 422
    assert client.get(f"/approvals/{aid}", headers=VIEWER).json()["decision"] is None


def test_a_rejection_needs_a_reason(api):
    client, worker = api
    _, aid = _start(client, worker)
    r = client.post(f"/approvals/{aid}/decision", headers=APPROVER,
                    json={"decision": "reject"})
    assert r.status_code == 422


def test_a_rejection_sends_the_agent_back_to_planning(api, conn):
    client, worker = api
    iid, aid = _start(client, worker)
    r = client.post(f"/approvals/{aid}/decision", headers=APPROVER,
                    json={"decision": "reject",
                          "reason": "checkout is a victim; the database is the cause"})
    assert r.status_code == 202
    worker.drain(180)

    events = [(e["event_type"], e["actor"])
              for e in client.get(f"/investigations/{iid}/audit", headers=VIEWER).json()]
    assert ("approval_rejected", "alice") in events
    scratchpad = conn.execute("SELECT scratchpad FROM investigations WHERE id = %s",
                              (iid,)).fetchone()[0]
    assert any("REJECTED" in line and "database is the cause" in line for line in scratchpad)
    # The point of a rejection: the next plan is made with the reason in view.
    plans_after = [c for c in worker.stub.calls if "Available tools:" in str(c["messages"])][1:]
    assert plans_after and "database is the cause" in str(plans_after[0]["messages"])
    applied = conn.execute("SELECT count(*) FROM control_plane_operations "
                           "WHERE idempotency_key = %s", (aid,)).fetchone()[0]
    assert applied == 0


def test_a_second_decision_is_a_conflict(api):
    client, worker = api
    _, aid = _start(client, worker)
    assert client.post(f"/approvals/{aid}/decision", headers=APPROVER,
                       json={"decision": "approve"}).status_code == 202
    r = client.post(f"/approvals/{aid}/decision", headers=APPROVER,
                    json={"decision": "reject", "reason": "changed my mind"})
    assert r.status_code == 409
    assert "alice" in r.json()["detail"]


def test_concurrent_decisions_have_exactly_one_winner(conn, keys):
    """Five approvers click at once. Whatever the interleaving, the conditional
    UPDATE lets one through and the rest get a 409."""
    iid = f"IQ-RACE-{uuid.uuid4().hex[:8]}"
    aid = f"APR-{uuid.uuid4().hex[:10]}"
    conn.execute("INSERT INTO investigations (id, user_query, status) "
                 "VALUES (%s, 'q', 'awaiting_approval')", (iid,))
    conn.execute("INSERT INTO approvals (id, investigation_id, proposed_tool, "
                 "proposed_arguments, reasoning, evidence) "
                 "VALUES (%s, %s, 'scale_service', %s, 'r', '{}')",
                 (aid, iid, json.dumps(PROPOSED)))

    class RecordingWorker:
        resumed: list[str] = []

        def resume_after_decision(self, investigation_id, _approval_id):
            self.resumed.append(investigation_id)

    recorder = RecordingWorker()
    app.state.worker = recorder
    client = TestClient(app)
    barrier = threading.Barrier(5)
    codes = []

    def decide():
        barrier.wait()
        codes.append(client.post(f"/approvals/{aid}/decision", headers=APPROVER,
                                 json={"decision": "approve"}).status_code)

    threads = [threading.Thread(target=decide) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    app.state.worker = None

    assert sorted(codes) == [202, 409, 409, 409, 409]
    assert recorder.resumed == [iid], "the graph must be resumed once, by the winner"
    status = conn.execute("SELECT status FROM investigations WHERE id = %s",
                          (iid,)).fetchone()[0]
    assert status == "queued", "a decided investigation must not still read awaiting_approval"


# ── Investigations ──────────────────────────────────────────
def test_resume_is_refused_while_an_approval_is_pending(api):
    """Resuming would run `act` with no decision and close the investigation
    without its action."""
    client, worker = api
    iid, _ = _start(client, worker)
    r = client.post(f"/investigations/{iid}/resume", headers=APPROVER)
    assert r.status_code == 409


def test_an_existing_incident_id_is_a_conflict(api):
    client, worker = api
    iid, _ = _start(client, worker)
    r = client.post("/investigations", headers=VIEWER,
                    json={"query": "the same incident again", "incident_id": iid})
    assert r.status_code == 409


def test_listing_filters_by_status(api):
    client, worker = api
    iid, _ = _start(client, worker)
    rows = client.get("/investigations?status=awaiting_approval", headers=VIEWER).json()
    assert iid in {r["id"] for r in rows}
    assert all(r["status"] == "awaiting_approval" for r in rows)


# ── Recovery (regressions from review) ──────────────────────
def _awaiting(conn, iid):
    """A synthetic investigation waiting on a pending approval."""
    aid = f"APR-{uuid.uuid4().hex[:10]}"
    conn.execute("INSERT INTO investigations (id, user_query, status) "
                 "VALUES (%s, 'q', 'awaiting_approval')", (iid,))
    conn.execute("INSERT INTO approvals (id, investigation_id, proposed_tool, "
                 "proposed_arguments, reasoning, evidence) "
                 "VALUES (%s, %s, 'scale_service', %s, 'r', '{}')",
                 (aid, iid, json.dumps(PROPOSED)))
    return aid


def test_resume_is_refused_while_the_investigation_has_a_job(conn, keys):
    """Resuming during a run used to queue a second job, which later ran `act`
    with no decision and closed the investigation without its action."""
    gate = threading.Event()

    class BlockedRunner:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def start(self, _iid, _query):
            gate.wait(10)

    worker = Worker(BlockedRunner)
    app.state.worker = worker
    client = TestClient(app)
    try:
        iid = client.post("/investigations", headers=VIEWER,
                          json={"query": "checkout-service is timing out"}).json()["id"]
        r = client.post(f"/investigations/{iid}/resume", headers=APPROVER)
        assert r.status_code == 409
        assert "already queued or running" in r.json()["detail"]
    finally:
        gate.set()
        worker.drain(10)
        app.state.worker = None


def test_resume_never_runs_act_for_an_undecided_approval(conn):
    stub = StubProvider(script=scripted_run())
    iid = f"IQ-GUARD-{uuid.uuid4().hex[:8]}"
    with InvestigationRunner(llm=stub) as runner:
        runner.start(iid, "checkout-service is timing out under load")
        assert runner.next_node(iid) == ("act",)
        state = runner.resume(iid)
        assert runner.next_node(iid) == ("act",), "act ran without a decision"
    assert state["status"] == "awaiting_approval"
    status = conn.execute("SELECT status FROM investigations WHERE id = %s",
                          (iid,)).fetchone()[0]
    assert status == "awaiting_approval"


def test_a_failed_action_is_retried_by_resume_under_the_same_key(conn, keys):
    """Control plane down for every attempt -> failed, paused before `act`.
    Once it is back, an HTTP resume executes the same approval exactly once."""
    down = {"now": True}
    plan = FaultPlan(decide=lambda _key, _n: Fault.UNAVAILABLE if down["now"] else None)
    stub = StubProvider(script=scripted_run())
    settings = get_settings().model_copy(update={"max_iterations": 3})

    def runner():
        client = ControlPlaneClient(in_process_http(ControlPlane(conn, plan)),
                                    max_attempts=3, sleep=lambda _s: None)
        return InvestigationRunner(settings=settings, llm=stub, control_plane=client)

    worker = Worker(runner)
    app.state.worker = worker
    client = TestClient(app)
    try:
        iid, aid = _start(client, worker)
        client.post(f"/approvals/{aid}/decision", headers=APPROVER, json={"decision": "approve"})
        worker.drain(120)
        inv = client.get(f"/investigations/{iid}", headers=VIEWER).json()
        assert inv["status"] == "failed"
        assert "did not accept" in inv["failure_reason"]
        assert [(a["outcome"], a["attempts"]) for a in inv["actions"]] == [("failed", 3)]

        down["now"] = False
        assert client.post(f"/investigations/{iid}/resume", headers=APPROVER).status_code == 202
        worker.drain(120)
        inv = client.get(f"/investigations/{iid}", headers=VIEWER).json()
        assert inv["status"] == "complete"
        assert inv["failure_reason"] is None
        assert [(a["outcome"], a["attempts"]) for a in inv["actions"]] == [("executed", 4)]
        applied = conn.execute("SELECT count(*) FROM control_plane_operations "
                               "WHERE idempotency_key = %s", (aid,)).fetchone()[0]
        assert applied == 1
    finally:
        worker.drain(60)
        app.state.worker = None


def test_act_rerun_after_a_crash_finds_the_action_done(conn):
    """The graph half of crash recovery: the action ran, the process died before
    `act` was checkpointed, so resuming runs `act` again."""
    stub = StubProvider(script=scripted_run())
    iid = f"IQ-CRASH-{uuid.uuid4().hex[:8]}"
    with InvestigationRunner(llm=stub) as runner:
        state = runner.start(iid, "checkout-service is timing out under load")
        aid = state["pending_approval"]["approval_id"]
        conn.execute("UPDATE approvals SET decision = 'approved', decided_by = 'alice', "
                     "decided_at = now() WHERE id = %s", (aid,))
        # What `act` did before the crash: execute, with no checkpoint after it.
        execute_action(conn, registry.get("scale_service"), PROPOSED,
                       investigation_id=iid, approval_id=aid)

        final = runner.resume(iid)

    assert final["status"] == "complete"
    assert final["action_result"]["execution"]["already_executed"] is True
    applied = conn.execute("SELECT count(*) FROM control_plane_operations "
                           "WHERE idempotency_key = %s", (aid,)).fetchone()[0]
    assert applied == 1
    events = [r[0] for r in conn.execute(
        "SELECT event_type FROM audit_log WHERE investigation_id = %s", (iid,)).fetchall()]
    assert "action_already_executed" in events and "action_executed" not in events


def test_a_failed_audit_write_rolls_the_decision_back(conn, keys):
    """The decision and its audit event commit together or not at all."""
    iid = f"IQ-ATOM-{uuid.uuid4().hex[:8]}"
    aid = _awaiting(conn, iid)

    class AuditDown:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def execute(self, query, *args, **kwargs):
            if query.lstrip().startswith("INSERT INTO audit_log"):
                raise RuntimeError("audit store unavailable")
            return self._inner.execute(query, *args, **kwargs)

    def failing_db():
        for inner in deps.db():
            yield AuditDown(inner)

    class RecordingWorker:
        resumed: list[str] = []

        def resume_after_decision(self, investigation_id, _approval_id):
            self.resumed.append(investigation_id)

    recorder = RecordingWorker()
    app.state.worker = recorder
    app.dependency_overrides[deps.db] = failing_db
    try:
        r = TestClient(app, raise_server_exceptions=False).post(
            f"/approvals/{aid}/decision", headers=APPROVER, json={"decision": "approve"})
    finally:
        app.dependency_overrides.clear()
        app.state.worker = None

    assert r.status_code == 500
    decision, status = conn.execute(
        "SELECT a.decision, i.status FROM approvals a JOIN investigations i "
        "ON i.id = a.investigation_id WHERE a.id = %s", (aid,)).fetchone()
    assert decision is None, "the decision committed without its audit event"
    assert status == "awaiting_approval"
    assert recorder.resumed == []


# ── Session and live updates ────────────────────────────────
def test_me_reports_the_role(keys):
    client = TestClient(app)
    assert client.get("/me", headers=VIEWER).json() == {"user": "vera", "role": "viewer"}
    assert client.get("/me", headers=APPROVER).json()["role"] == "approver"
    assert client.get("/me").status_code == 401


def test_the_stream_needs_a_key_and_a_real_id(conn, keys):
    client = TestClient(app)
    assert client.get("/investigations/IQ-NOPE/stream").status_code == 401
    assert client.get("/investigations/IQ-NOPE/stream", headers=VIEWER).status_code == 404


class PausingStub(StubProvider):
    """The scripted run, blocked inside its Nth model call until released."""

    def __init__(self, pause_at: int):
        super().__init__(script=scripted_run())
        self.pause_at, self.calls_made = pause_at, 0
        self.reached, self.release = threading.Event(), threading.Event()

    def complete_json(self, *args, **kwargs):
        self.calls_made += 1
        if self.calls_made == self.pause_at:
            self.reached.set()
            self.release.wait(30)
        return super().complete_json(*args, **kwargs)


def test_progress_is_visible_while_the_run_is_in_flight(conn):
    """Read from another connection while the run is blocked in `reflect` (the
    third model call): the steps so far must already be in the table."""
    stub = PausingStub(pause_at=3)
    iid = f"IQ-LIVE-{uuid.uuid4().hex[:8]}"
    done = threading.Event()

    def run():
        with InvestigationRunner(llm=stub) as runner:
            runner.start(iid, "checkout-service is timing out under load")
        done.set()

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert stub.reached.wait(120)
        status, events, proposal = conn.execute(
            "SELECT status, events, proposal FROM investigations WHERE id = %s", (iid,)).fetchone()
        nodes = [e["node"] for e in events]
        assert status == "running"
        assert nodes[:4] == ["intake", "retrieve", "plan", "execute_tool"]
        assert "reflect" not in nodes and proposal is None
    finally:
        stub.release.set()
        thread.join(60)
    assert done.is_set()
    events, proposal = conn.execute("SELECT events, proposal FROM investigations WHERE id = %s",
                                    (iid,)).fetchone()
    assert "await_approval" in [e["node"] for e in events]
    assert proposal["remediation_tool"] == "scale_service"


@pytest.fixture
def live_server():
    """The app under a real uvicorn server. TestClient buffers a streaming
    response until it ends, so it cannot show a stream arriving piece by piece."""
    import socket

    import uvicorn

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
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


def test_the_stream_shows_a_run_in_flight_then_its_end(conn, keys, live_server):
    """The first snapshot arrives while the run is blocked mid-way; releasing it
    produces later snapshots and finally `end` with the settled status."""
    import httpx

    stub = PausingStub(pause_at=3)
    worker = Worker(lambda: InvestigationRunner(llm=stub))
    app.state.worker = worker
    iid = f"IQ-SSE-{uuid.uuid4().hex[:8]}"
    received: list[tuple[str, str]] = []
    try:
        httpx.post(f"{live_server}/investigations", headers=VIEWER, timeout=10,
                   json={"query": "checkout-service is timing out under load", "incident_id": iid})
        assert stub.reached.wait(120)
        with httpx.stream("GET", f"{live_server}/investigations/{iid}/stream",
                          headers=VIEWER, timeout=60) as r:
            event = None
            for line in r.iter_lines():
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                elif line.startswith("data:") and event:
                    received.append((event, line.split(":", 1)[1].strip()))
                    if len(received) == 1:
                        first = json.loads(received[0][1])
                        nodes = [e["node"] for e in first["events"]]
                        assert first["status"] == "running"
                        assert nodes[:3] == ["intake", "retrieve", "plan"]   # progress so far
                        assert "reflect" not in nodes                        # and no further
                        stub.release.set()   # only now let the run continue
    finally:
        stub.release.set()
        worker.drain(60)
        app.state.worker = None

    snapshots = [json.loads(data) for event, data in received if event == "snapshot"]
    assert len(snapshots) >= 2
    assert snapshots[-1]["status"] == "awaiting_approval"
    assert received[-1] == ("end", "awaiting_approval")


def test_a_decision_made_while_its_job_is_still_finishing_is_not_lost(conn, keys):
    """The approval is visible before the job that created it has returned. A
    decision in that window used to be stored but never resumed."""
    gate, reached = threading.Event(), threading.Event()
    stub = StubProvider(script=scripted_run())

    class SlowToFinish:
        def __init__(self):
            self.inner = InvestigationRunner(llm=stub)

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            self.inner.close()
            return False

        def start(self, iid, query):
            self.inner.start(iid, query)
            reached.set()
            gate.wait(30)

        def resume(self, iid):
            return self.inner.resume(iid)

        def resume_for_decision(self, iid, aid):
            return self.inner.resume_for_decision(iid, aid)

    worker = Worker(SlowToFinish)
    app.state.worker = worker
    client = TestClient(app)
    try:
        iid = f"IQ-SLOW-{uuid.uuid4().hex[:8]}"
        client.post("/investigations", headers=VIEWER,
                    json={"query": "checkout-service is timing out under load", "incident_id": iid})
        assert reached.wait(120)
        aid = next(a["id"] for a in client.get("/approvals", headers=VIEWER).json()
                   if a["investigation_id"] == iid)
        assert client.post(f"/approvals/{aid}/decision", headers=APPROVER,
                           json={"decision": "approve"}).status_code == 202
        gate.set()
        worker.drain(120)
        inv = client.get(f"/investigations/{iid}", headers=VIEWER).json()
        assert inv["status"] == "complete"
        assert [a["outcome"] for a in inv["actions"]] == ["executed"]
    finally:
        gate.set()
        worker.drain(30)
        app.state.worker = None


def test_concurrent_first_requests_share_one_worker():
    """Two requests creating the worker at once must not end up with two
    workers, each with its own queue and its own idea of what is running."""
    app.state.worker = None

    class FakeRequest:
        def __init__(self):
            self.app = app

    barrier = threading.Barrier(8)
    seen = []

    def grab():
        barrier.wait()
        seen.append(deps.worker(FakeRequest()))

    threads = [threading.Thread(target=grab) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    app.state.worker = None
    assert len({id(w) for w in seen}) == 1



def test_concurrent_resumes_admit_exactly_one(conn, keys):
    """Both requests could pass a check-then-act; the worker slot is now claimed
    before anything is written, so one gets 202 and the other 409."""
    iid = f"IQ-RES-{uuid.uuid4().hex[:8]}"
    conn.execute("INSERT INTO investigations (id, user_query, status) "
                 "VALUES (%s, 'checkout-service is timing out', 'failed')", (iid,))
    gate = threading.Event()

    class Blocked:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def start(self, _iid, _query):
            gate.wait(10)

    worker = Worker(Blocked)
    app.state.worker = worker
    client = TestClient(app)
    barrier = threading.Barrier(2)
    codes = []

    def resume():
        barrier.wait()
        codes.append(client.post(f"/investigations/{iid}/resume", headers=APPROVER).status_code)

    threads = [threading.Thread(target=resume) for _ in range(2)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        gate.set()
        worker.drain(10)
        app.state.worker = None
    assert sorted(codes) == [202, 409]
    resumed = conn.execute("SELECT count(*) FROM audit_log WHERE investigation_id = %s "
                           "AND event_type = 'investigation_resumed'", (iid,)).fetchone()[0]
    assert resumed == 1


def _crash_after_approval_insert(client, conn, monkeypatch, decide_first):
    """The process dies after await_approval committed the approval (and its
    audit event) but before LangGraph checkpointed the node."""
    from incidentiq.agent import nodes

    real_event, died = nodes._event, {"done": False}

    def event(node, started, **extra):
        if node == "await_approval" and not died["done"]:
            died["done"] = True
            raise RuntimeError("process died")
        return real_event(node, started, **extra)

    monkeypatch.setattr(nodes, "_event", event)
    stub = StubProvider(script=scripted_run())
    worker = Worker(lambda: InvestigationRunner(llm=stub))
    app.state.worker = worker
    iid = f"IQ-APRCRASH-{uuid.uuid4().hex[:8]}"
    client.post("/investigations", headers=VIEWER,
                json={"query": "checkout-service is timing out under load", "incident_id": iid})
    worker.drain(120)
    inv = client.get(f"/investigations/{iid}", headers=VIEWER).json()
    assert inv["status"] == "failed" and len(inv["approvals"]) == 1
    aid = inv["approvals"][0]["id"]

    if decide_first:
        # Recover by deciding the approval that survived the crash.
        assert client.post(f"/approvals/{aid}/decision", headers=APPROVER,
                           json={"decision": "approve"}).status_code == 202
        worker.drain(120)
    else:
        # Recover by resuming: the run returns to the same approval and waits.
        assert client.post(f"/investigations/{iid}/resume", headers=APPROVER).status_code == 202
        worker.drain(120)
        again = client.get(f"/investigations/{iid}", headers=VIEWER).json()
        assert again["status"] == "awaiting_approval"
        assert [a["id"] for a in again["approvals"]] == [aid]
        client.post(f"/approvals/{aid}/decision", headers=APPROVER, json={"decision": "approve"})
        worker.drain(120)

    final = client.get(f"/investigations/{iid}", headers=VIEWER).json()
    app.state.worker = None
    assert final["status"] == "complete"
    assert [a["id"] for a in final["approvals"]] == [aid], "a second approval was created"
    assert [(a["approval_id"], a["outcome"]) for a in final["actions"]] == [(aid, "executed")]
    requested = conn.execute("SELECT count(*) FROM audit_log WHERE investigation_id = %s "
                             "AND event_type = 'approval_requested'", (iid,)).fetchone()[0]
    assert requested == 1


def test_a_crash_after_creating_an_approval_recovers_by_deciding_it(conn, keys, monkeypatch):
    _crash_after_approval_insert(TestClient(app), conn, monkeypatch, decide_first=True)


def test_a_crash_after_creating_an_approval_recovers_by_resuming(conn, keys, monkeypatch):
    _crash_after_approval_insert(TestClient(app), conn, monkeypatch, decide_first=False)


def test_a_decision_the_run_consumed_itself_is_not_acted_on_twice(conn, keys):
    """Decided before the finishing run checked, so the run consumes it and the
    decision's follow-up must not resume again. With the control plane down,
    acting twice would spend two retry budgets without anyone asking."""
    gate, reached = threading.Event(), threading.Event()
    stub = StubProvider(script=scripted_run())
    plan = FaultPlan(decide=lambda _key, _n: Fault.UNAVAILABLE)

    class PausesBeforeSettling(InvestigationRunner):
        def _settle(self, incident_id, final):
            if final.get("status") == "awaiting_approval" and not reached.is_set():
                reached.set()
                gate.wait(30)
            return super()._settle(incident_id, final)

    def runner():
        client = ControlPlaneClient(in_process_http(ControlPlane(conn, plan)),
                                    max_attempts=3, sleep=lambda _s: None)
        return PausesBeforeSettling(llm=stub, control_plane=client)

    worker = Worker(runner)
    app.state.worker = worker
    client = TestClient(app)
    try:
        iid = f"IQ-ONCE-{uuid.uuid4().hex[:8]}"
        client.post("/investigations", headers=VIEWER,
                    json={"query": "checkout-service is timing out under load", "incident_id": iid})
        assert reached.wait(120)
        aid = next(a["id"] for a in client.get("/approvals", headers=VIEWER).json()
                   if a["investigation_id"] == iid)
        client.post(f"/approvals/{aid}/decision", headers=APPROVER, json={"decision": "approve"})
        gate.set()
        worker.drain(120)
        inv = client.get(f"/investigations/{iid}", headers=VIEWER).json()
        assert inv["status"] == "failed"
        assert [(a["outcome"], a["attempts"]) for a in inv["actions"]] == [("failed", 3)]
    finally:
        gate.set()
        worker.drain(30)
        app.state.worker = None



@pytest.mark.parametrize("edit", [False, True], ids=["unchanged", "edited"])
def test_a_rerun_after_reset_never_reuses_an_earlier_runs_keys(conn, edit):
    """Same investigation id and the same proposal after reset=True (what the
    eval harness does): the new approval must execute again, not replay the old
    operation - and edited arguments must not collide with the old key."""
    stub = StubProvider(script=scripted_run() + scripted_run())
    iid = f"IQ-RERUN-{uuid.uuid4().hex[:8]}"
    with InvestigationRunner(llm=stub) as runner:
        first = runner.start(iid, "checkout-service is timing out under load")
        a1 = first["pending_approval"]["approval_id"]
        conn.execute("UPDATE approvals SET decision = 'approved' WHERE id = %s", (a1,))
        assert runner.resume(iid)["status"] == "complete"

        second = runner.start(iid, "checkout-service is timing out under load", reset=True)
        a2 = second["pending_approval"]["approval_id"]
        assert a2 != a1
        if edit:
            edited = {**PROPOSED, "replica_count": 4, "reason": "smaller step this time"}
            conn.execute("UPDATE approvals SET decision = 'modified', modified_arguments = %s "
                         "WHERE id = %s", (json.dumps(edited), a2))
        else:
            conn.execute("UPDATE approvals SET decision = 'approved' WHERE id = %s", (a2,))
        final = runner.resume(iid)

    assert final["status"] == "complete"
    assert final["action_result"]["execution"]["replayed"] is False
    params = conn.execute("SELECT params FROM control_plane_operations "
                          "WHERE idempotency_key = %s", (a2,)).fetchone()[0]
    assert params["replica_count"] == (4 if edit else PROPOSED["replica_count"])



def test_a_resume_cannot_overtake_a_start_that_has_not_been_submitted(conn, keys):
    """Between the start's commit and its submit, a resume used to be able to
    run the investigation first; the delayed start then failed it."""
    gate, paused = threading.Event(), threading.Event()
    started = []

    class Noop:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def start(self, iid, _query):
            started.append(iid)

    class SlowToSubmit(Worker):
        def start(self, investigation_id, query, *, reserved=False):
            paused.set()
            gate.wait(10)
            return super().start(investigation_id, query, reserved=reserved)

    worker = SlowToSubmit(Noop)
    app.state.worker = worker
    client = TestClient(app)
    iid = f"IQ-RACE2-{uuid.uuid4().hex[:8]}"
    codes = {}

    def start():
        codes["start"] = client.post("/investigations", headers=VIEWER, json={
            "query": "checkout-service is timing out", "incident_id": iid}).status_code

    t = threading.Thread(target=start)
    try:
        t.start()
        assert paused.wait(10)
        codes["resume"] = client.post(f"/investigations/{iid}/resume", headers=APPROVER).status_code
    finally:
        gate.set()
        t.join()
        worker.drain(10)
        app.state.worker = None
    assert codes == {"start": 202, "resume": 409}
    assert started == [iid]


def test_settling_fails_visibly_when_a_resume_makes_no_progress(conn):
    """If the graph re-pauses at the same interrupt without running `act`, the
    settle loop must not spin."""
    stub = StubProvider(script=scripted_run())
    iid = f"IQ-STUCK-{uuid.uuid4().hex[:8]}"
    with InvestigationRunner(llm=stub) as runner:
        state = runner.start(iid, "checkout-service is timing out under load")
        conn.execute("UPDATE approvals SET decision = 'approved' WHERE id = %s",
                     (state["pending_approval"]["approval_id"],))
        config = {"configurable": {"thread_id": iid}}
        runner._advance = lambda *_a, **_k: runner.graph.get_state(config).values
        with pytest.raises(RuntimeError, match="no progress"):
            runner._settle(iid, state)
