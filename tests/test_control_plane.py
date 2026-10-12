"""Executing an approved action at most once per approval: lost responses,
repeated calls, crashes between applying and recording, an unavailable control
plane, key reuse with a different body, and concurrent requests per key."""

import json
import sys
import threading
import uuid
from pathlib import Path

import httpx
import psycopg
import pytest
from conftest import database_missing

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))

from incidentiq.config import get_settings  # noqa: E402
from incidentiq.controlplane import (  # noqa: E402
    ControlPlane,
    ControlPlaneClient,
    ControlPlaneError,
    Fault,
    FaultPlan,
    in_process_http,
)
from incidentiq.tools import ToolError, registry  # noqa: E402
from incidentiq.tools.actions import execute_action  # noqa: E402

SCALE = registry.get("scale_service")


@pytest.fixture(scope="module")
def conn():
    try:
        c = psycopg.connect(get_settings().database_url, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        database_missing("Postgres not reachable")
    if c.execute("SELECT to_regclass('control_plane_operations')").fetchone()[0] is None:
        database_missing("migration 007 not applied")
    yield c
    c.close()


def _approval(conn, args, *, decision="approved", modified=None):
    iid = f"IQ-CP-{uuid.uuid4().hex[:8]}"
    aid = f"APR-{uuid.uuid4().hex[:10]}"
    conn.execute("INSERT INTO investigations (id, user_query, status) "
                 "VALUES (%s, 'q', 'running')", (iid,))
    conn.execute(
        "INSERT INTO approvals (id, investigation_id, proposed_tool, proposed_arguments, "
        "reasoning, evidence, decision, modified_arguments) "
        "VALUES (%s, %s, 'scale_service', %s, 'r', '{}', %s, %s)",
        (aid, iid, json.dumps(args), decision,
         json.dumps(modified) if modified else None))
    return iid, aid


def _client(conn, faults=None, attempts=4):
    plane = ControlPlane(conn, FaultPlan.scripted(faults or {}))
    return ControlPlaneClient(in_process_http(plane), max_attempts=attempts,
                              sleep=lambda _s: None)


def _applied(conn, key):
    return conn.execute("SELECT count(*) FROM control_plane_operations "
                        "WHERE idempotency_key = %s", (key,)).fetchone()[0]


def _action(conn, aid):
    return conn.execute("SELECT outcome, attempts FROM actions WHERE approval_id = %s",
                        (aid,)).fetchall()


ARGS = {"service": "checkout-service", "replica_count": 8,
        "reason": "pool saturation under peak load"}


def test_lost_response_is_retried_without_applying_twice(conn):
    """The dangerous case: applied, then the response was lost. The retry must
    get the stored result back, not apply the scale change a second time."""
    iid, aid = _approval(conn, ARGS)
    client = _client(conn, {aid: [Fault.LOST_RESPONSE]})

    result = execute_action(conn, SCALE, ARGS, investigation_id=iid, approval_id=aid,
                            control_plane=client)

    assert result["execution"]["attempts"] == 2
    assert result["execution"]["replayed"] is True
    assert _applied(conn, aid) == 1
    assert _action(conn, aid) == [("executed", 2)]


def test_executing_twice_for_one_approval_applies_once(conn):
    """The executor half of crash recovery: a second call for the same approval
    returns the recorded result. The graph half is in test_approval_api."""
    iid, aid = _approval(conn, ARGS)
    sent = []
    plan = FaultPlan(decide=lambda key, _n: sent.append(key))   # records, never faults
    client = ControlPlaneClient(in_process_http(ControlPlane(conn, plan)), sleep=lambda _s: None)

    first = execute_action(conn, SCALE, ARGS, investigation_id=iid, approval_id=aid,
                           control_plane=client)
    second = execute_action(conn, SCALE, ARGS, investigation_id=iid, approval_id=aid,
                            control_plane=client)

    assert second["operation_id"] == first["operation_id"]
    assert second["execution"]["already_executed"] is True
    assert sent == [aid], "the second call went to the control plane"
    assert _applied(conn, aid) == 1
    assert len(_action(conn, aid)) == 1


def test_crash_between_apply_and_record_is_recovered_by_the_key(conn):
    """Applied downstream, then the process died before the actions row was
    written. Nothing local says it ran - the idempotency key is what saves it."""
    iid, aid = _approval(conn, ARGS)
    client = _client(conn)
    client.submit("scale_service", SCALE.validate_args(ARGS).model_dump(mode="json"),
                  idempotency_key=aid)
    assert _action(conn, aid) == []

    result = execute_action(conn, SCALE, ARGS, investigation_id=iid, approval_id=aid,
                            control_plane=client)

    assert result["execution"]["replayed"] is True
    assert _applied(conn, aid) == 1


def test_an_unavailable_control_plane_fails_cleanly_and_can_be_retried_later(conn):
    iid, aid = _approval(conn, ARGS)
    down = _client(conn, {aid: [Fault.UNAVAILABLE] * 4}, attempts=4)

    with pytest.raises(ToolError, match="did not accept"):
        execute_action(conn, SCALE, ARGS, investigation_id=iid, approval_id=aid,
                       control_plane=down)
    assert _applied(conn, aid) == 0
    assert _action(conn, aid) == [("failed", 4)]

    # Later, once it is back: same approval, same key, and the attempt count
    # keeps the history of what it took.
    execute_action(conn, SCALE, ARGS, investigation_id=iid, approval_id=aid,
                   control_plane=_client(conn))
    assert _applied(conn, aid) == 1
    assert _action(conn, aid) == [("executed", 5)]
    error = conn.execute("SELECT error FROM actions WHERE approval_id = %s", (aid,)).fetchone()[0]
    assert "HTTP 503" in error, "the earlier failures were lost when it later succeeded"


def test_a_client_error_is_not_retried(conn):
    client = _client(conn)
    with pytest.raises(ControlPlaneError) as exc:
        client.submit("delete_everything", {}, idempotency_key=f"K-{uuid.uuid4().hex}")
    assert exc.value.attempts == 1
    assert exc.value.retryable is False


def test_reusing_a_key_for_a_different_request_is_refused(conn):
    """Same key, different body means the caller lost track of what it is
    retrying. Replaying the first response would hide that."""
    plane = ControlPlane(conn)
    key = f"K-{uuid.uuid4().hex}"
    assert plane.submit(key, "scale_service", {"replica_count": 4}).status == 201
    reply = plane.submit(key, "scale_service", {"replica_count": 40})
    assert reply.status == 422
    assert "different request" in reply.body["error"]


def test_concurrent_requests_with_one_key_apply_once():
    """Five callers racing with the same key: one insert wins, the rest replay it."""
    url = get_settings().database_url
    try:
        conns = [psycopg.connect(url, autocommit=True, connect_timeout=3) for _ in range(5)]
    except psycopg.OperationalError:
        database_missing("Postgres not reachable")
    key = f"K-{uuid.uuid4().hex}"
    barrier = threading.Barrier(5)
    replies = []

    def call(c):
        barrier.wait()
        replies.append(ControlPlane(c).submit(key, "restart_service", {"service": "x"}))

    threads = [threading.Thread(target=call, args=(c,)) for c in conns]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(r.status for r in replies) == [200, 200, 200, 200, 201]
    assert len({r.body["operation_id"] for r in replies}) == 1
    assert _applied(conns[0], key) == 1
    for c in conns:
        c.close()


def test_a_modified_decision_executes_the_approvers_arguments(conn):
    """Regression: the schema allowed decision='modified' but execute_action
    only accepted 'approved', so every edited approval failed."""
    human = {**ARGS, "replica_count": 5, "reason": "approver chose a smaller step"}
    iid, aid = _approval(conn, ARGS, decision="modified", modified=human)

    result = execute_action(conn, SCALE, ARGS, investigation_id=iid, approval_id=aid,
                            control_plane=_client(conn))

    assert result["target_replicas"] == 5
    params = conn.execute("SELECT params FROM control_plane_operations "
                          "WHERE idempotency_key = %s", (aid,)).fetchone()[0]
    assert params["replica_count"] == 5


def test_seeded_faults_are_reproducible():
    a = FaultPlan.seeded(rate=0.5, seed=7)
    b = FaultPlan.seeded(rate=0.5, seed=7)
    keys = [f"APR-{i}" for i in range(40)]
    assert [a.next(k) for k in keys] == [b.next(k) for k in keys]
    assert any(a.next(k) for k in keys), "a 50% rate injected nothing in 40 draws"


def test_arguments_other_than_the_approved_ones_are_refused(conn):
    """The approval row decides what runs, not the caller."""
    iid, aid = _approval(conn, ARGS)
    tampered = {**ARGS, "replica_count": 40}
    with pytest.raises(ToolError, match="differ from the ones that were approved"):
        execute_action(conn, SCALE, tampered, investigation_id=iid, approval_id=aid,
                       control_plane=_client(conn))
    assert _applied(conn, aid) == 0



def _scripted_http(responses):
    """An httpx client that answers with `responses` in order; an exception
    instance is raised as a transport error."""
    queue = list(responses)

    def handle(request):
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        status, body = item
        return httpx.Response(status, json=body, request=request)

    return httpx.Client(transport=httpx.MockTransport(handle), base_url="http://cp.test")


OK = (201, {"operation_id": "op-1"})


@pytest.mark.parametrize("first_failure", [
    httpx.ConnectError("refused"),
    httpx.ReadTimeout("slow"),
    (429, {"error": "slow down"}),
    (503, {"error": "unavailable"}),
], ids=["connect-error", "timeout", "429", "503"])
def test_transient_failures_are_retried(first_failure):
    delays = []
    client = ControlPlaneClient(_scripted_http([first_failure, OK]), max_attempts=3,
                                backoff_s=0.25, sleep=delays.append)
    sent = client.submit("scale_service", {}, idempotency_key="K")
    assert sent.attempts == 2 and sent.body["operation_id"] == "op-1"
    assert delays == [0.25]


@pytest.mark.parametrize("status", [400, 404, 409, 422])
def test_a_permanent_client_error_is_not_retried(status):
    delays = []
    client = ControlPlaneClient(_scripted_http([(status, {"error": "no"})]), max_attempts=3,
                                sleep=delays.append)
    with pytest.raises(ControlPlaneError) as exc:
        client.submit("scale_service", {}, idempotency_key="K")
    assert exc.value.attempts == 1 and not exc.value.retryable
    assert delays == []


def test_backoff_doubles_and_stops_after_the_last_attempt():
    delays = []
    client = ControlPlaneClient(_scripted_http([(503, {})] * 4), max_attempts=4,
                                backoff_s=0.25, sleep=delays.append)
    with pytest.raises(ControlPlaneError) as exc:
        client.submit("scale_service", {}, idempotency_key="K")
    assert exc.value.attempts == 4 and exc.value.retryable
    assert delays == [0.25, 0.5, 1.0]



def test_the_standalone_service_speaks_the_same_contract(conn, monkeypatch):
    """The HTTP app the compose stack runs: header parsing, replay header, the
    different-body refusal, and a lost response surfacing as a 504."""
    import incidentiq.controlplane.app as service
    from fastapi.testclient import TestClient

    key, lost = f"K-{uuid.uuid4().hex}", f"K-{uuid.uuid4().hex}"
    monkeypatch.setattr(service, "_faults", FaultPlan.scripted({lost: [Fault.LOST_RESPONSE]}))
    client = TestClient(service.app)
    body = {"operation": "scale_service", "params": {"replica_count": 3}}

    assert client.post("/v1/operations", json=body).status_code == 400
    first = client.post("/v1/operations", json=body, headers={"Idempotency-Key": key})
    again = client.post("/v1/operations", json=body, headers={"Idempotency-Key": key})
    assert first.status_code == 201 and again.status_code == 200
    assert again.headers["Idempotent-Replay"] == "true"
    assert again.json()["operation_id"] == first.json()["operation_id"]
    other = {"operation": "scale_service", "params": {"replica_count": 30}}
    assert client.post("/v1/operations", json=other,
                       headers={"Idempotency-Key": key}).status_code == 422

    assert client.post("/v1/operations", json=body,
                       headers={"Idempotency-Key": lost}).status_code == 504
    assert _applied(conn, lost) == 1, "a lost response still means the operation was applied"
