"""Proposed actions are checked before a human is asked to approve them.

Found on a real run: the local model proposed restart_service without its
required `reason`; the approval was created, approved, and only then failed
validation. Earlier, it proposed rolling back to a version never deployed.
"""

import json
import sys
import uuid
from pathlib import Path

import psycopg
import pytest
from conftest import database_missing
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))

from incidentiq.agent.graph import InvestigationRunner  # noqa: E402
from incidentiq.api.main import app  # noqa: E402
from incidentiq.config import get_settings  # noqa: E402
from incidentiq.llm import LLMResponse, StubProvider  # noqa: E402
from test_approval_api import APPROVER, VIEWER, scripted_run  # noqa: E402


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


def _reply(data: dict) -> LLMResponse:
    return LLMResponse(text=json.dumps(data), input_tokens=10, output_tokens=10, model="stub",
                       stop_reason="end_turn")


def _script(tool: str, arguments: dict, *repair: dict) -> list[LLMResponse]:
    """The scripted run with a different proposal, plus any repair answers."""
    script = scripted_run()
    proposal = json.loads(script[3].text)
    proposal.update(remediation_tool=tool, remediation_arguments=arguments,
                    remediation="roll back the bad deploy to the last good version")
    script[3] = _reply(proposal)
    return script[:4] + [_reply(r) for r in repair] + script[4:]


def _start(script: list[LLMResponse]) -> dict:
    with InvestigationRunner(llm=StubProvider(script=script)) as runner:
        return runner.start(f"IQ-PROP-{uuid.uuid4().hex[:8]}", "checkout-service is timing out")


def _versions(conn, service="checkout-service") -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT version FROM deploys WHERE service = %s ORDER BY deployed_at DESC", (service,))]


def test_a_missing_reason_is_filled_from_the_remediation(conn):
    state = _start(_script("restart_service", {"service": "checkout-service"}))
    proposal = state["proposal"]
    assert proposal["argument_problems"] == []
    assert proposal["remediation_arguments"]["reason"].startswith("roll back the bad deploy")
    assert any("filled the missing `reason`" in line for line in state["scratchpad"])


def test_a_hallucinated_version_is_repaired_from_the_real_history(conn):
    real = _versions(conn)[-1]
    bad = {"service": "checkout-service", "target_version": "v1.6.9", "reason": "regression"}
    state = _start(_script("rollback_deploy", bad,
                           {"remediation_arguments": {"service": "checkout-service",
                                                      "target_version": real}}))
    proposal = state["proposal"]
    assert proposal["argument_problems"] == []
    assert proposal["remediation_arguments"]["target_version"] == real
    assert any("repaired arguments" in line for line in state["scratchpad"])


def test_arguments_that_stay_invalid_cannot_be_approved_as_proposed(conn, keys):
    bad = {"service": "checkout-service", "target_version": "v1.6.9", "reason": "regression"}
    state = _start(_script("rollback_deploy", bad, {"remediation_arguments": bad}))
    assert "never deployed" in state["proposal"]["argument_problems"][0]
    aid = state["pending_approval"]["approval_id"]

    class RecordingWorker:
        def resume_after_decision(self, *_args):
            pass

    app.state.worker = RecordingWorker()
    try:
        client = TestClient(app)
        detail = client.get(f"/approvals/{aid}", headers=VIEWER).json()
        assert "never deployed" in detail["argument_problems"][0]

        r = client.post(f"/approvals/{aid}/decision", headers=APPROVER,
                        json={"decision": "approve"})
        assert r.status_code == 422
        assert "edited arguments" in r.json()["detail"]["message"]

        still_bad = client.post(f"/approvals/{aid}/decision", headers=APPROVER, json={
            "decision": "approve", "arguments": {**bad, "target_version": "v9.9.9"}})
        assert still_bad.status_code == 422

        fixed = {**bad, "target_version": _versions(conn)[-1]}
        ok = client.post(f"/approvals/{aid}/decision", headers=APPROVER,
                         json={"decision": "approve", "arguments": fixed})
        assert ok.status_code == 202 and ok.json()["decision"] == "modified"
    finally:
        app.state.worker = None


def test_a_repair_that_moves_the_action_to_an_unseen_service_is_withheld(conn):
    """The evidence is about checkout-service; a repair may not carry the action to a
    service nobody looked at."""
    bad = {"service": "checkout-service", "target_version": "v1.6.9", "reason": "regression"}
    elsewhere = {"service": "payment-service",
                 "target_version": _versions(conn, "payment-service")[-1], "reason": "regression"}
    state = _start(_script("rollback_deploy", bad, {"remediation_arguments": elsewhere}))
    proposal = state["proposal"]
    assert proposal["abstained"] and proposal["remediation_tool"] is None
    assert proposal["abstention_reason"].startswith(
        "no current evidence of a problem on payment-service")
    assert "pending_approval" not in state or not state["pending_approval"]


@pytest.mark.parametrize("name,arguments,problem", [
    ("restart_service", {"service": "no-such-service", "reason": "clear the stuck pool"},
     "unknown service"),
    ("scale_service", {"service": "no-such-service", "replica_count": 3,
                       "reason": "more capacity for the load"}, "unknown service"),
    ("rollback_deploy", {"service": "checkout-service", "target_version": "v1.6.9",
                         "reason": "a regression in the release"}, "never deployed"),
])
def test_arguments_are_checked_without_running_the_tool(conn, monkeypatch, name, arguments,
                                                        problem):
    from incidentiq.tools import argument_problems, registry

    tool = registry.get(name)

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("a proposal's arguments were checked by running the tool")

    monkeypatch.setattr(type(tool), "run", must_not_run)
    found = argument_problems(conn, tool, arguments)
    assert len(found) == 1 and problem in found[0]


def test_every_write_tool_has_its_own_check():
    from incidentiq.tools import registry
    from incidentiq.tools.base import Tool

    assert registry.write()
    for tool in registry.write():
        assert type(tool).check is not Tool.check, tool.name
