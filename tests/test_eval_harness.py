"""The harness decides approvals through the API's own code, records what the
agent concluded, and can rerun just the failures of a previous run."""

import json
import sys
import uuid
from pathlib import Path

import psycopg
import pytest
from conftest import database_missing

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "seeds"))

from incidentiq.config import get_settings  # noqa: E402
from test_proposals import _script, _start, _versions  # noqa: E402

from evals import run as harness  # noqa: E402
from evals.compare import render_scenarios  # noqa: E402
from evals.scenario import Scenario  # noqa: E402


@pytest.fixture(scope="module")
def conn():
    try:
        c = psycopg.connect(get_settings().database_url, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        database_missing("Postgres not reachable")
    yield c
    c.close()


def _scenario(decision: str = "approve") -> Scenario:
    return Scenario(id="SC-T", kind="approval_required", query="q", approval_decision=decision)


def test_the_harness_is_refused_where_a_human_approver_would_be(conn):
    bad = {"service": "checkout-service", "target_version": "v1.6.9", "reason": "regression"}
    state = _start(_script("rollback_deploy", bad, {"remediation_arguments": bad}))
    aid = state["pending_approval"]["approval_id"]

    refused, edited = harness._decide(get_settings(), aid, _scenario())
    assert refused is not None and refused.startswith("422") and "never deployed" in refused
    assert edited is None
    row = conn.execute("SELECT decision FROM approvals WHERE id = %s", (aid,)).fetchone()
    assert row[0] is None


def test_a_harness_decision_is_recorded_and_audited_like_a_human_one(conn):
    good = {"service": "checkout-service", "target_version": _versions(conn)[-1],
            "reason": "regression"}
    state = _start(_script("rollback_deploy", good))
    aid = state["pending_approval"]["approval_id"]

    assert harness._decide(get_settings(), aid, _scenario("reject")) == (None, None)
    decision, by, reason = conn.execute(
        "SELECT decision, decided_by, rejection_reason FROM approvals WHERE id = %s",
        (aid,)).fetchone()
    assert (decision, by) == ("rejected", "eval-harness") and reason
    assert conn.execute("SELECT count(*) FROM audit_log WHERE event_type = 'approval_decided' "
                        "AND payload->>'approval_id' = %s", (aid,)).fetchone()[0] == 1


def test_a_modify_scenario_runs_the_edited_arguments_and_says_so(conn):
    import uuid

    from incidentiq.agent.graph import InvestigationRunner
    from incidentiq.llm import StubProvider

    good = {"service": "checkout-service", "target_version": _versions(conn)[-1],
            "reason": "the last deploy broke checkout"}
    iid = f"IQ-MOD-{uuid.uuid4().hex[:8]}"
    with InvestigationRunner(llm=StubProvider(script=_script("rollback_deploy", good))) as runner:
        state = runner.start(iid, "checkout-service is timing out")
        aid = state["pending_approval"]["approval_id"]
        refused, edited = harness._decide(get_settings(), aid, _scenario("modify"))
        assert refused is None and edited["reason"].startswith("edited by the approver: ")
        assert harness._ran_edited(get_settings(), aid, edited) == \
            "the action did not run with the edited arguments"          # not yet
        runner.resume_for_decision(iid, aid)
    assert harness._ran_edited(get_settings(), aid, edited) is None
    decision, modified = conn.execute(
        "SELECT decision, modified_arguments FROM approvals WHERE id = %s", (aid,)).fetchone()
    assert decision == "modified" and modified == edited


def test_the_conclusion_is_recorded_with_the_grade():
    sc = Scenario(id="X", kind="single_service", query="q", true_cause_service="checkout-service")
    proposal = {"root_cause": "checkout-service exhausted its connection pool",
                "remediation_tool": "restart_service",
                "remediation_arguments": {"service": "checkout-service"}}
    r = harness._grade(sc, {"status": "complete", "proposal": proposal, "tool_calls": []})
    assert r.root_cause == proposal["root_cause"]
    assert (r.remediation_tool, r.remediation_arguments) == \
        ("restart_service", {"service": "checkout-service"})


def test_a_replay_reruns_only_the_failures_with_the_same_configuration(tmp_path, monkeypatch):
    from incidentiq.config import Settings

    original = Settings(failure_injection_seed=42, tool_max_retries=3)
    summary = harness.aggregate([harness.ScenarioResult("A", "single_service", "q")], original,
                                judge=True)
    run = tmp_path / "run_X.json"
    run.write_text(json.dumps({
        "summary": {**summary, "provider": "ollama", "model": "qwen2.5:14b",
                    "iteration_cap_override": 3, "seed": 7},
        "results": [{"scenario_id": "A", "task_success": True},
                    {"scenario_id": "B", "task_success": False},
                    {"scenario_id": "C", "task_success": False}]}))
    # The environment has changed since the run.
    monkeypatch.setenv("FAILURE_INJECTION_SEED", "1")
    monkeypatch.setenv("TOOL_MAX_RETRIES", "1")
    assert Settings().failure_injection_seed == 1

    failed, options, execution = harness.replay_plan(run)
    assert failed == ["B", "C"]
    assert options == {"provider": "ollama", "model": "qwen2.5:14b", "max_iterations": 3,
                       "no_judge": False, "seed": 7}
    restored = Settings().model_copy(update=execution)
    assert (restored.failure_injection_seed, restored.tool_max_retries) == (42, 3)
    assert {k: getattr(restored, k) for k in harness.EXECUTION} == summary["execution"]


def test_a_run_that_did_not_record_its_execution_settings_is_not_replayed(tmp_path):
    run = tmp_path / "run_old.json"
    run.write_text(json.dumps({"summary": {"provider": "ollama", "model": "m", "n_judged": 0},
                               "results": [{"scenario_id": "B", "task_success": False}]}))
    with pytest.raises(ValueError, match="does not record its execution settings"):
        harness.replay_plan(run)


def test_compare_lists_the_scenarios_whose_outcome_changed():
    a = [{"scenario_id": "A", "task_success": False, "failure_note": "did not name the service"},
         {"scenario_id": "B", "task_success": False, "failure_note": "named the service but not "
                                                                      "the failure mode (1 of 9)"},
         {"scenario_id": "C", "task_success": True, "failure_note": ""}]
    b = [{"scenario_id": "A", "task_success": True, "failure_note": ""},
         {"scenario_id": "B", "task_success": False, "failure_note": "named the service but not "
                                                                      "the failure mode (1 of 9)"}]
    table = render_scenarios(a, b, "before", "after")
    assert "| A | fail: did not name the service | pass |" in table
    assert "| B |" not in table and "| C |" not in table


def test_the_metrics_overview_leads_with_the_metric_each_failure_moves(conn):
    """For one scenario of every archetype, get_metrics without a metric_name
    leads with the metric that failure moves, instead of leaving the model to
    guess one (it usually guessed latency)."""
    import archetypes as A
    from incidentiq.tools import ToolStatus, execute_tool, registry

    from evals import fixtures
    from evals.scenario import load_scenarios

    first: dict[str, object] = {}
    for sc in load_scenarios(ROOT / "evals" / "scenarios"):
        if sc.expected_archetype and sc.expected_archetype not in first:
            first[sc.expected_archetype] = sc
    wrong = []
    for key, sc in first.items():
        seeded = fixtures.apply(conn, sc, run_id="test-overview")
        try:
            service = sc.true_cause_service or sc.expected_service
            r = execute_tool(conn, registry.get("get_metrics"), {"service": service},
                             investigation_id="T")
            assert r.status is ToolStatus.SUCCESS, r.error
            lead = r.data["metrics"][0]
            arch = A.BY_KEY[key]
            if lead["metric_name"] != arch.metric:
                wrong.append((key, lead))
            elif lead["pct_change_vs_yesterday"] is not None:
                # The full spike, not one averaged with normal readings.
                full = (arch.metric_spike - 1) * 100
                if abs(lead["pct_change_vs_yesterday"]) < 0.7 * abs(full):
                    wrong.append((key, lead, f"expected about {full:+.0f}%"))
        finally:
            fixtures.clear(conn, seeded)
    assert not wrong, wrong


def test_a_seed_reaches_the_local_model():
    import httpx
    from incidentiq.llm.providers import OllamaProvider

    sent = []

    def handler(request):
        sent.append(json.loads(request.content)["options"])
        return httpx.Response(200, json={"message": {"content": "{}"}, "done": True})

    provider = OllamaProvider("http://ollama", "m", seed=7)
    provider._client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://o")
    provider._post("system", [{"role": "user", "content": "hi"}], None, 10)
    assert sent[0]["seed"] == 7


def test_fixtures_left_by_a_failure_part_way_leave_the_corpus_as_it_was(conn, monkeypatch):
    """Scaled readings are only ever committed with the record of their originals."""
    from evals import fixtures
    from evals.scenario import load_scenarios

    sc = next(s for s in load_scenarios(ROOT / "evals" / "scenarios")
              if s.expected_archetype and not s.should_abstain)
    service = sc.true_cause_service or sc.expected_service

    def snapshot():
        return conn.execute("SELECT count(*), coalesce(sum(value), 0) FROM metric_points "
                            "WHERE service = %s", (service,)).fetchone()

    before = snapshot()

    def interrupted(*_args, **_kwargs):
        raise RuntimeError("killed while recording")

    monkeypatch.setattr(fixtures, "_record", interrupted)
    with pytest.raises(RuntimeError, match="killed while recording"):
        fixtures.apply(conn, sc, run_id="test-interrupted")
    assert snapshot() == before


def test_a_cleanup_that_fails_part_way_changes_nothing_and_the_purge_restores(conn, monkeypatch):
    """clear() is all or nothing: failing half way leaves the fixture rows and their
    tracking rows, and the next run's purge restores the corpus from those."""
    from evals import fixtures
    from evals.scenario import load_scenarios

    sc = next(s for s in load_scenarios(ROOT / "evals" / "scenarios")
              if s.expected_archetype and not s.should_abstain)
    service = sc.true_cause_service or sc.expected_service

    def snapshot():
        metrics = conn.execute("SELECT count(*), coalesce(sum(value), 0) FROM metric_points "
                               "WHERE service = %s", (service,)).fetchone()
        logs = conn.execute("SELECT count(*) FROM log_entries WHERE service = %s",
                            (service,)).fetchone()
        return (*metrics, *logs)

    before = snapshot()
    state = fixtures.apply(conn, sc, run_id="test-half-cleared")
    applied = snapshot()
    tracked = conn.execute("SELECT count(*) FROM eval_fixture_rows WHERE run_id = %s",
                           ("test-half-cleared",)).fetchone()[0]
    assert applied != before and tracked > 0

    calls = []
    real_execute = psycopg.Cursor.execute

    def fails_on_the_second(self, query, *args, **kwargs):
        calls.append(query)
        if len(calls) == 2:
            raise psycopg.OperationalError("connection lost")
        return real_execute(self, query, *args, **kwargs)

    monkeypatch.setattr(psycopg.Cursor, "execute", fails_on_the_second)
    with pytest.raises(psycopg.OperationalError):
        fixtures.clear(conn, state)
    monkeypatch.undo()
    assert snapshot() == applied                       # rolled back: nothing half removed
    assert conn.execute("SELECT count(*) FROM eval_fixture_rows WHERE run_id = %s",
                        ("test-half-cleared",)).fetchone()[0] == tracked
    fixtures.purge_orphans(conn, only_run_id="test-half-cleared")
    assert snapshot() == before


def test_a_modify_scenario_approves_its_own_edit_and_checks_it_ran():
    from evals.run import edit_arguments

    restart = {"service": "catalog-service", "reason": "the pool is exhausted"}
    assert edit_arguments("restart_service", restart) == {
        "service": "catalog-service", "reason": "edited by the approver: the pool is exhausted"}
    scale = {"service": "web", "replica_count": 3, "reason": "traffic surge"}
    assert edit_arguments("scale_service", scale)["replica_count"] == 4
    assert edit_arguments("scale_service", {**scale, "replica_count": 50})["replica_count"] == 50


def test_a_metric_rising_from_zero_reaches_the_evidence_gate(conn):
    """A zero yesterday came back as no reading, so a rise from it was no evidence."""
    from datetime import timedelta

    from incidentiq.agent.evidence import faults
    from incidentiq.tools import ToolStatus, execute_tool, registry
    from incidentiq.tools.diagnostics import DATA_NOW

    service, name = "checkout-service", "test_rise_from_zero"
    points = [(DATA_NOW - timedelta(hours=24, minutes=m), 0.0) for m in (10, 30, 50)] + \
             [(DATA_NOW - timedelta(minutes=m), 3.0) for m in (10, 30, 50)]
    insert = ("INSERT INTO metric_points (service, metric_name, ts, value) "
              "VALUES (%s, %s, %s, %s) RETURNING id")
    ids = [conn.execute(insert, (service, name, ts, v)).fetchone()[0] for ts, v in points]
    try:
        r = execute_tool(conn, registry.get("get_metrics"), {"service": service},
                         investigation_id="T")
        assert r.status is ToolStatus.SUCCESS, r.error
        metric = next(m for m in r.data["metrics"] if m["metric_name"] == name)
        assert (metric["same_hour_yesterday"], metric["latest_hour"]) == (0, 3)
        call = {"tool_name": "get_metrics", "status": "success",
                "arguments": {"service": service}, "result": r.data}
        assert f"{name} moved from zero" in faults([call], service)
    finally:
        conn.execute("DELETE FROM metric_points WHERE id = ANY(%s)", (ids,))


def call(tool, service, status="success", injected=False):
    return {"tool_name": tool, "arguments": {"service": service}, "status": status,
            "injected_failure": injected}


@pytest.mark.parametrize("calls,recovered", [
    ([call("get_service_logs", "a", "timeout", True), call("get_service_logs", "a")], True),
    ([call("get_service_logs", "a"), call("get_service_logs", "a", "timeout", True)], False),
    ([call("get_service_logs", "a", "timeout", True), call("get_service_logs", "b")], False),
    ([call("get_service_logs", "a", "timeout", True), call("get_metrics", "a")], False),
    ([call("get_service_logs", "a", "error"), call("get_service_logs", "a")], False),
])
def test_recovery_is_a_retry_that_succeeds_after_an_injected_failure(calls, recovered):
    assert harness.recovered(calls) is recovered


def test_a_second_evaluation_is_refused_while_the_database_is_held(conn):
    """The test session holds it (conftest), as a running evaluation would."""
    from evals import fixtures

    with pytest.raises(RuntimeError, match="another evaluation, or the test suite"):
        with fixtures.exclusive(get_settings().database_url):
            pass


def test_the_database_is_free_again_once_its_holder_lets_go(conn):
    from evals import fixtures

    url, key = get_settings().database_url, fixtures.EVAL_LOCK + 1   # not the session's
    with fixtures.exclusive(url, key):
        with pytest.raises(RuntimeError):
            with fixtures.exclusive(url, key):
                pass
    with fixtures.exclusive(url, key):
        pass


def test_a_results_file_without_saved_root_causes_is_not_regraded(tmp_path):
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"summary": {}, "results": [{"scenario_id": "SC-AP-001"}]}))
    with pytest.raises(ValueError, match="predates saved root causes"):
        harness.regrade(old)


def test_a_run_resumes_only_progress_started_by_the_same_code_on_the_same_scenarios(tmp_path):
    from incidentiq.config import Settings

    progress = tmp_path / "progress_x.jsonl"
    scenarios = [Scenario(id=i, kind="single_service", query="q") for i in ("A", "B")]
    started = harness.progress_stamp(Settings(tool_max_retries=3), scenarios)
    assert "does not record" in harness.resumable(progress, started)
    harness.stamp_path(progress).write_text(json.dumps(started))     # the run is interrupted
    clean = {**started, "revision": "abc1234"}
    harness.stamp_path(progress).write_text(json.dumps(clean))
    assert harness.resumable(progress, clean) is None
    # Resumed with other settings, other scenarios or other code: refused.
    retries = harness.progress_stamp(Settings(tool_max_retries=1), scenarios)
    assert "different execution" in harness.resumable(progress, {**retries, "revision": "abc1234"})
    other_model = harness.progress_stamp(Settings(tool_max_retries=3,
                                                  embedding_model="all-MiniLM-L6-v2"), scenarios)
    assert "different execution" in harness.resumable(progress, {**other_model,
                                                                 "revision": "abc1234"})
    fewer = harness.progress_stamp(Settings(tool_max_retries=3), scenarios[:1])
    assert "different scenario_ids" in harness.resumable(progress, {**fewer, "revision": "abc1234"})
    assert "different revision" in harness.resumable(progress, {**clean, "revision": "def5678"})
    dirty = {**clean, "revision": "abc1234-dirty"}
    harness.stamp_path(progress).write_text(json.dumps(dirty))
    assert "uncommitted" in harness.resumable(progress, dirty)


def test_compare_warns_when_the_runs_saw_different_scenarios():
    from evals.compare import _comparability

    base = {"n_scenarios": 2, "seed": 7, "by_kind": {"single_service": {}}}
    a = {**base, "scenario_ids": ["SC-SS-001", "SC-SS-002"]}
    b = {**base, "scenario_ids": ["SC-SS-001", "SC-SS-003"]}
    assert any("only in A: SC-SS-002; only in B: SC-SS-003" in w for w in _comparability(a, b))
    assert not any("different scenarios" in w for w in _comparability(a, dict(a)))



def test_approved_actions_go_out_with_the_runners_settings_not_the_environments():
    """A replay passes the recorded settings to the runner; the control plane must
    use them. The environment sends actions to the in-process plane with four
    attempts; the recorded settings name an address nothing listens on, with two."""
    from incidentiq.agent.graph import InvestigationRunner
    from incidentiq.config import get_settings
    from incidentiq.controlplane.client import ControlPlaneError
    from incidentiq.llm import StubProvider

    env = get_settings()
    assert (env.control_plane_url, env.action_max_attempts) == ("", 4)
    recorded = env.model_copy(update={"control_plane_url": "http://127.0.0.1:9",
                                      "control_plane_timeout_s": 0.5,
                                      "action_max_attempts": 2, "action_backoff_s": 0.0})
    try:
        runner = InvestigationRunner(settings=recorded, llm=StubProvider(script=[]))
    except psycopg.OperationalError:
        database_missing("Postgres not reachable")
    with runner:
        with pytest.raises(ControlPlaneError) as failed:
            runner.ctx.control_plane.submit(
                "restart_service", {"service": "checkout-service", "reason": "clear the pool"},
                idempotency_key=f"test-{uuid.uuid4().hex}")
    assert failed.value.attempts == 2 and "ConnectError" in "".join(failed.value.errors)



def test_uncommitted_changes_to_the_seeds_mark_the_revision(tmp_path):
    """Fixtures and the grader import seeds/; a change there changes what a run measures."""
    import subprocess

    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    for path in ("api/agent.py", "evals/scenarios/a.yaml", "seeds/archetypes.py",
                 "evals/results/run.json"):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text("x\n")
    git("init", "-q")
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "base")
    assert not harness.provenance(tmp_path)["revision"].endswith("-dirty")
    (tmp_path / "evals/results/run.json").write_text("regraded\n")
    assert not harness.provenance(tmp_path)["revision"].endswith("-dirty")
    (tmp_path / "seeds/archetypes.py").write_text("changed\n")
    assert harness.provenance(tmp_path)["revision"].endswith("-dirty")


def test_retrieval_and_the_search_tool_use_the_runners_embedding_model(monkeypatch):
    from incidentiq.agent import nodes
    from incidentiq.config import Settings
    from incidentiq.rag import embeddings, retrieval
    from incidentiq.tools.base import execute_tool, registry

    asked = []
    monkeypatch.setattr(embeddings, "get_embedding_provider",
                        lambda settings=None: asked.append(settings) or "the runner's model")
    used = []
    monkeypatch.setattr(retrieval, "search",
                        lambda *a, provider=None, **k: used.append(provider) or [])
    monkeypatch.setattr(retrieval, "hydrate", lambda conn, hits: [])

    recorded = Settings(embedding_model="all-MiniLM-L6-v2")
    ctx = nodes.NodeContext(conn=None, llm=None, settings=recorded)
    nodes.retrieve({"user_query": "q", "entities": {}, "tool_calls": []}, ctx)
    result = execute_tool(None, registry.get("search_similar_incidents"), {"query": "q"},
                          investigation_id="X", embeddings=ctx.embeddings)
    assert result.ok, result.error
    assert used == ["the runner's model", "the runner's model"]
    assert asked == [recorded, recorded]



def test_a_failed_cleanup_stops_the_run_before_the_next_scenario(tmp_path, monkeypatch):
    from incidentiq.config import Settings

    from evals import fixtures

    started = []

    class Runner:
        conn = None

        def __init__(self, settings=None, **_kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            pass

        def start(self, incident_id, query, reset=False):
            started.append(incident_id)
            return {"status": "complete", "proposal": None, "tool_calls": []}

    import incidentiq.agent.graph as graph
    monkeypatch.setattr(graph, "InvestigationRunner", Runner)
    monkeypatch.setattr(fixtures, "apply", lambda conn, sc, **kw: fixtures.SeededState(
        run_id="t", scenario_id=sc.id))

    def clear(conn, state):
        raise RuntimeError("connection lost")

    monkeypatch.setattr(fixtures, "clear", clear)

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            pass

    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: Conn())
    two = [Scenario(id=i, kind="single_service", query="q", true_cause_service="a")
           for i in ("A", "B")]
    with pytest.raises(harness.CleanupFailed, match="cleaning up A failed"):
        harness.run_all(two, Settings(llm_provider="stub"), None, 3, tmp_path / "progress.jsonl")
    assert started == ["EVAL-A"]


def test_a_replay_or_regrade_is_refused_when_the_scenarios_have_changed(tmp_path):
    from incidentiq.config import Settings

    summary = harness.aggregate([harness.ScenarioResult("A", "single_service", "q")],
                                Settings(llm_provider="stub"))
    run = tmp_path / "run_X.json"
    stale = {**summary, "scenarios_sha256": "0000000000000000"}
    run.write_text(json.dumps({"summary": stale, "results": [
        {"scenario_id": "SC-AP-001", "task_success": False, "root_cause": ""}]}))
    with pytest.raises(ValueError, match="scenario files have changed"):
        harness.replay_plan(run)
    with pytest.raises(ValueError, match="scenario files have changed"):
        harness.regrade(run)



def test_a_run_against_an_external_control_plane_is_not_replayed_or_resumed(tmp_path):
    from incidentiq.config import Settings

    external = Settings(llm_provider="stub", control_plane_url="http://control-plane:8001")
    summary = harness.aggregate([harness.ScenarioResult("A", "single_service", "q")], external)
    run = tmp_path / "run_X.json"
    run.write_text(json.dumps({"summary": {**summary, "judge": False}, "results": [
        {"scenario_id": "SC-AP-001", "task_success": False}]}))
    with pytest.raises(ValueError, match="external control plane"):
        harness.replay_plan(run)
    stamp = harness.progress_stamp(external, [Scenario(id="A", kind="single_service", query="q")])
    assert "external control plane" in harness.resumable(tmp_path / "progress.jsonl", stamp)


def test_a_metric_with_no_reading_yesterday_has_not_moved():
    from incidentiq.tools.diagnostics import _moved

    def metric(yesterday, latest):
        return {"pct_change_vs_yesterday": None, "same_hour_yesterday": yesterday,
                "latest_hour": latest}

    assert _moved(metric(0, 3)) == float("inf")        # from zero: further than any percentage
    assert _moved(metric(None, 3)) == 0.0              # nothing to have moved from
    assert _moved(metric(0, 0)) == 0.0
