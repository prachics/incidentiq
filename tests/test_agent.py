"""Agent tests: tools, routing, retry policy, and state recovery.

Everything here runs against the deterministic stub provider. That is the point
of the stub: the structural properties of the graph - does the loop terminate,
does state survive a crash, does a rejection re-enter planning, can an
unapproved action run - have nothing to do with model quality. Testing them
against a real model would make them slow, costly and flaky, so they would be
run rarely and trusted less.
"""

import sys
import uuid
from pathlib import Path

import psycopg
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))

from incidentiq.agent.graph import (  # noqa: E402
    route_after_act,
    route_after_propose,
    route_after_reflect,
)
from incidentiq.agent.state import (  # noqa: E402
    cited_doc_ids,
    initial_state,
    tool_success_rate,
)
from incidentiq.config import Settings, get_settings  # noqa: E402
from incidentiq.llm import LLMResponse, StubProvider, extract_json  # noqa: E402
from incidentiq.tools import ToolStatus, execute_tool, registry  # noqa: E402
from incidentiq.tools.base import FailureInjector, FailureMode  # noqa: E402


@pytest.fixture(scope="module")
def conn():
    from pgvector.psycopg import register_vector
    try:
        c = psycopg.connect(get_settings().database_url, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable")
    register_vector(c)
    if c.execute("SELECT count(*) FROM services").fetchone()[0] == 0:
        pytest.skip("no seed data")
    yield c
    c.close()


# ── State ───────────────────────────────────────────────────
class TestState:
    def test_initial_state_populates_every_key(self):
        s = initial_state("IQ-1", "q")
        for key in ("entities", "retrieved_docs", "tool_calls", "scratchpad",
                    "iteration", "proposal", "pending_approval", "status", "events"):
            assert key in s, f"{key} missing - a node reading it would KeyError after checkpoint"

    def test_state_is_json_serialisable(self):
        """A checkpoint that cannot be serialised fails exactly in the situation
        the checkpointing exists for."""
        import json
        json.dumps(initial_state("IQ-1", "q"))

    def test_tool_success_rate_counts_attempts_not_calls(self):
        s = initial_state("IQ-1", "q")
        s["tool_calls"] = [
            {"status": "timeout"}, {"status": "timeout"}, {"status": "success"},
        ]
        # One logical call that needed three attempts is 1/3, not 1/1.
        assert tool_success_rate(s) == pytest.approx(1 / 3)

    def test_partial_counts_as_success(self):
        s = initial_state("IQ-1", "q")
        s["tool_calls"] = [{"status": "partial"}, {"status": "success"}]
        assert tool_success_rate(s) == 1.0

    def test_tool_success_rate_none_when_no_calls(self):
        assert tool_success_rate(initial_state("IQ-1", "q")) is None

    def test_cited_doc_ids(self):
        s = initial_state("IQ-1", "q")
        s["retrieved_docs"] = [{"parent_doc_id": "INC-1"}, {"parent_doc_id": "INC-2"},
                               {"parent_doc_id": "INC-1"}]
        assert cited_doc_ids(s) == {"INC-1", "INC-2"}


# ── Routing ─────────────────────────────────────────────────
class TestRouting:
    def test_loop_terminates_at_cap_even_if_model_says_continue(self):
        """A safety property, not a reasoning one: a model that always answers
        'keep going' must still terminate."""
        assert route_after_reflect(
            {"iteration": 8, "max_iterations": 8, "should_continue": True}
        ) == "propose"

    def test_continues_while_budget_remains(self):
        assert route_after_reflect(
            {"iteration": 2, "max_iterations": 8, "should_continue": True}
        ) == "plan"

    def test_stops_when_evidence_sufficient(self):
        assert route_after_reflect(
            {"iteration": 2, "max_iterations": 8, "should_continue": False}
        ) == "propose"

    def test_write_action_goes_to_approval(self):
        assert route_after_propose(
            {"proposal": {"requires_approval": True, "remediation_tool": "restart_service"}}
        ) == "await_approval"

    def test_read_only_conclusion_skips_approval(self):
        assert route_after_propose(
            {"proposal": {"requires_approval": False, "remediation_tool": None}}
        ) == "summarize"

    def test_abstention_skips_approval(self):
        assert route_after_propose(
            {"proposal": {"abstained": True, "requires_approval": False,
                          "remediation_tool": None}}
        ) == "summarize"

    def test_rejection_reenters_planning(self):
        """A rejection is a correction, not a dead end."""
        assert route_after_act(
            {"pending_approval": {"decision": "rejected"},
             "iteration": 2, "max_iterations": 8}
        ) == "plan"

    def test_rejection_at_cap_does_not_loop_forever(self):
        assert route_after_act(
            {"pending_approval": {"decision": "rejected"},
             "iteration": 8, "max_iterations": 8}
        ) == "summarize"

    def test_approval_closes_out(self):
        assert route_after_act(
            {"pending_approval": {"decision": "approved"},
             "iteration": 2, "max_iterations": 8}
        ) == "summarize"


# ── Tools ───────────────────────────────────────────────────
class TestTools:
    def test_all_eight_tools_registered(self):
        assert len(registry.read_only()) == 5
        assert len(registry.write()) == 3

    def test_write_tools_all_require_approval(self):
        assert all(t.requires_approval for t in registry.write())

    def test_read_tools_never_require_approval(self):
        assert not any(t.requires_approval for t in registry.read_only())

    def test_every_tool_exposes_a_json_schema(self):
        for tool in registry.all():
            schema = tool.schema()
            assert schema["name"] and schema["description"]
            assert "properties" in schema["input_schema"]

    def test_unknown_service_is_malformed_not_failed(self, conn):
        """Argument errors must not be retried - retrying identical bad
        arguments burns the budget on something that cannot succeed."""
        r = execute_tool(conn, registry.get("get_service_logs"),
                         {"service": "nope-service"}, investigation_id="T")
        assert r.status is ToolStatus.MALFORMED

    def test_unknown_service_suggests_a_real_one(self, conn):
        r = execute_tool(conn, registry.get("get_service_logs"),
                         {"service": "checkout-svc"}, investigation_id="T")
        assert "checkout-service" in (r.error or "")

    def test_empty_service_rejected_by_schema(self, conn):
        r = execute_tool(conn, registry.get("get_service_logs"),
                         {"service": ""}, investigation_id="T")
        assert r.status is ToolStatus.MALFORMED

    def test_rollback_rejects_unknown_version(self, conn):
        r = execute_tool(conn, registry.get("rollback_deploy"),
                         {"service": "checkout-service", "target_version": "v99.99.99",
                          "reason": "testing the guard"}, investigation_id="T")
        assert r.status is ToolStatus.MALFORMED
        assert "never deployed" in (r.error or "")

    def test_tool_bug_does_not_kill_the_graph(self, conn):
        """An exception inside a tool becomes a recorded failure, never an
        unhandled raise - an unhandled raise leaves no checkpoint."""
        class Exploding(registry.get("get_service_logs").__class__):
            name = "exploding_tool"
            def run(self, conn, args):
                raise RuntimeError("boom")

        r = execute_tool(conn, Exploding(), {"service": "checkout-service"},
                         investigation_id="T")
        assert r.status is ToolStatus.FAILED
        assert "boom" in (r.error or "")


# ── Failure injection ───────────────────────────────────────
class TestFailureInjection:
    def _injector(self, **kw):
        return FailureInjector(Settings(failure_injection_enabled=True,
                                        failure_injection_rate=0.5,
                                        failure_injection_seed=42, **kw))

    def test_disabled_by_default(self):
        inj = FailureInjector(Settings(failure_injection_enabled=False))
        assert all(inj.decide("IQ", "t", a) is None for a in range(1, 40))

    def test_deterministic_for_the_same_key(self):
        a, b = self._injector(), self._injector()
        for attempt in range(1, 20):
            assert a.decide("IQ-1", "get_metrics", attempt) == \
                   b.decide("IQ-1", "get_metrics", attempt)

    def test_retry_is_a_different_draw(self):
        """If attempt were not part of the key, a failing call would fail
        forever and retry logic would be untestable."""
        inj = self._injector()
        verdicts = [inj.decide("IQ-1", "get_metrics", a) for a in range(1, 25)]
        assert any(v is None for v in verdicts), "no attempt ever succeeds"
        assert any(v is not None for v in verdicts), "no attempt ever fails"

    def test_all_three_modes_occur(self):
        inj = self._injector()
        modes = {inj.decide(f"IQ-{i}", "t", 1) for i in range(300)} - {None}
        assert modes == {FailureMode.TIMEOUT, FailureMode.MALFORMED, FailureMode.PARTIAL}

    def test_rate_is_approximately_honoured(self):
        inj = self._injector()
        failures = sum(inj.decide(f"IQ-{i}", "t", 1) is not None for i in range(2000))
        assert 0.44 < failures / 2000 < 0.56

    def test_partial_result_announces_itself(self, conn):
        """The dangerous mode: valid but incomplete. If it did not flag itself
        the agent would reason confidently from evidence it never saw."""
        inj = self._injector()
        for i in range(300):
            r = execute_tool(conn, registry.get("get_service_logs"),
                             {"service": "order-service", "time_window": "8h"},
                             investigation_id=f"IQ-{i}", injector=inj)
            if r.status is ToolStatus.PARTIAL:
                assert r.truncated
                assert r.data["_truncated"] is True
                assert "incomplete" in r.data["_note"].lower()
                assert "PARTIAL" in r.for_prompt()
                return
        pytest.fail("no partial result produced in 300 draws")


# ── LLM plumbing ────────────────────────────────────────────
class TestLLM:
    def test_extract_json_from_fenced_block(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_extract_json_from_prose(self):
        assert extract_json('Sure! {"a": 1} hope that helps') == {"a": 1}

    def test_extract_json_nested(self):
        assert extract_json('{"a": {"b": [1,2]}}') == {"a": {"b": [1, 2]}}

    def test_extract_json_returns_none_when_absent(self):
        assert extract_json("no json here") is None
        assert extract_json("") is None

    def test_stub_records_calls_for_inspection(self):
        p = StubProvider()
        p.complete("sys", [{"role": "user", "content": "hi"}])
        assert p.calls and p.calls[0]["system"] == "sys"

    def test_stub_follows_its_script_then_falls_back(self):
        p = StubProvider(script=[LLMResponse(text='{"sufficient": true}')])
        assert p.complete("s", []).text == '{"sufficient": true}'
        # Script exhausted: the stub keeps working rather than raising, so a
        # test that under-scripts still exercises the whole graph.
        assert p.complete("s", []).text != '{"sufficient": true}'


# ── State recovery: the headline claim ──────────────────────
class TestStateRecovery:
    """"State survives failures" is only worth claiming if it is demonstrated.

    Each test below kills or abandons a run mid-investigation and asserts that a
    completely fresh runner - new connections, new graph object, nothing shared
    in memory - resumes from the Postgres checkpoint rather than starting over.
    """

    @staticmethod
    def _settings():
        return Settings(llm_provider="stub", max_iterations=3,
                        failure_injection_enabled=False)

    @staticmethod
    def _scripted_llm(fail_after: int | None = None):
        """A stub that raises after N calls, simulating a process dying
        mid-investigation."""
        class Crashing(StubProvider):
            def __init__(self):
                super().__init__()
                self._n = 0

            def complete(self, system, messages, **kw):
                self._n += 1
                if fail_after is not None and self._n > fail_after:
                    raise RuntimeError("simulated process death")
                return super().complete(system, messages, **kw)

        return Crashing()

    def test_checkpoint_is_written_during_the_run(self, conn):
        from incidentiq.agent.graph import InvestigationRunner

        iid = f"IQ-REC-{uuid.uuid4().hex[:8]}"
        with InvestigationRunner(settings=self._settings(),
                                 llm=StubProvider()) as runner:
            runner.start(iid, "checkout-service is throwing errors")

        rows = conn.execute(
            "SELECT count(*) FROM checkpoints WHERE thread_id = %s", (iid,)
        ).fetchone()[0]
        assert rows > 1, "expected a checkpoint per node transition, not just one"

    def test_a_fresh_runner_reads_state_from_postgres(self, conn):
        """Nothing is shared between the two runners except the database."""
        from incidentiq.agent.graph import InvestigationRunner

        iid = f"IQ-REC-{uuid.uuid4().hex[:8]}"
        with InvestigationRunner(settings=self._settings(),
                                 llm=StubProvider()) as first:
            original = first.start(iid, "payment-service latency is way up")

        # Everything above is now out of scope: connections closed, graph gone.
        with InvestigationRunner(settings=self._settings(),
                                 llm=StubProvider()) as second:
            recovered = second.get_state(iid)

        assert recovered is not None, "state did not survive the runner"
        assert recovered["incident_id"] == original["incident_id"]
        assert recovered["user_query"] == original["user_query"]
        assert recovered["scratchpad"] == original["scratchpad"]
        assert len(recovered["tool_calls"]) == len(original["tool_calls"])

    def test_crash_mid_investigation_resumes_where_it_stopped(self, conn):
        """The actual claim: kill the process, restart, continue.

        A run that crashed part-way must resume from its last completed node,
        keeping the work it had already done - not restart from the beginning.
        """
        from incidentiq.agent.graph import InvestigationRunner

        iid = f"IQ-CRASH-{uuid.uuid4().hex[:8]}"

        # Run until the simulated crash.
        crashed_state = None
        with InvestigationRunner(settings=self._settings(),
                                 llm=self._scripted_llm(fail_after=3)) as runner:
            with pytest.raises(RuntimeError, match="simulated process death"):
                runner.start(iid, "order-service is unhealthy")
            crashed_state = runner.get_state(iid)

        assert crashed_state is not None, "no checkpoint survived the crash"
        work_done = len(crashed_state.get("scratchpad") or [])
        assert work_done > 0, "the crash left nothing - checkpointing did not happen"

        # A brand-new runner with a working model resumes.
        with InvestigationRunner(settings=self._settings(),
                                 llm=StubProvider()) as revived:
            final = revived.resume(iid)

        assert final["status"] in ("complete", "awaiting_approval", "running")
        # The pre-crash reasoning is still present, not regenerated.
        assert final["scratchpad"][:work_done] == crashed_state["scratchpad"][:work_done], \
            "resume restarted the investigation instead of continuing it"
        assert len(final["scratchpad"]) > work_done, "resume made no further progress"

    def test_investigation_row_mirrors_the_checkpoint(self, conn):
        """LangGraph's checkpoint is the resume source of truth; the
        `investigations` table is what the API and frontend read. Both must
        exist after a run."""
        from incidentiq.agent.graph import InvestigationRunner

        iid = f"IQ-MIRROR-{uuid.uuid4().hex[:8]}"
        with InvestigationRunner(settings=self._settings(),
                                 llm=StubProvider()) as runner:
            state = runner.start(iid, "search-service returning partial results")

        row = conn.execute(
            "SELECT status, iteration, final_summary FROM investigations WHERE id = %s",
            (iid,),
        ).fetchone()
        assert row is not None, "investigation row was never written"
        assert row[0] == state["status"]
        assert row[1] == state["iteration"]

    def test_tool_calls_are_persisted_per_attempt(self, conn):
        from incidentiq.agent.graph import InvestigationRunner

        iid = f"IQ-TC-{uuid.uuid4().hex[:8]}"
        with InvestigationRunner(settings=self._settings(),
                                 llm=StubProvider()) as runner:
            state = runner.start(iid, "inventory-service errors")

        n = conn.execute(
            "SELECT count(*) FROM tool_calls WHERE investigation_id = %s", (iid,)
        ).fetchone()[0]
        assert n == len(state["tool_calls"]), \
            "persisted attempt count differs from state - the tool success metric would be wrong"


# ── The approval gate ───────────────────────────────────────
class TestApprovalGate:
    """The gate is structural, not a prompt instruction. A tool cannot be made
    to run by phrasing the request differently."""

    def _make_approval(self, conn, iid, tool, args, decision=None):
        import json
        conn.execute(
            "INSERT INTO investigations (id, user_query, status) VALUES (%s,'q','running') "
            "ON CONFLICT (id) DO NOTHING", (iid,))
        aid = f"APR-{uuid.uuid4().hex[:10]}"
        conn.execute(
            "INSERT INTO approvals (id, investigation_id, proposed_tool, "
            "proposed_arguments, reasoning, evidence, decision) "
            "VALUES (%s,%s,%s,%s,'r','{}',%s)",
            (aid, iid, tool, json.dumps(args), decision))
        return aid

    def test_unapproved_action_is_refused(self, conn):
        from incidentiq.tools.actions import execute_action
        from incidentiq.tools.base import ToolError

        iid = f"IQ-GATE-{uuid.uuid4().hex[:8]}"
        args = {"service": "checkout-service", "replica_count": 10,
                "reason": "load is high right now"}
        aid = self._make_approval(conn, iid, "scale_service", args, decision=None)

        with pytest.raises(ToolError, match="pending"):
            execute_action(conn, registry.get("scale_service"), args,
                           investigation_id=iid, approval_id=aid)

    def test_rejected_action_is_refused(self, conn):
        from incidentiq.tools.actions import execute_action
        from incidentiq.tools.base import ToolError

        iid = f"IQ-GATE-{uuid.uuid4().hex[:8]}"
        args = {"service": "checkout-service", "replica_count": 10, "reason": "load is high"}
        aid = self._make_approval(conn, iid, "scale_service", args, decision="rejected")

        with pytest.raises(ToolError, match="rejected"):
            execute_action(conn, registry.get("scale_service"), args,
                           investigation_id=iid, approval_id=aid)

    def test_approval_for_a_different_tool_is_refused(self, conn):
        """An approval authorises one specific action, not any action."""
        from incidentiq.tools.actions import execute_action
        from incidentiq.tools.base import ToolError

        iid = f"IQ-GATE-{uuid.uuid4().hex[:8]}"
        aid = self._make_approval(conn, iid, "scale_service", {}, decision="approved")

        with pytest.raises(ToolError, match="authorises"):
            execute_action(conn, registry.get("restart_service"),
                           {"service": "checkout-service", "reason": "clearing the pool"},
                           investigation_id=iid, approval_id=aid)

    def test_nonexistent_approval_is_refused(self, conn):
        from incidentiq.tools.actions import execute_action
        from incidentiq.tools.base import ToolError

        with pytest.raises(ToolError, match="no approval"):
            execute_action(conn, registry.get("scale_service"),
                           {"service": "checkout-service", "replica_count": 5,
                            "reason": "testing the guard"},
                           investigation_id="IQ-NOPE", approval_id="APR-DOESNOTEXIST")

    def test_approved_action_executes_and_is_recorded(self, conn):
        from incidentiq.tools.actions import execute_action

        iid = f"IQ-GATE-{uuid.uuid4().hex[:8]}"
        args = {"service": "checkout-service", "replica_count": 12,
                "reason": "traffic surge beyond provisioned capacity"}
        aid = self._make_approval(conn, iid, "scale_service", args, decision="approved")

        result = execute_action(conn, registry.get("scale_service"), args,
                                investigation_id=iid, approval_id=aid)
        assert result["target_replicas"] == 12
        assert result["direction"] == "up"

        n = conn.execute(
            "SELECT count(*) FROM actions WHERE approval_id = %s", (aid,)
        ).fetchone()[0]
        assert n == 1, "approved action was not recorded in the actions table"

    def test_modified_arguments_override_the_agents(self, conn):
        """An approver who edits the arguments approved *their* version. Using
        the agent's original would silently discard the correction."""
        import json

        from incidentiq.tools.actions import execute_action

        iid = f"IQ-GATE-{uuid.uuid4().hex[:8]}"
        agent_args = {"service": "checkout-service", "replica_count": 40,
                      "reason": "agent wanted a large scale-up"}
        human_args = {"service": "checkout-service", "replica_count": 12,
                      "reason": "human reduced the scale-up"}
        aid = self._make_approval(conn, iid, "scale_service", agent_args, decision="approved")
        conn.execute("UPDATE approvals SET modified_arguments = %s WHERE id = %s",
                     (json.dumps(human_args), aid))

        result = execute_action(conn, registry.get("scale_service"), agent_args,
                                investigation_id=iid, approval_id=aid)
        assert result["target_replicas"] == 12, "the human's edit was ignored"


class TestStartSemantics:
    """`start()` on an existing thread used to silently resume it.

    LangGraph keys checkpoints by thread_id, so re-running an id continues the
    previous run. That is right for recovery and wrong for a fresh
    investigation - and left implicit it produced a trace that looked like one
    run but was two, which is how a confusing eval result gets reported as a
    real one.
    """

    @staticmethod
    def _settings():
        return Settings(llm_provider="stub", max_iterations=2)

    def test_reusing_a_thread_id_raises(self, conn):
        from incidentiq.agent.graph import InvestigationRunner

        iid = f"IQ-DUP-{uuid.uuid4().hex[:8]}"
        with InvestigationRunner(settings=self._settings(), llm=StubProvider()) as r:
            r.start(iid, "checkout-service errors")
            with pytest.raises(ValueError, match="already has"):
                r.start(iid, "checkout-service errors")

    def test_reset_allows_a_clean_restart(self, conn):
        from incidentiq.agent.graph import InvestigationRunner

        iid = f"IQ-DUP-{uuid.uuid4().hex[:8]}"
        with InvestigationRunner(settings=self._settings(), llm=StubProvider()) as r:
            first = r.start(iid, "checkout-service errors")
            second = r.start(iid, "checkout-service errors", reset=True)
        # A reset run starts from zero rather than continuing the first.
        assert len(second["scratchpad"]) <= len(first["scratchpad"])
        assert second["iteration"] <= first["iteration"]

    def test_reset_clears_derived_rows(self, conn):
        from incidentiq.agent.graph import InvestigationRunner

        iid = f"IQ-DUP-{uuid.uuid4().hex[:8]}"
        with InvestigationRunner(settings=self._settings(), llm=StubProvider()) as r:
            r.start(iid, "checkout-service errors")
            n_before = conn.execute(
                "SELECT count(*) FROM tool_calls WHERE investigation_id = %s", (iid,)
            ).fetchone()[0]
            r.reset(iid)
            n_after = conn.execute(
                "SELECT count(*) FROM tool_calls WHERE investigation_id = %s", (iid,)
            ).fetchone()[0]
        assert n_before > 0 and n_after == 0


class TestPlanFallback:
    """When the model names an invalid tool, the fallback must still produce
    arguments that can actually execute.

    The original fallback reused the model's arguments, which were usually empty
    in exactly that case - so it called get_service_logs with service="", failed
    schema validation every iteration, and burned the whole budget without a
    single successful tool call.
    """

    def test_fallback_uses_known_entities_not_empty_arguments(self):
        from incidentiq.agent.nodes import NodeContext, plan

        ctx = NodeContext(conn=None, llm=StubProvider(
            script=[LLMResponse(text='{"tool_name": "not_a_real_tool", "arguments": {}}')]
        ))
        state = initial_state("IQ-1", "order-service is unhealthy")
        state["entities"] = {"service": "order-service", "time_window": "6h"}

        out = plan(state, ctx=ctx)
        assert out["next_tool"]["tool_name"] == "get_service_logs"
        assert out["next_tool"]["arguments"]["service"] == "order-service", \
            "fallback dropped the known service and would fail schema validation"

    def test_fallback_prefers_model_arguments_when_they_name_a_service(self):
        from incidentiq.agent.nodes import NodeContext, plan

        ctx = NodeContext(conn=None, llm=StubProvider(script=[LLMResponse(
            text='{"tool_name": "bogus", "arguments": {"service": "payment-service"}}')]))
        state = initial_state("IQ-1", "q")
        state["entities"] = {"service": "order-service", "time_window": "6h"}

        out = plan(state, ctx=ctx)
        assert out["next_tool"]["arguments"]["service"] == "payment-service"


class TestProposeGuards:
    """`propose` must not pass through a read-only tool as a remediation, an
    out-of-range confidence, or a citation to a document never retrieved."""

    @staticmethod
    def _run(script_text, docs=None):
        from incidentiq.agent.nodes import NodeContext, propose

        ctx = NodeContext(conn=None, llm=StubProvider(script=[LLMResponse(text=script_text)]))
        state = initial_state("IQ-1", "q")
        state["retrieved_docs"] = docs or [
            {"parent_doc_id": "INC-00315", "doc_type": "incident", "title": "t",
             "service": "order-service", "excerpt": "e", "rrf_score": 0.1,
             "vector_rank": 1, "keyword_rank": 1},
        ]
        return propose(state, ctx=ctx)["proposal"]

    def test_read_only_tool_is_not_accepted_as_a_remediation(self):
        """Observed with the local model: it proposed `get_service_logs` as the
        remediation, which would open an approval request for an action that
        changes nothing."""
        p = self._run('{"root_cause": "pool exhaustion", "remediation_tool": '
                      '"get_service_logs", "confidence": 0.8, "evidence_citations": []}')
        assert p["remediation_tool"] is None
        assert p["requires_approval"] is False

    def test_write_tool_is_accepted(self):
        p = self._run('{"root_cause": "deadlock", "remediation_tool": "restart_service", '
                      '"confidence": 0.8, "evidence_citations": []}')
        assert p["remediation_tool"] == "restart_service"
        assert p["requires_approval"] is True

    def test_hallucinated_tool_is_dropped(self):
        p = self._run('{"root_cause": "x", "remediation_tool": "reboot_the_datacenter", '
                      '"confidence": 0.5, "evidence_citations": []}')
        assert p["remediation_tool"] is None

    @pytest.mark.parametrize("raw,expected", [
        (0.9, 0.9), (5, 0.5), (90, 0.9), (1.0, 1.0), (0, 0.0), (250, 1.0),
    ])
    def test_confidence_is_normalised_into_range(self, raw, expected):
        p = self._run(f'{{"root_cause": "x", "confidence": {raw}, "evidence_citations": []}}')
        assert p["confidence"] == pytest.approx(expected, abs=0.01)

    def test_citations_to_unretrieved_documents_are_dropped(self):
        """A citation to a document never in context is a fabrication. Passing
        it through would show the approver invented evidence."""
        p = self._run('{"root_cause": "x", "confidence": 0.5, '
                      '"evidence_citations": ["INC-00315", "INC-99999"]}')
        assert p["evidence_citations"] == ["INC-00315"]
