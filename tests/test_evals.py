"""Tests for the evaluation harness itself.

An eval suite that grades incorrectly is worse than no eval suite, because it
produces numbers people believe. These tests pin the grading rules so a change
to them is deliberate and visible in a diff.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT))

from evals.run import ScenarioResult, _grade, aggregate, failure_analysis  # noqa: E402
from evals.scenario import Scenario, breakdown, load_scenarios  # noqa: E402

SCENARIO_DIR = ROOT / "evals" / "scenarios"


def _state(**kw):
    base = {
        "status": "complete", "iteration": 3, "tool_calls": [], "retrieved_docs": [],
        "proposal": None, "total_input_tokens": 0, "total_output_tokens": 0,
    }
    base.update(kw)
    return base


def _proposal(**kw):
    base = {
        "root_cause": "", "confidence": 0.5, "evidence_citations": [],
        "remediation": "", "remediation_tool": None, "remediation_arguments": None,
        "requires_approval": False, "abstained": False, "abstention_reason": None,
    }
    base.update(kw)
    return base


class TestSuiteComposition:
    def test_the_suite_has_100_scenarios(self):
        assert len(load_scenarios(SCENARIO_DIR)) == 100

    def test_the_breakdown_matches_the_specification(self):
        counts = breakdown(load_scenarios(SCENARIO_DIR))
        assert counts == {
            "single_service": 40, "cascading": 25, "no_retrieval": 15,
            "approval_required": 10, "tool_failure": 10,
        }

    def test_scenario_ids_are_unique(self):
        ids = [s.id for s in load_scenarios(SCENARIO_DIR)]
        assert len(ids) == len(set(ids))

    def test_abstention_scenarios_expect_abstention(self):
        for s in load_scenarios(SCENARIO_DIR):
            if s.kind == "no_retrieval":
                assert s.should_abstain

    def test_cascading_scenarios_name_a_cause_distinct_from_the_query(self):
        """The whole point of a cascading scenario is that the reported service
        is not the broken one."""
        for s in load_scenarios(SCENARIO_DIR):
            if s.kind == "cascading":
                assert s.true_cause_service
                assert s.victim_services
                assert s.true_cause_service not in s.victim_services

    def test_tool_failure_scenarios_actually_inject_failures(self):
        for s in load_scenarios(SCENARIO_DIR):
            if s.kind == "tool_failure":
                assert s.inject_failures and s.failure_rate > 0

    def test_approval_scenarios_expect_a_write_tool(self):
        write_tools = {"restart_service", "scale_service", "rollback_deploy"}
        for s in load_scenarios(SCENARIO_DIR):
            if s.kind == "approval_required":
                assert s.requires_approval
                assert set(s.acceptable_remediation_tools) <= write_tools


class TestGrading:
    def test_abstaining_when_it_should_is_a_pass(self):
        sc = Scenario(id="X", kind="no_retrieval", query="q", should_abstain=True)
        r = _grade(sc, _state(proposal=_proposal(abstained=True)))
        assert r.task_success and r.abstention_correct

    def test_fabricating_when_it_should_abstain_is_a_fail(self):
        """The most damaging failure mode in the suite, so it is named
        explicitly in the failure analysis rather than lumped in."""
        sc = Scenario(id="X", kind="no_retrieval", query="q", should_abstain=True)
        r = _grade(sc, _state(proposal=_proposal(
            root_cause="the database was overloaded", abstained=False)))
        assert not r.task_success
        assert "fabricated" in r.failure_note

    def test_abstaining_on_a_solvable_scenario_is_a_fail(self):
        sc = Scenario(id="X", kind="single_service", query="q",
                      true_cause_service="checkout-service")
        r = _grade(sc, _state(proposal=_proposal(abstained=True)))
        assert not r.task_success
        assert "abstained" in r.failure_note

    def test_naming_the_cause_and_mechanism_passes(self):
        sc = Scenario(id="X", kind="single_service", query="q",
                      true_cause_service="checkout-service",
                      root_cause_keywords=["connection", "exhausted"],
                      acceptable_remediation_tools=["restart_service"])
        r = _grade(sc, _state(proposal=_proposal(
            root_cause="checkout-service had its connection pool exhausted",
            remediation_tool="restart_service")))
        assert r.task_success
        assert r.identified_cause and r.matched_keywords == 2

    def test_right_service_wrong_mechanism_fails(self):
        sc = Scenario(id="X", kind="single_service", query="q",
                      true_cause_service="checkout-service",
                      root_cause_keywords=["connection", "exhausted", "pooling",
                                           "saturated", "timeout"])
        r = _grade(sc, _state(proposal=_proposal(
            root_cause="checkout-service is unwell for unclear reasons")))
        assert not r.task_success
        assert "mechanism" in r.failure_note

    def test_blaming_the_victim_is_called_out_specifically(self):
        """A cascading scenario's characteristic wrong answer. Reporting it as a
        generic miss would hide the pattern the suite exists to detect."""
        sc = Scenario(id="X", kind="cascading", query="q",
                      true_cause_service="payment-db",
                      victim_services=["checkout-service"],
                      root_cause_keywords=["replication"])
        r = _grade(sc, _state(proposal=_proposal(
            root_cause="checkout-service is overloaded and should be restarted")))
        assert not r.task_success
        assert "victim" in r.failure_note

    def test_wrong_remediation_tool_fails(self):
        sc = Scenario(id="X", kind="single_service", query="q",
                      true_cause_service="checkout-service",
                      root_cause_keywords=["deadlock"],
                      acceptable_remediation_tools=["restart_service"])
        r = _grade(sc, _state(proposal=_proposal(
            root_cause="checkout-service hit a deadlock", remediation_tool="scale_service")))
        assert not r.task_success
        assert "scale_service" in r.failure_note

    def test_tool_success_counts_every_attempt(self):
        sc = Scenario(id="X", kind="single_service", query="q")
        r = _grade(sc, _state(tool_calls=[
            {"tool_name": "get_service_logs", "status": "timeout", "injected_failure": True},
            {"tool_name": "get_service_logs", "status": "timeout", "injected_failure": True},
            {"tool_name": "get_service_logs", "status": "success", "injected_failure": False},
        ], proposal=_proposal()))
        assert r.tool_attempts == 3 and r.tool_successes == 1

    def test_recovery_is_detected(self):
        sc = Scenario(id="X", kind="tool_failure", query="q")
        r = _grade(sc, _state(tool_calls=[
            {"tool_name": "get_metrics", "status": "timeout", "injected_failure": True},
            {"tool_name": "get_metrics", "status": "success", "injected_failure": False},
        ], proposal=_proposal()))
        assert r.recovered_from_failure

    def test_no_recovery_when_a_tool_never_succeeded(self):
        sc = Scenario(id="X", kind="tool_failure", query="q")
        r = _grade(sc, _state(tool_calls=[
            {"tool_name": "get_metrics", "status": "timeout", "injected_failure": True},
            {"tool_name": "get_metrics", "status": "timeout", "injected_failure": True},
        ], proposal=_proposal()))
        assert not r.recovered_from_failure

    def test_partial_counts_as_a_tool_success(self):
        sc = Scenario(id="X", kind="single_service", query="q")
        r = _grade(sc, _state(tool_calls=[
            {"tool_name": "get_service_logs", "status": "partial", "injected_failure": True},
        ], proposal=_proposal()))
        assert r.tool_successes == 1


class TestAggregation:
    def _settings(self):
        from incidentiq.config import Settings
        return Settings(llm_provider="stub")

    def test_aggregate_computes_the_headline_metrics(self):
        results = [
            ScenarioResult(scenario_id="a", kind="single_service", query="q",
                           task_success=True, tool_attempts=4, tool_successes=4,
                           latency_s=10.0),
            ScenarioResult(scenario_id="b", kind="single_service", query="q",
                           task_success=False, tool_attempts=6, tool_successes=3,
                           latency_s=20.0),
        ]
        s = aggregate(results, self._settings())
        assert s["task_completion"] == 0.5
        assert s["tool_success_rate"] == pytest.approx(7 / 10)
        assert s["median_latency_s"] == 15.0

    def test_grounded_rate_ignores_unjudged_scenarios(self):
        """Scenarios run with --no-judge must not be counted as ungrounded."""
        results = [
            ScenarioResult(scenario_id="a", kind="x", query="q", fully_grounded=True),
            ScenarioResult(scenario_id="b", kind="x", query="q", fully_grounded=None),
        ]
        s = aggregate(results, self._settings())
        assert s["grounded_response_rate"] == 1.0

    def test_by_kind_breakdown_is_present(self):
        results = [
            ScenarioResult(scenario_id="a", kind="cascading", query="q", task_success=True),
            ScenarioResult(scenario_id="b", kind="no_retrieval", query="q", task_success=False),
        ]
        s = aggregate(results, self._settings())
        assert set(s["by_kind"]) == {"cascading", "no_retrieval"}
        assert s["by_kind"]["cascading"]["task_completion"] == 1.0

    def test_failure_analysis_groups_by_cause(self):
        results = [
            ScenarioResult(scenario_id="a", kind="x", query="q",
                           failure_note="did not name the causing service"),
            ScenarioResult(scenario_id="b", kind="x", query="q",
                           failure_note="did not name the causing service"),
            ScenarioResult(scenario_id="c", kind="x", query="q",
                           failure_note="blamed the victim service instead of the cause"),
            ScenarioResult(scenario_id="d", kind="x", query="q", task_success=True),
        ]
        counts = failure_analysis(results)
        assert counts["did not name the causing service"] == 2
        assert sum(counts.values()) == 3   # the passing one is excluded
