"""The evaluation harness: the grading rules, pinned so that a change to them is
deliberate and visible in a diff."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "seeds"))

from evals.run import ScenarioResult, _grade, aggregate, failure_analysis  # noqa: E402
from evals.scenario import Scenario, breakdown, load_scenarios  # noqa: E402
from evals.words import keyword_hits  # noqa: E402

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
        assert r.matched_keywords == 0
        assert "failure mode" in r.failure_note

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


class TestPercentiles:
    def _settings(self):
        from incidentiq.config import Settings
        return Settings(llm_provider="stub")

    def _results(self, latencies):
        return [ScenarioResult(scenario_id=str(i), kind="x", query="q", latency_s=v)
                for i, v in enumerate(latencies)]

    def test_p95_on_a_small_sample_does_not_understate_the_tail(self):
        """int(0.95*4)-1 picks the third of four values. With small samples the
        tail is exactly what you are trying to see."""
        s = aggregate(self._results([60.0, 95.0, 140.0, 180.0]), self._settings())
        assert s["p95_latency_s"] == 180.0

    def test_p95_on_a_full_run(self):
        s = aggregate(self._results([float(i) for i in range(1, 101)]), self._settings())
        assert s["p95_latency_s"] == 95.0

    def test_p95_never_exceeds_the_maximum(self):
        for n in range(1, 40):
            s = aggregate(self._results([float(i) for i in range(n)]), self._settings())
            assert s["p95_latency_s"] <= float(n - 1)

    def test_single_scenario_p95_is_that_scenario(self):
        s = aggregate(self._results([42.0]), self._settings())
        assert s["p95_latency_s"] == 42.0 and s["median_latency_s"] == 42.0


class TestArchetypeVariantGrading:
    """Each archetype has two narratives for the same failure. Grading against
    the one a scenario happened to draw scored a correct diagnosis at zero.

    Observed: `disk_full` on catalog-db. The scenario drew "autovacuum could not
    keep up and dead tuples filled the volume"; the agent diagnosed disk
    exhaustion on the right service and described the other variant, "WAL
    segments accumulated because archiving was failing". Both are in the corpus
    and both are right. It matched 0 of 5 keywords.

    Ground truth is now the archetype, which is determinate, rather than the
    narrative, which is not.
    """

    def test_every_narrative_is_graded_as_its_own_failure_mode_and_no_other(self):
        """The grader's own accuracy, on the narratives the corpus is built from:
        with keywords taken from the first narrative only, 4 of 32 failed as
        their own archetype and 37 of 480 passed as someone else's."""
        import archetypes as A

        from evals import mechanism

        def render(text):
            return (text.replace("{dep}", "orders-db").replace("{service}", "checkout-service")
                        .replace("{version}", "v2.3.1").replace("{team}", "payments"))

        for arch in A.ARCHETYPES:
            for narrative in arch.root_cause:
                text = render(narrative)
                assert mechanism.names(arch.key, text), f"{arch.key} failed: {text[:70]}"
                others = [k for k in mechanism.VOCABULARY
                          if k != arch.key and mechanism.names(k, text)]
                assert not others, f"{arch.key} narrative also graded as {others}"

    def test_recorded_answers_are_graded_as_what_they_describe(self):
        """Conclusions recorded on a replay of the 2026-10-12 baseline."""
        from evals import mechanism

        pool = "connection pool exhaustion, source not yet isolated"
        assert mechanism.names("db_pool_exhausted", pool)
        # Run C: the agent examined the database and found the full disk; "disk" is
        # a tag of the archetype, four letters long.
        for disk in ["The user-db service is experiencing high disk usage and disk space",
                     "payment-db became slow due to disk exhaustion, causing timeouts"]:
            assert mechanism.names("disk_full", disk), disk
        cascade = "downstream dependency timeout cascading upstream on user-service"
        assert not mechanism.names("disk_full", cascade)
        assert mechanism.closest(cascade) == "downstream_timeout_cascade"

    def test_a_wrong_failure_mode_is_named_in_the_failure_note(self):
        sc = next(s for s in load_scenarios(SCENARIO_DIR) if s.id == "SC-CA-001")
        r = _grade(sc, _state(proposal=_proposal(
            root_cause="downstream dependency timeout cascading upstream on user-service",
            remediation_tool="restart_service",
            remediation_arguments={"service": sc.true_cause_service})))
        assert not r.task_success
        assert r.failure_note.startswith("named the service but not the failure mode "
                                         "(described downstream_timeout_cascade")

    def test_keywords_are_pooled_not_drawn_from_one_variant(self):
        """The pool must be big enough to cover both narratives."""
        for s in load_scenarios(SCENARIO_DIR):
            if s.expected_archetype and s.root_cause_keywords:
                assert len(s.root_cause_keywords) >= 6, (
                    f"{s.id} has only {len(s.root_cause_keywords)} keywords - "
                    "that looks like one variant rather than a pooled vocabulary"
                )

    def test_an_unrelated_answer_still_fails(self):
        from evals import mechanism

        wrong = "the TLS certificate expired and the load balancer rejected connections"
        assert not mechanism.names("disk_full", wrong)

    def test_other_forms_of_a_keyword_count(self):
        assert keyword_hits(["exhausted", "connection-pool"], "connection pool exhaustion") == 2
        assert keyword_hits(["saturation", "provisioned"], "saturated, under-provisioning") == 2

    @pytest.mark.parametrize("text", [
        "user-db disk is healthy",
        "the recommendation-service connection pool is healthy; restart it to be safe",
        "the connection pool is not exhausted",
        "no sign of disk pressure on user-db",
        "we checked the disk on user-db and the connection pool settings",
        "memory looked normal and the heap is stable",
        "disk_full is ruled out",
    ])
    def test_a_healthy_denied_or_merely_mentioned_mechanism_names_no_failure(self, text):
        from evals import mechanism
        assert mechanism.closest(text) is None, mechanism.asserted(text)

    def test_a_denied_mechanism_does_not_hide_the_one_asserted_beside_it(self):
        from evals import mechanism
        text = "The disk is healthy, but the connection pool is exhausted under load."
        assert mechanism.closest(text) == "db_pool_exhausted"

    @pytest.mark.parametrize("archetype,text,named", [
        ("disk_full", "user-db has no disk pressure", False),
        ("db_pool_exhausted", "No connection pool exhaustion on user-db", False),
        ("disk_full", "The disk on user-db is full and cpu usage is normal", True),
        ("disk_full", "cpu usage is normal and the disk on user-db is full", True),
        ("db_pool_exhausted", "the connection pool is exhausted, with no disk pressure", True),
        ("disk_full", "the connection pool is exhausted, with no disk pressure", False),
        ("disk_full", "writes to the disk failed with no space left on device", True),
    ])
    def test_a_denial_covers_its_own_part_of_the_sentence(self, archetype, text, named):
        from evals import mechanism
        assert mechanism.names(archetype, text) is named, mechanism.asserted(text)

    @pytest.mark.parametrize("archetype,text,named", [
        # Written apart from the corpus: the right failure in other words, and answers
        # that name the component without asserting its failure mode.
        ("db_pool_exhausted", "orders-db ran out of database connections; requests waited "
                              "for a free one", True),
        ("db_pool_exhausted", "the pool of connections to payments-db was saturated by "
                              "long-running queries", True),
        ("db_pool_exhausted", "the connection pool settings on orders-db look correct; "
                              "latency came from a slow query", False),
        ("memory_leak_oom", "checkout-service was OOM-killed repeatedly as its heap kept "
                            "growing", True),
        ("memory_leak_oom", "memory usage on checkout-service is steady at 40%", False),
        ("memory_leak_oom", "memory pressure on redis caused evictions", False),
        ("bad_deploy_regression", "the v4.2.0 release of cart-service introduced a "
                                  "serialization bug", True),
        ("bad_deploy_regression", "cart-service was deployed yesterday with no changes to "
                                  "the failing path", False),
        ("downstream_timeout_cascade", "search-service timed out because its downstream "
                                       "inventory-db was slow, and it cascaded upstream", True),
        ("cache_stampede", "after the cache was flushed, every request missed and hit the "
                           "database at once", True),
        ("cache_stampede", "the cache hit rate is normal", False),
        ("disk_full", "the data volume of payment-db filled up with WAL files", True),
        # Recorded answers (phase4 A and B, C):
        ("disk_full", "payment-db became slow due to disk space issues, causing timeouts in "
                      "payment-service", True),
        ("downstream_timeout_cascade", "Degradation in user-db causing timeouts and circuit "
                                       "breaker OPEN in user-service", True),
        ("disk_full", "payment-db disk is almost empty", False),
        ("disk_full", "Disk I/O on payment-db has high latency", False),
        ("disk_full", "payment-db has a disk hardware failure", False),
        ("replica_lag", "reads came from a replica that was minutes behind the primary", True),
        ("replica_lag", "the replica is in sync with the primary", False),
        ("cert_expiry", "clients failed the TLS handshake because the server certificate "
                        "had expired", True),
        ("cert_expiry", "the TLS certificate is valid until 2027", False),
        ("upstream_rate_limit", "the payment provider returned 429 Too Many Requests", True),
        ("upstream_rate_limit", "requests to the provider are well under its rate limit", False),
        ("consumer_lag", "the order-events consumers fell behind and a backlog of messages "
                         "built up", True),
        ("consumer_lag", "the consumer group is healthy", False),
        ("thread_pool_saturation", "all worker threads were stuck waiting on a lock, so the "
                                   "thread pool was exhausted", True),
        ("thread_pool_saturation", "the thread count is normal", False),
        ("config_drift", "the canary pods were missing an environment variable the other "
                         "pods had", True),
        ("config_drift", "the config for search-service was reviewed and is correct", False),
        ("search_shard_unassigned", "two primary shards were unassigned, so queries returned "
                                    "partial results", True),
        ("search_shard_unassigned", "the search cluster status is green", False),
        ("traffic_surge", "a marketing campaign tripled traffic beyond what checkout-service "
                          "was provisioned for", True),
        ("traffic_surge", "traffic to checkout-service was normal for a Tuesday", False),
        ("cache_node_eviction", "redis hit its maxmemory and started evicting frequently used "
                                "keys", True),
        ("cache_node_eviction", "redis memory use is stable", False),
        ("broker_partition_offline", "a kafka broker went down and its partitions were left "
                                     "without a leader", True),
        ("broker_partition_offline", "all brokers are online", False),
    ])
    def test_answers_written_apart_from_the_corpus(self, archetype, text, named):
        from evals import mechanism
        assert mechanism.names(archetype, text) is named, (mechanism.hits(text),
                                                           mechanism.asserted(text))

    @pytest.mark.parametrize("root_cause,success", [
        ("payment-db has no disk pressure", False),
        ("payment-db disk is almost empty", False),
        ("Disk I/O on payment-db has high latency", False),
        ("payment-db has a disk hardware failure", False),
        ("The disk on payment-db is full and cpu usage is normal", True),
    ])
    def test_denials_are_graded_through_the_whole_grader(self, root_cause, success):
        sc = next(s for s in load_scenarios(SCENARIO_DIR) if s.id == "SC-CA-002")
        r = _grade(sc, _state(proposal=_proposal(
            root_cause=root_cause, remediation_tool="restart_service",
            remediation_arguments={"service": "payment-db"})))
        assert r.task_success is success, r.failure_note

    @pytest.mark.parametrize("arguments", [None, {"service": "payment-db"}])
    @pytest.mark.parametrize("root_cause,success", [
        ("payment-db is healthy; payment-service has a full disk.", False),
        ("payment-db is unwell and payment-service has a full disk", False),
        ("payment-db does not have a full disk", False),
        ("payment-db is not running out of disk space", False),
        ("payment-db's disk is full, and payment-service times out waiting on it", True),
        ("payment-service is timing out because payment-db ran out of disk space", True),
        ("payment-db is slow because payment-service has a full disk", False),
        ("payment-service is slow because payment-db has a full disk", True),
        ("payment-db has neither a full disk nor memory pressure", False),
        ("payment-db is slow, but not due to disk exhaustion", False),
        ("payment-db is slow, not because its disk is full", False),
        ("payment-db is slow because payment-service is struggling with a full disk", False),
        ("payment-service is struggling because payment-db is slow with a full disk", True),
        ("payment-db is slow due to disk exhaustion, not because of a deploy", True),
        ("payment-db's disk is full; there is neither memory pressure nor cpu load", True),
        # A coordinated part without a service of its own is about the one before it.
        ("payment-db is slow because payment-service has high disk usage and disk space "
         "issues", False),
        ("payment-db is slow because payment-service has high disk usage, disk space issues",
         False),
        ("payment-db is slow because payment-service has high disk usage, and disk space "
         "issues", False),
        ("payment-db is slow because payment-service has high disk usage with disk space "
         "issues", False),
        ("payment-service is slow because payment-db has high disk usage and disk space "
         "issues", True),
        ("payment-service times out, and payment-db is out of disk space", True),
        ("payment-db is slow, and its disk is full", True),
        ("payment-service is slow because the disk is full and payment-db is stuck", True),
    ])
    def test_the_failure_must_be_asserted_of_the_service_at_fault(self, arguments, root_cause,
                                                                  success):
        sc = next(s for s in load_scenarios(SCENARIO_DIR) if s.id == "SC-CA-002")
        r = _grade(sc, _state(proposal=_proposal(
            root_cause=root_cause, remediation_tool="restart_service",
            remediation_arguments=arguments)))
        assert r.task_success is success, r.failure_note

    @pytest.mark.parametrize("scenario", [s for s in load_scenarios(SCENARIO_DIR)
                                          if not s.should_abstain], ids=lambda s: s.id)
    def test_every_scenarios_own_expected_answer_passes(self, scenario):
        """Its expected root cause, acting on the service at fault with an accepted tool."""
        from evals.run import _cause
        tools = scenario.acceptable_remediation_tools
        r = _grade(scenario, _state(proposal=_proposal(
            root_cause=scenario.expected_root_cause,
            remediation_tool=tools[0] if tools else None,
            remediation_arguments={"service": _cause(scenario)})))
        assert r.task_success, r.failure_note

    @pytest.mark.parametrize("scenario", [s for s in load_scenarios(SCENARIO_DIR)
                                          if not s.should_abstain], ids=lambda s: s.id)
    def test_every_expected_answer_blaming_another_service_fails(self, scenario):
        """The same answer with another service named where the failure is (or, if it
        names none, said to be on another service), acting on the right one."""
        import re

        from evals.run import SERVICE_NAMES, _cause, _mentions
        cause, text = _cause(scenario), scenario.expected_root_cause
        avoid = {cause, (scenario.expected_service or "").lower(),
                 *(v.lower() for v in scenario.victim_services)}
        wrong = next(s for s in sorted(SERVICE_NAMES)
                     if s not in avoid and not _mentions(text.lower(), s))
        blamed = (re.sub(rf"(?<![\w-]){re.escape(cause)}(?![\w-])", wrong, text, flags=re.I)
                  if _mentions(text.lower(), cause) else f"On {wrong}, {text[0].lower()}{text[1:]}")
        tools = scenario.acceptable_remediation_tools
        r = _grade(scenario, _state(proposal=_proposal(
            root_cause=blamed, remediation_tool=tools[0] if tools else None,
            remediation_arguments={"service": cause})))
        assert not r.task_success, blamed

    @pytest.mark.parametrize("root_cause", [
        "The connection pool on user-service is exhausted.",
        "The connection pool in user-service is exhausted.",
        "The connection pool of user-service is exhausted.",
        "user-service's connections to user-db leaked and were never returned to the pool",
    ])
    def test_the_holder_of_the_pool_named_any_way_is_the_service_at_fault(self, root_cause):
        from evals.run import _cause
        sc = next(s for s in load_scenarios(SCENARIO_DIR) if s.id == "SC-SS-004")
        r = _grade(sc, _state(proposal=_proposal(
            root_cause=root_cause, remediation_tool=sc.acceptable_remediation_tools[0],
            remediation_arguments={"service": _cause(sc)})))
        assert r.task_success, r.failure_note

    @pytest.mark.parametrize("scenario_id,root_cause", [
        # An application named with on, in or of holds the pool; only a store is where
        # connections go.
        ("SC-SS-004", "The connection pool on payment-service is exhausted."),
        ("SC-SS-004", "The connection pool in payment-service is exhausted."),
        ("SC-SS-004", "The connection pool of payment-service is exhausted."),
        # The holder named with a possessive keeps the failure.
        ("SC-SS-004", "payment-service's connections leaked and were never returned to the "
                      "pool"),
        # A sentence that goes on without naming a service is about the one before it.
        ("SC-CA-001", "The data volume on payment-service reached 100%. WAL accumulated "
                      "because archiving failed."),
        # The connections go to user-db, but payment-service holds and leaks them.
        ("SC-SS-004", "payment-service leaked its user-db connections, which were never "
                      "returned to the pool"),
        # The retries that amplified the load were user-db's, not user-service's.
        ("SC-TF-010", "user-db retried failed calls three times with no backoff, tripling "
                      "load on an already-struggling dependency"),
    ])
    def test_the_failure_stays_with_the_service_it_is_spread_over(self, scenario_id,
                                                                    root_cause):
        from evals.run import _cause
        sc = next(s for s in load_scenarios(SCENARIO_DIR) if s.id == scenario_id)
        r = _grade(sc, _state(proposal=_proposal(
            root_cause=root_cause, remediation_tool=(sc.acceptable_remediation_tools or [None])[0],
            remediation_arguments={"service": _cause(sc)})))
        assert not r.task_success and "failure mode" in r.failure_note

    def test_a_failure_spread_over_two_services_belongs_to_the_one_at_fault(self):
        """A timeout cascade is the upstream service's failure (no timeout, a breaker),
        started by a slow dependency; the part naming the service at fault carries no
        failure word of its own ("circuit breaker OPEN")."""
        sc = next(s for s in load_scenarios(SCENARIO_DIR) if s.id == "SC-TF-001")
        r = _grade(sc, _state(proposal=_proposal(
            root_cause="Degradation in user-db causing timeouts and circuit breaker OPEN in "
                       "user-service", remediation_tool="restart_service",
            remediation_arguments={"service": "user-service"})))
        assert r.failure_note == "" or "failure mode" not in r.failure_note

    def test_a_regrade_grades_the_service_again_too(self, tmp_path):
        import json

        from evals.run import ScenarioResult, regrade
        text = "payment-db is healthy; payment-service has a full disk."
        saved = ScenarioResult(scenario_id="SC-CA-002", kind="cascading", query="q",
                               completed=True, task_success=True, identified_cause=True,
                               correct_remediation=True, graded_text=text, root_cause=text)
        from incidentiq.config import Settings

        from evals.run import aggregate
        path = tmp_path / "run.json"
        path.write_text(json.dumps({"summary": aggregate([saved], Settings(llm_provider="stub")),
                                    "results": [saved.__dict__]}))
        old, new = regrade(path)
        result = json.loads(path.read_text())["results"][0]
        assert (old["task_completion"], new["task_completion"]) == (1.0, 0.0)
        assert result["identified_cause"] is False

    def test_advice_against_another_failure_mode_does_not_undo_the_diagnosis(self):
        """The failure mode is read from the root cause: advice in the remediation and
        summary (a timeout, a circuit breaker) is in another failure mode's words."""
        sc = next(s for s in load_scenarios(SCENARIO_DIR) if s.id == "SC-CA-002")
        r = _grade(sc, _state(
            proposal=_proposal(
                root_cause=("payment-db became slow due to disk exhaustion, causing "
                            "timeouts in payment-service"),
                remediation=("Free up disk space on payment-db and add a timeout and "
                             "circuit breaker on payment-service's calls to payment-db to "
                             "prevent future outages."),
                remediation_tool="restart_service",
                remediation_arguments={"service": "payment-db", "reason": "disk exhaustion"}),
            final_summary=(
                "The investigation concluded that payment-db became slow due to disk "
                "exhaustion, causing timeouts in payment-service. Evidence from logs and "
                "metrics indicated 'no space left on device' errors on payment-db, which "
                "coincided with timeouts observed in payment-service logs. The disk space "
                "issue was resolved by restarting payment-db, and a timeout and circuit "
                "breaker were proposed to prevent future outages.\n\nStill uncertain: "
                "whether the timeout and circuit breaker implementation will fully prevent "
                "future outages without impacting normal operations")))
        assert r.task_success, r.failure_note

    def test_a_keyword_must_be_a_whole_word_of_the_text(self):
        # Short tags inside longer words were read as a diagnosis.
        assert keyword_hits(["oom"], "the service has ample room") == 0
        assert keyword_hits(["wal"], "the worker walked away") == 0
        assert keyword_hits(["oom", "wal"], "OOM kills, then the WAL filled the disk") == 2
        assert keyword_hits(["connection pool"], "all connection-pools were busy") == 1
        assert keyword_hits(["connection pool"], "a connection from the pool") == 0

    def test_words_that_only_start_alike_do_not_count(self):
        assert keyword_hits(["capacity", "failing", "timeout"], "capable failure time") == 0
        assert keyword_hits(["configured"], "the configuration") == 0


class TestScenarioFixtures:
    """Scenarios must carry the infrastructure state their failure would leave.

    Without this most scenarios described a failure that left no trace anywhere
    the diagnostic tools could look, and the eval measured whether the agent
    could guess the archetype from retrieval alone. The symptom was an agent
    doing the right thing and being marked wrong: asked about a memory leak on
    user-service it found no ERROR logs, reasoned it was probably waiting on a
    slow dependency, and said so - correctly noting it had no evidence.
    """

    @pytest.fixture(scope="class")
    def conn(self):
        import psycopg
        from incidentiq.config import get_settings
        try:
            c = psycopg.connect(get_settings().database_url, autocommit=True, connect_timeout=3)
        except psycopg.OperationalError:
            pytest.skip("Postgres not reachable")
        yield c
        c.close()

    def _scenario(self, kind):
        return next(s for s in load_scenarios(SCENARIO_DIR) if s.kind == kind)

    def test_fixture_creates_matching_error_signatures(self, conn):
        from evals import fixtures
        sc = self._scenario("single_service")
        svc = sc.true_cause_service
        before = conn.execute(
            "SELECT count(*) FROM log_entries WHERE service=%s AND level IN ('ERROR','FATAL')",
            (svc,)).fetchone()[0]
        state = fixtures.apply(conn, sc)
        try:
            after = conn.execute(
                "SELECT count(*) FROM log_entries WHERE service=%s "
                "AND level IN ('ERROR','FATAL')", (svc,)).fetchone()[0]
            assert after > before, "no error evidence was planted"
        finally:
            fixtures.clear(conn, state)

    def test_cleanup_restores_the_base_corpus_exactly(self, conn):
        """Deleting by predicate would take the base corpus with it, and the
        damage would only surface as a later scenario mysteriously having no
        evidence."""
        from evals import fixtures
        sc = self._scenario("single_service")
        svc = sc.true_cause_service
        before = conn.execute(
            "SELECT count(*) FROM log_entries WHERE service=%s", (svc,)).fetchone()[0]
        state = fixtures.apply(conn, sc)
        fixtures.clear(conn, state)
        after = conn.execute(
            "SELECT count(*) FROM log_entries WHERE service=%s", (svc,)).fetchone()[0]
        assert after == before

    def test_metric_anomaly_is_planted(self, conn):
        import archetypes as A

        from evals import fixtures
        sc = self._scenario("single_service")
        arch = A.BY_KEY[sc.expected_archetype]
        before = dict(conn.execute(
            "SELECT id, value FROM metric_points WHERE service = %s AND metric_name = %s",
            (sc.true_cause_service, arch.metric)).fetchall())
        state = fixtures.apply(conn, sc)
        try:
            ids = state.metric_ids + [i for i, _ in state.scaled_metrics]
            rows = conn.execute(
                "SELECT value FROM metric_points WHERE id = ANY(%s)", (ids,)).fetchall()
            assert rows, f"no {arch.metric} anomaly planted"
            values = [r[0] for r in rows]
            assert max(values) != min(values), "the planted metric is flat"
        finally:
            fixtures.clear(conn, state)
        after = dict(conn.execute(
            "SELECT id, value FROM metric_points WHERE service = %s AND metric_name = %s",
            (sc.true_cause_service, arch.metric)).fetchall())
        assert after == before, "clearing must restore the corpus readings exactly"

    def test_abstention_scenarios_get_no_evidence(self, conn):
        """Their whole point is that there is nothing to find."""
        from evals import fixtures
        sc = self._scenario("no_retrieval")
        state = fixtures.apply(conn, sc)
        assert not state.log_ids and not state.metric_ids
        fixtures.clear(conn, state)

    def test_cascading_scenarios_plant_the_victim_symptoms(self, conn):
        """Without symptoms on the reported service, a cascading scenario gives
        the agent no reason to look downstream."""
        from evals import fixtures
        sc = self._scenario("cascading")
        state = fixtures.apply(conn, sc)
        try:
            services = {r[0] for r in conn.execute(
                "SELECT DISTINCT service FROM log_entries WHERE id = ANY(%s)",
                (state.log_ids,)).fetchall()}
            assert sc.true_cause_service in services
            assert services & set(sc.victim_services), "no victim symptoms planted"
        finally:
            fixtures.clear(conn, state)

    def test_orphaned_fixtures_are_tracked_and_purgeable(self, conn):
        """Cleanup state must survive an unclean exit.

        It did not: `kill -9` on an eval run skipped the finally block and
        orphaned 137 log rows, 1 deploy row, and ~153 metric rows. The metric
        rows had no marker at all, so they could not be told apart from the base
        corpus. Leaked evidence for one service becomes background noise for
        every later scenario, and the numbers drift with nothing failing.
        """
        import uuid as _uuid

        from evals import fixtures
        sc = self._scenario("single_service")
        run_id = f"test-orphan-{_uuid.uuid4().hex[:8]}"

        # Establish a clean baseline first. Any fixture rows left by an earlier
        # killed run would otherwise shift the counts this test compares, and
        # the test would fail for a reason that has nothing to do with what it
        # is checking.
        fixtures.purge_orphans(conn, only_run_id=run_id)
        baseline_logs = conn.execute("SELECT count(*) FROM log_entries").fetchone()[0]
        totals = "SELECT count(*), round(sum(value)::numeric, 3) FROM metric_points"
        baseline_metrics = conn.execute(totals).fetchone()

        try:
            # Simulate a run that dies before its cleanup: apply, then drop the
            # in-process handle entirely.
            state = fixtures.apply(conn, sc, run_id=run_id)
            assert state.log_ids and (state.metric_ids or state.scaled_metrics)
            tracked = conn.execute(
                "SELECT count(*) FROM eval_fixture_rows WHERE run_id = %s", (run_id,)
            ).fetchone()[0]
            assert tracked == (len(state.log_ids) + len(state.metric_ids)
                               + len(state.scaled_metrics) + len(state.deploy_ids))
            del state

            # A later run purges what the dead one left, from the database
            # alone. Scoped to a different owner, so this cannot disturb a live
            # eval - the unscoped form deleted a running eval's fixtures
            # mid-scenario when the suite happened to run alongside one.
            purged = fixtures.purge_orphans(conn, only_run_id=run_id)
            assert purged, "nothing was purged"
            assert conn.execute(
                "SELECT count(*) FROM log_entries").fetchone()[0] == baseline_logs
            # Same rows and the same readings: scaled ones are restored.
            assert conn.execute(totals).fetchone() == baseline_metrics
            assert conn.execute(
                "SELECT count(*) FROM eval_fixture_rows WHERE run_id = %s", (run_id,)
            ).fetchone()[0] == 0
        finally:
            # The test plants real rows, so it must clean up even when it fails.
            # Without this a failing run leaves its fixtures behind and the next
            # run sees double - which is exactly how this test first failed.
            fixtures.purge_orphans(conn, only_run_id=run_id)

    def test_clear_removes_its_own_tracking_rows(self, conn):
        from evals import fixtures
        sc = self._scenario("single_service")
        state = fixtures.apply(conn, sc, run_id="test-clear")
        fixtures.clear(conn, state)
        left = conn.execute(
            "SELECT count(*) FROM eval_fixture_rows WHERE run_id = 'test-clear'"
        ).fetchone()[0]
        assert left == 0, "clear left tracking rows behind, so a later purge would re-delete"


class TestGradingUsesTheWholeConclusion:
    """Which field a model puts the service name in is not something the agent
    should be scored on.

    Observed: the agent concluded "the root cause of the catalog-db incident is
    disk exhaustion, with high confidence", named the service in its summary and
    in the remediation arguments, and did not repeat it inside the root_cause
    string. Graded "did not name the causing service".
    """

    def test_service_named_only_in_the_summary_counts(self):
        sc = Scenario(id="X", kind="single_service", query="q",
                      true_cause_service="catalog-db",
                      root_cause_keywords=["disk", "volume"],
                      acceptable_remediation_tools=[])
        r = _grade(sc, _state(
            proposal=_proposal(root_cause="disk exhaustion filled the volume"),
            final_summary="The root cause of the catalog-db incident is disk exhaustion.",
        ))
        assert r.identified_cause, "service named in the summary was not counted"
        assert r.task_success

    def test_service_named_only_in_remediation_arguments_counts(self):
        sc = Scenario(id="X", kind="single_service", query="q",
                      true_cause_service="catalog-db",
                      root_cause_keywords=["disk"],
                      acceptable_remediation_tools=["scale_service"])
        r = _grade(sc, _state(proposal=_proposal(
            root_cause="the disk filled up",
            remediation_tool="scale_service",
            remediation_arguments={"service": "catalog-db", "replica_count": 4},
        )))
        assert r.identified_cause
        assert r.task_success

    def test_naming_the_wrong_service_everywhere_still_fails(self):
        """Widening the text must not make everything pass."""
        sc = Scenario(id="X", kind="single_service", query="q",
                      true_cause_service="catalog-db",
                      root_cause_keywords=["disk"])
        r = _grade(sc, _state(
            proposal=_proposal(root_cause="payment-service ran out of disk",
                               remediation_arguments={"service": "payment-service"}),
            final_summary="payment-service was the problem.",
        ))
        assert not r.identified_cause
        assert not r.task_success


class TestFailedActionGrading:
    def test_a_failed_action_is_not_task_success(self):
        """However good the diagnosis, an investigation whose approved action
        did not execute has not completed its task."""
        sc = Scenario(id="X", kind="approval_required", query="q",
                      true_cause_service="user-service",
                      root_cause_keywords=["memory", "leak"],
                      acceptable_remediation_tools=["rollback_deploy"],
                      requires_approval=True)
        state = _state(
            status="failed",
            failure_reason="approved action failed: 'v3.11.4' was never deployed",
            proposal=_proposal(root_cause="user-service memory leak from a recent deploy",
                               remediation_tool="rollback_deploy"),
        )
        r = _grade(sc, state)
        assert not r.completed
        assert not r.task_success
        assert "failed to execute" in r.failure_note


class TestVictimDetection:
    """Mentioning a service is not diagnosing it: naming or acting on a victim is
    reported as such."""

    def test_acting_on_the_victim_is_named_as_such(self):
        sc = Scenario(id="X", kind="cascading", query="q",
                      true_cause_service="order-db",
                      victim_services=["order-service", "api-gateway"],
                      root_cause_keywords=["replication", "lag"],
                      acceptable_remediation_tools=["restart_service"])
        r = _grade(sc, _state(proposal=_proposal(
            root_cause="thread pool saturation in order-service, involving the order-db",
            remediation_tool="restart_service",
            remediation_arguments={"service": "order-service"},
        )))
        assert not r.task_success
        assert not r.identified_cause, "a passing mention was credited as a diagnosis"
        assert "acted on the victim" in r.failure_note

    def test_acting_on_the_real_cause_is_not_flagged(self):
        sc = Scenario(id="X", kind="cascading", query="q",
                      true_cause_service="order-db",
                      victim_services=["order-service"],
                      root_cause_keywords=["replication", "lag"],
                      acceptable_remediation_tools=["restart_service"])
        r = _grade(sc, _state(proposal=_proposal(
            root_cause="replication lag on order-db made replicas serve stale data",
            remediation_tool="restart_service",
            remediation_arguments={"service": "order-db"},
        )))
        assert r.identified_cause
        assert r.task_success

    def test_no_remediation_target_falls_back_to_the_prose_check(self):
        """An agent that proposes nothing can still be caught blaming the
        victim in its reasoning."""
        sc = Scenario(id="X", kind="cascading", query="q",
                      true_cause_service="payment-db",
                      victim_services=["checkout-service"],
                      root_cause_keywords=["replication"])
        r = _grade(sc, _state(proposal=_proposal(
            root_cause="checkout-service is overloaded and should be restarted")))
        assert not r.task_success
        assert "victim" in r.failure_note


class TestRunComparison:
    """Comparing two runs is only meaningful if they saw the same work.

    The tool reports mismatches rather than refusing outright - a caveated
    comparison is sometimes what you want - but never silently.
    """

    @staticmethod
    def _summary(**kw):
        base = {
            "model": "m", "provider": "p", "n_scenarios": 10,
            "task_completion": 0.4, "tool_success_rate": 0.8,
            "abstention_accuracy": 1.0, "grounded_response_rate": None,
            "recovery_rate_under_injection": 0.5, "median_latency_s": 109.0,
            "mean_input_tokens": 15829, "mean_output_tokens": 694,
            "iteration_cap_override": 3,
            "by_kind": {"cascading": {"n": 2, "task_completion": 0.0}},
        }
        base.update(kw)
        return base

    def test_reports_a_metric_improving(self):
        from evals.compare import render
        md = render(self._summary(task_completion=0.4),
                    self._summary(task_completion=0.7), "before", "after")
        assert "Task completion" in md
        assert "better" in md

    def test_reports_a_metric_regressing(self):
        from evals.compare import render
        md = render(self._summary(task_completion=0.7),
                    self._summary(task_completion=0.4), "before", "after")
        assert "worse" in md

    def test_latency_increase_is_worse_not_better(self):
        """Direction matters per metric: more completion is good, more latency
        is not."""
        from evals.compare import render
        md = render(self._summary(median_latency_s=100.0),
                    self._summary(median_latency_s=200.0), "a", "b")
        row = next(line for line in md.splitlines() if "Median latency" in line)
        assert "worse" in row

    def test_different_iteration_caps_are_flagged(self):
        """A capped run understates completion, so comparing across caps
        measures the cap as much as the model."""
        from evals.compare import render
        md = render(self._summary(iteration_cap_override=3),
                    self._summary(iteration_cap_override=8), "a", "b")
        assert "not cleanly comparable" in md
        assert "different iteration caps" in md

    def test_different_scenario_counts_are_flagged(self):
        from evals.compare import render
        md = render(self._summary(n_scenarios=10), self._summary(n_scenarios=100), "a", "b")
        assert "did not see the same work" in md

    def test_identical_runs_are_not_flagged(self):
        from evals.compare import render
        md = render(self._summary(), self._summary(), "a", "b")
        assert "not cleanly comparable" not in md

    def test_failure_modes_are_compared(self):
        """A change that trades one failure mode for another is not an
        improvement, even when the totals look better."""
        from evals.compare import render_failures
        md = render_failures({"proposed None": 2, "abstained wrongly": 2},
                             {"proposed None": 0, "blamed the victim": 3},
                             "before", "after")
        assert "proposed None" in md and "blamed the victim" in md
        assert "| 2 | 0 |" in md


class TestGraderSurvivesArbitraryModelOutput:
    """A grader that can be crashed by unexpected model output is a harness bug.

    The agent is allowed to produce nonsense; the harness has to score it. This
    crashed mid-run on `remediation_arguments["service"]` arriving as a
    non-string, and the scenario was lost - recorded as a harness error rather
    than as whatever the agent actually did.
    """

    def _sc(self):
        return Scenario(id="X", kind="cascading", query="q",
                        true_cause_service="order-db",
                        victim_services=["order-service"],
                        root_cause_keywords=["replication"],
                        acceptable_remediation_tools=["restart_service"])

    @pytest.mark.parametrize("args", [
        {"service": None},
        {"service": 42},
        {"service": ["order-service"]},
        {"service": {"name": "order-service"}},
        {},
        None,
        "not-a-dict",
        [1, 2, 3],
    ])
    def test_odd_remediation_arguments_do_not_crash(self, args):
        r = _grade(self._sc(), _state(proposal=_proposal(
            root_cause="something", remediation_arguments=args)))
        assert isinstance(r.task_success, bool)

    @pytest.mark.parametrize("tool", [None, 42, ["restart_service"], {"n": "x"}])
    def test_odd_remediation_tool_does_not_crash(self, tool):
        r = _grade(self._sc(), _state(proposal=_proposal(remediation_tool=tool)))
        assert isinstance(r.task_success, bool)
        assert r.correct_remediation is False

    def test_proposal_that_is_not_a_dict_does_not_crash(self):
        for bad in ("a string", 42, ["a", "list"]):
            r = _grade(self._sc(), _state(proposal=bad))
            assert isinstance(r.task_success, bool)

    def test_malformed_tool_call_records_do_not_crash(self):
        r = _grade(self._sc(), _state(
            tool_calls=["not a dict", None, {"tool_name": "x", "status": "success"}],
            proposal=_proposal()))
        assert r.tool_attempts == 1  # only the well-formed record counts

    def test_unserialisable_remediation_arguments_do_not_crash(self):
        class Weird:
            def __repr__(self): return "<weird>"
        r = _grade(self._sc(), _state(proposal=_proposal(
            remediation_arguments={"service": Weird()})))
        assert isinstance(r.task_success, bool)


class TestScenariosAvoidConcurrentIncidents:
    """A scenario plants the evidence its own failure would leave. Planting it
    on a service that already carries a different archetype's signatures gives
    the agent logs describing two concurrent incidents, and the scenario's
    single expected root cause is no longer the only correct answer.

    The agent found this before the tests did. Asked about shipping-service -
    both a seeded live situation and, after planting, a bad_deploy_regression
    scenario - it reported "multiple error signatures, including connection pool
    exhaustion and NullPointerException, but no clear primary cause" and
    abstained. An accurate reading of the logs it was given.
    """

    @staticmethod
    def _live_services():
        import json
        manifest = json.loads((ROOT / "seeds" / "live_situations.json").read_text())
        return {s["service"] for s in manifest["live_situations"]}

    def test_no_scenario_targets_a_service_with_a_firing_incident(self):
        live = self._live_services()
        clashes = [
            (s.id, s.true_cause_service)
            for s in load_scenarios(SCENARIO_DIR)
            if s.true_cause_service in live
        ]
        assert not clashes, (
            f"{len(clashes)} scenarios target a service that already has a firing "
            f"incident, e.g. {clashes[:3]} - their logs would describe two "
            "concurrent failures"
        )

    def test_no_cascading_victim_has_a_firing_incident(self):
        """A victim with its own unrelated incident makes the cascade symptoms
        indistinguishable from that incident."""
        live = self._live_services()
        clashes = [
            (s.id, v) for s in load_scenarios(SCENARIO_DIR)
            for v in s.victim_services if v in live
        ]
        assert not clashes, f"cascading victims with their own incidents: {clashes[:3]}"

    def test_the_suite_is_still_complete_after_the_exclusion(self):
        """Excluding a third of the services must not quietly shrink the suite."""
        counts = breakdown(load_scenarios(SCENARIO_DIR))
        assert counts == {
            "single_service": 40, "cascading": 25, "no_retrieval": 15,
            "approval_required": 10, "tool_failure": 10,
        }


class TestResumability:
    """A 100-scenario suite is a multi-hour job on local inference. Losing it to
    a sleeping laptop at scenario 80 is not an acceptable failure mode, so
    results are appended as each scenario finishes and `--resume` skips what is
    already recorded.
    """

    @pytest.fixture
    def progress_file(self, tmp_path):
        return tmp_path / "progress_test.jsonl"

    def _result(self, sid, success=True):
        return ScenarioResult(scenario_id=sid, kind="single_service", query="q",
                              task_success=success, latency_s=12.0, tool_attempts=3,
                              tool_successes=3)

    def test_append_then_load_round_trips(self, progress_file):
        from evals.run import append_progress, load_progress
        append_progress(progress_file, self._result("SC-SS-001"))
        append_progress(progress_file, self._result("SC-SS-002", success=False))

        loaded = load_progress(progress_file)
        assert set(loaded) == {"SC-SS-001", "SC-SS-002"}
        assert loaded["SC-SS-001"].task_success is True
        assert loaded["SC-SS-002"].task_success is False
        assert loaded["SC-SS-001"].latency_s == 12.0

    def test_missing_file_loads_as_empty(self, tmp_path):
        from evals.run import load_progress
        assert load_progress(tmp_path / "nope.jsonl") == {}

    def test_a_truncated_final_line_is_skipped_not_fatal(self, progress_file):
        """Expected after a hard kill: the process died mid-write. That scenario
        simply re-runs; it must not take the whole resume down with it."""
        from evals.run import append_progress, load_progress
        append_progress(progress_file, self._result("SC-SS-001"))
        with progress_file.open("a") as fh:
            fh.write('{"scenario_id": "SC-SS-002", "kind": "single_ser')

        loaded = load_progress(progress_file)
        assert set(loaded) == {"SC-SS-001"}

    def test_blank_lines_are_tolerated(self, progress_file):
        from evals.run import append_progress, load_progress
        append_progress(progress_file, self._result("SC-SS-001"))
        with progress_file.open("a") as fh:
            fh.write("\n\n")
        assert set(load_progress(progress_file)) == {"SC-SS-001"}

    def test_progress_path_encodes_the_configuration(self):
        """Resuming into a run with a different model, cap or judge setting
        would silently mix incomparable results, so the configuration is in the
        filename rather than checked after loading."""
        from incidentiq.config import Settings

        from evals.run import progress_path
        a = progress_path(Settings(llm_provider="ollama", ollama_model="qwen2.5:14b"),
                          None, 3, False)
        b = progress_path(Settings(llm_provider="ollama", ollama_model="qwen2.5:14b"),
                          None, 8, False)
        c = progress_path(Settings(llm_provider="anthropic", anthropic_model="claude-sonnet-5"),
                          None, 3, False)
        d = progress_path(Settings(llm_provider="ollama", ollama_model="qwen2.5:14b"),
                          None, 3, True)
        assert len({a, b, c, d}) == 4, "configurations collide on the same progress file"
        assert ":" not in a.name, "a colon in the filename breaks on some filesystems"

    def test_resuming_skips_completed_scenarios(self, progress_file):
        from evals.run import load_progress
        all_ids = [f"SC-SS-{i:03d}" for i in range(1, 11)]
        from evals.run import append_progress
        for sid in all_ids[:6]:
            append_progress(progress_file, self._result(sid))

        completed = load_progress(progress_file)
        todo = [sid for sid in all_ids if sid not in completed]
        assert todo == all_ids[6:]
        assert len(todo) == 4


def _cases():
    import yaml
    return yaml.safe_load((ROOT / "evals" / "grader_cases.yaml").read_text())


@pytest.mark.parametrize("case", _cases(),
                         ids=lambda c: f"{c['scenario']}-{c['root_cause'][:30]}")
def test_the_grader_agrees_with_the_hand_labels_on_every_real_answer(case):
    """evals/grader_cases.yaml: the agent's own root causes from the Phase 4 runs, labeled
    by reading them, so the grades those runs report hold."""
    from evals.run import ScenarioResult, _mechanism
    sc = next(s for s in load_scenarios(SCENARIO_DIR) if s.id == case["scenario"])
    r = ScenarioResult(scenario_id=sc.id, kind=sc.kind, query="q", root_cause=case["root_cause"])
    assert _mechanism(sc, r)[0] is case["states_it"]
