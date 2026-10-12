"""What counts as current evidence on a service: something abnormal (a change), not a
judged failure."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))

from incidentiq.agent.evidence import faults  # noqa: E402


def call(tool, service, result, status="success"):
    return {"tool_name": tool, "status": status, "arguments": {"service": service},
            "result": result}


def test_error_lines_on_the_service_are_evidence():
    logs = call("get_service_logs", "user-db", {"sample_lines": [
        {"level": "ERROR", "message": "no space left on device"},
        {"level": "WARN", "message": "slow checkpoint"}]})
    assert faults([logs], "user-db") == ["1 error line in its logs"]
    assert faults([logs], "user-service") == [], "evidence is about the service it came from"


def test_a_metric_must_move_well_beyond_normal_variation():
    overview = call("get_metrics", "user-db", {"metrics": [
        {"metric_name": "disk_usage_pct", "pct_change_vs_yesterday": 41.0},
        {"metric_name": "cpu_pct", "pct_change_vs_yesterday": -12.0},
        {"metric_name": "unassigned_shards", "pct_change_vs_yesterday": None,
         "same_hour_yesterday": 0, "latest_hour": 3}]})
    assert faults([overview], "user-db") == ["disk_usage_pct moved +41%",
                                             "unassigned_shards moved from zero"]
    series = call("get_metrics", "user-db", {"metric_name": "cpu_pct",
                                             "pct_change_vs_baseline": 18.0})
    assert faults([series], "user-db") == []


def test_the_gate_detects_change_either_way_not_which_way_is_bad():
    better = call("get_metrics", "user-db", {"metric_name": "error_rate",
                                             "pct_change_vs_baseline": -60.0})
    assert faults([better], "user-db") == ["error_rate moved -60%"]


def test_failed_calls_and_quiet_logs_are_not_evidence():
    assert faults([call("get_service_logs", "x", {"sample_lines": []}),
                   call("get_service_logs", "x", None, status="timeout")], "x") == []


def test_a_previous_period_of_zero_is_evidence_too():
    series = call("get_metrics", "search-cluster", {"metric_name": "unassigned_shards",
                                                    "baseline_previous_period": 0,
                                                    "current_value": 3,
                                                    "pct_change_vs_baseline": None})
    assert faults([series], "search-cluster") == ["unassigned_shards moved from zero"]
