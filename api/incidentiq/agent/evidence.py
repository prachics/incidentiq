"""What this investigation's own tool calls showed happening on a service.

An action is proposed only for a service the agent examined and saw something
abnormal on: error lines in its logs, or a metric that moved well beyond normal
variation, in either direction. The gate detects change, not failure: it does not
judge which direction is bad for each metric. A similar past incident is a
hypothesis to check, not evidence: on a scenario with nothing to find, the agent
saw no errors, retrieved an old incident and proposed that incident's fix.
"""

from __future__ import annotations

from typing import Any

# The seeded series vary by up to about +/-20% hour to hour; a failure moves its
# metric by 40% or more. A move either way counts (a throughput falling to zero is a
# change too), so a large improvement also lets an action through.
MOVED_PCT = 30.0


def faults(tool_calls: list[dict[str, Any]], service: str) -> list[str]:
    """Abnormal current behaviour of `service` (error lines, or a metric that moved),
    from successful tool calls on it: what the gate on actions requires."""
    found = []
    for call in tool_calls:
        args, result = call.get("arguments") or {}, call.get("result") or {}
        if call.get("status") not in ("success", "partial") or \
                str(args.get("service") or "").lower() != service.lower():
            continue
        if call["tool_name"] == "get_service_logs":
            errors = [line for line in result.get("sample_lines") or []
                      if line.get("level") in ("ERROR", "FATAL")]
            if errors:
                found.append(f"{len(errors)} error line{'s' * (len(errors) != 1)} in its logs")
        elif call["tool_name"] == "get_metrics":
            for m in result.get("metrics") or [result]:
                change = m.get("pct_change_vs_yesterday", m.get("pct_change_vs_baseline"))
                was_zero = (m.get("same_hour_yesterday") == 0 and m.get("latest_hour")) or \
                    (m.get("baseline_previous_period") == 0 and m.get("current_value"))
                if (change is not None and abs(change) >= MOVED_PCT) or was_zero:
                    found.append(f"{m.get('metric_name')} moved "
                                 f"{'from zero' if was_zero else f'{change:+.0f}%'}")
    return found
