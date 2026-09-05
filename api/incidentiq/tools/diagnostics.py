"""Read-only diagnostic tools. Auto-executed; no approval required.

These read the seeded mock infrastructure rather than a live system. The
interfaces are deliberately the ones a real implementation would have - swapping
`get_service_logs` for something that queries Loki is a change inside one method,
not a change to the agent.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import psycopg
from pydantic import BaseModel, Field, field_validator

from incidentiq.tools.base import Tool, ToolError, ToolStatus, registry

# The seeded world is pinned to this instant (see seeds/generate.py). Tools
# resolve relative windows against it so "the last hour" means the last hour of
# the generated data, not of wall-clock time.
DATA_NOW = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)

_WINDOW = re.compile(r"^(\d+)\s*([mhd])$")


def parse_window(value: str) -> timedelta:
    match = _WINDOW.match(value.strip().lower())
    if not match:
        raise ValueError(f"time_window must look like '30m', '6h' or '2d', got {value!r}")
    amount, unit = int(match.group(1)), match.group(2)
    delta = {"m": timedelta(minutes=amount), "h": timedelta(hours=amount),
             "d": timedelta(days=amount)}[unit]
    if delta > timedelta(days=30):
        raise ValueError("time_window may not exceed 30d")
    return delta


class _WindowMixin(BaseModel):
    time_window: str = Field(
        default="1h",
        description="Relative lookback window: '30m', '6h', '2d'. Max 30d.",
    )

    @field_validator("time_window")
    @classmethod
    def _valid_window(cls, v: str) -> str:
        parse_window(v)
        return v


def _require_service(conn: psycopg.Connection, name: str) -> None:
    """Fail with a useful message rather than returning an empty result.

    An empty result reads to the agent as "this service is healthy", which is a
    much worse outcome than an error saying the name was wrong. The suggestion
    uses trigram similarity, which is what pg_trgm is installed for.
    """
    if conn.execute("SELECT 1 FROM services WHERE name = %s", (name,)).fetchone():
        return
    near = conn.execute(
        "SELECT name FROM services WHERE similarity(name, %s) > 0.3 "
        "ORDER BY similarity(name, %s) DESC LIMIT 3",
        (name, name),
    ).fetchall()
    hint = f" Did you mean: {', '.join(r[0] for r in near)}?" if near else ""
    # MALFORMED, not FAILED: the arguments are wrong, so retrying them
    # unchanged cannot succeed. The retry loop breaks on MALFORMED and the
    # message goes back to the model, which can correct the name next
    # iteration. Classifying this as FAILED burned all three attempts.
    raise ToolError(f"unknown service {name!r}.{hint}", status=ToolStatus.MALFORMED)


# ── get_service_logs ────────────────────────────────────────
class ServiceLogsArgs(_WindowMixin):
    service: str = Field(min_length=1, description="Exact service name, e.g. 'checkout-service'.")
    level: Literal["DEBUG", "INFO", "WARN", "ERROR", "FATAL"] = Field(
        default="ERROR",
        description="Minimum severity. ERROR is the useful default when diagnosing.",
    )
    limit: int = Field(default=50, ge=1, le=200)


class GetServiceLogs(Tool):
    name = "get_service_logs"
    description = (
        "Fetch recent log lines for a service. Returns the distinct error signatures "
        "with counts, plus a sample of raw lines. Start here: the signature usually "
        "identifies the failure mode directly."
    )
    args_model = ServiceLogsArgs

    def run(self, conn: psycopg.Connection, args: ServiceLogsArgs) -> dict[str, Any]:
        _require_service(conn, args.service)
        order = ["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]
        levels = order[order.index(args.level):]
        since = DATA_NOW - parse_window(args.time_window)

        rows = conn.execute(
            "SELECT ts, level, message FROM log_entries "
            "WHERE service = %s AND level = ANY(%s) AND ts >= %s "
            "ORDER BY ts DESC LIMIT %s",
            (args.service, levels, since, args.limit),
        ).fetchall()

        signatures = conn.execute(
            "SELECT message, count(*) AS n, min(ts) AS first_seen, max(ts) AS last_seen "
            "FROM log_entries WHERE service = %s AND level = ANY(%s) AND ts >= %s "
            "GROUP BY message ORDER BY n DESC LIMIT 10",
            (args.service, levels, since),
        ).fetchall()

        return {
            "service": args.service,
            "time_window": args.time_window,
            "min_level": args.level,
            "total_matching": len(rows),
            "distinct_signatures": [
                {"message": m, "count": n,
                 "first_seen": f.isoformat(), "last_seen": last.isoformat()}
                for m, n, f, last in signatures
            ],
            "sample_lines": [
                {"ts": ts.isoformat(), "level": lvl, "message": msg}
                for ts, lvl, msg in rows[:15]
            ],
        }


# ── get_metrics ─────────────────────────────────────────────
class MetricsArgs(_WindowMixin):
    service: str = Field(min_length=1, description="Exact service name.")
    metric_name: str = Field(
        description=(
            "One of: error_rate, latency_p99_ms, cpu_pct, memory_usage_pct, "
            "db_pool_active_connections, disk_usage_pct, replication_lag_seconds, "
            "cache_hit_rate, consumer_lag_messages, thread_pool_active, "
            "unassigned_shards, under_replicated_partitions."
        )
    )


class GetMetrics(Tool):
    name = "get_metrics"
    description = (
        "Fetch a metric time series for a service, bucketed hourly, with a comparison "
        "against the preceding period. Use this to confirm whether a metric actually "
        "changed rather than trusting a single reading."
    )
    args_model = MetricsArgs

    def run(self, conn: psycopg.Connection, args: MetricsArgs) -> dict[str, Any]:
        _require_service(conn, args.service)
        window = parse_window(args.time_window)
        since = DATA_NOW - window

        available = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT metric_name FROM metric_points WHERE service = %s",
                (args.service,),
            ).fetchall()
        ]
        if args.metric_name not in available:
            raise ToolError(
                f"metric {args.metric_name!r} is not collected for {args.service}. "
                f"Available: {', '.join(sorted(available))}",
                status=ToolStatus.MALFORMED,   # wrong argument, not a transient fault
            )

        series = conn.execute(
            "SELECT date_trunc('hour', ts) AS bucket, avg(value), min(value), max(value) "
            "FROM metric_points WHERE service = %s AND metric_name = %s AND ts >= %s "
            "GROUP BY bucket ORDER BY bucket",
            (args.service, args.metric_name, since),
        ).fetchall()

        baseline = conn.execute(
            "SELECT avg(value) FROM metric_points "
            "WHERE service = %s AND metric_name = %s AND ts >= %s AND ts < %s",
            (args.service, args.metric_name, since - window, since),
        ).fetchone()[0]

        current = series[-1][1] if series else None
        change = None
        if baseline and current is not None:
            change = round((current - baseline) / abs(baseline) * 100, 1)

        return {
            "service": args.service,
            "metric_name": args.metric_name,
            "time_window": args.time_window,
            "baseline_previous_period": round(baseline, 4) if baseline else None,
            "current_value": round(current, 4) if current is not None else None,
            "pct_change_vs_baseline": change,
            "series": [
                {"hour": b.isoformat(), "avg": round(a, 4),
                 "min": round(mn, 4), "max": round(mx, 4)}
                for b, a, mn, mx in series
            ],
        }


# ── get_recent_deploys ──────────────────────────────────────
class DeploysArgs(_WindowMixin):
    service: str = Field(min_length=1, description="Exact service name.")


class GetRecentDeploys(Tool):
    name = "get_recent_deploys"
    description = (
        "List deploys for a service in a window, most recent first. A sharp change in "
        "behaviour that coincides with a release is usually the release - check this "
        "before concluding a cause."
    )
    args_model = DeploysArgs

    def run(self, conn: psycopg.Connection, args: DeploysArgs) -> dict[str, Any]:
        _require_service(conn, args.service)
        since = DATA_NOW - parse_window(args.time_window)
        rows = conn.execute(
            "SELECT id, version, previous_version, deployed_at, deployed_by, status, "
            "changelog FROM deploys WHERE service = %s AND deployed_at >= %s "
            "ORDER BY deployed_at DESC LIMIT 20",
            (args.service, since),
        ).fetchall()
        return {
            "service": args.service,
            "time_window": args.time_window,
            "deploy_count": len(rows),
            "deploys": [
                {"id": i, "version": v, "previous_version": pv,
                 "deployed_at": at.isoformat(), "deployed_by": by,
                 "status": st, "changelog": cl}
                for i, v, pv, at, by, st, cl in rows
            ],
        }


# ── get_service_dependencies ────────────────────────────────
class DependenciesArgs(BaseModel):
    service: str = Field(min_length=1, description="Exact service name.")
    direction: Literal["downstream", "upstream", "both"] = Field(
        default="both",
        description=(
            "downstream = what this service calls (candidate causes). "
            "upstream = what calls it (who is affected). both = the full picture."
        ),
    )


class GetServiceDependencies(Tool):
    name = "get_service_dependencies"
    description = (
        "Map a service's dependencies and callers. Essential for distinguishing cause "
        "from victim: a service that is slow while its own CPU and memory are normal "
        "is usually waiting on something downstream."
    )
    args_model = DependenciesArgs

    def run(self, conn: psycopg.Connection, args: DependenciesArgs) -> dict[str, Any]:
        _require_service(conn, args.service)
        out: dict[str, Any] = {"service": args.service}

        if args.direction in ("downstream", "both"):
            rows = conn.execute(
                "SELECT d.to_service, d.kind, s.tier FROM service_dependencies d "
                "JOIN services s ON s.name = d.to_service WHERE d.from_service = %s",
                (args.service,),
            ).fetchall()
            out["depends_on"] = [{"service": t, "kind": k, "tier": tier}
                                 for t, k, tier in rows]

        if args.direction in ("upstream", "both"):
            rows = conn.execute(
                "SELECT d.from_service, d.kind, s.tier FROM service_dependencies d "
                "JOIN services s ON s.name = d.from_service WHERE d.to_service = %s",
                (args.service,),
            ).fetchall()
            out["called_by"] = [{"service": f, "kind": k, "tier": tier}
                                for f, k, tier in rows]

        meta = conn.execute(
            "SELECT tier, language, owner_team, replica_count FROM services WHERE name = %s",
            (args.service,),
        ).fetchone()
        out["tier"], out["language"], out["owner_team"], out["replica_count"] = meta
        return out


# ── search_similar_incidents ────────────────────────────────
class SimilarIncidentsArgs(BaseModel):
    query: str = Field(
        description=(
            "What you are looking for. Include observed error signatures verbatim - "
            "exact strings match far better than paraphrase."
        )
    )
    service: str | None = Field(
        default=None,
        description="Restrict to one service. Narrows results substantially when known.",
    )
    limit: int = Field(default=5, ge=1, le=10)


class SearchSimilarIncidents(Tool):
    name = "search_similar_incidents"
    description = (
        "Search historical incidents, runbooks, and service documentation using hybrid "
        "semantic and keyword retrieval. Returns cited passages with relevance scores. "
        "Every claim you make should be traceable to something this returns."
    )
    args_model = SimilarIncidentsArgs

    def run(self, conn: psycopg.Connection, args: SimilarIncidentsArgs) -> dict[str, Any]:
        from incidentiq.rag.retrieval import Filters, hydrate, search

        hits = hydrate(conn, search(
            conn, args.query,
            filters=Filters(service=args.service) if args.service else None,
            limit=args.limit, mode="hybrid",
        ))
        return {
            "query": args.query,
            "service_filter": args.service,
            "result_count": len(hits),
            "results": [
                {
                    "doc_id": h.parent_doc_id,
                    "doc_type": h.doc_type,
                    "title": h.parent_title,
                    "service": h.service,
                    "relevance": round(h.rrf_score, 5),
                    "vector_rank": h.vector_rank,
                    "keyword_rank": h.keyword_rank,
                    "excerpt": h.content[:600],
                }
                for h in hits
            ],
        }


for _tool in (GetServiceLogs(), GetMetrics(), GetRecentDeploys(),
              GetServiceDependencies(), SearchSimilarIncidents()):
    registry.register(_tool)
