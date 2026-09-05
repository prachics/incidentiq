#!/usr/bin/env python3
"""Generate and load the synthetic corpus and mock infrastructure.

Everything is derived from a fixed RNG seed, so two runs produce byte-identical
data. That matters more than it sounds: eval numbers are only comparable across
runs if the corpus they retrieve from is the same corpus.

What gets written:

  services, service_dependencies, service_instances   from seeds/catalog.py
  deploys, log_entries, metric_points                 mock infra the tools read
  incidents (500), runbooks (~50), service_docs (~30) the three RAG corpora
  seeds/live_situations.json                          manifest of active problems

"Live situations" are the bridge between the corpus and the agent. Each one is
an archetype currently firing on a service, with matching log lines and a metric
anomaly written into the mock infra — so when an investigation calls
get_service_logs, there is a real signal to find. Phase 3 builds eval scenarios
from this manifest.

Usage:
    python seeds/generate.py             # load (refuses if data present)
    python seeds/generate.py --force     # truncate and reload
    python seeds/generate.py --stats     # just report what is loaded
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg

SEEDS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SEEDS_DIR.parent
sys.path.insert(0, str(SEEDS_DIR))
sys.path.insert(0, str(REPO_ROOT / "api"))

import archetypes as A  # noqa: E402
import catalog as C  # noqa: E402

RANDOM_SEED = 20260101
N_INCIDENTS = 500
N_LIVE_SITUATIONS = 12
HISTORY_MONTHS = 18
LOG_WINDOW_HOURS = 48
METRIC_INTERVAL_MIN = 5

# "Now" is pinned so generated data is reproducible. Everything is relative to it.
NOW = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)

ENGINEERS = [
    "r.okafor", "s.venkatesan", "m.lindqvist", "j.park", "a.duarte", "n.bianchi",
    "t.almeida", "k.oyelaran", "d.novak", "h.tanaka", "l.moreau", "p.raghavan",
]

BASELINES = {
    "error_rate": 0.004, "latency_p99_ms": 210.0, "cpu_pct": 34.0,
    "memory_usage_pct": 48.0, "db_pool_active_connections": 6.0,
    "disk_usage_pct": 62.0, "replication_lag_seconds": 1.2,
    "cache_hit_rate": 0.94, "consumer_lag_messages": 40.0,
    "thread_pool_active": 22.0, "unassigned_shards": 0.0,
    "under_replicated_partitions": 0.0,
}

# Metrics whose healthy value is zero. A multiplicative spike cannot move them,
# so for these the archetype's metric_spike is read as the absolute value the
# metric reaches while the incident is firing.
ZERO_BASELINE_METRICS = frozenset({"unassigned_shards", "under_replicated_partitions"})
DEFAULT_METRICS = ("error_rate", "latency_p99_ms", "cpu_pct", "memory_usage_pct")


def rng() -> random.Random:
    return random.Random(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────
# Template rendering
# ─────────────────────────────────────────────────────────────

def render(template: str, *, service: str, dep: str = "", version: str = "", team: str = "") -> str:
    return template.format(service=service, dep=dep, version=version, team=team)


def pick_dep(r: random.Random, svc: C.Service, archetype: A.Archetype) -> str:
    """Pick a real dependency for archetypes whose templates reference one.

    This is what keeps the corpus internally consistent: an incident that blames
    a dependency blames one this service actually has.
    """
    if not archetype.needs_dep:
        return ""
    candidates = [t for t, _ in svc.depends_on]
    if archetype.key == "cache_stampede":
        cache = [t for t in candidates if C.BY_NAME[t].kind == "cache"]
        candidates = cache or candidates
    elif archetype.key == "db_pool_exhausted":
        stores = [t for t in candidates if C.BY_NAME[t].kind == "datastore"]
        candidates = stores or candidates
    elif archetype.key == "consumer_lag":
        queues = [t for t in candidates if C.BY_NAME[t].kind == "queue"]
        candidates = queues or candidates
    return r.choice(candidates) if candidates else ""


def eligible(svc: C.Service) -> list[A.Archetype]:
    """Archetypes valid for this service, dropping any that need a dependency
    the service does not have."""
    out = []
    for a in A.for_service(svc.kind, svc.language):
        if a.needs_dep and not svc.depends_on:
            continue
        if a.key == "consumer_lag":
            if not any(C.BY_NAME[t].kind == "queue" for t, _ in svc.depends_on):
                continue
        if a.key == "cache_stampede":
            if not any(C.BY_NAME[t].kind in ("cache", "datastore") for t, _ in svc.depends_on):
                continue
        out.append(a)
    return out


def version_string(r: random.Random) -> str:
    return f"v{r.randint(1, 4)}.{r.randint(0, 40)}.{r.randint(0, 9)}"


# ─────────────────────────────────────────────────────────────
# Corpus generation
# ─────────────────────────────────────────────────────────────

def gen_incidents(r: random.Random) -> list[dict]:
    """500 historical incidents, weighted so tier-1 services have more of them."""
    weights, services = [], []
    for svc in C.SERVICES:
        if not eligible(svc):
            continue
        services.append(svc)
        weights.append({"tier-1": 5, "tier-2": 3, "tier-3": 1}[svc.tier])

    out = []
    for i in range(N_INCIDENTS):
        svc = r.choices(services, weights=weights, k=1)[0]
        arch = r.choice(eligible(svc))
        dep = pick_dep(r, svc, arch)
        version = version_string(r)

        occurred = NOW - timedelta(
            days=r.randint(14, HISTORY_MONTHS * 30),
            hours=r.randint(0, 23), minutes=r.randint(0, 59),
        )
        severity = r.choices(
            ["SEV1", "SEV2", "SEV3", "SEV4"],
            weights={"tier-1": [3, 5, 6, 2], "tier-2": [1, 4, 8, 4],
                     "tier-3": [0, 2, 7, 8]}[svc.tier],
            k=1,
        )[0]
        ttr_minutes = {"SEV1": (12, 90), "SEV2": (20, 180),
                       "SEV3": (30, 420), "SEV4": (60, 1440)}[severity]

        kw = {"service": svc.name, "dep": dep, "version": version, "team": svc.owner_team}
        out.append({
            "id": f"INC-{i + 1:05d}",
            "title": f"{arch.name} on {svc.name}",
            "service": svc.name,
            "severity": severity,
            "occurred_at": occurred,
            "resolved_at": occurred + timedelta(minutes=r.randint(*ttr_minutes)),
            "symptoms": render(r.choice(arch.symptoms), **kw),
            "root_cause": render(r.choice(arch.root_cause), **kw),
            "resolution": render(r.choice(arch.resolution), **kw),
            "tags": list(arch.tags) + [svc.owner_team, svc.tier],
            # The error lines the on-call engineer quoted in the write-up.
            # Real postmortems include these; without them the corpus has no
            # distinctive identifiers and keyword retrieval has nothing to grip.
            "error_signatures": [
                render(line, **kw)
                for line in r.sample(arch.log_lines, k=min(2, len(arch.log_lines)))
            ],
            # Not stored in the DB - used to build the labelled retrieval set.
            "_archetype": arch.key,
            "_dep": dep,
        })
    return out


def gen_runbooks(r: random.Random) -> list[dict]:
    """One general runbook per archetype, plus service-specific variants for
    tier-1 services. Roughly 50 total."""
    out: list[dict] = []
    n = 0

    for arch in A.ARCHETYPES:
        n += 1
        steps = _runbook_steps(arch, service=None)
        out.append({
            "id": f"RB-{n:03d}",
            "title": f"Runbook: {arch.name}",
            "service": None,
            "category": arch.category,
            "body": steps,
            "applies_to": list(arch.tags),
        })

    scoped = [s for s in C.SERVICES if s.tier in ("tier-1", "tier-2") and eligible(s)]
    for svc in scoped:
        per = 3 if svc.tier == "tier-1" else 2
        for arch in r.sample(eligible(svc), k=min(per, len(eligible(svc)))):
            n += 1
            out.append({
                "id": f"RB-{n:03d}",
                "title": f"Runbook: {arch.name} — {svc.name}",
                "service": svc.name,
                "category": arch.category,
                "body": _runbook_steps(arch, service=svc),
                "applies_to": list(arch.tags) + [svc.name, svc.owner_team],
            })
    return out


def _runbook_steps(arch: A.Archetype, service: C.Service | None) -> str:
    name = service.name if service else "the affected service"
    owner = f"\n**Owner:** {service.owner_team}\n" if service else ""
    scoped = f" for `{service.name}`" if service else ""
    remediation = {
        "restart_service": "Restart the affected instances. This is disruptive to in-flight "
                           "requests — confirm with the service owner before proceeding.",
        "rollback_deploy": "Roll back to the last known-good version. Verify the target "
                           "version predates the onset of symptoms.",
        "scale_service": "Scale out to absorb load. Confirm the bottleneck is capacity and "
                         "not a downstream dependency before scaling — scaling a victim "
                         "service increases load on the thing that is actually broken.",
        None: "No standard write action. Remediation depends on the specific cause "
              "identified in step 3.",
    }[arch.remediation_tool]

    return f"""# {arch.name}{scoped}

**Category:** {arch.category}
**Applies to:** {", ".join(arch.tags)}{owner}

## Symptoms to confirm

Before acting, confirm you are actually looking at this failure mode and not
something that resembles it:

- Check `{arch.metric}` on {name}. This archetype moves it noticeably from its
  normal range; if that metric is flat, look elsewhere.
- Search recent logs for these signatures:
{chr(10).join(f"  - `{line}`" for line in arch.log_lines[:3])}

## Diagnosis

1. Pull the last 30 minutes of ERROR-level logs for {name} and look for the
   signatures above.
2. Compare `{arch.metric}` against the same window one day earlier. A step
   change matters more than an absolute value.
3. Check whether a deploy landed within 30 minutes before onset. A sharp edge
   that coincides with a release is usually the release.
4. If {name} looks healthy by its own metrics but is slow, check its
   dependencies before concluding the problem is local. A service that is
   waiting is not a service that is broken.

## Remediation

{remediation}

## Prevention

- Alert on `{arch.metric}` before it reaches the level that causes user impact.
- Review whether the failure was detectable earlier than it was detected. Most
  instances of this archetype are visible in metrics before they are visible in
  error rates.
"""


def gen_service_docs(r: random.Random) -> list[dict]:
    """Architecture, config, and SLA docs for the more significant services."""
    out: list[dict] = []
    n = 0
    targets = [s for s in C.SERVICES if s.tier in ("tier-1", "tier-2")][:15]

    for svc in targets:
        callers = C.dependents_of(svc.name)
        n += 1
        out.append({
            "id": f"DOC-{n:03d}", "title": f"{svc.name} — architecture",
            "service": svc.name, "doc_kind": "architecture",
            "body": f"""# {svc.name} — architecture

**Tier:** {svc.tier}  **Language:** {svc.language}  **Owner:** {svc.owner_team}
**Replicas:** {svc.replica_count}

{svc.description}

## Dependencies

{chr(10).join(f"- `{t}` ({k})" for t, k in svc.depends_on) or "- None. This service has no outbound dependencies."}

## Callers

{chr(10).join(f"- `{c}`" for c in callers) or "- None. This service is not called by other services."}

## Failure characteristics

{"Because this is a " + svc.tier + " service, degradation here is visible to customers within seconds." if svc.tier == "tier-1" else "Degradation here is usually absorbed by callers before customers notice, provided timeouts are configured correctly."}

{"Callers most affected by an outage here: " + ", ".join(f"`{c}`" for c in C.upstream_chain(svc.name)[:5]) if callers else "No upstream blast radius — nothing calls this service."}
""",
        })

        n += 1
        out.append({
            "id": f"DOC-{n:03d}", "title": f"{svc.name} — configuration",
            "service": svc.name, "doc_kind": "config",
            "body": f"""# {svc.name} — configuration reference

**Owner:** {svc.owner_team}

| Key | Default | Notes |
|---|---|---|
| `REPLICA_COUNT` | {svc.replica_count} | Baseline. Autoscales to {svc.replica_count * 3} on CPU. |
| `REQUEST_TIMEOUT_MS` | 2000 | Applies to all outbound sync calls. |
| `MAX_POOL_SIZE` | 20 | Per-replica connection pool to each datastore. |
| `RETRY_MAX_ATTEMPTS` | 3 | Exponential backoff, 100ms base. |
| `CIRCUIT_BREAKER_THRESHOLD` | 20 | Consecutive failures before the breaker opens. |
| `HEALTH_CHECK_PATH` | `/healthz` | Liveness. Does not check dependencies. |
| `READINESS_PATH` | `/readyz` | Readiness. Does check dependencies. |

## Notes

The liveness probe deliberately does not check dependencies. A service that is
healthy but whose dependency is down should not be restarted — restarting it
does not fix the dependency and drops in-flight work.

{"Connection pooling to " + ", ".join(f"`{t}`" for t, k in svc.depends_on if k == "datastore") + " is per-replica, so the effective pool size against each datastore is MAX_POOL_SIZE × REPLICA_COUNT." if any(k == "datastore" for _, k in svc.depends_on) else ""}
""",
        })

        if svc.tier == "tier-1":
            n += 1
            out.append({
                "id": f"DOC-{n:03d}", "title": f"{svc.name} — SLA and escalation",
                "service": svc.name, "doc_kind": "sla",
                "body": f"""# {svc.name} — SLA and escalation

**Owner:** {svc.owner_team}  **Tier:** {svc.tier}

| Objective | Target |
|---|---|
| Availability | 99.95% monthly |
| Latency p99 | < 400ms |
| Error rate | < 0.5% |

## Escalation

- **SEV1** — customer-facing outage. Page {svc.owner_team} immediately; engage
  the incident commander rotation if unresolved after 15 minutes.
- **SEV2** — significant degradation. Page {svc.owner_team} during business
  hours, on-call otherwise.
- **SEV3/SEV4** — ticket to {svc.owner_team}.

## Approval requirements

Any write action against {svc.name} — restart, scale, or rollback — requires
explicit approval from the on-call engineer. Rollback of a tier-1 service
additionally requires the deploying engineer to be notified.
""",
            })
    return out


# ─────────────────────────────────────────────────────────────
# Mock infrastructure
# ─────────────────────────────────────────────────────────────

def gen_live_situations(r: random.Random) -> list[dict]:
    """Problems that are 'happening now' — the signal the agent's tools find."""
    candidates = [s for s in C.SERVICES if eligible(s)]
    chosen = r.sample(candidates, k=min(N_LIVE_SITUATIONS, len(candidates)))
    out = []
    for i, svc in enumerate(chosen):
        arch = r.choice(eligible(svc))
        dep = pick_dep(r, svc, arch)
        onset = NOW - timedelta(hours=r.randint(1, 20), minutes=r.choice([0, 7, 13, 22, 41]))
        kw = {"service": svc.name, "dep": dep, "version": version_string(r), "team": svc.owner_team}
        out.append({
            "id": f"LIVE-{i + 1:02d}",
            "service": svc.name,
            "archetype": arch.key,
            "archetype_name": arch.name,
            "dep": dep,
            "onset": onset.isoformat(),
            "metric": arch.metric,
            "expected_root_cause": render(r.choice(arch.root_cause), **kw),
            "acceptable_remediation": render(r.choice(arch.resolution), **kw),
            "remediation_tool": arch.remediation_tool,
            "user_query": _phrase_as_oncall(r, svc.name, arch),
            "cascade_visible_in": C.upstream_chain(svc.name)[:3],
        })
    return out


def _phrase_as_oncall(r: random.Random, service: str, arch: A.Archetype) -> str:
    """How a tired on-call engineer would actually describe it — vague, partial,
    and never using the archetype's name. Retrieval has to bridge that gap."""
    return r.choice([
        f"{service} is throwing errors and I'm not sure why. Started maybe an hour ago.",
        f"getting paged for {service}, latency is way up. what's going on?",
        f"something's wrong with {service}. customers are complaining. help?",
        f"{service} looks unhealthy in the dashboard but I can't tell what changed",
        f"we're seeing failures on {service}. no idea if it's us or something downstream",
    ])


def gen_deploys(r: random.Random, live: list[dict]) -> list[dict]:
    """Deploy history. Live situations caused by a deploy get a matching one
    landing just before onset, so the timeline actually lines up."""
    out, n = [], 0
    deploy_caused = {"bad_deploy_regression", "memory_leak_oom", "config_drift"}

    for svc in C.SERVICES:
        if svc.kind in ("datastore", "cache", "queue"):
            continue
        version_major, version_minor = r.randint(1, 3), r.randint(5, 30)
        for d in range(r.randint(4, 9)):
            n += 1
            prev = f"v{version_major}.{version_minor}.{d}"
            cur = f"v{version_major}.{version_minor}.{d + 1}"
            out.append({
                "id": f"DEP-{n:05d}", "service": svc.name, "version": cur,
                "previous_version": prev,
                "deployed_at": NOW - timedelta(days=r.randint(2, 60), hours=r.randint(0, 23)),
                "deployed_by": r.choice(ENGINEERS),
                "status": r.choices(["succeeded", "rolled_back", "failed"],
                                    weights=[9, 1, 1], k=1)[0],
                "changelog": r.choice([
                    "dependency bumps and logging improvements",
                    "add caching layer for hot path",
                    "refactor request handler; no behaviour change intended",
                    "performance: reduce allocations in the serialisation path",
                    "add retry logic for transient downstream errors",
                    "migrate config loading to the new format",
                ]),
            })

    for sit in live:
        if sit["archetype"] not in deploy_caused:
            continue
        n += 1
        onset = datetime.fromisoformat(sit["onset"])
        out.append({
            "id": f"DEP-{n:05d}", "service": sit["service"],
            "version": "v3.12.0", "previous_version": "v3.11.4",
            "deployed_at": onset - timedelta(minutes=r.randint(4, 25)),
            "deployed_by": r.choice(ENGINEERS),
            "status": "succeeded",
            "changelog": "optimise response handling; add in-process cache for lookups",
        })
        sit["triggering_deploy"] = out[-1]["id"]
    return out


def gen_instances(r: random.Random) -> list[dict]:
    zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
    out = []
    for svc in C.SERVICES:
        for i in range(svc.replica_count):
            out.append({
                "id": f"{svc.name}-{r.randbytes(2).hex()}-{i:02d}",
                "service": svc.name, "zone": zones[i % 3],
                "status": "healthy",
                "started_at": NOW - timedelta(days=r.randint(1, 30), hours=r.randint(0, 23)),
            })
    return out


def gen_logs(r: random.Random, live: list[dict]) -> list[tuple]:
    """Background noise for every service, plus the archetype's real signature
    for services with a live situation."""
    rows: list[tuple] = []
    noise = [
        ("INFO", "request completed in {ms}ms status=200"),
        ("INFO", "healthcheck ok"),
        ("DEBUG", "cache lookup key=user:{n} hit=true"),
        ("WARN", "slow request: {ms}ms exceeds warn threshold 1000ms"),
        ("INFO", "connection pool: active={n} idle={m}"),
        ("WARN", "retrying downstream call (attempt 2/3)"),
        ("ERROR", "request failed: transient upstream error, retried successfully"),
    ]
    live_by_service = {s["service"]: s for s in live}

    for svc in C.SERVICES:
        start = NOW - timedelta(hours=LOG_WINDOW_HOURS)
        for _ in range(r.randint(180, 320)):
            level, tmpl = r.choices(noise, weights=[30, 10, 8, 6, 6, 4, 2], k=1)[0]
            ts = start + timedelta(minutes=r.uniform(0, LOG_WINDOW_HOURS * 60))
            msg = tmpl.format(ms=r.randint(20, 1800), n=r.randint(1, 9999), m=r.randint(0, 20))
            rows.append((svc.name, ts, level, msg, f"tr-{r.randbytes(6).hex()}"))

        sit = live_by_service.get(svc.name)
        if not sit:
            continue
        arch = A.BY_KEY[sit["archetype"]]
        onset = datetime.fromisoformat(sit["onset"])
        # Dense error signature from onset to now - this is what the agent finds.
        for _ in range(r.randint(60, 140)):
            ts = onset + timedelta(minutes=r.uniform(0, (NOW - onset).total_seconds() / 60))
            line = r.choice(arch.log_lines).format(service=svc.name, dep=sit["dep"] or "upstream")
            rows.append((svc.name, ts, r.choices(["ERROR", "FATAL"], weights=[9, 1], k=1)[0],
                         line, f"tr-{r.randbytes(6).hex()}"))
        # The cascade: callers see timeouts against this service.
        for caller in sit["cascade_visible_in"]:
            for _ in range(r.randint(15, 40)):
                ts = onset + timedelta(minutes=r.uniform(5, (NOW - onset).total_seconds() / 60))
                rows.append((caller, ts, "ERROR",
                             f"context deadline exceeded calling {svc.name}",
                             f"tr-{r.randbytes(6).hex()}"))
    return rows


def gen_metrics(r: random.Random, live: list[dict]) -> list[tuple]:
    """Time series at 5-minute granularity. Live situations get a real anomaly
    starting at onset, so the metric and the logs tell the same story."""
    rows: list[tuple] = []
    live_by_service = {s["service"]: s for s in live}
    points = LOG_WINDOW_HOURS * 60 // METRIC_INTERVAL_MIN
    start = NOW - timedelta(hours=LOG_WINDOW_HOURS)

    for svc in C.SERVICES:
        sit = live_by_service.get(svc.name)
        arch = A.BY_KEY[sit["archetype"]] if sit else None
        onset = datetime.fromisoformat(sit["onset"]) if sit else None

        names = set(DEFAULT_METRICS)
        if arch:
            names.add(arch.metric)
        if svc.kind == "datastore":
            names |= {"disk_usage_pct", "replication_lag_seconds"}
        if svc.kind == "cache":
            names.add("cache_hit_rate")
        if svc.kind == "queue":
            names |= {"under_replicated_partitions", "consumer_lag_messages"}

        for metric in names:
            base = BASELINES.get(metric, 1.0)
            for i in range(points):
                ts = start + timedelta(minutes=i * METRIC_INTERVAL_MIN)
                # Diurnal shape + noise.
                hour_factor = 0.75 + 0.5 * abs(((ts.hour - 3) % 24) / 24 - 0.5) * 2
                value = base * hour_factor * r.uniform(0.9, 1.1)
                if arch and metric == arch.metric and onset and ts >= onset:
                    ramp = min(1.0, (ts - onset).total_seconds() / 1800)
                    if metric in ZERO_BASELINE_METRICS:
                        # Absolute target, and integral - "2.4 unassigned shards"
                        # is not a thing that can be true.
                        value = round(arch.metric_spike * ramp)
                    else:
                        value = base * (1 + (arch.metric_spike - 1) * ramp) * r.uniform(0.95, 1.05)
                rows.append((svc.name, metric, ts, round(value, 4)))
    return rows


# ─────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────

def _copy(conn: psycopg.Connection, table: str, columns: list[str], rows: list[tuple]) -> None:
    """Bulk load via COPY. Orders of magnitude faster than executemany for the
    log and metric tables, which are the only ones large enough to notice."""
    cols = ", ".join(columns)
    with conn.cursor().copy(f"COPY {table} ({cols}) FROM STDIN") as copy:
        for row in rows:
            copy.write_row(row)


def load(conn: psycopg.Connection, force: bool) -> dict[str, int]:
    existing = conn.execute("SELECT count(*) FROM services").fetchone()[0]
    if existing and not force:
        raise SystemExit(
            f"{existing} services already loaded. Re-run with --force to truncate and reload."
        )
    if force:
        conn.execute(
            "TRUNCATE services, service_dependencies, service_instances, deploys, "
            "log_entries, metric_points, incidents, runbooks, service_docs, chunks "
            "RESTART IDENTITY CASCADE"
        )

    C.validate()
    r = rng()

    live = gen_live_situations(r)
    incidents = gen_incidents(r)
    runbooks = gen_runbooks(r)
    docs = gen_service_docs(r)
    deploys = gen_deploys(r, live)
    instances = gen_instances(r)
    logs = gen_logs(r, live)
    metrics = gen_metrics(r, live)

    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO services (name, tier, language, owner_team, replica_count, description)"
            " VALUES (%s,%s,%s,%s,%s,%s)",
            [(s.name, s.tier, s.language, s.owner_team, s.replica_count, s.description)
             for s in C.SERVICES],
        )
        cur.executemany(
            "INSERT INTO service_dependencies (from_service, to_service, kind) VALUES (%s,%s,%s)",
            C.dependencies(),
        )
        cur.executemany(
            "INSERT INTO service_instances (id, service, zone, status, started_at)"
            " VALUES (%s,%s,%s,%s,%s)",
            [(i["id"], i["service"], i["zone"], i["status"], i["started_at"]) for i in instances],
        )
        cur.executemany(
            "INSERT INTO deploys (id, service, version, previous_version, deployed_at,"
            " deployed_by, status, changelog) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            [(d["id"], d["service"], d["version"], d["previous_version"], d["deployed_at"],
              d["deployed_by"], d["status"], d["changelog"]) for d in deploys],
        )
        cur.executemany(
            "INSERT INTO incidents (id, title, service, severity, occurred_at, resolved_at,"
            " symptoms, root_cause, resolution, tags, error_signatures)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            [(i["id"], i["title"], i["service"], i["severity"], i["occurred_at"],
              i["resolved_at"], i["symptoms"], i["root_cause"], i["resolution"],
              i["tags"], i["error_signatures"])
             for i in incidents],
        )
        cur.executemany(
            "INSERT INTO runbooks (id, title, service, category, body, applies_to)"
            " VALUES (%s,%s,%s,%s,%s,%s)",
            [(b["id"], b["title"], b["service"], b["category"], b["body"], b["applies_to"])
             for b in runbooks],
        )
        cur.executemany(
            "INSERT INTO service_docs (id, title, service, doc_kind, body)"
            " VALUES (%s,%s,%s,%s,%s)",
            [(d["id"], d["title"], d["service"], d["doc_kind"], d["body"]) for d in docs],
        )

    _copy(conn, "log_entries", ["service", "ts", "level", "message", "trace_id"], logs)
    _copy(conn, "metric_points", ["service", "metric_name", "ts", "value"], metrics)

    # The manifest is how Phase 3 builds eval scenarios, and how the labelled
    # retrieval set knows which incidents *should* be retrieved for a query.
    manifest = {
        "generated_at": NOW.isoformat(),
        "random_seed": RANDOM_SEED,
        "live_situations": live,
        "labelled_retrieval": _labelled_retrieval_set(live, incidents),
    }
    (SEEDS_DIR / "live_situations.json").write_text(json.dumps(manifest, indent=2, default=str))

    return {
        "services": len(C.SERVICES), "dependencies": len(C.dependencies()),
        "instances": len(instances), "deploys": len(deploys),
        "incidents": len(incidents), "runbooks": len(runbooks), "service_docs": len(docs),
        "log_entries": len(logs), "metric_points": len(metrics),
        "live_situations": len(live),
    }


def _labelled_retrieval_set(live: list[dict], incidents: list[dict]) -> list[dict]:
    """For each live situation, which historical incidents are genuinely relevant.

    Relevance is defined structurally rather than by hand: an incident is
    relevant to a live situation if it is the same archetype AND either the same
    service or a service in the same dependency neighbourhood. This gives
    Recall@5 a ground truth that does not depend on anyone's judgement.
    """
    out = []
    for sit in live:
        neighbourhood = {sit["service"], *C.upstream_chain(sit["service"], depth=1)}
        if sit["dep"]:
            neighbourhood.add(sit["dep"])
        relevant = [
            inc["id"] for inc in incidents
            if inc["_archetype"] == sit["archetype"] and inc["service"] in neighbourhood
        ]
        out.append({
            "situation_id": sit["id"],
            "query": sit["user_query"],
            "service": sit["service"],
            "archetype": sit["archetype"],
            "relevant_incident_ids": relevant,
            "relevant_count": len(relevant),
            # A situation with no relevant history is a REQUIRED test case, not a
            # gap: the agent must say "I found nothing useful" rather than
            # stretching an unrelated incident into an answer. Recall@5 is
            # undefined here and these are excluded from that average; they are
            # scored on abstention instead.
            "expect_no_useful_retrieval": len(relevant) == 0,
        })
    return out


def stats(conn: psycopg.Connection) -> None:
    tables = ["services", "service_dependencies", "service_instances", "deploys",
              "log_entries", "metric_points", "incidents", "runbooks", "service_docs", "chunks"]
    print("\n  table                     rows")
    print("  " + "-" * 32)
    for t in tables:
        n = conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        print(f"  {t:<24} {n:>7,}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="truncate and reload")
    parser.add_argument("--stats", action="store_true", help="report row counts and exit")
    args = parser.parse_args()

    from incidentiq.config import get_settings

    with psycopg.connect(get_settings().database_url, autocommit=False) as conn:
        if args.stats:
            stats(conn)
            return 0
        print(f"generating with seed {RANDOM_SEED} ...")
        counts = load(conn, force=args.force)
        conn.commit()
        for k, v in counts.items():
            print(f"  {k:<20} {v:>7,}")
        print(f"\nmanifest written to {SEEDS_DIR / 'live_situations.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
