"""Failure archetypes: the patterns incidents are generated from.

A corpus of 500 randomly-worded incidents would be useless — retrieval would
either match everything or nothing, and there would be no correct answer to
evaluate against. Instead each incident is an instance of one archetype applied
to one service, with the symptom, root cause, resolution, log lines, and metric
signature all derived from the same template.

That gives three properties the eval harness needs:

1. **A known correct answer.** The scenario's expected root cause is the
   archetype's root cause. Grading is not a judgement call.
2. **Genuine near-misses.** The same archetype on a different service produces a
   document that is topically similar but factually wrong for this incident.
   Retrieval that cannot tell them apart will score badly — as it should.
3. **Consistency.** An archetype that blames a dependency picks a real
   dependency of that service from the catalog.

Multiple phrasings per field keep the corpus from being trivially matchable on
one distinctive string.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Which service kinds an archetype can apply to.
APP_KINDS = ("app", "gateway")
STORE_KINDS = ("datastore",)
CACHE_KINDS = ("cache",)


@dataclass(frozen=True)
class Archetype:
    key: str
    name: str
    category: str                       # maps to runbook category
    applies_to: tuple[str, ...]         # service kinds
    # Templates. Available fields: {service} {dep} {version} {team}
    symptoms: tuple[str, ...]
    root_cause: tuple[str, ...]
    resolution: tuple[str, ...]
    log_lines: tuple[str, ...]          # ERROR/FATAL lines the tool will return
    metric: str                         # metric that moves
    metric_spike: float                 # multiple of baseline at peak
    tags: tuple[str, ...]
    # Remediation the agent should converge on. None = no write action needed.
    remediation_tool: str | None = None
    needs_dep: bool = False             # template references {dep}
    languages: tuple[str, ...] = field(default=())   # empty = any


ARCHETYPES: tuple[Archetype, ...] = (
    Archetype(
        key="db_pool_exhausted",
        name="Database connection pool exhausted",
        category="database",
        applies_to=APP_KINDS,
        needs_dep=True,
        symptoms=(
            "Requests to {service} began timing out after ~30s. Error rate climbed to 40% "
            "within eight minutes. No deploy in the preceding six hours.",
            "{service} started returning 503s under normal traffic. Latency p99 went from "
            "180ms to over 30s. Healthchecks flapping across all replicas.",
            "Alert fired on {service} error rate. Threads appear stuck; the pods are up but "
            "not serving. Traffic volume is unremarkable.",
        ),
        root_cause=(
            "Connection pool to {dep} was exhausted. A slow query introduced earlier in the "
            "week held connections open far longer than the pool timeout assumed, so under "
            "normal concurrency every connection was checked out and new requests queued "
            "until they timed out.",
            "All {dep} connections were checked out and never returned. A code path added "
            "in the previous release failed to close its connection on the error branch, so "
            "each failed request permanently leaked one connection from the pool.",
        ),
        resolution=(
            "Restarted {service} to reset the pool, then raised max_pool_size from 20 to 50 "
            "and set a 5s statement_timeout on {dep} so a slow query can no longer hold a "
            "connection indefinitely.",
            "Rolled back the release that introduced the leak, restarted {service} to drain "
            "the stuck pool, and added a pool-utilisation alert at 80% so the next occurrence "
            "is caught before saturation.",
        ),
        log_lines=(
            "HikariPool-1 - Connection is not available, request timed out after 30000ms",
            "FATAL: remaining connection slots are reserved for non-replication superuser connections",
            "could not acquire connection from pool within 30s (active=20, idle=0, waiting=147)",
            "timeout acquiring connection to {dep}: pool exhausted",
        ),
        metric="db_pool_active_connections",
        metric_spike=4.0,
        tags=("database", "connection-pool", "timeout", "saturation"),
        remediation_tool="restart_service",
    ),
    Archetype(
        key="memory_leak_oom",
        name="Memory leak leading to OOM kills",
        category="runtime",
        applies_to=APP_KINDS,
        symptoms=(
            "{service} pods are restarting every 20-40 minutes. Each restart drops in-flight "
            "requests. Memory climbs steadily from deploy until the kill.",
            "Intermittent 502s from {service}. Container restart count is climbing across all "
            "replicas; no obvious pattern in the failing requests.",
            "{service} replicas cycling. RSS grows linearly after each start and the kernel "
            "OOM killer terminates the process at the memory limit.",
        ),
        root_cause=(
            "An unbounded in-process cache in {service} retained every response body it saw. "
            "Memory grew linearly with request volume until the container hit its limit and "
            "was OOM-killed, at which point the cycle restarted.",
            "A listener registered per request was never unregistered, so each request added "
            "a permanently reachable object. Heap grew until the container limit was reached.",
        ),
        resolution=(
            "Rolled back to the previous version, which does not contain the unbounded cache. "
            "Follow-up work replaced it with an LRU capped at 10k entries and added a "
            "container-memory alert at 85% of limit.",
            "Rolled back the release, then patched the listener registration to unregister in "
            "a finally block. Raised the memory limit as a temporary buffer while the fix shipped.",
        ),
        log_lines=(
            "java.lang.OutOfMemoryError: Java heap space",
            "Container killed due to memory limit (OOMKilled), restarting",
            "GC overhead limit exceeded - 98% of time in GC, 2% heap recovered",
            "fatal error: runtime: out of memory",
        ),
        metric="memory_usage_pct",
        metric_spike=1.9,
        tags=("memory", "oom", "restart", "leak"),
        remediation_tool="rollback_deploy",
    ),
    Archetype(
        key="bad_deploy_regression",
        name="Regression introduced by a deploy",
        category="deploy",
        applies_to=APP_KINDS,
        symptoms=(
            "Error rate on {service} jumped from 0.2% to 18% at 14:03, which lines up exactly "
            "with the {version} rollout. No infrastructure changes in that window.",
            "{service} started returning 500s on a subset of requests immediately after "
            "deploy. Roughly one request in five fails; the rest are fine.",
            "Customer reports of failed operations against {service}. Onset is sharp and "
            "coincides with a release, not a gradual degradation.",
        ),
        root_cause=(
            "Release {version} of {service} changed a response field from a string to an "
            "object without a compatibility shim. Callers still on the old contract failed to "
            "deserialise it, producing 500s for exactly the fraction of traffic on old clients.",
            "Release {version} introduced an unguarded null dereference on a rarely-populated "
            "optional field. Requests that happened to include it hit the exception path.",
        ),
        resolution=(
            "Rolled {service} back to the prior version. Error rate returned to baseline "
            "within 90 seconds. The change was reshipped behind a feature flag with the "
            "compatibility shim in place.",
            "Rolled back to the previous release and added a contract test covering the "
            "optional field so the regression cannot reach production again.",
        ),
        log_lines=(
            "JsonMappingException: cannot deserialize value of type String from Object",
            "NullPointerException at com.shop.{service}.handler.process(Handler.java:214)",
            "panic: interface conversion: interface {{}} is nil, not map[string]interface {{}}",
            "500 Internal Server Error - unhandled exception in request pipeline",
        ),
        metric="error_rate",
        metric_spike=90.0,
        tags=("deploy", "regression", "rollback", "release"),
        remediation_tool="rollback_deploy",
    ),
    Archetype(
        key="downstream_timeout_cascade",
        name="Downstream dependency timeout cascading upstream",
        category="network",
        applies_to=APP_KINDS,
        needs_dep=True,
        symptoms=(
            "{service} latency p99 went from 200ms to 12s. Its own CPU and memory are normal. "
            "Errors are all timeouts rather than exceptions.",
            "Requests through {service} hanging. The service looks healthy by every local "
            "metric, but almost every request is slow.",
            "{service} thread pool saturated. Threads are all in a waiting state rather than "
            "running, which points outward rather than inward.",
        ),
        root_cause=(
            "{dep} became slow, and {service} had no timeout configured on calls to it. "
            "Request threads blocked indefinitely waiting on {dep}, saturating {service}'s "
            "pool and making {service} appear to be the failure when it was the victim.",
            "Degradation in {dep} propagated upward because {service} retried failed calls "
            "three times with no backoff, tripling load on an already-struggling dependency "
            "and turning a partial outage into a full one.",
        ),
        resolution=(
            "Addressed the root problem in {dep}, which cleared {service} immediately. Added "
            "a 2s timeout and a circuit breaker on the {dep} client so the next {dep} "
            "degradation degrades {service} rather than taking it down.",
            "Recovered {dep}, then replaced the retry loop with exponential backoff and a "
            "concurrency limit on {dep} calls to stop {service} amplifying the next failure.",
        ),
        log_lines=(
            "context deadline exceeded calling {dep}",
            "upstream request timeout: {dep} did not respond within 30000ms",
            "circuit breaker OPEN for {dep} after 20 consecutive failures",
            "read timed out waiting for {dep} response",
        ),
        metric="latency_p99_ms",
        metric_spike=60.0,
        tags=("cascade", "timeout", "dependency", "circuit-breaker"),
        remediation_tool=None,
    ),
    Archetype(
        key="cache_stampede",
        name="Cache stampede after eviction",
        category="cache",
        applies_to=APP_KINDS,
        needs_dep=True,
        symptoms=(
            "Sharp load spike on {service} every time cached entries expire. Latency spikes "
            "for 30-60 seconds, recovers, then repeats on the next expiry boundary.",
            "{service} sees periodic bursts of database load with no corresponding traffic "
            "increase. The pattern is regular, roughly on the cache TTL interval.",
        ),
        root_cause=(
            "All {dep} entries were written with the same TTL, so they expired simultaneously. "
            "Every request that missed then recomputed the same value concurrently, hammering "
            "the backing store — a thundering herd on each expiry boundary.",
            "A cache flush during deploy left {dep} empty. Every in-flight request missed and "
            "recomputed at once rather than the first one populating the entry for the rest.",
        ),
        resolution=(
            "Added jitter to the TTL so entries expire over a spread rather than together, "
            "and introduced single-flight so concurrent misses for the same key wait on one "
            "recomputation instead of all doing it.",
            "Switched to a warm-on-deploy strategy for {dep} and added request coalescing so "
            "an empty cache no longer converts directly into backend load.",
        ),
        log_lines=(
            "cache miss storm: {dep} miss rate 98% over 30s window",
            "recompute queue depth 4200, workers 16 - falling behind",
            "backend load spike correlated with {dep} eviction cycle",
        ),
        metric="cache_hit_rate",
        metric_spike=0.05,
        tags=("cache", "stampede", "ttl", "thundering-herd"),
        remediation_tool=None,
    ),
    Archetype(
        key="disk_full",
        name="Disk exhaustion on a datastore",
        category="database",
        applies_to=STORE_KINDS,
        symptoms=(
            "Writes to {service} are failing. Reads still work. The failure is total for "
            "anything that needs to persist.",
            "{service} rejecting transactions with a disk-related error. Onset was sudden "
            "after a long, gradual climb in volume usage.",
        ),
        root_cause=(
            "The data volume on {service} reached 100%. WAL segments accumulated faster than "
            "they were archived because the archive command had been failing silently for "
            "nine days, and nothing alerted on archive failure.",
            "Autovacuum had been unable to keep up on the largest table in {service}, so dead "
            "tuples accumulated until the volume filled.",
        ),
        resolution=(
            "Cleared archived WAL segments to recover space, fixed the archive command, and "
            "added an alert on both volume utilisation at 80% and archive-command failure, "
            "since the second is what made the first a surprise.",
            "Ran a manual vacuum full on the affected table during a maintenance window, "
            "expanded the volume, and tuned autovacuum thresholds for the table's write rate.",
        ),
        log_lines=(
            "PANIC: could not write to file \"pg_wal/000000010000004A000000FF\": No space left on device",
            "ERROR: could not extend file \"base/16384/2619\": No space left on device",
            "archive command failed with exit code 1 (repeated 12847 times)",
        ),
        metric="disk_usage_pct",
        metric_spike=1.4,
        tags=("database", "disk", "wal", "capacity"),
        remediation_tool="scale_service",
    ),
    Archetype(
        key="replica_lag",
        name="Read replica lag serving stale data",
        category="database",
        applies_to=STORE_KINDS,
        symptoms=(
            "Users report seeing stale data from {service} — an update succeeds, then the "
            "next read shows the old value. Intermittent, roughly half of reads.",
            "Inconsistent reads against {service}. Writes confirm successfully but do not "
            "appear on subsequent queries for several minutes.",
        ),
        root_cause=(
            "Replication lag on {service} reached 240 seconds. A bulk backfill job saturated "
            "the primary's write throughput, and the replicas could not apply WAL as fast as "
            "it was produced. Read traffic load-balanced across replicas therefore returned "
            "data up to four minutes old.",
            "A long-running analytical query on the replica blocked WAL replay, so the replica "
            "fell progressively further behind while continuing to serve reads.",
        ),
        resolution=(
            "Throttled the backfill job to run in smaller batches with pauses, which let "
            "replicas catch up within 15 minutes. Added a lag alert at 30s and configured the "
            "read router to remove a replica from rotation when its lag exceeds 10s.",
            "Terminated the blocking query and set a statement timeout on the replica so "
            "analytical work can no longer stall replay.",
        ),
        log_lines=(
            "replica lag 240s exceeds threshold 30s",
            "WAL receiver falling behind: last_replay_lsn 4A/FF000000, primary at 4B/12000000",
            "canceling statement due to conflict with recovery",
        ),
        metric="replication_lag_seconds",
        metric_spike=24.0,
        tags=("database", "replication", "lag", "consistency"),
        remediation_tool=None,
    ),
    Archetype(
        key="cert_expiry",
        name="TLS certificate expiry",
        category="network",
        applies_to=("gateway", "adapter"),
        symptoms=(
            "All outbound calls from {service} failing with TLS errors, starting exactly on "
            "the hour. Total failure, not partial.",
            "{service} cannot establish connections to its upstream. The failure began "
            "abruptly with no deploy and no traffic change.",
        ),
        root_cause=(
            "The TLS certificate used by {service} expired. Automated renewal had been "
            "disabled during a migration eight months earlier and was never re-enabled, and "
            "the expiry monitor was watching the load balancer's certificate rather than "
            "this one.",
            "An intermediate CA certificate in {service}'s trust chain expired. The leaf "
            "certificate was still valid, which is why expiry monitoring did not fire.",
        ),
        resolution=(
            "Issued and deployed a replacement certificate, restoring service in 12 minutes. "
            "Re-enabled automated renewal and extended expiry monitoring to cover every "
            "certificate in the chain rather than leaf certificates only.",
            "Updated the trust bundle with the current intermediate and added the full chain "
            "to the expiry monitor.",
        ),
        log_lines=(
            "x509: certificate has expired or is not yet valid",
            "SSL handshake failed: certificate verify failed (certificate has expired)",
            "tls: failed to verify certificate chain for upstream",
        ),
        metric="error_rate",
        metric_spike=100.0,
        tags=("tls", "certificate", "expiry", "network"),
        remediation_tool=None,
    ),
    Archetype(
        key="upstream_rate_limit",
        name="Third-party rate limit exceeded",
        category="external",
        applies_to=("adapter",),
        symptoms=(
            "{service} returning 429s from the provider. Failures are bursty and correlate "
            "with traffic peaks rather than being constant.",
            "Intermittent failures through {service} during peak hours only. Off-peak is "
            "completely clean.",
        ),
        root_cause=(
            "The provider's rate limit was exceeded during peak traffic. A retry loop with no "
            "backoff turned each rate-limited request into four, so the limit was breached "
            "further the harder the system tried to recover.",
            "A batch job began running during peak hours after a schedule change, consuming "
            "the same rate-limit quota as live customer traffic.",
        ),
        resolution=(
            "Added a token-bucket limiter in {service} sized just under the provider's quota, "
            "and replaced the retry loop with exponential backoff honouring the Retry-After "
            "header. Peak-hour failures went to zero.",
            "Moved the batch job back to off-peak hours and gave it a separate, lower-priority "
            "quota so it cannot starve live traffic.",
        ),
        log_lines=(
            "429 Too Many Requests - rate limit exceeded, retry after 30s",
            "provider quota exhausted: 10000/10000 requests in current window",
            "retrying request (attempt 4/4) after 429",
        ),
        metric="error_rate",
        metric_spike=35.0,
        tags=("rate-limit", "external", "retry", "backoff"),
        remediation_tool=None,
    ),
    Archetype(
        key="consumer_lag",
        name="Event consumer falling behind",
        category="queue",
        applies_to=APP_KINDS,
        needs_dep=True,
        symptoms=(
            "Downstream effects of {service} are delayed by tens of minutes. The work "
            "eventually completes, but far too late.",
            "{service} consumer lag climbing steadily. No errors — it is simply not keeping "
            "up with the incoming rate.",
        ),
        root_cause=(
            "{service} consumers could not keep pace with producer throughput on {dep}. A "
            "per-message synchronous call added in the last release roughly tripled "
            "per-message processing time, pushing consumption below the production rate.",
            "Two of three consumer instances of {service} were unhealthy and had dropped out "
            "of the group, leaving a single consumer to handle all partitions.",
        ),
        resolution=(
            "Scaled {service} consumers from 3 to 9 to drain the backlog, then batched the "
            "synchronous call so it happens once per 100 messages instead of once per "
            "message. Lag returned to under a second.",
            "Restored the unhealthy consumers and added an alert on consumer group size so a "
            "silent drop-out is caught immediately.",
        ),
        log_lines=(
            "consumer group lag: 1847293 messages behind on topic order-events",
            "rebalance triggered: 2 members left the group",
            "processing rate 120 msg/s below production rate 890 msg/s",
        ),
        metric="consumer_lag_messages",
        metric_spike=50.0,
        tags=("kafka", "queue", "lag", "throughput"),
        remediation_tool="scale_service",
    ),
    Archetype(
        key="thread_pool_saturation",
        name="Thread pool saturation",
        category="runtime",
        applies_to=APP_KINDS,
        languages=("java",),
        symptoms=(
            "{service} accepting connections but not responding. CPU is low, which rules out "
            "a compute bottleneck.",
            "Requests to {service} queueing without being served. Load average is near zero "
            "while the request queue grows.",
        ),
        root_cause=(
            "The request thread pool in {service} was fully occupied by threads blocked on a "
            "synchronous call, with no timeout. Because the threads were waiting rather than "
            "running, CPU stayed low and every CPU-based alert stayed silent.",
            "A deadlock between two locks acquired in inconsistent order gradually consumed "
            "the pool as more requests hit the same path.",
        ),
        resolution=(
            "Restarted {service} to clear the blocked pool, then added timeouts to the "
            "blocking call and a bulkhead limiting how much of the pool any one dependency "
            "can occupy. Added an alert on pool utilisation, since CPU alerts cannot see this.",
            "Fixed the lock ordering, restarted the service, and added a deadlock detector to "
            "the health check.",
        ),
        log_lines=(
            "Thread pool exhausted: active=200, queued=5000, rejected=1247",
            "RejectedExecutionException: Task rejected from ThreadPoolExecutor",
            "Found one Java-level deadlock: thread http-nio-8080-exec-42",
        ),
        metric="thread_pool_active",
        metric_spike=5.0,
        tags=("threads", "saturation", "deadlock", "runtime"),
        remediation_tool="restart_service",
    ),
    Archetype(
        key="config_drift",
        name="Configuration drift between environments",
        category="config",
        applies_to=APP_KINDS,
        symptoms=(
            "{service} behaving differently across replicas — some requests succeed, others "
            "fail, with no pattern in the request itself.",
            "Roughly a third of requests to {service} fail. Which third appears random, and "
            "retries usually succeed.",
        ),
        root_cause=(
            "A config change was applied to two of three replicas of {service} and not the "
            "third. Requests routed to the stale replica used the old value and failed. The "
            "apparent randomness was just load balancing.",
            "An environment variable was set in the deployment manifest but absent from the "
            "canary overlay, so canary pods fell back to a default that pointed at the wrong "
            "endpoint.",
        ),
        resolution=(
            "Reapplied the configuration to all replicas and restarted {service} to pick it "
            "up. Added a config-hash label to pods and an alert when replicas of the same "
            "deployment disagree.",
            "Corrected the canary overlay and added a required-variable check to the startup "
            "path so a missing value fails fast instead of silently defaulting.",
        ),
        log_lines=(
            "config checksum mismatch: expected a3f9c1, got 7b2e88",
            "connecting to legacy endpoint (deprecated) - CONFIG_ENDPOINT unset, using default",
            "configuration reload failed: key 'timeout_ms' not found",
        ),
        metric="error_rate",
        metric_spike=30.0,
        tags=("config", "drift", "deploy", "consistency"),
        remediation_tool="restart_service",
    ),
    Archetype(
        key="search_shard_unassigned",
        name="Search cluster shards unassigned",
        category="search",
        applies_to=("search",),
        symptoms=(
            "Search results from {service} are incomplete — some products missing entirely. "
            "Queries succeed, so nothing errors.",
            "{service} cluster health is yellow. Queries return, but result counts are lower "
            "than expected.",
        ),
        root_cause=(
            "Two primary shards on {service} were unassigned after a node left the cluster "
            "and disk-based allocation rules prevented reassignment — the remaining nodes had "
            "crossed the high watermark. Queries silently returned partial results rather "
            "than failing.",
            "A rolling restart proceeded faster than shard recovery, so multiple nodes holding "
            "copies of the same shard were down simultaneously.",
        ),
        resolution=(
            "Freed disk on the remaining nodes to drop below the watermark, which let shards "
            "reallocate automatically. Set search requests to fail rather than return partial "
            "results, so incomplete data surfaces as an error instead of quiet wrongness.",
            "Paused the rolling restart until green, then added a health gate between nodes.",
        ),
        log_lines=(
            "cluster health status YELLOW - 2 unassigned shards",
            "low disk watermark [85%] exceeded on node es-data-3, shards will be relocated",
            "search returned partial results: 3 of 5 shards responded",
        ),
        metric="unassigned_shards",
        metric_spike=2.0,
        tags=("search", "elasticsearch", "shards", "capacity"),
        remediation_tool="scale_service",
    ),
    Archetype(
        key="traffic_surge",
        name="Traffic surge beyond provisioned capacity",
        category="capacity",
        applies_to=APP_KINDS,
        symptoms=(
            "{service} latency degraded steadily as traffic climbed. Errors began once CPU "
            "saturated across all replicas.",
            "Marketing campaign drove a 6x traffic increase. {service} degraded gracefully at "
            "first, then began shedding load.",
        ),
        root_cause=(
            "Traffic exceeded {service}'s provisioned capacity. Autoscaling was configured on "
            "CPU with a five-minute stabilisation window, so it reacted far too slowly for a "
            "spike that arrived in under a minute.",
            "A campaign launched without capacity planning notice. {service} was provisioned "
            "for typical peak, not six times it.",
        ),
        resolution=(
            "Scaled {service} manually to absorb the surge, then lowered the autoscaling "
            "target and shortened the stabilisation window. Added a pre-scale step to the "
            "campaign launch checklist.",
            "Scaled out and added a scheduled scale-up ahead of known campaign windows.",
        ),
        log_lines=(
            "CPU throttling: container exceeded quota, throttled 4200ms in last 10s",
            "request queue depth 8400 exceeds capacity, shedding load",
            "503 Service Unavailable - upstream at capacity",
        ),
        metric="cpu_pct",
        metric_spike=3.2,
        tags=("capacity", "autoscaling", "traffic", "saturation"),
        remediation_tool="scale_service",
    ),
    Archetype(
        key="cache_node_eviction",
        name="Cache node evicting under memory pressure",
        category="cache",
        applies_to=CACHE_KINDS,
        symptoms=(
            "Hit rate on {service} dropped from 94% to under 40%. Backing stores are seeing "
            "a proportional load increase.",
            "{service} evicting keys far faster than expected. Clients are functionally "
            "correct but every read is now a miss.",
        ),
        root_cause=(
            "{service} reached its maxmemory limit and the allkeys-lru policy began evicting "
            "hot keys. A new key pattern introduced upstream roughly doubled the working set "
            "without any change to the memory allocation.",
            "A single oversized key on {service} consumed a disproportionate share of the "
            "node's memory, forcing eviction of the entries that actually mattered.",
        ),
        resolution=(
            "Raised maxmemory and set a TTL on the new key pattern so it cannot grow "
            "unbounded. Added a working-set-size alert so the next growth is visible before "
            "it becomes eviction.",
            "Removed the oversized key and added a value-size guard in the client so writes "
            "above 1MB are rejected rather than accepted and silently harmful.",
        ),
        log_lines=(
            "evicted_keys rate 8400/s, maxmemory policy allkeys-lru",
            "OOM command not allowed when used memory > 'maxmemory'",
            "used_memory 15.8G approaching maxmemory 16G",
        ),
        metric="cache_hit_rate",
        metric_spike=0.35,
        tags=("cache", "redis", "eviction", "memory"),
        remediation_tool="scale_service",
    ),
    Archetype(
        key="broker_partition_offline",
        name="Broker partition offline",
        category="queue",
        applies_to=("queue",),
        symptoms=(
            "Producers writing to {service} are failing for a subset of partitions. Other "
            "partitions are completely unaffected.",
            "Intermittent publish failures against {service}. Which messages fail appears "
            "to depend on the partition key.",
        ),
        root_cause=(
            "A broker in the {service} cluster went offline and the partitions it led had no "
            "in-sync replica to take over, because min.insync.replicas had been lowered "
            "during an earlier incident and never restored. Those partitions became "
            "unavailable for writes.",
            "Disk pressure on one {service} broker caused it to drop out of the ISR for its "
            "partitions, and leadership could not move while it was still nominally alive.",
        ),
        resolution=(
            "Restored the failed broker and let leadership rebalance. Reset "
            "min.insync.replicas to 2 and added an alert on under-replicated partitions, "
            "which is the signal that predicts this rather than reports it.",
            "Freed disk on the affected broker, which let it rejoin the ISR. Added a "
            "per-broker disk alert at 80%.",
        ),
        log_lines=(
            "NOT_ENOUGH_REPLICAS: messages rejected, isr=1 min.insync.replicas=2",
            "partition order-events-7 has no leader, producer requests failing",
            "broker 3 removed from ISR for 14 partitions",
        ),
        metric="under_replicated_partitions",
        metric_spike=14.0,
        tags=("kafka", "queue", "replication", "broker"),
        remediation_tool="restart_service",
    ),
)

BY_KEY: dict[str, Archetype] = {a.key: a for a in ARCHETYPES}


def for_service(kind: str, language: str) -> list[Archetype]:
    """Archetypes that can plausibly apply to a service of this kind/language."""
    return [
        a for a in ARCHETYPES
        if kind in a.applies_to and (not a.languages or language in a.languages)
    ]


CATEGORIES: tuple[str, ...] = tuple(sorted({a.category for a in ARCHETYPES}))
