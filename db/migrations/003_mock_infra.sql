-- 003_mock_infra.sql
-- The fake production estate the read-only diagnostic tools query.
--
-- Nothing here talks to a real system. `get_service_logs` reads log_entries,
-- `get_metrics` reads metric_points, and so on. This keeps the project fully
-- self-contained (`docker compose up` gives you a working world) while the
-- tool interfaces stay identical to ones that would hit Datadog or Loki.

CREATE TABLE services (
    name            TEXT PRIMARY KEY,
    tier            TEXT        NOT NULL CHECK (tier IN ('tier-1','tier-2','tier-3')),
    language        TEXT        NOT NULL,
    owner_team      TEXT        NOT NULL,
    replica_count   INT         NOT NULL DEFAULT 3,
    description     TEXT        NOT NULL DEFAULT ''
);

-- The dependency graph. `from_service` calls `to_service`.
-- Cascading-failure scenarios walk these edges.
CREATE TABLE service_dependencies (
    from_service    TEXT NOT NULL REFERENCES services(name) ON DELETE CASCADE,
    to_service      TEXT NOT NULL REFERENCES services(name) ON DELETE CASCADE,
    kind            TEXT NOT NULL CHECK (kind IN ('sync','async','datastore')),
    PRIMARY KEY (from_service, to_service),
    CHECK (from_service <> to_service)
);
CREATE INDEX idx_deps_to ON service_dependencies (to_service);

CREATE TABLE service_instances (
    id              TEXT PRIMARY KEY,           -- e.g. 'checkout-service-7d9f-a1b2'
    service         TEXT        NOT NULL REFERENCES services(name) ON DELETE CASCADE,
    zone            TEXT        NOT NULL,
    status          TEXT        NOT NULL CHECK (status IN ('healthy','degraded','crashlooping','terminated')),
    started_at      TIMESTAMPTZ NOT NULL
);
CREATE INDEX idx_instances_service ON service_instances (service);

CREATE TABLE deploys (
    id              TEXT PRIMARY KEY,           -- e.g. 'DEP-01843'
    service         TEXT        NOT NULL REFERENCES services(name) ON DELETE CASCADE,
    version         TEXT        NOT NULL,       -- e.g. 'v2.14.1'
    previous_version TEXT,
    deployed_at     TIMESTAMPTZ NOT NULL,
    deployed_by     TEXT        NOT NULL,
    status          TEXT        NOT NULL CHECK (status IN ('succeeded','failed','rolled_back','in_progress')),
    changelog       TEXT        NOT NULL DEFAULT ''
);
CREATE INDEX idx_deploys_service_time ON deploys (service, deployed_at DESC);

CREATE TABLE log_entries (
    id              BIGSERIAL PRIMARY KEY,
    service         TEXT        NOT NULL REFERENCES services(name) ON DELETE CASCADE,
    ts              TIMESTAMPTZ NOT NULL,
    level           TEXT        NOT NULL CHECK (level IN ('DEBUG','INFO','WARN','ERROR','FATAL')),
    message         TEXT        NOT NULL,
    trace_id        TEXT
);
-- Composite index matching the tool's access pattern:
-- WHERE service = ? AND ts BETWEEN ? AND ? AND level >= ?
CREATE INDEX idx_logs_service_ts ON log_entries (service, ts DESC);
CREATE INDEX idx_logs_level      ON log_entries (level) WHERE level IN ('ERROR','FATAL');

CREATE TABLE metric_points (
    id              BIGSERIAL PRIMARY KEY,
    service         TEXT        NOT NULL REFERENCES services(name) ON DELETE CASCADE,
    metric_name     TEXT        NOT NULL,       -- 'latency_p99_ms' | 'error_rate' | 'cpu_pct' | ...
    ts              TIMESTAMPTZ NOT NULL,
    value           DOUBLE PRECISION NOT NULL
);
CREATE INDEX idx_metrics_lookup ON metric_points (service, metric_name, ts DESC);
