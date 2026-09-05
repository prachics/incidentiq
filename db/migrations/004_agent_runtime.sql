-- 004_agent_runtime.sql
-- Everything the running agent writes: investigation state, tool calls,
-- approvals, executed actions, and the immutable audit log.
--
-- DESIGN NOTE (see docs/DECISIONS.md #1):
-- Investigation state is persisted here, not held in process memory. This is
-- what makes the "state survives failures" claim demonstrable: kill the API
-- mid-investigation, restart it, and the agent resumes from the last
-- checkpoint rather than starting over.
--
-- LangGraph's own checkpointer creates its tables separately (see
-- scripts/migrate.py). This table is IncidentIQ's human-readable view of an
-- investigation - what the API and frontend read.

CREATE TABLE investigations (
    id                  TEXT PRIMARY KEY,       -- incident_id, e.g. 'IQ-2026-0007'
    user_query          TEXT        NOT NULL,
    status              TEXT        NOT NULL
                        CHECK (status IN ('running','awaiting_approval','complete','failed')),
    entities            JSONB       NOT NULL DEFAULT '{}',   -- extracted service, error sig, window
    scratchpad          JSONB       NOT NULL DEFAULT '[]',   -- reasoning trace, append-only
    iteration           INT         NOT NULL DEFAULT 0,
    retrieved_docs      JSONB       NOT NULL DEFAULT '[]',   -- [{chunk_id, score, doc_type, parent_doc_id}]
    final_summary       TEXT,
    failure_reason      TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_investigations_status ON investigations (status);
CREATE INDEX idx_investigations_created ON investigations (created_at DESC);

-- One row per tool invocation attempt, including failed attempts.
-- Tool success rate = COUNT(status='success') / COUNT(*) over this table.
CREATE TABLE tool_calls (
    id                  BIGSERIAL PRIMARY KEY,
    investigation_id    TEXT        NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
    tool_name           TEXT        NOT NULL,
    arguments           JSONB       NOT NULL,
    attempt             INT         NOT NULL DEFAULT 1,   -- 1-based; retries increment
    status              TEXT        NOT NULL
                        CHECK (status IN ('success','failed','timeout','malformed','partial')),
    result              JSONB,
    error               TEXT,
    latency_ms          INT,
    injected_failure    BOOLEAN     NOT NULL DEFAULT false, -- was this failure deliberate?
    started_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at         TIMESTAMPTZ
);
CREATE INDEX idx_tool_calls_investigation ON tool_calls (investigation_id, started_at);
CREATE INDEX idx_tool_calls_status ON tool_calls (status);

-- A pending or decided human decision on a write action.
CREATE TABLE approvals (
    id                  TEXT PRIMARY KEY,
    investigation_id    TEXT        NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
    proposed_tool       TEXT        NOT NULL,   -- e.g. 'rollback_deploy'
    proposed_arguments  JSONB       NOT NULL,
    reasoning           TEXT        NOT NULL,   -- why the agent proposes this
    evidence            JSONB       NOT NULL,   -- cited chunks + tool results behind it
    decision            TEXT        CHECK (decision IN ('approved','rejected','modified')),
    decided_by          TEXT,
    decided_at          TIMESTAMPTZ,
    rejection_reason    TEXT,                   -- fed back into the scratchpad
    modified_arguments  JSONB,                  -- set when decision='modified'
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_approvals_investigation ON approvals (investigation_id);
CREATE INDEX idx_approvals_pending ON approvals (created_at) WHERE decision IS NULL;

-- Write tools record INTENT here. Nothing in this table touches a real system.
CREATE TABLE actions (
    id                  BIGSERIAL PRIMARY KEY,
    investigation_id    TEXT        NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
    approval_id         TEXT        NOT NULL REFERENCES approvals(id) ON DELETE CASCADE,
    tool_name           TEXT        NOT NULL,
    arguments           JSONB       NOT NULL,
    outcome             TEXT        NOT NULL CHECK (outcome IN ('recorded','simulated_failure')),
    executed_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Append-only. No UPDATE, no DELETE - enforced by the trigger below.
CREATE TABLE audit_log (
    id                  BIGSERIAL PRIMARY KEY,
    investigation_id    TEXT        NOT NULL,
    event_type          TEXT        NOT NULL,   -- 'approval_requested' | 'approved' | 'rejected' | ...
    actor               TEXT        NOT NULL,   -- 'agent' | a user identifier
    payload             JSONB       NOT NULL,
    occurred_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_audit_investigation ON audit_log (investigation_id, occurred_at);

CREATE OR REPLACE FUNCTION reject_audit_mutation() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'audit_log is append-only; % is not permitted', TG_OP;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER audit_log_no_update
    BEFORE UPDATE OR DELETE ON audit_log
    FOR EACH ROW EXECUTE FUNCTION reject_audit_mutation();

-- Keep investigations.updated_at honest without the app having to remember.
CREATE OR REPLACE FUNCTION touch_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER investigations_touch
    BEFORE UPDATE ON investigations
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();
