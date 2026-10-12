-- Approved actions are sent to a control plane, can fail after approval, and
-- must never be applied twice.

-- The mock control plane's own ledger. It stands in for an external system, so
-- only incidentiq.controlplane writes here. The unique idempotency_key is what
-- makes a retry safe: a second request with the same key replays the stored
-- response instead of applying again.
CREATE TABLE control_plane_operations (
    id                  BIGSERIAL PRIMARY KEY,
    idempotency_key     TEXT        NOT NULL UNIQUE,
    operation           TEXT        NOT NULL,
    request_hash        TEXT        NOT NULL,   -- same key with a different body is refused
    params              JSONB       NOT NULL,
    response            JSONB       NOT NULL,
    applied_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- An investigation accepted by the API but not yet picked up by the worker.
ALTER TABLE investigations DROP CONSTRAINT IF EXISTS investigations_status_check;
ALTER TABLE investigations ADD CONSTRAINT investigations_status_check
    CHECK (status IN ('queued','running','awaiting_approval','complete','failed'));

ALTER TABLE actions DROP CONSTRAINT IF EXISTS actions_outcome_check;
ALTER TABLE actions ADD CONSTRAINT actions_outcome_check
    CHECK (outcome IN ('recorded','simulated_failure','executed','failed'));

ALTER TABLE actions ADD COLUMN idempotency_key TEXT;
ALTER TABLE actions ADD COLUMN attempts        INT  NOT NULL DEFAULT 1;
ALTER TABLE actions ADD COLUMN result          JSONB;
ALTER TABLE actions ADD COLUMN error           TEXT;

-- One execution record per approval. Partial, so rows written before this
-- migration (which have no key) are left as they are.
CREATE UNIQUE INDEX uq_actions_idempotency_key
    ON actions (idempotency_key) WHERE idempotency_key IS NOT NULL;
