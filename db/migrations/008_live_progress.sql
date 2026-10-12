-- The UI shows an investigation while it runs, so the readable tables are
-- mirrored after every node rather than once at the end of a run.
-- `events` is the per-node timeline (node, duration, outcome) already kept in
-- the checkpoint; `proposal` is the agent's conclusion, which exists even when
-- no approval is needed.
ALTER TABLE investigations ADD COLUMN events   JSONB NOT NULL DEFAULT '[]';
ALTER TABLE investigations ADD COLUMN proposal JSONB;
