-- 006_eval_fixtures.sql
--
-- Durable bookkeeping for evaluation fixture rows.
--
-- The eval harness plants evidence into log_entries, metric_points and deploys
-- before each scenario and removes it afterwards. The row ids lived in a Python
-- object, which is fine until the process does not exit cleanly - a SIGKILL
-- skips the cleanup and orphans every row it was holding.
--
-- That happened, and the consequence is worse than untidy: leaked evidence for
-- one service silently becomes background noise for every later scenario, and
-- the eval numbers drift without anything failing. log_entries and deploys were
-- at least identifiable by their markers; metric_points had none, so ~153 rows
-- could not be distinguished from the base corpus at all.
--
-- Cleanup state therefore lives in the database, next to the rows it describes,
-- and the harness purges orphans at startup.

CREATE TABLE eval_fixture_rows (
    id              BIGSERIAL PRIMARY KEY,
    run_id          TEXT        NOT NULL,   -- one eval process
    scenario_id     TEXT        NOT NULL,
    target_table    TEXT        NOT NULL CHECK (target_table IN
                                ('log_entries','metric_points','deploys')),
    target_id       TEXT        NOT NULL,   -- text, since deploys.id is not numeric
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_eval_fixture_run ON eval_fixture_rows (run_id);
CREATE INDEX idx_eval_fixture_scenario ON eval_fixture_rows (scenario_id);
