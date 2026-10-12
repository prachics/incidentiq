-- Eval fixtures scale the corpus's own metric readings during an incident
-- instead of adding a second series beside them: hourly averages of the two
-- hid the anomaly. The reading's original value is kept here, so cleanup can
-- restore it, including the purge after a run that died.
ALTER TABLE eval_fixture_rows ADD COLUMN original_value DOUBLE PRECISION;
