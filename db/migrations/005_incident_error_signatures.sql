-- 005_incident_error_signatures.sql
--
-- Incidents as originally generated contained no error strings: the archetype's
-- log signatures went to `log_entries` but never into the incident write-up.
-- That is unrealistic - a real postmortem quotes the error it was diagnosed
-- from - and it had a measurable consequence. With no identifiers in the
-- corpus, the keyword half of hybrid retrieval had nothing distinctive to
-- match, and hybrid search could not outperform vector search alone.
--
-- Measured before this change (enriched query, service filter, incidents):
--     vector only   Recall@5 0.662   MRR 0.725
--     hybrid best   Recall@5 0.662   MRR 0.767
--
-- See docs/EVALS.md for the after figures.

ALTER TABLE incidents ADD COLUMN error_signatures TEXT[] NOT NULL DEFAULT '{}';

COMMENT ON COLUMN incidents.error_signatures IS
    'Log lines quoted in the incident write-up. The keyword half of hybrid '
    'retrieval depends on these: embeddings cannot represent an identifier '
    'like HikariPool-1, but full-text search matches it exactly.';
