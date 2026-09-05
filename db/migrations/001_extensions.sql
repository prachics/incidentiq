-- 001_extensions.sql
-- Postgres extensions IncidentIQ depends on.
--
-- vector   : pgvector. Adds the `vector(N)` column type plus similarity
--            operators (<=> cosine, <-> L2, <#> inner product) and the
--            HNSW / IVFFlat index types that make them fast.
-- pg_trgm  : trigram matching. Used for fuzzy service-name lookup
--            ("checkout-svc" vs "checkout-service") during entity extraction.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
