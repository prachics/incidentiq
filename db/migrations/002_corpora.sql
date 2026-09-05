-- 002_corpora.sql
-- The three RAG corpora, plus the unified chunk table that carries embeddings.
--
-- DESIGN NOTE (see docs/DECISIONS.md #2):
-- Documents live in three typed tables so each corpus keeps its own natural
-- columns (an incident has a resolution; a runbook does not). Chunks live in
-- ONE table so retrieval is a single indexed query with a WHERE clause,
-- rather than a three-way UNION ALL that no index can serve efficiently.

-- ── Corpus 1: historical incidents ──────────────────────────
CREATE TABLE incidents (
    id              TEXT PRIMARY KEY,           -- e.g. 'INC-00042'
    title           TEXT        NOT NULL,
    service         TEXT        NOT NULL,
    severity        TEXT        NOT NULL CHECK (severity IN ('SEV1','SEV2','SEV3','SEV4')),
    occurred_at     TIMESTAMPTZ NOT NULL,
    resolved_at     TIMESTAMPTZ,
    symptoms        TEXT        NOT NULL,       -- what the on-call saw
    root_cause      TEXT        NOT NULL,
    resolution      TEXT        NOT NULL,       -- what actually fixed it
    tags            TEXT[]      NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_incidents_service  ON incidents (service);
CREATE INDEX idx_incidents_severity ON incidents (severity);
CREATE INDEX idx_incidents_occurred ON incidents (occurred_at DESC);

-- ── Corpus 2: runbooks ──────────────────────────────────────
CREATE TABLE runbooks (
    id              TEXT PRIMARY KEY,           -- e.g. 'RB-012'
    title           TEXT        NOT NULL,
    service         TEXT,                       -- NULL = applies to all services
    category        TEXT        NOT NULL,       -- 'database' | 'network' | 'deploy' | ...
    body            TEXT        NOT NULL,       -- the procedure, markdown
    applies_to      TEXT[]      NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_runbooks_service  ON runbooks (service);
CREATE INDEX idx_runbooks_category ON runbooks (category);

-- ── Corpus 3: service documentation ─────────────────────────
CREATE TABLE service_docs (
    id              TEXT PRIMARY KEY,           -- e.g. 'DOC-007'
    title           TEXT        NOT NULL,
    service         TEXT        NOT NULL,
    doc_kind        TEXT        NOT NULL,       -- 'architecture' | 'config' | 'sla'
    body            TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_service_docs_service ON service_docs (service);

-- ── Unified chunk table ─────────────────────────────────────
-- One row per chunk of one parent document, across all three corpora.
CREATE TABLE chunks (
    id              BIGSERIAL PRIMARY KEY,
    doc_type        TEXT        NOT NULL CHECK (doc_type IN ('incident','runbook','service_doc')),
    parent_doc_id   TEXT        NOT NULL,       -- FK by convention; targets 3 tables
    chunk_index     INT         NOT NULL,       -- 0-based position within the parent
    content         TEXT        NOT NULL,       -- the chunk text that gets embedded
    token_count     INT         NOT NULL,

    -- Metadata denormalised from the parent so filtering never needs a JOIN.
    -- This is what makes "filter before vector search" cheap.
    service         TEXT,
    severity        TEXT,
    doc_date        TIMESTAMPTZ,

    -- 384 dims = BAAI/bge-small-en-v1.5. Changing EMBEDDING_MODEL means
    -- changing this number and re-running the embed step.
    embedding       vector(384),

    -- Generated tsvector for the keyword half of hybrid retrieval.
    -- STORED = computed on write, so the GIN index below stays valid.
    content_tsv     tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (doc_type, parent_doc_id, chunk_index)
);

-- Vector index: HNSW with cosine distance.
-- HNSW is an approximate nearest-neighbour graph - much faster than a full
-- scan, at the cost of occasionally missing a true neighbour. That miss rate
-- is exactly what Recall@5 measures.
CREATE INDEX idx_chunks_embedding ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Keyword index: GIN over the generated tsvector.
CREATE INDEX idx_chunks_tsv ON chunks USING gin (content_tsv);

-- Metadata indexes, used by the pre-filter.
CREATE INDEX idx_chunks_service  ON chunks (service);
CREATE INDEX idx_chunks_doc_type ON chunks (doc_type);
CREATE INDEX idx_chunks_parent   ON chunks (doc_type, parent_doc_id);
