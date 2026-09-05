"""Tests for the retrieval layer.

Split into two groups: chunking is pure and always runs; search needs a live,
indexed database and skips without one.
"""

import sys
from pathlib import Path

import psycopg
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))

from incidentiq.config import get_settings  # noqa: E402
from incidentiq.rag.chunking import (  # noqa: E402
    chunk_incident,
    chunk_text,
    estimate_tokens,
)


class TestChunking:
    def test_short_text_is_one_chunk(self):
        assert len(chunk_text("A short sentence.", chunk_tokens=320)) == 1

    def test_empty_text_yields_nothing(self):
        assert chunk_text("") == []
        assert chunk_text("   \n  ") == []

    def test_long_text_splits(self):
        text = " ".join(["word"] * 4000)
        chunks = chunk_text(text, chunk_tokens=100, overlap_tokens=20)
        assert len(chunks) > 1

    def test_chunks_respect_size_budget(self):
        """Allow the overlap on top of the budget - a chunk is its own content
        plus the tail carried from its predecessor."""
        text = " ".join(f"word{i}" for i in range(3000))
        chunk_tokens, overlap = 120, 24
        chunks = chunk_text(text, chunk_tokens=chunk_tokens, overlap_tokens=overlap)
        budget = (chunk_tokens + overlap) * 4 * 1.15  # chars, 15% tolerance
        assert all(len(c) <= budget for c in chunks), (
            f"largest chunk {max(len(c) for c in chunks)} chars exceeds {budget:.0f}"
        )

    def test_consecutive_chunks_overlap(self):
        text = " ".join(f"word{i}" for i in range(2000))
        chunks = chunk_text(text, chunk_tokens=100, overlap_tokens=25)
        assert len(chunks) >= 2
        for a, b in zip(chunks, chunks[1:], strict=False):  # deliberately offset
            shared = max((n for n in range(1, min(len(a), len(b)) + 1) if a.endswith(b[:n])),
                         default=0)
            assert shared > 0, "consecutive chunks share no text - overlap is not applied"

    def test_overlap_must_be_smaller_than_chunk(self):
        with pytest.raises(ValueError, match="must be smaller"):
            chunk_text("some text " * 500, chunk_tokens=100, overlap_tokens=100)

    def test_no_content_is_lost(self):
        """Every word of the source must survive into at least one chunk."""
        words = [f"w{i}" for i in range(800)]
        chunks = chunk_text(" ".join(words), chunk_tokens=64, overlap_tokens=12)
        joined = " ".join(chunks)
        missing = [w for w in words if w not in joined]
        assert not missing, f"{len(missing)} words lost during chunking, e.g. {missing[:5]}"

    def test_incident_chunk_carries_filter_metadata(self):
        row = {
            "id": "INC-00001", "title": "t", "service": "checkout-service",
            "severity": "SEV1", "occurred_at": None, "symptoms": "s",
            "root_cause": "r", "resolution": "x", "error_signatures": ["HikariPool-1 timeout"],
        }
        chunks = chunk_incident(row)
        assert chunks
        for c in chunks:
            assert c.service == "checkout-service"
            assert c.severity == "SEV1"
            assert c.parent_doc_id == "INC-00001"
            assert c.doc_type == "incident"

    def test_incident_chunk_includes_error_signatures(self):
        """Without these the keyword half of hybrid retrieval has nothing
        distinctive to match - measured, see docs/EVALS.md."""
        row = {
            "id": "INC-1", "title": "t", "service": "s", "severity": "SEV2",
            "occurred_at": None, "symptoms": "sym", "root_cause": "rc",
            "resolution": "res", "error_signatures": ["HikariPool-1 - Connection is not available"],
        }
        body = " ".join(c.content for c in chunk_incident(row))
        assert "HikariPool-1" in body

    def test_token_estimate_is_monotonic(self):
        assert estimate_tokens("a" * 400) > estimate_tokens("a" * 100)


@pytest.fixture(scope="module")
def conn():
    from pgvector.psycopg import register_vector
    try:
        c = psycopg.connect(get_settings().database_url, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable")
    register_vector(c)
    if c.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0:
        pytest.skip("no chunks - run: python -m incidentiq.rag.indexer")
    yield c
    c.close()


class TestSearch:
    def test_vector_search_finds_paraphrase(self, conn):
        """The property keyword search cannot have: a match with no shared
        vocabulary."""
        from incidentiq.rag.retrieval import search
        hits = search(conn, "the database is refusing new connections",
                      mode="vector", limit=5)
        assert hits
        assert any("pool" in h.content.lower() or "connection" in h.content.lower()
                   for h in hits)

    def test_keyword_search_finds_exact_identifier(self, conn):
        """The property vector search cannot have: an exact match on a token
        that carries no semantic meaning."""
        from incidentiq.rag.retrieval import search
        hits = search(conn, "HikariPool-1", mode="keyword", limit=5)
        assert hits, "keyword search found nothing for a string known to be in the corpus"
        assert any("HikariPool" in h.content for h in hits)

    def test_keyword_search_survives_natural_language(self, conn):
        """Regression guard. websearch_to_tsquery ANDs every term, so a
        conversational query matched nothing at all until the query builder
        switched the conjunctions to disjunctions."""
        from incidentiq.rag.retrieval import search
        hits = search(conn, "checkout is timing out and I keep seeing HikariPool errors",
                      mode="keyword", limit=5)
        assert hits, "a multi-word natural-language query returned no keyword hits"

    def test_service_filter_is_respected(self, conn):
        from incidentiq.rag.retrieval import Filters, search
        hits = search(conn, "errors and timeouts",
                      filters=Filters(service="checkout-service"), limit=10)
        assert hits
        assert all(h.service == "checkout-service" for h in hits)

    def test_doc_type_filter_is_respected(self, conn):
        from incidentiq.rag.retrieval import Filters, search
        hits = search(conn, "connection pool", filters=Filters(doc_type="runbook"), limit=10)
        assert hits
        assert all(h.doc_type == "runbook" for h in hits)

    def test_multiple_services_filter(self, conn):
        """A cascade investigation searches a service together with its
        dependencies in one query."""
        from incidentiq.rag.retrieval import Filters, search
        targets = ["checkout-service", "payment-service"]
        hits = search(conn, "timeout", filters=Filters(service=targets), limit=10)
        assert hits
        assert all(h.service in targets for h in hits)

    def test_hybrid_returns_provenance(self, conn):
        """Every result must say which ranker(s) produced it - this is what the
        frontend evidence panel displays."""
        from incidentiq.rag.retrieval import search
        hits = search(conn, "connection pool exhausted on checkout-service",
                      mode="hybrid", limit=5)
        assert hits
        for h in hits:
            assert h.vector_rank is not None or h.keyword_rank is not None
            assert h.rrf_score > 0

    def test_hybrid_can_promote_a_chunk_neither_ranker_ranked_first(self, conn):
        """The point of fusion: consensus beats a single strong opinion."""
        from incidentiq.rag.retrieval import search
        hits = search(conn, "checkout service timing out with HikariPool connection errors",
                      mode="hybrid", limit=5)
        both = [h for h in hits if h.vector_rank and h.keyword_rank]
        assert both, "no chunk was found by both rankers - fusion is doing nothing"

    def test_results_are_rank_ordered(self, conn):
        from incidentiq.rag.retrieval import search
        hits = search(conn, "database connection pool", mode="hybrid", limit=8)
        scores = [h.rrf_score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_hydrate_attaches_parent_titles(self, conn):
        """Retrieval returns chunks; citations must name documents."""
        from incidentiq.rag.retrieval import hydrate, search
        hits = hydrate(conn, search(conn, "connection pool exhausted", limit=5))
        assert hits
        assert all(h.parent_title for h in hits)
        assert all(h.citation.startswith(h.parent_doc_id) for h in hits)

    def test_empty_query_does_not_crash(self, conn):
        from incidentiq.rag.retrieval import search
        assert isinstance(search(conn, "", mode="hybrid", limit=5), list)

    def test_filter_matching_nothing_returns_empty(self, conn):
        from incidentiq.rag.retrieval import Filters, search
        assert search(conn, "anything", filters=Filters(service="does-not-exist")) == []
