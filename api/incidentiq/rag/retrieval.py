"""Hybrid retrieval: vector similarity + keyword search, fused.

Why both
--------
Vector search understands meaning and is blind to exact strings. "the database
is refusing connections" finds an incident about pool exhaustion with no shared
vocabulary. But `HikariPool-1` is a meaningless token to an embedding model,
and it is the single most diagnostic string in that incident's logs.

Keyword search is the mirror image: exact on identifiers, useless on paraphrase.

Neither alone is enough for incident response, where a query is usually part
paraphrase ("checkout is slow") and part identifier ("502", "payment-service").

Why reciprocal rank fusion rather than blending scores
------------------------------------------------------
The two searches return incomparable numbers. Cosine distance is 0..2;
`ts_rank` is an unbounded relevance score whose scale depends on document
length and term frequency. Normalising them into a weighted sum means choosing a
weight with no principled basis, and the choice silently changes with corpus
composition.

RRF discards the scores and uses only the *ranks*:

    score(d) = sum over rankers of  1 / (k + rank_of_d_in_that_ranker)

A document ranked 1st by either method scores highly; a document ranked well by
both scores highest. k=60 is the constant from the original paper - it damps the
difference between ranks 1 and 2 so a single ranker cannot dominate on its own.

Why filter before ranking
-------------------------
`WHERE service = 'checkout-service'` is applied inside the same query as the
similarity ordering, so Postgres narrows the candidate set before HNSW ranks it.
Filtering afterwards would mean retrieving many chunks to keep few, and the
relevant ones may not survive the initial top-N at all. The measured effect of
this is in docs/EVALS.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
import psycopg

from incidentiq.rag.embeddings import EmbeddingProvider, get_embedding_provider

# Fusion parameters, chosen by measurement rather than by citing a default.
# See `evals/retrieval_eval.py --fusion-sweep` and docs/EVALS.md.
#
# k turned out not to matter on this corpus (5, 10, 20 and 60 all scored
# identically once the corpus contained error signatures), so the canonical 60
# is kept. The keyword weight does matter: at 1.0 the weaker ranker drags
# results down (Recall@5 0.783); at 0.5 it adds without dominating (0.800).
RRF_K = 60
DEFAULT_KEYWORD_WEIGHT = 0.5

SearchMode = Literal["hybrid", "vector", "keyword"]


@dataclass
class Filters:
    """Metadata constraints applied before ranking.

    `service` accepts several names so a cascade investigation can search a
    service together with its dependencies in one query.
    """
    service: str | list[str] | None = None
    doc_type: str | list[str] | None = None
    severity: str | list[str] | None = None
    since: datetime | None = None
    until: datetime | None = None

    def to_sql(self, alias: str = "c") -> tuple[str, dict]:
        """Build a WHERE fragment and its parameters."""
        clauses: list[str] = []
        params: dict = {}

        def add_in(column: str, value, key: str) -> None:
            if value is None:
                return
            values = [value] if isinstance(value, str) else list(value)
            if not values:
                return
            clauses.append(f"{alias}.{column} = ANY(%({key})s)")
            params[key] = values

        add_in("service", self.service, "f_service")
        add_in("doc_type", self.doc_type, "f_doc_type")
        add_in("severity", self.severity, "f_severity")
        if self.since:
            clauses.append(f"{alias}.doc_date >= %(f_since)s")
            params["f_since"] = self.since
        if self.until:
            clauses.append(f"{alias}.doc_date <= %(f_until)s")
            params["f_until"] = self.until

        return (" AND ".join(clauses) if clauses else "TRUE"), params


@dataclass
class RetrievedDoc:
    """One retrieved chunk, with everything needed to cite it and to explain
    why it was returned."""
    chunk_id: int
    doc_type: str
    parent_doc_id: str
    chunk_index: int
    content: str
    service: str | None
    severity: str | None
    # Provenance. A rank of None means that ranker did not return this chunk.
    vector_rank: int | None = None
    vector_similarity: float | None = None
    keyword_rank: int | None = None
    keyword_score: float | None = None
    rrf_score: float = 0.0
    # Filled by hydrate(): the parent document's title, for display and citation.
    parent_title: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"{self.parent_doc_id} ({self.doc_type})"


def _vector_search(
    conn: psycopg.Connection, qvec: list[float], filters: Filters, limit: int
) -> list[tuple[int, float]]:
    where, params = filters.to_sql()
    params["qvec"] = np.asarray(qvec, dtype=np.float32)
    params["limit"] = limit
    rows = conn.execute(
        f"""
        SELECT c.id, 1 - (c.embedding <=> %(qvec)s) AS similarity
        FROM chunks c
        WHERE {where} AND c.embedding IS NOT NULL
        ORDER BY c.embedding <=> %(qvec)s
        LIMIT %(limit)s
        """,
        params,
    ).fetchall()
    return [(r[0], float(r[1])) for r in rows]


def _keyword_search(
    conn: psycopg.Connection, query: str, filters: Filters, limit: int
) -> list[tuple[int, float]]:
    where, params = filters.to_sql()
    params["q"] = query
    params["limit"] = limit
    # `websearch_to_tsquery` ANDs every term:
    #     'checkout is timing out ... HikariPool errors'
    #       -> 'checkout' & 'time' & 'keep' & 'see' & 'hikaripool' & 'error'
    # For a natural-language incident report that matches nothing, because no
    # single chunk contains every word - so the keyword half of the hybrid
    # silently contributes zero. Switching the conjunctions to disjunctions
    # (`&` -> `|`) makes it "any of these terms", and `ts_rank` then does the
    # discriminating: it rewards chunks containing more of the terms, and rarer
    # terms weigh more than common ones.
    #
    # Rewriting the operator on the *parsed* tsquery rather than on the raw
    # string keeps websearch_to_tsquery's stemming, stopword removal, and
    # quoting behaviour, and keeps user text from ever reaching tsquery syntax.
    # Phrase operators (`<->`, from quoted input) survive untouched.
    rows = conn.execute(
        f"""
        WITH q AS (
            SELECT replace(
                websearch_to_tsquery('english', %(q)s)::text, ' & ', ' | '
            )::tsquery AS tsq
        )
        SELECT c.id, ts_rank(c.content_tsv, q.tsq) AS score
        FROM chunks c, q
        WHERE {where}
          AND c.content_tsv @@ q.tsq
        ORDER BY score DESC
        LIMIT %(limit)s
        """,
        params,
    ).fetchall()
    return [(r[0], float(r[1])) for r in rows]


def _fuse(
    vector_hits: list[tuple[int, float]],
    keyword_hits: list[tuple[int, float]],
    k: int = RRF_K,
    vector_weight: float = 1.0,
    keyword_weight: float = 1.0,
) -> dict[int, dict]:
    """Weighted reciprocal rank fusion. Returns chunk_id -> provenance dict.

    The two parameters exist because plain RRF has a specific and easily
    overlooked bias: it rewards *agreement* over *confidence*. With k=60, a
    chunk ranked 10th by both rankers scores 2/70 = 0.029, beating a chunk
    ranked 1st by one ranker alone at 1/61 = 0.016. When one ranker is
    substantially weaker than the other, its mediocre picks get promoted purely
    for being seconded.

    - `k` controls how sharply top ranks are favoured. Lower k means rank 1
      dominates; the canonical k=60 deliberately flattens the curve.
    - The weights let a ranker contribute proportionally to how good it
      actually is on this corpus.

    Both are measured rather than assumed - see `evals/retrieval_eval.py
    --fusion-sweep` and the results table in docs/EVALS.md.
    """
    merged: dict[int, dict] = {}

    for rank, (chunk_id, score) in enumerate(vector_hits, start=1):
        merged.setdefault(chunk_id, {"rrf": 0.0})
        merged[chunk_id]["vector_rank"] = rank
        merged[chunk_id]["vector_similarity"] = score
        merged[chunk_id]["rrf"] += vector_weight / (k + rank)

    for rank, (chunk_id, score) in enumerate(keyword_hits, start=1):
        merged.setdefault(chunk_id, {"rrf": 0.0})
        merged[chunk_id]["keyword_rank"] = rank
        merged[chunk_id]["keyword_score"] = score
        merged[chunk_id]["rrf"] += keyword_weight / (k + rank)

    return merged


def search(
    conn: psycopg.Connection,
    query: str,
    *,
    filters: Filters | None = None,
    limit: int = 5,
    mode: SearchMode = "hybrid",
    candidate_multiplier: int = 4,
    provider: EmbeddingProvider | None = None,
    rrf_k: int = RRF_K,
    vector_weight: float = 1.0,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
) -> list[RetrievedDoc]:
    """Retrieve the `limit` most relevant chunks.

    Each ranker retrieves `limit * candidate_multiplier` candidates before
    fusion, because a chunk ranked 12th by vector search and 3rd by keyword
    search can legitimately win overall - but only if it is in both candidate
    lists to begin with.
    """
    filters = filters or Filters()
    provider = provider or get_embedding_provider()
    candidates = limit * candidate_multiplier

    vector_hits: list[tuple[int, float]] = []
    keyword_hits: list[tuple[int, float]] = []

    if mode in ("hybrid", "vector"):
        vector_hits = _vector_search(conn, provider.embed_query(query), filters, candidates)
    if mode in ("hybrid", "keyword"):
        keyword_hits = _keyword_search(conn, query, filters, candidates)

    if mode == "vector":
        ordered = [(cid, {"rrf": 1.0 / (RRF_K + i), "vector_rank": i,
                          "vector_similarity": s})
                   for i, (cid, s) in enumerate(vector_hits, start=1)]
    elif mode == "keyword":
        ordered = [(cid, {"rrf": 1.0 / (RRF_K + i), "keyword_rank": i, "keyword_score": s})
                   for i, (cid, s) in enumerate(keyword_hits, start=1)]
    else:
        merged = _fuse(vector_hits, keyword_hits, rrf_k, vector_weight, keyword_weight)
        ordered = sorted(merged.items(), key=lambda kv: kv[1]["rrf"], reverse=True)

    top = ordered[:limit]
    if not top:
        return []

    ids = [cid for cid, _ in top]
    rows = conn.execute(
        "SELECT id, doc_type, parent_doc_id, chunk_index, content, service, severity "
        "FROM chunks WHERE id = ANY(%s)",
        (ids,),
    ).fetchall()
    by_id = {r[0]: r for r in rows}

    out: list[RetrievedDoc] = []
    for chunk_id, prov in top:
        r = by_id.get(chunk_id)
        if r is None:
            continue
        out.append(RetrievedDoc(
            chunk_id=r[0], doc_type=r[1], parent_doc_id=r[2], chunk_index=r[3],
            content=r[4], service=r[5], severity=r[6],
            vector_rank=prov.get("vector_rank"),
            vector_similarity=prov.get("vector_similarity"),
            keyword_rank=prov.get("keyword_rank"),
            keyword_score=prov.get("keyword_score"),
            rrf_score=prov["rrf"],
        ))
    return out


def hydrate(conn: psycopg.Connection, docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
    """Attach each chunk's parent document title.

    Retrieval returns chunks because chunks are what got embedded, but a
    citation should name the document. This is what `parent_doc_id` is for.
    """
    by_type: dict[str, list[str]] = {}
    for d in docs:
        by_type.setdefault(d.doc_type, []).append(d.parent_doc_id)

    titles: dict[tuple[str, str], str] = {}
    table = {"incident": "incidents", "runbook": "runbooks", "service_doc": "service_docs"}
    for doc_type, ids in by_type.items():
        for pid, title in conn.execute(
            f"SELECT id, title FROM {table[doc_type]} WHERE id = ANY(%s)", (ids,)
        ).fetchall():
            titles[(doc_type, pid)] = title

    for d in docs:
        d.parent_title = titles.get((d.doc_type, d.parent_doc_id))
    return docs
