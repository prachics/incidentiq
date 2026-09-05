"""Chunk every document in the three corpora, embed the chunks, store them.

Run after seeding:
    python -m incidentiq.rag.indexer

This is the step that turns a database of documents into something searchable.
It is idempotent per configuration: re-running with the same chunk parameters
replaces the chunk table rather than appending to it.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from incidentiq.config import get_settings
from incidentiq.rag.chunking import (
    Chunk,
    chunk_incident,
    chunk_runbook,
    chunk_service_doc,
)
from incidentiq.rag.embeddings import get_embedding_provider

log = logging.getLogger(__name__)

DEFAULT_CHUNK_TOKENS = 320
DEFAULT_OVERLAP_TOKENS = 64


def _fetch(conn: psycopg.Connection, sql: str) -> list[dict]:
    cur = conn.execute(sql)
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def build_chunks(
    conn: psycopg.Connection,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    kw = {"chunk_tokens": chunk_tokens, "overlap_tokens": overlap_tokens}
    chunks: list[Chunk] = []

    for row in _fetch(conn, """
        SELECT id, title, service, severity, occurred_at, symptoms, root_cause,
               resolution, error_signatures
        FROM incidents ORDER BY id
    """):
        chunks.extend(chunk_incident(row, **kw))

    for row in _fetch(conn, "SELECT id, title, service, category, body FROM runbooks ORDER BY id"):
        chunks.extend(chunk_runbook(row, **kw))

    for row in _fetch(conn, """
        SELECT id, title, service, doc_kind, body FROM service_docs ORDER BY id
    """):
        chunks.extend(chunk_service_doc(row, **kw))

    return chunks


def index(
    conn: psycopg.Connection,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    quiet: bool = False,
) -> dict[str, float | int]:
    provider = get_embedding_provider()

    t0 = time.perf_counter()
    chunks = build_chunks(conn, chunk_tokens, overlap_tokens)
    t_chunk = time.perf_counter() - t0

    if not quiet:
        print(f"  chunked  {len(chunks):,} chunks in {t_chunk:.1f}s")

    t0 = time.perf_counter()
    vectors = provider.embed_documents([c.content for c in chunks])
    t_embed = time.perf_counter() - t0

    if not quiet:
        rate = len(chunks) / t_embed if t_embed else 0
        print(f"  embedded {len(chunks):,} chunks in {t_embed:.1f}s ({rate:.0f}/s)")

    t0 = time.perf_counter()
    with conn.transaction():
        conn.execute("TRUNCATE chunks RESTART IDENTITY")
        with conn.cursor().copy(
            "COPY chunks (doc_type, parent_doc_id, chunk_index, content, token_count, "
            "service, severity, doc_date, embedding) FROM STDIN"
        ) as copy:
            # strict=True: a length mismatch here would silently pair chunks
            # with the wrong vectors, which no test downstream would catch.
            for c, vec in zip(chunks, vectors, strict=True):
                copy.write_row((
                    c.doc_type, c.parent_doc_id, c.chunk_index, c.content,
                    c.token_count, c.service, c.severity, c.doc_date,
                    # A plain Python list is dumped as a Postgres array
                    # ("{1,2}"); pgvector needs vector syntax ("[1,2]"). Its
                    # psycopg adapter produces that for a numpy float32 array.
                    np.asarray(vec, dtype=np.float32),
                ))
    t_write = time.perf_counter() - t0

    if not quiet:
        print(f"  wrote    {len(chunks):,} rows in {t_write:.1f}s")

    tokens = [c.token_count for c in chunks]
    return {
        "chunks": len(chunks),
        "chunk_tokens": chunk_tokens,
        "overlap_tokens": overlap_tokens,
        "mean_tokens": sum(tokens) / len(tokens) if tokens else 0,
        "max_tokens": max(tokens) if tokens else 0,
        "seconds_chunk": round(t_chunk, 2),
        "seconds_embed": round(t_embed, 2),
        "seconds_write": round(t_write, 2),
    }


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=DEFAULT_OVERLAP_TOKENS)
    args = parser.parse_args()

    settings = get_settings()
    with psycopg.connect(settings.database_url, autocommit=False) as conn:
        register_vector(conn)
        n_docs = conn.execute(
            "SELECT (SELECT count(*) FROM incidents) + (SELECT count(*) FROM runbooks) "
            "+ (SELECT count(*) FROM service_docs)"
        ).fetchone()[0]
        if n_docs == 0:
            print("no documents to index - run: python seeds/generate.py", file=sys.stderr)
            return 1

        print(f"indexing {n_docs:,} documents "
              f"(chunk={args.chunk_tokens} overlap={args.overlap_tokens})")
        stats = index(conn, args.chunk_tokens, args.overlap_tokens)
        conn.commit()

    print(f"\n  {stats['chunks']:,} chunks, mean {stats['mean_tokens']:.0f} tokens, "
          f"max {stats['max_tokens']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
