"""Splitting documents into chunks before embedding.

Why chunk at all: an embedding is a fixed-size vector, so a 2000-word runbook
and a one-sentence symptom both compress to 384 numbers. The runbook's vector
ends up as an average of everything it discusses, which is close to nothing in
particular. Splitting it means each vector represents one coherent idea.

Why overlap: a naive split can cut a sentence — or worse, separate a claim from
its condition ("restart the service" / "only after confirming the dependency is
healthy"). Overlapping windows mean any span of text up to the overlap size
appears intact in at least one chunk.

The two parameters (size, overlap) are not guessed. `evals/retrieval_sweep.py`
measures Recall@5 across a grid of them, and docs/EVALS.md reports the result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Rough token estimate. The real tokenizer is the embedding model's, but this
# is within ~10% for English prose and avoids loading the model just to chunk.
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class Chunk:
    """One chunk, ready to embed."""
    parent_doc_id: str
    doc_type: str
    chunk_index: int
    content: str
    token_count: int
    # Denormalised from the parent so the retriever can filter without a join.
    service: str | None = None
    severity: str | None = None
    doc_date: object | None = None


# Split points, most preferred first. Splitting on a paragraph boundary
# preserves more meaning than splitting mid-sentence, which preserves more than
# splitting mid-word.
_SPLIT_PATTERNS = (
    re.compile(r"\n\n+"),        # paragraph
    re.compile(r"(?<=[.!?])\s+"),  # sentence
    re.compile(r"\n"),            # line
    re.compile(r"\s+"),           # word
)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _split_recursive(text: str, max_chars: int, depth: int = 0) -> list[str]:
    """Split text into pieces no larger than max_chars, preferring the
    highest-level boundary that works.

    This is the 'recursive character' strategy: try paragraphs; any piece still
    too big gets split by sentences; still too big, by lines; then by words. A
    piece that is under the limit is never split further, so natural structure
    survives wherever it can.
    """
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    if depth >= len(_SPLIT_PATTERNS):
        # Out of separators: hard-cut. Only reachable for a single token longer
        # than max_chars, e.g. a base64 blob or a very long stack frame.
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    parts = _SPLIT_PATTERNS[depth].split(text)
    if len(parts) == 1:
        return _split_recursive(text, max_chars, depth + 1)

    # Recombine adjacent parts greedily up to the limit, so we do not end up
    # with one chunk per sentence when several sentences would fit together.
    out: list[str] = []
    buf = ""
    for part in parts:
        candidate = f"{buf}\n\n{part}" if buf and depth == 0 else (f"{buf} {part}" if buf else part)
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                out.append(buf)
            if len(part) > max_chars:
                out.extend(_split_recursive(part, max_chars, depth + 1))
                buf = ""
            else:
                buf = part
    if buf:
        out.append(buf)
    return [c for c in out if c.strip()]


def chunk_text(
    text: str,
    *,
    chunk_tokens: int = 320,
    overlap_tokens: int = 64,
) -> list[str]:
    """Split text into overlapping chunks of roughly `chunk_tokens` each.

    Overlap is applied by carrying the tail of each chunk onto the front of the
    next, measured in characters derived from the token estimate.
    """
    if not text.strip():
        return []

    max_chars = chunk_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN
    if overlap_chars >= max_chars:
        raise ValueError(
            f"overlap ({overlap_tokens}) must be smaller than chunk size ({chunk_tokens})"
        )

    pieces = _split_recursive(text.strip(), max_chars)
    if len(pieces) <= 1:
        return pieces

    with_overlap: list[str] = [pieces[0]]
    for piece in pieces[1:]:
        prev = with_overlap[-1]
        # Carry the tail of the previous chunk, cut back to a word boundary so
        # the overlap does not start mid-word.
        tail = prev[-overlap_chars:]
        if " " in tail:
            tail = tail[tail.index(" ") + 1:]
        with_overlap.append(f"{tail} {piece}".strip())
    return with_overlap


def chunk_incident(row: dict, **kw) -> list[Chunk]:
    """Chunk one incident.

    An incident is short and has strong internal structure, so each section is
    prefixed with its role. Without the prefix, a chunk containing only a
    resolution reads as an assertion about the world ("Restarted the service and
    raised max_pool_size") rather than as what someone did about one incident.
    The prefix is also a retrieval signal: a query about how something was fixed
    matches text that says 'Resolution'.
    """
    header = f"Incident {row['id']}: {row['title']} ({row['severity']}, {row['service']})"
    # Error signatures are placed with the symptoms, which is where they belong
    # semantically and where they do the most retrieval work: a query quoting an
    # error is describing what was observed, not what fixed it.
    signatures = row.get("error_signatures") or []
    observed = ("\n" + "\n".join(signatures)) if signatures else ""
    body = (
        f"{header}\n\n"
        f"Symptoms: {row['symptoms']}{observed}\n\n"
        f"Root cause: {row['root_cause']}\n\n"
        f"Resolution: {row['resolution']}"
    )
    return _to_chunks(
        body, row["id"], "incident",
        service=row["service"], severity=row["severity"], doc_date=row["occurred_at"], **kw
    )


def chunk_runbook(row: dict, **kw) -> list[Chunk]:
    header = f"Runbook {row['id']}: {row['title']} (category: {row['category']})"
    body = f"{header}\n\n{row['body']}"
    return _to_chunks(body, row["id"], "runbook", service=row["service"], **kw)


def chunk_service_doc(row: dict, **kw) -> list[Chunk]:
    header = f"{row['title']} ({row['doc_kind']} documentation for {row['service']})"
    body = f"{header}\n\n{row['body']}"
    return _to_chunks(body, row["id"], "service_doc", service=row["service"], **kw)


def _to_chunks(
    body: str,
    parent_doc_id: str,
    doc_type: str,
    *,
    service: str | None = None,
    severity: str | None = None,
    doc_date: object | None = None,
    chunk_tokens: int = 320,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    texts = chunk_text(body, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
    return [
        Chunk(
            parent_doc_id=parent_doc_id,
            doc_type=doc_type,
            chunk_index=i,
            content=text,
            token_count=estimate_tokens(text),
            service=service,
            severity=severity,
            doc_date=doc_date,
        )
        for i, text in enumerate(texts)
    ]
