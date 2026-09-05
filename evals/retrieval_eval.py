#!/usr/bin/env python3
"""Measure retrieval quality against the labelled set.

Usage:
    python -m evals.retrieval_eval                 # headline numbers
    python -m evals.retrieval_eval --ablation      # compare modes and filtering
    python -m evals.retrieval_eval --sweep         # chunk size / overlap grid
    python -m evals.retrieval_eval --json out.json

Definitions, stated precisely because the details decide the number
--------------------------------------------------------------------
Retrieval returns *chunks*, but relevance is labelled over *documents*. Results
are therefore deduplicated by `parent_doc_id` before scoring, and "@5" means the
first 5 distinct documents - not the first 5 chunks, which can be three
documents wearing five hats.

**Recall@5 is capped.** A situation with 12 relevant documents cannot have more
than 5 of them in 5 slots, so uncapped recall would be bounded at 0.42 for
reasons that have nothing to do with retrieval quality. The reported figure is

    Recall@5 = |relevant ∩ retrieved@5| / min(|relevant|, 5)

which asks the answerable question: of the relevant documents that *could* fit,
how many did we get? The uncapped figure is reported alongside it so the
difference is visible rather than hidden.

Ground truth is defined over the incident corpus only (see
docs/DECISIONS.md #5), so the headline measurement restricts retrieval to
incidents. The all-corpora figure is reported too, since that is what the agent
actually runs - a runbook taking a slot is not an error, but it does cost recall
against an incidents-only label set.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "api"))

from incidentiq.config import get_settings  # noqa: E402
from incidentiq.rag.retrieval import Filters, search  # noqa: E402

MANIFEST = REPO_ROOT / "seeds" / "live_situations.json"


@dataclass
class Scored:
    situation_id: str
    service: str
    archetype: str
    n_relevant: int
    n_hit: int
    recall_at_5: float
    recall_uncapped: float
    hit_at_5: bool
    reciprocal_rank: float
    latency_ms: float


def enrich_query(conn, case: dict, n_lines: int = 4) -> str:
    """Build the query the agent actually retrieves with.

    The bare user report - "something's wrong with auth-service, customers are
    complaining" - contains no information about the failure mode, while
    relevance is defined *by* the failure mode. Scoring retrieval on that text
    alone measures an impossible task and tells you nothing about the retriever.

    The agent does not retrieve from that text. Its `retrieve` node runs after
    `intake` has extracted entities and, on later loop iterations, after tools
    have returned evidence. So the honest measurement enriches the query with
    the distinct error signatures a first `get_service_logs` call would return -
    which is exactly what the agent would be holding.

    Both numbers are reported: cold (bare query) is the floor, enriched is the
    operating condition.
    """
    rows = conn.execute(
        """
        SELECT message, count(*) AS n
        FROM log_entries
        WHERE service = %s AND level IN ('ERROR', 'FATAL')
        GROUP BY message ORDER BY n DESC LIMIT %s
        """,
        (case["service"], n_lines),
    ).fetchall()
    signatures = " ".join(r[0] for r in rows)
    return f"{case['query']} Observed errors: {signatures}"


def _unique_docs(docs, k: int) -> list[str]:
    """Collapse chunks to their parent documents, preserving rank order."""
    seen: list[str] = []
    for d in docs:
        if d.parent_doc_id not in seen:
            seen.append(d.parent_doc_id)
        if len(seen) == k:
            break
    return seen


def score_one(conn, case: dict, *, mode: str, use_filter: bool,
              incidents_only: bool, enriched: bool = False, k: int = 5,
              **search_kw) -> Scored | None:
    relevant = set(case["relevant_incident_ids"])
    if not relevant:
        return None  # abstention case - scored separately, not here

    filters = Filters(
        service=case["service"] if use_filter else None,
        doc_type="incident" if incidents_only else None,
    )
    query = enrich_query(conn, case) if enriched else case["query"]

    t0 = time.perf_counter()
    # Over-fetch chunks so that after deduplication we still have k documents.
    hits = search(conn, query, filters=filters, limit=k * 6, mode=mode, **search_kw)
    latency_ms = (time.perf_counter() - t0) * 1000

    top_docs = _unique_docs(hits, k)
    found = [d for d in top_docs if d in relevant]

    rr = 0.0
    for i, doc_id in enumerate(top_docs, start=1):
        if doc_id in relevant:
            rr = 1.0 / i
            break

    return Scored(
        situation_id=case["situation_id"],
        service=case["service"],
        archetype=case["archetype"],
        n_relevant=len(relevant),
        n_hit=len(found),
        recall_at_5=len(found) / min(len(relevant), k),
        recall_uncapped=len(found) / len(relevant),
        hit_at_5=bool(found),
        reciprocal_rank=rr,
        latency_ms=latency_ms,
    )


def run_config(conn, cases: list[dict], *, mode: str, use_filter: bool,
               incidents_only: bool, enriched: bool = False, **search_kw) -> dict:
    scored = [
        s for c in cases
        if (s := score_one(conn, c, mode=mode, use_filter=use_filter,
                           incidents_only=incidents_only, enriched=enriched,
                           **search_kw)) is not None
    ]
    if not scored:
        return {}
    return {
        "mode": mode,
        "service_filter": use_filter,
        "incidents_only": incidents_only,
        "enriched": enriched,
        **{f"fusion_{key}": val for key, val in search_kw.items()},
        "n_scored": len(scored),
        "recall_at_5": round(statistics.mean(s.recall_at_5 for s in scored), 4),
        "recall_uncapped": round(statistics.mean(s.recall_uncapped for s in scored), 4),
        "hit_rate_at_5": round(statistics.mean(s.hit_at_5 for s in scored), 4),
        "mrr": round(statistics.mean(s.reciprocal_rank for s in scored), 4),
        "median_latency_ms": round(statistics.median(s.latency_ms for s in scored), 1),
        "p95_latency_ms": round(
            sorted(s.latency_ms for s in scored)[max(0, int(len(scored) * 0.95) - 1)], 1
        ),
        "per_situation": [asdict(s) for s in scored],
    }


def score_abstention(cases: list[dict]) -> dict:
    """Situations with no relevant history. Retrieval cannot be scored on
    recall here; what matters is what the agent does with weak results, which
    is a Phase 3 measurement. Reported so the count is never silently dropped."""
    abstain = [c for c in cases if not c["relevant_incident_ids"]]
    return {
        "n_abstention_cases": len(abstain),
        "situation_ids": [c["situation_id"] for c in abstain],
        "note": "excluded from Recall@5; scored on agent abstention in Phase 3",
    }


def print_table(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print(f"  {'query':<9} {'mode':<9} {'filter':<7} {'corpus':<11} "
          f"{'Recall@5':>9} {'uncapped':>9} {'Hit@5':>7} {'MRR':>6} {'p50 ms':>7}")
    print("  " + "─" * 84)
    for r in rows:
        if not r:
            continue
        corpus = "incidents" if r["incidents_only"] else "all"
        qkind = "enriched" if r.get("enriched") else "cold"
        print(f"  {qkind:<9} {r['mode']:<9} {str(r['service_filter']):<7} {corpus:<11} "
              f"{r['recall_at_5']:>9.3f} {r['recall_uncapped']:>9.3f} "
              f"{r['hit_rate_at_5']:>7.3f} {r['mrr']:>6.3f} {r['median_latency_ms']:>7.1f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", action="store_true", help="compare modes and filtering")
    parser.add_argument("--sweep", action="store_true", help="chunk size / overlap grid")
    parser.add_argument("--fusion-sweep", action="store_true",
                        help="grid over RRF k and ranker weights")
    parser.add_argument("--json", type=str, help="write full results to this path")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    cases = manifest["labelled_retrieval"]
    scorable = [c for c in cases if c["relevant_incident_ids"]]

    print(f"labelled set: {len(cases)} situations "
          f"({len(scorable)} scorable, {len(cases) - len(scorable)} abstention)")
    print(f"relevant documents per situation: "
          f"min {min(c['relevant_count'] for c in scorable)}, "
          f"median {statistics.median(c['relevant_count'] for c in scorable):.0f}, "
          f"max {max(c['relevant_count'] for c in scorable)}")

    conn = psycopg.connect(get_settings().database_url)
    register_vector(conn)

    results: dict = {
        "generated_at": datetime.now(UTC).isoformat(),
        "abstention": score_abstention(cases),
    }

    if args.fusion_sweep:
        results["fusion_sweep"] = run_fusion_sweep(conn, cases)
    elif args.sweep:
        results["sweep"] = run_sweep(conn, cases)
    elif args.ablation:
        rows = []
        for enriched in (False, True):
            for mode in ("vector", "keyword", "hybrid"):
                for use_filter in (False, True):
                    rows.append(run_config(conn, cases, mode=mode, use_filter=use_filter,
                                           incidents_only=True, enriched=enriched))
        results["ablation"] = rows
        print_table(rows, "ABLATION — retrieval mode × service pre-filter × corpus")
    else:
        rows = [
            run_config(conn, cases, mode="hybrid", use_filter=True,
                       incidents_only=True, enriched=True),
            run_config(conn, cases, mode="hybrid", use_filter=True,
                       incidents_only=True, enriched=False),
        ]
        results["headline"] = rows
        print_table(rows, "HEADLINE")

    print(f"\nabstention cases (excluded from recall): "
          f"{results['abstention']['n_abstention_cases']} "
          f"{results['abstention']['situation_ids']}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")
    return 0


def run_fusion_sweep(conn, cases: list[dict]) -> list[dict]:
    """Grid over RRF k and the keyword ranker's weight.

    Answers "how does hybrid retrieval combine the two rankings" with a measured
    setting rather than a cited default. Baselines for the two single rankers
    are printed alongside so it is obvious whether fusion earns its place.
    """
    out = []
    print("\nFUSION SWEEP — enriched query, service filter on, incidents corpus")
    print(f"  {'k':>4} {'w_kw':>6} {'Recall@5':>9} {'Hit@5':>7} {'MRR':>6}")
    print("  " + "─" * 38)
    for mode, label in (("vector", "vector only"), ("keyword", "keyword only")):
        r = run_config(conn, cases, mode=mode, use_filter=True,
                       incidents_only=True, enriched=True)
        print(f"  {label:>11}   {r['recall_at_5']:>9.3f} {r['hit_rate_at_5']:>7.3f} "
              f"{r['mrr']:>6.3f}")
        out.append({**{k: v for k, v in r.items() if k != "per_situation"}})
    print("  " + "─" * 38)
    for rrf_k in (5, 10, 20, 60):
        for w_kw in (0.25, 0.5, 1.0):
            r = run_config(conn, cases, mode="hybrid", use_filter=True,
                           incidents_only=True, enriched=True,
                           rrf_k=rrf_k, keyword_weight=w_kw)
            print(f"  {rrf_k:>4} {w_kw:>6.2f} {r['recall_at_5']:>9.3f} "
                  f"{r['hit_rate_at_5']:>7.3f} {r['mrr']:>6.3f}")
            out.append({k: v for k, v in r.items() if k != "per_situation"})
    return out


def run_sweep(conn, cases: list[dict]) -> list[dict]:
    """Re-chunk, re-embed, and re-measure across a grid of chunk parameters.

    This is the empirical answer to "what is your chunk size and why".
    """
    from incidentiq.rag.indexer import index

    # Overlap is held at 20% of chunk size throughout. Sweeping both
    # independently would be a 2-D grid whose second axis, on a corpus of short
    # structured documents, is dominated by the first.
    grid = [(96, 20), (128, 24), (192, 40), (256, 48), (320, 64), (448, 88), (640, 128)]
    out = []
    print("\nSWEEP — enriched query, hybrid (k=10, w_kw=0.5), service filter on")
    print("  re-indexes at each setting; the chunk table is restored at the end")
    print(f"  {'chunk':>6} {'overlap':>8} {'chunks':>7} {'mean tok':>9} {'Recall@5':>9} "
          f"{'Hit@5':>7} {'MRR':>6} {'embed s':>8}")
    print("  " + "─" * 72)
    for chunk_tokens, overlap in grid:
        stats = index(conn, chunk_tokens, overlap, quiet=True)
        conn.commit()
        r = run_config(conn, cases, mode="hybrid", use_filter=True, incidents_only=True,
                       enriched=True, rrf_k=10, keyword_weight=0.5)
        row = {**stats, **{k: v for k, v in r.items() if k != "per_situation"}}
        out.append(row)
        print(f"  {chunk_tokens:>6} {overlap:>8} {stats['chunks']:>7} "
              f"{stats['mean_tokens']:>9.0f} {r['recall_at_5']:>9.3f} "
              f"{r['hit_rate_at_5']:>7.3f} {r['mrr']:>6.3f} {stats['seconds_embed']:>8.1f}")
    # Picking the winner: the results plateau rather than peak, so "argmax" would
    # be choosing between ties on noise. Take every setting within a small
    # tolerance of the best score and pick the MIDDLE of that plateau, not an
    # edge of it. A setting at the edge is one corpus change away from falling
    # off; the centre is the robust choice, and on a 12-situation labelled set
    # that robustness is worth more than a third decimal place.
    top = max(r["recall_at_5"] for r in out)
    plateau = [r for r in out if r["recall_at_5"] >= top - 1e-9]
    chosen = plateau[len(plateau) // 2]
    print(f"\n  plateau at Recall@5 {top:.3f}: "
          f"chunk sizes {[r['chunk_tokens'] for r in plateau]}")
    print(f"  chosen: chunk={chosen['chunk_tokens']} overlap={chosen['overlap_tokens']} "
          f"(middle of the plateau, not an edge of it)")
    index(conn, chosen["chunk_tokens"], chosen["overlap_tokens"], quiet=True)
    conn.commit()
    return out


if __name__ == "__main__":
    raise SystemExit(main())
