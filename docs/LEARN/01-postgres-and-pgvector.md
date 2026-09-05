# 01 — Postgres, pgvector, and why the schema looks like that

Every design choice in `db/migrations/` explained. Read alongside the SQL files.

---

## Why one database instead of two

The conventional stack for this kind of system is a relational database for
application data plus a dedicated vector database (Pinecone, Weaviate, Qdrant)
for embeddings. IncidentIQ uses Postgres for both, via the **pgvector**
extension.

The reason is that retrieval here is never purely a vector question. A real
query is *"find chunks similar to this, but only for `checkout-service`, only
SEV1 and SEV2, and only from the last 90 days."* With two datastores you either

- fetch a large candidate set from the vector DB and filter in application code
  — which means retrieving 500 chunks to keep 5, and Recall@5 quietly collapses
  because the relevant ones fell outside the candidate set; or
- duplicate all your metadata into the vector store and keep it in sync forever.

With pgvector the filter is a `WHERE` clause on the same table as the vector, so
Postgres applies it *before* ranking. That is the "metadata filtering before
vector search" requirement, and it is one SQL statement rather than an
architecture.

The tradeoff is real and worth being able to state: a dedicated vector database
will outperform pgvector at very large scale (hundreds of millions of vectors)
and offers features pgvector lacks. At this corpus size — thousands of chunks —
that advantage is entirely theoretical, and the operational cost of a second
datastore is not.

---

## The `vector` type and distance operators

```sql
embedding vector(384)
```

384 is fixed by the embedding model (`BAAI/bge-small-en-v1.5`). **Changing
models means changing this number and re-embedding everything** — vectors from
different models are not comparable, not convertible, and produce silently
meaningless similarity scores if mixed. This is why `EMBEDDING_DIM` is in
`.env.example` next to `EMBEDDING_MODEL`: they move together.

pgvector adds three distance operators:

| Operator | Distance | Use |
|---|---|---|
| `<=>` | Cosine | **What we use.** Compares direction, ignores magnitude |
| `<->` | L2 (Euclidean) | Straight-line distance; magnitude matters |
| `<#>` | Negative inner product | Fastest, requires normalised vectors |

Cosine is the right default for text. Two documents about the same topic point
the same direction in embedding space regardless of length, and cosine measures
exactly that. L2 would score a long document as further away simply for being
long.

Note that these are **distance**, not similarity — smaller is closer. Identical
vectors give 0:

```sql
SELECT embedding <=> '[...]'::vector AS distance FROM chunks ORDER BY distance LIMIT 5;
```

Similarity for display is `1 - distance`.

---

## The HNSW index, and what "approximate" costs you

```sql
CREATE INDEX idx_chunks_embedding ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

Without an index, finding the 5 nearest vectors means computing distance to
every row — fine at 5,000 chunks, hopeless at 5,000,000.

**HNSW** (Hierarchical Navigable Small World) builds a layered graph where each
vector links to its neighbours. A search enters at the top layer, greedily walks
toward the query, and drops down a layer. It examines a few hundred candidates
instead of all of them.

The catch is in the name: it is **approximate**. It can miss a true nearest
neighbour. Two parameters trade that off:

| Parameter | Meaning | Raising it |
|---|---|---|
| `m` | Links per node | Better recall, more memory, slower build |
| `ef_construction` | Candidates considered at build | Better recall, slower build only |
| `ef_search` (query time) | Candidates considered at search | Better recall, slower query |

`m = 16, ef_construction = 64` are the pgvector defaults and a sensible starting
point. **The right way to tune them is to measure**, which is precisely what
Recall@5 does in Phase 2 — that metric is measuring the index's miss rate as
much as the embedding model's quality.

The alternative index type, IVFFlat, clusters vectors and searches the nearest
clusters. It builds much faster but has worse recall at the same speed, and it
must be built *after* data is loaded (it needs the data to compute clusters).
HNSW can be built on an empty table, which is why it works with a migration that
runs before seeding.

---

## Full-text search: the generated tsvector column

```sql
content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
```

Vector search is weak at exactly the things incident response cares about most:
error codes, service names, stack-trace fragments. `HikariPool-1` is a
meaningless token to an embedding model, but it is the single most diagnostic
string in a log line.

Postgres full-text search handles these. `to_tsvector('english', ...)` does
three things:

1. Tokenises the text
2. Stems words to a root form (`connections`, `connection`, `connecting` → `connect`)
3. Drops stopwords

The result is queryable with `@@`:

```sql
SELECT * FROM chunks WHERE content_tsv @@ plainto_tsquery('english', 'connection pool exhausted');
```

Two details worth noticing in the actual output:

```
'checkout':7 'checkout-servic':6 'connect':2 'exhaust':4 'pool':3 'postgr':1 'servic':8
```

- The numbers are **positions**, which is how `ts_rank` scores proximity.
- `checkout-service` is indexed **three ways**: the whole hyphenated token, plus
  each half. So a query for "checkout" finds it, and so does "checkout-service".
  For a corpus full of hyphenated service names, this matters a lot.

`GENERATED ALWAYS AS ... STORED` computes the column on write. The alternative —
a trigger, or computing it in the application — can drift out of sync with
`content`. A generated column cannot: Postgres will not let you write it
directly.

`GIN` (Generalized Inverted Index) is the right index for it — an inverted index
from each lexeme to the rows containing it, which is the same structure a search
engine uses.

---

## Why chunks are one table and documents are three

Documents are read by humans and cited by the agent. An incident has a
`root_cause` and a `resolution`; a runbook has neither. Forcing all three into
one table means a wide table of mostly-NULL columns, or a JSON blob that no
constraint can check.

Chunks are read only by the retriever, which wants one indexed table. Three chunk
tables would require:

```sql
SELECT ... FROM incident_chunks UNION ALL
SELECT ... FROM runbook_chunks   UNION ALL
SELECT ... FROM doc_chunks       ORDER BY embedding <=> $1 LIMIT 5
```

Postgres cannot use an HNSW index to satisfy an `ORDER BY` across a `UNION ALL`
— it would scan each branch. One table with a `doc_type` column turns that into
an ordinary indexed query with a `WHERE` clause.

The cost is that `parent_doc_id` cannot be a real foreign key, because it points
at one of three tables depending on `doc_type`. That is a genuine loss of
referential integrity, and the delete path has to handle it manually.

### Denormalised metadata

`chunks` carries `service`, `severity`, and `doc_date` copied from the parent
document. Duplicated data is normally a smell. Here it is deliberate: it means
the pre-filter is a `WHERE` on the same table as the vector, with no join.
Postgres can then use `idx_chunks_service` to cut the candidate set before HNSW
ever runs.

If the filter required a join to `incidents`, the planner would either join first
(losing the vector index) or rank first (losing the filter's benefit). Neither is
what you want.

---

## The append-only audit log

```sql
CREATE TRIGGER audit_log_no_update
    BEFORE UPDATE OR DELETE ON audit_log
    FOR EACH ROW EXECUTE FUNCTION reject_audit_mutation();
```

An audit log that application code *could* modify is not an audit log; it is a
table that currently happens to be append-only. Enforcing it in the database
means the guarantee holds against every path — a bug, a migration, a `psql`
session, an ORM doing something clever.

It genuinely works:

```
=> UPDATE audit_log SET actor='attacker' WHERE investigation_id='IQ-TEST';
ERROR:  audit_log is append-only; UPDATE is not permitted
```

The honest limit: a superuser can drop the trigger. This is tamper-*evident*
against application-level mistakes, not tamper-proof against a database
administrator. Real immutability needs an external append-only store. Saying so
is better than overclaiming.

---

## What to be able to explain

- Why one database rather than a dedicated vector store, **and** the case for
  the other choice
- Why cosine rather than L2 for text
- What "approximate" costs in HNSW, and which knob trades recall for speed
- Why vector search alone is insufficient for error codes and service names
- Why the chunk metadata is deliberately denormalised
- Why the audit-log guarantee is in the database and where it still stops

---

**Next:** 02 — Chunking, embeddings, and hybrid retrieval *(Phase 2)*
