# 02 — Chunking, embeddings, and hybrid retrieval

How text becomes searchable, and why it takes three techniques rather than one.

---

## Embeddings: text as coordinates

An **embedding model** converts text into a fixed-length list of numbers — a
**vector**. Ours produces 384 of them. The useful property is that texts with
similar *meaning* land near each other in that space, even with no words in
common.

Run against the real corpus:

```
query: "the database is refusing new connections"

  0.707  Connection pool to payment-db was exhausted; all connections checked out.
  0.581  Replication lag reached 240 seconds; replicas served data four minutes old.
  0.510  The TLS certificate used by stripe-adapter expired at midnight.
```

The top match shares almost no vocabulary with the query — "refusing" and
"exhausted" are different words for the same situation. That is the whole point:
**this is search by meaning, not by string**.

### Similarity, distance, and normalisation

pgvector's `<=>` returns cosine **distance** — smaller is closer, 0 means
identical. Similarity for display is `1 - distance`.

Our vectors are **normalised** to unit length (`normalize_embeddings=True`).
With unit vectors, cosine similarity is just the dot product, so distance
behaves predictably in the 0..2 range. Without normalisation, a longer document
could score as more similar simply for having a larger magnitude.

### The asymmetry that is easy to miss

BGE models are trained with an **instruction prefix on the query side only**:

```
passage:  "Connection pool to payment-db was exhausted..."      (as-is)
query:    "Represent this sentence for searching relevant passages: the database is..."
```

Skip the prefix and queries land in a slightly different region of the space
than passages, costing recall. Our `EmbeddingProvider` interface therefore has
two methods, `embed_documents` and `embed_query`, rather than one `embed` — the
asymmetry is impossible to forget at a call site.

A caution worth internalising: when I compared one query-document pair with and
without the prefix, the prefix made the similarity *lower* (0.707 vs 0.756).
That looks like the prefix hurting, and it is not. **The absolute similarity of
one pair is meaningless.** What matters is whether the correct document ranks
above the incorrect ones. Only a ranking metric over many queries can tell you
that — which is what Recall@5 is for.

---

## Chunking: why documents get split

An embedding is fixed-size. A 2000-word runbook and a one-line symptom both
compress to 384 numbers, so the runbook's vector becomes an average of
everything it discusses — close to nothing in particular.

Splitting means each vector represents one coherent idea.

### Recursive character splitting

Naively splitting every N characters cuts sentences in half. Better is to try
the highest-level boundary that fits, and only descend when a piece is still too
large:

```
paragraph  (\n\n)   →  sentence  (. ! ?)   →  line  (\n)   →  word  (space)
```

A piece already under the limit is never split further, so natural structure
survives wherever it can.

### Why overlap

A split can separate a claim from its condition — *"restart the service"* in one
chunk, *"only after confirming the dependency is healthy"* in the next. With
64 tokens of overlap, the tail of each chunk is carried onto the front of the
next, so any span shorter than the overlap appears intact somewhere.

Verified on a real runbook: chunk 0 ends with the first 248 characters of chunk
1 — 62 tokens against the 64 requested.

### Choosing the size: a plateau, not a peak

| chunk | chunks produced | mean tokens | Recall@5 |
|---|---|---|---|
| 96 | 1967 | 79 | 0.783 |
| 128 | 1485 | 101 | 0.783 |
| 192 | 986 | 148 | 0.800 |
| 256 | 703 | 193 | 0.800 |
| **320** | **680** | **200** | **0.800** |
| 640 | 609 | 216 | 0.800 |

Everything from 192 up is identical. The mean-tokens column explains why: even
with a 640-token budget the mean chunk is 216 tokens, because most documents are
shorter than any of these limits and never split at all.

Given a flat region, taking the argmax is choosing between ties on noise. The
rule used here: take everything within tolerance of the best, then pick the
**middle of the plateau**. An edge setting is one corpus change from falling
off.

If asked "what's your chunk size and how did you choose it", the answer is not
"320, it's a common default". It is: *"I swept it. On this corpus it plateaus
above 192, because the documents are short enough that most never split. I took
the middle of the plateau for robustness. On a corpus of long postmortems I'd
expect a real peak and would re-sweep."*

---

## Why vector search alone is not enough

Embeddings are strong on meaning and blind to exact strings.

`HikariPool-1` carries no semantic content — it is a connection-pool
implementation's internal name. An embedding model has no useful representation
for it. But it is the single most diagnostic string in a pool-exhaustion
incident.

Postgres full-text search handles exactly this. `to_tsvector` tokenises, stems,
and drops stopwords:

```
'checkout-servic':6 'checkout':7 'connect':2 'exhaust':4 'pool':3 'postgr':1 'servic':8
```

Two details worth noticing:

- The numbers are **positions**, which is how `ts_rank` scores proximity.
- `checkout-service` is indexed **three ways** — the whole hyphenated token and
  each half. For a corpus full of hyphenated service names, that matters.

### A bug that made keyword search contribute nothing

`websearch_to_tsquery` **ANDs every term**:

```sql
websearch_to_tsquery('english', 'checkout is timing out and I keep seeing HikariPool errors')
  →  'checkout' & 'time' & 'keep' & 'see' & 'hikaripool' & 'error'
```

No single chunk contains all six words, so this returned **zero rows** — despite
`HikariPool` genuinely appearing in the corpus. The keyword half of "hybrid"
retrieval was silently contributing nothing at all.

The fix converts the conjunctions to disjunctions on the *parsed* tsquery:

```sql
replace(websearch_to_tsquery('english', %s)::text, ' & ', ' | ')::tsquery
```

Rewriting the parsed query rather than the raw string keeps the stemming,
stopword removal, and quote handling, and keeps user text from ever reaching
tsquery syntax. `ts_rank` then does the discriminating — chunks matching more
terms, and rarer terms, rank higher.

This is worth remembering as a class of bug: **a component that returns zero
results looks identical to a component that is working and finding nothing.**
There is no error. The only way to catch it is to measure each half separately,
which is why the ablation table has vector-only and keyword-only rows.

---

## Reciprocal rank fusion

Two rankers return incomparable numbers. Cosine distance is 0..2; `ts_rank` is
unbounded and scale-dependent. Normalising them into a weighted sum means
choosing a weight with no principled basis, and the right weight drifts as the
corpus changes.

**RRF discards the scores and uses only the ranks:**

```
score(d) = Σ over rankers  weight / (k + rank_of_d_in_that_ranker)
```

Worked, with k=60:

| document | vector rank | keyword rank | score |
|---|---|---|---|
| A | 1 | — | 1/61 = 0.0164 |
| B | 10 | 10 | 2/70 = 0.0286 |

**B beats A.** A document ranked 10th by both rankers outranks one ranked 1st by
a single ranker. That is RRF working as designed — it rewards consensus. But it
means that when one ranker is weaker, its mediocre picks get promoted purely for
being seconded.

Hence the weight. Measured on this corpus:

| keyword weight | Recall@5 |
|---|---|
| 0.25 | 0.800 |
| 0.50 | 0.800 |
| 1.00 (naive default) | 0.783 |

`k` made no difference at all (5, 10, 20, 60 identical), so the canonical 60 is
kept rather than tuned to noise.

Seen live, on the query *"checkout is timing out and I keep seeing HikariPool
errors"*:

```
1. [v#1  k#1]  INC-00407   ← ranked first by both, wins outright
2. [v#5  k#2]  RB-026      ← promoted; neither ranker put it second
3. [v#2 k#12]  INC-00042
4. [v#3   k—]  INC-00368   ← wrong service; keyword missed it, demoted
```

---

## Metadata filtering before ranking

The retrieval query is not purely a vector question. A real one is *"find chunks
similar to this, but only for `checkout-service`, only SEV1 and SEV2, only the
last 90 days."*

Because chunks carry `service`, `severity`, and `doc_date` **denormalised from
their parent documents**, the filter is a `WHERE` clause on the same table as the
vector:

```sql
SELECT c.id, 1 - (c.embedding <=> %(qvec)s) AS similarity
FROM chunks c
WHERE c.service = ANY(%(services)s)          -- applied BEFORE ranking
ORDER BY c.embedding <=> %(qvec)s
LIMIT 20
```

Filtering *after* ranking would mean retrieving 500 chunks to keep 5, and the
relevant ones may not survive the initial top-N at all.

Duplicated data is usually a smell; here it is deliberate. If the filter needed
a join to `incidents`, the planner would either join first — losing the vector
index — or rank first, losing the filter's benefit.

---

## The finding I did not expect

| | filter off | filter on |
|---|---|---|
| vector only | 0.728 | **0.800** |
| hybrid | **0.800** | 0.800 |

With the service filter on, hybrid and vector **tie exactly**. With it off,
hybrid recovers the entire 0.072 the filter was providing.

So hybrid retrieval's value on this corpus is not that it beats dense retrieval.
It is that it is **insensitive to whether the filter is available** — which is
the condition on the first turn of an investigation, when the agent does not yet
know which service is at fault.

And a second, more general one: hybrid retrieval initially performed at or
*below* vector-only everywhere. The cause was a corpus defect. `HikariPool`,
`OutOfMemoryError`, and `x509` appeared in the logs and **zero** times in the
incident write-ups, so the keyword ranker had no identifiers to match.

**Hybrid retrieval is not universally better than dense retrieval. It is better
when your documents contain tokens embeddings cannot represent.** A corpus of
pure prose gains little from it. That is a more useful thing to know than
"hybrid is best practice".

---

## What to be able to explain

- What an embedding is, and why cosine rather than Euclidean for text
- Why the query gets an instruction prefix and the passage does not
- Why absolute similarity of one pair tells you nothing
- Your chunk size, and that you swept it and got a plateau
- Why overlap exists and how it is applied
- Why `websearch_to_tsquery` returns nothing for conversational queries
- How RRF combines two incomparable rankings, and its consensus bias
- Why the chunk table denormalises parent metadata
- The condition under which hybrid retrieval earns its cost

---

**Next:** 03 — Measuring retrieval: Recall@5 and what it does not tell you
