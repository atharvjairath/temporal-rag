# Temporal RAG — Personal Context Retrieval

A retrieval pipeline designed for the hard problem at the core of personal AI assistants: handling broad, time-anchored queries like *"what was I working on last week?"*

Standard RAG systems were designed for document Q&A, not personal context. This project explores what changes when the corpus is someone's digital life — and queries are vague, temporal, and deeply personal.

---

## The Problem

General-purpose RAG breaks down on personal context queries for three reasons:

**1. Temporal vagueness is the primary constraint, not topic.**
"What did I do last week?" has almost no semantic content. The time range IS the query. A single embedding lookup finds nothing useful because there's no topic to embed.

**2. Broad queries need multi-strategy retrieval.**
"What was I working on last week?" could match meeting notes, PRs, Slack messages, browser tabs, or docs — across wildly different vocabulary. No single retrieval strategy covers all of them.

**3. Standard re-ranking ignores time and source.**
Cross-encoder re-rankers maximise semantic relevance. But a semantically perfect result from 6 months ago is less useful than a slightly weaker result from yesterday. A perfect Slack result is also wrong if the user asked for meetings.

This pipeline addresses all three.

---

## Pipeline Architecture

```
Raw Query
    │
    ▼
┌─────────────────────────────────────────────┐
│  Stage 1: Query Decomposer                  │
│  "what meetings did I have last week?"      │
│  → time_start: May 13  time_end: May 20     │
│  → topic: ""         intent: recall         │
│  → sources: ["meeting"]                     │
│  → entities: []                             │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  Stage 2: Query Filters                     │
│  Hard-filter by time window + source type   │
│  before retrieval, not after                │
└────────────────────┬────────────────────────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
┌──────────────┐ ┌──────────┐ ┌────────────────┐
│ Dense        │ │ Sparse   │ │ Recency        │
│ Retriever    │ │ BM25     │ │ Retriever      │
│              │ │          │ │                │
│ Open-source  │ │ Exact    │ │ Time-window    │
│ embeddings   │ │ keyword  │ │ ordering       │
│              │ │ match    │ │ + decay score  │
└──────┬───────┘ └────┬─────┘ └───────┬────────┘
       │              │               │
       └──────────────┼───────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│  Stage 3: Temporal Re-ranker                │
│  RRF fusion + small temporal tie-breaker    │
│  → scale-invariant rank combination         │
│  → recency helps, but does not dominate     │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  Stage 4: Context Pruner                    │
│  Greedy token budget + max-doc cap          │
│  → keep high-signal docs, truncate large    │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  Stage 5: Narrative Grouper                 │
│  Cluster retrieved chunks into work threads │
│  → "Payments", "Auth", "Search", ...        │
└─────────────────────────────────────────────┘
```

---

## Design Decisions

### Why three retrievers in parallel?

Each retriever captures different signal:

| Retriever | Strengths | Weaknesses |
|-----------|-----------|------------|
| Dense | Semantic similarity, paraphrase matching | Misses exact names, dates, jargon |
| Sparse (BM25) | Exact terms, proper nouns, technical vocabulary | Can't handle synonyms or vague queries |
| Recency | Always surfaces recent context regardless of topic | Ignores semantic relevance entirely |

For personal context queries, you need all three. "What did I discuss with Sarah about the API?" requires: dense (semantics of the discussion), sparse (name "Sarah" + keyword "API"), and potentially recency (if you mean recently).

The dense retriever uses `sentence-transformers/all-MiniLM-L6-v2`, an open-source local embedding model.

### Why source-aware decomposition?

"What was I reading about AI recently?" and "what meetings did I have about architecture?" are not just topic queries. They contain source constraints.

The decomposer extracts:

```
"what meetings did I have about architecture?"
       │                         │
    source                   topic
  ("meeting")            ("architecture")
```

Those filters are pushed before retrieval. That matters because source filtering after ranking wastes top-k slots on the wrong modality.

### Why Reciprocal Rank Fusion (RRF)?

The naive alternative — weighted sum of normalised scores — sounds reasonable until you notice:
- BM25 scores range from 0 to ~25 depending on corpus size and term frequency
- Cosine similarity scores range from 0 to 1
- Recency decay scores range from 0 to 1

Normalising brings them to similar ranges, but the distributions are completely different (BM25 is Zipf-distributed; cosine is roughly normal). A doc with BM25=8.3 and cosine=0.12 produces different linear combinations depending on how you normalise.

RRF sidesteps this entirely by using rank position, which is scale-invariant:

```
RRF_score(doc) = Σ  w_i / (k + rank_i(doc))
               retrievers
```

A document ranked #1 in dense and #3 in sparse always beats one ranked #2 in both, regardless of the raw score magnitudes.

### Why temporal boost is additive, but small?

```python
final_score = rrf_score + temporal_weight * 0.03 * recency_score
```

A multiplier (`final_score = rrf_score * (1 + boost)`) would suppress documents that score near zero on RRF even if they're perfectly within the time window. Additive boost gives recent docs a lift.

But the boost is intentionally small. Once the candidate set has already been filtered to the requested time window, recency should break ties — not bury the best semantic match.

### Context pruning — the 80% reduction problem

The goal: pass only what the LLM needs, not everything retrieved.

The greedy approach used here:
1. Sort by final re-rank score
2. Walk the sorted list, adding docs until the token budget or max-doc cap is hit
3. For high-scoring oversized docs, include a truncated version rather than skipping

What this still misses:
- **Semantic deduplication**: two docs about the same meeting can both survive
- **Extractive compression**: pull just the relevant sentences, not the whole chunk
- **Cross-doc synthesis**: merge three related docs into one coherent context block

The grouper is the first step toward solving that last problem: it clusters retrieved chunks into work narratives like Payments, Auth, Search, and Architecture.

### Privacy before indexing

Personal context retrieval has a trust boundary that normal document RAG does not.

The pipeline redacts common sensitive patterns before documents enter the index:
- emails
- API keys
- tokens
- passwords
- long card-like numbers

It also exposes deletion primitives (`delete_since`, `delete_recent`) to model the product requirement that users can remove recent context.

---

## Running the Demo

No API key required. The embedding model is open-source and local.

```bash
git clone https://github.com/yourusername/temporal-rag
cd temporal-rag
pip install -r requirements.txt

# Run demo queries + eval suite + ablation report
python demo.py

# Demo only
python demo.py --demo

# Eval only, without ablation
python demo.py --eval --no-ablation
```

Expected embedding backend:

```
Embedding backend: sentence-transformers:sentence-transformers/all-MiniLM-L6-v2
```

---

## Eval Results

These results use `sentence-transformers/all-MiniLM-L6-v2` on a deterministic 37-doc synthetic corpus spanning 90 days.

The pass criteria are intentionally stricter than "some relevant doc exists somewhere." A query must return enough context, hit a relevant result in the top ranks, and meet a minimum Precision@5 threshold.

```
Passed: 8/8
Avg temporal precision@5:  0.62
Avg temporal recall@5:     0.72
Avg semantic hit rate:     1.00
Avg MRR:                   0.917
Avg compression ratio:     0.73
```

### Ablation results

```
Variant                         Pass   P@5   R@5   MRR   Compression
----------------------------------------------------------------------
dense only                       8/8   0.57  0.67  0.94     0.79
sparse only                      8/8   0.68  0.76  0.91     0.79
recency only                     6/8   0.42  0.35  0.53     0.76
hybrid / no filters              6/8   0.45  0.59  0.89     0.35
hybrid + time/source filters     8/8   0.62  0.72  0.92     0.73
```

The synthetic corpus is intentionally lexical, so BM25 is strong. The important result is not that hybrid wins every scalar metric. The important result is that recency alone fails broad semantic cases, and hybrid retrieval needs query filters to avoid wasting top-k on the wrong time window or source type.

---

## Project Structure

```
temporal-rag/
├── src/
│   ├── __init__.py
│   └── temporal_rag.py      # Core pipeline
├── data/
│   ├── __init__.py
│   └── synthetic_corpus.py  # 37-doc synthetic personal context corpus
├── evals/
│   ├── __init__.py
│   └── eval_harness.py      # Eval cases, metrics, ablations
├── demo.py                  # Entry point
├── requirements.txt
└── README.md
```

---

## Current Limitations & What Would Fix Them

### 1. The corpus is synthetic and too clean

The corpus has realistic shapes, but real personal context is messier: repeated screenshots, incomplete meeting notes, notification fragments, private windows, and contradictory follow-ups.

**Fix**: Add a noisier corpus with duplicate events, partial snippets, irrelevant app activity, and near-identical chunks.

### 2. Query decomposition is still rule-based

"The week before I joined" or "sometime around the product launch" requires understanding personal calendar context. The rule-based parser doesn't handle relative references anchored to events.

**Fix**: LLM-based decomposition with a structured output schema and access to personal timeline metadata.

### 3. Narrative grouping is heuristic

The current grouper uses tags and simple labels. It can group obvious work threads, but it does not yet merge evidence into a true answer-ready summary.

**Fix**: Cluster by embedding similarity, then run extractive compression or a small summarisation pass per group.

### 4. Top-5 precision is still uneven

The architecture and incident queries pass because the first relevant item appears early, but their Precision@5 is only 0.20. That is acceptable for a small demo corpus, but not good enough for production.

**Fix**: Add a cross-encoder re-ranker and train/evaluate on human relevance judgements.

### 5. Flat temporal decay misses natural work rhythms

The current exponential decay treats all time uniformly. But Monday morning docs about "what I'm working on this week" are more valuable for weekly recall than Friday afternoon docs, even if the Friday docs are more recent.

**Fix**: Learn a personal temporal importance model from user feedback signals.

---

## What Production Would Add

1. **Better local embeddings** — BGE-M3 or a fine-tuned personal context model
2. **LLM-based query decomposition** — structured JSON output, handles arbitrarily complex temporal expressions
3. **Cross-encoder re-ranking** — a local cross-encoder for final ranking quality
4. **Extractive compression** — sentence-level pruning rather than whole-document selection
5. **Online learning** — user feedback signals update temporal decay and re-ranking weights
6. **Vector database** — Qdrant or pgvector for ANN search at scale
7. **Metadata filtering in storage** — push time and source filters into the vector DB query
8. **Privacy policy engine** — app-level allow/deny lists, private-window handling, and per-source retention

---

## Why This Problem Is Hard

The deeper issue: personal context retrieval sits at the intersection of three unsolved problems in retrieval.

**The vagueness problem**: "What was I working on?" has almost zero semantic content to embed. The query is mostly stop words. Standard dense retrieval degrades to random for these queries.

**The recency-relevance tension**: Older documents can be more relevant (the original design decision) than newer ones (the daily standup that referenced it). No fixed decay function resolves this without user feedback.

**The context assembly problem**: Retrieved documents are chunks of a larger story. The meeting note, the Slack follow-up, and the PR description are three pieces of the same narrative — but the retriever sees them as independent documents with independent scores.

This pipeline solves part of all three. The remaining work is turning ranked chunks into reliable, privacy-aware memory.
