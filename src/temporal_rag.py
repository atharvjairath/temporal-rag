"""
temporal_rag.py
~~~~~~~~~~~~~~~
Temporal RAG pipeline for personal context queries.

Handles broad, time-anchored queries like:
  - "what was I working on last week?"
  - "what did I discuss with Sarah in Q4?"
  - "summarize my work on the payments feature"

Pipeline stages:
  1. QueryDecomposer   — extracts time range + topic intent from raw query
  2. MultiRetriever    — runs dense, sparse, and recency-boosted retrieval in parallel
  3. TemporalReranker  — fuses results with temporal decay + semantic relevance
  4. ContextPruner     — trims context to token budget while preserving critical info
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, Optional
import numpy as np

_MODEL_CACHE = {}
_CAUSAL_LM_CACHE = {}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A single chunk of personal context (meeting note, screen capture, message)."""
    id: str
    content: str
    timestamp: datetime
    source: str                         # "meeting", "screen", "message", "doc"
    tags: list[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def age_days(self, now: datetime) -> float:
        return max(0.0, (now - self.timestamp).total_seconds() / 86400)


class PrivacyFilter:
    """
    Redacts high-risk personal data before it enters the retrieval index.

    This is intentionally small, but it models the production invariant:
    retrieval should never need raw secrets, passwords, or long-lived tokens.
    """

    _PATTERNS: list[tuple[re.Pattern, str]] = [
        (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I), "[email]"),
        (re.compile(r"\b(?:api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,;]+", re.I), "[secret]"),
        (re.compile(r"\bsk-[A-Za-z0-9]{12,}\b"), "[secret]"),
        (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[card]"),
    ]

    def redact(self, content: str) -> str:
        redacted = content
        for pattern, replacement in self._PATTERNS:
            redacted = pattern.sub(replacement, redacted)
        return redacted


@dataclass
class DecomposedQuery:
    """Structured representation of a user query after parsing."""
    raw: str
    topic: str                          # semantic content to search for
    time_start: Optional[datetime]
    time_end: Optional[datetime]
    time_label: str                     # human-readable: "last week", "Q4", etc.
    intent: str                         # "recall", "summarise", "find_entity"
    named_entities: list[str] = field(default_factory=list)
    source_filters: list[str] = field(default_factory=list)


@dataclass
class ScoredDocument:
    doc: Document
    dense_score: float = 0.0
    sparse_score: float = 0.0
    recency_score: float = 0.0
    final_score: float = 0.0


@dataclass
class ContextGroup:
    """A small cross-document narrative assembled from retrieved context."""
    label: str
    documents: list[ScoredDocument]
    evidence_count: int


# ---------------------------------------------------------------------------
# Stage 1 — Query Decomposer
# ---------------------------------------------------------------------------

class QueryDecomposer:
    """
    Extracts temporal constraints and topic intent from a natural language query.

    In production this would call an LLM with a structured output schema.
    This implementation uses deterministic rules so the demo runs without an API key.

    Design notes
    ------------
    The key insight: personal context queries have TWO independent axes:
      - WHEN  — often vague ("last week", "a few months ago", "yesterday")
      - WHAT  — often vague too ("that thing with Sarah", "the API stuff")

    Decomposing them lets us apply different retrieval strategies per axis
    rather than treating the raw query as a single embedding lookup.
    """

    # Relative time patterns → (start_offset_days, end_offset_days)
    _RELATIVE_PATTERNS: list[tuple[re.Pattern, tuple[int, int], str]] = [
        (re.compile(r"\byesterday\b", re.I),            (1, 0),    "yesterday"),
        (re.compile(r"\btoday\b", re.I),                (0, 0),    "today"),
        (re.compile(r"\blast\s+week\b", re.I),          (7, 0),    "last week"),
        (re.compile(r"\bthis\s+week\b", re.I),          (7, 0),    "this week"),
        (re.compile(r"\blast\s+month\b", re.I),         (30, 0),   "last month"),
        (re.compile(r"\bthis\s+month\b", re.I),         (30, 0),   "this month"),
        (re.compile(r"\blast\s+(\d+)\s+days?\b", re.I), None,      "last N days"),
        (re.compile(r"\brecently\b|\brecent\b", re.I),  (14, 0),   "recently"),
        (re.compile(r"\bq([1-4])\b", re.I),             None,      "quarter"),
    ]

    _SOURCE_KEYWORDS = {
        "meeting": ["meeting", "meetings", "call", "calls", "sync", "standup", "1:1"],
        "screen": ["reading", "read", "browser", "article", "website", "screen", "page"],
        "message": ["slack", "message", "messages", "chat", "dm"],
        "doc": ["doc", "docs", "document", "note", "notes", "rfc"],
    }

    _INTENT_KEYWORDS = {
        "summarise": ["summarize", "summarise", "overview", "recap", "summary"],
        "find_entity": ["who", "which", "find", "where", "when exactly"],
        "recall":  ["what", "did i", "was i", "working on", "talked about"],
    }

    def decompose(self, query: str, now: Optional[datetime] = None) -> DecomposedQuery:
        now = now or datetime.now()
        time_start, time_end, time_label = self._parse_time(query, now)
        topic = self._extract_topic(query)
        intent = self._classify_intent(query)
        entities = self._extract_entities(query)
        sources = self._extract_sources(query)

        return DecomposedQuery(
            raw=query,
            topic=topic,
            time_start=time_start,
            time_end=time_end,
            time_label=time_label,
            intent=intent,
            named_entities=entities,
            source_filters=sources,
        )

    def _parse_time(
        self, query: str, now: datetime
    ) -> tuple[Optional[datetime], Optional[datetime], str]:
        q = query.lower()

        # "last N days"
        m = re.search(r"last\s+(\d+)\s+days?", q)
        if m:
            n = int(m.group(1))
            return now - timedelta(days=n), now, f"last {n} days"

        # quarter references: "Q2", "q3 last year"
        m = re.search(r"\bq([1-4])\b", q)
        if m:
            qn = int(m.group(1))
            year = now.year if "last year" not in q else now.year - 1
            starts = {1: (1,1), 2: (4,1), 3: (7,1), 4: (10,1)}
            ends   = {1: (3,31), 2: (6,30), 3: (9,30), 4: (12,31)}
            sm, sd = starts[qn]
            em, ed = ends[qn]
            return (
                datetime(year, sm, sd),
                datetime(year, em, ed, 23, 59),
                f"Q{qn} {year}",
            )

        for pattern, offsets, label in self._RELATIVE_PATTERNS:
            if pattern.search(query) and offsets is not None:
                start_off, end_off = offsets
                return (
                    now - timedelta(days=start_off),
                    now - timedelta(days=end_off),
                    label,
                )

        # No time anchor found — no constraint
        return None, None, "any time"

    def _extract_topic(self, query: str) -> str:
        # Strip temporal + filler phrases to isolate the semantic topic
        stopwords = {
            "what", "was", "were", "have", "had", "did", "does", "has",
            "i", "me", "my", "working", "on", "do", "last", "week", "weeks",
            "month", "months", "this", "recently", "yesterday", "today",
            "about", "tell", "summarize", "summarise", "show", "the", "a",
            "an", "any", "which", "that", "how", "many", "much", "happened",
            "meeting", "meetings", "call", "calls", "reading", "read",
            "browser", "article", "website", "screen", "slack", "message",
            "messages", "chat", "doc", "docs", "document", "note", "notes",
        }
        normalised_stopwords = {_normalise_token(t) for t in stopwords}
        tokens = _tokenize(query)
        topic_tokens = [t for t in tokens if t not in normalised_stopwords and len(t) > 1]
        return " ".join(topic_tokens) if topic_tokens else query

    def _classify_intent(self, query: str) -> str:
        q = query.lower()
        for intent, keywords in self._INTENT_KEYWORDS.items():
            if any(kw in q for kw in keywords):
                return intent
        return "recall"

    def _extract_entities(self, query: str) -> list[str]:
        # Naive: capitalised words not at sentence start
        return re.findall(r"(?<!\. )\b[A-Z][a-z]+\b", query)

    def _extract_sources(self, query: str) -> list[str]:
        q = query.lower()
        sources = []
        for source, keywords in self._SOURCE_KEYWORDS.items():
            if any(re.search(rf"\b{re.escape(keyword)}\b", q) for keyword in keywords):
                sources.append(source)
        return sources


class LocalLLMQueryDecomposer(QueryDecomposer):
    """
    Query decomposer backed by a lightweight local Qwen model.

    The model proposes a JSON decomposition. Deterministic parsing still owns
    date arithmetic and schema repair because bad JSON should not break search.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        super().__init__()
        self.model_name = model_name
        self.backend_name = f"qwen:{model_name}"
        self._tokenizer = None
        self._model = None

    def decompose(self, query: str, now: Optional[datetime] = None) -> DecomposedQuery:
        now = now or datetime.now()
        base = super().decompose(query, now)
        proposal = self._generate_json(query)

        proposed_topic = self._clean_topic(proposal.get("topic"))
        topic = self._merge_topic(base.topic, proposed_topic)
        intent = base.intent
        entities = self._merge_unique(base.named_entities, self._clean_entities(proposal.get("entities")))
        sources = base.source_filters

        return DecomposedQuery(
            raw=query,
            topic=topic,
            time_start=base.time_start,
            time_end=base.time_end,
            time_label=base.time_label,
            intent=intent,
            named_entities=entities,
            source_filters=sources,
        )

    def _generate_json(self, query: str) -> dict:
        prompt = self._prompt(query)
        tokenizer, model = self._load_model()
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
        output = model.generate(
            **encoded,
            max_new_tokens=120,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = output[0][encoded["input_ids"].shape[-1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        return self._parse_jsonish(text)

    def _prompt(self, query: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You extract retrieval query metadata. Return only valid JSON "
                    "with keys topic, intent, entities, sources. Allowed intents: "
                    "recall, summarise, find_entity. Allowed sources: meeting, "
                    "screen, message, doc."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Query: " + query + "\n"
                    "Return compact JSON. Example: "
                    '{"topic":"payments latency","intent":"recall",'
                    '"entities":["Sarah"],"sources":["meeting"]}'
                ),
            },
        ]
        tokenizer, _ = self._load_model()
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return messages[0]["content"] + "\n" + messages[1]["content"] + "\nJSON:"

    def _load_model(self):
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        if self.model_name in _CAUSAL_LM_CACHE:
            self._tokenizer, self._model = _CAUSAL_LM_CACHE[self.model_name]
            return self._tokenizer, self._model

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            model.eval()
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the local Qwen query decomposition model. "
                "Install dependencies with `pip install -r requirements.txt` and "
                f"ensure `{self.model_name}` can be downloaded or is cached locally."
            ) from exc

        self._tokenizer = tokenizer
        self._model = model
        _CAUSAL_LM_CACHE[self.model_name] = (tokenizer, model)
        return tokenizer, model

    def _parse_jsonish(self, text: str) -> dict:
        text = text.strip()
        if not text:
            return {}

        match = re.search(r"\{.*\}", text, re.S)
        candidate = match.group(0) if match else text
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return self._parse_key_value_text(text)

    def _parse_key_value_text(self, text: str) -> dict:
        fields = {}
        for key in ["topic", "intent", "entities", "sources"]:
            m = re.search(rf"{key}\s*[:=]\s*([^;\n]+)", text, re.I)
            if m:
                fields[key] = m.group(1).strip()
        return fields

    def _clean_topic(self, value) -> str:
        if not isinstance(value, str):
            return ""
        value = value.strip().strip('"')
        if not value or value.lower() in {"none", "null", "n/a"}:
            return ""
        return value.lower()

    def _clean_intent(self, value) -> str:
        if not isinstance(value, str):
            return ""
        value = value.lower().strip()
        if value in {"recall", "summarise", "summarize"}:
            return "summarise" if value == "summarize" else value
        if value in {"find", "find_entity", "entity"}:
            return "find_entity"
        return ""

    def _clean_entities(self, value) -> list[str]:
        blocked = {
            "today", "yesterday", "recently", "recent", "last_week", "this_week",
            "last_month", "this_month", "week", "month", "q1", "q2", "q3", "q4",
        }
        if isinstance(value, str):
            parts = re.split(r"[,|]", value)
            return [
                p.strip()
                for p in parts
                if p.strip() and p.strip().lower() not in {"none", "null"} | blocked
            ]
        if isinstance(value, list):
            return [
                str(v).strip()
                for v in value
                if str(v).strip() and str(v).strip().lower() not in blocked
            ]
        return []

    def _clean_sources(self, value) -> list[str]:
        allowed = {"meeting", "screen", "message", "doc"}
        if isinstance(value, str):
            raw = re.split(r"[,|]", value)
        elif isinstance(value, list):
            raw = [str(v) for v in value]
        else:
            raw = []
        return [source for source in (item.lower().strip() for item in raw) if source in allowed]

    def _is_useful_topic(self, topic: str) -> bool:
        if not topic:
            return False
        weak = {"meeting", "meetings", "screen", "message", "messages", "doc", "docs"}
        tokens = set(_tokenize(topic))
        return bool(tokens - weak)

    def _merge_topic(self, base_topic: str, proposed_topic: str) -> str:
        if not self._is_useful_topic(proposed_topic):
            return base_topic
        if not base_topic or base_topic == proposed_topic:
            return proposed_topic

        merged = []
        seen = set()
        for token in _tokenize(f"{proposed_topic} {base_topic}"):
            if token not in seen:
                merged.append(token)
                seen.add(token)
        return " ".join(merged)

    def _merge_unique(self, first: list[str], second: list[str]) -> list[str]:
        merged = []
        seen = set()
        for item in first + second:
            key = item.lower()
            if key not in seen:
                merged.append(item)
                seen.add(key)
        return merged


# ---------------------------------------------------------------------------
# Stage 2 — Retrievers
# ---------------------------------------------------------------------------

def _normalise_token(token: str) -> str:
    token = token.lower()
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _tokenize(text: str) -> list[str]:
    return [_normalise_token(t) for t in re.findall(r"\b[a-z0-9]+\b", text.lower())]


def _searchable_text(doc: Document) -> str:
    return " ".join([doc.content, doc.source, *doc.tags])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


class OpenSourceEmbedder:
    """
    Embedding backend.

    Uses the Apache-2.0 `sentence-transformers/all-MiniLM-L6-v2` model locally.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self.backend_name = f"sentence-transformers:{model_name}"

    def embed(self, text: str) -> np.ndarray:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[np.ndarray]:
        model = self._load_model()
        vectors = model.encode(texts, normalize_embeddings=True)
        return [np.asarray(v, dtype=float) for v in vectors]

    def _load_model(self):
        if self._model is not None:
            return self._model
        if self.model_name in _MODEL_CACHE:
            self._model = _MODEL_CACHE[self.model_name]
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            _MODEL_CACHE[self.model_name] = self._model
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the local sentence-transformers embedding model. "
                "Install dependencies with `pip install -r requirements.txt` and "
                f"ensure `{self.model_name}` can be downloaded or is cached locally."
            ) from exc
        return self._model


class DenseRetriever:
    """
    Embedding-based semantic retrieval.

    Uses cosine similarity between query embedding and document embeddings.
    The default model is open-source and local: all-MiniLM-L6-v2.

    Design notes
    ------------
    Dense retrieval excels at semantic similarity ("API work" ↔ "REST endpoints")
    but struggles with exact terms, dates, and named entities.
    That's why we run it alongside sparse retrieval.
    """

    def __init__(self, embedder: Optional[OpenSourceEmbedder] = None):
        self.embedder = embedder or OpenSourceEmbedder()

    def retrieve(
        self,
        query: DecomposedQuery,
        documents: list[Document],
        top_k: int = 20,
    ) -> list[ScoredDocument]:
        q_vec = self._embed(query.topic)
        results = []
        for doc in documents:
            if doc.embedding is None:
                doc.embedding = self._embed(_searchable_text(doc))
            score = cosine_similarity(q_vec, doc.embedding)
            results.append(ScoredDocument(doc=doc, dense_score=score))
        results.sort(key=lambda x: x.dense_score, reverse=True)
        return results[:top_k]

    def embed_documents(self, documents: list[Document]) -> None:
        texts = [_searchable_text(doc) for doc in documents]
        vectors = self.embedder.embed_many(texts)
        for doc, vector in zip(documents, vectors):
            doc.embedding = vector

    def _embed(self, text: str) -> np.ndarray:
        return self.embedder.embed(text)


class SparseRetriever:
    """
    BM25-style keyword retrieval.

    Captures exact term matches, names, and technical jargon that dense
    retrieval often misses. Especially important for personal context where
    proper nouns (project names, people) are high-signal.

    Design notes
    ------------
    BM25 parameters:
      k1 = 1.5  (term frequency saturation — higher = more weight on rare terms)
      b  = 0.75 (length normalisation — 0 = no normalisation, 1 = full)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def retrieve(
        self,
        query: DecomposedQuery,
        documents: list[Document],
        top_k: int = 20,
    ) -> list[ScoredDocument]:
        query_terms = set(_tokenize(query.topic))
        if query.named_entities:
            query_terms.update(_normalise_token(e) for e in query.named_entities)

        corpus = [_tokenize(_searchable_text(d)) for d in documents]
        avg_len = sum(len(c) for c in corpus) / max(len(corpus), 1)
        doc_freq = {}
        for tokens in corpus:
            for term in set(tokens):
                doc_freq[term] = doc_freq.get(term, 0) + 1

        N = len(documents)
        results = []
        for doc, tokens in zip(documents, corpus):
            score = 0.0
            tf_map: dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1
            doc_len = len(tokens)

            for term in query_terms:
                if term not in doc_freq:
                    continue
                tf = tf_map.get(term, 0)
                df = doc_freq[term]
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                tf_norm = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / avg_len)
                )
                score += idf * tf_norm

            results.append(ScoredDocument(doc=doc, sparse_score=score))

        results.sort(key=lambda x: x.sparse_score, reverse=True)
        return results[:top_k]


class RecencyRetriever:
    """
    Pure recency-based retrieval — always returns recent docs within time window.

    Exists because dense + sparse retrieval can bury temporally relevant docs
    if the query topic is vague. For "what did I do last week?", recency IS
    the primary signal — we want everything from that window, semantics second.

    Recency score uses exponential decay: score = exp(-λ * age_days)
    λ controls the half-life. λ=0.1 → docs older than ~10 days score < 0.37
    """

    def __init__(self, decay_lambda: float = 0.1):
        self.decay_lambda = decay_lambda

    def retrieve(
        self,
        query: DecomposedQuery,
        documents: list[Document],
        now: datetime,
        top_k: int = 20,
    ) -> list[ScoredDocument]:
        results = []
        for doc in documents:
            # Hard filter: apply time window if one was parsed
            if query.time_start and doc.timestamp < query.time_start:
                continue
            if query.time_end and doc.timestamp > query.time_end:
                continue

            age = doc.age_days(now)
            recency = math.exp(-self.decay_lambda * age)
            results.append(ScoredDocument(doc=doc, recency_score=recency))

        results.sort(key=lambda x: x.recency_score, reverse=True)
        return results[:top_k]


# ---------------------------------------------------------------------------
# Stage 3 — Temporal Re-ranker
# ---------------------------------------------------------------------------

class TemporalReranker:
    """
    Fuses dense, sparse, and recency scores into a single ranked list.

    Score fusion strategy: Reciprocal Rank Fusion (RRF) + temporal boost.

    Why RRF instead of simple weighted sum?
    - Score scales differ wildly across retrievers (cosine 0–1, BM25 0–20+)
    - RRF uses rank position, which is scale-invariant
    - Empirically outperforms linear combination on most benchmarks

    Temporal boost is additive, not multiplicative — so a doc inside the
    query's time window always gets a lift, even if semantically weak.

    Parameters
    ----------
    rrf_k : int
        RRF smoothing constant. k=10 is intentionally sharper for the small
        personal-memory corpora used in this demo.
    temporal_weight : float
        How much to boost documents within the queried time window.
        0.0 = pure semantic, 1.0 = strongly temporal.
    weights : dict
        Per-retriever weights for the RRF combination.
    """

    def __init__(
        self,
        rrf_k: int = 10,
        temporal_weight: float = 0.3,
        weights: Optional[dict[str, float]] = None,
    ):
        self.rrf_k = rrf_k
        self.temporal_weight = temporal_weight
        self.weights = weights or {"dense": 0.5, "sparse": 0.3, "recency": 0.2}

    def rerank(
        self,
        dense_results: list[ScoredDocument],
        sparse_results: list[ScoredDocument],
        recency_results: list[ScoredDocument],
        query: DecomposedQuery,
        now: datetime,
    ) -> list[ScoredDocument]:

        # Merge all unique docs
        all_docs: dict[str, ScoredDocument] = {}
        for sd in dense_results + sparse_results + recency_results:
            if sd.doc.id not in all_docs:
                all_docs[sd.doc.id] = ScoredDocument(
                    doc=sd.doc,
                    dense_score=sd.dense_score,
                    sparse_score=sd.sparse_score,
                    recency_score=sd.recency_score,
                )
            else:
                existing = all_docs[sd.doc.id]
                existing.dense_score = max(existing.dense_score, sd.dense_score)
                existing.sparse_score = max(existing.sparse_score, sd.sparse_score)
                existing.recency_score = max(existing.recency_score, sd.recency_score)

        merged = list(all_docs.values())

        # Compute RRF score per retriever
        rrf_scores: dict[str, dict[str, float]] = {
            "dense": self._rrf_scores(dense_results),
            "sparse": self._rrf_scores(sparse_results),
            "recency": self._rrf_scores(recency_results),
        }

        has_time_constraint = query.time_start is not None

        for sd in merged:
            rrf = sum(
                self.weights[name] * rrf_scores[name].get(sd.doc.id, 0.0)
                for name in ["dense", "sparse", "recency"]
            )

            # Temporal boost: add extra weight if doc is inside the query window
            temporal_boost = 0.0
            if has_time_constraint and sd.recency_score > 0:
                temporal_boost = self.temporal_weight * 0.03 * sd.recency_score

            sd.final_score = rrf + temporal_boost

        merged.sort(key=lambda x: x.final_score, reverse=True)
        return merged

    def _rrf_scores(self, results: list[ScoredDocument]) -> dict[str, float]:
        return {
            sd.doc.id: 1.0 / (self.rrf_k + rank + 1)
            for rank, sd in enumerate(results)
        }


# ---------------------------------------------------------------------------
# Stage 4 — Context Pruner
# ---------------------------------------------------------------------------

class ContextPruner:
    """
    Trims the retrieved context to a token budget while preserving signal.

    Strategy: greedy selection by marginal utility.

    For each candidate document (already ranked by final_score):
      1. Estimate token cost
      2. If adding it stays within budget → include
      3. If it would overflow → try to include a truncated version if it's
         high-signal enough (final_score above the include_threshold)

    The "80% reduction without losing the critical 1%" problem is real.
    The key insight: high-rank docs are not always the most information-dense.
    A one-line "Meeting with Sarah — decided to launch Thursday" is worth
    more than 3 paragraphs of meeting transcript that say the same thing.

    In production, replace _estimate_tokens with tiktoken and add an
    LLM-based extractive summarisation pass for oversized high-value docs.
    """

    def __init__(
        self,
        token_budget: int = 2000,
        include_threshold: float = 0.01,
        max_docs: int = 12,
    ):
        self.token_budget = token_budget
        self.include_threshold = include_threshold
        self.max_docs = max_docs

    def prune(self, ranked: list[ScoredDocument]) -> list[ScoredDocument]:
        selected = []
        remaining_budget = self.token_budget

        for sd in ranked:
            if len(selected) >= self.max_docs:
                break
            if sd.final_score < self.include_threshold:
                break

            cost = self._estimate_tokens(sd.doc.content)

            if cost <= remaining_budget:
                selected.append(sd)
                remaining_budget -= cost
            elif remaining_budget > 50 and sd.final_score > 0.05:
                # High-value but oversized: truncate to fit
                truncated_content = self._truncate(sd.doc.content, remaining_budget)
                truncated_doc = Document(
                    id=sd.doc.id + "_trunc",
                    content=truncated_content,
                    timestamp=sd.doc.timestamp,
                    source=sd.doc.source,
                    tags=sd.doc.tags,
                )
                selected.append(ScoredDocument(
                    doc=truncated_doc,
                    dense_score=sd.dense_score,
                    sparse_score=sd.sparse_score,
                    recency_score=sd.recency_score,
                    final_score=sd.final_score,
                ))
                break

        return selected

    def _estimate_tokens(self, text: str) -> int:
        # ~4 chars per token is a reasonable approximation (use tiktoken in prod)
        return max(1, len(text) // 4)

    def _truncate(self, text: str, token_budget: int) -> str:
        max_chars = token_budget * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rsplit(" ", 1)[0] + " [truncated]"


# ---------------------------------------------------------------------------
# Stage 5 — Narrative grouping
# ---------------------------------------------------------------------------

class NarrativeGrouper:
    """
    Groups retrieved chunks into small work narratives.

    Retrieval returns fragments. Personal recall often wants the story:
    the meeting, the follow-up message, the PR, and the deployment note
    should read as one thread of work.
    """

    _GENERIC_TAGS = {
        "research", "reading", "meeting", "standup", "doc", "screen",
        "message", "production", "deploy", "testing", "coverage",
    }

    def group(self, context: list[ScoredDocument], max_groups: int = 5) -> list[ContextGroup]:
        buckets: dict[str, list[ScoredDocument]] = {}
        for sd in context:
            label = self._label(sd.doc)
            buckets.setdefault(label, []).append(sd)

        groups = [
            ContextGroup(label=label, documents=docs, evidence_count=len(docs))
            for label, docs in buckets.items()
        ]
        groups.sort(
            key=lambda g: (g.evidence_count, max(sd.final_score for sd in g.documents)),
            reverse=True,
        )
        return groups[:max_groups]

    def _label(self, doc: Document) -> str:
        for tag in doc.tags:
            if tag not in self._GENERIC_TAGS:
                return tag.replace("-", " ").title()
        tokens = _tokenize(doc.content)
        for token in tokens:
            if token not in {"the", "and", "for", "with", "from", "that"} and len(token) > 4:
                return token.title()
        return doc.source.title()


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

class TemporalRAGPipeline:
    """
    End-to-end pipeline: raw query → pruned, ranked context.

    Usage
    -----
    pipeline = TemporalRAGPipeline()
    pipeline.index(documents)
    result = pipeline.query("what was I working on last week?")
    for item in result.context:
        print(item.doc.timestamp, item.doc.content[:80])
    """

    def __init__(
        self,
        token_budget: int = 2000,
        temporal_weight: float = 0.3,
        top_k: int = 20,
        active_retrievers: Iterable[str] = ("dense", "sparse", "recency"),
        strict_query_filters: bool = True,
        query_decomposer: Optional[QueryDecomposer] = None,
    ):
        self.privacy = PrivacyFilter()
        self.decomposer = query_decomposer or QueryDecomposer()
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()
        self.recency = RecencyRetriever()
        self.reranker = TemporalReranker(temporal_weight=temporal_weight)
        self.pruner = ContextPruner(token_budget=token_budget)
        self.grouper = NarrativeGrouper()
        self.documents: list[Document] = []
        self.top_k = top_k
        self.active_retrievers = set(active_retrievers)
        self.strict_query_filters = strict_query_filters

    def index(self, documents: list[Document]) -> None:
        self.documents = [
            Document(
                id=doc.id,
                content=self.privacy.redact(doc.content),
                timestamp=doc.timestamp,
                source=doc.source,
                tags=list(doc.tags),
            )
            for doc in documents
        ]
        self.dense.embed_documents(self.documents)

    def delete_since(self, cutoff: datetime) -> int:
        """Delete indexed context at or after a timestamp."""
        before = len(self.documents)
        self.documents = [doc for doc in self.documents if doc.timestamp < cutoff]
        return before - len(self.documents)

    def delete_recent(self, now: datetime, days: int = 1) -> int:
        """Delete indexed context from the last N days."""
        return self.delete_since(now - timedelta(days=days))

    def query(self, raw_query: str, now: Optional[datetime] = None):
        now = now or datetime.now()
        dq = self.decomposer.decompose(raw_query, now)
        candidates = self._candidate_documents(dq)

        dense_r = (
            self.dense.retrieve(dq, candidates, top_k=self.top_k)
            if "dense" in self.active_retrievers else []
        )
        sparse_r = (
            self.sparse.retrieve(dq, candidates, top_k=self.top_k)
            if "sparse" in self.active_retrievers else []
        )
        recency_r = (
            self.recency.retrieve(dq, candidates, now=now, top_k=self.top_k)
            if "recency" in self.active_retrievers else []
        )

        ranked = self.reranker.rerank(dense_r, sparse_r, recency_r, dq, now)
        pruned = self.pruner.prune(ranked)
        groups = self.grouper.group(pruned)

        return QueryResult(
            query=dq,
            dense_results=dense_r,
            sparse_results=sparse_r,
            recency_results=recency_r,
            ranked=ranked,
            context=pruned,
            groups=groups,
        )

    def _candidate_documents(self, query: DecomposedQuery) -> list[Document]:
        if not self.strict_query_filters:
            return self.documents

        candidates = []
        for doc in self.documents:
            if query.time_start and doc.timestamp < query.time_start:
                continue
            if query.time_end and doc.timestamp > query.time_end:
                continue
            if query.source_filters and doc.source not in query.source_filters:
                continue
            candidates.append(doc)

        return candidates or self.documents


@dataclass
class QueryResult:
    query: DecomposedQuery
    dense_results: list[ScoredDocument]
    sparse_results: list[ScoredDocument]
    recency_results: list[ScoredDocument]
    ranked: list[ScoredDocument]
    context: list[ScoredDocument]
    groups: list[ContextGroup] = field(default_factory=list)
