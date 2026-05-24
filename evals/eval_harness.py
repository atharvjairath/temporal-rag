"""
evals/eval_harness.py
~~~~~~~~~~~~~~~~~~~~~
Evaluation harness for the temporal RAG pipeline.

Metrics
-------
  Temporal Precision@K  — of the top-K results, what fraction are within
                          the queried time window?
  Temporal Recall@K     — of all relevant docs (in time window), what
                          fraction appear in top-K?
  Semantic Hit Rate     — do the top-K results contain at least one doc
                          matching the topic keyword(s)?
  Mean Reciprocal Rank  — where does the first relevant result appear?
  Context Compression   — ratio of docs returned vs docs retrieved

These are custom metrics designed around the personal context retrieval problem.
In production, complement with human relevance judgements or LLM-as-judge.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from src.temporal_rag import Document, QueryResult, TemporalRAGPipeline


@dataclass
class EvalCase:
    """A single test case with a ground-truth relevance function."""
    query: str
    description: str
    is_relevant: Callable[[Document], bool]   # ground truth oracle
    expected_min_results: int = 1             # fail if fewer results returned
    min_precision_at_5: float = 0.2
    min_recall_at_5: float = 0.0
    max_first_relevant_rank: int = 5


@dataclass
class EvalResult:
    case: EvalCase
    result: QueryResult
    temporal_precision_at_5: float
    temporal_recall_at_5: float
    semantic_hit_rate: float
    mrr: float
    compression_ratio: float
    first_relevant_rank: int | None
    passed: bool


@dataclass
class AblationResult:
    name: str
    passed: int
    total: int
    avg_precision_at_5: float
    avg_recall_at_5: float
    avg_mrr: float
    avg_compression: float


def evaluate(pipeline: TemporalRAGPipeline, cases: list[EvalCase], now: datetime) -> list[EvalResult]:
    results = []
    for case in cases:
        qr = pipeline.query(case.query, now=now)
        all_relevant = [d for d in pipeline.documents if case.is_relevant(d)]

        # Temporal precision@5
        top5 = qr.context[:5]
        tp5 = sum(1 for sd in top5 if case.is_relevant(sd.doc)) / max(len(top5), 1)

        # Temporal recall@5
        tr5 = sum(1 for sd in top5 if case.is_relevant(sd.doc)) / max(len(all_relevant), 1)

        # Semantic hit rate: any relevant doc in full context?
        shr = any(case.is_relevant(sd.doc) for sd in qr.context)

        # MRR over full ranked list
        mrr = 0.0
        first_relevant_rank = None
        for rank, sd in enumerate(qr.ranked, start=1):
            if case.is_relevant(sd.doc):
                mrr = 1.0 / rank
                first_relevant_rank = rank
                break

        # Compression: pruned / total retrieved (lower = more aggressive pruning)
        total_retrieved = len(set(
            sd.doc.id for lst in [qr.dense_results, qr.sparse_results, qr.recency_results]
            for sd in lst
        ))
        compression = len(qr.context) / max(total_retrieved, 1)

        passed = (
            len(qr.context) >= case.expected_min_results
            and shr
            and tp5 >= case.min_precision_at_5
            and tr5 >= case.min_recall_at_5
            and first_relevant_rank is not None
            and first_relevant_rank <= case.max_first_relevant_rank
        )

        results.append(EvalResult(
            case=case,
            result=qr,
            temporal_precision_at_5=tp5,
            temporal_recall_at_5=tr5,
            semantic_hit_rate=float(shr),
            mrr=mrr,
            compression_ratio=compression,
            first_relevant_rank=first_relevant_rank,
            passed=passed,
        ))
    return results


def print_report(eval_results: list[EvalResult]) -> None:
    passed = sum(1 for r in eval_results if r.passed)
    total = len(eval_results)

    print("\n" + "=" * 70)
    print("TEMPORAL RAG — EVAL REPORT")
    print("=" * 70)

    for er in eval_results:
        status = "✓ PASS" if er.passed else "✗ FAIL"
        dq = er.result.query
        print(f"\n  {status}  \"{er.case.query}\"")
        print(f"           ({er.case.description})")
        print(f"           Parsed time: {dq.time_label}  |  Topic: '{dq.topic}'  |  Intent: {dq.intent}")
        print(f"           Context docs returned: {len(er.result.context)}")
        print(f"           Temporal precision@5: {er.temporal_precision_at_5:.2f}")
        print(f"           Temporal recall@5:    {er.temporal_recall_at_5:.2f}")
        print(f"           Semantic hit rate:    {er.semantic_hit_rate:.2f}")
        print(f"           MRR:                  {er.mrr:.3f}")
        print(f"           First relevant rank:  {er.first_relevant_rank or 'none'}")
        print(f"           Compression ratio:    {er.compression_ratio:.2f}")

        if er.result.context:
            print(f"           Top result: [{er.result.context[0].doc.source}] "
                  f"{er.result.context[0].doc.timestamp.strftime('%b %d')} — "
                  f"{er.result.context[0].doc.content[:70]}...")

    print("\n" + "-" * 70)
    print(f"  Passed: {passed}/{total}")

    if eval_results:
        avg_tp5  = statistics.mean(r.temporal_precision_at_5 for r in eval_results)
        avg_tr5  = statistics.mean(r.temporal_recall_at_5 for r in eval_results)
        avg_shr  = statistics.mean(r.semantic_hit_rate for r in eval_results)
        avg_mrr  = statistics.mean(r.mrr for r in eval_results)
        avg_comp = statistics.mean(r.compression_ratio for r in eval_results)

        print(f"  Avg temporal precision@5:  {avg_tp5:.2f}")
        print(f"  Avg temporal recall@5:     {avg_tr5:.2f}")
        print(f"  Avg semantic hit rate:     {avg_shr:.2f}")
        print(f"  Avg MRR:                   {avg_mrr:.3f}")
        print(f"  Avg compression ratio:     {avg_comp:.2f}  (lower = more pruning)")
    print("=" * 70 + "\n")


def run_ablation(documents: list[Document], cases: list[EvalCase], now: datetime) -> list[AblationResult]:
    variants = [
        ("dense only", {"active_retrievers": ("dense",)}),
        ("sparse only", {"active_retrievers": ("sparse",)}),
        ("recency only", {"active_retrievers": ("recency",)}),
        ("hybrid / no filters", {"active_retrievers": ("dense", "sparse", "recency"), "strict_query_filters": False}),
        ("hybrid + time/source filters", {"active_retrievers": ("dense", "sparse", "recency"), "strict_query_filters": True}),
    ]

    rows = []
    for name, kwargs in variants:
        pipeline = TemporalRAGPipeline(token_budget=2000, temporal_weight=0.3, top_k=20, **kwargs)
        pipeline.index(documents)
        results = evaluate(pipeline, cases, now=now)
        rows.append(AblationResult(
            name=name,
            passed=sum(r.passed for r in results),
            total=len(results),
            avg_precision_at_5=statistics.mean(r.temporal_precision_at_5 for r in results),
            avg_recall_at_5=statistics.mean(r.temporal_recall_at_5 for r in results),
            avg_mrr=statistics.mean(r.mrr for r in results),
            avg_compression=statistics.mean(r.compression_ratio for r in results),
        ))
    return rows


def print_ablation_report(rows: list[AblationResult]) -> None:
    print("\n" + "=" * 70)
    print("TEMPORAL RAG — ABLATION REPORT")
    print("=" * 70)
    print("Variant                         Pass   P@5   R@5   MRR   Compression")
    print("-" * 70)
    for row in rows:
        print(
            f"{row.name:<31} "
            f"{row.passed:>2}/{row.total:<2}  "
            f"{row.avg_precision_at_5:>4.2f}  "
            f"{row.avg_recall_at_5:>4.2f}  "
            f"{row.avg_mrr:>4.2f}  "
            f"{row.avg_compression:>7.2f}"
        )
    print("=" * 70 + "\n")


def build_eval_cases(now: datetime) -> list[EvalCase]:
    from datetime import timedelta

    last_week_start = now - timedelta(days=7)
    last_week_end = now

    return [
        EvalCase(
            query="what was I working on last week?",
            description="Broad temporal recall — last 7 days, no topic constraint",
            is_relevant=lambda d: last_week_start <= d.timestamp <= last_week_end,
            expected_min_results=3,
            min_precision_at_5=0.8,
            min_recall_at_5=0.4,
        ),
        EvalCase(
            query="what did I do on the payments project recently?",
            description="Topic-filtered recency — payments + recent",
            is_relevant=lambda d: (
                "payments" in d.tags or "payment" in d.content.lower()
            ) and d.timestamp >= now - timedelta(days=30),
            expected_min_results=2,
            min_precision_at_5=0.6,
            min_recall_at_5=0.3,
        ),
        EvalCase(
            query="what did I discuss with Sarah?",
            description="Entity-anchored recall — person name filter",
            is_relevant=lambda d: "sarah" in d.tags or "Sarah" in d.content,
            expected_min_results=2,
            min_precision_at_5=0.4,
        ),
        EvalCase(
            query="summarize my work on search this month",
            description="Summarise intent — search + this month",
            is_relevant=lambda d: (
                "search" in d.tags or "search" in d.content.lower()
            ) and d.timestamp >= now - timedelta(days=30),
            expected_min_results=1,
            min_precision_at_5=0.2,
        ),
        EvalCase(
            query="what bugs did I fix last week?",
            description="Intent + topic — bug fixing in last 7 days",
            is_relevant=lambda d: (
                "bug" in d.tags or "fix" in d.content.lower() or "bug" in d.content.lower()
            ) and last_week_start <= d.timestamp <= last_week_end,
            expected_min_results=1,
            min_precision_at_5=0.2,
        ),
        EvalCase(
            query="what meetings did I have about architecture?",
            description="Source filter + topic — meetings about architecture",
            is_relevant=lambda d: (
                d.source == "meeting"
                and ("architecture" in d.tags or "architecture" in d.content.lower())
            ),
            expected_min_results=1,
            min_precision_at_5=0.2,
        ),
        EvalCase(
            query="what was I reading about AI recently?",
            description="Topic + source — screen captures about AI/ML research",
            is_relevant=lambda d: (
                d.source == "screen"
                and any(t in d.tags for t in ["ai", "llm", "rag", "ml", "research", "embeddings"])
            ) and d.timestamp >= now - timedelta(days=30),
            expected_min_results=1,
            min_precision_at_5=0.4,
        ),
        EvalCase(
            query="what production incidents happened last month?",
            description="Topic recall — incidents in last 30 days",
            is_relevant=lambda d: (
                "incident" in d.tags or "postmortem" in d.tags
            ) and d.timestamp >= now - timedelta(days=30),
            expected_min_results=1,
            min_precision_at_5=0.2,
        ),
    ]
