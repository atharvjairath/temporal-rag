"""
demo.py
~~~~~~~
Interactive demo + eval runner for the temporal RAG pipeline.

Run:
    python demo.py          # runs demo queries + eval suite
    python demo.py --eval   # eval only
    python demo.py --demo   # demo only
"""

import sys
from datetime import datetime

sys.path.insert(0, ".")

from data.synthetic_corpus import build_corpus
from src.temporal_rag import TemporalRAGPipeline
from evals.eval_harness import (
    build_eval_cases,
    evaluate,
    print_ablation_report,
    print_report,
    run_ablation,
)


NOW = datetime(2025, 5, 20, 12, 0)   # fixed "now" for reproducibility

DEMO_QUERIES = [
    "what was I working on last week?",
    "what did I do on the payments project recently?",
    "what did I discuss with Sarah?",
    "summarize my work on search this month",
    "what bugs did I fix last week?",
]


def run_demo(pipeline: TemporalRAGPipeline) -> None:
    print("\n" + "=" * 70)
    print("TEMPORAL RAG — DEMO QUERIES")
    print("=" * 70)

    for query in DEMO_QUERIES:
        result = pipeline.query(query, now=NOW)
        dq = result.query

        print(f"\n┌─ Query: \"{query}\"")
        print(f"│  Parsed time: {dq.time_label}  |  Topic: '{dq.topic}'  |  Intent: {dq.intent}")
        if dq.named_entities:
            print(f"│  Entities: {dq.named_entities}")
        if dq.source_filters:
            print(f"│  Source filters: {dq.source_filters}")
        print(f"│  Retrieved: dense={len(result.dense_results)} "
              f"sparse={len(result.sparse_results)} "
              f"recency={len(result.recency_results)}")
        print(f"│  After fusion + pruning: {len(result.context)} docs")
        if result.groups:
            group_summary = ", ".join(
                f"{g.label} ({g.evidence_count})" for g in result.groups[:3]
            )
            print(f"│  Narrative groups: {group_summary}")
        print("│")
        print("│  Top results:")
        for i, sd in enumerate(result.context[:4], 1):
            ts = sd.doc.timestamp.strftime("%b %d")
            src = sd.doc.source.upper()[:4]
            score = sd.final_score
            snippet = sd.doc.content[:90].replace("\n", " ")
            print(f"│    {i}. [{src}] {ts}  score={score:.4f}")
            print(f"│       {snippet}...")
        print("└" + "─" * 60)


def run_evals(pipeline: TemporalRAGPipeline) -> None:
    cases = build_eval_cases(NOW)
    results = evaluate(pipeline, cases, now=NOW)
    print_report(results)


def run_ablation_suite(corpus) -> None:
    cases = build_eval_cases(NOW)
    rows = run_ablation(corpus, cases, now=NOW)
    print_ablation_report(rows)


def main() -> None:
    args = sys.argv[1:]
    do_demo = "--eval" not in args
    do_eval = "--demo" not in args
    do_ablation = "--demo" not in args and "--no-ablation" not in args

    print("Building corpus and indexing documents...")
    corpus = build_corpus(base_date=NOW)
    pipeline = TemporalRAGPipeline(token_budget=2000, temporal_weight=0.3, top_k=20)
    pipeline.index(corpus)
    print(f"Indexed {len(corpus)} documents spanning 90 days of context.")
    print(f"Embedding backend: {pipeline.dense.embedder.backend_name}")

    if do_demo:
        run_demo(pipeline)

    if do_eval:
        run_evals(pipeline)

    if do_ablation:
        run_ablation_suite(corpus)


if __name__ == "__main__":
    main()
