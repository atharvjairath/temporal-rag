"""
data/synthetic_corpus.py
~~~~~~~~~~~~~~~~~~~~~~~~
Generates a realistic synthetic corpus of personal context documents
for demonstrating and evaluating the temporal RAG pipeline.

Simulates 90 days of a software engineer's digital footprint:
  - Meeting notes
  - Slack/chat messages
  - Browser/screen captures (page titles + snippets)
  - Code-related notes
  - Task descriptions
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Optional

from src.temporal_rag import Document


def _ago(days: float, base: Optional[datetime] = None, rng: Optional[random.Random] = None) -> datetime:
    base = base or datetime(2025, 5, 20, 12, 0)
    rng = rng or random.Random(42)
    return base - timedelta(days=days, hours=rng.uniform(-4, 4))


def build_corpus(base_date: Optional[datetime] = None) -> list[Document]:
    """Returns a deterministic synthetic corpus spanning 90 days of personal context."""
    base = base_date or datetime(2025, 5, 20, 12, 0)
    rng = random.Random(42)

    docs = []
    _id = 0

    def add(content: str, days_ago: float, source: str, tags: list[str] = None):
        nonlocal _id
        docs.append(Document(
            id=f"doc_{_id:03d}",
            content=content,
            timestamp=_ago(days_ago, base, rng),
            source=source,
            tags=tags or [],
        ))
        _id += 1

    # --- LAST WEEK (0-7 days ago) ---
    add("Sprint planning meeting — agreed to prioritize the payments API refactor. "
        "Sarah flagged latency issues in production. We decided to tackle the "
        "database connection pooling first. Target: ship by end of sprint.",
        2.5, "meeting", ["sprint", "payments", "sarah"])

    add("Reviewed PR #847 — refactored the payment gateway timeout handling. "
        "Left comments about error propagation. John pushed a fix. Merged.",
        3.1, "screen", ["pr", "payments", "john"])

    add("Slack: Sarah — 'the payments dashboard latency is down to 180ms, "
        "great improvement!' Response: 'nice, pooling change did the trick'",
        3.4, "message", ["sarah", "payments", "latency"])

    add("Fixed bug in auth token refresh — tokens were expiring silently "
        "on mobile clients. Root cause: clock skew between services. "
        "Added 30s buffer to expiry check.",
        4.0, "screen", ["auth", "bug", "mobile"])

    add("1:1 with manager — discussed promotion timeline. Mentioned payments "
        "work as key impact. Suggested I also take on the search infrastructure "
        "project. Agreed to revisit in 6 weeks.",
        4.5, "meeting", ["career", "manager", "search"])

    add("Read: 'Optimizing PostgreSQL connection pools at scale' — notes: "
        "PgBouncer transaction mode, max_client_conn=200, pool_size=10 per DB user.",
        5.0, "screen", ["database", "postgresql", "research"])

    add("Deployed payments API v2.1 to staging. Ran load tests — p99 latency 210ms "
        "under 500 RPS. Will promote to production Thursday after sign-off from Sarah.",
        5.5, "doc", ["payments", "deploy", "latency"])

    add("Wrote design doc for new search ranking pipeline — covers hybrid BM25 + "
        "dense retrieval, re-ranking with cross-encoders, eval harness using NDCG.",
        6.0, "doc", ["search", "design", "ranking"])

    add("Team standup — mentioned auth bug fix. Tom asked about the mobile release "
        "date. Confirmed Thursday. Sarah confirmed payments sign-off on track.",
        6.5, "meeting", ["standup", "mobile", "sarah", "tom"])

    add("Reviewed Stripe webhook documentation — researching idempotency keys "
        "for retry safety. Added notes to the payments design doc.",
        7.0, "screen", ["payments", "stripe", "research"])

    add("Debug note: OAuth callback failed in staging. Test login used email "
        "emma@example.com and token=lb_test_secret_12345. Verified the privacy "
        "filter redacts this before indexing.",
        7.2, "doc", ["security", "privacy", "oauth"])

    add("Browser tab: watched a movie review ranking the best search scenes in sci-fi. "
        "Unrelated personal reading, not connected to the search infrastructure project.",
        1.2, "screen", ["personal", "movie", "search"])

    add("Slack: Sarah — lunch was fun, let's try the new Thai place next time. "
        "No work decisions or project updates in this thread.",
        1.4, "message", ["sarah", "personal"])

    add("Calendar hold: Incident response tabletop rehearsal. Simulated checkout outage "
        "for training only; no production issue occurred.",
        1.8, "meeting", ["incident", "training", "calendar"])

    add("Notes from a payments design article: the author argued that split payments "
        "are mostly a UX problem. Bookmarked for later; unrelated to current sprint.",
        2.1, "screen", ["payments", "reading", "ux"])

    # --- TWO WEEKS AGO (7-14 days) ---
    add("Sprint retrospective — team agreed the auth service refactor went well. "
        "Identified deployment process as a bottleneck. Action item: automate "
        "staging promotion with GitHub Actions.",
        9.0, "meeting", ["retro", "auth", "deployment"])

    add("Pair programming session with John on the search indexing pipeline. "
        "Explored using HNSW index in pgvector. Decided to prototype this week.",
        9.5, "meeting", ["search", "john", "vector", "pgvector"])

    add("Slack: John — 'pgvector prototype is looking good, query time under 20ms "
        "for 1M vectors'. Added note: need to test recall at that latency.",
        10.0, "message", ["search", "john", "pgvector", "latency"])

    add("Read 'Lessons from running RAG in production' blog post — key takeaways: "
        "chunk size matters more than model choice, metadata filtering saves tokens, "
        "eval harness is non-negotiable.",
        10.5, "screen", ["rag", "research", "retrieval"])

    add("Wrote unit tests for payment retry logic — edge cases: duplicate webhooks, "
        "partial captures, currency mismatch. All green. Coverage now at 91%.",
        11.0, "screen", ["payments", "testing", "coverage"])

    add("Architecture review with CTO — proposed microservice split for the "
        "notification system. CTO concerned about operational overhead. "
        "Agreed to keep monolith, add async queue instead.",
        11.5, "meeting", ["architecture", "cto", "notifications"])

    add("Investigated memory leak in the agent service — traced to unbounded "
        "in-memory cache for tool call results. Fixed with LRU eviction (max 1000 entries).",
        12.0, "screen", ["agent", "bug", "memory", "cache"])

    add("Deployed auth service v1.8 to production. Zero-downtime rolling deploy. "
        "Monitoring for 24h — no incidents. Closed 3 related Jira tickets.",
        13.0, "doc", ["auth", "deploy", "production"])

    add("Copied meeting transcript fragment: payments latency payments latency payments "
        "latency. Duplicate capture from a noisy recorder, no new decision beyond the "
        "existing pooling work.",
        8.0, "doc", ["payments", "duplicate", "noise"])

    # --- LAST MONTH (14-30 days) ---
    add("Q2 planning session — committed to: (1) search v1 launch, "
        "(2) payments v3 with split payments, (3) mobile performance audit. "
        "Timeline: all by end of June.",
        16.0, "meeting", ["planning", "q2", "search", "payments", "mobile"])

    add("Onboarding session for new hire Emma — walked through codebase, "
        "deployment process, and oncall runbook. Emma focused on frontend.",
        17.0, "meeting", ["onboarding", "emma", "frontend"])

    add("Wrote RFC for context pruning in the AI pipeline — proposed using "
        "LLMLingua for token compression, targeting 60% reduction with <2% recall loss.",
        18.0, "doc", ["ai", "context", "pruning", "llm", "rfc"])

    add("Resolved production incident — search service returned empty results "
        "for 12 minutes due to index rebuild lock. Postmortem: add read replica "
        "for queries during index rebuild.",
        20.0, "doc", ["incident", "search", "postmortem", "production"])

    add("Attended company all-hands — CEO announced Series A target for Q4. "
        "Product roadmap revealed: enterprise tier, SSO, audit logs.",
        22.0, "meeting", ["all-hands", "company", "roadmap"])

    add("Performance review self-assessment submitted — highlighted payments "
        "refactor (40% latency reduction), auth service (zero downtime), "
        "and search design doc.",
        24.0, "doc", ["review", "performance", "career"])

    add("Research: surveyed embedding models for personal context retrieval — "
        "compared OpenAI text-embedding-3-small vs BGE-M3 vs E5-large. "
        "BGE-M3 best recall on personal text at 3x lower cost.",
        26.0, "screen", ["research", "embeddings", "bge", "retrieval"])

    add("Kickoff meeting for mobile performance audit — identified 3 main issues: "
        "bundle size, image loading, excessive re-renders. Sprint plan created.",
        28.0, "meeting", ["mobile", "performance", "audit"])

    add("Slack: John asked whether vector recall should be measured before latency "
        "optimization. We agreed recall@10 must stay above 0.92 before shipping HNSW.",
        29.0, "message", ["john", "search", "vector", "recall"])

    add("Meeting notes: Emma mentioned an architecture blog post about notification "
        "systems. It was interesting, but no product architecture decision was made.",
        29.4, "meeting", ["architecture", "emma", "reading"])

    # --- OLDER (30-90 days) ---
    add("Completed HackerNews reading session — articles on LLM context windows, "
        "RLHF updates, and Postgres 17 features. Bookmarked 4 links.",
        35.0, "screen", ["reading", "llm", "postgres"])

    add("Q1 retrospective — payments team delivered 2/3 milestones. Auth "
        "service migration was delayed by 2 weeks due to dependencies. "
        "Lessons: flag cross-team blockers earlier.",
        40.0, "meeting", ["q1", "retro", "payments", "auth"])

    add("Built proof of concept for AI meeting summarisation — used Whisper "
        "for transcription, GPT-4 for structured summaries. Demo'd to team.",
        45.0, "doc", ["ai", "meetings", "poc", "whisper"])

    add("1:1 with Sarah — discussed career growth. She encouraged taking on "
        "technical leadership for the search project. Agreed to be DRI.",
        50.0, "meeting", ["career", "sarah", "search", "leadership"])

    add("Read 'The Bitter Lesson' and surrounding ML scaling literature. "
        "Notes: compute-efficient approaches outperform human priors long-term.",
        55.0, "screen", ["research", "ml", "scaling", "reading"])

    add("Completed AWS certification study — passed Solutions Architect Associate. "
        "Key study areas: VPC design, IAM policies, RDS failover.",
        60.0, "doc", ["aws", "certification", "career"])

    add("Started payments v3 design — split payments requires idempotency at "
        "every step. Researched Stripe's approach. Draft doc shared with team.",
        65.0, "doc", ["payments", "design", "stripe"])

    add("Old Slack: Sarah discussed the payments dashboard redesign. This was before "
        "the latency work and should not answer recent payments queries.",
        66.0, "message", ["sarah", "payments", "old"])

    add("Incident: database failover during peak traffic caused 8 min outage. "
        "Root cause: replica lag exceeded threshold. Fix: increase sync_commit.",
        70.0, "doc", ["incident", "database", "outage", "postmortem"])

    add("Team offsite — 2-day strategy session. Agreed on Q3 priorities: "
        "enterprise features, improved search, mobile parity with web.",
        80.0, "meeting", ["offsite", "strategy", "q3"])

    add("First week at company — onboarding, codebase walkthrough, met the team. "
        "Initial impression: strong engineering culture, good test coverage.",
        90.0, "doc", ["onboarding", "first-week"])

    rng.shuffle(docs)
    return docs
