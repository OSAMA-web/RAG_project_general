"""Lightweight chat analytics (Phase 7) — logs each question to SQLite and
provides simple aggregate queries. From-scratch implementation (no external
analytics service) to keep the dependency footprint small and the mechanics
fully explainable.
"""

from datetime import datetime, timezone
from typing import Optional

from .db import get_connection


def log_event(
    user_id: Optional[str], intent: str, knowledge_base: Optional[str], response_time_ms: float
) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO analytics_events (user_id, intent, knowledge_base, response_time_ms, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, intent, knowledge_base, response_time_ms, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


def get_summary(user_id: Optional[str] = None) -> dict:
    """Aggregate stats, optionally scoped to one user."""
    with get_connection() as conn:
        query = "SELECT * FROM analytics_events"
        params: tuple = ()
        if user_id:
            query += " WHERE user_id = ?"
            params = (user_id,)
        rows = conn.execute(query, params).fetchall()

    total = len(rows)
    if total == 0:
        return {
            "total_questions": 0,
            "by_intent": {},
            "by_knowledge_base": {},
            "average_response_time_ms": 0,
        }

    by_intent: dict = {}
    by_kb: dict = {}
    total_time = 0.0
    for row in rows:
        by_intent[row["intent"]] = by_intent.get(row["intent"], 0) + 1
        kb = row["knowledge_base"] or "none"
        by_kb[kb] = by_kb.get(kb, 0) + 1
        total_time += row["response_time_ms"]

    return {
        "total_questions": total,
        "by_intent": by_intent,
        "by_knowledge_base": by_kb,
        "average_response_time_ms": round(total_time / total, 1),
    }
