"""Stores user feedback (thumbs up/down) on individual answers, in SQLite (Phase 7)."""

from datetime import datetime, timezone
from typing import Optional

from .db import get_connection


def add_feedback(
    user_id: Optional[str], thread_id: str, question: str, answer: str, rating: str
) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO feedback (user_id, thread_id, question, answer, rating, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, thread_id, question, answer, rating, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


def get_summary() -> dict:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT rating, COUNT(*) as count FROM feedback GROUP BY rating"
        ).fetchall()
    return {row["rating"]: row["count"] for row in rows}
