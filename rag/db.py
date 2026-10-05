"""Shared SQLite database for data that needs to survive restarts: users,
analytics events, and feedback (Phase 7).

Chat threads/memory intentionally stay in-process (see rag/threads.py and
rag/memory.py) — this file is specifically for the professional-features data
that genuinely needs to persist across restarts.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path("app_data.db")


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analytics_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                intent TEXT NOT NULL,
                knowledge_base TEXT,
                response_time_ms REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                thread_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                rating TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


init_db()
