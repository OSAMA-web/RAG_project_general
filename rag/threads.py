"""In-memory registry of chat threads (id, title, timestamps, owner).

Each thread's id doubles as the session_id used by rag/memory.py to store
that conversation's messages. Phase 7 adds a user_id field so threads can be
scoped to the account that created them — enforced at the API layer.

Same caveat as memory.py: a plain in-process dict, resets on server restart.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4

DEFAULT_TITLE = "New Chat"
AUTOTITLE_MAX_CHARS = 40

_threads: Dict[str, dict] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_thread(title: str = DEFAULT_TITLE, user_id: Optional[str] = None) -> dict:
    thread_id = str(uuid4())
    thread = {
        "id": thread_id,
        "title": title,
        "user_id": user_id,
        "created_at": _now(),
        "updated_at": _now(),
    }
    _threads[thread_id] = thread
    return thread


def ensure_thread(thread_id: str, user_id: Optional[str] = None) -> dict:
    """Returns the thread if it exists; otherwise registers one with this exact id.

    Lets /ask work even if a caller supplies a session_id that was never
    explicitly created via POST /threads (e.g. an older client).
    """
    if thread_id not in _threads:
        _threads[thread_id] = {
            "id": thread_id,
            "title": DEFAULT_TITLE,
            "user_id": user_id,
            "created_at": _now(),
            "updated_at": _now(),
        }
    return _threads[thread_id]


def get_thread(thread_id: str) -> Optional[dict]:
    return _threads.get(thread_id)


def list_threads(user_id: Optional[str] = None) -> List[dict]:
    """Most recently active thread first. Scoped to one user if user_id is given."""
    all_threads = sorted(_threads.values(), key=lambda t: t["updated_at"], reverse=True)
    if user_id is None:
        return all_threads
    return [t for t in all_threads if t.get("user_id") == user_id]


def rename_thread(thread_id: str, new_title: str) -> Optional[dict]:
    thread = _threads.get(thread_id)
    if not thread:
        return None
    thread["title"] = new_title.strip() or DEFAULT_TITLE
    thread["updated_at"] = _now()
    return thread


def delete_thread(thread_id: str) -> bool:
    return _threads.pop(thread_id, None) is not None


def touch_thread(thread_id: str) -> None:
    """Bumps updated_at so the thread rises to the top of the sidebar after use."""
    thread = _threads.get(thread_id)
    if thread:
        thread["updated_at"] = _now()


def maybe_autotitle(thread_id: str, first_question: str) -> None:
    """Sets the thread's title from its first question, but only if it's still the default.

    Mirrors how ChatGPT names new chats after your first message instead of
    leaving everything labeled "New Chat".
    """
    thread = _threads.get(thread_id)
    if not thread or thread["title"] != DEFAULT_TITLE:
        return
    title = first_question.strip()
    if len(title) > AUTOTITLE_MAX_CHARS:
        title = title[:AUTOTITLE_MAX_CHARS].rstrip() + "…"
    thread["title"] = title or DEFAULT_TITLE
