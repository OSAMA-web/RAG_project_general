"""In-memory conversation history, keyed by session id.

Stores the COMPLETE conversation for each session — nothing is ever
discarded automatically. When building a prompt for the LLM, only a
recent window of turns is used (PROMPT_WINDOW_TURNS) to keep the prompt
short and fast; the full transcript stays available for other uses
(displaying full scrollback, exporting a chat, searching past chats, etc.).

Intentionally simple — a plain Python dict living in the server process.
Fine for a single-user local demo; it resets when the server restarts.
For persistence across restarts, swap this for a small database table.
"""

from collections import defaultdict
from typing import Dict, List, Tuple
from uuid import uuid4

PROMPT_WINDOW_TURNS = 5  # how many of the most recent exchanges to feed into the LLM prompt

_history: Dict[str, List[Tuple[str, str]]] = defaultdict(list)


def new_session_id() -> str:
    return str(uuid4())


def get_history(session_id: str) -> List[Tuple[str, str]]:
    """Returns the FULL conversation for a session — unlimited length, nothing trimmed."""
    return _history.get(session_id, [])


def get_recent_history(session_id: str, turns: int = PROMPT_WINDOW_TURNS) -> List[Tuple[str, str]]:
    """Returns only the most recent `turns` exchanges — used for the LLM prompt window."""
    return get_history(session_id)[-turns:]


def add_turn(session_id: str, question: str, answer: str) -> None:
    """Appends a turn. Nothing is ever dropped — the full conversation is retained."""
    _history[session_id].append((question, answer))


def clear_session(session_id: str) -> None:
    _history.pop(session_id, None)


def format_history(session_id: str, turns: int = PROMPT_WINDOW_TURNS) -> str:
    """Formats the most recent `turns` exchanges as plain text for a prompt.

    Deliberately windowed (not the full history) so the prompt sent to the LLM
    stays a manageable size no matter how long the conversation has run.
    """
    recent = get_recent_history(session_id, turns)
    if not recent:
        return ""
    lines: List[str] = []
    for q, a in recent:
        lines.append(f"User: {q}")
        lines.append(f"Assistant: {a}")
    return "\n".join(lines)
