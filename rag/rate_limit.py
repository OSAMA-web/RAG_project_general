"""Simple in-memory rate limiter (Phase 7).

A sliding-window log: for each key (typically a user id) it keeps the timestamps
of recent requests and allows a new one only if fewer than MAX_REQUESTS_PER_WINDOW
happened in the last WINDOW_SECONDS — so no rolling 60-second span can ever hold
more than 20 requests. No new external dependency (no Redis) — correct for a
single-process deployment. A multi-process production deployment would need a
shared store instead, since this dict lives in one process's memory only.
"""

import os
import time
from collections import defaultdict
from typing import Dict, List

WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))
MAX_REQUESTS_PER_WINDOW = int(os.environ.get("RATE_LIMIT_MAX_REQUESTS", "20"))

_requests: Dict[str, List[float]] = defaultdict(list)


def is_allowed(key: str) -> bool:
    """Returns True and records this request if the key is under its quota."""
    now = time.time()
    window_start = now - WINDOW_SECONDS
    _requests[key] = [t for t in _requests[key] if t > window_start]

    if len(_requests[key]) >= MAX_REQUESTS_PER_WINDOW:
        return False

    _requests[key].append(now)
    return True


def reset(key: str) -> None:
    _requests.pop(key, None)
