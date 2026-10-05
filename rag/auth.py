"""Authentication (Phase 7): password hashing, JWT issuance/verification, user storage.

No third-party auth service — this is a real, self-contained implementation
using bcrypt for password hashing and PyJWT for tokens. JWT_SECRET_KEY should
be set via the environment in any deployment beyond local personal use (see
.env.example) — the fallback here is only for convenience during local dev.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

import bcrypt
import jwt

from .db import get_connection

JWT_SECRET = os.environ.get("JWT_SECRET_KEY", "dev-secret-change-me-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = int(os.environ.get("JWT_EXPIRY_HOURS", "24"))


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_user(username: str, password: str) -> Optional[dict]:
    """Creates a new user. Returns the user dict, or None if the username is already taken."""
    with get_connection() as conn:
        existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            return None

        user_id = str(uuid4())
        password_hash = hash_password(password)
        created_at = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO users (id, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (user_id, username, password_hash, created_at),
        )
        conn.commit()
        return {"id": user_id, "username": username, "created_at": created_at}


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Returns the user dict if the credentials are valid, else None."""
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if not row:
            return None
        if not verify_password(password, row["password_hash"]):
            return None
        return {"id": row["id"], "username": row["username"], "created_at": row["created_at"]}


def create_access_token(user_id: str, username: str) -> str:
    expiry = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS)
    payload = {"sub": user_id, "username": username, "exp": expiry}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    """Returns the decoded payload if the token is valid, else None (expired/invalid/tampered)."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None
