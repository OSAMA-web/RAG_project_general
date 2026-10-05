import pytest

from rag import auth, db


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path, monkeypatch):
    """Redirects the SQLite file to a throwaway path so tests never touch the
    real project database and are fully isolated from each other."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test_app_data.db")
    db.init_db()
    yield


def test_create_user_succeeds_and_returns_user_dict():
    user = auth.create_user("alice", "supersecret")
    assert user is not None
    assert user["username"] == "alice"
    assert "id" in user


def test_create_user_rejects_duplicate_username():
    auth.create_user("alice", "supersecret")
    duplicate = auth.create_user("alice", "anotherpassword")
    assert duplicate is None


def test_authenticate_user_with_correct_password():
    auth.create_user("bob", "correct-password")
    user = auth.authenticate_user("bob", "correct-password")
    assert user is not None
    assert user["username"] == "bob"


def test_authenticate_user_with_wrong_password_fails():
    auth.create_user("bob", "correct-password")
    assert auth.authenticate_user("bob", "wrong-password") is None


def test_authenticate_unknown_user_fails():
    assert auth.authenticate_user("nobody", "whatever") is None


def test_password_hash_is_not_stored_as_plaintext():
    hashed = auth.hash_password("mypassword")
    assert hashed != "mypassword"
    assert auth.verify_password("mypassword", hashed) is True
    assert auth.verify_password("wrongpassword", hashed) is False


def test_create_and_decode_access_token_roundtrip():
    token = auth.create_access_token("user-123", "alice")
    payload = auth.decode_access_token(token)
    assert payload["sub"] == "user-123"
    assert payload["username"] == "alice"


def test_decode_invalid_token_returns_none():
    assert auth.decode_access_token("not-a-real-token") is None
