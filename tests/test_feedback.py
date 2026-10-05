import pytest

from rag import db, feedback


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test_feedback.db")
    db.init_db()
    yield


def test_add_feedback_and_get_summary():
    feedback.add_feedback("user-1", "thread-1", "Q1", "A1", "up")
    feedback.add_feedback("user-1", "thread-1", "Q2", "A2", "up")
    feedback.add_feedback("user-1", "thread-2", "Q3", "A3", "down")

    summary = feedback.get_summary()

    assert summary["up"] == 2
    assert summary["down"] == 1


def test_get_summary_empty_returns_empty_dict():
    assert feedback.get_summary() == {}
