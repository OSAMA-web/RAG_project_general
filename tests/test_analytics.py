import pytest

from rag import analytics, db


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test_analytics.db")
    db.init_db()
    yield


def test_log_event_and_get_summary_counts_correctly():
    analytics.log_event("user-1", "rag", "Air India", 120.5)
    analytics.log_event("user-1", "flight", None, 45.0)
    analytics.log_event("user-1", "rag", "Air India", 80.0)

    summary = analytics.get_summary(user_id="user-1")

    assert summary["total_questions"] == 3
    assert summary["by_intent"]["rag"] == 2
    assert summary["by_intent"]["flight"] == 1
    assert summary["by_knowledge_base"]["Air India"] == 2
    assert summary["by_knowledge_base"]["none"] == 1
    assert summary["average_response_time_ms"] == round((120.5 + 45.0 + 80.0) / 3, 1)


def test_get_summary_empty_returns_zeroed_stats():
    summary = analytics.get_summary(user_id="nobody-yet")
    assert summary["total_questions"] == 0
    assert summary["average_response_time_ms"] == 0


def test_get_summary_scoped_per_user():
    analytics.log_event("user-a", "rag", None, 100.0)
    analytics.log_event("user-b", "flight", None, 200.0)

    summary_a = analytics.get_summary(user_id="user-a")
    assert summary_a["total_questions"] == 1
    assert summary_a["by_intent"] == {"rag": 1}
