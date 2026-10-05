import pytest

from rag import threads


@pytest.fixture(autouse=True)
def _reset_registry():
    """Threads live in a plain module-level dict — clear it before every test
    so tests can't see each other's threads."""
    threads._threads.clear()
    yield


def test_create_thread_has_default_title_and_timestamps():
    thread = threads.create_thread()
    assert thread["title"] == threads.DEFAULT_TITLE
    assert thread["id"]
    assert thread["created_at"]
    assert thread["updated_at"]


def test_list_threads_sorted_most_recently_updated_first():
    a = threads.create_thread()
    b = threads.create_thread()
    threads.touch_thread(a["id"])  # bump a's updated_at after b was created

    ordered = threads.list_threads()
    assert ordered[0]["id"] == a["id"]
    assert ordered[1]["id"] == b["id"]


def test_ensure_thread_creates_if_missing_and_is_idempotent():
    thread_id = "custom-id-123"
    first = threads.ensure_thread(thread_id)
    second = threads.ensure_thread(thread_id)
    assert first["id"] == thread_id
    assert first["created_at"] == second["created_at"]  # wasn't recreated


def test_rename_thread_updates_title_and_none_if_missing():
    thread = threads.create_thread()
    renamed = threads.rename_thread(thread["id"], "Fleet questions")
    assert renamed["title"] == "Fleet questions"
    assert threads.rename_thread("does-not-exist", "x") is None


def test_delete_thread_removes_it():
    thread = threads.create_thread()
    assert threads.delete_thread(thread["id"]) is True
    assert threads.get_thread(thread["id"]) is None
    assert threads.delete_thread(thread["id"]) is False  # already gone


def test_maybe_autotitle_only_applies_once():
    thread = threads.create_thread()
    threads.maybe_autotitle(thread["id"], "How many aircraft does Air India have?")
    assert threads.get_thread(thread["id"])["title"] == "How many aircraft does Air India have?"

    # A later call must not override a title that's already been set
    threads.maybe_autotitle(thread["id"], "A completely different question")
    assert threads.get_thread(thread["id"])["title"] == "How many aircraft does Air India have?"


def test_maybe_autotitle_truncates_long_questions():
    thread = threads.create_thread()
    long_question = "A" * 80
    threads.maybe_autotitle(thread["id"], long_question)
    title = threads.get_thread(thread["id"])["title"]
    assert len(title) <= threads.AUTOTITLE_MAX_CHARS + 1  # +1 for the trailing ellipsis
    assert title.endswith("…")


# ---------- Phase 7: user ownership ----------


def test_create_thread_stores_user_id():
    thread = threads.create_thread(user_id="user-123")
    assert thread["user_id"] == "user-123"


def test_list_threads_filters_by_user_id():
    a = threads.create_thread(user_id="user-1")
    threads.create_thread(user_id="user-2")

    only_user_1 = threads.list_threads(user_id="user-1")
    assert len(only_user_1) == 1
    assert only_user_1[0]["id"] == a["id"]


def test_list_threads_returns_everything_when_no_user_filter():
    threads.create_thread(user_id="user-1")
    threads.create_thread(user_id="user-2")
    assert len(threads.list_threads()) == 2


def test_ensure_thread_stores_user_id_for_a_newly_registered_thread():
    thread = threads.ensure_thread("custom-id", user_id="user-1")
    assert thread["user_id"] == "user-1"
