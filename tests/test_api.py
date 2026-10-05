import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import api
import rag.config as rag_config
import rag.db as rag_db
from rag import memory as rag_memory
from rag import rate_limit as rag_rate_limit
from rag import threads as rag_threads

FAKE_USER = {"id": "test-user-id", "username": "testuser"}


def _fake_pipeline(answer: str):
    """Stands in for rag.pipeline.get_response_with_sources, INCLUDING its side effect
    of recording the turn in memory — tests that read history back (search, export)
    need that, and a bare return_value mock silently skips it."""

    def _fake(question, session_id=None, knowledge_base=None):
        if session_id:
            rag_memory.add_turn(session_id, question, answer)
        return answer, []

    return _fake


def _parse_sse(text: str):
    """Turns a raw SSE response body back into a list of event dicts, for assertions."""
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block.startswith("data:"):
            continue
        payload = block[len("data:") :].strip()
        if payload:
            events.append(json.loads(payload))
    return events


@pytest.fixture(autouse=True)
def _reset_state(tmp_path, monkeypatch):
    """Threads, message history, and the rate limiter are plain in-process
    structures — clear them before every test. The SQLite DB (users/analytics/
    feedback) is redirected to a fresh temp file per test for full isolation.

    Auth is bypassed via FastAPI's dependency_overrides so most existing tests
    don't need a real token — a handful of dedicated tests below explicitly
    clear the override to exercise the real login/register/401 behavior.
    """
    rag_memory._history.clear()
    rag_threads._threads.clear()
    rag_rate_limit._requests.clear()
    monkeypatch.setattr(rag_db, "DB_PATH", tmp_path / "test_app_data.db")
    rag_db.init_db()

    api.app.dependency_overrides[api.get_current_user] = lambda: FAKE_USER
    yield
    api.app.dependency_overrides.clear()


@patch("api.build_index_if_needed", return_value=0)
def test_health_endpoint(mock_build):
    with TestClient(api.app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


@patch(
    "api.get_response_with_sources",
    return_value=(
        "Air India was founded by JRD Tata in 1932.",
        ["Air India's first flight was on 15 October 1932."],
    ),
)
@patch("api.build_index_if_needed", return_value=0)
def test_ask_endpoint_returns_answer_sources_and_session_id(mock_build, mock_get_response):
    with TestClient(api.app) as client:
        response = client.post("/ask", json={"question": "Who founded Air India?"})
        assert response.status_code == 200
        body = response.json()
        assert body["question"] == "Who founded Air India?"
        assert "JRD Tata" in body["answer"]
        assert len(body["sources"]) == 1
        assert isinstance(body["session_id"], str) and len(body["session_id"]) > 0


@patch(
    "api.get_response_with_sources",
    return_value=("Sure, following up on that — yes.", []),
)
@patch("api.build_index_if_needed", return_value=0)
def test_ask_endpoint_reuses_provided_session_id(mock_build, mock_get_response):
    with TestClient(api.app) as client:
        response = client.post(
            "/ask", json={"question": "And the follow-up?", "session_id": "abc-123"}
        )
        assert response.status_code == 200
        assert response.json()["session_id"] == "abc-123"
        mock_get_response.assert_called_once_with(
            "And the follow-up?", "abc-123", knowledge_base=None
        )


@patch("api.build_index_if_needed", return_value=0)
def test_reset_endpoint_clears_session(mock_build):
    with TestClient(api.app) as client:
        # /reset only works on a thread the caller owns (Phase 7), so create one first
        thread_id = client.post("/threads").json()["id"]
        rag_memory.add_turn(thread_id, "Q?", "A.")

        response = client.post("/reset", json={"session_id": thread_id})
        assert response.status_code == 200
        assert response.json() == {"status": "cleared"}
        assert rag_memory.get_history(thread_id) == []


@patch("api.build_index_if_needed", return_value=0)
def test_reset_endpoint_404s_for_a_thread_the_user_does_not_own(mock_build):
    with TestClient(api.app) as client:
        response = client.post("/reset", json={"session_id": "abc-123"})
        assert response.status_code == 404


@patch("api.build_index_if_needed", return_value=0)
def test_ask_endpoint_rejects_empty_question(mock_build):
    with TestClient(api.app) as client:
        response = client.post("/ask", json={"question": "   "})
        assert response.status_code == 400


@patch("api.build_index_if_needed", return_value=0)
def test_create_and_list_threads(mock_build):
    with TestClient(api.app) as client:
        created = client.post("/threads")
        assert created.status_code == 200
        thread = created.json()
        assert thread["title"] == "New Chat"

        listed = client.get("/threads")
        assert listed.status_code == 200
        ids = [t["id"] for t in listed.json()]
        assert thread["id"] in ids


@patch("api.build_index_if_needed", return_value=0)
def test_rename_and_delete_thread(mock_build):
    with TestClient(api.app) as client:
        thread = client.post("/threads").json()

        renamed = client.patch(f"/threads/{thread['id']}", json={"title": "Fleet Qs"})
        assert renamed.status_code == 200
        assert renamed.json()["title"] == "Fleet Qs"

        deleted = client.delete(f"/threads/{thread['id']}")
        assert deleted.status_code == 200
        assert deleted.json() == {"status": "deleted"}

        missing = client.patch(f"/threads/{thread['id']}", json={"title": "x"})
        assert missing.status_code == 404


@patch(
    "api.get_response_with_sources",
    return_value=("JRD Tata founded it.", ["source snippet"]),
)
@patch("api.build_index_if_needed", return_value=0)
def test_asking_auto_creates_and_autotitles_thread(mock_build, mock_get_response):
    with TestClient(api.app) as client:
        response = client.post("/ask", json={"question": "Who founded Air India?"})
        assert response.status_code == 200
        session_id = response.json()["session_id"]

        all_threads = client.get("/threads").json()
        matching = next(t for t in all_threads if t["id"] == session_id)
        assert matching["title"] == "Who founded Air India?"


@patch("api.build_index_if_needed", return_value=0)
def test_thread_messages_endpoint_returns_404_for_unknown_thread(mock_build):
    with TestClient(api.app) as client:
        response = client.get("/threads/does-not-exist/messages")
        assert response.status_code == 404


@patch("api.stream_response_with_sources")
@patch("api.build_index_if_needed", return_value=0)
def test_ask_stream_emits_session_sources_tokens_then_done(mock_build, mock_stream):
    mock_stream.return_value = iter(
        [
            {"type": "sources", "sources": ["snippet one"]},
            {"type": "token", "text": "Hello"},
            {"type": "token", "text": " world"},
        ]
    )

    with TestClient(api.app) as client:
        response = client.post("/ask/stream", json={"question": "Hi"})
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        events = _parse_sse(response.text)
        types = [e["type"] for e in events]

        assert types[0] == "session"
        assert {"type": "sources", "sources": ["snippet one"]} in events
        assert {"type": "token", "text": "Hello"} in events
        assert {"type": "token", "text": " world"} in events
        assert types[-1] == "done"


@patch("api.build_index_if_needed", return_value=0)
def test_ask_stream_rejects_empty_question(mock_build):
    with TestClient(api.app) as client:
        response = client.post("/ask/stream", json={"question": "   "})
        assert response.status_code == 400


@patch("api.stream_response_with_sources")
@patch("api.build_index_if_needed", return_value=0)
def test_ask_stream_reuses_provided_session_id(mock_build, mock_stream):
    mock_stream.return_value = iter([{"type": "token", "text": "ok"}])

    with TestClient(api.app) as client:
        response = client.post(
            "/ask/stream", json={"question": "Follow-up?", "session_id": "abc-123"}
        )
        events = _parse_sse(response.text)
        session_event = next(e for e in events if e["type"] == "session")
        assert session_event["session_id"] == "abc-123"


@patch("api.add_uploaded_document", return_value=7)
@patch("api.build_index_if_needed", return_value=0)
def test_upload_endpoint_accepts_pdf_and_returns_chunk_count(mock_build, mock_add, tmp_path):
    with patch("api.UPLOAD_DIR", tmp_path):
        with TestClient(api.app) as client:
            response = client.post(
                "/upload",
                files={"file": ("policy.pdf", b"%PDF-1.4 fake content", "application/pdf")},
            )
            assert response.status_code == 200
            body = response.json()
            assert body["filename"] == "policy.pdf"
            assert body["chunks_added"] == 7


@patch("api.build_index_if_needed", return_value=0)
def test_upload_endpoint_rejects_unsupported_extension(mock_build, tmp_path):
    with patch("api.UPLOAD_DIR", tmp_path):
        with TestClient(api.app) as client:
            response = client.post(
                "/upload",
                files={"file": ("virus.exe", b"binary", "application/octet-stream")},
            )
            assert response.status_code == 400


@patch("api.add_uploaded_document", side_effect=RuntimeError("corrupt file"))
@patch("api.build_index_if_needed", return_value=0)
def test_upload_endpoint_returns_500_when_processing_fails(mock_build, mock_add, tmp_path):
    with patch("api.UPLOAD_DIR", tmp_path):
        with TestClient(api.app) as client:
            response = client.post(
                "/upload",
                files={"file": ("policy.pdf", b"%PDF-1.4 fake content", "application/pdf")},
            )
            assert response.status_code == 500


@patch("api.list_knowledge_bases", return_value=["Air India", "HR Policy"])
@patch("api.build_index_if_needed", return_value=0)
def test_knowledge_bases_endpoint_returns_list(mock_build, mock_list):
    with TestClient(api.app) as client:
        response = client.get("/knowledge-bases")
        assert response.status_code == 200
        assert response.json() == ["Air India", "HR Policy"]


@patch(
    "api.get_response_with_sources",
    return_value=("Fleet size is 138 aircraft.", ["snippet"]),
)
@patch("api.build_index_if_needed", return_value=0)
def test_ask_endpoint_passes_knowledge_base_filter(mock_build, mock_get_response):
    with TestClient(api.app) as client:
        client.post(
            "/ask",
            json={"question": "Fleet size?", "session_id": "abc-123", "knowledge_base": "Air India"},
        )
        mock_get_response.assert_called_once_with(
            "Fleet size?", "abc-123", knowledge_base="Air India"
        )


@patch("api.add_uploaded_document", return_value=5)
@patch("api.build_index_if_needed", return_value=0)
def test_upload_endpoint_passes_custom_knowledge_base(mock_build, mock_add, tmp_path):
    with patch("api.UPLOAD_DIR", tmp_path):
        with TestClient(api.app) as client:
            response = client.post(
                "/upload",
                files={"file": ("policy.pdf", b"%PDF-1.4 fake content", "application/pdf")},
                data={"knowledge_base": "HR Policy"},
            )
            assert response.status_code == 200
            _, kwargs = mock_add.call_args
            assert kwargs["knowledge_base"] == "HR Policy"


@patch("api.add_uploaded_document", return_value=3)
@patch("api.build_index_if_needed", return_value=0)
def test_upload_endpoint_defaults_knowledge_base(mock_build, mock_add, tmp_path):
    with patch("api.UPLOAD_DIR", tmp_path):
        with TestClient(api.app) as client:
            response = client.post(
                "/upload",
                files={"file": ("notes.txt", b"plain text", "text/plain")},
            )
            assert response.status_code == 200
            _, kwargs = mock_add.call_args
            assert kwargs["knowledge_base"] == rag_config.DEFAULT_UPLOAD_KNOWLEDGE_BASE


@patch("api.build_index_if_needed", return_value=0)
def test_search_threads_matches_by_title(mock_build):
    with TestClient(api.app) as client:
        created = client.post("/threads").json()
        client.patch(f"/threads/{created['id']}", json={"title": "Fleet size questions"})

        results = client.get("/threads/search", params={"q": "fleet"}).json()
        ids = [t["id"] for t in results]
        assert created["id"] in ids


@patch("api.get_response_with_sources", side_effect=_fake_pipeline("Air India has 138 aircraft."))
@patch("api.build_index_if_needed", return_value=0)
def test_search_threads_matches_by_message_content(mock_build, mock_get_response):
    with TestClient(api.app) as client:
        ask_response = client.post("/ask", json={"question": "How many aircraft?"})
        session_id = ask_response.json()["session_id"]

        results = client.get("/threads/search", params={"q": "138 aircraft"}).json()
        ids = [t["id"] for t in results]
        assert session_id in ids


@patch("api.build_index_if_needed", return_value=0)
def test_search_threads_empty_query_returns_all(mock_build):
    with TestClient(api.app) as client:
        client.post("/threads")
        client.post("/threads")
        results = client.get("/threads/search", params={"q": ""}).json()
        assert len(results) == 2


@patch("api.build_index_if_needed", return_value=0)
def test_export_endpoint_returns_404_for_unknown_thread(mock_build):
    with TestClient(api.app) as client:
        response = client.get("/threads/does-not-exist/export/txt")
        assert response.status_code == 404


@patch("api.build_index_if_needed", return_value=0)
def test_export_endpoint_rejects_unsupported_format(mock_build):
    with TestClient(api.app) as client:
        thread = client.post("/threads").json()
        response = client.get(f"/threads/{thread['id']}/export/exe")
        assert response.status_code == 400


@patch("api.get_response_with_sources", side_effect=_fake_pipeline("Air India was founded by JRD Tata."))
@patch("api.build_index_if_needed", return_value=0)
def test_export_txt_contains_transcript(mock_build, mock_get_response):
    with TestClient(api.app) as client:
        ask_response = client.post("/ask", json={"question": "Who founded Air India?"})
        session_id = ask_response.json()["session_id"]

        response = client.get(f"/threads/{session_id}/export/txt")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "Who founded Air India?" in response.text
        assert "JRD Tata" in response.text


@patch(
    "api.get_response_with_sources",
    return_value=("Air India was founded by JRD Tata.", []),
)
@patch("api.build_index_if_needed", return_value=0)
def test_export_pdf_returns_pdf_bytes(mock_build, mock_get_response):
    with TestClient(api.app) as client:
        ask_response = client.post("/ask", json={"question": "Who founded Air India?"})
        session_id = ask_response.json()["session_id"]

        response = client.get(f"/threads/{session_id}/export/pdf")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert response.content.startswith(b"%PDF")


@patch(
    "api.get_response_with_sources",
    return_value=("Air India was founded by JRD Tata.", []),
)
@patch("api.build_index_if_needed", return_value=0)
def test_export_docx_returns_valid_zip_container(mock_build, mock_get_response):
    with TestClient(api.app) as client:
        ask_response = client.post("/ask", json={"question": "Who founded Air India?"})
        session_id = ask_response.json()["session_id"]

        response = client.get(f"/threads/{session_id}/export/docx")
        assert response.status_code == 200
        # .docx files are zip containers — real files start with the zip magic bytes
        assert response.content[:2] == b"PK"


# ---------- Phase 7: real auth flow (override cleared for these specific tests) ----------


def test_register_and_login_flow_with_real_tokens():
    api.app.dependency_overrides.pop(api.get_current_user, None)
    try:
        with TestClient(api.app) as client:
            register_response = client.post(
                "/auth/register", json={"username": "newuser123", "password": "supersecret"}
            )
            assert register_response.status_code == 200
            assert register_response.json()["access_token"]

            login_response = client.post(
                "/auth/login", json={"username": "newuser123", "password": "supersecret"}
            )
            assert login_response.status_code == 200
            assert login_response.json()["username"] == "newuser123"
    finally:
        api.app.dependency_overrides[api.get_current_user] = lambda: FAKE_USER


def test_register_rejects_duplicate_username():
    api.app.dependency_overrides.pop(api.get_current_user, None)
    try:
        with TestClient(api.app) as client:
            client.post("/auth/register", json={"username": "dupeuser", "password": "password1"})
            second = client.post("/auth/register", json={"username": "dupeuser", "password": "password2"})
            assert second.status_code == 400
    finally:
        api.app.dependency_overrides[api.get_current_user] = lambda: FAKE_USER


def test_login_rejects_wrong_password():
    api.app.dependency_overrides.pop(api.get_current_user, None)
    try:
        with TestClient(api.app) as client:
            client.post("/auth/register", json={"username": "loginuser", "password": "correctpass"})
            response = client.post("/auth/login", json={"username": "loginuser", "password": "wrongpass"})
            assert response.status_code == 401
    finally:
        api.app.dependency_overrides[api.get_current_user] = lambda: FAKE_USER


def test_protected_endpoint_rejects_missing_token():
    api.app.dependency_overrides.pop(api.get_current_user, None)
    try:
        with TestClient(api.app) as client:
            response = client.get("/threads")
            assert response.status_code in (401, 403)  # FastAPI's HTTPBearer uses 403 for a missing header
    finally:
        api.app.dependency_overrides[api.get_current_user] = lambda: FAKE_USER


def test_protected_endpoint_works_with_a_real_token():
    api.app.dependency_overrides.pop(api.get_current_user, None)
    try:
        with TestClient(api.app) as client:
            register_response = client.post(
                "/auth/register", json={"username": "realflowuser", "password": "supersecret"}
            )
            token = register_response.json()["access_token"]
            response = client.get("/threads", headers={"Authorization": f"Bearer {token}"})
            assert response.status_code == 200
    finally:
        api.app.dependency_overrides[api.get_current_user] = lambda: FAKE_USER


# ---------- Phase 7: user-scoped threads ----------


def test_users_cannot_access_each_others_threads():
    api.app.dependency_overrides[api.get_current_user] = lambda: {"id": "user-alpha", "username": "alpha"}
    with TestClient(api.app) as client:
        thread = client.post("/threads").json()

    api.app.dependency_overrides[api.get_current_user] = lambda: {"id": "user-beta", "username": "beta"}
    with TestClient(api.app) as client:
        response = client.get(f"/threads/{thread['id']}/messages")
        assert response.status_code == 404  # existence not revealed to a non-owner


def test_users_only_see_their_own_threads_in_list():
    api.app.dependency_overrides[api.get_current_user] = lambda: {"id": "user-alpha", "username": "alpha"}
    with TestClient(api.app) as client:
        client.post("/threads")

    api.app.dependency_overrides[api.get_current_user] = lambda: {"id": "user-beta", "username": "beta"}
    with TestClient(api.app) as client:
        client.post("/threads")
        beta_threads = client.get("/threads").json()
        assert len(beta_threads) == 1  # only beta's own thread, not alpha's


# ---------- Phase 7: rate limiting ----------


@patch("api.get_response_with_sources", return_value=("Test answer", []))
@patch("api.build_index_if_needed", return_value=0)
def test_ask_endpoint_enforces_rate_limit(mock_build, mock_get_response, monkeypatch):
    monkeypatch.setattr(rag_rate_limit, "MAX_REQUESTS_PER_WINDOW", 3)
    with TestClient(api.app) as client:
        for _ in range(3):
            response = client.post("/ask", json={"question": "Test?"})
            assert response.status_code == 200
        blocked = client.post("/ask", json={"question": "Test?"})
        assert blocked.status_code == 429


# ---------- Phase 7: feedback ----------


@patch("api.build_index_if_needed", return_value=0)
def test_feedback_endpoint_records_rating(mock_build):
    with TestClient(api.app) as client:
        response = client.post(
            "/feedback",
            json={"thread_id": "thread-1", "question": "Q?", "answer": "A.", "rating": "up"},
        )
        assert response.status_code == 200
        assert response.json() == {"status": "recorded"}


@patch("api.build_index_if_needed", return_value=0)
def test_feedback_endpoint_rejects_invalid_rating(mock_build):
    with TestClient(api.app) as client:
        response = client.post(
            "/feedback",
            json={"thread_id": "thread-1", "question": "Q?", "answer": "A.", "rating": "sideways"},
        )
        assert response.status_code == 400


# ---------- Phase 7: analytics ----------


@patch("api.get_response_with_sources", return_value=("Air India has 138 aircraft.", []))
@patch("api.build_index_if_needed", return_value=0)
def test_analytics_summary_reflects_asked_questions(mock_build, mock_get_response):
    with TestClient(api.app) as client:
        client.post("/ask", json={"question": "How many aircraft?"})
        summary = client.get("/analytics/summary").json()
        assert summary["total_questions"] == 1


# ---------- Generalization: the UI gets its identity from the active profile ----------


@patch("api.build_index_if_needed", return_value=0)
def test_config_endpoint_is_public_and_reflects_the_active_profile(mock_build):
    api.app.dependency_overrides.pop(api.get_current_user, None)  # prove it needs no token
    try:
        with TestClient(api.app) as client:
            response = client.get("/config")
            assert response.status_code == 200
            body = response.json()
            assert body["profile"] == rag_config.PROFILE_NAME
            assert body["app_name"] == rag_config.APP_NAME
            assert body["suggested_questions"] == rag_config.SUGGESTED_QUESTIONS
            assert ".pdf" in body["supported_extensions"]
    finally:
        api.app.dependency_overrides[api.get_current_user] = lambda: FAKE_USER
