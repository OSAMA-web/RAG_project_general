"""REST API for the RAG assistant.

The domain (name, prompts, documents, enabled tools) comes from the active
profile — see rag/config.py. Pick one with APP_PROFILE (default: "general").

Run locally:
    uvicorn api:app --reload

Docs at http://localhost:8000/docs

Phase 7 adds: JWT authentication, per-user chat ownership, rate limiting on
the expensive endpoints, request logging, chat analytics, and answer feedback.
Public endpoints (no auth required): /, /health, /config, /auth/register, /auth/login.
Everything else requires a valid Bearer token.
"""

import io
import json
import shutil
import time
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from docx import Document as DocxDocument
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fpdf import FPDF
from pydantic import BaseModel, Field

from rag import analytics, auth, feedback, memory, rate_limit, threads
import rag.config as rag_config
from rag.ingestion import (
    SUPPORTED_UPLOAD_EXTENSIONS,
    add_uploaded_document,
    build_index_if_needed,
    list_knowledge_bases,
)
from rag.logging_config import logger
from rag.pipeline import get_response_with_sources, stream_response_with_sources
from rag.tools.router import classify_intent

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title=f"{rag_config.APP_NAME} API", version="1.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

security_scheme = HTTPBearer()


@app.middleware("http")
async def log_requests(request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(f"Unhandled error on {request.method} {request.url.path}")
        raise
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.1f}ms)")
    return response


# ---------- Auth dependencies ----------

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)) -> dict:
    payload = auth.decode_access_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    return {"id": payload["sub"], "username": payload["username"]}


def rate_limited_user(current_user: dict = Depends(get_current_user)) -> dict:
    if not rate_limit.is_allowed(current_user["id"]):
        raise HTTPException(status_code=429, detail="Rate limit exceeded — please slow down.")
    return current_user


# ---------- Request/response models ----------

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")
    session_id: Optional[str] = Field(
        None,
        description="Thread id to continue. Omit to auto-create a new, unregistered thread.",
    )
    knowledge_base: Optional[str] = Field(
        None,
        description="Restrict search to this knowledge base (see GET /knowledge-bases). Omit to search everything.",
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    session_id: str


class ResetRequest(BaseModel):
    session_id: str


class ThreadSummary(BaseModel):
    id: str
    title: str
    user_id: Optional[str] = None
    created_at: str
    updated_at: str


class RenameThreadRequest(BaseModel):
    title: str = Field(..., min_length=1)


class ThreadMessage(BaseModel):
    question: str
    answer: str


class ThreadMessagesResponse(BaseModel):
    thread_id: str
    messages: List[ThreadMessage]


class UploadResponse(BaseModel):
    filename: str
    chunks_added: int


class AppConfigResponse(BaseModel):
    profile: str
    app_name: str
    icon: str
    greeting: str
    input_placeholder: str
    suggested_questions: List[str]
    supported_extensions: List[str]
    default_upload_knowledge_base: str


class FeedbackRequest(BaseModel):
    thread_id: str
    question: str
    answer: str
    rating: str = Field(..., description="'up' or 'down'")


# ---------- Helpers ----------

def _require_owned_thread(thread_id: str, current_user: dict) -> dict:
    """Returns the thread if it exists AND belongs to current_user; otherwise 404
    (not 403 — we don't reveal whether a thread id exists for someone else)."""
    thread = threads.get_thread(thread_id)
    if not thread or thread.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=404, detail="Thread not found.")
    return thread


def _pdf_safe(text: str) -> str:
    """fpdf2's core fonts only support latin-1 — replace anything outside that
    range rather than crashing the export on an em-dash, emoji, etc."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _safe_filename(title: str) -> str:
    cleaned = "".join(c for c in title if c.isalnum() or c in " _-").strip()
    return cleaned or "chat"


# ---------- Startup ----------

@app.on_event("startup")
def startup() -> None:
    build_index_if_needed()


# ---------- Public endpoints ----------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def serve_frontend() -> FileResponse:
    return FileResponse("web/index.html")


@app.get("/config", response_model=AppConfigResponse)
def get_app_config() -> AppConfigResponse:
    """Branding and UI text for the active profile, so the web UI hardcodes no
    domain. Public on purpose: the login screen needs the app name before auth."""
    return AppConfigResponse(
        profile=rag_config.PROFILE_NAME,
        app_name=rag_config.APP_NAME,
        icon=rag_config.APP_ICON,
        greeting=rag_config.GREETING,
        input_placeholder=rag_config.INPUT_PLACEHOLDER,
        suggested_questions=rag_config.SUGGESTED_QUESTIONS,
        supported_extensions=sorted(SUPPORTED_UPLOAD_EXTENSIONS),
        default_upload_knowledge_base=rag_config.DEFAULT_UPLOAD_KNOWLEDGE_BASE,
    )


@app.post("/auth/register", response_model=TokenResponse)
def register(payload: RegisterRequest) -> TokenResponse:
    user = auth.create_user(payload.username, payload.password)
    if not user:
        raise HTTPException(status_code=400, detail="Username already taken.")
    token = auth.create_access_token(user["id"], user["username"])
    logger.info(f"New user registered: {user['username']}")
    return TokenResponse(access_token=token, username=user["username"])


@app.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest) -> TokenResponse:
    user = auth.authenticate_user(payload.username, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = auth.create_access_token(user["id"], user["username"])
    return TokenResponse(access_token=token, username=user["username"])


# ---------- Authenticated endpoints ----------

@app.get("/knowledge-bases", response_model=List[str])
def get_knowledge_bases(current_user: dict = Depends(get_current_user)) -> List[str]:
    return list_knowledge_bases()


@app.post("/threads", response_model=ThreadSummary)
def create_new_thread(current_user: dict = Depends(get_current_user)) -> ThreadSummary:
    thread = threads.create_thread(user_id=current_user["id"])
    return ThreadSummary(**thread)


@app.get("/threads", response_model=List[ThreadSummary])
def list_all_threads(current_user: dict = Depends(get_current_user)) -> List[ThreadSummary]:
    return [ThreadSummary(**t) for t in threads.list_threads(user_id=current_user["id"])]


@app.get("/threads/search", response_model=List[ThreadSummary])
def search_threads(q: str = "", current_user: dict = Depends(get_current_user)) -> List[ThreadSummary]:
    """Searches the current user's chats by title AND message content."""
    query = q.strip().lower()
    own_threads = threads.list_threads(user_id=current_user["id"])
    if not query:
        return [ThreadSummary(**t) for t in own_threads]

    matches = []
    for t in own_threads:
        if query in t["title"].lower():
            matches.append(t)
            continue
        history = memory.get_history(t["id"])
        if any(query in q_text.lower() or query in a_text.lower() for q_text, a_text in history):
            matches.append(t)
    return [ThreadSummary(**t) for t in matches]


@app.get("/threads/{thread_id}/messages", response_model=ThreadMessagesResponse)
def get_thread_messages(thread_id: str, current_user: dict = Depends(get_current_user)) -> ThreadMessagesResponse:
    _require_owned_thread(thread_id, current_user)
    history = memory.get_history(thread_id)
    return ThreadMessagesResponse(
        thread_id=thread_id,
        messages=[ThreadMessage(question=q, answer=a) for q, a in history],
    )


@app.patch("/threads/{thread_id}", response_model=ThreadSummary)
def update_thread_title(
    thread_id: str, payload: RenameThreadRequest, current_user: dict = Depends(get_current_user)
) -> ThreadSummary:
    _require_owned_thread(thread_id, current_user)
    updated = threads.rename_thread(thread_id, payload.title)
    return ThreadSummary(**updated)


@app.delete("/threads/{thread_id}")
def delete_thread_endpoint(thread_id: str, current_user: dict = Depends(get_current_user)) -> dict:
    _require_owned_thread(thread_id, current_user)
    threads.delete_thread(thread_id)
    memory.clear_session(thread_id)
    return {"status": "deleted"}


@app.get("/threads/{thread_id}/export/{export_format}")
def export_thread(
    thread_id: str, export_format: str, current_user: dict = Depends(get_current_user)
) -> Response:
    if export_format not in {"txt", "pdf", "docx"}:
        raise HTTPException(status_code=400, detail="format must be one of: txt, pdf, docx")

    thread = _require_owned_thread(thread_id, current_user)
    history = memory.get_history(thread_id)
    filename_base = _safe_filename(thread["title"])

    if export_format == "txt":
        lines = [thread["title"], "=" * len(thread["title"]), ""]
        for question, answer in history:
            lines.append(f"You: {question}")
            lines.append(f"Assistant: {answer}")
            lines.append("")
        return Response(
            content="\n".join(lines),
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="{filename_base}.txt"'},
        )

    if export_format == "pdf":
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(0, 10, _pdf_safe(thread["title"]))
        pdf.set_font("Helvetica", "", 11)
        pdf.ln(4)
        for question, answer in history:
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(0, 7, _pdf_safe(f"You: {question}"))
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 7, _pdf_safe(f"Assistant: {answer}"))
            pdf.ln(3)
        return Response(
            content=bytes(pdf.output()),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename_base}.pdf"'},
        )

    doc = DocxDocument()
    doc.add_heading(thread["title"], level=1)
    for question, answer in history:
        doc.add_paragraph(f"You: {question}")
        doc.add_paragraph(f"Assistant: {answer}")
        doc.add_paragraph("")
    buffer = io.BytesIO()
    doc.save(buffer)
    return Response(
        content=buffer.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename_base}.docx"'},
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    knowledge_base: str = Form(rag_config.DEFAULT_UPLOAD_KNOWLEDGE_BASE),
    current_user: dict = Depends(get_current_user),
) -> UploadResponse:
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_UPLOAD_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Allowed: {allowed}.")

    destination = UPLOAD_DIR / file.filename
    if destination.exists():
        destination = UPLOAD_DIR / f"{uuid4().hex[:8]}_{file.filename}"

    with destination.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunk_count = add_uploaded_document(
            str(destination),
            file.filename,
            knowledge_base=knowledge_base or rag_config.DEFAULT_UPLOAD_KNOWLEDGE_BASE,
        )
    except Exception as e:
        logger.exception(f"Upload processing failed for {file.filename}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")

    logger.info(f"User {current_user['username']} uploaded {file.filename} ({chunk_count} chunks)")
    return UploadResponse(filename=file.filename, chunks_added=chunk_count)


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, current_user: dict = Depends(rate_limited_user)) -> AskResponse:
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = payload.session_id or memory.new_session_id()
    existing = threads.get_thread(session_id)
    if existing and existing.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=403, detail="This chat belongs to another user.")
    threads.ensure_thread(session_id, user_id=current_user["id"])

    is_first_turn = len(memory.get_history(session_id)) == 0
    intent = classify_intent(payload.question)
    start = time.perf_counter()

    answer, sources = get_response_with_sources(
        payload.question, session_id, knowledge_base=payload.knowledge_base
    )

    response_time_ms = (time.perf_counter() - start) * 1000
    analytics.log_event(current_user["id"], intent, payload.knowledge_base, response_time_ms)

    if is_first_turn:
        threads.maybe_autotitle(session_id, payload.question)
    threads.touch_thread(session_id)

    return AskResponse(question=payload.question, answer=answer, sources=sources, session_id=session_id)


@app.post("/reset")
def reset(payload: ResetRequest, current_user: dict = Depends(get_current_user)) -> dict:
    _require_owned_thread(payload.session_id, current_user)
    memory.clear_session(payload.session_id)
    return {"status": "cleared"}


@app.post("/ask/stream")
def ask_stream(payload: AskRequest, current_user: dict = Depends(rate_limited_user)) -> StreamingResponse:
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = payload.session_id or memory.new_session_id()
    existing = threads.get_thread(session_id)
    if existing and existing.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=403, detail="This chat belongs to another user.")
    threads.ensure_thread(session_id, user_id=current_user["id"])
    is_first_turn = len(memory.get_history(session_id)) == 0
    intent = classify_intent(payload.question)
    start = time.perf_counter()

    def event_generator():
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
        try:
            for event in stream_response_with_sources(
                payload.question, session_id, knowledge_base=payload.knowledge_base
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.exception(f"Streaming failure for session {session_id}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        response_time_ms = (time.perf_counter() - start) * 1000
        analytics.log_event(current_user["id"], intent, payload.knowledge_base, response_time_ms)

        if is_first_turn:
            threads.maybe_autotitle(session_id, payload.question)
        threads.touch_thread(session_id)
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/feedback")
def submit_feedback(payload: FeedbackRequest, current_user: dict = Depends(get_current_user)) -> dict:
    if payload.rating not in {"up", "down"}:
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")
    feedback.add_feedback(
        current_user["id"], payload.thread_id, payload.question, payload.answer, payload.rating
    )
    return {"status": "recorded"}


@app.get("/analytics/summary")
def get_analytics_summary(current_user: dict = Depends(get_current_user)) -> dict:
    return analytics.get_summary(user_id=current_user["id"])
