"""Incremental docs-folder sync (rag/ingestion.py: build_index_if_needed).

Uses a REAL Chroma collection (in a temp dir) with tiny fake embeddings, so the
metadata filters and deletes are exercised against the actual vector store API
rather than a mock — no model download needed.
"""

import hashlib
from uuid import uuid4

import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from rag import config, ingestion


class _FakeEmbeddings(Embeddings):
    def _vec(self, text):
        return [b / 255 for b in hashlib.sha256(text.encode("utf-8")).digest()[:8]]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def _text(topic):
    return f"This paragraph is about {topic}. " * 6  # comfortably above MIN_CHUNK_LENGTH


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Isolated vector store, parent store and manifest for each test."""
    vector_store = Chroma(
        collection_name=f"test_{uuid4().hex[:8]}",
        embedding_function=_FakeEmbeddings(),
        persist_directory=str(tmp_path / "chroma"),
    )
    monkeypatch.setattr(ingestion, "vector_store", vector_store)
    monkeypatch.setattr(config, "PARENT_STORE_FILE", str(tmp_path / "parent_store.json"))
    monkeypatch.setattr(config, "INDEX_MANIFEST_FILE", str(tmp_path / "manifest.json"))
    monkeypatch.setattr(config, "DEFAULT_KNOWLEDGE_BASE", "Documents")
    ingestion._parent_store.clear()
    yield vector_store
    ingestion._parent_store.clear()


@pytest.fixture
def docs(tmp_path):
    folder = tmp_path / "docs"
    (folder / "HR Policy").mkdir(parents=True)
    (folder / "handbook.txt").write_text(_text("the company handbook"), encoding="utf-8")
    (folder / "HR Policy" / "leave.md").write_text(_text("annual leave"), encoding="utf-8")
    return folder


def _chunks(vector_store, source=None):
    result = vector_store.get(include=["metadatas", "documents"])
    rows = list(zip(result["metadatas"], result["documents"]))
    return [(m, d) for m, d in rows if source is None or m["source"] == source]


# ---------- Scanning & knowledge-base naming ----------


def test_scan_finds_supported_files_and_skips_hidden_and_unsupported(tmp_path):
    (tmp_path / "Finance").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / "a.pdf").write_bytes(b"x")
    (tmp_path / "Finance" / "b.TXT").write_text("x")
    (tmp_path / ".gitkeep").write_text("")
    (tmp_path / ".git" / "config.txt").write_text("x")
    (tmp_path / "image.png").write_bytes(b"x")

    assert set(ingestion.scan_docs_folder(tmp_path)) == {"a.pdf", "Finance/b.TXT"}


def test_knowledge_base_comes_from_the_top_level_subfolder(monkeypatch):
    monkeypatch.setattr(config, "DEFAULT_KNOWLEDGE_BASE", "Documents")
    assert ingestion.knowledge_base_for("handbook.pdf") == "Documents"
    assert ingestion.knowledge_base_for("HR Policy/leave.pdf") == "HR Policy"
    assert ingestion.knowledge_base_for("HR Policy/2024/leave.pdf") == "HR Policy"


# ---------- Sync behaviour ----------


def test_first_sync_indexes_every_file_with_kb_and_origin(store, docs):
    total = ingestion.build_index_if_needed(str(docs))

    assert total == len(_chunks(store)) > 0
    handbook = _chunks(store, "handbook.txt")
    leave = _chunks(store, "HR Policy/leave.md")
    assert handbook and all(m["knowledge_base"] == "Documents" for m, _ in handbook)
    assert leave and all(m["knowledge_base"] == "HR Policy" for m, _ in leave)
    assert all(m["origin"] == "folder" for m, _ in _chunks(store))
    assert ingestion.list_knowledge_bases() == ["Documents", "HR Policy"]


def test_second_sync_with_no_changes_embeds_nothing(store, docs, monkeypatch):
    ingestion.build_index_if_needed(str(docs))
    before = len(_chunks(store))

    calls = []
    real_loader_for = ingestion._loader_for
    monkeypatch.setattr(ingestion, "_loader_for", lambda path: calls.append(path) or real_loader_for(path))
    ingestion.build_index_if_needed(str(docs))

    assert calls == []  # nothing re-read, nothing re-embedded
    assert len(_chunks(store)) == before


def test_changed_file_is_replaced_not_duplicated(store, docs):
    ingestion.build_index_if_needed(str(docs))
    old_count = len(_chunks(store, "handbook.txt"))

    (docs / "handbook.txt").write_text(_text("the REVISED handbook"), encoding="utf-8")
    ingestion.build_index_if_needed(str(docs))

    handbook = _chunks(store, "handbook.txt")
    assert len(handbook) == old_count
    assert all("REVISED" in text for _, text in handbook)


def test_deleted_file_loses_its_chunks_and_parents_but_uploads_survive(store, docs, tmp_path):
    ingestion.build_index_if_needed(str(docs))
    upload = tmp_path / "upload.txt"
    upload.write_text(_text("an uploaded memo"), encoding="utf-8")
    ingestion.add_uploaded_document(str(upload), "memo.txt", knowledge_base="Memos")
    leave_parents = {m["parent_id"] for m, _ in _chunks(store, "HR Policy/leave.md")}

    (docs / "HR Policy" / "leave.md").unlink()
    ingestion.build_index_if_needed(str(docs))

    assert _chunks(store, "HR Policy/leave.md") == []
    assert not leave_parents & set(ingestion._parent_store)       # parent texts cleaned up too
    assert _chunks(store, "memo.txt")                                # uploads are never touched
    assert _chunks(store, "handbook.txt")


def test_new_file_dropped_in_later_is_picked_up(store, docs):
    ingestion.build_index_if_needed(str(docs))
    (docs / "Finance").mkdir()
    (docs / "Finance" / "budget.txt").write_text(_text("the budget"), encoding="utf-8")

    ingestion.build_index_if_needed(str(docs))

    budget = _chunks(store, "Finance/budget.txt")
    assert budget and all(m["knowledge_base"] == "Finance" for m, _ in budget)


def test_unreadable_file_is_skipped_and_retried_later(store, docs):
    (docs / "broken.pdf").write_bytes(b"this is not really a pdf")

    ingestion.build_index_if_needed(str(docs))  # must not raise

    assert _chunks(store, "handbook.txt")
    assert "broken.pdf" not in ingestion._load_manifest()  # so the next startup tries it again


def test_missing_docs_folder_leaves_the_index_alone(store, docs, tmp_path):
    ingestion.build_index_if_needed(str(docs))
    before = len(_chunks(store))

    assert ingestion.build_index_if_needed(str(tmp_path / "typo_in_path")) == before
    assert len(_chunks(store)) == before


def test_wiped_vector_store_triggers_a_full_rebuild(store, docs):
    ingestion.build_index_if_needed(str(docs))
    store.delete(ids=store.get(include=[])["ids"])  # e.g. someone deleted storage/<profile>/chroma

    total = ingestion.build_index_if_needed(str(docs))

    assert total > 0 and _chunks(store, "handbook.txt") and _chunks(store, "HR Policy/leave.md")
