"""Loads documents, splits them into chunks, embeds them, and persists a local Chroma store.

Domain-agnostic: it indexes whatever is in the active profile's docs folder (see
rag/config.py), plus anything uploaded at runtime. Supported: PDF, DOCX, TXT, MD.

Docs-folder sync (build_index_if_needed):
  - Every supported file under the docs folder is fingerprinted (SHA-256 of its bytes).
  - A small manifest remembers which fingerprint was indexed for each file, so each
    startup only (re-)embeds NEW or CHANGED files and removes the chunks of DELETED
    ones. Drop a file into the folder, restart, and it's searchable — no need to
    wipe the vector store.
  - Files directly inside the folder go into the profile's default knowledge base;
    files inside a subfolder go into a knowledge base named after that (top-level)
    subfolder.
  - Every chunk records its origin: "folder" chunks are owned by the sync, while
    "upload" chunks (added at runtime through the API) are never touched by it.

Phase 3 (unchanged):
  - Two-level parent/child chunking (Parent Document Retriever pattern): small
    child chunks get embedded for precise matching, larger parent chunks are
    persisted to the parent store and looked up at retrieval time for the
    actual context handed to the LLM.
  - Near-empty child chunks (stray page numbers, headers) are dropped.
  - get_all_child_documents() feeds the BM25 keyword index built in retrieval.py.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from . import config

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
SUPPORTED_UPLOAD_EXTENSIONS = SUPPORTED_EXTENSIONS  # uploads and the docs folder accept the same types

ORIGIN_FOLDER = "folder"  # chunk came from the profile's docs folder — managed by the sync
ORIGIN_UPLOAD = "upload"  # chunk came from a runtime upload — never touched by the sync

Path(config.STORAGE_DIR).mkdir(parents=True, exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

vector_store = Chroma(
    collection_name=config.COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=config.PERSIST_DIR,
)

_parent_store: Dict[str, str] = {}


def _load_parent_store() -> None:
    if os.path.exists(config.PARENT_STORE_FILE):
        with open(config.PARENT_STORE_FILE, "r", encoding="utf-8") as f:
            _parent_store.clear()
            _parent_store.update(json.load(f))


def _save_parent_store() -> None:
    with open(config.PARENT_STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(_parent_store, f)


_load_parent_store()


def get_parent_text(parent_id: str) -> Optional[str]:
    """Looks up a parent chunk's full text by id — used by retrieval to expand
    a precisely-matched (small) child chunk back out to its full surrounding context."""
    return _parent_store.get(parent_id)


def _chunk_with_parent_child(documents: List[Document]) -> List[Document]:
    """Splits documents into parent chunks (persisted for lookup) and child chunks
    (returned for embedding), linking each child to its parent via parent_id metadata.
    Child chunks shorter than MIN_CHUNK_LENGTH are dropped as noise.
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.PARENT_CHUNK_SIZE, chunk_overlap=config.PARENT_CHUNK_OVERLAP
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE, chunk_overlap=config.CHILD_CHUNK_OVERLAP
    )

    parent_chunks = parent_splitter.split_documents(documents)
    child_chunks: List[Document] = []

    for parent in parent_chunks:
        parent_id = str(uuid4())
        _parent_store[parent_id] = parent.page_content

        for child in child_splitter.split_documents([parent]):
            if len(child.page_content.strip()) < config.MIN_CHUNK_LENGTH:
                continue
            child.metadata["parent_id"] = parent_id
            child_chunks.append(child)

    _save_parent_store()
    return child_chunks


def _add_chunks(documents: List[Document]) -> int:
    """Chunks and embeds already-tagged documents. Returns the number of child chunks
    stored. A file with no extractable text (e.g. a scanned PDF) yields 0 chunks —
    Chroma rejects an empty insert, so that case is skipped rather than crashing."""
    child_chunks = _chunk_with_parent_child(documents)
    if child_chunks:
        ids = [str(uuid4()) for _ in range(len(child_chunks))]
        vector_store.add_documents(documents=child_chunks, ids=ids)
    return len(child_chunks)


def _count_chunks() -> int:
    return len(vector_store.get(include=[]).get("ids") or [])


def _loader_for(file_path: str):
    """Picks the right LangChain loader based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    if ext == ".docx":
        return Docx2txtLoader(file_path)
    if ext in (".txt", ".md"):
        return TextLoader(file_path, encoding="utf-8")
    raise ValueError(f"Unsupported file type: {ext}")


# ---------- Runtime uploads ----------


def add_uploaded_document(
    file_path: str,
    original_filename: str,
    knowledge_base: str = config.DEFAULT_UPLOAD_KNOWLEDGE_BASE,
) -> int:
    """Loads, chunks (parent/child), embeds, and stores a single uploaded file.

    Returns the number of child chunks added. They're added to the SAME
    vector_store used for retrieval, so they become searchable immediately.
    """
    documents = _loader_for(file_path).load()

    for doc in documents:
        doc.metadata["source"] = original_filename
        doc.metadata["knowledge_base"] = knowledge_base
        doc.metadata["origin"] = ORIGIN_UPLOAD

    return _add_chunks(documents)


# ---------- Docs-folder sync ----------


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_docs_folder(folder) -> Dict[str, str]:
    """Maps every supported file under `folder` (as a relative POSIX path) to a
    fingerprint of its contents. Hidden files and folders (.gitkeep, .git, ...) are skipped."""
    root = Path(folder)
    found: Dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or any(part.startswith(".") for part in relative.parts):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        found[relative.as_posix()] = _fingerprint(path)
    return found


def knowledge_base_for(relative_path: str) -> str:
    """"leave.pdf" -> the profile's default knowledge base; "HR Policy/leave.pdf" -> "HR Policy"."""
    parts = relative_path.split("/")
    return parts[0] if len(parts) > 1 else config.DEFAULT_KNOWLEDGE_BASE


def _load_manifest() -> Dict[str, str]:
    if not os.path.exists(config.INDEX_MANIFEST_FILE):
        return {}
    with open(config.INDEX_MANIFEST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_manifest(manifest: Dict[str, str]) -> None:
    with open(config.INDEX_MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def _remove_folder_document(relative_path: str) -> int:
    """Deletes every chunk (and its parent text) that came from this docs-folder file."""
    existing = vector_store.get(
        where={"$and": [{"origin": ORIGIN_FOLDER}, {"source": relative_path}]},
        include=["metadatas"],
    )
    ids = existing.get("ids") or []
    if not ids:
        return 0
    for meta in existing.get("metadatas") or []:
        _parent_store.pop((meta or {}).get("parent_id"), None)
    vector_store.delete(ids=ids)
    _save_parent_store()
    return len(ids)


def _index_folder_document(folder: Path, relative_path: str) -> int:
    documents = _loader_for(str(folder / relative_path)).load()
    for doc in documents:
        doc.metadata["source"] = relative_path
        doc.metadata["knowledge_base"] = knowledge_base_for(relative_path)
        doc.metadata["origin"] = ORIGIN_FOLDER
    return _add_chunks(documents)


def build_index_if_needed(docs_folder: Optional[str] = None) -> int:
    """Brings the vector store in line with the docs folder (see module docstring).

    Safe to call on every startup: when nothing changed it only hashes the files.
    Returns the total number of child chunks in the store (folder + uploads).
    """
    folder = Path(docs_folder or config.DOCS_FOLDER)
    if not folder.is_dir():
        # A missing folder is far more likely a misconfiguration (typo, unmounted
        # volume) than a request to delete everything — so don't touch the index.
        print(f"[index] Docs folder not found: {folder} — skipping sync, existing index left as-is.")
        return _count_chunks()

    current = scan_docs_folder(folder)
    manifest = _load_manifest()
    if manifest and _count_chunks() == 0:
        manifest = {}  # the vector store was wiped but the manifest survived — rebuild everything

    removed = [p for p in manifest if p not in current]
    changed = [p for p in current if p in manifest and manifest[p] != current[p]]
    added = [p for p in current if p not in manifest]
    unchanged = len(current) - len(changed) - len(added)

    for relative_path in removed:
        _remove_folder_document(relative_path)
        manifest.pop(relative_path)
        _save_manifest(manifest)
        print(f"[index] Removed {relative_path} (no longer in the docs folder)")

    for relative_path in changed + added:
        # Clears the previous version of a changed file — and any leftovers from a
        # run that crashed halfway through this file — so chunks never duplicate.
        _remove_folder_document(relative_path)
        try:
            chunk_count = _index_folder_document(folder, relative_path)
        except Exception as e:
            # One unreadable file shouldn't stop the whole app from starting. It's
            # left out of the manifest, so it's retried on the next startup.
            manifest.pop(relative_path, None)
            _save_manifest(manifest)
            print(f"[index] Skipped {relative_path}: {e}")
            continue
        manifest[relative_path] = current[relative_path]
        _save_manifest(manifest)
        print(
            f"[index] Indexed {relative_path} -> {chunk_count} chunks "
            f"(knowledge base: {knowledge_base_for(relative_path)})"
        )

    total = _count_chunks()
    print(
        f"[index] Profile '{config.PROFILE_NAME}': {len(added)} new, {len(changed)} changed, "
        f"{len(removed)} removed, {unchanged} unchanged file(s). {total} chunks in the store."
    )
    return total


# ---------- Queries over the store ----------


def list_knowledge_bases() -> List[str]:
    """Returns the distinct knowledge base names currently present in the vector store."""
    existing = vector_store.get(include=["metadatas"])
    names = set()
    for meta in existing.get("metadatas", []) or []:
        meta = meta or {}
        names.add(meta.get("knowledge_base", config.DEFAULT_KNOWLEDGE_BASE))
    return sorted(names)


def get_all_child_documents(knowledge_base: Optional[str] = None) -> List[Document]:
    """Pulls every child chunk currently stored — used to (re)build the BM25
    keyword index for hybrid search. Optionally restricted to one knowledge base.
    """
    existing = vector_store.get(include=["documents", "metadatas"])
    docs: List[Document] = []
    texts = existing.get("documents", []) or []
    metas = existing.get("metadatas", []) or []
    for text, meta in zip(texts, metas):
        meta = meta or {}
        if knowledge_base and meta.get("knowledge_base", config.DEFAULT_KNOWLEDGE_BASE) != knowledge_base:
            continue
        docs.append(Document(page_content=text, metadata=meta))
    return docs
