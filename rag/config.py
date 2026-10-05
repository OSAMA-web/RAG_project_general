"""Central configuration.

Everything domain-specific lives in a *profile*; everything in this file is
domain-agnostic plumbing. Pick the active profile with the APP_PROFILE
environment variable (default: "general"). A profile is just a folder:

    profiles/<name>/
        profile.json   branding, prompt wording, suggested questions, enabled tools
        docs/          documents indexed at startup. Files directly inside go into the
                       profile's default knowledge base; each subfolder becomes its own
                       knowledge base named after the folder (docs/HR Policy/*.pdf -> "HR Policy")
        eval.json      optional golden Q&A set used by evaluate.py

Adding a new domain = adding a folder. No Python changes needed.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from .tools import AVAILABLE_TOOLS

load_dotenv()  # loads variables from a .env file if present (see .env.example)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROFILES_DIR = PROJECT_ROOT / "profiles"

# Every key a profile.json may set, and the value used when it's left out — so a
# new profile only needs the keys it actually wants to change.
PROFILE_DEFAULTS: Dict[str, Any] = {
    "app_name": "Document Assistant",
    "icon": "📚",
    # Completes the sentence "You are <app_name>, an assistant that answers questions about ___."
    "domain_description": "the documents in its knowledge bases",
    "default_knowledge_base": "Documents",
    "greeting": (
        "Ask me anything about the documents loaded here. I'll remember what we've discussed "
        "so you can ask follow-ups, and I'll show you exactly what I based each answer on."
    ),
    "input_placeholder": "Ask a question about your documents…",
    "suggested_questions": [],
    "enabled_tools": [],
    # None -> profiles/<name>/docs. A relative path is resolved against the project root,
    # so a profile can point at an existing folder of documents without copying it.
    "docs_folder": None,
}

_LIST_KEYS = ("suggested_questions", "enabled_tools")


def load_profile(name: str, profiles_dir: Path = PROFILES_DIR) -> Dict[str, Any]:
    """Reads profiles/<name>/profile.json, validates it, fills in defaults, and
    resolves paths. Fails loudly on typos rather than silently ignoring them."""
    profile_dir = Path(profiles_dir) / name
    profile_file = profile_dir / "profile.json"
    if not profile_file.is_file():
        available = (
            sorted(p.name for p in Path(profiles_dir).iterdir() if (p / "profile.json").is_file())
            if Path(profiles_dir).is_dir()
            else []
        )
        raise FileNotFoundError(
            f"Profile '{name}' not found (expected {profile_file}). "
            f"Available profiles: {', '.join(available) or 'none'}."
        )

    with open(profile_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Keys starting with "_" are treated as comments (JSON has no comment syntax).
    data = {k: v for k, v in raw.items() if not k.startswith("_")}
    unknown = sorted(set(data) - set(PROFILE_DEFAULTS))
    if unknown:
        raise ValueError(
            f"Unknown key(s) {unknown} in {profile_file}. Allowed keys: {sorted(PROFILE_DEFAULTS)}."
        )

    profile = {**PROFILE_DEFAULTS, **data}

    for key in _LIST_KEYS:
        value = profile[key]
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"'{key}' in {profile_file} must be a list of strings.")

    unknown_tools = sorted(set(profile["enabled_tools"]) - set(AVAILABLE_TOOLS))
    if unknown_tools:
        raise ValueError(
            f"Unknown tool(s) {unknown_tools} in {profile_file}. Available tools: {list(AVAILABLE_TOOLS)}."
        )

    docs_folder = profile["docs_folder"]
    if docs_folder:
        docs_path = Path(docs_folder)
        profile["docs_folder"] = str(docs_path if docs_path.is_absolute() else PROJECT_ROOT / docs_path)
    else:
        profile["docs_folder"] = str(profile_dir / "docs")

    eval_file = profile_dir / "eval.json"
    profile["eval_dataset"] = str(eval_file) if eval_file.is_file() else None
    profile["name"] = name
    return profile


# ---------- Active profile ----------

PROFILE_NAME = os.environ.get("APP_PROFILE", "").strip() or "general"
PROFILE = load_profile(PROFILE_NAME)

APP_NAME = PROFILE["app_name"]
APP_ICON = PROFILE["icon"]
DOMAIN_DESCRIPTION = PROFILE["domain_description"]
GREETING = PROFILE["greeting"]
INPUT_PLACEHOLDER = PROFILE["input_placeholder"]
SUGGESTED_QUESTIONS = PROFILE["suggested_questions"]
ENABLED_TOOLS = PROFILE["enabled_tools"]
DOCS_FOLDER = PROFILE["docs_folder"]
EVAL_DATASET = PROFILE["eval_dataset"]

# Knowledge bases — every stored chunk is tagged with one of these (or a
# subfolder name, or a custom name typed in at upload time), so retrieval can
# optionally filter by it.
DEFAULT_KNOWLEDGE_BASE = PROFILE["default_knowledge_base"]
DEFAULT_UPLOAD_KNOWLEDGE_BASE = "Uploaded Files"

# ---------- Models (local, free — no API keys required) ----------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")  # try "phi3" if your machine has limited RAM

# ---------- Storage (one isolated index per profile) ----------
# Switching profiles never mixes one domain's chunks into another's answers.
# If you change EMBEDDING_MODEL or the chunk sizes, delete storage/<profile>/ to rebuild.

STORAGE_DIR = PROJECT_ROOT / "storage" / PROFILE_NAME
PERSIST_DIR = str(STORAGE_DIR / "chroma")
COLLECTION_NAME = "documents"
PARENT_STORE_FILE = str(STORAGE_DIR / "parent_store.json")
INDEX_MANIFEST_FILE = str(STORAGE_DIR / "index_manifest.json")

# ---------- Chunking ----------

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Two-level "parent document" chunking: small CHILD chunks get embedded for
# precise matching; larger PARENT chunks are what's actually fed to the LLM as
# context, looked up via a parent_id link on each child chunk.
CHILD_CHUNK_SIZE = CHUNK_SIZE
CHILD_CHUNK_OVERLAP = CHUNK_OVERLAP
PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200
MIN_CHUNK_LENGTH = 40  # drop near-empty child chunks (stray page numbers, headers, etc.)

# ---------- Retrieval ----------

RETRIEVE_K = 10   # dense-search candidates pulled per query, before reranking
RERANK_K = 3      # final chunks kept after cross-encoder reranking

# Hybrid search + multi-query (Phase 3)
BM25_TOP_K = 10          # keyword-search candidates pulled per query variant
MULTI_QUERY_VARIANTS = 3  # LLM-generated rephrasings of the question, in addition to the original
MIN_RERANK_SCORE = 0.0    # cross-encoder score floor (tuned for ms-marco-MiniLM's score range) — chunks below this are dropped as irrelevant rather than force-included
COMPRESSED_SENTENCES_PER_SOURCE = 4  # sentences kept per source after context compression
