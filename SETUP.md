# RAG Assistant — profile-driven, fully local (all 7 phases)

A retrieval-augmented chat assistant over your own documents. Fully local core
stack (local embeddings, local reranking, local LLM via Ollama), with live-data
tools, an AI agent, and professional-grade features (auth, rate limiting,
logging, analytics, feedback).

**Nothing in the code is tied to one domain.** Everything domain-specific — the
name and icon, the wording of the prompts, suggested questions, the documents,
and which live tools are allowed — lives in a *profile*: one folder under
`profiles/`. Two ship with the project:

| Profile | What it is |
|---|---|
| `general` (default) | A neutral document assistant. Its `docs/` folder starts empty — add your files or upload them in the UI. |
| `air_india` | The original Air India assistant, unchanged in behaviour: same branding and greeting, the 5 source PDFs, flight/weather/currency tools, and the 7-question eval set. |

Adding a new domain = adding a folder. No Python changes.

## What's included

- **Chat**: session memory, multiple independent chat threads (new/rename/delete/switch),
  streaming responses (see Known limitations)
- **Documents**: drop PDF/DOCX/TXT/MD files into the profile's `docs/` folder or
  upload them on the fly; organize them into multiple named knowledge bases
- **RAG quality**: hybrid search (keyword + semantic), multi-query expansion,
  parent-document context, sentence-level compression, cross-encoder reranking
- **UI**: markdown/tables/code rendering, copy button, dark mode, suggested
  questions, chat search, export to TXT/PDF/DOCX — branding comes from the profile
- **Live data + AI Agent**: flight status, weather, currency conversion (each
  enabled per profile), with a two-tier router (a strict regex fast path for
  clear-cut live-data questions + genuine LLM tool-calling fallback)
- **Professional features**: user accounts with JWT login, per-user chat
  ownership, rate limiting, structured logging, chat analytics, answer
  feedback (👍/👎), environment-variable config, CI via GitHub Actions

## 1. Install Ollama and pull a model

```bash
ollama pull llama3        # or "ollama pull llama3.1" / "ollama pull qwen3:8b" / "ollama pull phi3"
```

If you use a different model, set `OLLAMA_MODEL` in `.env` (step 3).

The agent's tool calling (the LLM deciding by itself to call the weather,
currency or flight tool) needs a model that supports tools in Ollama, such as
`llama3.1` or `qwen3`. Ollama rejects tool requests for models that don't
(`llama3`, `phi3`); the app notices that, logs a warning once, and answers from
the documents only. The fast path still handles clear-cut live-data questions
with any model.

## 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

First run downloads two small local models from HuggingFace (embedding model
~90MB, reranker ~90MB) — needs internet once, then works offline.

## 3. Configure environment variables

```bash
cp .env.example .env
```

Then edit `.env`:
- **APP_PROFILE** — which profile to run: `general` (default), `air_india`, or
  any folder you add under `profiles/`.
- **OLLAMA_MODEL** — the Ollama model to generate with (default `llama3`).
- **JWT_SECRET_KEY** — change this to a long random string (generate one with
  `python -c "import secrets; print(secrets.token_hex(32))"`). The app works
  without changing it, but every restart with the default secret means any
  previously-issued login tokens stay valid forever — fine for local use, not
  for anything shared.
- **AVIATIONSTACK_API_KEY** — optional, free signup at aviationstack.com. Only
  used by profiles that enable the `flight` tool (`air_india` does). Weather
  and currency need no key at all.

## 4. Run it

```bash
uvicorn api:app --reload
```

On startup the active profile's `docs/` folder is synced into its index (see
[How indexing works](#how-indexing-works)). Open **http://localhost:8000** —
you'll land on a login/register screen first. Register a username + password
(this creates your account), and you're in. API docs at
`http://localhost:8000/docs`.

To run a different profile once without editing `.env`:

```bash
APP_PROFILE=air_india uvicorn api:app --reload            # macOS / Linux
$env:APP_PROFILE="air_india"; uvicorn api:app --reload    # Windows PowerShell
```

**Docker Compose (API + Ollama, fully containerized):**
```bash
docker compose up --build                          # uses APP_PROFILE from .env, else "general"
docker compose exec ollama ollama pull llama3      # first time only
```
`profiles/` and `storage/` are mounted from the host, so you can add documents
or whole profiles without rebuilding the image, and the index survives
container rebuilds.

**Streamlit UI (simpler, single-thread, no auth, no streaming):**
```bash
streamlit run app.py
```

## Profiles

### Layout

```
profiles/
  general/
    profile.json        branding, prompt wording, suggested questions, enabled tools
    docs/               documents indexed at startup (empty to begin with)
  air_india/
    profile.json
    docs/               the 5 Air India PDFs
    eval.json           golden Q&A set for evaluate.py (optional)
```

Inside `docs/`, files at the top level go into the profile's
`default_knowledge_base`; each top-level subfolder becomes its own knowledge
base named after the folder (`docs/HR Policy/leave.pdf` → "HR Policy"; deeper
nesting stays in that top-level folder's knowledge base). Hidden files are ignored.

### Create your own

1. Copy `profiles/general` to `profiles/<your_name>` — the folder name is the profile name.
2. Edit `profile.json`. Every key is optional; anything you leave out falls back to the default.
3. Put your documents in `docs/` (PDF, DOCX, TXT, MD).
4. Optionally add `eval.json`: `[{"question": "...", "reference_answer": "..."}, ...]`.
5. Set `APP_PROFILE=<your_name>` and start the app.

For example, `profiles/hr_policies/profile.json`:

```json
{
  "app_name": "HR Policy Assistant",
  "icon": "🏢",
  "domain_description": "the company's HR policies — leave, benefits, and conduct",
  "default_knowledge_base": "HR Policies",
  "suggested_questions": ["How many days of paid leave do I get?"],
  "enabled_tools": []
}
```

### `profile.json` reference

| Key | Used for | Default |
|---|---|---|
| `app_name` | Header, browser tab, API docs title, and the system prompt ("You are *app_name*, …") | `"Document Assistant"` |
| `icon` | Emoji shown next to the name | `"📚"` |
| `domain_description` | Completes "an assistant that answers questions about ___" in the system prompt | `"the documents in its knowledge bases"` |
| `default_knowledge_base` | Knowledge base for files at the top level of `docs/` | `"Documents"` |
| `greeting` | First message in the web UI | a generic greeting |
| `input_placeholder` | Placeholder text in the chat box | `"Ask a question about your documents…"` |
| `suggested_questions` | Clickable suggestion chips in the web UI; the first one is also `main.py`'s sample question | `[]` |
| `enabled_tools` | Any of `"flight"`, `"weather"`, `"currency"`. Tools not listed are invisible to both the router and the agent | `[]` |
| `docs_folder` | Index a folder somewhere else instead of `profiles/<name>/docs` (relative paths are resolved from the project root) | `profiles/<name>/docs` |

Keys starting with `_` are treated as comments. A misspelled key, an unknown
tool name, or a non-list where a list belongs stops the app at startup with a
message naming the problem — a typo can't silently fall back to a default.
The test suite also loads every folder in `profiles/`, so CI catches it too.

### What's per-profile and what's shared

Per profile: the `profile.json` settings, the documents, and the index —
`storage/<profile>/` holds its vector store, parent-chunk store, and sync
manifest. Switching profiles never mixes one domain's chunks into another's
answers, and files uploaded while a profile is running go into that profile's
index. Shared across profiles: `app_data.db` (user accounts, analytics,
feedback) and the raw copies of uploads in `uploads/`. Chat threads are kept
in memory and reset on restart, as before.

## How indexing works

On every startup (API, Streamlit, `main.py`, `evaluate.py`), each file in the
docs folder is fingerprinted (SHA-256 of its bytes) and compared with
`storage/<profile>/index_manifest.json`:

- **new or changed** files are (re)indexed — the old chunks of a changed file are removed first, so nothing duplicates;
- **deleted** files have their chunks removed from the index;
- **unchanged** files are skipped, so a restart with no changes only hashes the files.

So: drop a file in, restart, and it's searchable. Files uploaded through the UI
are never touched by the sync. A file that can't be read is skipped with a log
line and retried on the next startup. A missing docs folder is treated as a
misconfiguration, not as "delete everything" — the index is left as it is.

**Rebuilding from scratch:** delete `storage/<profile>/`. Do this after
changing `EMBEDDING_MODEL` or the chunk sizes in `rag/config.py` — vectors from
different models (or chunks cut differently) can't be mixed in one index. Note
that this also drops that profile's uploaded files from the index; re-upload them.

**Coming from the Air India-only version:** the index moved from
`chroma_vectorestore_local/` + `parent_store.json` to `storage/<profile>/`.
The old ones are no longer used and can be deleted; the first start
re-indexes the documents.

## Live tools and the router

A question can be answered by a live tool in two ways: a regex **fast path**
that answers clear-cut live-data questions without any LLM call, and the
**agent fallback**, where the LLM sees the retrieved context plus the enabled
tools and decides itself whether to call one.

The fast path is deliberately strict, because its two kinds of mistakes don't
cost the same. If it fires wrongly, document retrieval is skipped and the user
gets a live-data answer to a document question. If it stays quiet wrongly, the
question just goes through the normal pipeline, where the agent can still call
the same tool — one extra LLM call. So it only fires when a question contains
both the thing to look up (a real flight number, a whole-word city, an amount
plus two currencies) and a clear live-data cue ("where is", "right now",
"convert", …). "What happened to Air India in 1985?" goes to the documents,
not to a flight lookup for "IN1985".

Tools are opt-in per profile through `enabled_tools`: `flight`
(AviationStack, needs the key above), `weather` (Open-Meteo, no key),
`currency` (Frankfurter, no key). With no tools enabled, the agent simply
answers from the retrieved documents.

## 5. Run the test suite

```bash
pytest
```

191 tests — all mocked/isolated (temp SQLite DBs, temp vector stores, cleared
in-memory state between tests), so they run in seconds with no LLM, PDFs, or
network. They pass under every shipped profile (`APP_PROFILE=air_india pytest`).

## 6. CI/CD

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs the full test
suite automatically on every push/PR if you push this to a GitHub repo — no
extra setup needed beyond having the repo on GitHub.

## 7. Run the evaluation harness

```bash
python main.py                          # syncs the index, then answers the profile's first suggested question
python evaluate.py                      # runs the active profile's eval.json through the real pipeline
python evaluate.py path/to/golden.json  # or any dataset in the same format
```

With `APP_PROFILE=air_india`, `evaluate.py` runs the original 7-question golden set.

## Known limitations

- **Route maps are images.** In the `air_india` profile, "Domestic Routes Feb
  2025.pdf" and "International Routes Feb 2025.pdf" are single-page maps: only
  the city labels are extractable text, and the routes themselves are drawn
  lines. Text RAG can tell you which cities appear on the map but not which
  cities are connected. Fixes: a hand-written routes table (e.g. a `.md` or
  `.txt` file in `docs/`), or a vision model at ingestion time.
- **Scanned PDFs** (images of text) yield no chunks — run OCR first.
- **Document answers arrive in one piece.** The agent has to see the model's
  complete first reply to know whether it wants to call a tool, so for ordinary
  document answers the stream delivers the whole answer at once; live-data
  answers do stream token by token. Fix: stream the first call and watch the
  chunks for tool calls.
- **Follow-up questions are retrieved without their context.** Chat history
  goes into the answer prompt, but retrieval only sees the latest question, so
  "and how many of those are wide-body?" is searched as written. Fix: rewrite
  the question into a standalone one (using the history) before retrieving.

## Project structure

```
main.py, app.py, api.py        Entry points (CLI / Streamlit / FastAPI + web UI)
evaluate.py                     LLM-as-judge evaluation against a profile's eval.json
web/index.html                  The chat website (single file; branding fetched from GET /config)
profiles/<name>/                One folder per domain: profile.json, docs/, optional eval.json
storage/<name>/                 That profile's index (created at runtime; safe to delete)
rag/
  config.py                     Profile loading + all tunable settings
  ingestion.py                  PDF/DOCX/TXT/MD loading, parent/child chunking, KB tagging, docs-folder sync
  retrieval.py                  Hybrid search, multi-query, compression, reranking
  generation.py                 Ollama calls (sync + streaming)
  agent.py                      AI Agent: fast-path router + LLM tool-calling fallback
  pipeline.py                   Combines everything above
  memory.py, threads.py         Per-thread conversation history + registry
  tools/                        router.py, flight.py, weather.py, currency.py
  auth.py                       Password hashing, JWT
  db.py                         Shared SQLite (users, analytics, feedback)
  analytics.py, feedback.py, rate_limit.py   Professional features
  logging_config.py             Console + rotating file logging
tests/                          191 tests, one file per module above
.env.example                    Environment variable template
.github/workflows/ci.yml        GitHub Actions CI
Dockerfile, docker-compose.yml
requirements.txt
```

## Roadmap status — all 7 phases complete

| Phase | Status |
|---|---|
| 1 — Chat History, Threads, Streaming | ✅ |
| 2 — Upload Documents, Multiple Knowledge Bases | ✅ |
| 3 — RAG Quality | ✅ |
| 4 — UX Polish | ✅ |
| 5 — Live APIs | ✅ |
| 6 — AI Agent | ✅ |
| 7 — Professional Features | ✅ |
| Generalization — domain-agnostic core, profiles, incremental docs sync | ✅ |
