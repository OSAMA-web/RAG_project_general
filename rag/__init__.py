"""RAG assistant package.

.env is loaded here, before any submodule runs, because several modules read
environment variables at import time — auth (JWT_SECRET_KEY, JWT_EXPIRY_HOURS),
rate_limit (RATE_LIMIT_*), config (APP_PROFILE, OLLAMA_MODEL). Loading it only in
config.py was too late whenever another module was imported first, as api.py does.
Variables already set in the real environment still take precedence.
"""

from dotenv import load_dotenv

load_dotenv()
