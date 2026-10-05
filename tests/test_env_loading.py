"""Settings in .env must reach every module, whatever order the app imports them in."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_dotenv_values_reach_modules_imported_before_config(tmp_path):
    shutil.copytree(PROJECT_ROOT / "rag", tmp_path / "rag", ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / ".env").write_text("JWT_SECRET_KEY=from-dotenv\nRATE_LIMIT_MAX_REQUESTS=7\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("JWT_", "RATE_LIMIT_"))}
    env["PYTHONPATH"] = str(tmp_path)

    # auth and rate_limit read their settings at import time and don't import config —
    # the same situation as api.py, which imports them before rag.config.
    code = "from rag import auth, rate_limit; print(auth.JWT_SECRET, rate_limit.MAX_REQUESTS_PER_WINDOW)"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["from-dotenv", "7"]
