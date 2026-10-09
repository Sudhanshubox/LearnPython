"""Settings for the AI mentor. Override any of these with environment variables."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CURRICULUM_DIR = ROOT / "curriculum"
PROGRESS_FILE = Path(os.environ.get("MENTOR_PROGRESS_FILE", ROOT / "progress.json"))
CHATS_DIR = Path(os.environ.get("MENTOR_CHATS_DIR", ROOT / ".mentor" / "chats"))

MODEL = os.environ.get("MENTOR_MODEL", "claude-opus-5-5")
# low | medium | high | xhigh | max -- higher means deeper thinking but slower and costlier.
EFFORT = os.environ.get("MENTOR_EFFORT", "medium")
MAX_TOKENS = int(os.environ.get("MENTOR_MAX_TOKENS", "32000"))

# If the model declines a request, the API retries it on a recommended fallback model.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

HOST = os.environ.get("MENTOR_HOST", "127.0.0.1")
PORT = int(os.environ.get("MENTOR_PORT", "8765"))
