"""
Centralised configuration — reads from environment variables so that
secrets are never hard-coded in source files.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure we always load from the directory where config.py lives
_env_path = Path(__file__).resolve().parent / ".env"
# Platform-provided variables (for example Render's PORT) must take precedence
# over values in the local development file.
load_dotenv(dotenv_path=_env_path, override=False)


def _clean_str(val: str) -> str:
    return val.strip().strip("'\"") if val else ""


def _groq_model() -> str:
    """Use the current model when an older deployment setting is still present."""
    configured = os.getenv("GROQ_MODEL", "").strip()
    if not configured or configured == "llama-3.3-70b-versatile":
        return "openai/gpt-oss-120b"
    return configured


class Config:
    # ── LLM providers ──────────────────────────────────────────────────────────
    GROQ_API_KEY: str = _clean_str(os.getenv("GROQ_API_KEY", ""))
    GROQ_MODEL: str = _groq_model()

    GEMINI_API_KEY: str = _clean_str(os.getenv("GEMINI_API_KEY", ""))
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()

    # ── Search ─────────────────────────────────────────────────────────────────
    TAVILY_API_KEY: str = _clean_str(os.getenv("TAVILY_API_KEY", ""))

    # ── Vector DB (Qdrant serverless) ──────────────────────────────────────────
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333").strip()
    QDRANT_API_KEY: str = _clean_str(os.getenv("QDRANT_API_KEY", ""))
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "research_docs").strip()

    # ── Langfuse observability ─────────────────────────────────────────────────
    LANGFUSE_PUBLIC_KEY: str = _clean_str(os.getenv("LANGFUSE_PUBLIC_KEY", ""))
    LANGFUSE_SECRET_KEY: str = _clean_str(os.getenv("LANGFUSE_SECRET_KEY", ""))
    LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").strip()

    # ── Agent settings ─────────────────────────────────────────────────────────
    MAX_MESSAGES_BEFORE_COMPACTION: int = int(
        os.getenv("MAX_MESSAGES_BEFORE_COMPACTION", "10")
    )
    MESSAGES_TO_SUMMARISE: int = int(os.getenv("MESSAGES_TO_SUMMARISE", "8"))
    CHECKPOINT_DB_PATH: str = os.getenv("CHECKPOINT_DB_PATH", "checkpoints.sqlite").strip()

    # ── FastAPI ────────────────────────────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0").strip()
    PORT: int = int(os.getenv("PORT", "8000"))


cfg = Config()
