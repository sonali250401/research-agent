# ─────────────────────────────────────────────────────────────────────────────
# ResearchBot Agent — Dockerfile
# Optimised for Hugging Face Spaces (Python 3.11, uvicorn, FastAPI)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Hugging Face Spaces requires port 7860
ENV PORT=7860
ENV HOST=0.0.0.0

# Non-root user (security best practice)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose
EXPOSE 7860

# Health-check (Render / HF Spaces compatible)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.getenv('PORT', '7860')}/health\")"

# Start the FastAPI server
CMD ["sh", "-c", "exec python -m uvicorn api:app --host \"${HOST:-0.0.0.0}\" --port \"${PORT:-7860}\" --workers 1"]
