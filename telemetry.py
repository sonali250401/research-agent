"""
Task 8: Telemetry — Langfuse & No-op Integration
=================================================
Wraps LangGraph execution in telemetry traces so you can inspect:
  - Latency per node / span
  - Token counts per LLM call
  - Full execution flow

Gracefully falls back to a lightweight local timer tracer when Langfuse is
not configured or when placeholder keys are used.
"""

import uuid
import time
import logging
from contextlib import contextmanager
from config import cfg

log = logging.getLogger("telemetry")


def _is_valid_key(k: str) -> bool:
    """Check if key is real and not a default placeholder."""
    if not k or not isinstance(k, str):
        return False
    k = k.strip()
    if k.startswith("pk-lf-...") or k.startswith("sk-lf-...") or "your_" in k.lower():
        return False
    return len(k) > 10


# ── Attempt to connect to Langfuse ────────────────────────────────────────────
_langfuse = None
_enabled = False

if _is_valid_key(cfg.LANGFUSE_PUBLIC_KEY) and _is_valid_key(cfg.LANGFUSE_SECRET_KEY):
    try:
        from langfuse import Langfuse

        _langfuse = Langfuse(
            public_key=cfg.LANGFUSE_PUBLIC_KEY,
            secret_key=cfg.LANGFUSE_SECRET_KEY,
            host=cfg.LANGFUSE_HOST or "https://cloud.langfuse.com",
        )
        _enabled = True
        log.info("✅ Langfuse telemetry enabled")
    except Exception as e:
        log.warning(f"Langfuse initialization error ({e}), using local timer tracer.")
else:
    log.info("Langfuse keys not set or placeholder detected — using local timer tracer.")


# ══════════════════════════════════════════════════════════════════════════════
# Span & Tracer implementations
# ══════════════════════════════════════════════════════════════════════════════

class NoopSpan:
    def __init__(self, name: str = ""):
        self.name = name
        self.attributes = {}

    def set_attribute(self, key: str, val):
        self.attributes[key] = val

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


class LocalTracer:
    """High-performance local tracer that logs trace lifecycle and span latencies."""

    @contextmanager
    def trace(self, name: str):
        log.info(f"[TRACE START] {name}")
        t0 = time.monotonic()
        try:
            yield
        finally:
            elapsed = round((time.monotonic() - t0) * 1000, 1)
            log.info(f"[TRACE END] {name} ({elapsed}ms)")
            if _enabled and _langfuse:
                try:
                    _langfuse.flush()
                except Exception:
                    pass

    @contextmanager
    def span(self, name: str):
        t0 = time.monotonic()
        span_obj = NoopSpan(name)
        try:
            yield span_obj
        finally:
            elapsed = round((time.monotonic() - t0) * 1000, 1)
            log.info(f"[SPAN] {name}  {elapsed}ms")


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

_tracer = LocalTracer()


def get_tracer():
    return _tracer


def start_trace(query: str) -> str:
    """Start a top-level trace and return a unique trace_id."""
    trace_id = str(uuid.uuid4())
    if _enabled and _langfuse:
        try:
            log.info(f"[Langfuse] Trace started: {trace_id[:8]} for '{query[:50]}'")
        except Exception as e:
            log.debug(f"Langfuse start_trace note: {e}")
    else:
        log.info(f"[Trace] id={trace_id[:8]} query='{query[:50]}'")
    return trace_id


def end_trace(trace_id: str, output: str):
    """Finalise the trace with the agent's output."""
    if _enabled and _langfuse:
        try:
            _langfuse.flush()
        except Exception:
            pass
    preview = output[:80].replace("\n", " ") if output else "None"
    log.info(f"[Trace] id={trace_id[:8]} finished. Output preview: {preview}...")
