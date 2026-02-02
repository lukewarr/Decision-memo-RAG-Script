# app/core/llm_helpers.py
import json
import random
import time
from typing import Any, Optional

RETRIABLE_STATUS = {429, 500, 502, 503, 504}

def http_status_from_exc(e: Exception) -> Optional[int]:
    """
    Best-effort extraction of HTTP status from OpenAI SDK exceptions.
    Shapes vary across versions.
    """
    resp = getattr(e, "response", None)
    if resp is not None:
        return getattr(resp, "status_code", None)
    return None

def call_with_retry(
    fn,
    *,
    max_retries: int = 5,
    base_delay: float = 0.4,
    max_delay: float = 6.0,
    retriable_statuses: set[int] = RETRIABLE_STATUS,
):
    """
    Exponential backoff + jitter for transient errors (429/5xx).
    """
    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            status = http_status_from_exc(e)
            if status not in retriable_statuses or attempt == max_retries:
                raise

            delay = min(max_delay, base_delay * (2 ** attempt))
            delay *= (0.5 + random.random())  # jitter 0.5x–1.5x
            time.sleep(delay)
    raise last_err  # should be unreachable

def safe_json_load(text: str) -> dict[str, Any]:
    """
    Strict-ish: returns {} on failure or non-dict JSON.
    """
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def extract_usage(resp: Any) -> dict[str, Any]:
    """
    Best-effort extraction of token usage from OpenAI Responses API.
    """
    usage: dict[str, Any] = {}
    try:
        u = getattr(resp, "usage", None)
        if u:
            if isinstance(u, dict):
                usage = u
            else:
                usage = {
                    k: getattr(u, k)
                    for k in ("input_tokens", "output_tokens", "total_tokens")
                    if getattr(u, k, None) is not None
                }
    except Exception:
        pass
    return usage

def format_sources(hits: list[Any]) -> str:
    """
    Formats retrieved chunks into a text block for the model.
    Expects each hit to have: chunk_id, path, heading, content.
    """
    blocks: list[str] = []
    for i, h in enumerate(hits, start=1):
        blocks.append(
            f"[SOURCE {i} | chunk_id={getattr(h, 'chunk_id', None)} | "
            f"path={getattr(h, 'path', '')} | heading={getattr(h, 'heading', '') or ''}]\n"
            f"{getattr(h, 'content', '')}\n"
        )
    return "\n".join(blocks)
