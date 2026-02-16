from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from app.schemas.rag import Citation

def _safe_excerpt(text: Optional[str], limit: int = 280) -> Optional[str]:
    if not text:
        return None
    t = " ".join(text.split())  # collapse whitespace
    if len(t) <= limit:
        return t
    return t[:limit].rstrip() + "…"

def build_citations_from_hits(hits: List[Dict[str, Any]], max_items: int = 8) -> List[Citation]:
    """
    Dedupe citations so UI looks clean.
    Assumes hits may contain: path/title/heading/chunk_id/content/distance
    """
    out: List[Citation] = []
    seen: set[Tuple[str, Optional[int], Optional[str]]] = set()

    for h in hits:
        path = h.get("path") or h.get("source") or h.get("doc_path") or ""
        chunk_id = h.get("chunk_id")
        heading = h.get("heading")

        key = (path, chunk_id, heading)
        if path and key in seen:
            continue
        seen.add(key)

        out.append(
            Citation(
                path=path,
                title=h.get("title"),
                heading=heading,
                chunk_id=chunk_id,
                distance=h.get("distance") if isinstance(h.get("distance"), (int, float)) else h.get("score"),
                excerpt=_safe_excerpt(h.get("content")),
            )
        )
        if len(out) >= max_items:
            break

    return out
