import json
import os
import time
import uuid
from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from openai import OpenAI

from app.core.db import get_db
from app.rag.embed import embed_text
from app.rag.retriever import retrieve_top_k, RetrievedChunk

router = APIRouter(tags=["rag"])

# ---- Config ----
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
RAG_MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "0.65"))
RAG_CONTEXT_MARGIN = float(os.getenv("RAG_CONTEXT_MARGIN", "0.08"))

# -------------------------
# API Models
# -------------------------
class MemoRequest(BaseModel):
    topic: str = Field(..., min_length=1, description="Decision topic or question")
    k: int = Field(8, ge=1, le=20)
    include_hits: bool = False

class CitationRef(BaseModel):
    chunk_id: int

class SectionWithCitations(BaseModel):
    text: str
    citations: list[CitationRef] = Field(default_factory=list)

class OptionItem(BaseModel):
    option: str
    tradeoffs: str
    citations: list[CitationRef] = Field(default_factory=list)

class RiskItem(BaseModel):
    risk: str
    mitigation: str
    citations: list[CitationRef] = Field(default_factory=list)

class QuestionItem(BaseModel):
    question: str
    citations: list[CitationRef] = Field(default_factory=list)

class ChangeMindItem(BaseModel):
    item: str
    citations: list[CitationRef] = Field(default_factory=list)

class Citation(BaseModel):
    chunk_id: int
    path: str
    title: Optional[str] = None
    heading: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    distance: Optional[float] = None

class MemoResponse(BaseModel):
    # Memo sections
    tldr: SectionWithCitations
    options_tradeoffs: list[OptionItem]
    risks_mitigations: list[RiskItem]
    open_questions: list[QuestionItem]
    what_would_change_my_mind: list[ChangeMindItem]

    # Product/debug fields
    citations: list[Citation]  # flattened, mapped to full objects
    best_distance: Optional[float] = None
    weak_match: bool = False
    hits: Optional[list[dict[str, Any]]] = None
    request_id: str

# -------------------------
# Helpers
# -------------------------
def _format_sources(hits: list[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for i, h in enumerate(hits, start=1):
        blocks.append(
            f"[SOURCE {i} | chunk_id={h.chunk_id} | path={h.path} | heading={h.heading or ''}]\n"
            f"{h.content}\n"
        )
    return "\n".join(blocks)

def _extract_usage(resp: Any) -> dict[str, Any]:
    usage = {}
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

def _safe_json_load(text: str) -> dict[str, Any]:
    """
    Strict-but-forgiving JSON parse: returns {} on failure.
    """
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _normalize_memo_json(data: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure all required keys exist with correct-ish types so response_model won’t explode.
    """
    def _list(x): return x if isinstance(x, list) else []
    def _str(x): return x if isinstance(x, str) else ""

    out = {}
    out["tldr"] = {
        "text": _str((data.get("tldr") or {}).get("text") if isinstance(data.get("tldr"), dict) else ""),
        "citations": _list((data.get("tldr") or {}).get("citations") if isinstance(data.get("tldr"), dict) else []),
    }
    out["options_tradeoffs"] = _list(data.get("options_tradeoffs"))
    out["risks_mitigations"] = _list(data.get("risks_mitigations"))
    out["open_questions"] = _list(data.get("open_questions"))
    out["what_would_change_my_mind"] = _list(data.get("what_would_change_my_mind"))
    return out

def _collect_chunk_ids(memo_json: dict[str, Any]) -> list[int]:
    """
    Walk the memo JSON and collect chunk_ids from all citations fields.
    """
    ids: list[int] = []

    def pull_citations(obj: Any):
        if isinstance(obj, dict):
            # If this dict has "citations": [...]
            cits = obj.get("citations")
            if isinstance(cits, list):
                for c in cits:
                    if isinstance(c, dict) and "chunk_id" in c:
                        try:
                            ids.append(int(c["chunk_id"]))
                        except Exception:
                            pass
            # Recurse
            for v in obj.values():
                pull_citations(v)
        elif isinstance(obj, list):
            for it in obj:
                pull_citations(it)

    pull_citations(memo_json)
    # unique preserve order
    seen = set()
    uniq = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq

def _llm_memo_with_citations(client: OpenAI, topic: str, context_hits: list[RetrievedChunk]) -> tuple[dict[str, Any], dict[str, Any]]:
    sources_text = _format_sources(context_hits)

    system = (
        "You generate a decision memo using ONLY the provided SOURCES.\n"
        "Rules:\n"
        "1) If the SOURCES do not contain enough evidence for a claim, say so in the text and use empty citations for that claim.\n"
        "2) Return STRICT JSON only (no markdown, no extra text).\n"
        "3) Every citations list must contain objects like {\"chunk_id\": <int>} referencing ONLY chunk_ids present in SOURCES.\n"
        "4) Keep sections concise and practical.\n"
        "Output JSON schema:\n"
        "{\n"
        "  \"tldr\": {\"text\": string, \"citations\": [{\"chunk_id\": int}]},\n"
        "  \"options_tradeoffs\": [{\"option\": string, \"tradeoffs\": string, \"citations\": [{\"chunk_id\": int}]}],\n"
        "  \"risks_mitigations\": [{\"risk\": string, \"mitigation\": string, \"citations\": [{\"chunk_id\": int}]}],\n"
        "  \"open_questions\": [{\"question\": string, \"citations\": [{\"chunk_id\": int}]}],\n"
        "  \"what_would_change_my_mind\": [{\"item\": string, \"citations\": [{\"chunk_id\": int}]}]\n"
        "}\n"
    )

    user = (
        f"Decision topic/question: {topic}\n\n"
        f"SOURCES:\n{sources_text}\n\n"
        "Return JSON now."
    )

    resp = client.responses.create(
        model=CHAT_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    text = (resp.output_text or "").strip()
    data = _safe_json_load(text)
    return data, _extract_usage(resp)

# -------------------------
# Endpoint
# -------------------------
@router.post("/memo", response_model=MemoResponse)
def memo(req: MemoRequest, db: Session = Depends(get_db)):
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())

    topic = req.topic.strip()

    # ---- Embedding + Retrieval ----
    t_embed0 = time.perf_counter()
    vec_literal = embed_text(topic)
    t_embed1 = time.perf_counter()

    t_ret0 = time.perf_counter()
    hits = retrieve_top_k(db, query_embedding=vec_literal, k=req.k)
    t_ret1 = time.perf_counter()

    best_distance = hits[0].distance if hits and hits[0].distance is not None else None
    weak_match = (best_distance is None) or (best_distance > RAG_MAX_DISTANCE)

    # ---- Gate: insufficient evidence ----
    if not hits or best_distance is None or best_distance > RAG_MAX_DISTANCE:
        print({
            "event": "memo",
            "request_id": request_id,
            "best_distance": best_distance,
            "weak_match": True,
            "returned_hits": len(hits),
            "filtered_hits": 0,
            "embed_ms": round((t_embed1 - t_embed0) * 1000, 2),
            "retrieval_ms": round((t_ret1 - t_ret0) * 1000, 2),
            "total_ms": round((time.perf_counter() - t0) * 1000, 2),
        })

        # Minimal “honest” memo structure with no real claims.
        empty = {
            "tldr": SectionWithCitations(text="I don’t have enough evidence in the indexed documents to write a grounded decision memo on this topic.", citations=[]),
            "options_tradeoffs": [],
            "risks_mitigations": [],
            "open_questions": [{"question": "Which documents should be added to support this decision?", "citations": []}],
            "what_would_change_my_mind": [{"item": "Add relevant sources and re-run the memo.", "citations": []}],
        }

        return MemoResponse(
            **empty,
            citations=[Citation(**hits[0].__dict__)] if hits else [],
            best_distance=best_distance,
            weak_match=True,
            hits=[h.__dict__ for h in hits] if req.include_hits else None,
            request_id=request_id,
        )

    # ---- Context filter (best + margin) ----
    filtered_hits = [
        h for h in hits
        if h.distance is not None and h.distance <= (best_distance + RAG_CONTEXT_MARGIN)
    ]
    if len(filtered_hits) < 2:
        filtered_hits = hits[:2]

    # ---- LLM Memo ----
    client = OpenAI()
    t_llm0 = time.perf_counter()
    raw_json, usage = _llm_memo_with_citations(client, topic, filtered_hits)
    t_llm1 = time.perf_counter()

    memo_json = _normalize_memo_json(raw_json)

    # ---- Map chunk_id citations back to full citation objects (flattened) ----
    by_id = {h.chunk_id: h for h in hits}
    cited_ids = _collect_chunk_ids(memo_json)

    full_citations: list[Citation] = []
    for cid in cited_ids:
        if cid in by_id:
            full_citations.append(Citation(**by_id[cid].__dict__))

    # If model cited nothing, at least cite the top hit (so the client can trace provenance)
    if not full_citations and hits:
        full_citations = [Citation(**hits[0].__dict__)]

    print({
        "event": "memo",
        "request_id": request_id,
        "best_distance": best_distance,
        "weak_match": False,
        "returned_hits": len(hits),
        "filtered_hits": len(filtered_hits),
        "paths": list({h.path for h in filtered_hits}),
        "embed_ms": round((t_embed1 - t_embed0) * 1000, 2),
        "retrieval_ms": round((t_ret1 - t_ret0) * 1000, 2),
        "llm_ms": round((t_llm1 - t_llm0) * 1000, 2),
        "total_ms": round((time.perf_counter() - t0) * 1000, 2),
        "usage": usage,
    })

    return MemoResponse(
        tldr=memo_json["tldr"],
        options_tradeoffs=memo_json["options_tradeoffs"],
        risks_mitigations=memo_json["risks_mitigations"],
        open_questions=memo_json["open_questions"],
        what_would_change_my_mind=memo_json["what_would_change_my_mind"],
        citations=full_citations,
        best_distance=best_distance,
        weak_match=False,
        hits=[h.__dict__ for h in hits] if req.include_hits else None,
        request_id=request_id,
    )
