import os
import time
import uuid
from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from sqlalchemy.orm import Session
from openai import OpenAI

from app.rag.gating import gate_hits
from app.core.db import get_db
from app.core.llm_helpers import call_with_retry, extract_usage, format_sources, safe_json_load
from app.core.cache import TTLCache, stable_key
from app.core.metrics import estimate_cost_usd
from app.rag.embed import embed_text
from app.rag.retriever import retrieve_top_k, RetrievedChunk
from app.schemas.rag import AskResponse, Hit
from app.services.rag_format import build_citations_from_hits


router = APIRouter(tags=["rag"])

# ---- Config ----
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
RAG_MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "0.65"))
RAG_CONTEXT_MARGIN = float(os.getenv("RAG_CONTEXT_MARGIN", "0.08"))

EMBED_TTL_S = int(os.getenv("EMBED_CACHE_TTL_S", "600"))
RETR_TTL_S = int(os.getenv("RETR_CACHE_TTL_S", "300"))

_embed_cache = TTLCache(ttl_seconds=EMBED_TTL_S, max_items=2048)
_retr_cache = TTLCache(ttl_seconds=RETR_TTL_S, max_items=2048)

# -------------------------
# API Models
# -------------------------

class Citation(BaseModel):
    model_config = ConfigDict(extra="ignore")  # <-- KEY: ignore extra keys in __dict__

    # Make path non-fatal if missing in some flows
    path: str = ""

    title: Optional[str] = None
    heading: Optional[str] = None
    chunk_id: Optional[int] = None

    # Add fields you already use everywhere
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    distance: Optional[float] = None

    # Nice for UI
    excerpt: Optional[str] = None
    
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(6, ge=1, le=20)
    include_hits: bool = False  # handy during dev

# -------------------------
# LLM Output Models
# -------------------------
class LlmCitation(BaseModel):
    chunk_id: int

class LlmAsk(BaseModel):
    answer: str = ""
    citations: list[LlmCitation] = Field(default_factory=list)

# -------------------------
# LLM Call
# -------------------------
def _llm_answer_with_citations(
    client: OpenAI,
    question: str,
    context_hits: list[RetrievedChunk],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sources_text = format_sources(context_hits)

    system = (
        "You are a careful assistant answering ONLY from the provided SOURCES.\n"
        "Rules:\n"
        "1) If the answer is not clearly supported by the SOURCES, say you have insufficient evidence.\n"
        "2) Return STRICT JSON only (no markdown, no extra text).\n"
        "3) Citations must be a list of objects like {\"chunk_id\": int}.\n"
        "4) You may ONLY cite chunk_id values that appear in the SOURCES.\n"
        "Output schema: {\"answer\": string, \"citations\": [{\"chunk_id\": int}]}\n"
        "5) If insufficient evidence, set citations to an empty list.\n"
    )

    user = (
        f"Question: {question}\n\n"
        f"SOURCES:\n{sources_text}\n\n"
        "Return JSON now."
    )

    def _call():
        return client.responses.create(
            model=CHAT_MODEL,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )

    resp = call_with_retry(_call)
    text = (resp.output_text or "").strip()
    return safe_json_load(text), extract_usage(resp)

# -------------------------
# Endpoint
# -------------------------
@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, db: Session = Depends(get_db)):
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())

    q = req.question.strip()

    # Guard: very short queries are often ambiguous in embedding space
    MIN_CHARS = int(os.getenv("RAG_MIN_QUERY_CHARS", "6"))
    if len(q) < MIN_CHARS:
        return AskResponse(
            answer=(
                "That query is too short / ambiguous for semantic search. "
                "Try adding a few more words (e.g., '429 rate limit during demo timeline')."
            ),
            citations=[],
            best_distance=None,
            weak_match=True,
            hits=None if not req.include_hits else [],
            request_id=request_id,
        )

    # ---- Embedding (cached) ----
    t_embed0 = time.perf_counter()
    EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    embed_key = stable_key("embed", EMBED_MODEL,q)
    vec_literal = _embed_cache.get(embed_key)
    embed_cache_hit = vec_literal is not None
    if vec_literal is None:
        vec_literal = embed_text(q)
        _embed_cache.set(embed_key, vec_literal)
    t_embed1 = time.perf_counter()

    # ---- Retrieval (cached) ----
    t_ret0 = time.perf_counter()
    retr_key = stable_key("retr", vec_literal, req.k)
    hits = _retr_cache.get(retr_key)
    retr_cache_hit = hits is not None
    if hits is None:
        hits = retrieve_top_k(db, query_embedding=vec_literal, k=req.k)
        _retr_cache.set(retr_key, hits)
    t_ret1 = time.perf_counter()

    # ---- Gate ----
    gate = gate_hits(
        hits,
        max_distance=RAG_MAX_DISTANCE,
        weak_band=float(os.getenv("RAG_WEAK_BAND", "0.05")),
        min_gap=float(os.getenv("RAG_MIN_GAP", "0.03")),
    )
    best_distance = gate.best_distance
    weak_match = gate.decision != "confident"

    valid_hits = [h for h in hits if getattr(h, "distance", None) is not None]
    is_weak = (gate.decision == "weak")

    if gate.decision == "insufficient":
        print({
            "event": "ask",
            "request_id": request_id,
            "best_distance": best_distance,
            "weak_match": True,
            "gate_decision": gate.decision,
            "gate_reason": gate.reason,
            "gate_gap": gate.gap,
            "returned_hits": len(hits),
            "valid_hits": len(valid_hits),
            "filtered_hits": 0,
            "embed_ms": round((t_embed1 - t_embed0) * 1000, 2),
            "retrieval_ms": round((t_ret1 - t_ret0) * 1000, 2),
            "embed_cache_hit": embed_cache_hit,
            "retr_cache_hit": retr_cache_hit,
            "total_ms": round((time.perf_counter() - t0) * 1000, 2),
        })

        top_cit = [Citation(**valid_hits[0].__dict__)] if valid_hits else []
        return AskResponse(
            answer=(
                "I don’t have enough evidence in the indexed documents to answer that confidently.\n"
                "If you can, point me to the doc/section that covers it (or add it to the corpus)."
            ),
            citations=top_cit,
            best_distance=best_distance,
            weak_match=True,
            hits=[h.__dict__ for h in hits] if req.include_hits else None,
            request_id=request_id,
        )

    if best_distance is None:
        top_cit = [Citation(**valid_hits[0].__dict__)] if valid_hits else []
        return AskResponse(
            answer="I don’t have enough evidence in the indexed documents to answer that confidently.",
            citations=top_cit,
            best_distance=best_distance,
            weak_match=True,
            hits=[h.__dict__ for h in hits] if req.include_hits else None,
            request_id=request_id,
        )

    # ---- Context filter (best + margin) ----
    filtered_hits = [h for h in valid_hits if h.distance <= (best_distance + RAG_CONTEXT_MARGIN)]
    if len(filtered_hits) < 2:
        filtered_hits = valid_hits[:2]

    allowed_by_id = {h.chunk_id: h for h in filtered_hits}

    # ---- LLM answer ----
    client = OpenAI()
    t_llm0 = time.perf_counter()
    raw_json, usage = _llm_answer_with_citations(client, q, filtered_hits)
    t_llm1 = time.perf_counter()

    try:
        llm_obj = LlmAsk.model_validate(raw_json)
    except ValidationError:
        cost_usd = estimate_cost_usd(CHAT_MODEL, usage)

        print({
            "event": "ask_llm_invalid_json",
            "request_id": request_id,
            "best_distance": best_distance,
            "weak_match": weak_match,
            "gate_decision": gate.decision,
            "gate_reason": gate.reason,
            "gate_gap": gate.gap,
            "returned_hits": len(hits),
            "filtered_hits": len(filtered_hits),
            "llm_ms": round((t_llm1 - t_llm0) * 1000, 2),
            "usage": usage,
            "cost_usd": round(cost_usd, 6),
        })

        top_cit = [Citation(**filtered_hits[0].__dict__)] if filtered_hits else []
        return AskResponse(
            answer="I couldn’t produce a valid grounded answer from the sources (LLM formatting error).",
            citations=top_cit,
            best_distance=best_distance,
            weak_match=weak_match,
            hits=[h.__dict__ for h in hits] if req.include_hits else None,
            request_id=request_id,
        )

    answer = (llm_obj.answer or "").strip()

    citations: list[Citation] = []
    for c in llm_obj.citations:
        if c.chunk_id in allowed_by_id:
            citations.append(Citation(**allowed_by_id[c.chunk_id].__dict__))

    if not citations and filtered_hits:
        citations = [Citation(**filtered_hits[0].__dict__)]

    if not answer:
        answer = "I don’t have enough evidence in the indexed documents to answer that."

    if is_weak:
        answer = "Note: evidence is partial / near-threshold.\n" + answer

    cost_usd = estimate_cost_usd(CHAT_MODEL, usage)

    print({
        "event": "ask",
        "request_id": request_id,
        "best_distance": best_distance,
        "weak_match": weak_match,
        "gate_decision": gate.decision,
        "gate_reason": gate.reason,
        "gate_gap": gate.gap,
        "returned_hits": len(hits),
        "valid_hits": len(valid_hits),
        "filtered_hits": len(filtered_hits),
        "paths": list({h.path for h in filtered_hits}),
        "embed_ms": round((t_embed1 - t_embed0) * 1000, 2),
        "retrieval_ms": round((t_ret1 - t_ret0) * 1000, 2),
        "llm_ms": round((t_llm1 - t_llm0) * 1000, 2),
        "embed_cache_hit": embed_cache_hit,
        "retr_cache_hit": retr_cache_hit,
        "total_ms": round((time.perf_counter() - t0) * 1000, 2),
        "usage": usage,
        "cost_usd": round(cost_usd, 6),
    })
    hits_dicts = [h.__dict__ for h in hits]
    citations = build_citations_from_hits(hits)
    gating = gate.decision  # "confident" | "weak" | "insufficient"


    return AskResponse(
    answer=answer,
    gating=gating,
    best_distance=best_distance,
    weak_match=bool(weak_match),
    citations=citations,
    hits=[Hit(**h) for h in hits] if req.include_hits else None,
)
