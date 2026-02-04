import os
import time
import uuid
from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session
from openai import OpenAI

from app.rag.gating import gate_hits
from app.core.db import get_db
from app.core.llm_helpers import call_with_retry, extract_usage, format_sources, safe_json_load
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
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(6, ge=1, le=20)
    include_hits: bool = False  # handy during dev

class Citation(BaseModel):
    chunk_id: int
    path: str
    title: Optional[str] = None
    heading: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    distance: Optional[float] = None

class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    best_distance: Optional[float] = None
    weak_match: bool = False
    hits: Optional[list[dict[str, Any]]] = None
    request_id: str

# -------------------------
# LLM Output Models (Hardening #2)
# -------------------------
class LlmCitation(BaseModel):
    chunk_id: int

class LlmAsk(BaseModel):
    answer: str = ""
    citations: list[LlmCitation] = Field(default_factory=list)

# -------------------------
# LLM Call
# -------------------------
def _llm_answer_with_citations(client: OpenAI, question: str, context_hits: list[RetrievedChunk]) -> tuple[dict[str, Any], dict[str, Any]]:
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

    # ---- Embedding + Retrieval ----
    t_embed0 = time.perf_counter()
    vec_literal = embed_text(q)
    t_embed1 = time.perf_counter()

    t_ret0 = time.perf_counter()
    hits = retrieve_top_k(db, query_embedding=vec_literal, k=req.k)
    t_ret1 = time.perf_counter()

    gate = gate_hits(
    hits,
    max_distance=RAG_MAX_DISTANCE,
    weak_band=float(os.getenv("RAG_WEAK_BAND", "0.05")),
    min_gap=float(os.getenv("RAG_MIN_GAP", "0.03")),
    )
    best_distance = gate.best_distance
    weak_match = gate.decision != "confident"


    # Safer: ignore None distances
    if gate.decision == "insufficient":
        top_cit = [Citation(**hits[0].__dict__)] if hits else []
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
    valid_hits = [h for h in hits if h.distance is not None]


    # ---- Gate: insufficient evidence ----
    if not valid_hits or best_distance is None or best_distance > RAG_MAX_DISTANCE:
        print({
            "event": "ask",
            "request_id": request_id,
            "best_distance": best_distance,
            "weak_match": True,
            "returned_hits": len(hits),
            "valid_hits": len(valid_hits),
            "filtered_hits": 0,
            "embed_ms": round((t_embed1 - t_embed0) * 1000, 2),
            "retrieval_ms": round((t_ret1 - t_ret0) * 1000, 2),
            "total_ms": round((time.perf_counter() - t0) * 1000, 2),
        })

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
    filtered_hits = [
        h for h in valid_hits
        if h.distance is not None and h.distance <= (best_distance + RAG_CONTEXT_MARGIN)
    ]
    if len(filtered_hits) < 2:
        filtered_hits = valid_hits[:2]

    # Hardening #1: citations must come ONLY from filtered context
    allowed_by_id = {h.chunk_id: h for h in filtered_hits}

    # ---- LLM answer ----
    client = OpenAI()
    t_llm0 = time.perf_counter()
    raw_json, usage = _llm_answer_with_citations(client, q, filtered_hits)
    t_llm1 = time.perf_counter()

    # Hardening #2: strict validate LLM output, fail closed
    try:
        llm_obj = LlmAsk.model_validate(raw_json)
    except ValidationError:
        print({
            "event": "ask_llm_invalid_json",
            "request_id": request_id,
            "best_distance": best_distance,
            "returned_hits": len(hits),
            "filtered_hits": len(filtered_hits),
        })

        top_cit = [Citation(**filtered_hits[0].__dict__)] if filtered_hits else []
        return AskResponse(
            answer="I couldn’t produce a valid grounded answer from the sources (LLM formatting error).",
            citations=top_cit,
            best_distance=best_distance,
            weak_match=False,
            hits=[h.__dict__ for h in hits] if req.include_hits else None,
            request_id=request_id,
        )

    answer = (llm_obj.answer or "").strip()

    # Map citations -> full objects, but ONLY if allowed
    citations: list[Citation] = []
    for c in llm_obj.citations:
        if c.chunk_id in allowed_by_id:
            citations.append(Citation(**allowed_by_id[c.chunk_id].__dict__))

    # Hardening #3: enforce citations on confident answers
    if not citations and filtered_hits:
        citations = [Citation(**filtered_hits[0].__dict__)]

    if not answer:
        answer = "I don’t have enough evidence in the indexed documents to answer that."

    print({
        "event": "ask",
        "request_id": request_id,
        "best_distance": best_distance,
        "weak_match": False,
        "returned_hits": len(hits),
        "valid_hits": len(valid_hits),
        "filtered_hits": len(filtered_hits),
        "paths": list({h.path for h in filtered_hits}),
        "embed_ms": round((t_embed1 - t_embed0) * 1000, 2),
        "retrieval_ms": round((t_ret1 - t_ret0) * 1000, 2),
        "llm_ms": round((t_llm1 - t_llm0) * 1000, 2),
        "total_ms": round((time.perf_counter() - t0) * 1000, 2),
        "usage": usage,
    })

    return AskResponse(
        answer=answer,
        citations=citations,
        best_distance=best_distance,
        weak_match=False,
        hits=[h.__dict__ for h in hits] if req.include_hits else None,
        request_id=request_id,
    )
