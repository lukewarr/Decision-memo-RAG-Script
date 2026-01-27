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

# Models (explicit schema prevents Swagger "additionalProp1" confusion)
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

def _format_sources(hits: list[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for i, h in enumerate(hits, start=1):
        blocks.append(
            f"[SOURCE {i} | chunk_id={h.chunk_id} | path={h.path} | heading={h.heading or ''}]\n"
            f"{h.content}\n"
        )
    return "\n".join(blocks)

def _extract_usage(resp: Any) -> dict[str, Any]:
    """
    Best-effort extraction of token usage from OpenAI Responses API response.
    Shape can vary; we avoid hard failures.
    """
    usage = {}
    try:
        # Common: resp.usage exists
        u = getattr(resp, "usage", None)
        if u:
            # Some SDKs store usage as dict-like or object-like
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

def _llm_answer_with_citations(client: OpenAI, question: str, context_hits: list[RetrievedChunk]) -> tuple[dict[str, Any], dict[str, Any]]:
    sources_text = _format_sources(context_hits)

    system = (
        "You are a careful assistant answering ONLY from the provided SOURCES.\n"
        "Rules:\n"
        "1) If the answer is not clearly supported by the SOURCES, say you have insufficient evidence.\n"
        "2) Return STRICT JSON only (no markdown, no extra text).\n"
        "3) Citations must reference ONLY chunk_id values that appear in the SOURCES.\n"
        'Output schema: {"answer": string, "citations": [{"chunk_id": int}]}\n'
        "4) If insufficient evidence, set citations to an empty list.\n"
    )

    user = (
        f"Question: {question}\n\n"
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

    # Parse JSON robustly
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Model did not return a JSON object.")
        # Normalize fields
        data.setdefault("answer", "")
        data.setdefault("citations", [])
        if not isinstance(data.get("citations"), list):
            data["citations"] = []
        return data, _extract_usage(resp)
    except Exception:
        # Fallback: treat model output as plain answer with no citations
        return {"answer": text, "citations": []}, _extract_usage(resp)

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

    best_distance = hits[0].distance if hits and hits[0].distance is not None else None
    weak_match = (best_distance is None) or (best_distance > RAG_MAX_DISTANCE)

    # ---- Gate: insufficient evidence ----
    if not hits or best_distance is None or best_distance > RAG_MAX_DISTANCE:
        print({
            "event": "ask",
            "request_id": request_id,
            "best_distance": best_distance,
            "weak_match": True,
            "returned_hits": len(hits),
            "filtered_hits": 0,
            "embed_ms": round((t_embed1 - t_embed0) * 1000, 2),
            "retrieval_ms": round((t_ret1 - t_ret0) * 1000, 2),
            "total_ms": round((time.perf_counter() - t0) * 1000, 2),
        })

        return AskResponse(
            answer="I don’t have enough evidence in the indexed documents to answer that confidently.",
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
    # Always keep at least 2 chunks so the model has context
    if len(filtered_hits) < 2:
        filtered_hits = hits[:2]

    # ---- LLM Answer (grounded) ----
    client = OpenAI()
    t_llm0 = time.perf_counter()
    llm_json, usage = _llm_answer_with_citations(client, q, filtered_hits)
    t_llm1 = time.perf_counter()

    answer = (llm_json.get("answer") or "").strip()
    cited = llm_json.get("citations") or []

    # Map cited chunk_ids to full citation objects from retrieved hits (not just filtered)
    by_id = {h.chunk_id: h for h in hits}
    citations: list[Citation] = []

    for c in cited:
        try:
            cid = int(c.get("chunk_id"))
        except Exception:
            continue
        if cid in by_id:
            citations.append(Citation(**by_id[cid].__dict__))

    # If model forgot citations, at least attach top hit
    if not citations and hits:
        citations = [Citation(**hits[0].__dict__)]

    # If model answered but said "insufficient evidence", keep it honest
    if not answer:
        answer = "I don’t have enough evidence in the indexed documents to answer that."

    print({
        "event": "ask",
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

    return AskResponse(
        answer=answer,
        citations=citations,
        best_distance=best_distance,
        weak_match=False,
        hits=[h.__dict__ for h in hits] if req.include_hits else None,
        request_id=request_id,
    )
