import json
import os
import uuid
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from openai import OpenAI

from app.core.db import get_db
from app.rag.embed import embed_text
from app.rag.retriever import retrieve_top_k, RetrievedChunk

router = APIRouter(tags=["rag"])

# Models (explicit schema prevents Swagger "additionalProp1" confusion)
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(6, ge=1, le=20)
    include_hits: bool = False  # handy during dev

class Citation(BaseModel):
    chunk_id: int
    path: str
    title: str | None = None
    heading: str | None = None
    start_char: int | None = None
    end_char: int | None = None
    distance: float | None = None

class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    hits: list[dict[str, Any]] | None = None
    request_id: str

# ---- Config ----
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
# This is a *starting* heuristic. You’ll tune later by looking at debug/retrieve.
MAX_DISTANCE_FOR_CONFIDENT_ANSWER = float(os.getenv("RAG_MAX_DISTANCE", "0.80"))

def _format_sources(hits: list[RetrievedChunk]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(
            f"[SOURCE {i} | chunk_id={h.chunk_id} | {h.path} | {h.heading or ''}]\n"
            f"{h.content}\n"
        )
    return "\n".join(blocks)

def _llm_answer_with_citations(client: OpenAI, question: str, hits: list[RetrievedChunk]) -> dict[str, Any]:
    sources_text = _format_sources(hits)

    system = (
        "You are a careful assistant answering ONLY from the provided SOURCES.\n"
        "Rules:\n"
        "1) If the answer is not in the SOURCES, say you have insufficient evidence.\n"
        "2) Return STRICT JSON only (no markdown, no extra text).\n"
        "3) Citations must be a list of chunk_id integers that support the answer.\n"
        'Output schema: {"answer": string, "citations": [{"chunk_id": int}]}'
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

    text = resp.output_text.strip()

    # Parse JSON robustly (models sometimes add whitespace/newlines)
    try:
        return json.loads(text)
    except Exception:
        # Fallback: treat model output as plain answer with no citations
        return {"answer": text, "citations": []}

@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, db: Session = Depends(get_db)):
    request_id = str(uuid.uuid4())

    q = req.question.strip()
    vec_literal = embed_text(q)
    hits = retrieve_top_k(db, query_embedding=vec_literal, k=req.k)

    # --- Insufficient evidence gate (v1 heuristic) ---
    if not hits:
        return AskResponse(
            answer="I don’t have enough evidence in the indexed documents to answer that.",
            citations=[],
            hits=[h.__dict__ for h in hits] if req.include_hits else None,
            request_id=request_id,
        )

    best_distance = hits[0].distance
    if best_distance is None or best_distance > MAX_DISTANCE_FOR_CONFIDENT_ANSWER:
        return AskResponse(
            answer="I don’t have enough evidence in the indexed documents to answer that confidently.",
            citations=[Citation(**h.__dict__) for h in hits[:3]],  # show what we found
            hits=[h.__dict__ for h in hits] if req.include_hits else None,
            request_id=request_id,
        )

    # --- LLM answer ---
    client = OpenAI()
    llm_json = _llm_answer_with_citations(client, q, hits)

    answer = (llm_json.get("answer") or "").strip()
    cited = llm_json.get("citations") or []

    # Map cited chunk_ids to full citation objects from hits
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
    if not citations:
        citations = [Citation(**hits[0].__dict__)]

    return AskResponse(
        answer=answer or "I don’t have enough evidence in the indexed documents to answer that.",
        citations=citations,
        hits=[h.__dict__ for h in hits] if req.include_hits else None,
        request_id=request_id,
    )
