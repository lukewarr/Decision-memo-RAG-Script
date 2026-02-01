import json
import os
import random
import time
import uuid
from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, ValidationError
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

    # Flattened, mapped to full objects (ONLY from filtered context)
    citations: list[Citation]

    # Product/debug fields
    best_distance: Optional[float] = None
    weak_match: bool = False
    hits: Optional[list[dict[str, Any]]] = None
    request_id: str

# -------------------------
# LLM Output Models (Hardening #2)
# -------------------------
class LlmCitation(BaseModel):
    chunk_id: int

class LlmTLDR(BaseModel):
    text: str = ""
    citations: list[LlmCitation] = Field(default_factory=list)

class LlmOption(BaseModel):
    option: str = ""
    tradeoffs: str = ""
    citations: list[LlmCitation] = Field(default_factory=list)

class LlmRisk(BaseModel):
    risk: str = ""
    mitigation: str = ""
    citations: list[LlmCitation] = Field(default_factory=list)

class LlmQuestion(BaseModel):
    question: str = ""
    citations: list[LlmCitation] = Field(default_factory=list)

class LlmChangeMind(BaseModel):
    item: str = ""
    citations: list[LlmCitation] = Field(default_factory=list)

class LlmMemo(BaseModel):
    tldr: LlmTLDR = Field(default_factory=LlmTLDR)
    options_tradeoffs: list[LlmOption] = Field(default_factory=list)
    risks_mitigations: list[LlmRisk] = Field(default_factory=list)
    open_questions: list[LlmQuestion] = Field(default_factory=list)
    what_would_change_my_mind: list[LlmChangeMind] = Field(default_factory=list)

# -------------------------
# Retry wrapper (Hardening #4)
# -------------------------
RETRIABLE_STATUS = {429, 500, 502, 503, 504}

def _http_status_from_exc(e: Exception) -> Optional[int]:
    # Best-effort; SDK exception shapes can vary across versions
    resp = getattr(e, "response", None)
    if resp is not None:
        return getattr(resp, "status_code", None)
    return None

def call_with_retry(fn, *, max_retries: int = 5, base_delay: float = 0.4, max_delay: float = 6.0):
    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            status = _http_status_from_exc(e)
            if status not in RETRIABLE_STATUS or attempt == max_retries:
                raise

            # exponential backoff + jitter (0.5x–1.5x)
            delay = min(max_delay, base_delay * (2 ** attempt))
            delay *= (0.5 + random.random())
            time.sleep(delay)
    raise last_err  # should be unreachable

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
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _mark_if_uncited(text: str, citations: list[LlmCitation]) -> str:
    if citations:
        return text
    if "insufficient evidence" in (text or "").lower():
        return text
    # keep short & explicit
    return f"{text} (insufficient evidence in sources)".strip()

def _collect_chunk_ids_from_llm(memo: LlmMemo) -> list[int]:
    ids: list[int] = []

    # TLDR
    ids += [c.chunk_id for c in memo.tldr.citations]

    # Lists
    for opt in memo.options_tradeoffs:
        ids += [c.chunk_id for c in opt.citations]
    for r in memo.risks_mitigations:
        ids += [c.chunk_id for c in r.citations]
    for q in memo.open_questions:
        ids += [c.chunk_id for c in q.citations]
    for w in memo.what_would_change_my_mind:
        ids += [c.chunk_id for c in w.citations]

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
        "1) If SOURCES do not support a claim, write it as uncertain and set citations to an empty list.\n"
        "2) Return STRICT JSON only (no markdown, no extra text).\n"
        "3) Every citations list must contain objects like {\"chunk_id\": <int>}.\n"
        "4) You may ONLY cite chunk_id values that appear in SOURCES.\n"
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
    return _safe_json_load(text), _extract_usage(resp)

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

    # Safer: ignore None distances
    valid_hits = [h for h in hits if h.distance is not None]
    best_distance = valid_hits[0].distance if valid_hits else None
    weak_match = (best_distance is None) or (best_distance > RAG_MAX_DISTANCE)

    # ---- Gate: insufficient evidence ----
    if not valid_hits or best_distance is None or best_distance > RAG_MAX_DISTANCE:
        print({
            "event": "memo",
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

        # Honest empty memo with optional top citation for debugging
        top_cit = [Citation(**valid_hits[0].__dict__)] if valid_hits else []

        return MemoResponse(
            tldr=SectionWithCitations(
                text="I don’t have enough evidence in the indexed documents to write a grounded decision memo on this topic.",
                citations=[],
            ),
            options_tradeoffs=[],
            risks_mitigations=[],
            open_questions=[QuestionItem(question="Which documents should be added to support this decision?", citations=[])],
            what_would_change_my_mind=[ChangeMindItem(item="Add relevant sources and re-run the memo.", citations=[])],
            citations=top_cit,
            best_distance=best_distance,
            weak_match=True,
            hits=[h.__dict__ for h in hits] if req.include_hits else None,
            request_id=request_id,
        )

    # ---- Context filter (best + margin) ----
    filtered_hits = [h for h in valid_hits if h.distance is not None and h.distance <= (best_distance + RAG_CONTEXT_MARGIN)]
    if len(filtered_hits) < 2:
        filtered_hits = valid_hits[:2]

    # ---- LLM Memo ----
    client = OpenAI()
    t_llm0 = time.perf_counter()
    raw_json, usage = _llm_memo_with_citations(client, topic, filtered_hits)
    t_llm1 = time.perf_counter()

    # Hardening #2: strict validate LLM output, fail closed
    try:
        memo_obj = LlmMemo.model_validate(raw_json)
    except ValidationError:
        print({
            "event": "memo_llm_invalid_json",
            "request_id": request_id,
            "best_distance": best_distance,
            "returned_hits": len(hits),
            "filtered_hits": len(filtered_hits),
        })

        # Fail closed: don’t ship a hallucination-shaped memo
        top_cit = [Citation(**filtered_hits[0].__dict__)] if filtered_hits else []
        return MemoResponse(
            tldr=SectionWithCitations(
                text="I couldn’t produce a valid grounded memo from the sources (LLM formatting error).",
                citations=[],
            ),
            options_tradeoffs=[],
            risks_mitigations=[],
            open_questions=[QuestionItem(question="Try again, or reduce k/context, or inspect /debug/retrieve.", citations=[])],
            what_would_change_my_mind=[ChangeMindItem(item="If a valid JSON memo is produced from the sources.", citations=[])],
            citations=top_cit,
            best_distance=best_distance,
            weak_match=False,
            hits=[h.__dict__ for h in hits] if req.include_hits else None,
            request_id=request_id,
        )

    # Hardening #3: enforce memo rules
    # A) TL;DR must have ≥1 citation
    if not memo_obj.tldr.citations and filtered_hits:
        memo_obj.tldr.citations = [LlmCitation(chunk_id=filtered_hits[0].chunk_id)]

    # B) Mark uncited items explicitly
    for item in memo_obj.open_questions:
        item.question = _mark_if_uncited(item.question, item.citations)
    for item in memo_obj.what_would_change_my_mind:
        item.item = _mark_if_uncited(item.item, item.citations)

    # Hardening #1: citations only from filtered context
    allowed_by_id = {h.chunk_id: h for h in filtered_hits}

    # Build flattened citation objects from memo citations (only allowed IDs)
    cited_ids = [cid for cid in _collect_chunk_ids_from_llm(memo_obj) if cid in allowed_by_id]
    full_citations: list[Citation] = [Citation(**allowed_by_id[cid].__dict__) for cid in cited_ids]

    # If somehow still empty, force top filtered citation
    if not full_citations and filtered_hits:
        full_citations = [Citation(**filtered_hits[0].__dict__)]

    print({
        "event": "memo",
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

    # Convert memo_obj -> API response models
    return MemoResponse(
        tldr=SectionWithCitations(
            text=memo_obj.tldr.text,
            citations=[CitationRef(chunk_id=c.chunk_id) for c in memo_obj.tldr.citations if c.chunk_id in allowed_by_id],
        ),
        options_tradeoffs=[
            OptionItem(
                option=o.option,
                tradeoffs=o.tradeoffs,
                citations=[CitationRef(chunk_id=c.chunk_id) for c in o.citations if c.chunk_id in allowed_by_id],
            )
            for o in memo_obj.options_tradeoffs
        ],
        risks_mitigations=[
            RiskItem(
                risk=r.risk,
                mitigation=r.mitigation,
                citations=[CitationRef(chunk_id=c.chunk_id) for c in r.citations if c.chunk_id in allowed_by_id],
            )
            for r in memo_obj.risks_mitigations
        ],
        open_questions=[
            QuestionItem(
                question=q.question,
                citations=[CitationRef(chunk_id=c.chunk_id) for c in q.citations if c.chunk_id in allowed_by_id],
            )
            for q in memo_obj.open_questions
        ],
        what_would_change_my_mind=[
            ChangeMindItem(
                item=w.item,
                citations=[CitationRef(chunk_id=c.chunk_id) for c in w.citations if c.chunk_id in allowed_by_id],
            )
            for w in memo_obj.what_would_change_my_mind
        ],
        citations=full_citations,
        best_distance=best_distance,
        weak_match=False,
        hits=[h.__dict__ for h in hits] if req.include_hits else None,
        request_id=request_id,
    )
