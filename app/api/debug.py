from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.rag.embed import embed_text
from app.rag.retriever import retrieve_top_k

router = APIRouter(prefix="/debug", tags=["debug"])

@router.get("/retrieve")
def debug_retrieve(
    q: str = Query(..., min_length=1),
    k: int = Query(6, ge=1, le=20),
    db: Session = Depends(get_db),
):
    vec_literal = embed_text(q)
    hits = retrieve_top_k(db, query_embedding=vec_literal, k=k)
    return {}