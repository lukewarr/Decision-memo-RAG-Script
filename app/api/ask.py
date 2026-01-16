from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.db import get_db
from app.rag.embed import embed_text
from app.rag.retriever import retrieve_top_k
from app.rag.retriever import to_pgvector

router = APIRouter()

@router.post("/ask")
def ask(payload: dict, db: Session = Depends(get_db)):
    question = payload["question"]
    k = int(payload.get("k", 6))

    embedding = embed_text(question)
    vec_literal = to_pgvector(embedding)

    hits = retrieve_top_k(db, query_embedding=vec_literal, k=k)
    return {"hits": [h.__dict__ for h in hits]
    }