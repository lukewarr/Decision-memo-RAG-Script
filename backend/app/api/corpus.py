from fastapi import APIRouter
from app.services.ingest_service import get_corpus_stats

router = APIRouter(tags=["corpus"])

@router.get("/corpus/stats")
def corpus_stats():
    return get_corpus_stats()
