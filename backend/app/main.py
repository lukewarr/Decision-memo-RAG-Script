from fastapi import FastAPI

from app.api.ask import router as ask_router
from app.api.memo import router as memo_router
from app.api.debug import router as debug_router
from app.api.ingest import router as ingest_router
from app.api.corpus import router as corpus_router

app = FastAPI()

app.include_router(ingest_router)
app.include_router(corpus_router)

app.include_router(ask_router)
app.include_router(memo_router)
app.include_router(debug_router) 
