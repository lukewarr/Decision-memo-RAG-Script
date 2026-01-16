from fastapi import FastAPI
from app.api.ask import router as ask_router

app = FastAPI()
app.include_router(ask_router)