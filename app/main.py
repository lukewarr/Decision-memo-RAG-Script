from fastapi import FastAPI

from app.api.ask import router as ask_router
from app.api.debug import router as debug_router  # <-- add this

app = FastAPI()

app.include_router(ask_router)
app.include_router(debug_router)  # <-- add this
