from openai import OpenAI
from app.core.settings import settings

client = OpenAI(api_key=settings.openai_api_key)

EMBED_MODEL = "text-embedding-3-small"

def embed_text(text: str) -> list[float]:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
    return resp.data[0].embedding
