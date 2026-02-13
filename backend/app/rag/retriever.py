from dataclasses import dataclass
from typing import List
from sqlalchemy import text
from sqlalchemy.orm import Session

@dataclass
class RetrievedChunk:
    chunk_id: int
    path: str
    title: str | None
    heading: str | None
    content: str
    start_char: int | None
    end_char: int | None
    distance: float

RETRIEVAL_SQL = """
select
    c.id as chunk_id,
    d.path,
    d.title,
    c.heading,
    c.content as content,
    c.start_char,
    c.end_char,
    (e.embedding <=> cast(:query_embedding as vector)) as distance
from embeddings e
join chunks c on c.id = e.chunk_id
join documents d on d.id = c.document_id
order by e.embedding <=> cast(:query_embedding as vector)
limit :k;
"""

def retrieve_top_k(db: Session, query_embedding: str, k: int = 6) -> List[RetrievedChunk]:
    """
    query_embedding MUST be a pgvector-compatible literal string, e.g. [0.1, 0.2, ...]
    (We'll generate this from embedding array returned by OpenAI)
    """

    rows = db.execute(
        text(RETRIEVAL_SQL),
        {"query_embedding": query_embedding, "k": k},
    ).mappings().all()

    return [
        RetrievedChunk(
            chunk_id =r["chunk_id"],
            path = r["path"],
            title = r["title"],
            heading = r["heading"],
            content=r["content"],
            start_char = r["start_char"],
            end_char = r["end_char"],
            distance = float(r["distance"]),
        )
        for r in rows
    ]

def to_pgvector(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"