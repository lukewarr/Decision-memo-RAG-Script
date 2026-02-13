from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from openai import OpenAI

# Optional PDF support (only if installed)
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


# -------------------------
# Config (same as script)
# -------------------------
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

MAX_CHARS = 3500
OVERLAP_CHARS = 400

DEFAULT_BATCH_SIZE = int(os.getenv("INGEST_EMBED_BATCH_SIZE", "32"))

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

_ENGINE = create_engine(DATABASE_URL, pool_pre_ping=True)
_OAI = OpenAI(api_key=OPENAI_API_KEY)


# -------------------------
# Data structures
# -------------------------
@dataclass
class DocChunk:
    chunk_index: int
    heading: str | None
    content: str
    start_char: int | None
    end_char: int | None
    content_hash: str
    token_count: int | None


# -------------------------
# Helpers (same as script)
# -------------------------
def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def estimate_tokens(text_str: str) -> int:
    return max(1, len(text_str) // 4)

def to_pgvector_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

def embed_batch(client: OpenAI, texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    out = [d.embedding for d in resp.data]
    for v in out:
        if len(v) != EMBED_DIM:
            raise RuntimeError(f"Unexpected embedding dim: {len(v)} (expected {EMBED_DIM})")
    return out


# -------------------------
# Markdown chunking (same)
# -------------------------
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)\s*$")

def split_markdown_sections(md: str) -> List[Tuple[str | None, str]]:
    lines = md.splitlines()
    sections: List[Tuple[str | None, List[str]]] = []
    current_heading: str | None = None
    current_body: List[str] = []

    def flush():
        nonlocal current_heading, current_body
        body = "\n".join(current_body).strip()
        if body:
            sections.append((current_heading, [body]))
        current_body = []

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            flush()
            current_heading = m.group(2).strip()
        else:
            current_body.append(line)

    flush()

    out: List[Tuple[str | None, str]] = []
    for h, bodies in sections:
        out.append((h, "\n\n".join(bodies).strip()))
    return out if out else [(None, md.strip())]

def chunk_text_with_overlap(text_str: str, max_chars: int, overlap_chars: int) -> List[Tuple[int, int, str]]:
    t = text_str.strip()
    if not t:
        return []
    chunks: List[Tuple[int, int, str]] = []
    i = 0
    n = len(t)
    while i < n:
        end = min(n, i + max_chars)
        chunk = t[i:end].strip()
        if chunk:
            chunks.append((i, end, chunk))
        if end >= n:
            break
        i = max(0, end - overlap_chars)
    return chunks

def chunk_document(suffix: str, full_text: str) -> List[DocChunk]:
    chunks: List[DocChunk] = []
    idx = 0

    if suffix.lower() == ".md":
        sections = split_markdown_sections(full_text)
        for heading, body in sections:
            for (s, e, chunk_txt) in chunk_text_with_overlap(body, MAX_CHARS, OVERLAP_CHARS):
                chash = sha256_text(f"{heading or ''}\n{chunk_txt}")
                chunks.append(
                    DocChunk(
                        chunk_index=idx,
                        heading=heading,
                        content=chunk_txt,
                        start_char=s,
                        end_char=e,
                        content_hash=chash,
                        token_count=estimate_tokens(chunk_txt),
                    )
                )
                idx += 1
    else:
        for (s, e, chunk_txt) in chunk_text_with_overlap(full_text, MAX_CHARS, OVERLAP_CHARS):
            chash = sha256_text(chunk_txt)
            chunks.append(
                DocChunk(
                    chunk_index=idx,
                    heading=None,
                    content=chunk_txt,
                    start_char=s,
                    end_char=e,
                    content_hash=chash,
                    token_count=estimate_tokens(chunk_txt),
                )
            )
            idx += 1

    return chunks


# -------------------------
# PDF bytes -> text
# -------------------------
def read_pdf_bytes(pdf_bytes: bytes) -> str:
    if PdfReader is None:
        raise RuntimeError("PDF support not installed. Run: pip install pypdf")
    import io
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts: List[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


# -------------------------
# SQL (unchanged)
# -------------------------
SQL_GET_DOC = text("""select id, content_hash from documents where path = :path""")

SQL_INSERT_DOC = text("""
insert into documents (path, title, content_hash)
values (:path, :title, :content_hash)
returning id
""")

SQL_UPDATE_DOC = text("""
update documents
set title = :title, content_hash = :content_hash, updated_at = now()
where id = :id
""")

SQL_DELETE_CHUNKS_FOR_DOC = text("""delete from chunks where document_id = :document_id""")

SQL_INSERT_CHUNK = text("""
insert into chunks (
  document_id, chunk_index, heading, content, content_hash, start_char, end_char, token_count
) values (
  :document_id, :chunk_index, :heading, :content, :content_hash, :start_char, :end_char, :token_count
)
returning id
""")

SQL_INSERT_EMBEDDING = text("""
insert into embeddings (chunk_id, embedding)
values (:chunk_id, (:embedding)::vector)
""")


# -------------------------
# Public API functions
# -------------------------
def ingest_file_bytes(
    filename: str,
    data: bytes,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    path_key: Optional[str] = None,
) -> Dict:
    """
    API-friendly ingestion:
    - path_key: stable doc key (defaults to filename). Use this if you want folder-like keys.
    Returns {"chunks_created": int, "embeddings_written": int, "doc_skipped": bool}
    """
    suffix = Path(filename).suffix.lower()
    title = Path(filename).stem

    # 1) Load text from bytes
    if suffix in (".md", ".txt"):
        raw_text = data.decode("utf-8", errors="ignore").strip()
    elif suffix == ".pdf":
        raw_text = read_pdf_bytes(data).strip()
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    if not raw_text:
        return {"chunks_created": 0, "embeddings_written": 0, "doc_skipped": False}

    doc_hash = sha256_text(raw_text)
    doc_path = (path_key or filename).replace("\\", "/")  # same normalization idea

    chunks_written = 0
    embeds_written = 0

    with Session(_ENGINE) as db:
        row = db.execute(SQL_GET_DOC, {"path": doc_path}).mappings().first()

        # skip unchanged
        if row and row["content_hash"] == doc_hash:
            return {"chunks_created": 0, "embeddings_written": 0, "doc_skipped": True}

        # upsert document
        if row:
            doc_id = int(row["id"])
            db.execute(SQL_UPDATE_DOC, {"id": doc_id, "title": title, "content_hash": doc_hash})
            db.execute(SQL_DELETE_CHUNKS_FOR_DOC, {"document_id": doc_id})
        else:
            doc_id = int(db.execute(SQL_INSERT_DOC, {"path": doc_path, "title": title, "content_hash": doc_hash}).scalar_one())

        # chunk
        chunks = chunk_document(suffix, raw_text)
        if not chunks:
            db.commit()
            return {"chunks_created": 0, "embeddings_written": 0, "doc_skipped": False}

        # insert chunks
        chunk_ids: List[int] = []
        chunk_texts: List[str] = []
        for ch in chunks:
            chunk_id = int(
                db.execute(
                    SQL_INSERT_CHUNK,
                    {
                        "document_id": doc_id,
                        "chunk_index": ch.chunk_index,
                        "heading": ch.heading,
                        "content": ch.content,
                        "content_hash": ch.content_hash,
                        "start_char": ch.start_char,
                        "end_char": ch.end_char,
                        "token_count": ch.token_count,
                    },
                ).scalar_one()
            )
            chunk_ids.append(chunk_id)
            chunk_texts.append(ch.content)
            chunks_written += 1

        # embed + insert embeddings in batches
        bs = max(1, int(batch_size))
        for i in range(0, len(chunk_texts), bs):
            batch_texts = chunk_texts[i : i + bs]
            batch_ids = chunk_ids[i : i + bs]
            vectors = embed_batch(_OAI, batch_texts)
            for cid, vec in zip(batch_ids, vectors):
                db.execute(SQL_INSERT_EMBEDDING, {"chunk_id": cid, "embedding": to_pgvector_literal(vec)})
                embeds_written += 1

        db.commit()

    return {
        "chunks_created": chunks_written,
        "embeddings_written": embeds_written,
        "doc_skipped": False,
    }


def get_corpus_stats() -> Dict:
    """
    Minimal stats for UI.
    """
    # These assume your tables exist and are not massive; count(*) is fine for MVP.
    SQL_DOC_COUNT = text("select count(*) from documents")
    SQL_CHUNK_COUNT = text("select count(*) from chunks")
    # If you have updated_at on documents, you can use:
    SQL_LAST_INGEST = text("select max(updated_at) from documents")

    with Session(_ENGINE) as db:
        total_docs = int(db.execute(SQL_DOC_COUNT).scalar_one())
        total_chunks = int(db.execute(SQL_CHUNK_COUNT).scalar_one())
        last_ingest = db.execute(SQL_LAST_INGEST).scalar_one()
        if last_ingest is not None:
            # last_ingest might already be tz-aware depending on DB settings
            last_ingest_time = last_ingest.isoformat()
        else:
            last_ingest_time = None

    return {
        "total_docs": total_docs,
        "total_chunks": total_chunks,
        "last_ingest_time": last_ingest_time,
    }
