"""
Ingest a local folder of .md/.txt/.pdf files into Supabase Postgres + pgvector.

- Upserts documents by path
- Skips unchanged docs using content_hash
- For changed docs: deletes existing chunks (embeddings cascade) and re-inserts
- Chunks markdown by headings/paragraphs (simple + debuggable)
- Embeds chunks with OpenAI text-embedding-3-small (1536 dim)
- Stores embeddings as pgvector literal strings cast to vector in SQL

Usage:
  python scripts/ingest_folder.py data/test_demo
Env:
  DATABASE_URL=postgresql+psycopg://...  (Supabase pooler ok)
  OPENAI_API_KEY=...
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from pypdf import PdfReader
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()


EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# Chunking defaults (simple + good enough)
MAX_CHARS = 3500          # roughly ~800-900 tokens depending on text
OVERLAP_CHARS = 400       # overlap to preserve context across boundaries


@dataclass
class DocChunk:
    chunk_index: int
    heading: str | None
    content: str
    start_char: int | None
    end_char: int | None
    content_hash: str
    token_count: int | None

#Skips unchanged docs
def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

#Estimates tokens used
def estimate_tokens(text_str: str) -> int:
    # cheap estimate: ~4 chars/token (varies by content)
    return max(1, len(text_str) // 4)

#Reads .md/.txt files
def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

#Reads .pdf files
def read_pdf_file(path: Path) -> str:
    if PdfReader is None:
        raise RuntimeError("PDF support not installed. Run: pip install pypdf")
    reader = PdfReader(str(path))
    parts: List[str] = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        parts.append(txt)
    return "\n".join(parts)

#Loads read files
def load_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in [".md", ".txt"]:
        return read_text_file(path)
    if suffix == ".pdf":
        return read_pdf_file(path)
    raise ValueError(f"Unsupported file type: {path}")


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)\s*$")

#Keeps structures
def split_markdown_sections(md: str) -> List[Tuple[str | None, str]]:
    """
    Returns list of (heading, body_text). heading None for preface text.
    """
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
            # flush previous
            flush()
            current_heading = m.group(2).strip()
        else:
            current_body.append(line)

    flush()

    # collapse list-of-list
    out: List[Tuple[str | None, str]] = []
    for h, bodies in sections:
        out.append((h, "\n\n".join(bodies).strip()))
    return out if out else [(None, md.strip())]

# Make consistent with overlap
def chunk_text_with_overlap(text_str: str, max_chars: int, overlap_chars: int) -> List[Tuple[int, int, str]]:
    """
    Returns list of (start_char, end_char, chunk_text) over the given text.
    """
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

# Choose a chunking strategy depending on file type
def chunk_document(path: Path, full_text: str) -> List[DocChunk]:
    """
    Chunk strategy:
    - If .md: split into sections by headings, then chunk each section with overlap
    - Else: chunk full text with overlap
    """
    chunks: List[DocChunk] = []
    idx = 0

    if path.suffix.lower() == ".md":
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

#Converts a list of floats into the string format
def to_pgvector_literal(vec: List[float]) -> str:
    # pgvector accepts: '[1,2,3]'::vector
    # Keep it compact to reduce SQL payload size.
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

#Embeds multiple chunks inot one openAI api call
def embed_batch(client: OpenAI, texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    out = [d.embedding for d in resp.data]
    # basic sanity check
    for v in out:
        if len(v) != EMBED_DIM:
            raise RuntimeError(f"Unexpected embedding dim: {len(v)} (expected {EMBED_DIM})")
    return out


SQL_GET_DOC = text("""
select id, content_hash from documents where path = :path
""")

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

SQL_DELETE_CHUNKS_FOR_DOC = text("""
delete from chunks where document_id = :document_id
""")

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
""")  # cast the string literal to vector


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".md", ".txt", ".pdf"]:
            yield p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="Folder to ingest, e.g. data/test_demo")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")

    if not database_url:
        print("ERROR: DATABASE_URL is not set in environment.", file=sys.stderr)
        return 2
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set in environment.", file=sys.stderr)
        return 2

    root = Path(args.folder).resolve()
    if not root.exists():
        print(f"ERROR: Folder not found: {root}", file=sys.stderr)
        return 2

    engine = create_engine(database_url, pool_pre_ping=True)
    client = OpenAI(api_key=api_key)

    docs_total = 0
    docs_skipped = 0
    docs_updated = 0
    chunks_written = 0
    embeds_written = 0

    t0 = time.time()

    with Session(engine) as db:
        for fp in iter_files(root):
            rel_path = str(fp.relative_to(root)).replace("\\", "/")
            full_path = f"{root.name}/{rel_path}"  # stable-ish path key
            title = fp.stem

            raw_text = load_document_text(fp).strip()
            if not raw_text:
                continue

            doc_hash = sha256_text(raw_text)

            row = db.execute(SQL_GET_DOC, {"path": full_path}).mappings().first()

            if row and row["content_hash"] == doc_hash:
                docs_total += 1
                docs_skipped += 1
                continue

            # upsert document
            if row:
                doc_id = int(row["id"])
                db.execute(SQL_UPDATE_DOC, {"id": doc_id, "title": title, "content_hash": doc_hash})
                # simplest correctness: delete all existing chunks; embeddings cascade
                db.execute(SQL_DELETE_CHUNKS_FOR_DOC, {"document_id": doc_id})
            else:
                doc_id = int(db.execute(SQL_INSERT_DOC, {"path": full_path, "title": title, "content_hash": doc_hash}).scalar_one())

            # chunk
            chunks = chunk_document(fp, raw_text)
            if not chunks:
                docs_total += 1
                docs_updated += 1
                continue

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
            bs = max(1, int(args.batch_size))
            for i in range(0, len(chunk_texts), bs):
                batch_texts = chunk_texts[i : i + bs]
                batch_ids = chunk_ids[i : i + bs]
                vectors = embed_batch(client, batch_texts)
                for cid, vec in zip(batch_ids, vectors):
                    db.execute(SQL_INSERT_EMBEDDING, {"chunk_id": cid, "embedding": to_pgvector_literal(vec)})
                    embeds_written += 1

            db.commit()
            docs_total += 1
            docs_updated += 1

    dt = time.time() - t0
    print("Ingest complete")
    print(f"  Folder: {root}")
    print(f"  Docs processed: {docs_total}")
    print(f"    Skipped (unchanged): {docs_skipped}")
    print(f"    Updated/Inserted:    {docs_updated}")
    print(f"  Chunks written:   {chunks_written}")
    print(f"  Embeddings written: {embeds_written}")
    print(f"  Time: {dt:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
