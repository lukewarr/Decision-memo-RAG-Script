# Decision Memo RAG (Constraint-Driven RAG System)

A Retrieval-Augmented Generation (RAG) service designed to be **shippable**: constrained, observable, and debuggable.  
The core product flow answers questions with **citations to source documents** and generates structured **decision memos** grounded in the indexed corpus.

This project prioritizes:
- engineering judgment over flashy demos
- cost/latency constraints
- observability + failure modes
- simplicity and debuggability

---

## Features (Current)

- Local folder ingestion (Markdown/text; PDF support can be added later)
- Chunking + embeddings using OpenAI embeddings
- Storage in Postgres + pgvector (Supabase)
- Vector similarity retrieval (top-k)
- FastAPI service with `/ask` (in-progress) and debug retrieval (recommended)

---

## Roadmap (High-Level)

**Phase 1 — Constraint-Driven RAG System**
- [x] Supabase Postgres + pgvector schema
- [x] Ingest local folder → chunks + embeddings stored in DB
- [x] Retrieval query returns relevant chunks
- [x] `/debug/retrieve` endpoint for retrieval debugging
- [x] `/ask` endpoint: answer + citations
- [x] “Insufficient evidence” gating
- [x] Basic metrics (latency, tokens, cost estimates)
- [x] Caching for query embeddings / retrieval results (optional)

**Phase 2 - Frontend development**
- [x] Developed ingestion landing page
- [x] Developed processing landing page (ask/memo)
- [ ] Refine page & process

**Phase 3 — Evaluation Harness**
- [ ] Regression tests for retrieval/citation accuracy
- [ ] Prompt robustness tests
- [ ] Cost/latency drift detection
- [ ] Failure mode detection + reporting

---
## Key RAG Constraints / “Hard Teaches”

The system enforces a **distance-gated** retrieval policy to avoid hallucinated answers:

- `RAG_MAX_DISTANCE` (default ~0.65): beyond this is generally **insufficient**
- `RAG_CONTEXT_MARGIN` (default ~0.08): include context hits within `(best_distance + margin)`
- Gating outputs: `confident | weak | insufficient`
- Recommended short-query guard: block ambiguous queries under ~6 chars (configurable)

These constraints are intentionally simple and debuggable.

---

## API (Local Dev)

### Ingest
**POST `/ingest`** (multipart form-data)
- Field name: `files` (repeatable)
- Returns:
  - `files_received`, `files_ingested`, `files_skipped` (if enabled)
  - `chunks_created`, `embeddings_written`
  - `errors`
  - `last_ingest_time`

### Corpus Stats
**GET `/corpus/stats`**
- Returns:
  - `total_docs`, `total_chunks`, `last_ingest_time`

### Ask (RAG QA)
**POST `/ask`**
- Request:
  - `question: str`
  - `k: int`
  - `include_hits: bool` (optional)
- Response (intended stable shape):
  - `answer: str`
  - `gating: confident|weak|insufficient`
  - `best_distance: float | null`
  - `weak_match: bool`
  - `citations: [...]`
  - `hits: [...]` (optional)

### Memo (GEN)
**POST `/memo`**
- Request:
  - `topic: str`
  - `k: int`
  - `include_hits: bool` (optional)
- Response (intended stable shape):
  - `sections: {tldr, options_tradeoffs, risks_mitigations, open_questions, what_would_change_my_mind}`
  - `gating`, `best_distance`, `weak_match`, `citations`
  - `hits` optional

---

## Running Locally

### 1) Backend
From the `backend/` directory:

```bash
uvicorn app.main:app --reload --port 8000

