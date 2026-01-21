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
- [ ] `/debug/retrieve` endpoint for retrieval debugging
- [ ] `/ask` endpoint: answer + citations
- [ ] “Insufficient evidence” gating
- [ ] Basic metrics (latency, tokens, cost estimates)
- [ ] Caching for query embeddings / retrieval results (optional)

**Phase 2 — Evaluation Harness**
- [ ] Regression tests for retrieval/citation accuracy
- [ ] Prompt robustness tests
- [ ] Cost/latency drift detection
- [ ] Failure mode detection + reporting

---

## Repo Structure (Current / Intended)

