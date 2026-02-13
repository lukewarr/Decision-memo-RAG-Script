# Meeting Notes - Week 1

Date: 2026-01-16

- Agreed to start with Supabase pgvector.
- Decided MVP ingestion is local folder only.
- Noted we should add caching early to avoid repeated embedding costs.
- Discussed failure mode: empty retrieval should lead to "insufficient evidence" not hallucination.
- Next: build ingestion CLI and then /ask endpoint.
