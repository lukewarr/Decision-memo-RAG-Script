# Decision: Use Supabase Postgres + pgvector

Date: 2026-01-16
Owner: Luke

## Context
We need a vector store with reliable operations and simple infra.

## Options
1) Local FAISS + files
2) Postgres + pgvector locally
3) Supabase hosted Postgres + pgvector

## Decision
Use Supabase Postgres + pgvector.

## Rationale
- Realistic for startups
- Supports SQL joins for citations and metadata
- HNSW index supports fast similarity search

## Consequences
- Must handle connection pooling (transaction vs session)
- Need migrations discipline

## What would change my mind
If cost scales poorly or we need on-prem, move to self-hosted Postgres.
