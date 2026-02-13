# Decision: Add caching layer for retrieval results

Date: 2026-01-16
Owner: Luke

## Context
Vector search is fast, but repeated queries (same or near-same) are common in a demo.
We want to reduce latency and cost while preserving correctness.

## Options
1) No cache
2) In-memory LRU cache (single process)
3) Redis cache (shared)

## Decision
Start with in-memory LRU caching for:
- query embedding results (keyed by normalized question)
- retrieval results (top-k chunk ids)

## Rationale
- Lowest operational overhead for MVP
- Easy to instrument hit rate and latency improvements

## Risks
- Cache invalidation after re-index
- Multi-worker deployment will reduce cache effectiveness

## Mitigations
- Add a corpus_version (ingest run id) to cache keys
- If/when multi-worker, move to Redis

## What would change my mind
If we deploy multiple workers or need shared caching across instances, use Redis.
