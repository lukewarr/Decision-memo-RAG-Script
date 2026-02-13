# Postmortem: OpenAI rate limit errors during demo

Date: 2026-01-15

## Impact
Demo failed for ~3 minutes due to 429 responses.

## Timeline
- 10:01 demo started
- 10:02 repeated /ask calls triggered bursts
- 10:03 first 429 observed
- 10:04 added backoff and retry
- 10:05 recovered

## Root cause
No client-side retry/backoff policy and no caching of embeddings.

## Fixes
- Add exponential backoff with jitter for embeddings and completion calls
- Add in-memory cache for query embeddings for 10 minutes
- Add request-level timeout and clear error messages

## Follow-ups
Add metrics: retry count, 429 rate, and cache hit rate.
