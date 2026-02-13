# PRD: Decision Memo RAG (MVP)

## Goals
- Ingest local folder of .md/.txt/.pdf
- Answer questions with citations (doc path + chunk id)
- Generate decision memos with: TL;DR, options & tradeoffs, risks, open questions, what changes my mind
- Enforce cost and latency constraints
- Provide structured logs: cost estimate, tokens, retrieval stats, latency breakdown

## Non-goals
- Web upload UI
- Multi-user auth
- Re-ranking models
- Agent frameworks

## Success metrics
- Citation coverage: >= 80% of factual claims have citations
- p95 latency meets targets
- Cost caps enforced with zero overages
