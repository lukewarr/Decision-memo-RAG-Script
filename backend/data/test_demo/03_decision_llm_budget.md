# Decision: Hard budget enforcement per request

Date: 2026-01-16
Owner: Luke

## Requirements
- /ask must cost <= $0.02 per request
- /memo must cost <= $0.06 per request
- p95 latency targets: /ask <= 1.8s, /memo <= 3.5s

## Policy
If estimated budget would be exceeded:
1) Reduce top_k
2) Reduce context token budget
3) Reduce output token max
4) Return "insufficient evidence" + follow-up question

## Rationale
Budgets prevent surprise bills and keep the system ship-worthy.

## What would change my mind
If users explicitly opt-in to "high quality mode" with a higher cap.
