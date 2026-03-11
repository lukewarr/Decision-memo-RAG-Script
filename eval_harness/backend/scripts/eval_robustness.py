import argparse
from typing import Any, Dict, List

from backend.scripts.eval_utils import (
    read_jsonl,
    write_json,
    write_text,
    now_iso,
    http_post_json,
    gating_geq,
    gating_leq,
    summarize_table,
)

def call_ask(backend: str, question: str, k: int) -> Dict[str, Any]:
    url = backend.rstrip("/") + "/ask"
    payload = {"question": question, "k": k, "include_hits": True}
    status, data, dt = http_post_json(url, payload, timeout_s=180)
    if isinstance(data, dict):
        data["_http_ms"] = round(dt * 1000, 2)
        data["_http_status"] = status
    return data if isinstance(data, dict) else {"_raw": data, "_http_status": status, "_http_ms": round(dt * 1000, 2)}

def citations_subset_of_hits(resp: Dict[str, Any]) -> bool:
    cits = resp.get("citations") or []
    hits = resp.get("hits") or []
    hit_ids = {h.get("chunk_id") for h in hits if isinstance(h, dict)}
    for c in cits:
        cid = c.get("chunk_id") if isinstance(c, dict) else None
        if cid is None:
            continue
        if cid not in hit_ids and hit_ids:
            return False
    return True

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="http://127.0.0.1:8000")
    ap.add_argument("--spec", default="eval/robustness.jsonl")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out_json", default="eval/reports/robustness_latest.json")
    ap.add_argument("--out_md", default="eval/reports/robustness_latest.md")
    args = ap.parse_args()

    tests = read_jsonl(args.spec)
    rows: List[Dict[str, Any]] = []
    failures = 0

    for t in tests:
        tid = t["id"]
        base = t["base"]
        variants = t.get("variants", [])
        expected_g = t.get("expected_gating")
        min_g = t.get("expected_min_gating")
        max_g = t.get("expected_max_gating")

        all_qs = [base] + variants
        for q in all_qs:
            resp = call_ask(args.backend, q, args.k)
            gating = resp.get("gating", "")
            http_ms = resp.get("_http_ms")

            ok = True
            if expected_g:
                ok = ok and (gating == expected_g)
            if min_g:
                ok = ok and gating_geq(gating, min_g)
            if max_g:
                ok = ok and gating_leq(gating, max_g)

            # basic grounding sanity
            ok = ok and citations_subset_of_hits(resp)

            if not ok:
                failures += 1

            rows.append({
                "id": tid,
                "pass": ok,
                "lat_ms": http_ms,
                "best_distance": resp.get("best_distance"),
                "top_path": (resp.get("citations")[0].get("path") if resp.get("citations") else ""),
                "query": q,
                "gating": gating,
                "raw": resp if not ok else None,
            })

    md = []
    md.append(f"# Robustness Report\n\nRun: `{now_iso()}`\n")
    md.append(f"- Backend: `{args.backend}`")
    md.append(f"- Cases: `{len(rows)}`")
    md.append(f"- Failures: `{failures}`\n")

    md.append(summarize_table(rows))
    md.append("\n")

    if failures:
        md.append("## Failures (details)\n")
        for r in rows:
            if r["pass"]:
                continue
            md.append(f"### {r['id']}\n")
            md.append(f"- Query: {r['query']}\n")
            md.append(f"- Gating: `{r['gating']}`\n")
            md.append("```json\n" + __import__("json").dumps(r.get("raw") or {}, indent=2) + "\n```\n")

    write_json(args.out_json, {"run": now_iso(), "backend": args.backend, "failures": failures, "rows": rows})
    write_text(args.out_md, "\n".join(md))
    print(f"Wrote: {args.out_md}")
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())