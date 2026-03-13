import argparse
from typing import Any, Dict, List, Optional

from eval_utils import (
    read_jsonl,
    write_json,
    write_text,
    now_iso,
    http_post_json,
    normalize_hits,
    best_distance_from_hits,
    path_matches_expected,
    summarize_table,
)

def call_debug_retrieve(backend: str, query: str, k: int) -> Dict[str, Any]:
    # Try common payload shapes (your debug endpoint may use query vs question)
    url = backend.rstrip("/") + "/debug/retrieve"
    for payload in ({"query": query, "k": k}, {"question": query, "k": k}):
        status, data, dt = http_post_json(url, payload, timeout_s=120)
        if status < 300 and isinstance(data, (dict, list)):
            data["_http_ms"] = round(dt * 1000, 2)
            data["_http_status"] = status
            return data
    return {"hits": [], "_http_status": status, "_http_ms": round(dt * 1000, 2), "_raw": data}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="http://127.0.0.1:8000")
    ap.add_argument("--golden", default="eval/golden_retrieval.jsonl")
    ap.add_argument("--out_json", default="eval/reports/retrieval_latest.json")
    ap.add_argument("--out_md", default="eval/reports/retrieval_latest.md")
    args = ap.parse_args()

    tests = read_jsonl(args.golden)
    results: List[Dict[str, Any]] = []
    failures = 0

    for t in tests:
        tid = t["id"]
        q = t["query"]
        k = int(t.get("k", 8))
        expected = t.get("expected_paths", [])
        max_bd = t.get("max_best_distance", None)

        resp = call_debug_retrieve(args.backend, q, k)
        hits = normalize_hits(resp)
        bd = best_distance_from_hits(hits)
        http_ms = resp.get("_http_ms")

        top_path = hits[0].get("path") if hits else ""
        hit_ok = False
        if hits and expected:
            hit_ok = any(path_matches_expected(h.get("path", ""), expected) for h in hits)
        elif hits and not expected:
            hit_ok = True  # if no expectation, any hit is fine

        bd_ok = True
        if max_bd is not None:
            bd_ok = (bd is not None) and (bd <= float(max_bd))

        passed = bool(hit_ok and bd_ok)
        if not passed:
            failures += 1

        results.append({
            "id": tid,
            "pass": passed,
            "query": q,
            "k": k,
            "expected_paths": expected,
            "best_distance": bd,
            "http_ms": http_ms,
            "top_path": top_path,
            "raw": resp if not passed else None,
        })

    report_rows = []
    for r in results:
        report_rows.append({
            "id": r["id"],
            "pass": r["pass"],
            "lat_ms": r["http_ms"],
            "best_distance": r["best_distance"],
            "top_path": r["top_path"],
        })

    md = []
    md.append(f"# Retrieval Regression Report\n\nRun: `{now_iso()}`\n")
    md.append(f"- Backend: `{args.backend}`")
    md.append(f"- Tests: `{len(results)}`")
    md.append(f"- Failures: `{failures}`\n")
    md.append(summarize_table(report_rows))
    md.append("\n")

    if failures:
        md.append("## Failures (details)\n")
        for r in results:
            if r["pass"]:
                continue
            md.append(f"### {r['id']}\n")
            md.append(f"- Query: {r['query']}\n")
            md.append(f"- Expected tokens: `{r['expected_paths']}`\n")
            md.append(f"- best_distance: `{r['best_distance']}`\n")
            md.append(f"- top_path: `{r['top_path']}`\n")
            raw = r.get("raw") or {}
            md.append("```json\n" + __import__("json").dumps(raw, indent=2) + "\n```\n")

    write_json(args.out_json, {"run": now_iso(), "backend": args.backend, "failures": failures, "results": results})
    write_text(args.out_md, "\n".join(md))

    print(f"Wrote: {args.out_md}")
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())