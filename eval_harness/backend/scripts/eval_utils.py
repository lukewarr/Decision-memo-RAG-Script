import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


GATING_ORDER = {"insufficient": 0, "weak": 1, "confident": 2}


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def http_post_json(url: str, payload: Dict[str, Any], timeout_s: int = 120) -> Tuple[int, Dict[str, Any], float]:
    t0 = time.perf_counter()
    r = requests.post(url, json=payload, timeout=timeout_s)
    dt = time.perf_counter() - t0
    try:
        data = r.json()
    except Exception:
        data = {"_raw": r.text}
    return r.status_code, data, dt


def http_get_json(url: str, timeout_s: int = 60) -> Tuple[int, Dict[str, Any], float]:
    t0 = time.perf_counter()
    r = requests.get(url, timeout=timeout_s)
    dt = time.perf_counter() - t0
    try:
        data = r.json()
    except Exception:
        data = {"_raw": r.text}
    return r.status_code, data, dt


def gating_leq(a: str, b: str) -> bool:
    return GATING_ORDER.get(a, -1) <= GATING_ORDER.get(b, -1)


def gating_geq(a: str, b: str) -> bool:
    return GATING_ORDER.get(a, -1) >= GATING_ORDER.get(b, -1)


def best_distance_from_hits(hits: List[Dict[str, Any]]) -> Optional[float]:
    vals = []
    for h in hits:
        d = h.get("distance")
        if isinstance(d, (int, float)):
            vals.append(float(d))
    return min(vals) if vals else None


def normalize_hits(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    # supports either {"hits":[...]} or direct list
    if isinstance(resp, dict) and isinstance(resp.get("hits"), list):
        return resp["hits"]
    if isinstance(resp, list):
        return resp
    return []


def path_matches_expected(path: str, expected_tokens: List[str]) -> bool:
    p = (path or "").lower()
    return any(tok.lower() in p for tok in expected_tokens if tok)


def summarize_table(rows: List[Dict[str, Any]]) -> str:
    # simple markdown table
    headers = ["id", "pass", "lat_ms", "best_distance", "top_path"]
    out = ["| " + " | ".join(headers) + " |", "|---|---:|---:|---:|---|"]
    for r in rows:
        out.append(
            "| {id} | {pass_} | {lat_ms} | {bd} | {tp} |".format(
                id=r.get("id", ""),
                pass_="✅" if r.get("pass") else "❌",
                lat_ms=str(r.get("lat_ms", "")),
                bd=str(r.get("best_distance", "")),
                tp=(r.get("top_path", "") or "").replace("|", "\\|"),
            )
        )
    return "\n".join(out)