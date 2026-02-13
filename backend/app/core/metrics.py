# app/core/metrics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import os
import time

# Rough pricing (USD per 1M tokens). Override via env if you want.
DEFAULT_PRICES_PER_1M = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

def _get_price_per_1m(model: str) -> tuple[float, float]:
    key = model.upper().replace("-", "_").replace(".", "_")
    env_in = os.getenv(f"PRICE_{key}_IN_PER_1M")
    env_out = os.getenv(f"PRICE_{key}_OUT_PER_1M")

    if env_in and env_out:
        return float(env_in), float(env_out)

    d = DEFAULT_PRICES_PER_1M.get(model)
    if d:
        return float(d["input"]), float(d["output"])

    return 0.0, 0.0

def estimate_cost_usd(model: str, usage: dict[str, Any]) -> float:
    """
    usage expects keys like: input_tokens, output_tokens
    (OpenAI Responses API usage shape)
    """
    in_toks = int(usage.get("input_tokens") or 0)
    out_toks = int(usage.get("output_tokens") or 0)
    in_price, out_price = _get_price_per_1m(model)

    return (in_toks / 1_000_000.0) * in_price + (out_toks / 1_000_000.0) * out_price

@dataclass
class Timer:
    start: float
    def ms(self) -> float:
        return (time.perf_counter() - self.start) * 1000.0

def now_timer() -> Timer:
    return Timer(time.perf_counter())
