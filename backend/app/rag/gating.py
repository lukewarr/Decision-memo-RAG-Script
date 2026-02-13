# app/rag/gating.py
from dataclasses import dataclass
from typing import Optional, Sequence

@dataclass
class GateResult:
    decision: str  # "confident" | "weak" | "insufficient"
    best_distance: Optional[float]
    gap: Optional[float]        # hit2 - hit1
    reason: str

def gate_hits(
    hits: Sequence,
    *,
    max_distance: float = 0.65,
    weak_band: float = 0.05,          # within 0.05 of the threshold -> "weak"
    min_gap: float = 0.03,            # if top2 are too close -> "weak" (optional)
) -> GateResult:
    """
    hits: list of RetrievedChunk-like objects with .distance (float|None)
    """
    valid = [h for h in hits if getattr(h, "distance", None) is not None]
    if not valid:
        return GateResult("insufficient", None, None, "no_valid_hits")
    valid.sort(key=lambda h: h.distance)

    best = valid[0].distance
    second = valid[1].distance if len(valid) > 1 else None
    gap = (second - best) if (second is not None and best is not None) else None

    if best is None:
        return GateResult("insufficient", None, None, "best_distance_none")

    if best > max_distance:
        return GateResult("insufficient", best, gap, f"best_distance>{max_distance}")

    # Weak band: near-threshold OR ambiguous top2
    if best > (max_distance - weak_band):
        return GateResult("weak", best, gap, "near_threshold")

    if gap is not None and gap < min_gap:
        return GateResult("weak", best, gap, "low_separation")

    return GateResult("confident", best, gap, "ok")
