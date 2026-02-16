from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


Gating = Literal["confident", "weak", "insufficient"]


class Citation(BaseModel):
    path: str
    title: Optional[str] = None
    heading: Optional[str] = None
    chunk_id: Optional[int] = None
    distance: Optional[float] = None
    excerpt: Optional[str] = None  # short snippet for UI


class Hit(BaseModel):
    # Keep this flexible because your retrieve may return extra fields
    chunk_id: Optional[int] = None
    path: Optional[str] = None
    title: Optional[str] = None
    heading: Optional[str] = None
    content: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    distance: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class AskResponse(BaseModel):
    answer: str
    gating: Gating
    best_distance: Optional[float] = None
    weak_match: bool = False
    citations: List[Citation] = Field(default_factory=list)
    hits: Optional[List[Hit]] = None  # only returned when include_hits=True


class MemoSections(BaseModel):
    tldr: str = ""
    options_tradeoffs: str = ""
    risks_mitigations: str = ""
    open_questions: str = ""
    what_would_change_my_mind: str = ""


class MemoResponse(BaseModel):
    sections: MemoSections
    gating: Gating
    best_distance: Optional[float] = None
    weak_match: bool = False
    citations: List[Citation] = Field(default_factory=list)
    hits: Optional[List[Hit]] = None
