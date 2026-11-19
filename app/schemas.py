"""Request/response models (Pydantic = validated, self-documenting JSON)."""
from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)
    docs: Optional[list[str]] = Field(default=None, max_length=20, description="Only search these doc_name values (see GET /documents)")


class Citation(BaseModel):
    id: int = Field(description="The [n] marker used inside the answer")
    doc_name: str
    doc_label: str
    chapter: str = ""
    article: str
    article_title: str = ""
    clause: str = ""
    score: float
    snippet: str


class Latency(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class ChatResponse(BaseModel):
    answer: str
    grounded: bool = Field(description="True if the answer cites at least one retrieved source")
    confidence: float = Field(description="Top-1 retrieval score (0-1). A heuristic, not a probability")
    citations: list[Citation]
    retrieved: int
    refusal_reason: Optional[str] = None
    retrieval_mode: str = ""
    search_query: Optional[str] = Field(default=None, description="Rewritten query, if rewriting is on")
    latency_ms: Latency
    model: str


class SearchResponse(BaseModel):
    query: str
    results: list[Citation]
    latency_ms: float
