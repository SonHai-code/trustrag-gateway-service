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
    text: Optional[str] = Field(default=None, description="Full chunk text; only returned by /search with include_text=true")


class Latency(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    ttft_ms: Optional[float] = Field(default=None, description="Time to first token (streaming)")
    stages: dict[str, float] = Field(default_factory=dict, description="rewrite_ms / embed_ms / vector_ms / rerank_ms")


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
    cached: bool = Field(default=False, description="True if the answer was replayed from the in-memory cache")


class SearchRequest(ChatRequest):
    include_text: bool = Field(default=False, description="Return the full chunk text (the agent grades on it)")


class SearchResponse(BaseModel):
    query: str
    results: list[Citation]
    latency_ms: float
    best_score: float = 0.0
    sufficient: bool = Field(default=False, description="Would /chat answer this, or refuse for low retrieval score?")
    retrieval_mode: str = ""
    stages: dict[str, float] = Field(default_factory=dict)


class ArticleResponse(BaseModel):
    doc_name: str
    doc_label: str
    article: str
    article_title: str = ""
    found: bool
    text: str = ""
