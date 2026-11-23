"""Request/response models (Pydantic = validated, self-documenting JSON)."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    model: Optional[str] = Field(default=None, max_length=80, description="Model id from GET /models; empty = the default model")
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)
    docs: Optional[list[str]] = Field(default=None, max_length=20, description="Only search these doc_name values (see GET /documents)")


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=6000)


class AgentChatRequest(ChatRequest):
    """Agent mode: the agent keeps no state, so the UI sends the earlier turns (oldest first) with every question."""
    history: list[HistoryMessage] = Field(default_factory=list, max_length=40)


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


class OutlineArticle(BaseModel):
    article: str
    title: str = ""
    chapter: str = ""


class OutlineSection(BaseModel):
    section: int
    title: str
    count: int
    first: str = ""
    last: str = ""
    chapters: list[str] = Field(default_factory=list)
    articles: list[OutlineArticle] = Field(default_factory=list)


class OutlineResponse(BaseModel):
    doc_name: str
    doc_label: str
    found: bool
    total_articles: int = 0
    section_count: int = 0
    numbering_restarts: bool = False
    sections: list[OutlineSection] = Field(default_factory=list)
