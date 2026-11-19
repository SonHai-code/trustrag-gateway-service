"""The RAG pipeline: (rewrite) -> retrieve -> gate -> generate -> extract citations.

Kept free of FastAPI imports so it can be unit-tested and reused (agent in Phase 4).
"""
import asyncio
import re
import time
from typing import AsyncIterator

from app import llm, settings
from app.prompts import NO_ANSWER, REWRITE_PROMPT, SYSTEM_PROMPT, build_user_prompt, doc_label
from app.schemas import ChatResponse, Citation, Latency

CITE_RE = re.compile(r"\[(\d+)\]")


async def retrieve_hits(question: str, top_k: int, docs: list[str] | None = None):
    from app.retrieval import retrieve
    return await asyncio.to_thread(retrieve, question, top_k, docs)


async def maybe_rewrite(question: str) -> str:
    """Optional: let the LLM turn a chatty question into a tight search query. Falls back silently."""
    if not settings.REWRITE_QUERY or settings.LLM_PROVIDER == "mock":
        return question
    try:
        text = (await llm.generate(REWRITE_PROMPT, question)).strip().strip('"')
    except llm.LLMError:
        return question
    return text[:300] if text else question


def extract_cited_ids(text: str, n_sources: int) -> list[int]:
    """Valid [n] markers in order of first appearance. Markers like [9] with no source are ignored."""
    seen: list[int] = []
    for m in CITE_RE.finditer(text):
        i = int(m.group(1))
        if 1 <= i <= n_sources and i not in seen:
            seen.append(i)
    return seen


def to_citation(i: int, hit) -> Citation:
    p = hit.payload
    text = p.get("text", "")
    return Citation(
        id=i,
        doc_name=p.get("doc_name", ""),
        doc_label=doc_label(p.get("doc_name", "")),
        chapter=p.get("chapter", ""),
        article=str(p.get("article", "")),
        article_title=p.get("article_title", ""),
        clause=str(p.get("clause", "")),
        score=round(hit.score, 4),
        snippet=text[:300],
    )


def _best_score(hits) -> float:
    # max, not hits[0]: in hybrid mode the first hit may be a keyword-only match without a dense score
    return max((h.score for h in hits), default=0.0)


def _confidence(hits) -> float:
    return round(max(0.0, min(1.0, _best_score(hits))), 4)


def _threshold(hits) -> float:
    reranked = any(getattr(h, "rerank", None) is not None for h in hits)
    return settings.RERANK_MIN_SCORE if reranked else settings.MIN_SCORE


def _mode_label(hits) -> str:
    reranked = any(getattr(h, "rerank", None) is not None for h in hits)
    return settings.SEARCH_MODE + ("+rerank" if reranked else "")


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def _should_refuse(hits) -> bool:
    return not hits or _best_score(hits) < _threshold(hits)


def _refusal(hits, reason: str, t_start: float, t_ret: float, search_query: str | None) -> ChatResponse:
    return ChatResponse(
        answer=NO_ANSWER, grounded=False, confidence=_confidence(hits), citations=[],
        retrieved=len(hits), refusal_reason=reason, retrieval_mode=_mode_label(hits), search_query=search_query,
        latency_ms=Latency(retrieval_ms=t_ret, generation_ms=0.0, total_ms=_ms(t_start)),
        model=settings.LLM_MODEL,
    )


def _finalize(answer: str, hits, t_start: float, t_ret: float, t_gen: float, search_query: str | None) -> ChatResponse:
    answer = answer.strip()
    refused = NO_ANSWER in answer
    cited = [] if refused else extract_cited_ids(answer, len(hits))
    return ChatResponse(
        answer=answer, grounded=bool(cited), confidence=_confidence(hits),
        citations=[to_citation(i, hits[i - 1]) for i in cited], retrieved=len(hits),
        refusal_reason="model_declined" if refused else (None if cited else "no_citation"),
        retrieval_mode=_mode_label(hits), search_query=search_query,
        latency_ms=Latency(retrieval_ms=t_ret, generation_ms=t_gen, total_ms=_ms(t_start)),
        model=settings.LLM_MODEL,
    )


async def answer(question: str, top_k: int | None = None, docs: list[str] | None = None) -> ChatResponse:
    top_k = top_k or settings.TOP_K
    t0 = time.perf_counter()
    search_query = await maybe_rewrite(question)
    shown_query = search_query if search_query != question else None
    hits = await retrieve_hits(search_query, top_k, docs)
    t_ret = _ms(t0)
    if _should_refuse(hits):
        return _refusal(hits, "low_retrieval_score", t0, t_ret, shown_query)
    t1 = time.perf_counter()
    text = await llm.generate(SYSTEM_PROMPT, build_user_prompt(question, hits))
    return _finalize(text, hits, t0, t_ret, _ms(t1), shown_query)


async def answer_stream(question: str, top_k: int | None = None,
                        docs: list[str] | None = None) -> AsyncIterator[tuple[str, dict]]:
    """Yields (event_name, data): sources -> token* -> done."""
    top_k = top_k or settings.TOP_K
    t0 = time.perf_counter()
    search_query = await maybe_rewrite(question)
    shown_query = search_query if search_query != question else None
    hits = await retrieve_hits(search_query, top_k, docs)
    t_ret = _ms(t0)
    if _should_refuse(hits):
        yield "sources", {"sources": []}
        yield "token", {"text": NO_ANSWER}
        yield "done", _refusal(hits, "low_retrieval_score", t0, t_ret, shown_query).model_dump()
        return
    yield "sources", {"sources": [to_citation(i, h).model_dump() for i, h in enumerate(hits, 1)]}
    t1 = time.perf_counter()
    parts: list[str] = []
    async for piece in llm.stream(SYSTEM_PROMPT, build_user_prompt(question, hits)):
        parts.append(piece)
        yield "token", {"text": piece}
    yield "done", _finalize("".join(parts), hits, t0, t_ret, _ms(t1), shown_query).model_dump()
