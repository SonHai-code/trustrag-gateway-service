"""Retrieval: dense, hybrid (dense + BM25 fused with RRF) and optional reranking, all through remote services.

Pipeline:  question -> embedding service -> [dense top-N] + [BM25 top-N] in Qdrant -> RRF -> (reranker service) -> top-k
`Hit.score` is what the API gates on and shows as confidence:
reranker probability if reranked, otherwise dense cosine similarity (0.0 for keyword-only hits).
If the reranker is slow or down we log a warning and return the un-reranked order: a worse answer beats no answer.
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache

from app import clients, settings
from app.cache import TTLCache, normalize_query

log = logging.getLogger(__name__)
RRF_K = 60


class RetrievalError(RuntimeError):
    pass


_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="qdrant")  # dense + BM25 searches run side by side
_result_cache = TTLCache(maxsize=settings.RETRIEVAL_CACHE_SIZE, ttl=settings.RETRIEVAL_CACHE_TTL)


def cache_stats() -> dict:
    return {"retrieval": _result_cache.stats(), "query_embedding": clients.query_cache_stats()}


def clear_caches() -> None:
    _result_cache.clear()


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


@dataclass
class Hit:
    score: float
    payload: dict
    dense: float | None = None   # cosine similarity from the dense search
    rrf: float | None = None     # fused rank score (hybrid only)
    rerank: float | None = None  # cross-encoder relevance (0-1), if reranked


# ---------- thin wrappers around external systems (easy to fake in tests) ----------

@lru_cache(maxsize=1)
def _client():
    from qdrant_client import QdrantClient
    return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)


def _encode_query(question: str):
    return clients.embed_query(question)  # (dense, (indices, values))


def _build_filter(docs: list[str] | None):
    if not docs:
        return None
    from qdrant_client import models
    return models.Filter(must=[models.FieldCondition(key="doc_name", match=models.MatchAny(any=docs))])


def _query(using: str, vector, limit: int, flt):
    from qdrant_client import models
    query = models.SparseVector(indices=vector[0], values=vector[1]) if using == "bm25" else vector
    return _client().query_points(
        settings.COLLECTION, query=query, using=using, limit=limit, with_payload=True, query_filter=flt
    ).points


def _rerank_scores(query: str, texts: list[str]) -> list[float]:
    return clients.rerank(query, texts)


# ---------- pure logic ----------

def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank). Rank-based, so scores of
    different retrievers (cosine vs BM25) never need to be comparable."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, pid in enumerate(ranking, 1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _fuse(dense_points, sparse_points, hybrid: bool) -> list[Hit]:
    payloads = {str(p.id): (p.payload or {}) for p in dense_points}
    dense = {str(p.id): float(p.score) for p in dense_points}
    if hybrid and sparse_points:
        for p in sparse_points:
            payloads.setdefault(str(p.id), p.payload or {})
        order = rrf_fuse([[str(p.id) for p in dense_points], [str(p.id) for p in sparse_points]])
    else:
        order = [(str(p.id), None) for p in dense_points]
    return [Hit(score=dense.get(pid, 0.0), payload=payloads[pid], dense=dense.get(pid), rrf=rrf) for pid, rrf in order]


def _apply_rerank(question: str, hits: list[Hit]) -> list[Hit]:
    scores = _rerank_scores(question, [h.payload.get("text", "") for h in hits])
    for h, s in zip(hits, scores):
        h.rerank, h.score = s, s
    return sorted(hits, key=lambda h: h.rerank, reverse=True)


# ---------- public API ----------

def retrieve(question: str, top_k: int = 5, docs: list[str] | None = None,
             mode: str | None = None, rerank: bool | None = None, timings: dict | None = None) -> list[Hit]:
    """Blocking (network): run it in a thread from async code.
    `mode`/`rerank` override the settings (the baseline script uses them to compare configurations).
    `timings`, if given, is filled with per-stage milliseconds (embed_ms, vector_ms, rerank_ms) or cache=True."""
    mode = (mode or settings.SEARCH_MODE).lower()
    do_rerank = settings.RERANK if rerank is None else rerank
    hybrid = mode == "hybrid"
    timings = timings if timings is not None else {}
    key = (normalize_query(question), top_k, tuple(sorted(docs or [])), mode, do_rerank)
    cached = _result_cache.get(key)
    if cached is not None:
        timings["cache"] = True
        return list(cached)
    pool = max(top_k, settings.RERANK_CANDIDATES) if do_rerank else (max(top_k * 3, 20) if hybrid else top_k)
    try:
        t = time.perf_counter()
        dense_vec, sparse_vec = _encode_query(question)
        timings["embed_ms"] = _ms(t)
        flt = _build_filter(docs)
        t = time.perf_counter()
        dense_future = _pool.submit(_query, "dense", dense_vec, pool, flt)
        sparse_future = _pool.submit(_query, "bm25", sparse_vec, pool, flt) if hybrid and sparse_vec[0] else None
        dense_points = dense_future.result()
        sparse_points = sparse_future.result() if sparse_future else []
        timings["vector_ms"] = _ms(t)
    except clients.ServiceError as exc:
        raise RetrievalError(str(exc)) from exc
    except Exception as exc:
        raise RetrievalError(f"Qdrant unavailable or collection missing/outdated (re-run ingestion): {exc}") from exc
    hits = _fuse(dense_points, sparse_points, hybrid)
    if do_rerank and hits:
        t = time.perf_counter()
        try:
            hits = _apply_rerank(question, hits[:pool])
        except Exception as exc:  # degrade, do not fail the request
            log.warning("reranker unavailable, returning un-reranked results: %s", exc)
        timings["rerank_ms"] = _ms(t)
    result = hits[:top_k]
    _result_cache.set(key, list(result))
    return result


def fetch_article(doc_name: str, article: str, limit: int = 20) -> list[dict]:
    """Exact lookup (no embedding): every chunk of 'Điều <article>' in one document, in stored order.
    Used when the question names the article ("Điều 5 Nghị định 399 quy định gì?")."""
    try:
        from qdrant_client import models
        flt = models.Filter(must=[
            models.FieldCondition(key="doc_name", match=models.MatchValue(value=doc_name)),
            models.FieldCondition(key="article", match=models.MatchValue(value=str(article))),
        ])
        points, _ = _client().scroll(settings.COLLECTION, scroll_filter=flt, limit=limit, with_payload=True)
    except Exception as exc:
        raise RetrievalError(f"Qdrant unavailable or collection missing: {exc}") from exc
    return [p.payload or {} for p in points]


def list_documents() -> list[str]:
    try:
        names: set[str] = set()
        offset = None
        while True:
            points, offset = _client().scroll(settings.COLLECTION, limit=500, offset=offset, with_payload=["doc_name"])
            names.update(p.payload["doc_name"] for p in points if p.payload)
            if offset is None:
                return sorted(names)
    except Exception as exc:
        raise RetrievalError(f"Qdrant unavailable or collection missing: {exc}") from exc


def count_points() -> int:
    try:
        return _client().count(settings.COLLECTION).count
    except Exception as exc:
        raise RetrievalError(f"Qdrant unavailable or collection missing: {exc}") from exc
