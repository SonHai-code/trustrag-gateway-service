"""Retrieval: dense, hybrid (dense + BM25 fused with RRF) and optional reranking, all through remote services.

Pipeline:  question -> embedding service -> [dense top-N] + [BM25 top-N] in Qdrant -> RRF -> (reranker service) -> top-k
`Hit.score` is what the API gates on and shows as confidence:
reranker probability if reranked, otherwise dense cosine similarity (0.0 for keyword-only hits).
If the reranker is slow or down we log a warning and return the un-reranked order: a worse answer beats no answer.
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import lru_cache

from app import clients, settings
from app.cache import TTLCache, normalize_query
from app.expand import expandable, join_article

log = logging.getLogger(__name__)
RRF_K = 60


class RetrievalError(RuntimeError):
    pass


_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="qdrant")  # dense + BM25 searches run side by side
_result_cache = TTLCache(maxsize=settings.RETRIEVAL_CACHE_SIZE, ttl=settings.RETRIEVAL_CACHE_TTL)
_article_cache = TTLCache(maxsize=settings.RETRIEVAL_CACHE_SIZE, ttl=settings.RETRIEVAL_CACHE_TTL)  # (doc, article) -> chunks


def cache_stats() -> dict:
    return {"retrieval": _result_cache.stats(), "query_embedding": clients.query_cache_stats()}


def clear_caches() -> None:
    global _rerank_down_until
    _result_cache.clear()
    _article_cache.clear()
    _rerank_down_until = 0.0


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


_rerank_down_until = 0.0  # circuit breaker: monotonic time until which the reranker is skipped after a failure


def _apply_rerank(question: str, hits: list[Hit]) -> list[Hit]:
    # the context line (document > chapter > Điều) helps the cross-encoder just as it helps the embedding
    scores = _rerank_scores(question, [(h.payload.get("context", "") + "\n" + h.payload.get("text", "")).strip() for h in hits])
    for h, s in zip(hits, scores):
        h.rerank, h.score = s, s
    return sorted(hits, key=lambda h: h.rerank, reverse=True)


# ---------- public API ----------

def retrieve(question: str, top_k: int = 5, docs: list[str] | None = None,
             mode: str | None = None, rerank: bool | None = None, timings: dict | None = None,
             expand: bool | None = None) -> list[Hit]:
    """Blocking (network): run it in a thread from async code.
    `mode`/`rerank` override the settings (the baseline script uses them to compare configurations).
    `expand` (default: settings.EXPAND_ARTICLES) replaces each hit by its whole Điều (small-to-big).
    `timings`, if given, is filled with per-stage milliseconds (embed_ms, vector_ms, rerank_ms) or cache=True."""
    mode = (mode or settings.SEARCH_MODE).lower()
    do_rerank = settings.RERANK if rerank is None else rerank
    do_expand = settings.EXPAND_ARTICLES if expand is None else expand
    hybrid = mode == "hybrid"
    timings = timings if timings is not None else {}
    key = (normalize_query(question), top_k, tuple(sorted(docs or [])), mode, do_rerank, do_expand)
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
    degraded = False
    if do_rerank and hits:
        global _rerank_down_until
        t = time.perf_counter()
        if time.monotonic() < _rerank_down_until:  # it failed a moment ago: do not make every search wait for it again
            degraded, timings["rerank_skipped"] = True, True
        else:
            try:
                hits = _apply_rerank(question, hits[:pool])
            except Exception as exc:  # degrade, do not fail the request
                _rerank_down_until = time.monotonic() + settings.RERANK_COOLDOWN
                degraded = True
                log.warning("reranker unavailable, returning un-reranked results (skipping it for %.0fs): %s",
                            settings.RERANK_COOLDOWN, exc)
        timings["rerank_ms"] = _ms(t)
    if do_expand:
        t = time.perf_counter()
        hits = _one_hit_per_article(hits)  # several parts of one Điều would otherwise fill the top-k with duplicates
        result = expand_hits(hits[:top_k])
        timings["expand_ms"] = _ms(t)
    else:
        result = hits[:top_k]
    if not degraded:  # never cache a degraded ordering: the next search may have the reranker back
        _result_cache.set(key, list(result))
    return result


def fetch_article(doc_name: str, article: str, limit: int = 64, section: int = 0) -> list[dict]:
    """Exact lookup (no embedding): every chunk of 'Điều <article>' in one document, in stored order.
    Used when the question names the article ("Điều 5 Nghị định 399 quy định gì?").
    A document can restart its numbering (a Quy chế attached to a Quyết định): `section` picks the part, 0 = main text.
    Points ingested before sections existed have no `section` field and count as part 0."""
    try:
        from qdrant_client import models
        must = [models.FieldCondition(key="doc_name", match=models.MatchValue(value=doc_name)),
                models.FieldCondition(key="article", match=models.MatchValue(value=str(article)))]
        must_not = []
        if section > 0:
            must.append(models.FieldCondition(key="section", match=models.MatchValue(value=int(section))))
        else:
            must_not.append(models.FieldCondition(key="section", range=models.Range(gt=0)))
        flt = models.Filter(must=must, must_not=must_not or None)
        points, _ = _client().scroll(settings.COLLECTION, scroll_filter=flt, limit=limit, with_payload=True)
    except Exception as exc:
        raise RetrievalError(f"Qdrant unavailable or collection missing: {exc}") from exc
    return [p.payload or {} for p in points]


def fetch_document_chunks(doc_name: str) -> list[dict]:
    """The structural payload (no text) of every chunk of one document: the input of the outline."""
    try:
        from qdrant_client import models
        flt = models.Filter(must=[models.FieldCondition(key="doc_name", match=models.MatchValue(value=doc_name))])
        out: list[dict] = []
        offset = None
        while True:
            points, offset = _client().scroll(settings.COLLECTION, scroll_filter=flt, limit=500, offset=offset,
                                              with_payload=["doc_name", "article", "article_title", "chapter", "section", "section_title"])
            out.extend(p.payload or {} for p in points)
            if offset is None:
                return out
    except Exception as exc:
        raise RetrievalError(f"Qdrant unavailable or collection missing: {exc}") from exc


def _article_key(payload: dict) -> tuple[str, str, int]:
    try:
        section = int(payload.get("section") or 0)
    except (TypeError, ValueError):
        section = 0
    return str(payload.get("doc_name", "")), str(payload.get("article", "")), section


def _one_hit_per_article(hits: list[Hit]) -> list[Hit]:
    """Keep the best-ranked hit of every multi-part Điều (they will all be expanded to the same text anyway)."""
    seen: set[tuple[str, str, int]] = set()
    out: list[Hit] = []
    for h in hits:
        if expandable(h.payload):
            key = _article_key(h.payload)
            if key in seen:
                continue
            seen.add(key)
        out.append(h)
    return out


def _siblings(key: tuple[str, str, int]) -> list[dict]:
    cached = _article_cache.get(key)
    if cached is not None:
        return cached
    parts = fetch_article(key[0], key[1], section=key[2])
    _article_cache.set(key, parts)
    return parts


def expand_hits(hits: list[Hit]) -> list[Hit]:
    """Small-to-big: a hit on one chunk of a long Điều becomes the whole Điều (the matched part is kept in
    `match_text`, so citations still show what was actually found). A Điều that cannot be fetched stays as it is."""
    wanted = [_article_key(h.payload) for h in hits if expandable(h.payload)]
    if not wanted:
        return list(hits)
    fetched = dict(zip(wanted, _pool.map(_safe_siblings, wanted)))
    budget = settings.EXPAND_TOTAL_CHARS
    out: list[Hit] = []
    for h in hits:
        p = h.payload
        parts = fetched.get(_article_key(p)) if expandable(p) else None
        if not parts or len(parts) < 2:
            out.append(h)
            continue
        own = len(p.get("text", ""))
        limit = max(min(settings.EXPAND_MAX_CHARS, budget), own)
        text, truncated = join_article(parts, p.get("part"), limit)
        budget -= len(text)
        out.append(replace(h, payload={**p, "text": text, "clause": "", "expanded": True, "truncated": truncated,
                                       "match_text": p.get("text", ""), "match_clause": p.get("clause", "")}))
    return out


def _safe_siblings(key: tuple[str, str, int]) -> list[dict] | None:
    try:
        return _siblings(key)
    except RetrievalError as exc:  # keep the small chunk rather than failing the search
        log.warning("could not expand Điều %s of %s: %s", key[1], key[0], exc)
        return None


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
