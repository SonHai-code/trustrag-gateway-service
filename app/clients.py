"""HTTP clients for the embedding and reranker services. Blocking: call from a thread (asyncio.to_thread)."""
import logging
import time

import httpx

from app import settings

log = logging.getLogger(__name__)


class ServiceError(RuntimeError):
    pass


def _post(name: str, base_url: str, path: str, payload: dict, api_key: str, timeout: float, wait: float) -> dict:
    """POST with retries while the service is starting (503) or waking up from zero replicas (connection errors)."""
    headers = {"X-API-Key": api_key} if api_key else {}
    deadline = time.monotonic() + wait
    delay = 1.0
    while True:
        try:
            r = httpx.post(f"{base_url.rstrip('/')}{path}", json=payload, headers=headers, timeout=timeout)
            if r.status_code in (502, 503, 504) and time.monotonic() < deadline:
                time.sleep(delay)
                delay = min(delay * 2, 8)
                continue
            r.raise_for_status()
            return r.json()
        except httpx.TransportError as exc:
            if time.monotonic() >= deadline:
                raise ServiceError(f"{name} unreachable: {exc}") from exc
            time.sleep(delay)
            delay = min(delay * 2, 8)
        except httpx.HTTPStatusError as exc:
            raise ServiceError(f"{name} error {exc.response.status_code}: {exc.response.text[:200]}") from exc


def embed_query(question: str) -> tuple[list[float], tuple[list[int], list[float]]]:
    """-> (dense vector, (sparse indices, sparse values)) for the question."""
    body = _post("embedding service", settings.EMBEDDING_URL, "/embed", {"texts": [question], "kind": "query"},
                 settings.EMBEDDING_API_KEY, settings.DOWNSTREAM_TIMEOUT, settings.WARMUP_WAIT)
    s = body["sparse"][0]
    return body["dense"][0], (s["indices"], s["values"])


def rerank(query: str, texts: list[str]) -> list[float]:
    """Relevance 0-1 per text. Short wait on purpose: the caller degrades gracefully if this fails."""
    body = _post("reranker service", settings.RERANKER_URL, "/rerank", {"query": query, "documents": texts},
                 settings.RERANKER_API_KEY, settings.RERANK_TIMEOUT, wait=3.0)
    return [float(s) for s in body["scores"]]


def dependency_status() -> dict:
    """Is each downstream service ready? (None = not configured)"""
    def ready(url: str) -> bool:
        try:
            return httpx.get(f"{url.rstrip('/')}/health/ready", timeout=3).status_code == 200
        except httpx.HTTPError:
            return False

    return {
        "embedding": ready(settings.EMBEDDING_URL),
        "reranker": ready(settings.RERANKER_URL) if settings.RERANKER_URL else None,
    }
