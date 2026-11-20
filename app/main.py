"""Gateway API.   Run:  uvicorn app.main:app --reload --port 8000   (docs at /docs)"""
import asyncio
import base64
import contextlib
import json
import logging
import secrets
import time

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from app import rag, settings
from app.llm import LLMConfigError, LLMError
from app.prompts import doc_label
from app.retrieval import RetrievalError
from app.schemas import ArticleResponse, ChatRequest, ChatResponse, SearchRequest, SearchResponse

log = logging.getLogger(__name__)


async def _keep_warm_loop():
    """Optional (KEEPWARM_SECONDS > 0): a tiny request every N seconds so the embedding replica stays loaded."""
    from app import clients
    while True:
        await asyncio.sleep(settings.KEEPWARM_SECONDS)
        ok = await asyncio.to_thread(clients.keep_warm)
        if not ok:
            log.warning("keep-warm: embedding service did not answer")


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    task = asyncio.create_task(_keep_warm_loop()) if settings.KEEPWARM_SECONDS > 0 else None
    try:
        yield
    finally:
        if task:
            task.cancel()


app = FastAPI(title="TrustRAG gateway", version="1.1.0", lifespan=lifespan)


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    """Optional password wall so a public URL cannot burn your LLM budget (real auth: Phase 7)."""
    if not (settings.BASIC_USER and settings.BASIC_PASSWORD) or request.url.path == "/health":
        return await call_next(request)
    key = request.headers.get("x-api-key", "")
    if settings.INTERNAL_API_KEY and key and secrets.compare_digest(key.encode(), settings.INTERNAL_API_KEY.encode()):
        return await call_next(request)  # the MCP server / agent: service-to-service, no browser login
    ok = False
    header = request.headers.get("authorization", "")
    if header.lower().startswith("basic "):
        try:
            user, _, password = base64.b64decode(header[6:]).decode("utf-8").partition(":")
            ok = secrets.compare_digest(user.encode(), settings.BASIC_USER.encode()) and secrets.compare_digest(
                password.encode(), settings.BASIC_PASSWORD.encode()
            )
        except Exception:
            ok = False
    if not ok:
        return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="TrustRAG"'})
    return await call_next(request)


@app.exception_handler(RetrievalError)
async def _retrieval_error(request, exc: RetrievalError):
    log.error("retrieval failed on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(LLMConfigError)
async def _llm_config_error(_, exc: LLMConfigError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(LLMError)
async def _llm_error(_, exc: LLMError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.get("/health")
async def health():
    return {"status": "ok", "llm_provider": settings.LLM_PROVIDER, "model": settings.LLM_MODEL}


@app.get("/health/ready")
async def ready():
    import asyncio
    from app import clients
    from app.retrieval import count_points
    deps = await asyncio.to_thread(clients.dependency_status)
    return {"status": "ready", "points": await asyncio.to_thread(count_points), **deps}


@app.get("/documents")
async def documents():
    """doc_name values you can pass in the `docs` filter of /search and /chat."""
    import asyncio
    from app.retrieval import list_documents
    return {"documents": await asyncio.to_thread(list_documents)}


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """Retrieval only (no LLM): debugging, baseline metrics and the agent's `search` tool.
    `sufficient` tells the caller whether /chat would answer or refuse, using the same gate."""
    t0 = time.perf_counter()
    stages: dict = {}
    hits = await rag.retrieve_hits(req.question, req.top_k, req.docs, stages)
    results = [rag.to_citation(i, h) for i, h in enumerate(hits, 1)]
    if req.include_text:
        for c, h in zip(results, hits):
            c.text = h.payload.get("text", "")
    return SearchResponse(
        query=req.question, results=results,
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        best_score=rag._confidence(hits), sufficient=not rag._should_refuse(hits),
        retrieval_mode=rag._mode_label(hits), stages=rag._clean(stages),
    )


@app.get("/articles", response_model=ArticleResponse)
async def article(doc_name: str, article: str):
    """Exact lookup of 'Điều <article>' in one document (no embedding, ~ms). Lets the agent answer
    questions that name the article directly."""
    from app.retrieval import fetch_article
    payloads = await asyncio.to_thread(fetch_article, doc_name, article)
    payloads.sort(key=lambda p: str(p.get("clause", "")))
    text = "\n\n".join(p.get("text", "") for p in payloads)
    title = next((p.get("article_title", "") for p in payloads if p.get("article_title")), "")
    return ArticleResponse(doc_name=doc_name, doc_label=doc_label(doc_name), article=article,
                           article_title=title, found=bool(payloads), text=text)


@app.get("/cache/stats")
async def cache_stats():
    from app.retrieval import cache_stats as stats
    return {**stats(), "answer": rag._answer_cache.stats()}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    return await rag.answer(req.question, req.top_k, req.docs)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def events():
        try:
            async for name, data in rag.answer_stream(req.question, req.top_k, req.docs):
                yield _sse(name, data)
        except (RetrievalError, LLMError) as exc:  # headers are already sent: report in-band
            log.error("stream failed: %s", exc)
            yield _sse("error", {"detail": str(exc)})

    return StreamingResponse(
        events(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------- Phase 4: forward /agent/* to the agent service ----------------
# The browser only ever talks to the gateway (one login, one URL). The agent is optional: no AGENT_URL = 503.

async def _agent_stream(path: str, body: bytes):
    headers = {"content-type": "application/json"}
    if settings.AGENT_API_KEY:
        headers["x-api-key"] = settings.AGENT_API_KEY
    try:
        async with httpx.AsyncClient(timeout=settings.AGENT_TIMEOUT) as client:
            async with client.stream("POST", f"{settings.AGENT_URL.rstrip('/')}{path}", content=body, headers=headers) as r:
                if r.status_code >= 400:
                    detail = (await r.aread()).decode("utf-8", "ignore")[:300]
                    yield _sse("error", {"detail": f"agent returned {r.status_code}: {detail}"})
                    return
                async for chunk in r.aiter_raw():
                    yield chunk
    except httpx.HTTPError as exc:
        yield _sse("error", {"detail": f"agent unreachable: {exc}"})


@app.post("/agent/chat/stream")
async def agent_chat_stream(req: ChatRequest):
    if not settings.AGENT_URL:
        return JSONResponse(status_code=503, content={"detail": "Agent service is not configured (set AGENT_URL)."})
    return StreamingResponse(
        _agent_stream("/chat/stream", req.model_dump_json().encode()), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/agent/health")
async def agent_health():
    if not settings.AGENT_URL:
        return JSONResponse(status_code=503, content={"detail": "Agent service is not configured (set AGENT_URL)."})
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{settings.AGENT_URL.rstrip('/')}/health/ready")
        return JSONResponse(status_code=r.status_code, content=r.json())
    except httpx.HTTPError as exc:
        return JSONResponse(status_code=503, content={"detail": f"agent unreachable: {exc}"})
