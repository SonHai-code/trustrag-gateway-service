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

from app import auth, models, rag, settings, userproxy
from app.llm import LLMConfigError, LLMError
from app.prompts import doc_label
from app.retrieval import RetrievalError
from app.schemas import AgentChatRequest, ArticleResponse, ChatRequest, ChatResponse, SearchRequest, SearchResponse

log = logging.getLogger(__name__)


async def _keep_warm_loop():
    """Optional (KEEPWARM_SECONDS > 0): a tiny request every N seconds so the embedding replica stays loaded."""
    from app import clients
    while True:
        await asyncio.sleep(settings.KEEPWARM_SECONDS)
        ok = await asyncio.to_thread(clients.keep_warm)
        if not ok:
            log.warning("keep-warm: embedding service did not answer")


_user_http: httpx.AsyncClient | None = None  # shared connection pool towards the user service
_limiter = auth.FixedWindowLimiter()


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    if settings.AUTH_REQUIRED and len(settings.JWT_SECRET) < 32:
        # fail at start-up with a clear message instead of rejecting every login with a confusing 401
        raise RuntimeError("AUTH_REQUIRED=true needs JWT_SECRET (>= 32 characters, the same value as the user service)")
    task = asyncio.create_task(_keep_warm_loop()) if settings.KEEPWARM_SECONDS > 0 else None
    try:
        yield
    finally:
        if task:
            task.cancel()
        global _user_http
        if _user_http is not None:
            await _user_http.aclose()
            _user_http = None


app = FastAPI(title="TrustRAG gateway", version="1.1.0", lifespan=lifespan)


LLM_PATHS = {"/chat", "/chat/stream", "/agent/chat/stream", "/search"}  # the routes that cost money: rate-limited per user
ADMIN_PATHS = {"/cache/stats"}


def _client_ip(request: Request) -> str:
    peer = request.client.host if request.client else ""
    return auth.client_ip(request.headers.get("x-forwarded-for", ""), peer, settings.TRUSTED_PROXY_HOPS)


def _deny(status: int, detail: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": detail}, headers=headers or {})


def _is_internal(request: Request) -> bool:
    key = request.headers.get("x-api-key", "")
    return bool(settings.INTERNAL_API_KEY and key and secrets.compare_digest(key.encode(), settings.INTERNAL_API_KEY.encode()))


def _basic_ok(request: Request) -> bool:
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("basic "):
        return False
    try:
        user, _, password = base64.b64decode(header[6:]).decode("utf-8").partition(":")
        return secrets.compare_digest(user.encode(), settings.BASIC_USER.encode()) and secrets.compare_digest(
            password.encode(), settings.BASIC_PASSWORD.encode())
    except Exception:
        return False


@app.middleware("http")
async def access_control(request: Request, call_next):
    """Who may call what. With AUTH_REQUIRED=true every route needs a valid access token from the user service (except
    /health and the sign-in endpoints); the older shared Basic login still works when AUTH_REQUIRED is off."""
    path = request.url.path
    if path == "/health" or _is_internal(request):  # /health: probes; internal key: the MCP server and the agent
        return await call_next(request)

    if settings.AUTH_REQUIRED:
        if path in userproxy.PUBLIC_PATHS:
            return await call_next(request)
        try:
            who = auth.verify_token(auth.bearer_token(request.headers.get("authorization", "")), settings.JWT_SECRET,
                                    settings.JWT_ISSUER, settings.JWT_AUDIENCE)
        except auth.AuthError:
            return _deny(401, "Bạn cần đăng nhập.", {"WWW-Authenticate": "Bearer"})
        request.state.user = who
        if path in ADMIN_PATHS and not who.is_admin:
            return _deny(403, "Bạn không có quyền thực hiện thao tác này.")
        if path in LLM_PATHS:
            for tag, limit, window, msg in (("m", settings.RATE_LIMIT_PER_MIN, 60, "Bạn gửi câu hỏi quá nhanh. Hãy đợi một chút rồi thử lại."),
                                            ("d", settings.RATE_LIMIT_PER_DAY, 86400, "Bạn đã dùng hết số câu hỏi cho hôm nay.")):
                ok, retry = _limiter.allow(f"{tag}:{who.user_id}", limit, window)
                if not ok:
                    return _deny(429, msg, {"Retry-After": str(retry)})
        return await call_next(request)

    if settings.BASIC_USER and settings.BASIC_PASSWORD and not _basic_ok(request):
        return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="TrustRAG"'})
    return await call_next(request)


@app.exception_handler(RetrievalError)
async def _retrieval_error(_, exc: RetrievalError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(LLMConfigError)
async def _llm_config_error(_, exc: LLMConfigError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(LLMError)
async def _llm_error(_, exc: LLMError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(models.ModelConfigError)
async def _model_error(_, exc: models.ModelConfigError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/models")
async def list_models():
    """The models a user may choose (ids and labels only: never URLs or keys). The same ids are accepted by /chat and /agent/*."""
    return {"default": models.default_id(), "models": models.public_list()}


@app.get("/health")
async def health():
    spec = models.resolve(None)
    return {"status": "ok", "llm_provider": spec.provider, "model": spec.model, "models": len(models.registry())}


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
    return await rag.answer(req.question, req.top_k, req.docs, req.model)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    models.resolve(req.model)  # refuse an unknown model with a clean 400 before any streaming starts
    async def events():
        try:
            async for name, data in rag.answer_stream(req.question, req.top_k, req.docs, req.model):
                yield _sse(name, data)
        except (RetrievalError, LLMError) as exc:  # headers are already sent: report in-band
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
async def agent_chat_stream(req: AgentChatRequest):
    models.resolve(req.model)  # the same allow-list: the browser can only name a model, never an endpoint or a key
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


# ---------------- accounts: forward /auth, /users, /conversations, /admin to trustrag-user-service ----------------
# The browser keeps talking to one origin. The user service verifies the token again itself, so it stays safe on its own.

@app.api_route("/auth/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"], include_in_schema=False)
@app.api_route("/users/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"], include_in_schema=False)
@app.api_route("/admin/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"], include_in_schema=False)
@app.api_route("/conversations", methods=["GET", "POST", "DELETE"], include_in_schema=False)
@app.api_route("/conversations/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"], include_in_schema=False)
async def user_service(request: Request):
    global _user_http
    path = request.url.path
    if not userproxy.is_user_path(path):
        return _deny(404, "Not Found")
    if not settings.USER_SERVICE_URL:
        return _deny(503, "Dịch vụ tài khoản chưa được cấu hình (USER_SERVICE_URL).")
    body = await request.body()
    if len(body) > 1 << 20:
        return _deny(413, "Dữ liệu gửi lên quá lớn.")
    if _user_http is None:
        _user_http = httpx.AsyncClient(timeout=settings.USER_TIMEOUT, limits=httpx.Limits(max_connections=50, max_keepalive_connections=10))
    headers = userproxy.forward_headers(request.headers, _client_ip(request), settings.USER_API_KEY)
    try:
        r = await _user_http.request(request.method, f"{settings.USER_SERVICE_URL.rstrip('/')}{path}",
                                     params=request.query_params, content=body, headers=headers)
    except httpx.HTTPError as exc:
        log.warning("user service unreachable: %s", exc)
        return _deny(503, "Dịch vụ tài khoản đang khởi động hoặc tạm thời không truy cập được. Hãy thử lại sau ít giây.")
    out = Response(content=r.content, status_code=r.status_code)
    for k, v in userproxy.response_headers(r.headers.multi_items()):
        out.headers.append(k, v)
    return out
