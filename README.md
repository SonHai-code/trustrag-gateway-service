# trustrag-gateway-service

The public brain of TrustRAG: receives a question, retrieves evidence, asks the LLM, verifies citations.
It holds **no ML model** (~100 MB image), so it starts in seconds and scales cheaply.

```
question -> embedding service (dense + sparse query vectors)
         -> Qdrant: dense top-N + BM25 top-N -> RRF fusion
         -> reranker service (optional; skipped automatically if it is down)
         -> gate (refuse when evidence is weak) -> LLM -> [n] citations
```

| Endpoint | Purpose |
|---|---|
| `POST /chat` | answer + citations + confidence + latency |
| `POST /chat/stream` | the same as Server-Sent Events: `sources`, `token`*, `done` |
| `POST /search` | retrieval only; `include_text=true` returns full passages; `sufficient` / `best_score` say whether `/chat` would answer |
| `GET /articles?doc_name=&article=` | exact lookup of one "Điều" (no embedding): used by the agent's `get_article` tool |
| `POST /agent/chat/stream` | forwards to the agent service (Phase 4) so the browser has one URL and one login; 503 if `AGENT_URL` is empty |
| `GET /cache/stats` | hit/miss counters of the query-embedding, retrieval and answer caches |
| `GET /documents` | `doc_name` values usable in the `docs` filter |
| `GET /health` | liveness |
| `GET /health/ready` | Qdrant point count + whether embedding / reranker services are ready |

Every response carries `latency_ms.ttft_ms` (time to first token) and `latency_ms.stages` (`embed_ms`, `vector_ms`,
`rerank_ms`, `rewrite_ms`): that is how you see what to tune (`trustrag-infra/docs/LATENCY.md`).

Speed features (all in-process, no extra service): keep-alive connection pool, query-embedding cache, dense and BM25
searches in parallel, retrieval-result cache (5 min), optional answer cache (`ANSWER_CACHE_TTL`, off by default),
optional keep-warm ping. Service-to-service calls (MCP server, agent) authenticate with `X-API-Key = INTERNAL_API_KEY`
instead of the browser's Basic login.

Failure behaviour: embedding service down -> 503 with a clear message; reranker down/slow -> results are returned
un-reranked (a warning is logged); LLM error -> 502.

```bash
cp .env.example .env
pip install -r requirements-dev.txt && pytest -q
uvicorn app.main:app --reload --port 8000
```

## Evaluation
```bash
python -m scripts.baseline --label cloud --matrix        # dense / hybrid / +rerank, appends to eval/baseline_results.md
```
Needs the embedding service (and reranker for the `+rerank` rows) to be reachable via `EMBEDDING_URL` / `RERANKER_URL`.
