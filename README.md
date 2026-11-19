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
| `POST /search` | retrieval only (debugging, metrics) |
| `GET /documents` | `doc_name` values usable in the `docs` filter |
| `GET /health` | liveness |
| `GET /health/ready` | Qdrant point count + whether embedding / reranker services are ready |

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
