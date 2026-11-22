"""Settings, read from environment variables (see .env.example)."""
import os

# ---- LLM ----
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()  # anthropic | openai | mock
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
_DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4o-mini", "mock": "mock"}
LLM_MODEL = os.getenv("LLM_MODEL") or _DEFAULT_MODELS.get(LLM_PROVIDER, "")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
# Several selectable models (see app/models.py): a JSON list. Empty = the single model defined by the LLM_* settings above.
LLM_MODELS = os.getenv("LLM_MODELS", "")
LLM_DEFAULT_MODEL = os.getenv("LLM_DEFAULT_MODEL", "")   # id from LLM_MODELS used when the client does not choose
# JSON merged into every OpenAI-compatible request body (vLLM extras), e.g. {"chat_template_kwargs": {"enable_thinking": false}}
LLM_EXTRA_BODY = os.getenv("LLM_EXTRA_BODY", "")

# ---- Retrieval ----
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.35"))  # refusal threshold on dense cosine
SEARCH_MODE = os.getenv("SEARCH_MODE", "hybrid").lower()  # dense | hybrid
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0.05"))  # refusal threshold on reranker probability
REWRITE_QUERY = os.getenv("REWRITE_QUERY", "false").lower() == "true"

# ---- Downstream services ----
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION = os.getenv("QDRANT_COLLECTION", "legal_docs")
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://localhost:8001")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
RERANKER_URL = os.getenv("RERANKER_URL", "")  # empty => no reranking
RERANKER_API_KEY = os.getenv("RERANKER_API_KEY", "")
RERANK = os.getenv("RERANK", "true").lower() == "true" and bool(RERANKER_URL)
DOWNSTREAM_TIMEOUT = float(os.getenv("DOWNSTREAM_TIMEOUT", "30"))  # seconds per call
RERANK_COOLDOWN = float(os.getenv("RERANK_COOLDOWN", "60"))  # after a reranker failure, skip it this many seconds (no 8 s wait per search)
RERANK_TIMEOUT = float(os.getenv("RERANK_TIMEOUT", "8"))  # reranking is optional: do not wait long for it
WARMUP_WAIT = float(os.getenv("WARMUP_WAIT", "45"))  # keep retrying while a service wakes up from zero replicas

# ---- Speed (see docs/LATENCY in the infra repo) ----
QUERY_CACHE_SIZE = int(os.getenv("QUERY_CACHE_SIZE", "512"))  # query embeddings kept in memory (0 = off)
RETRIEVAL_CACHE_SIZE = int(os.getenv("RETRIEVAL_CACHE_SIZE", "256"))  # retrieval results kept in memory (0 = off)
RETRIEVAL_CACHE_TTL = float(os.getenv("RETRIEVAL_CACHE_TTL", "300"))  # seconds; re-ingestion becomes visible after this
ANSWER_CACHE_TTL = float(os.getenv("ANSWER_CACHE_TTL", "0"))  # seconds a finished answer is replayed (0 = off)
ANSWER_CACHE_SIZE = int(os.getenv("ANSWER_CACHE_SIZE", "128"))
KEEPWARM_SECONDS = float(os.getenv("KEEPWARM_SECONDS", "0"))  # >0: ping the embedding service this often (0 = off)
HTTP_POOL_SIZE = int(os.getenv("HTTP_POOL_SIZE", "20"))  # keep-alive connections to embedding/reranker

# ---- Agent (Phase 4) ----
AGENT_URL = os.getenv("AGENT_URL", "")  # e.g. http://trustrag-agent ; empty = /agent/* returns 503
AGENT_API_KEY = os.getenv("AGENT_API_KEY", "")  # sent to the agent service as X-API-Key
AGENT_TIMEOUT = float(os.getenv("AGENT_TIMEOUT", "120"))

# ---- Protection ----
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "")  # services (MCP server, agent) send it as X-API-Key instead of the Basic login
BASIC_USER = os.getenv("BASIC_USER", "")
BASIC_PASSWORD = os.getenv("BASIC_PASSWORD", "")  # both set => every route except /health needs a login
