"""Gateway side of multi-model support: registry, request dialects, learning from a 400, and the chosen model reaching the answer."""
import asyncio
import json

import pytest

from app import llm, models, rag, settings
from app.models import ModelConfigError, ModelSpec
from app.retrieval import Hit

REGISTRY = [
    {"id": "gpt-4o-mini", "provider": "openai", "model": "gpt-4o-mini", "api_key_env": "T_KEY"},
    {"id": "gpt-6.2-sol", "provider": "openai", "model": "gpt-6.2-sol", "api_key_env": "T_KEY"},
    {"id": "claude", "provider": "anthropic", "model": "claude-sonnet-5-5", "api_key_env": "T_KEY"},
    {"id": "mock", "provider": "mock", "model": "mock-model"},
]


@pytest.fixture(autouse=True)
def reg(monkeypatch):
    monkeypatch.setenv("T_KEY", "sk-test")
    monkeypatch.setattr(settings, "LLM_MODELS", json.dumps(REGISTRY))
    monkeypatch.setattr(settings, "LLM_DEFAULT_MODEL", "")
    models.reset()
    yield
    models.reset()


def test_registry_and_allow_list():
    assert [m["id"] for m in models.public_list()] == ["gpt-4o-mini", "gpt-6.2-sol", "claude", "mock"]
    assert "sk-test" not in json.dumps(models.public_list())
    with pytest.raises(ModelConfigError):
        models.resolve("https://evil.example/v1")


def test_request_dialects():
    spec = models.resolve("gpt-4o-mini")
    _, h, b = llm._request(spec, "s", "u", False)
    assert b["max_tokens"] == 1024 and b["temperature"] == 0.0 and h["authorization"] == "Bearer sk-test"
    _, _, new = llm._request(models.resolve("gpt-6.2-sol"), "s", "u", False)
    assert "max_tokens" not in new and "temperature" not in new and new["max_completion_tokens"] == 1024
    url, h, claude = llm._request(models.resolve("claude"), "s", "u", True)
    assert url.endswith("/v1/messages") and h["x-api-key"] == "sk-test" and claude["stream"] is True


def test_generate_learns_from_a_400_and_retries(monkeypatch):
    spec = ModelSpec(id="x", provider="openai", model="mystery", base_url="https://compat.example/v1", api_key="k")
    sent = []

    class Resp:
        def __init__(self, status, payload):
            self.status_code, self._p, self.text = status, payload, json.dumps(payload)
        def json(self): return self._p

    class Client:
        def __init__(self, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, headers=None, json=None):
            sent.append(sorted(k for k in json if k.startswith("max")))
            if "max_tokens" in json:
                return Resp(400, {"error": {"message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."}})
            return Resp(200, {"choices": [{"message": {"content": "xong"}}]})

    monkeypatch.setattr(llm.httpx, "AsyncClient", Client, raising=False)
    assert asyncio.run(llm.generate("s", "u", spec)) == "xong"
    assert sent == [["max_tokens"], ["max_completion_tokens"]]


def test_chosen_model_is_used_reported_and_keeps_its_own_cache_entry(monkeypatch):
    monkeypatch.setattr(settings, "MIN_SCORE", 0.35)
    monkeypatch.setattr(settings, "ANSWER_CACHE_TTL", 0)
    used = []

    async def fake_retrieve(q, k, docs=None, timings=None):
        return [Hit(0.9, {"doc_name": "399_2026_ND-CP_1", "article": "3", "article_title": "t", "chapter": "", "clause": "", "text": "Điều 3."})]

    async def fake_generate(system, user, spec=None):
        used.append(spec.id)
        return "Đáp án [1]."

    monkeypatch.setattr(rag, "retrieve_hits", fake_retrieve)
    monkeypatch.setattr(llm, "generate", fake_generate)
    a = asyncio.run(rag.answer("mức lương tối thiểu?", model="gpt-6.2-sol"))
    b = asyncio.run(rag.answer("mức lương tối thiểu?"))
    assert used == ["gpt-6.2-sol", "gpt-4o-mini"] and a.model == "gpt-6.2-sol" and b.model == "gpt-4o-mini"
    assert rag._answer_key("q", 5, None, "a") != rag._answer_key("q", 5, None, "b")
    with pytest.raises(ModelConfigError):
        asyncio.run(rag.answer("mức lương tối thiểu?", model="nope"))
