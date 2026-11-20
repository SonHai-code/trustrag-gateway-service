"""HTTP-level tests. Skipped automatically if fastapi is not installed."""
import os

import pytest

pytest.importorskip("fastapi")
os.environ["LLM_PROVIDER"] = "mock"

from fastapi.testclient import TestClient  # noqa: E402

from app import rag, settings  # noqa: E402
from app.main import app  # noqa: E402
from app.retrieval import Hit  # noqa: E402


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")

    async def fake_retrieve(question, top_k, docs=None, timings=None):
        if timings is not None:
            timings["embed_ms"] = 3.0
        return [Hit(0.8, {"doc_name": "399_2026_ND-CP_1", "article": "3", "article_title": "Mức lương tối thiểu",
                          "chapter": "", "clause": "", "text": "Điều 3. ..."})]

    monkeypatch.setattr(rag, "retrieve_hits", fake_retrieve)
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_chat_returns_structured_answer(client):
    body = client.post("/chat", json={"question": "Mức lương tối thiểu là gì?"}).json()
    assert body["grounded"] is True and body["citations"][0]["article"] == "3"
    assert set(body["latency_ms"]) == {"retrieval_ms", "generation_ms", "total_ms"}


def test_chat_validates_input(client):
    assert client.post("/chat", json={"question": "a"}).status_code == 422
    assert client.post("/chat", json={"question": "hợp lệ", "top_k": 99}).status_code == 422


def test_stream_emits_sse_events(client):
    text = client.post("/chat/stream", json={"question": "Mức lương tối thiểu?"}).text
    assert "event: sources" in text and "event: token" in text and text.rstrip().endswith("}")
    assert "event: done" in text


def test_documents_and_docs_filter_are_forwarded(client, monkeypatch):
    from app import retrieval
    monkeypatch.setattr(retrieval, "list_documents", lambda: ["399_2026_ND-CP_1"])
    assert client.get("/documents").json() == {"documents": ["399_2026_ND-CP_1"]}
    body = client.post("/search", json={"question": "lương tối thiểu", "docs": ["399_2026_ND-CP_1"]}).json()
    assert body["results"][0]["article"] == "3"


def test_search_reports_sufficiency_and_can_include_full_text(client):
    body = client.post("/search", json={"question": "lương tối thiểu", "include_text": True}).json()
    assert body["sufficient"] is True and body["best_score"] == 0.8
    assert body["results"][0]["text"].startswith("Điều 3") and body["stages"] == {"embed_ms": 3.0}
    plain = client.post("/search", json={"question": "lương tối thiểu"}).json()
    assert plain["results"][0]["text"] is None


def test_search_flags_insufficient_evidence(client, monkeypatch):
    async def weak(question, top_k, docs=None, timings=None):
        return [Hit(0.1, {"doc_name": "399_2026_ND-CP_1", "article": "9", "text": "x"})]

    monkeypatch.setattr(rag, "retrieve_hits", weak)
    assert client.post("/search", json={"question": "ngoài phạm vi"}).json()["sufficient"] is False


def test_articles_endpoint_joins_the_chunks_of_one_article(client, monkeypatch):
    from app import retrieval
    monkeypatch.setattr(retrieval, "fetch_article", lambda doc, art, limit=20: [
        {"article_title": "Mức lương", "clause": "2", "text": "Điều 3. ...\nKhoản 2"},
        {"article_title": "Mức lương", "clause": "1", "text": "Điều 3. ...\nKhoản 1"},
    ])
    body = client.get("/articles", params={"doc_name": "399_2026_ND-CP_1", "article": "3"}).json()
    assert body["found"] and body["article_title"] == "Mức lương" and body["doc_label"] == "Nghị định 399/2026/NĐ-CP"
    assert body["text"].index("Khoản 1") < body["text"].index("Khoản 2")
    monkeypatch.setattr(retrieval, "fetch_article", lambda *a, **k: [])
    assert client.get("/articles", params={"doc_name": "x", "article": "99"}).json()["found"] is False


def test_internal_api_key_bypasses_the_basic_login(client, monkeypatch):
    monkeypatch.setattr(settings, "BASIC_USER", "demo")
    monkeypatch.setattr(settings, "BASIC_PASSWORD", "secret")
    monkeypatch.setattr(settings, "INTERNAL_API_KEY", "svc-key")
    body = {"question": "lương tối thiểu"}
    assert client.post("/search", json=body).status_code == 401
    assert client.post("/search", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/search", json=body, headers={"X-API-Key": "svc-key"}).status_code == 200
    assert client.get("/health").status_code == 200


def test_agent_routes_return_503_when_not_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "AGENT_URL", "")
    assert client.post("/agent/chat/stream", json={"question": "hợp lệ nhé"}).status_code == 503
    assert client.get("/agent/health").status_code == 503
