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

    async def fake_retrieve(question, top_k, docs=None):
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
