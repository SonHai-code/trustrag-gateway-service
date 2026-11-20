import httpx
import pytest

from app import clients, settings


class Resp:
    def __init__(self, status, body=None):
        self.status_code, self._body, self.text = status, body or {}, str(body)

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=self)


class FakeHttp:
    """Stands in for the shared httpx.Client."""
    def __init__(self, post):
        self.post = post


def _use(monkeypatch, post):
    monkeypatch.setattr(clients, "_http", lambda: FakeHttp(post))


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    clients._query_cache.clear()
    monkeypatch.setattr(clients._query_cache, "maxsize", 512)
    monkeypatch.setattr(clients.time, "sleep", lambda s: None)
    monkeypatch.setattr(settings, "EMBEDDING_URL", "http://embedding")
    monkeypatch.setattr(settings, "RERANKER_URL", "http://reranker")


def test_embed_query_retries_while_the_service_warms_up(monkeypatch):
    answers = iter([Resp(503), Resp(200, {"dense": [[0.5]], "sparse": [{"indices": [3], "values": [1.0]}]})])
    seen = {}

    def fake_post(url, json, headers, timeout):
        seen["url"], seen["json"] = url, json
        return next(answers)

    _use(monkeypatch, fake_post)
    dense, sparse = clients.embed_query("mức lương")
    assert dense == [0.5] and sparse == ([3], [1.0])
    assert seen["url"] == "http://embedding/embed" and seen["json"] == {"texts": ["mức lương"], "kind": "query"}


def test_rerank_returns_scores(monkeypatch):
    _use(monkeypatch, lambda *a, **k: Resp(200, {"scores": [0.9, 0.1]}))
    assert clients.rerank("q", ["a", "b"]) == [0.9, 0.1]


def test_client_error_is_not_retried(monkeypatch):
    _use(monkeypatch, lambda *a, **k: Resp(422, {"detail": "bad"}))
    with pytest.raises(clients.ServiceError, match="422"):
        clients.embed_query("x")


def test_repeated_question_is_served_from_the_query_cache(monkeypatch):
    calls = []

    def fake_post(url, json, headers, timeout):
        calls.append(json)
        return Resp(200, {"dense": [[0.5]], "sparse": [{"indices": [3], "values": [1.0]}]})

    _use(monkeypatch, fake_post)
    first = clients.embed_query("Mức lương tối thiểu?")
    again = clients.embed_query("  mức LƯƠNG   tối thiểu ")  # same question: case, spaces, '?' do not matter
    assert first == again and len(calls) == 1
    assert clients.query_cache_stats()["hits"] >= 1


def test_query_cache_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(clients._query_cache, "maxsize", 0)
    calls = []

    def fake_post(url, json, headers, timeout):
        calls.append(1)
        return Resp(200, {"dense": [[0.5]], "sparse": [{"indices": [3], "values": [1.0]}]})

    _use(monkeypatch, fake_post)
    clients.embed_query("câu hỏi")
    clients.embed_query("câu hỏi")
    assert len(calls) == 2
