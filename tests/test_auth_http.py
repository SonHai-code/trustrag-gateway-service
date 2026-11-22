"""HTTP-level tests of the login wall and the user-service proxy. Skipped automatically if fastapi is not installed."""
import base64
import os

import pytest

pytest.importorskip("fastapi")
os.environ.setdefault("LLM_PROVIDER", "mock")

from fastapi.testclient import TestClient  # noqa: E402

from app import auth, main, models, rag, settings  # noqa: E402
from app.retrieval import Hit  # noqa: E402
from tests.test_auth import SECRET, make_token  # noqa: E402


class _FakeUserHTTP:
    def __init__(self):
        self.calls = []

    async def request(self, method, url, params=None, content=None, headers=None):
        self.calls.append({"method": method, "url": url, "headers": headers, "content": content})

        class R:
            status_code = 200
            content = b'{"ok": true}'

            class headers:  # noqa: N801
                @staticmethod
                def multi_items():
                    return [("content-type", "application/json"), ("set-cookie", "trustrag_rt=abc; Path=/api/auth; HttpOnly"), ("set-cookie", "x=1")]
        return R

    async def aclose(self):
        pass


@pytest.fixture()
def app_client(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")
    monkeypatch.setattr(settings, "LLM_MODELS", "")
    monkeypatch.setattr(settings, "AUTH_REQUIRED", True)
    monkeypatch.setattr(settings, "JWT_SECRET", SECRET)
    monkeypatch.setattr(settings, "USER_SERVICE_URL", "http://trustrag-user")
    monkeypatch.setattr(settings, "USER_API_KEY", "svc-key")
    monkeypatch.setattr(settings, "RATE_LIMIT_PER_MIN", 2)
    monkeypatch.setattr(settings, "INTERNAL_API_KEY", "internal-key")
    monkeypatch.setattr(main, "_limiter", auth.FixedWindowLimiter())
    fake = _FakeUserHTTP()
    monkeypatch.setattr(main, "_user_http", fake)
    models.reset()

    async def fake_retrieve(question, top_k, docs=None, timings=None):
        return [Hit(0.8, {"doc_name": "d", "article": "3", "article_title": "t", "chapter": "", "clause": "", "text": "Điều 3."})]

    monkeypatch.setattr(rag, "retrieve_hits", fake_retrieve)
    c = TestClient(main.app)
    c.fake = fake
    return c


def _tok(**kw):
    import time as _t
    return make_token(kw, now=int(_t.time()))


def test_protected_routes_need_a_token(app_client):
    for path in ("/models", "/documents"):
        r = app_client.get(path)
        assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer" and "detail" in r.json()
    assert app_client.get("/health").status_code == 200
    assert app_client.get("/models", headers={"Authorization": "Bearer " + _tok()}).status_code == 200
    head, payload, sig = _tok().split(".")
    forged = f"{head}.{payload}.{'B' if sig[0] == 'A' else 'A'}{sig[1:]}"  # first signature character changed: always a different MAC
    assert app_client.get("/models", headers={"Authorization": "Bearer " + forged}).status_code == 401


def test_internal_key_still_bypasses_the_wall(app_client):
    assert app_client.get("/models", headers={"x-api-key": "internal-key"}).status_code == 200
    assert app_client.get("/models", headers={"x-api-key": "wrong"}).status_code == 401


def test_llm_routes_are_rate_limited_per_user(app_client):
    a, b = {"Authorization": "Bearer " + _tok(sub="a")}, {"Authorization": "Bearer " + _tok(sub="b")}
    q = {"question": "Mức lương tối thiểu là gì?"}
    assert [app_client.post("/chat", json=q, headers=a).status_code for _ in range(2)] == [200, 200]
    r = app_client.post("/chat", json=q, headers=a)
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert app_client.post("/chat", json=q, headers=b).status_code == 200


def test_admin_only_route(app_client):
    user = {"Authorization": "Bearer " + _tok(role="user")}
    admin = {"Authorization": "Bearer " + _tok(role="admin")}
    assert app_client.get("/cache/stats", headers=user).status_code == 403
    assert app_client.get("/cache/stats", headers=admin).status_code == 200


def test_login_is_public_and_proxied_with_cookies_and_client_ip(app_client):
    r = app_client.post("/auth/login", json={"email": "a@example.com", "password": "x"},
                        headers={"X-Forwarded-For": "6.6.6.6, 203.0.113.9, 10.0.0.4", "x-api-key": "browser-sent"})
    assert r.status_code == 200
    call = app_client.fake.calls[-1]
    assert call["url"] == "http://trustrag-user/auth/login" and call["method"] == "POST"
    assert call["headers"]["x-client-ip"] == "203.0.113.9" and call["headers"]["x-api-key"] == "svc-key"
    cookies = r.headers.get_list("set-cookie")
    assert any(c.startswith("trustrag_rt=abc") for c in cookies) and any(c.startswith("x=1") for c in cookies)


def test_user_routes_need_a_token_at_the_gateway_too(app_client):
    assert app_client.get("/conversations").status_code == 401
    assert app_client.get("/users/me").status_code == 401
    assert not app_client.fake.calls
    r = app_client.get("/conversations", headers={"Authorization": "Bearer " + _tok()})
    assert r.status_code == 200 and app_client.fake.calls[-1]["url"] == "http://trustrag-user/conversations"


def test_user_service_not_configured(app_client, monkeypatch):
    monkeypatch.setattr(settings, "USER_SERVICE_URL", "")
    assert app_client.post("/auth/login", json={}).status_code == 503


def test_basic_login_still_works_when_auth_is_off(app_client, monkeypatch):
    monkeypatch.setattr(settings, "AUTH_REQUIRED", False)
    monkeypatch.setattr(settings, "BASIC_USER", "u")
    monkeypatch.setattr(settings, "BASIC_PASSWORD", "p")
    assert app_client.get("/models").status_code == 401
    good = "Basic " + base64.b64encode(b"u:p").decode()
    assert app_client.get("/models", headers={"Authorization": good}).status_code == 200


def test_startup_refuses_a_short_secret(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_REQUIRED", True)
    monkeypatch.setattr(settings, "JWT_SECRET", "short")
    with pytest.raises(RuntimeError):
        with TestClient(main.app):
            pass
