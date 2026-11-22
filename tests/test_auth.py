"""Pure-function tests for the login wall (no fastapi needed). The HTTP-level tests are in test_auth_http.py."""
import base64
import hashlib
import hmac
import json
import os

import pytest

from app import auth, userproxy

SECRET = "k" * 40
# Issued by the Go user service (Sign with these claims, see trustrag-user-service/internal/auth): proves both sides agree.
GO_TOKEN = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1LTEyMyIsImVtYWlsIjoiYW5AZXhhbXBsZS5jb20iLCJuYW1lIjoiTmd1eeG7hW4gQW4i"
            "LCJyb2xlIjoiYWRtaW4iLCJzaWQiOiJzLTkiLCJpc3MiOiJ0cnVzdHJhZy11c2VyIiwiYXVkIjoidHJ1c3RyYWciLCJpYXQiOjE3MDAwMDAwMDAsImV4cCI6MTcw"
            "MDAwMDkwMH0.HlTj3S9LbP6V8Aj7dQyLspfp6iT5PzvpS9z-B3hlCrI")
T0 = 1_700_000_000


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def make_token(claims: dict | None = None, secret: str = SECRET, header: dict | None = None, now: int = T0) -> str:
    c = {"sub": "u1", "email": "a@example.com", "name": "An", "role": "user", "sid": "s1", "iss": "trustrag-user",
         "aud": "trustrag", "iat": now, "exp": now + 900}
    c.update(claims or {})
    h = _b64(json.dumps(header or {"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    p = _b64(json.dumps(c, separators=(",", ":")).encode())
    sig = hmac.new(secret.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
    return f"{h}.{p}.{_b64(sig)}"


def verify(token, now=T0 + 100):
    return auth.verify_token(token, SECRET, "trustrag-user", "trustrag", now=now)


def test_accepts_a_token_signed_by_the_go_service():
    who = verify(GO_TOKEN)
    assert (who.user_id, who.email, who.name, who.role, who.session_id) == ("u-123", "an@example.com", "Nguyễn An", "admin", "s-9")
    assert who.is_admin


def test_roundtrip_and_defaults():
    who = verify(make_token({"role": "user"}))
    assert who.user_id == "u1" and not who.is_admin


@pytest.mark.parametrize("mutate", [
    lambda: make_token(secret="x" * 40),                                           # wrong key
    lambda: make_token({"exp": T0 + 10}),                                          # expired (checked at T0+100, leeway 30)
    lambda: make_token({"iss": "someone-else"}),
    lambda: make_token({"aud": "other-app"}),
    lambda: make_token({"sub": ""}),
    lambda: make_token({"iat": T0 + 1000}),                                        # issued in the future
    lambda: make_token(header={"alg": "none", "typ": "JWT"}),
    lambda: make_token(header={"alg": "HS512", "typ": "JWT"}),
    lambda: GO_TOKEN[:-3] + "AAA",                                                 # tampered signature
    lambda: GO_TOKEN.rsplit(".", 1)[0],                                            # no signature
    lambda: "not-a-jwt",
    lambda: "",
])
def test_rejects_bad_tokens(mutate):
    with pytest.raises(auth.AuthError):
        verify(mutate())


def test_payload_swap_is_detected():
    h, _p, s = GO_TOKEN.split(".")
    forged = _b64(json.dumps({"sub": "attacker", "role": "admin", "iss": "trustrag-user", "aud": "trustrag", "iat": T0, "exp": T0 + 900}).encode())
    with pytest.raises(auth.AuthError):
        verify(f"{h}.{forged}.{s}")


def test_clock_leeway_is_30_seconds():
    tok = make_token()
    assert verify(tok, now=T0 + 900 + 29)
    with pytest.raises(auth.AuthError):
        verify(tok, now=T0 + 900 + 31)


def test_bearer_token_parsing():
    assert auth.bearer_token("Bearer abc.def.ghi") == "abc.def.ghi"
    assert auth.bearer_token("bearer   x.y.z ") == "x.y.z"
    assert auth.bearer_token("Basic Zm9vOmJhcg==") == ""
    assert auth.bearer_token("") == ""


def test_client_ip_counts_trusted_proxies_from_the_right():
    # browser -> ingress -> nginx -> gateway: two trusted hops append; the client's own forged prefix is ignored
    assert auth.client_ip("1.1.1.1, 203.0.113.9, 10.0.0.4", "10.0.0.5", 2) == "203.0.113.9"
    assert auth.client_ip("203.0.113.9, 10.0.0.4", "10.0.0.5", 2) == "203.0.113.9"
    assert auth.client_ip("203.0.113.9", "10.0.0.5", 2) == "203.0.113.9"      # fewer entries than expected: best available
    assert auth.client_ip("", "10.0.0.5", 2) == "10.0.0.5"
    assert auth.client_ip("6.6.6.6", "10.0.0.5", 0) == "10.0.0.5"            # hops=0: do not trust the header at all


def test_limiter_window_and_independent_keys():
    now = [100.0]
    lim = auth.FixedWindowLimiter(clock=lambda: now[0])
    assert [lim.allow("u1", 2, 60)[0] for _ in range(3)] == [True, True, False]
    ok, retry = lim.allow("u1", 2, 60)
    assert not ok and 1 <= retry <= 61
    assert lim.allow("u2", 2, 60)[0]                 # another user is unaffected
    now[0] += 61
    assert lim.allow("u1", 2, 60)[0]                 # window passed
    assert lim.allow("u1", 0, 60) == (True, 0)       # 0 = unlimited


def test_limiter_table_does_not_grow_forever():
    now = [0.0]
    lim = auth.FixedWindowLimiter(clock=lambda: now[0])
    for i in range(20001):
        lim.allow(f"k{i}", 1, 10)
    now[0] = 100
    lim.allow("trigger", 1, 10)
    assert len(lim._hits) < 5


def test_user_paths_are_an_allow_list():
    for p in ("/auth/login", "/users/me", "/conversations", "/conversations/abc/messages", "/admin/users"):
        assert userproxy.is_user_path(p), p
    for p in ("/health", "/health/ready", "/chat", "/authx", "/conversationsx", "/users", "/auth"):
        assert not userproxy.is_user_path(p), p


def test_forward_headers_pass_credentials_and_add_identity():
    h = userproxy.forward_headers({"authorization": "Bearer t", "cookie": "trustrag_rt=1", "content-type": "application/json",
                                   "host": "evil", "x-client-ip": "6.6.6.6", "x-api-key": "browser-sent"}, "203.0.113.9", "svc")
    assert h["authorization"] == "Bearer t" and h["cookie"] == "trustrag_rt=1"
    assert h["x-client-ip"] == "203.0.113.9" and h["x-api-key"] == "svc"       # overwritten, never taken from the browser
    assert "host" not in h
    assert "x-api-key" not in userproxy.forward_headers({"x-api-key": "browser-sent"}, "1.1.1.1", "")


def test_response_headers_keep_every_set_cookie():
    items = [("Content-Type", "application/json"), ("Set-Cookie", "a=1; Path=/"), ("set-cookie", "b=2"), ("Server", "x"),
             ("Retry-After", "5"), ("Content-Length", "10")]
    got = userproxy.response_headers(items)
    assert [v for k, v in got if k == "set-cookie"] == ["a=1; Path=/", "b=2"]
    assert ("retry-after", "5") in got and all(k not in ("server", "content-length") for k, _ in got)
