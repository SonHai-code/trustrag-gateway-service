"""Login wall: verify the access tokens issued by trustrag-user-service.

Tokens are HS256 JWTs signed with the secret both services share (JWT_SECRET), so the gateway checks them locally, with no
network call per request. Only the standard library is used. The algorithm is never read from the token: the header must be
exactly the one the user service writes, so "alg": "none" and algorithm-confusion tricks are refused.
"""
import base64
import hashlib
import hmac
import json
import threading
import time
from dataclasses import dataclass

_HEADER = {"alg": "HS256", "typ": "JWT"}


class AuthError(Exception):
    """The token is missing, malformed, forged, expired or meant for someone else."""


@dataclass(frozen=True)
class Principal:
    user_id: str
    email: str
    name: str
    role: str
    session_id: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def _b64d(part: str) -> bytes:
    try:
        return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))
    except Exception as exc:  # binascii.Error, ValueError
        raise AuthError("bad encoding") from exc


def verify_token(token: str, secret: str, issuer: str, audience: str, now: float | None = None, leeway: int = 30) -> Principal:
    if not token or not secret:
        raise AuthError("missing token")
    parts = token.split(".")
    if len(parts) != 3:
        raise AuthError("malformed token")
    signing_input = f"{parts[0]}.{parts[1]}".encode("ascii", "ignore")
    expected = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    if not hmac.compare_digest(expected, _b64d(parts[2])):
        raise AuthError("bad signature")
    try:
        header = json.loads(_b64d(parts[0]))
        claims = json.loads(_b64d(parts[1]))
    except ValueError as exc:
        raise AuthError("bad json") from exc
    if header != _HEADER or not isinstance(claims, dict):
        raise AuthError("unexpected header")
    now = time.time() if now is None else now
    exp, iat = claims.get("exp"), claims.get("iat")
    if not isinstance(exp, (int, float)) or not isinstance(iat, (int, float)):
        raise AuthError("missing times")
    if now > exp + leeway:
        raise AuthError("token expired")
    if iat > now + leeway:
        raise AuthError("token issued in the future")
    if claims.get("iss") != issuer or claims.get("aud") != audience:
        raise AuthError("wrong issuer or audience")
    sub = claims.get("sub")
    if not isinstance(sub, str) or not sub:
        raise AuthError("no subject")
    return Principal(user_id=sub, email=str(claims.get("email", "")), name=str(claims.get("name", "")),
                     role=str(claims.get("role", "user")), session_id=str(claims.get("sid", "")))


def bearer_token(authorization: str) -> str:
    """'Bearer abc' -> 'abc'; anything else -> ''."""
    if len(authorization) > 8 and authorization[:7].lower() == "bearer ":
        return authorization[7:].strip()
    return ""


def client_ip(forwarded_for: str, peer: str, hops: int) -> str:
    """The browser's address behind `hops` trusted proxies that each append to X-Forwarded-For.
    Entries further left than the trusted ones can be forged by the client and are ignored."""
    entries = [e.strip() for e in forwarded_for.split(",") if e.strip()]
    if hops > 0 and entries:
        return entries[-hops] if len(entries) >= hops else entries[0]
    return peer or "unknown"


class FixedWindowLimiter:
    """Per-key request counter in memory (one gateway replica; with several, each enforces its own share)."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._hits: dict[str, tuple[float, int]] = {}

    def allow(self, key: str, limit: int, window: float) -> tuple[bool, int]:
        """(allowed, seconds until the window resets). limit <= 0 means unlimited."""
        if limit <= 0:
            return True, 0
        now = self._clock()
        with self._lock:
            if len(self._hits) > 20000:  # drop expired windows so the table cannot grow without bound
                self._hits = {k: v for k, v in self._hits.items() if v[0] > now}
            end, count = self._hits.get(key, (0.0, 0))
            if end <= now:
                end, count = now + window, 0
            if count >= limit:
                return False, max(1, int(end - now) + 1)
            self._hits[key] = (end, count + 1)
            return True, 0
