"""Pure helpers for forwarding /auth, /users, /conversations and /admin to trustrag-user-service."""
from collections.abc import Iterable, Mapping

# the only paths the browser may reach on the user service (its /health is not one of them)
_PREFIXES = ("/auth/", "/users/", "/admin/", "/conversations/")
_EXACT = {"/conversations"}
# sign-in endpoints that do not need a token (the user service rate-limits them itself)
PUBLIC_PATHS = {"/auth/register", "/auth/login", "/auth/refresh", "/auth/logout"}

_REQUEST_HEADERS = ("authorization", "cookie", "content-type", "user-agent", "origin", "x-request-id", "accept-language")
_RESPONSE_HEADERS = {"content-type", "set-cookie", "retry-after", "www-authenticate", "cache-control", "x-request-id"}


def is_user_path(path: str) -> bool:
    return path in _EXACT or path.startswith(_PREFIXES)


def forward_headers(incoming: Mapping[str, str], client_ip: str, api_key: str) -> dict[str, str]:
    """What the user service receives: the caller's credentials plus who the caller really is (X-Client-IP, trusted
    by the user service only together with the service key)."""
    out = {k: incoming[k] for k in _REQUEST_HEADERS if incoming.get(k)}
    out["x-client-ip"] = client_ip
    if api_key:
        out["x-api-key"] = api_key
    return out


def response_headers(items: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """Copy the headers worth passing on, keeping repeated ones (several Set-Cookie)."""
    return [(k.lower(), v) for k, v in items if k.lower() in _RESPONSE_HEADERS]
