"""Minimal async LLM client over plain HTTP (httpx). No vendor SDK needed.

Providers: anthropic | openai (any OpenAI-compatible server, e.g. vLLM in Phase 6) | mock.
"""
import json
from typing import AsyncIterator

import httpx

from app import settings


class LLMError(RuntimeError):
    """The LLM call failed (network, bad status, ...)."""


class LLMConfigError(LLMError):
    """Missing or invalid configuration, e.g. no API key."""


def _check_config() -> None:
    if settings.LLM_PROVIDER == "mock":
        return
    if settings.LLM_PROVIDER not in ("anthropic", "openai"):
        raise LLMConfigError(f"Unknown LLM_PROVIDER '{settings.LLM_PROVIDER}' (use anthropic, openai or mock)")
    if not settings.LLM_API_KEY and not (settings.LLM_PROVIDER == "openai" and settings.LLM_BASE_URL):
        raise LLMConfigError("LLM_API_KEY is empty. Set it in .env, or use LLM_PROVIDER=mock to test offline.")


def _request(system: str, user: str, stream: bool) -> tuple[str, dict, dict]:
    if settings.LLM_PROVIDER == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": settings.LLM_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": settings.LLM_MODEL,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "stream": stream,
        }
    else:
        base = (settings.LLM_BASE_URL or "https://api.openai.com/v1").rstrip("/")
        url = f"{base}/chat/completions"
        headers = {"content-type": "application/json"}
        if settings.LLM_API_KEY:
            headers["authorization"] = f"Bearer {settings.LLM_API_KEY}"
        body = {
            "model": settings.LLM_MODEL,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": stream,
        }
        if settings.LLM_EXTRA_BODY:  # vLLM / OpenAI-compatible extras; a bad value must not take the service down
            try:
                extra = json.loads(settings.LLM_EXTRA_BODY)
                body.update(extra if isinstance(extra, dict) else {})
            except json.JSONDecodeError:
                pass
    return url, headers, body


def _mock_answer() -> str:
    return "Đây là câu trả lời mô phỏng (LLM_PROVIDER=mock), dựa trên nguồn [1]."


async def generate(system: str, user: str) -> str:
    _check_config()
    if settings.LLM_PROVIDER == "mock":
        return _mock_answer()
    url, headers, body = _request(system, user, stream=False)
    try:
        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT) as client:
            resp = await client.post(url, headers=headers, json=body)
    except httpx.HTTPError as exc:
        raise LLMError(f"LLM request failed: {exc}") from exc
    if resp.status_code >= 400:
        raise LLMError(f"LLM returned {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    if settings.LLM_PROVIDER == "anthropic":
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    return data["choices"][0]["message"]["content"] or ""


async def stream(system: str, user: str) -> AsyncIterator[str]:
    _check_config()
    if settings.LLM_PROVIDER == "mock":
        for word in _mock_answer().split(" "):
            yield word + " "
        return
    url, headers, body = _request(system, user, stream=True)
    try:
        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT) as client:
            async with client.stream("POST", url, headers=headers, json=body) as resp:
                if resp.status_code >= 400:
                    detail = (await resp.aread()).decode("utf-8", "ignore")[:300]
                    raise LLMError(f"LLM returned {resp.status_code}: {detail}")
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    event = json.loads(raw)
                    if settings.LLM_PROVIDER == "anthropic":
                        delta = event.get("delta", {})
                        if event.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
                            yield delta.get("text", "")
                    else:
                        choices = event.get("choices") or [{}]
                        text = choices[0].get("delta", {}).get("content")
                        if text:
                            yield text
    except httpx.HTTPError as exc:
        raise LLMError(f"LLM stream failed: {exc}") from exc
