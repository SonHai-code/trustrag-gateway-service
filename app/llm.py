"""Minimal async LLM client over plain HTTP (httpx). No vendor SDK needed.

Providers: anthropic | openai (any OpenAI-compatible server, e.g. vLLM in Phase 6) | mock.
"""
import json
from typing import AsyncIterator

import httpx

from app import models, settings
from app.models import ModelSpec


class LLMError(RuntimeError):
    """The LLM call failed (network, bad status, ...)."""


class LLMConfigError(LLMError):
    """Missing or invalid configuration, e.g. no API key."""


def _check_config(spec: ModelSpec) -> None:
    if spec.provider == "mock":
        return
    if spec.provider not in ("anthropic", "openai"):
        raise LLMConfigError(f"Unknown provider '{spec.provider}' for model '{spec.id}' (use anthropic, openai or mock)")
    if not spec.api_key and not (spec.provider == "openai" and spec.base_url):
        raise LLMConfigError(f"No API key for model '{spec.id}'. Set its secret (api_key_env / LLM_API_KEY), or choose another model.")


def _request(spec: ModelSpec, system: str, user: str, stream: bool) -> tuple[str, dict, dict]:
    """One request in the dialect of the model's provider. Differences between model families live in app/models.py."""
    max_tokens = spec.max_tokens or settings.LLM_MAX_TOKENS
    if spec.provider == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": spec.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        body = {"model": spec.model, "max_tokens": max_tokens, "system": system,
                "messages": [{"role": "user", "content": user}], "stream": stream}
    else:
        url = f"{spec.endpoint}/chat/completions"
        headers = {"content-type": "application/json"}
        if spec.api_key:
            headers["authorization"] = f"Bearer {spec.api_key}"
        body = {"model": spec.model, "stream": stream, models.token_param(spec, spec.model): models.output_budget(spec, spec.model, max_tokens),
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if models.send_temperature(spec, spec.model):
            body["temperature"] = spec.temperature
    if spec.extra_body:
        body.update(spec.extra_body)  # explicit per-model settings win over the defaults above
    return url, headers, body


def _mock_answer() -> str:
    return "Đây là câu trả lời mô phỏng (LLM_PROVIDER=mock), dựa trên nguồn [1]."


async def generate(system: str, user: str, spec: ModelSpec | None = None) -> str:
    spec = spec or models.resolve(None)
    _check_config(spec)
    if spec.provider == "mock":
        return _mock_answer()
    for attempt in (1, 2, 3):  # a 400 that names a parameter this model does not accept is fixed once and retried
        url, headers, body = _request(spec, system, user, stream=False)
        try:
            async with httpx.AsyncClient(timeout=spec.timeout or settings.LLM_TIMEOUT) as client:
                resp = await client.post(url, headers=headers, json=body)
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        if resp.status_code == 400 and attempt < 3 and spec.provider == "openai" and models.learn_from_error(spec, spec.model, resp.text):
            continue
        if resp.status_code >= 400:
            raise LLMError(f"LLM returned {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        if spec.provider == "anthropic":
            return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        return data["choices"][0]["message"]["content"] or ""
    raise LLMError("LLM request could not be adapted to the model")  # pragma: no cover


async def stream(system: str, user: str, spec: ModelSpec | None = None) -> AsyncIterator[str]:
    spec = spec or models.resolve(None)
    _check_config(spec)
    if spec.provider == "mock":
        for word in _mock_answer().split(" "):
            yield word + " "
        return
    try:
        async with httpx.AsyncClient(timeout=spec.timeout or settings.LLM_TIMEOUT) as client:
            for attempt in (1, 2, 3):
                url, headers, body = _request(spec, system, user, stream=True)
                async with client.stream("POST", url, headers=headers, json=body) as resp:
                    if resp.status_code >= 400:
                        detail = (await resp.aread()).decode("utf-8", "ignore")[:300]
                        if resp.status_code == 400 and attempt < 3 and spec.provider == "openai" and models.learn_from_error(spec, spec.model, detail):
                            continue  # nothing was streamed yet: rebuild the request for this model and try again
                        raise LLMError(f"LLM returned {resp.status_code}: {detail}")
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if not raw or raw == "[DONE]":
                            continue
                        event = json.loads(raw)
                        if spec.provider == "anthropic":
                            delta = event.get("delta", {})
                            if event.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
                                yield delta.get("text", "")
                        else:
                            choices = event.get("choices") or [{}]
                            text = choices[0].get("delta", {}).get("content")
                            if text:
                                yield text
                    return
    except httpx.HTTPError as exc:
        raise LLMError(f"LLM stream failed: {exc}") from exc
