"""Model registry + per-model capabilities: one place that knows WHICH models can be chosen and HOW each one is called.

LLM_MODELS (JSON list) declares the models users may pick. Each entry:
  id            short name shown in the UI and sent by the client (required, unique)
  provider      "openai" (OpenAI and every OpenAI-compatible server: vLLM, Groq, OpenRouter, Gemini's compat endpoint...) | "anthropic" | "mock"
  model         the vendor's model name (required)
  label         text for the dropdown (default: id)
  base_url      endpoint for provider=openai (default https://api.openai.com/v1)
  api_key_env   NAME of the environment variable that holds the key (the key itself never goes in this JSON)
  fast_model    cheaper model of the SAME endpoint for the short internal steps (router, grading); default: the model itself
  token_param   auto | max_tokens | max_completion_tokens (auto = by model family, then learned from the server's error)
  temperature   number, or null to omit it (auto-omitted for model families that reject it)
  min_completion_tokens  floor for the output budget (reasoning models spend tokens thinking before they answer)
  extra_body    JSON merged into every request (e.g. {"reasoning_effort": "low"})
Without LLM_MODELS the old LLM_PROVIDER / LLM_MODEL / LLM_API_KEY / LLM_BASE_URL / LLM_FAST_MODEL settings define one model, so
nothing changes for existing deployments. Clients only ever send an id; they can never supply a URL or a key."""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field

from app import settings

log = logging.getLogger(__name__)
PROVIDERS = ("openai", "anthropic", "mock")


class ModelConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    id: str
    provider: str
    model: str
    label: str = ""
    base_url: str = ""
    api_key: str = field(default="", repr=False)
    fast_model: str = ""
    token_param: str = "auto"
    temperature: float | None = 0.0
    min_completion_tokens: int = 0
    extra_body: dict = field(default_factory=dict)
    max_tokens: int = 0
    timeout: float = 0.0

    @property
    def endpoint(self) -> str:
        return (self.base_url or "https://api.openai.com/v1").rstrip("/")

    @property
    def is_openai_cloud(self) -> bool:
        return self.provider == "openai" and "api.openai.com" in self.endpoint

    def public(self, default: bool = False) -> dict:
        return {"id": self.id, "label": self.label or self.id, "provider": self.provider, "model": self.model, "default": default}


def _legacy_spec() -> ModelSpec:
    extra = {}
    if settings.LLM_EXTRA_BODY:
        try:
            parsed = json.loads(settings.LLM_EXTRA_BODY)
            extra = parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            log.warning("LLM_EXTRA_BODY is not valid JSON: ignored")
    return ModelSpec(id=settings.LLM_MODEL or "default", provider=settings.LLM_PROVIDER, model=settings.LLM_MODEL,
                     base_url=settings.LLM_BASE_URL, api_key=settings.LLM_API_KEY,
                     fast_model=getattr(settings, "LLM_FAST_MODEL", ""), extra_body=extra)


def _parse_entry(raw: dict, index: int) -> ModelSpec:
    if not isinstance(raw, dict):
        raise ModelConfigError(f"LLM_MODELS[{index}] must be an object")
    mid, provider, model = str(raw.get("id", "")).strip(), str(raw.get("provider", "openai")).lower(), str(raw.get("model", "")).strip()
    if not mid or not model:
        raise ModelConfigError(f"LLM_MODELS[{index}] needs 'id' and 'model'")
    if provider not in PROVIDERS:
        raise ModelConfigError(f"LLM_MODELS[{index}] ('{mid}'): unknown provider '{provider}' (use {', '.join(PROVIDERS)})")
    token_param = str(raw.get("token_param", "auto"))
    if token_param not in ("auto", "max_tokens", "max_completion_tokens"):
        raise ModelConfigError(f"LLM_MODELS[{index}] ('{mid}'): token_param must be auto, max_tokens or max_completion_tokens")
    key_env = str(raw.get("api_key_env", "")).strip()
    api_key = os.getenv(key_env, "") if key_env else ""
    if key_env and not api_key:
        log.warning("model '%s': environment variable %s is empty: calls will fail until it is set as a secret", mid, key_env)
    temp = raw.get("temperature", 0.0)
    return ModelSpec(id=mid, provider=provider, model=model, label=str(raw.get("label", "")), base_url=str(raw.get("base_url", "")),
                     api_key=api_key, fast_model=str(raw.get("fast_model", "")), token_param=token_param,
                     temperature=None if temp is None else float(temp), min_completion_tokens=int(raw.get("min_completion_tokens", 0)),
                     extra_body=raw.get("extra_body") or {}, max_tokens=int(raw.get("max_tokens", 0)), timeout=float(raw.get("timeout", 0)))


_registry: dict[str, ModelSpec] | None = None
_default_id = ""


def registry() -> dict[str, ModelSpec]:
    global _registry, _default_id
    if _registry is None:
        raw = getattr(settings, "LLM_MODELS", "").strip()
        specs: list[ModelSpec] = []
        if raw:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ModelConfigError(f"LLM_MODELS is not valid JSON: {exc}") from exc
            if not isinstance(data, list) or not data:
                raise ModelConfigError("LLM_MODELS must be a non-empty JSON list")
            specs = [_parse_entry(item, i) for i, item in enumerate(data)]
            if len({s.id for s in specs}) != len(specs):
                raise ModelConfigError("LLM_MODELS: ids must be unique")
        else:
            specs = [_legacy_spec()]
        _registry = {s.id: s for s in specs}
        wanted = getattr(settings, "LLM_DEFAULT_MODEL", "")
        _default_id = wanted if wanted in _registry else specs[0].id
        if wanted and wanted not in _registry:
            log.warning("LLM_DEFAULT_MODEL '%s' is not in LLM_MODELS: using '%s'", wanted, _default_id)
    return _registry


def default_id() -> str:
    registry()
    return _default_id


def resolve(model_id: str | None) -> ModelSpec:
    """The spec for a client-supplied id (None/empty = the default). Unknown ids are refused: the registry is an allow-list."""
    reg = registry()
    if not model_id:
        return reg[_default_id]
    if model_id not in reg:
        raise ModelConfigError(f"Unknown model '{model_id}'. Available: {', '.join(reg)}")
    return reg[model_id]


def public_list() -> list[dict]:
    d = default_id()
    return [s.public(s.id == d) for s in registry().values()]


def reset() -> None:
    global _registry
    _registry = None
    _learned.clear()


# ---------------- capabilities ----------------
# OpenAI's newer model families (o-series, gpt-5 and later) take `max_completion_tokens`, not `max_tokens`, and accept only the
# default temperature. Names we have never seen are handled by LEARNING: the first 400 that says "Unsupported parameter: 'x'"
# changes the request, the retry is remembered for this endpoint + model, and later calls are right the first time.
_NEW_OPENAI = re.compile(r"^(?:o\d|gpt-(?:[5-9]|\d{2,}))", re.IGNORECASE)
_learned: dict[tuple, dict] = {}


def _key(spec: ModelSpec, model: str) -> tuple:
    return (spec.provider, spec.endpoint, model)


def token_param(spec: ModelSpec, model: str) -> str:
    if spec.token_param != "auto":
        return spec.token_param
    learned = _learned.get(_key(spec, model), {}).get("token_param")
    if learned:
        return learned
    return "max_completion_tokens" if spec.is_openai_cloud and _NEW_OPENAI.match(model) else "max_tokens"


def send_temperature(spec: ModelSpec, model: str) -> bool:
    if spec.temperature is None:
        return False
    learned = _learned.get(_key(spec, model), {}).get("temperature")
    if learned is not None:
        return learned
    return not (spec.is_openai_cloud and _NEW_OPENAI.match(model))


def learn_from_error(spec: ModelSpec, model: str, error_text: str) -> bool:
    """Read a 400 body; True if the request should be rebuilt and retried (the learned capability changed)."""
    t = error_text.lower()
    m = re.search(r"unsupported parameter: '(\w+)'", t) or re.search(r"'(\w+)' is not supported with this model", t)
    param = m.group(1) if m else ""
    if not param and re.search(r"unsupported value: 'temperature'|'temperature' does not support", t):
        param = "temperature"
    entry = _learned.setdefault(_key(spec, model), {})
    if param == "max_tokens" and token_param(spec, model) != "max_completion_tokens":
        entry["token_param"] = "max_completion_tokens"
    elif param == "max_completion_tokens" and token_param(spec, model) != "max_tokens":
        entry["token_param"] = "max_tokens"
    elif param == "temperature" and entry.get("temperature") is not False:
        entry["temperature"] = False
    else:
        return False
    log.warning("model '%s' @ %s: learned %s -> retrying once", model, spec.endpoint, entry)
    return True


def output_budget(spec: ModelSpec, model: str, wanted: int) -> int:
    """Reasoning models count their hidden thinking against the limit: a 30-token router call would come back empty."""
    floor = spec.min_completion_tokens or (1024 if token_param(spec, model) == "max_completion_tokens" else 0)
    return max(wanted, floor)
