"""LLM abstraction with production reliability patterns.

Patterns implemented (each is a known latency/reliability lever):
  * Provider-agnostic client (mock | ollama | anthropic | openai-compatible/vLLM).
  * Model cascade: try the small model first; escalate to the large model only when the
    small model's self-reported confidence or schema validation fails. Most traffic stays cheap.
  * Response cache keyed on (model, normalized prompt) with TTL: repeated drawings / FAQ chats
    return in microseconds.
  * Retries with exponential backoff + jitter, a per-call timeout, and a circuit breaker so a
    failing provider degrades the system to "rules only + human review" instead of hanging it.
  * Structured output: JSON is parsed and validated against a field schema; one repair
    round-trip is attempted before giving up (no silent garbage enters the pipeline).

The MockLLM is deterministic. Modules register task handlers on it so the whole system runs
offline in CI and in the demo, and tests are reproducible.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from .config import settings
from .observability import tracer


@dataclass
class LLMRequest:
    task: str                      # routing / mock-dispatch key, e.g. "extract_fields"
    prompt: str
    system: str = "You are a precise industrial-document assistant. Answer only from the provided context."
    payload: dict[str, Any] = field(default_factory=dict)  # structured inputs (used by the mock + prompts)
    json_schema: dict[str, str] | None = None  # {"field": "type"}; triggers JSON validation
    max_tokens: int = 800
    temperature: float = 0.0


@dataclass
class LLMResponse:
    text: str
    model: str
    latency_ms: float
    cached: bool = False
    parsed: dict[str, Any] | None = None
    confidence: float | None = None
    escalated: bool = False


class LLMError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------------------------
class Provider:
    name = "base"

    def complete(self, req: LLMRequest, model: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


MockHandler = Callable[[LLMRequest, str], str]


class MockLLM(Provider):
    """Deterministic stand-in. `model` is passed so tests can simulate a weak small model."""

    name = "mock"
    _handlers: dict[str, MockHandler] = {}

    @classmethod
    def register(cls, task: str) -> Callable[[MockHandler], MockHandler]:
        def deco(fn: MockHandler) -> MockHandler:
            cls._handlers[task] = fn
            return fn

        return deco

    def complete(self, req: LLMRequest, model: str) -> str:
        handler = self._handlers.get(req.task)
        if handler is None:
            return json.dumps({"answer": "I don't have enough grounded information to answer.", "confidence": 0.3})
        return handler(req, model)


class OllamaProvider(Provider):
    name = "ollama"

    def complete(self, req: LLMRequest, model: str) -> str:
        body = {
            "model": model,
            "system": req.system,
            "prompt": req.prompt,
            "stream": False,
            "options": {"temperature": req.temperature, "num_predict": req.max_tokens},
        }
        if req.json_schema:
            body["format"] = "json"  # grammar-constrained decoding on the server side
        r = httpx.post(f"{settings.ollama_url}/api/generate", json=body, timeout=settings.request_timeout_s)
        r.raise_for_status()
        return r.json()["response"]


class AnthropicProvider(Provider):
    name = "anthropic"

    def complete(self, req: LLMRequest, model: str) -> str:
        import os

        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY not set")
        r = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={
                "model": model,
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
                "system": req.system,
                "messages": [{"role": "user", "content": req.prompt}],
            },
            timeout=settings.request_timeout_s,
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"])


class OpenAICompatProvider(Provider):
    """Works with OpenAI and any OpenAI-compatible server (vLLM, TGI, LM Studio)."""

    name = "openai"

    def complete(self, req: LLMRequest, model: str) -> str:
        import os

        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'not-needed')}"}
        body: dict[str, Any] = {
            "model": model,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "messages": [{"role": "system", "content": req.system}, {"role": "user", "content": req.prompt}],
        }
        if req.json_schema:
            body["response_format"] = {"type": "json_object"}
        r = httpx.post(f"{settings.openai_base_url}/chat/completions", headers=headers, json=body,
                       timeout=settings.request_timeout_s)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class RateLimited(LLMError):
    """Provider said 429. Not a provider *failure*: it must not open the circuit breaker."""

    def __init__(self, retry_after: float, detail: str = "") -> None:
        super().__init__(f"rate limited, retry after {retry_after:.1f}s {detail}".strip())
        self.retry_after = retry_after


class Pacer:
    """Client-side request pacing (requests/minute) so a burst of work never trips the free-tier limit."""

    def __init__(self, rpm: float) -> None:
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self.next_at = 0.0
        self.lock = threading.Lock()

    def wait(self) -> float:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + self.interval
        if delay:
            time.sleep(delay)
        return delay


class GroqProvider(Provider):
    """Groq's OpenAI-compatible endpoint (free tier). Llama 3.1 8B / Llama 3.3 70B by default.

    * JSON mode (`response_format: json_object`) when a schema is requested — Groq requires the word
      "JSON" in the prompt, which every structured prompt here already contains.
    * 429 → RateLimited(retry_after) using the `retry-after` header; the router waits and retries
      without counting it as an outage.
    * Remaining-token/request headers are recorded so the UI can show quota headroom.
    """

    name = "groq"
    pacer: Pacer | None = None
    last_limits: dict[str, str] = {}

    def __init__(self) -> None:
        if GroqProvider.pacer is None:
            GroqProvider.pacer = Pacer(settings.groq_rpm)

    def complete(self, req: LLMRequest, model: str) -> str:
        import os

        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise LLMError("GROQ_API_KEY not set — create a free key at console.groq.com and put it in backend/.env")
        body: dict[str, Any] = {
            "model": model,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "messages": [{"role": "system", "content": req.system}, {"role": "user", "content": req.prompt}],
        }
        if req.json_schema:
            body["response_format"] = {"type": "json_object"}
        waited = self.pacer.wait() if self.pacer else 0.0
        if waited:
            tracer.incr("llm.groq.paced")
        r = httpx.post(f"{settings.groq_base_url}/chat/completions", json=body, timeout=settings.request_timeout_s,
                       headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        GroqProvider.last_limits = {k: v for k, v in r.headers.items() if k.lower().startswith("x-ratelimit")}
        if r.status_code == 429:
            tracer.incr("llm.groq.rate_limited")
            try:
                ra = float(r.headers.get("retry-after", "2"))
            except ValueError:
                ra = 2.0
            raise RateLimited(ra, r.headers.get("x-ratelimit-reset-tokens", ""))
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        return msg.get("content") or ""


PROVIDERS: dict[str, type[Provider]] = {
    "mock": MockLLM, "groq": GroqProvider, "ollama": OllamaProvider, "anthropic": AnthropicProvider,
    "openai": OpenAICompatProvider,
}


# ---------------------------------------------------------------------------------------------
# Reliability primitives
# ---------------------------------------------------------------------------------------------
class TTLCache:
    def __init__(self, maxsize: int = 4096, ttl_s: float = 3600) -> None:
        self.maxsize, self.ttl = maxsize, ttl_s
        self._d: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    @staticmethod
    def key(model: str, req: LLMRequest) -> str:
        norm = re.sub(r"\s+", " ", (req.system + "\n" + req.prompt).strip().lower())
        return hashlib.sha256(f"{model}|{req.task}|{norm}".encode()).hexdigest()

    def get(self, k: str) -> str | None:
        with self._lock:
            item = self._d.get(k)
            if item and time.time() - item[0] < self.ttl:
                self._d.move_to_end(k)
                self.hits += 1
                return item[1]
            self.misses += 1
            return None

    def put(self, k: str, v: str) -> None:
        with self._lock:
            self._d[k] = (time.time(), v)
            self._d.move_to_end(k)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)


class CircuitBreaker:
    """closed → (N consecutive failures) → open → (after reset_s) → half-open → closed/open."""

    def __init__(self, fail_threshold: int, reset_s: float) -> None:
        self.fail_threshold, self.reset_s = fail_threshold, reset_s
        self.failures = 0
        self.opened_at: float | None = None

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        if time.time() - self.opened_at >= self.reset_s:
            return "half-open"
        return "open"

    def allow(self) -> bool:
        return self.state != "open"

    def success(self) -> None:
        self.failures, self.opened_at = 0, None

    def failure(self) -> None:
        self.failures += 1
        if self.failures >= self.fail_threshold:
            self.opened_at = time.time()


def parse_json_loose(text: str) -> dict[str, Any] | None:
    """Accept raw JSON or JSON inside prose/code fences (models often wrap output)."""
    text = text.strip()
    try:
        out = json.loads(text)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            out = json.loads(m.group(0))
            return out if isinstance(out, dict) else None
        except json.JSONDecodeError:
            return None
    return None


_TYPES = {"str": str, "float": (int, float), "int": int, "bool": bool, "list": list, "dict": dict}


def validate_schema(obj: dict[str, Any], schema: dict[str, str]) -> list[str]:
    errors = []
    for key, typ in schema.items():
        optional = typ.endswith("?")
        typ = typ.rstrip("?")
        if key not in obj or obj[key] is None:
            if not optional:
                errors.append(f"missing '{key}'")
            continue
        if not isinstance(obj[key], _TYPES[typ]):
            errors.append(f"'{key}' should be {typ}")
    return errors


# ---------------------------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------------------------
class LLMRouter:
    def __init__(self, provider: str | None = None, small: str | None = None, large: str | None = None) -> None:
        self.provider_name = provider or settings.llm_provider
        self.provider = PROVIDERS[self.provider_name]()
        self.small = small or settings.small_model
        self.large = large or settings.large_model
        self.cache = TTLCache(ttl_s=settings.cache_ttl_s)
        self.breaker = CircuitBreaker(settings.circuit_fail_threshold, settings.circuit_reset_s)

    # single model call with cache + retries + breaker
    def _call(self, req: LLMRequest, model: str) -> LLMResponse:
        k = self.cache.key(model, req)
        hit = self.cache.get(k)
        if hit is not None:
            tracer.incr("llm.cache_hit")
            return LLMResponse(text=hit, model=model, latency_ms=0.0, cached=True)
        if not self.breaker.allow():
            tracer.incr("llm.circuit_open")
            raise LLMError("circuit open: provider unhealthy, degrading to rules + human review")
        last: Exception | None = None
        for attempt in range(settings.max_retries + 1):
            t0 = time.perf_counter()
            try:
                with tracer.span("llm.call", model=model, task=req.task, attempt=attempt):
                    text = self.provider.complete(req, model)
                self.breaker.success()
                self.cache.put(k, text)
                return LLMResponse(text=text, model=model, latency_ms=(time.perf_counter() - t0) * 1000)
            except RateLimited as exc:  # quota, not an outage: wait as told, don't trip the breaker
                last = exc
                if exc.retry_after > settings.max_rate_limit_wait_s:
                    break  # caller degrades to rules + human review instead of hanging
                time.sleep(exc.retry_after)
                continue
            except Exception as exc:  # network, 5xx, timeout
                last = exc
                self.breaker.failure()
                tracer.incr("llm.retry")
                if attempt < settings.max_retries:
                    time.sleep(min(2.0, 0.2 * 2**attempt) * (0.5 + random.random()))
        raise LLMError(f"LLM call failed after retries: {last}")

    def _structured(self, req: LLMRequest, model: str) -> LLMResponse:
        resp = self._call(req, model)
        if not req.json_schema:
            return resp
        parsed = parse_json_loose(resp.text)
        errors = validate_schema(parsed, req.json_schema) if parsed is not None else ["not JSON"]
        if errors:  # one repair round-trip
            tracer.incr("llm.repair")
            repair = LLMRequest(
                task=req.task, payload=req.payload, json_schema=req.json_schema, system=req.system,
                prompt=f"{req.prompt}\n\nYour previous output was invalid ({'; '.join(errors)}). "
                       f"Return ONLY a JSON object with keys {list(req.json_schema)}.",
            )
            resp = self._call(repair, model)
            parsed = parse_json_loose(resp.text)
            errors = validate_schema(parsed, req.json_schema) if parsed is not None else ["not JSON"]
            if errors:
                raise LLMError(f"schema validation failed: {errors}")
        resp.parsed = parsed
        conf = parsed.get("confidence") if parsed else None
        resp.confidence = float(conf) if isinstance(conf, (int, float)) else None
        return resp

    def complete(self, req: LLMRequest, cascade: bool = True) -> LLMResponse:
        """Small model first; escalate when invalid or under-confident."""
        with tracer.span("llm.complete", task=req.task):
            if not cascade:
                return self._structured(req, self.large)
            try:
                resp = self._structured(req, self.small)
                if resp.confidence is None or resp.confidence >= settings.escalate_conf:
                    return resp
                tracer.incr("llm.escalate.low_conf")
            except LLMError as exc:
                if "circuit open" in str(exc):
                    raise
                tracer.incr("llm.escalate.invalid")
            resp = self._structured(req, self.large)
            resp.escalated = True
            return resp


_router: LLMRouter | None = None
_judge: LLMRouter | None = None


def get_router() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router


def get_judge_router() -> LLMRouter:
    """Judge uses a separate provider/model family (mitigates self-preference bias)."""
    global _judge
    if _judge is None:
        _judge = LLMRouter(provider=settings.judge_provider, small=settings.judge_model, large=settings.judge_model)
    return _judge
