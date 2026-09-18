"""OpenAI-совместимый Chat Completions клиент."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)

from app.config import Settings
from app.prometheus_metrics import observe_llm_call
from app.schemas import ErrorCode, LlmHealth

logger = logging.getLogger("app.llm")

CompleteFn = Callable[[Settings, str, str], Awaitable["ChatResult"]]

complete_override: CompleteFn | None = None
status_override: LlmHealth | None = None

_probe_cache: _ProbeSnapshot | None = None
_probe_lock: asyncio.Lock | None = None


@dataclass
class _ProbeSnapshot:
    key: str
    status: LlmHealth
    at: float


class LlmCallError(Exception):
    def __init__(self, code: ErrorCode, message: str | None = None) -> None:
        super().__init__(message or code.value)
        self.code = code
        self.message = message


@dataclass
class ChatResult:
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_sec: float = 0.0
    http_status: int | None = None


def stubs_enabled() -> bool:
    return os.environ.get("ISUMMARIZE_STUBS", "").lower() in {"1", "true", "yes"}


def _redact(text: str, secret: str) -> str:
    if not secret:
        return text
    return text.replace(secret, "***")


def _safe_message(exc: BaseException, api_key: str) -> str:
    return _redact(str(exc), api_key)


def _timeout(settings: Settings) -> httpx.Timeout | float | None:
    if settings.LLM_TIMEOUT_SEC <= 0:
        return None
    return float(settings.LLM_TIMEOUT_SEC)


async def stub_chat_complete(_settings: Settings, _system: str, _user: str) -> ChatResult:
    return ChatResult(content="stub-summary", prompt_tokens=10, completion_tokens=5, latency_sec=0.0)


def _client(settings: Settings) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=settings.BASE_URL.strip() or None,
        api_key=settings.API_KEY,
        timeout=_timeout(settings),
        max_retries=0,
    )


async def _one_call(settings: Settings, system: str, user: str) -> ChatResult:
    kwargs: dict[str, Any] = {
        "model": settings.MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": settings.TEMPERATURE,
    }
    if settings.MAX_TOKENS is not None:
        kwargs["max_tokens"] = settings.MAX_TOKENS
    client = _client(settings)
    started = time.perf_counter()
    try:
        response = await client.chat.completions.create(**kwargs)
    except APITimeoutError as exc:
        raise LlmCallError(ErrorCode.llm_timeout, _safe_message(exc, settings.API_KEY)) from exc
    except (AuthenticationError, PermissionDeniedError) as exc:
        raise LlmCallError(ErrorCode.llm_unavailable, _safe_message(exc, settings.API_KEY)) from exc
    except RateLimitError:
        raise
    except InternalServerError:
        raise
    except APIConnectionError as exc:
        raise LlmCallError(ErrorCode.llm_unavailable, _safe_message(exc, settings.API_KEY)) from exc
    except APIStatusError as exc:
        status = getattr(exc, "status_code", None)
        if status in {401, 403}:
            raise LlmCallError(ErrorCode.llm_unavailable, _safe_message(exc, settings.API_KEY)) from exc
        if status == 429 or (isinstance(status, int) and status >= 500):
            raise
        raise LlmCallError(ErrorCode.llm_unavailable, _safe_message(exc, settings.API_KEY)) from exc
    finally:
        await client.close()
    latency = time.perf_counter() - started
    choice = response.choices[0] if response.choices else None
    content = ""
    if choice is not None and choice.message is not None and choice.message.content:
        content = choice.message.content
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage is not None else None
    completion_tokens = usage.completion_tokens if usage is not None else None
    return ChatResult(
        content=content or "",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_sec=latency,
        http_status=200,
    )


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, (RateLimitError, InternalServerError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        return status == 429 or (isinstance(status, int) and status >= 500)
    return False


async def chat_complete(
    settings: Settings,
    system: str,
    user: str,
    *,
    task_id: str | None = None,
) -> ChatResult:
    if complete_override is not None:
        result = await complete_override(settings, system, user)
        observe_llm_call(settings.MODEL, result.latency_sec)
        return result
    if not settings.llm_configured():
        raise LlmCallError(ErrorCode.llm_unconfigured, "BASE_URL, API_KEY or MODEL is empty")

    attempts = settings.LLM_MAX_RETRIES + 1
    last_retryable: BaseException | None = None
    for attempt in range(attempts):
        try:
            result = await _one_call(settings, system, user)
        except LlmCallError:
            raise
        except Exception as exc:
            if _retryable(exc) and attempt + 1 < attempts:
                last_retryable = exc
                delay = min(8.0, 0.5 * (2**attempt))
                logger.info(
                    "llm retry task_id=%s model=%s attempt=%s delay=%.1fs",
                    task_id,
                    settings.MODEL,
                    attempt + 1,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            if isinstance(exc, APITimeoutError):
                raise LlmCallError(ErrorCode.llm_timeout, _safe_message(exc, settings.API_KEY)) from exc
            raise LlmCallError(ErrorCode.llm_unavailable, _safe_message(exc, settings.API_KEY)) from exc
        observe_llm_call(settings.MODEL, result.latency_sec)
        logger.info(
            "llm call task_id=%s model=%s prompt_chars=%s completion_chars=%s "
            "latency_ms=%.0f http_status=%s prompt_tokens=%s completion_tokens=%s",
            task_id,
            settings.MODEL,
            len(system) + len(user),
            len(result.content),
            result.latency_sec * 1000,
            result.http_status,
            result.prompt_tokens,
            result.completion_tokens,
        )
        return result
    assert last_retryable is not None
    raise LlmCallError(
        ErrorCode.llm_unavailable, _safe_message(last_retryable, settings.API_KEY)
    ) from last_retryable


def reset_probe_cache() -> None:
    global _probe_cache, _probe_lock
    _probe_cache = None
    _probe_lock = None


def _probe_cache_key(settings: Settings) -> str:
    return f"{settings.BASE_URL.strip()}\n{settings.MODEL.strip()}"


def _cached_probe(key: str, ttl: float) -> LlmHealth | None:
    if ttl <= 0:
        return None
    cached = _probe_cache
    if cached is None or cached.key != key:
        return None
    if time.monotonic() - cached.at >= ttl:
        return None
    return cached.status


def _store_probe(key: str, status: LlmHealth) -> None:
    global _probe_cache
    _probe_cache = _ProbeSnapshot(key=key, status=status, at=time.monotonic())


def _get_probe_lock() -> asyncio.Lock:
    global _probe_lock
    if _probe_lock is None:
        _probe_lock = asyncio.Lock()
    return _probe_lock


async def _probe_http(settings: Settings) -> LlmHealth:
    url = settings.BASE_URL.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {settings.API_KEY}"}
    timeout = httpx.Timeout(5.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers)
    except Exception:
        logger.info("llm probe unavailable model=%s", settings.MODEL)
        return LlmHealth.unavailable
    if 200 <= response.status_code < 300:
        return LlmHealth.ready
    logger.info(
        "llm probe unavailable model=%s status=%s",
        settings.MODEL,
        response.status_code,
    )
    return LlmHealth.unavailable


async def probe_llm(settings: Settings) -> LlmHealth:
    if status_override is not None:
        return status_override
    if stubs_enabled():
        return LlmHealth.ready if settings.llm_configured() else LlmHealth.unconfigured
    if not settings.llm_configured():
        return LlmHealth.unconfigured
    key = _probe_cache_key(settings)
    ttl = settings.LLM_PROBE_TTL_SEC
    cached = _cached_probe(key, ttl)
    if cached is not None:
        return cached
    async with _get_probe_lock():
        cached = _cached_probe(key, ttl)
        if cached is not None:
            return cached
        status = await _probe_http(settings)
        _store_probe(key, status)
        return status
