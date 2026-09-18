from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config import Settings
from app.llm import (
    LlmCallError,
    _redact,
    chat_complete,
    probe_llm,
    reset_probe_cache,
)
from app import llm as llm_mod
from app.schemas import ErrorCode, LlmHealth


@pytest.fixture(autouse=True)
def _clear_llm_hooks() -> None:
    llm_mod.complete_override = None
    llm_mod.status_override = None
    reset_probe_cache()
    yield
    llm_mod.complete_override = None
    llm_mod.status_override = None
    reset_probe_cache()


def _settings(**kwargs: Any) -> Settings:
    values = {
        "BASE_URL": "http://llm.test/v1",
        "API_KEY": "sk-SECRET-KEY",
        "MODEL": "test-model",
        "LLM_TIMEOUT_SEC": 5,
        "LLM_MAX_RETRIES": 2,
        "LLM_PROBE_TTL_SEC": 15,
        "TEMPERATURE": 0.2,
    }
    values.update(kwargs)
    return Settings(**values)


def test_redact_api_key() -> None:
    assert "sk-SECRET-KEY" not in _redact("boom sk-SECRET-KEY here", "sk-SECRET-KEY")
    assert _redact("boom sk-SECRET-KEY here", "sk-SECRET-KEY") == "boom *** here"


@pytest.mark.asyncio
async def test_unconfigured_raises() -> None:
    settings = _settings(BASE_URL="", API_KEY="", MODEL="")
    with pytest.raises(LlmCallError) as exc:
        await chat_complete(settings, "sys", "user")
    assert exc.value.code is ErrorCode.llm_unconfigured


@pytest.mark.asyncio
async def test_timeout_maps(monkeypatch: pytest.MonkeyPatch) -> None:
    from openai import APITimeoutError

    settings = _settings()

    async def boom(*_args, **_kwargs):
        raise APITimeoutError(request=MagicMock())

    monkeypatch.setattr("app.llm._one_call", boom)
    with pytest.raises(LlmCallError) as exc:
        await chat_complete(settings, "sys", "user")
    assert exc.value.code is ErrorCode.llm_timeout
    assert "sk-SECRET-KEY" not in str(exc.value)


@pytest.mark.asyncio
async def test_retries_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    from openai import InternalServerError

    settings = _settings(LLM_MAX_RETRIES=2)
    from app.llm import ChatResult

    calls = {"n": 0}

    async def flaky(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise InternalServerError(
                message="nope",
                response=MagicMock(status_code=500),
                body=None,
            )
        return ChatResult(content="ok", prompt_tokens=1, completion_tokens=1, latency_sec=0.01)

    monkeypatch.setattr("app.llm._one_call", flaky)
    monkeypatch.setattr("app.llm.asyncio.sleep", AsyncMock())
    result = await chat_complete(settings, "sys", "user")
    assert result.content == "ok"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_retries_exhausted_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    from openai import APIConnectionError

    settings = _settings(LLM_MAX_RETRIES=1)

    async def boom(*_args, **_kwargs):
        raise APIConnectionError(request=MagicMock(), message="sk-SECRET-KEY down")

    monkeypatch.setattr("app.llm._one_call", boom)
    monkeypatch.setattr("app.llm.asyncio.sleep", AsyncMock())
    with pytest.raises(LlmCallError) as exc:
        await chat_complete(settings, "sys", "user")
    assert exc.value.code is ErrorCode.llm_unavailable
    assert "sk-SECRET-KEY" not in (exc.value.message or "")


@pytest.mark.asyncio
async def test_probe_unconfigured() -> None:
    status = await probe_llm(_settings(BASE_URL="", API_KEY="", MODEL=""))
    assert status is LlmHealth.unconfigured


def _fake_httpx_client(status_code: int = 200, on_get=None):
    class FakeResponse:
        def __init__(self) -> None:
            self.status_code = status_code

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, headers):
            assert "sk-SECRET-KEY" in headers["Authorization"]
            if on_get is not None:
                await on_get(url, headers)
            return FakeResponse()

    return FakeClient


@pytest.mark.asyncio
async def test_probe_http_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.llm.stubs_enabled", lambda: False)
    monkeypatch.setattr("app.llm.status_override", None)
    monkeypatch.setattr("app.llm.httpx.AsyncClient", _fake_httpx_client(200))
    status = await probe_llm(_settings())
    assert status is LlmHealth.ready


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [401, 403, 404, 429, 500, 503])
async def test_probe_non_2xx_unavailable(monkeypatch: pytest.MonkeyPatch, code: int) -> None:
    monkeypatch.setattr("app.llm.stubs_enabled", lambda: False)
    monkeypatch.setattr("app.llm.status_override", None)
    monkeypatch.setattr("app.llm.httpx.AsyncClient", _fake_httpx_client(code))
    status = await probe_llm(_settings())
    assert status is LlmHealth.unavailable


@pytest.mark.asyncio
async def test_probe_network_error_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    class BoomClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, headers):
            raise ConnectionError("gateway down")

    monkeypatch.setattr("app.llm.stubs_enabled", lambda: False)
    monkeypatch.setattr("app.llm.status_override", None)
    monkeypatch.setattr("app.llm.httpx.AsyncClient", BoomClient)
    status = await probe_llm(_settings())
    assert status is LlmHealth.unavailable


@pytest.mark.asyncio
async def test_probe_cache_skips_second_http(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    async def on_get(_url, _headers):
        calls["n"] += 1

    monkeypatch.setattr("app.llm.stubs_enabled", lambda: False)
    monkeypatch.setattr("app.llm.status_override", None)
    monkeypatch.setattr("app.llm.httpx.AsyncClient", _fake_httpx_client(401, on_get))
    settings = _settings()
    assert await probe_llm(settings) is LlmHealth.unavailable
    assert await probe_llm(settings) is LlmHealth.unavailable
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_probe_ttl_zero_always_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    async def on_get(_url, _headers):
        calls["n"] += 1

    monkeypatch.setattr("app.llm.stubs_enabled", lambda: False)
    monkeypatch.setattr("app.llm.status_override", None)
    monkeypatch.setattr("app.llm.httpx.AsyncClient", _fake_httpx_client(200, on_get))
    settings = _settings(LLM_PROBE_TTL_SEC=0)
    assert await probe_llm(settings) is LlmHealth.ready
    assert await probe_llm(settings) is LlmHealth.ready
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_probe_cache_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    now = {"t": 100.0}

    async def on_get(_url, _headers):
        calls["n"] += 1

    monkeypatch.setattr("app.llm.stubs_enabled", lambda: False)
    monkeypatch.setattr("app.llm.status_override", None)
    monkeypatch.setattr("app.llm.httpx.AsyncClient", _fake_httpx_client(403, on_get))
    monkeypatch.setattr("app.llm.time.monotonic", lambda: now["t"])
    settings = _settings(LLM_PROBE_TTL_SEC=15)
    assert await probe_llm(settings) is LlmHealth.unavailable
    now["t"] = 114.9
    assert await probe_llm(settings) is LlmHealth.unavailable
    assert calls["n"] == 1
    now["t"] = 115.0
    assert await probe_llm(settings) is LlmHealth.unavailable
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_probe_singleflight(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    started = asyncio.Event()
    release = asyncio.Event()

    async def on_get(_url, _headers):
        calls["n"] += 1
        started.set()
        await release.wait()

    monkeypatch.setattr("app.llm.stubs_enabled", lambda: False)
    monkeypatch.setattr("app.llm.status_override", None)
    monkeypatch.setattr("app.llm.httpx.AsyncClient", _fake_httpx_client(200, on_get))
    settings = _settings()
    first = asyncio.create_task(probe_llm(settings))
    await started.wait()
    second = asyncio.create_task(probe_llm(settings))
    await asyncio.sleep(0.01)
    release.set()
    results = await asyncio.gather(first, second)
    assert results == [LlmHealth.ready, LlmHealth.ready]
    assert calls["n"] == 1


def test_chat_result_json_roundtrip() -> None:
    payload = {"content": "hi", "prompt_tokens": 1}
    assert json.dumps(payload, ensure_ascii=False)
