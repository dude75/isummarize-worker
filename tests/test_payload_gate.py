from __future__ import annotations

from collections.abc import Iterator

import pytest
from starlette.requests import ClientDisconnect, Request

from app.main import _read_body_capped, app, prometheus_http_middleware


def _request_from_chunks(
    chunks: list[bytes],
    *,
    disconnect: bool = False,
) -> Request:
    messages: list[dict[str, object]] = []
    for i, chunk in enumerate(chunks):
        last = i == len(chunks) - 1
        messages.append(
            {
                "type": "http.request",
                "body": chunk,
                "more_body": (not last) or disconnect,
            }
        )
    if disconnect:
        messages.append({"type": "http.disconnect"})
    it: Iterator[dict[str, object]] = iter(messages)

    async def receive() -> dict[str, object]:
        try:
            return next(it)
        except StopIteration:
            return {"type": "http.disconnect"}

    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/summarize",
            "raw_path": b"/summarize",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1),
            "server": ("test", 80),
            "app": app,
        },
        receive,
    )


async def test_read_body_capped_joins_chunks() -> None:
    request = _request_from_chunks([b"ab", b"cd"])
    assert await _read_body_capped(request, 10) == b"abcd"


async def test_read_body_capped_allows_exact_limit() -> None:
    request = _request_from_chunks([b"x" * 80])
    assert await _read_body_capped(request, 80) == b"x" * 80


async def test_read_body_capped_stops_before_eof_when_over_limit() -> None:
    """Без Content-Length лимит режет поток, а не ждёт полного тела."""
    request = _request_from_chunks([b"a" * 40, b"b" * 40, b"c" * 40])
    assert await _read_body_capped(request, 80) is None


async def test_read_body_capped_propagates_disconnect() -> None:
    request = _request_from_chunks([b"hello"], disconnect=True)
    with pytest.raises(ClientDisconnect):
        await _read_body_capped(request, 10_000)


async def test_prometheus_middleware_swallows_client_disconnect() -> None:
    async def receive() -> dict[str, object]:
        return {"type": "http.disconnect"}

    request = Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/summarize",
            "raw_path": b"/summarize",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1),
            "server": ("test", 80),
            "app": app,
        },
        receive,
    )

    async def call_next(_request: Request):
        raise ClientDisconnect()

    response = await prometheus_http_middleware(request, call_next)
    assert response.status_code == 499
