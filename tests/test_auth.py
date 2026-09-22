from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.auth import api_token_is_valid
from app.config import get_settings
from app import llm as llm_mod
from app.schemas import LlmHealth
from app.version import read_version
from tests.conftest import auth_headers

_UNAUTHORIZED = {"status": "error", "error": {"code": "unauthorized"}}
_PROTECTED = (
    ("GET", "/tasks"),
    ("DELETE", "/tasks"),
    ("GET", "/tasks/00000000-0000-0000-0000-000000000001"),
    ("DELETE", "/tasks/00000000-0000-0000-0000-000000000001"),
    ("GET", "/metrics"),
    ("POST", "/summarize"),
)


def _assert_unauthorized(response) -> None:
    assert response.status_code == 401
    body = response.json()
    assert body == _UNAUTHORIZED
    assert "meta" not in body
    assert "summary" not in body


def test_health_without_token(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == read_version()
    assert body["model"] == "stub-model"
    assert body["llm"] in {"ready", "unconfigured", "unavailable"}
    workers = body["workers"]
    assert workers == {"max": 1, "active": 0, "available": 1}


def test_ready_without_token(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == read_version()
    assert body["model"] == "stub-model"
    assert body["llm"] == "ready"
    assert body["workers"] == {"max": 1, "active": 0, "available": 1}


def test_ready_unavailable_is_503(client: TestClient) -> None:
    llm_mod.status_override = LlmHealth.unavailable
    ready = client.get("/ready")
    assert ready.status_code == 503
    assert ready.json()["status"] == "ok"
    assert ready.json()["llm"] == "unavailable"
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["llm"] == "unavailable"


def test_summarize_rejects_when_llm_unavailable(client: TestClient) -> None:
    llm_mod.status_override = LlmHealth.unavailable
    response = client.post(
        "/summarize",
        json={"text": "hello transcript", "skill": "sum briefly"},
        headers=auth_headers(),
    )
    assert response.status_code == 503
    assert response.json() == {"status": "error", "error": {"code": "llm_unavailable"}}


def test_summarize_rejects_when_llm_unconfigured(client: TestClient) -> None:
    llm_mod.status_override = LlmHealth.unconfigured
    response = client.post(
        "/summarize",
        json={"text": "hello transcript", "skill": "sum briefly"},
        headers=auth_headers(),
    )
    assert response.status_code == 503
    assert response.json() == {"status": "error", "error": {"code": "llm_unconfigured"}}


@pytest.mark.parametrize("method, path", _PROTECTED)
def test_protected_routes_reject_missing_token(client: TestClient, method: str, path: str) -> None:
    kwargs: dict = {}
    if method == "POST":
        kwargs["json"] = {"text": "hello", "skill": "sum"}
    response = client.request(method, path, **kwargs)
    _assert_unauthorized(response)


@pytest.mark.parametrize("method, path", _PROTECTED)
def test_protected_routes_reject_wrong_token(client: TestClient, method: str, path: str) -> None:
    kwargs: dict = {"headers": {"Authorization": "Bearer not-the-token"}}
    if method == "POST":
        kwargs["json"] = {"text": "hello", "skill": "sum"}
    response = client.request(method, path, **kwargs)
    _assert_unauthorized(response)


def test_summarize_accepts_valid_bearer(client: TestClient) -> None:
    response = client.post(
        "/summarize",
        json={"text": "hello transcript", "skill": "sum briefly"},
        headers=auth_headers(),
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert body["error"] is None
    assert body["summary"] is None
    assert body["meta"]["task_id"]


def test_summarize_form_without_token_is_401(client: TestClient) -> None:
    response = client.post("/summarize", data={"text": "hello", "skill": "sum"})
    _assert_unauthorized(response)


def test_empty_api_token_rejects_all(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("API_TOKEN", "")
    get_settings.cache_clear()
    response = client.get("/tasks", headers={"Authorization": "Bearer test"})
    _assert_unauthorized(response)
    get_settings.cache_clear()


def test_metrics_requires_token(client: TestClient) -> None:
    _assert_unauthorized(client.get("/metrics"))
    ok = client.get("/metrics", headers=auth_headers())
    assert ok.status_code == 200


def test_api_token_is_valid_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_TOKEN", "test")
    get_settings.cache_clear()
    assert api_token_is_valid("Bearer test") is True
    assert api_token_is_valid("bearer test") is True
    assert api_token_is_valid("BEARER test") is True
    assert api_token_is_valid(None) is False
    assert api_token_is_valid("") is False
    assert api_token_is_valid("Bearer") is False
    assert api_token_is_valid("Bearer ") is False
    assert api_token_is_valid("Bearer nope") is False
    assert api_token_is_valid("Basic test") is False
    assert api_token_is_valid("test") is False
    get_settings.cache_clear()


def test_openapi_matches_stock_fastapi_shape(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    dumped = str(spec)
    assert "gpt-4o-mini" not in dumped
    assert "4f8b9e12-87c2-4911-bca4-d832e12cf900" not in dumped
    assert "Модель из MODEL" not in dumped
    assert "сам текст не возвращается" not in dumped
    summarize = spec["paths"]["/summarize"]["post"]
    content_202 = summarize["responses"]["202"]["content"]["application/json"]
    assert "example" not in content_202
    assert content_202["schema"]["$ref"] == "#/components/schemas/TaskResponse"
    props = spec["components"]["schemas"]["TaskResponse"]["properties"]
    assert "status" in props
    assert "meta" in props
    meta_props = spec["components"]["schemas"]["TaskMeta"]["properties"]
    assert set(meta_props) >= {"timestamp", "task_id", "model", "text_chars", "skill_chars"}
    assert "description" not in meta_props["model"]
    assert "description" not in meta_props["text_chars"]
