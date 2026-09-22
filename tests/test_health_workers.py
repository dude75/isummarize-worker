from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import llm as llm_mod
from app.config import get_settings
from tests.conftest import auth_headers, isolate_env
from tests.test_api_tasks import _install_slow_stub, _poll_client, _post_summarize


def _assert_workers(body: dict, *, max_workers: int, active: int, available: int) -> None:
    workers = body["workers"]
    assert set(workers) == {"max", "active", "available"}
    assert workers["max"] == max_workers
    assert workers["active"] == active
    assert workers["available"] == available
    assert workers["active"] <= workers["max"]
    assert workers["available"] == workers["max"] - workers["active"]


def test_health_workers_idle(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "stub-model"
    _assert_workers(body, max_workers=1, active=0, available=1)


def test_ready_includes_workers(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    _assert_workers(response.json(), max_workers=1, active=0, available=1)


def test_health_workers_max_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolate_env(tmp_path, monkeypatch, workers="2", queue_size="4")
    from app.main import app

    with TestClient(app) as test_client:
        _assert_workers(test_client.get("/health").json(), max_workers=2, active=0, available=2)
    get_settings.cache_clear()


def test_health_workers_max_alias(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolate_env(tmp_path, monkeypatch, queue_size="4")
    monkeypatch.delenv("WORKERS", raising=False)
    monkeypatch.setenv("WORKERS_MAX", "3")
    get_settings.cache_clear()
    from app.main import app

    with TestClient(app) as test_client:
        _assert_workers(test_client.get("/health").json(), max_workers=3, active=0, available=3)
    get_settings.cache_clear()


def test_health_workers_during_and_after_task(client: TestClient) -> None:
    gate = threading.Event()
    original = _install_slow_stub(gate)
    try:
        idle = client.get("/health").json()
        _assert_workers(idle, max_workers=1, active=0, available=1)

        created = _post_summarize(client)
        assert created.status_code == 202
        task_id = created.json()["meta"]["task_id"]
        deadline = time.time() + 3
        while time.time() < deadline:
            status = client.get(f"/tasks/{task_id}", headers=auth_headers()).json()["status"]
            if status == "running":
                break
            time.sleep(0.02)
        else:
            raise AssertionError("task did not become running")

        running = client.get("/health").json()
        _assert_workers(running, max_workers=1, active=1, available=0)

        gate.set()
        _poll_client(client, task_id)

        finished = client.get("/health").json()
        _assert_workers(finished, max_workers=1, active=0, available=1)
    finally:
        gate.set()
        llm_mod.complete_override = original
