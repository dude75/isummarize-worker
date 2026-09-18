"""Общие фикстуры. Не трогать живой {DATA_DIR}/tmp сервиса."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings


@pytest.fixture(scope="session", autouse=True)
def _test_api_token() -> Iterator[None]:
    """Герметичный Bearer, без зависимости от локального .env."""
    mp = pytest.MonkeyPatch()
    mp.setenv("API_TOKEN", "test")
    get_settings.cache_clear()
    yield
    mp.undo()
    get_settings.cache_clear()


def isolate_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    workers: str = "1",
    queue_size: str = "4",
    max_restarts: str | None = None,
    extra: dict[str, str] | None = None,
) -> None:
    monkeypatch.setenv("API_TOKEN", "test")
    monkeypatch.setenv("ISUMMARIZE_STUBS", "1")
    monkeypatch.setenv("BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("API_KEY", "sk-test-secret-do-not-log")
    monkeypatch.setenv("MODEL", "stub-model")
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "tasks.db"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("PERFORMANCE_LOG", str(tmp_path / "logs" / "performance_log.csv"))
    monkeypatch.setenv("WORKERS", workers)
    monkeypatch.setenv("WORKER_QUEUE_SIZE", queue_size)
    if max_restarts is not None:
        monkeypatch.setenv("TASK_MAX_RESTARTS", max_restarts)
    if extra:
        for key, value in extra.items():
            monkeypatch.setenv(key, value)
    get_settings.cache_clear()


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    isolate_env(tmp_path, monkeypatch, workers="1", queue_size="1")
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()


def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {get_settings().API_TOKEN}"}
