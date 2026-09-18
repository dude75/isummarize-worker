from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from prometheus_client import generate_latest
from prometheus_client.parser import text_string_to_metric_families

from app import llm as llm_mod
from app.config import get_settings
from app.llm import ChatResult
from app.prometheus_metrics import Metrics, set_active
from app.queueing import TaskRunner
from app.tasks import TaskStore
from tests.conftest import auth_headers, isolate_env
from tests.test_api_tasks import _install_slow_stub, _poll_client, _post_summarize


def _sample(text: str, name: str, labels: dict[str, str] | None = None) -> float | None:
    wanted = labels or {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name != name:
                continue
            if all(sample.labels.get(key) == value for key, value in wanted.items()):
                return float(sample.value)
    return None


def test_metrics_without_token(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 401


def test_metrics_rejects_bad_token(client: TestClient) -> None:
    response = client.get("/metrics", headers={"Authorization": "Bearer not-the-token"})
    assert response.status_code == 401


def test_metrics_with_token(client: TestClient) -> None:
    response = client.get("/metrics", headers=auth_headers())
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    body = response.text
    assert _sample(body, "isummarize_up") == 1.0
    assert _sample(body, "isummarize_ready") == 1.0
    assert _sample(body, "isummarize_queue_depth") == 0.0
    assert _sample(body, "isummarize_worker_slots") == 1.0
    assert _sample(body, "isummarize_queue_limit") == 1.0


def test_submitted_and_completed_success(client: TestClient) -> None:
    created = _post_summarize(client)
    assert created.status_code == 202
    task_id = created.json()["meta"]["task_id"]
    payload = _poll_client(client, task_id)
    assert payload["status"] == "success"

    body = client.get("/metrics", headers=auth_headers()).text
    labels = {"model": "stub-model"}
    assert _sample(body, "isummarize_tasks_submitted_total", labels) == 1.0
    assert (
        _sample(
            body,
            "isummarize_tasks_completed_total",
            {**labels, "status": "success"},
        )
        == 1.0
    )
    assert _sample(body, "isummarize_pipeline_duration_seconds_count", labels) == 1.0


def test_queue_full_increments_rejected(client: TestClient) -> None:
    gate = threading.Event()
    original = _install_slow_stub(gate)
    try:
        first = _post_summarize(client)
        assert first.status_code == 202
        first_id = first.json()["meta"]["task_id"]
        deadline = time.time() + 3
        while time.time() < deadline:
            status = client.get(f"/tasks/{first_id}", headers=auth_headers()).json()["status"]
            if status == "running":
                break
            time.sleep(0.02)
        else:
            raise AssertionError("first task did not become running")

        second = _post_summarize(client)
        assert second.status_code == 202
        third = _post_summarize(client)
        assert third.status_code == 503
        body = client.get("/metrics", headers=auth_headers()).text
        assert _sample(body, "isummarize_queue_rejected_total") == 1.0
        assert "isummarize_queue_rejected_total" in body
    finally:
        gate.set()
        llm_mod.complete_override = original


def test_http_path_uses_template_not_uuid(client: TestClient) -> None:
    created = _post_summarize(client)
    task_id = created.json()["meta"]["task_id"]
    _poll_client(client, task_id)
    body = client.get("/metrics", headers=auth_headers()).text
    assert "/tasks/{task_id}" in body
    for line in body.splitlines():
        if 'path="' in line:
            assert task_id not in line


def test_csv_still_written(client: TestClient) -> None:
    created = _post_summarize(client)
    _poll_client(client, created.json()["meta"]["task_id"])
    csv_path = Path(get_settings().PERFORMANCE_LOG)
    assert csv_path.is_file()
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 2


def test_metrics_disabled_does_not_break_summarize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolate_env(tmp_path, monkeypatch, extra={"METRICS_ENABLED": "false", "WORKER_QUEUE_SIZE": "1"})
    from app.main import app as local_app

    with TestClient(local_app) as test_client:
        created = _post_summarize(test_client)
        assert created.status_code == 202
        _poll_client(test_client, created.json()["meta"]["task_id"])
        response = test_client.get("/metrics", headers=auth_headers())
        assert response.status_code == 200
        assert _sample(response.text, "isummarize_queue_depth") is None
        assert _sample(response.text, "isummarize_tasks_submitted_total") is None
    get_settings.cache_clear()


def test_restore_missing_payload_metric(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolate_env(tmp_path, monkeypatch)
    settings = get_settings()
    metrics = Metrics(enabled=True)
    set_active(metrics)
    store = TaskStore(settings.SQLITE_PATH)
    store.create(
        "missing-payload-metric",
        model=settings.MODEL,
        text_chars=1,
        skill_chars=1,
        payload_dir=str(tmp_path / "no-such"),
    )
    store.close()

    async def scenario() -> None:
        runner = TaskRunner(settings)
        await runner.start()
        await runner.stop()

    try:
        asyncio.run(scenario())
        body = generate_latest(metrics.registry).decode()
        assert _sample(body, "isummarize_restore_tasks_total", {"outcome": "missing_payload"}) == 1.0
    finally:
        set_active(None)
        get_settings.cache_clear()


def test_restore_process_killed_metric(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolate_env(tmp_path, monkeypatch, extra={"TASK_MAX_RESTARTS": "0"})
    settings = get_settings()
    metrics = Metrics(enabled=True)
    set_active(metrics)
    store = TaskStore(settings.SQLITE_PATH)
    dest_dir = tmp_path / "tmp" / "poison-metric"
    dest_dir.mkdir(parents=True)
    (dest_dir / "text.txt").write_text("t", encoding="utf-8")
    (dest_dir / "skill.md").write_text("s", encoding="utf-8")
    store.create(
        "poison-metric",
        model=settings.MODEL,
        text_chars=1,
        skill_chars=1,
        payload_dir=str(dest_dir),
    )
    assert store.mark_running("poison-metric")
    store.close()

    async def scenario() -> None:
        runner = TaskRunner(settings)
        await runner.start()
        rec = runner.store.get("poison-metric")
        assert rec is not None
        assert rec.status.value == "error"
        await runner.stop()

    try:
        asyncio.run(scenario())
        body = generate_latest(metrics.registry).decode()
        assert _sample(body, "isummarize_restore_tasks_total", {"outcome": "process_killed"}) == 1.0
    finally:
        set_active(None)
        get_settings.cache_clear()


def test_llm_tokens_recorded(client: TestClient) -> None:
    original = llm_mod.complete_override

    async def counted(settings, system, user):
        return ChatResult(content="sum", prompt_tokens=11, completion_tokens=7, latency_sec=0.01)

    llm_mod.complete_override = counted
    try:
        created = _post_summarize(client)
        payload = _poll_client(client, created.json()["meta"]["task_id"])
        assert payload["status"] == "success"
        assert payload["meta"]["prompt_tokens"] == 11
        assert payload["meta"]["completion_tokens"] == 7
        body = client.get("/metrics", headers=auth_headers()).text
        assert _sample(body, "isummarize_prompt_tokens_total", {"model": "stub-model"}) == 11.0
        assert _sample(body, "isummarize_completion_tokens_total", {"model": "stub-model"}) == 7.0
    finally:
        llm_mod.complete_override = original
