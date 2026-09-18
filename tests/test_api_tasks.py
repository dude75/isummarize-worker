from __future__ import annotations

import asyncio
import threading
import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import llm as llm_mod
from app.config import get_settings
from app.llm import ChatResult, stub_chat_complete
from app.main import app
from app.queueing import TaskRunner
from app.schemas import TaskStatus
from app.storage import create_tmp, tmp_dir
from app.tasks import TaskStore
from tests.conftest import auth_headers, isolate_env


def _post_summarize(client: TestClient, text: str = "hello transcript", skill: str = "sum briefly"):
    return client.post(
        "/summarize",
        json={"text": text, "skill": skill},
        headers=auth_headers(),
    )


def _poll_client(client: TestClient, task_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    result = None
    while time.time() < deadline:
        result = client.get(f"/tasks/{task_id}", headers=auth_headers())
        assert result.status_code == 200
        if result.json()["status"] in {"success", "error"}:
            return result.json()
        time.sleep(0.02)
    raise AssertionError(f"timeout polling {task_id}: {result.json() if result else None}")


async def _wait_store(store: TaskStore, task_id: str, statuses: set[TaskStatus], timeout: float = 5.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = store.get(task_id)
        if last is not None and last.status in statuses:
            return last
        await asyncio.sleep(0.02)
    raise AssertionError(f"timeout waiting {task_id} in {statuses}: {last}")


def _install_slow_stub(gate: threading.Event):
    original = llm_mod.complete_override

    async def slow(settings, system, user):
        await asyncio.to_thread(gate.wait, 10)
        return ChatResult(content="stub-summary", prompt_tokens=1, completion_tokens=1, latency_sec=0.01)

    llm_mod.complete_override = slow
    return original


def test_summarize_empty_fields_422(client: TestClient) -> None:
    missing = client.post("/summarize", json={"text": "hi"}, headers=auth_headers())
    assert missing.status_code == 422
    blank = client.post(
        "/summarize",
        json={"text": "   ", "skill": "rules"},
        headers=auth_headers(),
    )
    assert blank.status_code == 422
    extra = client.post(
        "/summarize",
        json={"text": "hi", "skill": "rules", "model": "gpt-4"},
        headers=auth_headers(),
    )
    assert extra.status_code == 422


def test_summarize_form_fields(client: TestClient) -> None:
    created = client.post(
        "/summarize",
        data={"text": "hello transcript", "skill": "sum briefly"},
        headers=auth_headers(),
    )
    assert created.status_code == 202
    payload = _poll_client(client, created.json()["meta"]["task_id"])
    assert payload["status"] == "success"


def test_summarize_docs_form_fields(client: TestClient) -> None:
    root = client.get("/", follow_redirects=False)
    assert root.status_code in (307, 302)
    assert root.headers["location"] == "/docs"
    docs = client.get("/docs")
    assert docs.status_code == 200
    html = docs.text
    wrap_at = html.find("window.SwaggerUIBundle = wrapped")
    init_at = html.find("const ui = SwaggerUIBundle({")
    fetch_at = html.find("window.fetch = function")
    assert wrap_at != -1
    assert init_at != -1
    assert fetch_at != -1
    assert wrap_at < init_at
    assert "summarize-paste" in html
    assert "requestInterceptor" in html
    assert "JSON.stringify(fields)" in html
    assert "defaultModelRendering" not in html
    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    content = openapi.json()["paths"]["/summarize"]["post"]["requestBody"]["content"]
    assert "application/x-www-form-urlencoded" in content
    props = content["application/x-www-form-urlencoded"]["schema"]["properties"]
    assert set(props) == {"text", "skill"}
    assert "example" not in props["text"]
    assert "example" not in props["skill"]


def test_summarize_json_keeps_special_characters(client: TestClient) -> None:
    text = 'a&b=c + % / "quotes"\nline'
    skill = "rule: x=1&y=2 #md"
    created = client.post(
        "/summarize",
        json={"text": text, "skill": skill},
        headers=auth_headers(),
    )
    assert created.status_code == 202
    assert created.json()["meta"]["text_chars"] == len(text)
    assert created.json()["meta"]["skill_chars"] == len(skill)


def test_payload_too_large(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_PAYLOAD_BYTES", "80")
    get_settings.cache_clear()
    response = _post_summarize(client, text="x" * 200, skill="y" * 200)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
    listing = client.get("/tasks", headers=auth_headers())
    assert listing.status_code == 200
    assert listing.json() == []


def test_poll_until_success_and_tmp_cleaned(client: TestClient) -> None:
    created = _post_summarize(client)
    assert created.status_code == 202
    body = created.json()
    assert body["status"] == "queued"
    assert body["summary"] is None
    assert body["error"] is None
    task_id = body["meta"]["task_id"]
    assert body["meta"]["model"] == "stub-model"
    assert body["meta"]["text_chars"] == len("hello transcript")

    payload = _poll_client(client, task_id)
    assert payload["status"] == "success"
    assert payload["summary"]
    assert payload["meta"]["chunk_count"] == 1
    assert not tmp_dir(task_id).exists()

    listing = client.get("/tasks", headers=auth_headers())
    assert listing.status_code == 200
    assert listing.json()[0]["task_id"] == task_id
    assert "summary" not in listing.json()[0]


def test_unknown_task_404(client: TestClient) -> None:
    response = client.get(f"/tasks/{uuid.uuid4()}", headers=auth_headers())
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_queue_running_queued_and_503(client: TestClient) -> None:
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
        assert second.json()["status"] == "queued"
        third = _post_summarize(client)
        assert third.status_code == 503
        assert third.json()["error"]["code"] == "queue_full"
    finally:
        gate.set()
        llm_mod.complete_override = original


def test_delete_queued_ok_running_409(client: TestClient) -> None:
    gate = threading.Event()
    original = _install_slow_stub(gate)
    try:
        first = _post_summarize(client)
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
        second_id = second.json()["meta"]["task_id"]
        deleted = client.delete(f"/tasks/{second_id}", headers=auth_headers())
        assert deleted.status_code == 200
        missing = client.get(f"/tasks/{second_id}", headers=auth_headers())
        assert missing.status_code == 404

        running = client.delete(f"/tasks/{first_id}", headers=auth_headers())
        assert running.status_code == 409
        assert running.json()["error"]["code"] == "task_running"
    finally:
        gate.set()
        llm_mod.complete_override = original


def test_pipeline_error_does_not_increment_attempts(client: TestClient) -> None:
    original = llm_mod.complete_override

    async def boom(settings, system, user):
        raise RuntimeError("simulated python exception")

    llm_mod.complete_override = boom
    try:
        created = _post_summarize(client)
        assert created.status_code == 202
        task_id = created.json()["meta"]["task_id"]
        payload = _poll_client(client, task_id)
        assert payload["status"] == "error"
        assert payload["error"]["code"] == "pipeline_error"
        rec = app.state.runner.store.get(task_id)
        assert rec is not None
        assert rec.attempts == 0
    finally:
        llm_mod.complete_override = original


def test_stub_timeout_maps_to_llm_timeout(client: TestClient) -> None:
    from app.llm import LlmCallError
    from app.schemas import ErrorCode

    original = llm_mod.complete_override

    async def timeout(settings, system, user):
        raise LlmCallError(ErrorCode.llm_timeout, "stub timeout")

    llm_mod.complete_override = timeout
    try:
        created = _post_summarize(client)
        payload = _poll_client(client, created.json()["meta"]["task_id"])
        assert payload["status"] == "error"
        assert payload["error"]["code"] == "llm_timeout"
    finally:
        llm_mod.complete_override = original


def test_task_timeout_cancels_hung_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolate_env(tmp_path, monkeypatch, extra={"TASK_TIMEOUT_SEC": "0.15"})
    original = llm_mod.complete_override

    async def hang(settings, system, user):
        await asyncio.sleep(10)
        return ChatResult(content="too-late", prompt_tokens=1, completion_tokens=1)

    try:
        with TestClient(app) as test_client:
            llm_mod.complete_override = hang
            created = _post_summarize(test_client)
            task_id = created.json()["meta"]["task_id"]
            payload = _poll_client(test_client, task_id)
            assert payload["status"] == "error"
            assert payload["error"]["code"] == "task_timeout"
            assert not tmp_dir(task_id, tmp_path).exists()
    finally:
        llm_mod.complete_override = original
        get_settings.cache_clear()


def test_stub_empty_content_bad_response(client: TestClient) -> None:
    original = llm_mod.complete_override

    async def empty(settings, system, user):
        return ChatResult(content="   ", prompt_tokens=1, completion_tokens=0)

    llm_mod.complete_override = empty
    try:
        created = _post_summarize(client)
        payload = _poll_client(client, created.json()["meta"]["task_id"])
        assert payload["status"] == "error"
        assert payload["error"]["code"] == "llm_bad_response"
    finally:
        llm_mod.complete_override = original


def test_long_text_chunk_count(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.chunking as chunking

    monkeypatch.setattr(chunking, "MAX_PROMPT_CHARS", 400)
    original = llm_mod.complete_override
    calls = {"n": 0}

    async def counted(settings, system, user):
        calls["n"] += 1
        return ChatResult(
            content=f"part-{calls['n']}",
            prompt_tokens=2,
            completion_tokens=2,
            latency_sec=0.001,
        )

    llm_mod.complete_override = counted
    try:
        text = ("paragraph about a meeting.\n\n" * 40) + "end."
        created = _post_summarize(client, text=text, skill="short skill")
        payload = _poll_client(client, created.json()["meta"]["task_id"])
        assert payload["status"] == "success"
        assert payload["meta"]["chunk_count"] > 1
        assert payload["summary"]
        assert calls["n"] > 1
    finally:
        llm_mod.complete_override = original


def test_api_key_not_in_logs(client: TestClient) -> None:
    created = _post_summarize(client)
    _poll_client(client, created.json()["meta"]["task_id"])
    secret = get_settings().API_KEY
    log_path = Path(get_settings().LOG_DIR) / "app.log"
    if log_path.is_file():
        assert secret not in log_path.read_text(encoding="utf-8")


def test_ttl_deletes_finished(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolate_env(tmp_path, monkeypatch)
    store = TaskStore(str(tmp_path / "tasks.db"))
    rec = store.create(
        "ttl-1",
        model="stub-model",
        text_chars=1,
        skill_chars=1,
        payload_dir=str(tmp_path / "tmp" / "ttl-1"),
    )
    store.mark_success(
        rec.task_id,
        summary="done",
        chunk_count=1,
        prepare_time_sec=0.1,
        llm_time_sec=0.1,
        total_time_sec=0.2,
        prompt_tokens=1,
        completion_tokens=1,
    )
    assert store.purge_expired(3600) == 0
    assert store.purge_expired(1) == 1 or store.get("ttl-1") is not None
    # finished_at is now; TTL 1s may not have elapsed. Force old finished_at.
    with store._lock:
        store._conn.execute(
            "UPDATE tasks SET finished_at = ? WHERE task_id = ?",
            ("2000-01-01T00:00:00", "ttl-1"),
        )
        store._conn.commit()
    # re-create if already deleted
    if store.get("ttl-1") is None:
        store.create(
            "ttl-1",
            model="stub-model",
            text_chars=1,
            skill_chars=1,
            payload_dir=str(tmp_path / "tmp" / "ttl-1"),
        )
        store.mark_success(
            "ttl-1",
            summary="done",
            chunk_count=1,
            prepare_time_sec=0.1,
            llm_time_sec=0.1,
            total_time_sec=0.2,
            prompt_tokens=1,
            completion_tokens=1,
        )
        with store._lock:
            store._conn.execute(
                "UPDATE tasks SET finished_at = ? WHERE task_id = ?",
                ("2000-01-01T00:00:00", "ttl-1"),
            )
            store._conn.commit()
    assert store.purge_expired(1) == 1
    assert store.get("ttl-1") is None
    store.close()


def test_two_workers_run_two_tasks_in_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolate_env(tmp_path, monkeypatch, workers="2", queue_size="4")
    entered = threading.Semaphore(0)
    release = threading.Event()

    async def slow(settings, system, user):
        entered.release()
        await asyncio.to_thread(release.wait, 15)
        return ChatResult(content="ok", prompt_tokens=1, completion_tokens=1)

    llm_mod.complete_override = slow

    async def scenario() -> None:
        settings = get_settings()
        runner = TaskRunner(settings)
        await runner.start()
        first = await runner.submit("one", "skill")
        second = await runner.submit("two", "skill")
        assert await asyncio.to_thread(entered.acquire, True, 5)
        assert await asyncio.to_thread(entered.acquire, True, 5)
        running = {
            rec.status
            for rec in (
                runner.store.get(first.task_id),
                runner.store.get(second.task_id),
            )
            if rec is not None
        }
        assert running == {TaskStatus.running}
        release.set()
        done_first = await _wait_store(
            runner.store, first.task_id, {TaskStatus.success, TaskStatus.error}
        )
        done_second = await _wait_store(
            runner.store, second.task_id, {TaskStatus.success, TaskStatus.error}
        )
        await runner.stop()
        assert done_first.status is TaskStatus.success
        assert done_second.status is TaskStatus.success

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        llm_mod.complete_override = stub_chat_complete
        get_settings.cache_clear()


def test_purge_unauthorized(client: TestClient) -> None:
    response = client.delete("/tasks")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_purge_tasks_clears_queue_history_and_orphan_tmp(
    client: TestClient, tmp_path: Path
) -> None:
    created = _post_summarize(client)
    finished_id = created.json()["meta"]["task_id"]
    payload = _poll_client(client, finished_id)
    assert payload["status"] == "success"

    runner = app.state.runner
    queued_ids: list[str] = []
    for _ in range(2):
        tid = str(uuid.uuid4())
        dest = create_tmp(tid, tmp_path)
        (dest / "text.txt").write_text("t", encoding="utf-8")
        (dest / "skill.md").write_text("s", encoding="utf-8")
        runner.store.create(
            tid,
            model=runner.settings.MODEL,
            text_chars=1,
            skill_chars=1,
            payload_dir=str(dest),
        )
        queued_ids.append(tid)

    orphan_id = "orphan-no-row"
    create_tmp(orphan_id, tmp_path)
    assert tmp_dir(orphan_id, tmp_path).is_dir()

    response = client.delete("/tasks", headers=auth_headers())
    assert response.status_code == 200
    body_json = response.json()
    assert body_json["status"] == "ok"
    assert body_json["purged_queued"] == 2
    assert body_json["purged_finished"] == 1
    assert body_json["skipped_running"] == 0
    assert body_json["purged_tmp"] == 3
    listing = client.get("/tasks", headers=auth_headers())
    assert listing.status_code == 200
    assert listing.json() == []
    for tid in queued_ids + [orphan_id]:
        assert not tmp_dir(tid, tmp_path).exists()


def test_purge_skips_running_returns_200_not_409(client: TestClient, tmp_path: Path) -> None:
    gate = threading.Event()
    original = _install_slow_stub(gate)
    try:
        first = _post_summarize(client)
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
        second_id = second.json()["meta"]["task_id"]
        assert second.json()["status"] == "queued"

        response = client.delete("/tasks", headers=auth_headers())
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["purged_queued"] == 1
        assert body["skipped_running"] == 1
        assert body["purged_finished"] == 0
        assert body["purged_tmp"] >= 1
        assert not tmp_dir(second_id, tmp_path).exists()
        assert tmp_dir(first_id, tmp_path).is_dir()

        still = client.get(f"/tasks/{first_id}", headers=auth_headers())
        assert still.status_code == 200
        assert still.json()["status"] == "running"

        listing = client.get("/tasks", headers=auth_headers())
        ids = {item["task_id"] for item in listing.json()}
        assert first_id in ids
        assert second_id not in ids
    finally:
        gate.set()
        llm_mod.complete_override = original

    done = _poll_client(client, first_id)
    assert done["status"] == "success"


def test_purge_counts_legacy_cwd_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    isolate_env(data_dir, monkeypatch)
    get_settings.cache_clear()
    llm_mod.complete_override = stub_chat_complete
    legacy = tmp_path / "tmp_legacy-orphan"
    legacy.mkdir()

    async def scenario() -> None:
        runner = TaskRunner(get_settings())
        await runner.start()
        result = await runner.purge()
        assert result.purged_queued == 0
        assert result.purged_finished == 0
        assert result.skipped_running == 0
        assert result.purged_tmp == 1
        await runner.stop()

    try:
        asyncio.run(scenario())
        assert not legacy.exists()
    finally:
        get_settings.cache_clear()
