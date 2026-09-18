from __future__ import annotations

import asyncio
import sqlite3
import threading
from pathlib import Path

import pytest

from app import llm as llm_mod
from app.config import get_settings
from app.llm import ChatResult, stub_chat_complete
from app.queueing import TaskRunner
from app.schemas import TaskStatus
from app.storage import write_payload
from app.tasks import TaskStore
from tests.conftest import isolate_env
from tests.test_api_tasks import _wait_store


def _seed_payload(tmp_path: Path, task_id: str, store: TaskStore, model: str = "stub-model") -> None:
    dest = write_payload(
        task_id,
        tmp_path,
        text="transcript text",
        skill="skill rules",
        model=model,
    )
    store.create(
        task_id,
        model=model,
        text_chars=len("transcript text"),
        skill_chars=len("skill rules"),
        payload_dir=str(dest),
    )


def test_store_create_get_list(tmp_path: Path) -> None:
    store = TaskStore(str(tmp_path / "tasks.db"))
    rec = store.create(
        "t1",
        model="m",
        text_chars=10,
        skill_chars=4,
        payload_dir=str(tmp_path / "tmp" / "t1"),
    )
    assert rec.status is TaskStatus.queued
    assert rec.attempts == 0
    got = store.get("t1")
    assert got is not None
    assert got.model == "m"
    assert store.count_queued() == 1
    assert store.list_tasks()[0].task_id == "t1"
    store.close()


def test_attempts_column_migrated_on_existing_db(tmp_path: Path) -> None:
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """
        CREATE TABLE tasks (
            task_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            model TEXT NOT NULL,
            text_chars INTEGER,
            skill_chars INTEGER,
            chunk_count INTEGER,
            prepare_time_sec REAL,
            llm_time_sec REAL,
            total_time_sec REAL,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            summary TEXT,
            error TEXT,
            payload_dir TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO tasks (task_id, status, timestamp, model)
        VALUES (?, ?, ?, ?)
        """,
        ("legacy-1", "queued", "2026-01-01T00:00:00", "m"),
    )
    conn.commit()
    conn.close()

    store = TaskStore(str(db))
    rec = store.get("legacy-1")
    assert rec is not None
    assert rec.attempts == 0
    assert store.bump_attempts("legacy-1") == 1
    rec = store.get("legacy-1")
    assert rec is not None
    assert rec.attempts == 1
    store.close()


def test_restart_queued_survives_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolate_env(tmp_path, monkeypatch, workers="1", queue_size="4")
    gate = threading.Event()
    released = threading.Event()

    async def slow(settings, system, user):
        await asyncio.to_thread(gate.wait, 15)
        released.set()
        return ChatResult(content="ok", prompt_tokens=1, completion_tokens=1)

    llm_mod.complete_override = slow

    async def scenario() -> None:
        settings = get_settings()
        runner = TaskRunner(settings)
        await runner.start()
        first = await runner.submit("one", "skill")
        await _wait_store(runner.store, first.task_id, {TaskStatus.running})
        second = await runner.submit("two", "skill")
        assert second.status is TaskStatus.queued
        payload = Path(second.payload_dir or "")
        assert (payload / "text.txt").is_file()
        await runner.stop()
        assert (payload / "text.txt").is_file()
        gate.set()
        released.wait(timeout=5)
        await asyncio.sleep(0.05)
        llm_mod.complete_override = stub_chat_complete
        runner2 = TaskRunner(settings)
        await runner2.start()
        done = await _wait_store(
            runner2.store, second.task_id, {TaskStatus.success, TaskStatus.error}
        )
        assert done.status is TaskStatus.success
        await runner2.stop()

    try:
        asyncio.run(scenario())
    finally:
        gate.set()
        llm_mod.complete_override = stub_chat_complete
        get_settings.cache_clear()


def test_restart_running_requeued_not_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolate_env(tmp_path, monkeypatch, workers="1", queue_size="4")
    gate = threading.Event()
    released = threading.Event()

    async def slow(settings, system, user):
        await asyncio.to_thread(gate.wait, 15)
        released.set()
        return ChatResult(content="ok", prompt_tokens=1, completion_tokens=1)

    llm_mod.complete_override = slow

    async def scenario() -> None:
        settings = get_settings()
        runner = TaskRunner(settings)
        await runner.start()
        first = await runner.submit("one", "skill")
        await _wait_store(runner.store, first.task_id, {TaskStatus.running})
        assert Path(first.payload_dir or "").joinpath("text.txt").is_file()
        await runner.stop()
        assert Path(first.payload_dir or "").joinpath("text.txt").is_file()
        gate.set()
        released.wait(timeout=5)
        await asyncio.sleep(0.05)
        llm_mod.complete_override = stub_chat_complete
        runner2 = TaskRunner(settings)
        await runner2.start()
        rec = runner2.store.get(first.task_id)
        assert rec is not None
        if rec.status is TaskStatus.error:
            assert rec.error is None or rec.error.get("code") != "interrupted"
        else:
            assert rec.status in {
                TaskStatus.queued,
                TaskStatus.running,
                TaskStatus.success,
            }
        done = await _wait_store(
            runner2.store, first.task_id, {TaskStatus.success, TaskStatus.error}
        )
        assert done.status is TaskStatus.success
        await runner2.stop()

    try:
        asyncio.run(scenario())
    finally:
        gate.set()
        llm_mod.complete_override = stub_chat_complete
        get_settings.cache_clear()


def test_restore_missing_payload_errors_not_queued(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolate_env(tmp_path, monkeypatch)
    settings = get_settings()
    store = TaskStore(settings.SQLITE_PATH)
    missing = store.create(
        "missing-payload-1",
        model=settings.MODEL,
        text_chars=1,
        skill_chars=1,
        payload_dir=str(tmp_path / "no-such"),
    )
    store.close()

    async def scenario() -> None:
        runner = TaskRunner(settings)
        await runner.start()
        rec = runner.store.get(missing.task_id)
        assert rec is not None
        assert rec.status is TaskStatus.error
        assert rec.error is not None
        assert rec.error["code"] == "missing_payload"
        assert runner.store.count_queued() == 0
        await runner.stop()

    try:
        asyncio.run(scenario())
    finally:
        get_settings.cache_clear()


def test_restore_running_without_files_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolate_env(tmp_path, monkeypatch)
    settings = get_settings()
    store = TaskStore(settings.SQLITE_PATH)
    store.create(
        "gone-running",
        model=settings.MODEL,
        text_chars=1,
        skill_chars=1,
        payload_dir=str(tmp_path / "missing"),
    )
    assert store.mark_running("gone-running")
    store.close()

    async def scenario() -> None:
        runner = TaskRunner(settings)
        await runner.start()
        rec = runner.store.get("gone-running")
        assert rec is not None
        assert rec.status is TaskStatus.error
        assert rec.error is not None
        assert rec.error["code"] == "interrupted"
        await runner.stop()

    try:
        asyncio.run(scenario())
    finally:
        get_settings.cache_clear()


def test_restore_fifo_order_and_queue_limit_not_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolate_env(tmp_path, monkeypatch, workers="1", queue_size="1")
    llm_mod.complete_override = stub_chat_complete
    settings = get_settings()
    store = TaskStore(settings.SQLITE_PATH)
    ids = ["fifo-a", "fifo-b"]
    stamps = ["2026-01-01T00:00:00", "2026-01-01T00:00:01"]
    for task_id, stamp in zip(ids, stamps, strict=True):
        _seed_payload(tmp_path, task_id, store, settings.MODEL)
        with store._lock:
            store._conn.execute(
                "UPDATE tasks SET timestamp = ? WHERE task_id = ?",
                (stamp, task_id),
            )
            store._conn.commit()
    store.close()

    async def scenario() -> None:
        runner = TaskRunner(settings)
        await runner.start()
        first = await _wait_store(
            runner.store, "fifo-a", {TaskStatus.success, TaskStatus.error}
        )
        second = await _wait_store(
            runner.store, "fifo-b", {TaskStatus.success, TaskStatus.error}
        )
        assert first.status is TaskStatus.success
        assert second.status is TaskStatus.success
        assert first.started_at is not None
        assert second.started_at is not None
        assert first.started_at <= second.started_at
        await runner.stop()

    try:
        asyncio.run(scenario())
    finally:
        get_settings.cache_clear()


def _seed_running_with_payload(tmp_path: Path, task_id: str) -> None:
    settings = get_settings()
    store = TaskStore(settings.SQLITE_PATH)
    _seed_payload(tmp_path, task_id, store, settings.MODEL)
    assert store.mark_running(task_id)
    store.close()


def test_restore_running_zero_restarts_process_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolate_env(tmp_path, monkeypatch, max_restarts="0")
    _seed_running_with_payload(tmp_path, "poison-0")

    async def scenario() -> None:
        runner = TaskRunner(get_settings())
        await runner.start()
        rec = runner.store.get("poison-0")
        assert rec is not None
        assert rec.status is TaskStatus.error
        assert rec.error is not None
        assert rec.error["code"] == "process_killed"
        assert rec.attempts == 1
        assert runner.store.count_queued() == 0
        assert not (tmp_path / "tmp" / "poison-0").exists()
        await runner.delete("poison-0")
        assert runner.store.get("poison-0") is None
        await runner.stop()

    try:
        asyncio.run(scenario())
    finally:
        get_settings.cache_clear()


def test_restore_running_under_limit_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolate_env(tmp_path, monkeypatch)
    llm_mod.complete_override = stub_chat_complete
    _seed_running_with_payload(tmp_path, "retry-1")

    async def scenario() -> None:
        runner = TaskRunner(get_settings())
        await runner.start()
        done = await _wait_store(
            runner.store, "retry-1", {TaskStatus.success, TaskStatus.error}
        )
        assert done.status is TaskStatus.success
        assert done.attempts == 1
        await runner.stop()

    try:
        asyncio.run(scenario())
    finally:
        get_settings.cache_clear()


def test_restore_running_errors_when_attempts_exceed_max(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolate_env(tmp_path, monkeypatch)
    _seed_running_with_payload(tmp_path, "poison-2")
    store = TaskStore(get_settings().SQLITE_PATH)
    assert store.bump_attempts("poison-2") == 1
    store.close()

    async def scenario() -> None:
        runner = TaskRunner(get_settings())
        await runner.start()
        rec = runner.store.get("poison-2")
        assert rec is not None
        assert rec.status is TaskStatus.error
        assert rec.error is not None
        assert rec.error["code"] == "process_killed"
        assert rec.attempts == 2
        assert runner.store.count_queued() == 0
        assert not (tmp_path / "tmp" / "poison-2").exists()
        await runner.stop()

    try:
        asyncio.run(scenario())
    finally:
        get_settings.cache_clear()
