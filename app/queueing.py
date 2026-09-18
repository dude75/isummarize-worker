"""Слоты WORKERS и очередь queued (WORKER_QUEUE_SIZE)."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from app.config import Settings, get_settings
from app.pipeline import run_pipeline
from app.prometheus_metrics import observe_restore, observe_submitted
from app.schemas import ErrorCode, ErrorDetail, PurgeResult, TaskStatus
from app.storage import (
    cleanup_legacy_cwd_tmp,
    cleanup_tmp,
    cleanup_tmp_except,
    payload_exists,
    write_payload,
)
from app.tasks import TaskRecord, TaskStore


class QueueFullError(Exception):
    code = ErrorCode.queue_full


class TaskRunningError(Exception):
    code = ErrorCode.task_running


class TaskRunner:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = TaskStore(self.settings.SQLITE_PATH)
        self._free_slots: asyncio.Queue[int] = asyncio.Queue()
        for index in range(self.settings.WORKERS):
            self._free_slots.put_nowait(index)
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._submit_lock = asyncio.Lock()
        self._cancelled: set[str] = set()
        self._tasks: set[asyncio.Task[None]] = set()
        self._dispatcher: asyncio.Task[None] | None = None
        self._ttl_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        Path(self.settings.LOG_DIR).mkdir(parents=True, exist_ok=True)
        self._restore_unfinished()
        self._dispatcher = asyncio.create_task(self._dispatch_loop())
        self._ttl_task = asyncio.create_task(self._ttl_loop())

    def _restore_unfinished(self) -> None:
        data_dir = self.settings.DATA_DIR
        for record in self.store.list_tasks(TaskStatus.running):
            if not payload_exists(record.task_id, data_dir):
                self.store.mark_error(
                    record.task_id, ErrorDetail(code=ErrorCode.interrupted)
                )
                observe_restore(ErrorCode.interrupted.value)
                cleanup_tmp(record.task_id, data_dir)
                continue
            attempts = self.store.bump_attempts(record.task_id)
            if attempts > self.settings.TASK_MAX_RESTARTS:
                self.store.mark_error(
                    record.task_id, ErrorDetail(code=ErrorCode.process_killed)
                )
                observe_restore(ErrorCode.process_killed.value)
                cleanup_tmp(record.task_id, data_dir)
            else:
                self.store.reset_to_queued(record.task_id)
        for record in self.store.list_tasks(TaskStatus.queued):
            if not payload_exists(record.task_id, data_dir):
                self.store.mark_error(
                    record.task_id, ErrorDetail(code=ErrorCode.missing_payload)
                )
                observe_restore(ErrorCode.missing_payload.value)
                cleanup_tmp(record.task_id, data_dir)
        # WORKER_QUEUE_SIZE не применяется: после рестарта очередь может быть длиннее лимита.
        for record in self.store.list_queued_fifo():
            self._queue.put_nowait(record.task_id)
            observe_restore("requeued")

    async def stop(self) -> None:
        if self._dispatcher is not None:
            self._dispatcher.cancel()
        if self._ttl_task is not None:
            self._ttl_task.cancel()
        pending = list(self._tasks)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        # queued/running tmp сохраняем — после рестарта пайплайн стартует с payload.
        for record in self.store.list_tasks():
            if record.status in {TaskStatus.success, TaskStatus.error}:
                cleanup_tmp(record.task_id, self.settings.DATA_DIR)
        self.store.close()

    async def submit(self, text: str, skill: str) -> TaskRecord:
        async with self._submit_lock:
            if self.store.count_queued() >= self.settings.WORKER_QUEUE_SIZE:
                raise QueueFullError()
            task_id = str(uuid.uuid4())
            payload_dir = await asyncio.to_thread(
                write_payload,
                task_id,
                self.settings.DATA_DIR,
                text=text,
                skill=skill,
                model=self.settings.MODEL,
            )
            record = self.store.create(
                task_id,
                model=self.settings.MODEL,
                text_chars=len(text),
                skill_chars=len(skill),
                payload_dir=str(payload_dir),
            )
            await self._queue.put(task_id)
            observe_submitted(self.settings.MODEL)
            return record

    async def delete(self, task_id: str) -> None:
        status = self.store.delete_if_not_running(task_id)
        if status is None:
            raise KeyError(task_id)
        if status is TaskStatus.running:
            raise TaskRunningError()
        if status is TaskStatus.queued:
            self._cancelled.add(task_id)
        cleanup_tmp(task_id, self.settings.DATA_DIR)

    async def purge(self) -> PurgeResult:
        """Снести queued и историю; running не трогать. SQL только в store."""
        async with self._submit_lock:
            running = self.store.list_tasks(TaskStatus.running)
            keep_ids = {item.task_id for item in running}
            queued = self.store.delete_by_statuses((TaskStatus.queued,))
            for record in queued:
                self._cancelled.add(record.task_id)
            finished = self.store.delete_by_statuses(
                (TaskStatus.success, TaskStatus.error)
            )
            purged_tmp = cleanup_tmp_except(keep_ids, self.settings.DATA_DIR)
            purged_tmp += cleanup_legacy_cwd_tmp()
            return PurgeResult(
                status="ok",
                purged_queued=len(queued),
                purged_finished=len(finished),
                purged_tmp=purged_tmp,
                skipped_running=len(running),
            )

    async def _dispatch_loop(self) -> None:
        while True:
            task_id = await self._queue.get()
            task = asyncio.create_task(self._run_one(task_id))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _run_one(self, task_id: str) -> None:
        try:
            if task_id in self._cancelled:
                return
            slot = await self._free_slots.get()
            try:
                if task_id in self._cancelled or self.store.get(task_id) is None:
                    return
                await run_pipeline(self.store, self.settings, task_id, slot)
            finally:
                self._free_slots.put_nowait(slot)
        finally:
            self._queue.task_done()

    async def _ttl_loop(self) -> None:
        while True:
            self.store.purge_expired(self.settings.TASK_TTL_SEC)
            await asyncio.sleep(30)
