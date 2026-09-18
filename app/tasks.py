"""SQLite-реестр задач."""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from cryptography.fernet import Fernet

from app.config import get_settings
from app.schemas import ErrorDetail, TaskStatus


@dataclass
class TaskRecord:
    task_id: str
    status: TaskStatus
    timestamp: str
    model: str
    started_at: str | None = None
    finished_at: str | None = None
    text_chars: int | None = None
    skill_chars: int | None = None
    chunk_count: int | None = None
    prepare_time_sec: float | None = None
    llm_time_sec: float | None = None
    total_time_sec: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    summary: str | None = None
    error: dict[str, object] | None = None
    payload_dir: str | None = None
    attempts: int = 0


def _now() -> str:
    return datetime.now().isoformat()


def _summary_fernet() -> Fernet:
    digest = hashlib.sha256(get_settings().API_TOKEN.encode()).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def _encrypt_summary(payload: str) -> str:
    return _summary_fernet().encrypt(payload.encode()).decode()


def _decode_summary(raw: str | None) -> str | None:
    if not raw:
        return None
    if raw.startswith("gAAAAA"):
        return _summary_fernet().decrypt(raw.encode()).decode()
    return raw


class TaskStore:
    def __init__(self, sqlite_path: str) -> None:
        path = Path(sqlite_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
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
                    payload_dir TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            columns = {
                row[1]
                for row in self._conn.execute("PRAGMA table_info(tasks)").fetchall()
            }
            if "attempts" not in columns:
                self._conn.execute(
                    "ALTER TABLE tasks ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0"
                )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def create(
        self,
        task_id: str,
        *,
        model: str,
        text_chars: int,
        skill_chars: int,
        payload_dir: str,
    ) -> TaskRecord:
        timestamp = _now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO tasks (
                    task_id, status, timestamp, model,
                    text_chars, skill_chars, payload_dir
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    TaskStatus.queued.value,
                    timestamp,
                    model,
                    text_chars,
                    skill_chars,
                    payload_dir,
                ),
            )
            self._conn.commit()
        record = self.get(task_id)
        assert record is not None
        return record

    def get(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        if row is None:
            return None
        return _row_to_record(row)

    def list_tasks(self, status: TaskStatus | None = None) -> list[TaskRecord]:
        query = "SELECT * FROM tasks"
        params: tuple[object, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status.value,)
        query += " ORDER BY timestamp DESC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [_row_to_record(row) for row in rows]

    def list_queued_fifo(self) -> list[TaskRecord]:
        """Queued в порядке постановки (timestamp ASC) — для восстановления после рестарта."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM tasks WHERE status = ? ORDER BY timestamp ASC",
                (TaskStatus.queued.value,),
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def bump_attempts(self, task_id: str) -> int:
        """Увеличить счётчик смертей процесса на running-задаче. Возвращает новое значение."""
        with self._lock:
            self._conn.execute(
                "UPDATE tasks SET attempts = attempts + 1 WHERE task_id = ?",
                (task_id,),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT attempts FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return int(row["attempts"]) if row is not None else 0

    def reset_to_queued(self, task_id: str) -> None:
        """Вернуть running в queued и сбросить поля прогресса, чтобы mark_running снова сработал."""
        with self._lock:
            self._conn.execute(
                """
                UPDATE tasks SET
                    status = ?,
                    started_at = NULL,
                    finished_at = NULL,
                    chunk_count = NULL,
                    prepare_time_sec = NULL,
                    llm_time_sec = NULL,
                    total_time_sec = NULL,
                    prompt_tokens = NULL,
                    completion_tokens = NULL,
                    summary = NULL,
                    error = NULL
                WHERE task_id = ?
                """,
                (TaskStatus.queued.value, task_id),
            )
            self._conn.commit()

    def delete_by_statuses(self, statuses: tuple[TaskStatus, ...]) -> list[TaskRecord]:
        if not statuses:
            return []
        values = tuple(item.value for item in statuses)
        placeholders = ",".join("?" * len(values))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM tasks WHERE status IN ({placeholders})",
                values,
            ).fetchall()
            self._conn.execute(
                f"DELETE FROM tasks WHERE status IN ({placeholders})",
                values,
            )
            self._conn.commit()
        return [_row_to_record(row) for row in rows]

    def count_queued(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM tasks WHERE status = ?",
                (TaskStatus.queued.value,),
            ).fetchone()
        return int(row["n"])

    def mark_running(self, task_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                """
                UPDATE tasks SET status = ?, started_at = ?
                WHERE task_id = ? AND status = ?
                """,
                (TaskStatus.running.value, _now(), task_id, TaskStatus.queued.value),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def mark_success(
        self,
        task_id: str,
        *,
        summary: str,
        chunk_count: int,
        prepare_time_sec: float,
        llm_time_sec: float,
        total_time_sec: float,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> None:
        payload = _encrypt_summary(summary)
        with self._lock:
            self._conn.execute(
                """
                UPDATE tasks SET
                    status = ?, finished_at = ?,
                    chunk_count = ?, prepare_time_sec = ?, llm_time_sec = ?,
                    total_time_sec = ?, prompt_tokens = ?, completion_tokens = ?,
                    summary = ?, error = NULL
                WHERE task_id = ?
                """,
                (
                    TaskStatus.success.value,
                    _now(),
                    chunk_count,
                    prepare_time_sec,
                    llm_time_sec,
                    total_time_sec,
                    prompt_tokens,
                    completion_tokens,
                    payload,
                    task_id,
                ),
            )
            self._conn.commit()

    def mark_error(
        self,
        task_id: str,
        error: ErrorDetail,
        *,
        chunk_count: int | None = None,
        prepare_time_sec: float | None = None,
        llm_time_sec: float | None = None,
        total_time_sec: float | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE tasks SET
                    status = ?, finished_at = ?, error = ?,
                    chunk_count = COALESCE(?, chunk_count),
                    prepare_time_sec = COALESCE(?, prepare_time_sec),
                    llm_time_sec = COALESCE(?, llm_time_sec),
                    total_time_sec = COALESCE(?, total_time_sec),
                    prompt_tokens = COALESCE(?, prompt_tokens),
                    completion_tokens = COALESCE(?, completion_tokens)
                WHERE task_id = ?
                """,
                (
                    TaskStatus.error.value,
                    _now(),
                    error.model_dump_json(),
                    chunk_count,
                    prepare_time_sec,
                    llm_time_sec,
                    total_time_sec,
                    prompt_tokens,
                    completion_tokens,
                    task_id,
                ),
            )
            self._conn.commit()

    def delete(self, task_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def delete_if_not_running(self, task_id: str) -> TaskStatus | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is None:
                return None
            status = TaskStatus(row["status"])
            if status is TaskStatus.running:
                return status
            self._conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
            self._conn.commit()
            return status

    def purge_expired(self, ttl_sec: int) -> int:
        if ttl_sec <= 0:
            return 0
        cutoff = (datetime.now() - timedelta(seconds=ttl_sec)).isoformat()
        with self._lock:
            cur = self._conn.execute(
                """
                DELETE FROM tasks
                WHERE status IN (?, ?)
                  AND finished_at IS NOT NULL
                  AND finished_at <= ?
                """,
                (TaskStatus.success.value, TaskStatus.error.value, cutoff),
            )
            self._conn.commit()
            return cur.rowcount


def _row_to_record(row: sqlite3.Row) -> TaskRecord:
    error = json.loads(row["error"]) if row["error"] else None
    return TaskRecord(
        task_id=row["task_id"],
        status=TaskStatus(row["status"]),
        timestamp=row["timestamp"],
        model=row["model"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        text_chars=row["text_chars"],
        skill_chars=row["skill_chars"],
        chunk_count=row["chunk_count"],
        prepare_time_sec=row["prepare_time_sec"],
        llm_time_sec=row["llm_time_sec"],
        total_time_sec=row["total_time_sec"],
        prompt_tokens=row["prompt_tokens"],
        completion_tokens=row["completion_tokens"],
        summary=_decode_summary(row["summary"]),
        error=error,
        payload_dir=row["payload_dir"],
        attempts=int(row["attempts"]) if "attempts" in row.keys() else 0,
    )
