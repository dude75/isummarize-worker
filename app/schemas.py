"""Схемы API: queued / running / success / error."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskStatus(str, Enum):
    queued = "queued"
    running = "running"
    success = "success"
    error = "error"


class LlmHealth(str, Enum):
    ready = "ready"
    unconfigured = "unconfigured"
    unavailable = "unavailable"


class ErrorCode(str, Enum):
    unauthorized = "unauthorized"
    payload_too_large = "payload_too_large"
    queue_full = "queue_full"
    not_found = "not_found"
    task_running = "task_running"
    missing_payload = "missing_payload"
    interrupted = "interrupted"
    process_killed = "process_killed"
    llm_unconfigured = "llm_unconfigured"
    llm_unavailable = "llm_unavailable"
    llm_timeout = "llm_timeout"
    llm_bad_response = "llm_bad_response"
    task_timeout = "task_timeout"
    text_too_long = "text_too_long"
    pipeline_error = "pipeline_error"


def error_payload(code: ErrorCode) -> dict[str, Any]:
    return {"status": "error", "error": {"code": code.value}}


class ErrorDetail(BaseModel):
    code: ErrorCode
    message: str | None = None


class TaskMeta(BaseModel):
    timestamp: str
    task_id: str
    model: str
    text_chars: int | None = None
    skill_chars: int | None = None
    chunk_count: int | None = None
    prepare_time_sec: float | None = None
    llm_time_sec: float | None = None
    total_time_sec: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class TaskResponse(BaseModel):
    status: TaskStatus
    meta: TaskMeta
    summary: str | None = None
    error: ErrorDetail | None = None


class TaskListItem(BaseModel):
    task_id: str
    status: TaskStatus
    timestamp: str
    model: str
    text_chars: int | None = None
    skill_chars: int | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    llm: LlmHealth


class PurgeResult(BaseModel):
    status: str = "ok"
    purged_queued: int
    purged_finished: int
    purged_tmp: int
    skipped_running: int


class SummarizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, description="Транскрипт")
    skill: str = Field(..., min_length=1, description="Правила саммаризации (SKILL)")

    @field_validator("text", "skill")
    @classmethod
    def _strip_nonempty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped
