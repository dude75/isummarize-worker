"""Пайплайн задачи: payload → prompt/chunk → LLM → summary, метрики, чистка tmp."""

from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
import time
from contextlib import asynccontextmanager

from app.chunking import (
    fits,
    plan_chunks,
    user_chunk,
    user_reduce,
    user_single,
)
from app.config import Settings
from app import llm as llm_mod
from app.llm import ChatResult, LlmCallError, chat_complete
from app.metrics import MetricEvent, write_metric
from app.prometheus_metrics import observe_task_finished, queue_wait_sec
from app.schemas import ErrorCode, ErrorDetail, TaskStatus
from app.storage import (
    cleanup_tmp,
    payload_exists,
    read_payload,
    write_chunk_summary,
    write_reduce_summary,
)
from app.tasks import TaskRecord, TaskStore

logger = logging.getLogger("app.pipeline")

_FENCE_RE = re.compile(r"^```[^\n]*\n(.*)\n```$", re.DOTALL)


class TaskFailed(Exception):
    def __init__(self, code: ErrorCode, message: str | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.message = message


@asynccontextmanager
async def _task_deadline(settings: Settings):
    timeout = settings.TASK_TIMEOUT_SEC
    if timeout <= 0:
        yield
        return
    async with asyncio.timeout(timeout):
        yield


def _skill_wants_fence(skill: str) -> bool:
    lower = skill.lower()
    needles = ("```", "code fence", "markdown fence", "огражд", "кодблок", "code block")
    return any(needle in lower for needle in needles)


def normalize_output(text: str, skill: str) -> str:
    stripped = text.strip()
    if not stripped:
        raise TaskFailed(ErrorCode.llm_bad_response, "empty model content")
    match = _FENCE_RE.match(stripped)
    if match is not None and not _skill_wants_fence(skill):
        inner = match.group(1).strip()
        if inner:
            return inner
    return stripped


def _add_tokens(current: int | None, incoming: int | None) -> int | None:
    if incoming is None:
        return current
    return (current or 0) + incoming


def _emit_task_metrics(
    settings: Settings,
    record: TaskRecord,
    task_id: str,
    *,
    status: str,
    prepare_time: float | None,
    llm_time: float | None,
    total_time: float | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    chunk_count: int | None,
    error_code: ErrorCode | None = None,
) -> None:
    write_metric(
        settings.PERFORMANCE_LOG,
        MetricEvent(
            timestamp=record.timestamp,
            task_id=task_id,
            model=record.model,
            text_chars=record.text_chars,
            skill_chars=record.skill_chars,
            chunk_count=chunk_count,
            prepare_time_sec=prepare_time,
            llm_time_sec=llm_time,
            total_time_sec=total_time,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            status=status,
        ),
    )
    observe_task_finished(
        model=record.model,
        status=status,
        error_code=None if error_code is None else error_code.value,
        llm_time_sec=llm_time,
        total_time_sec=total_time,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        queue_wait=queue_wait_sec(record.timestamp),
    )


async def _call_llm(
    settings: Settings,
    skill: str,
    user: str,
    task_id: str,
) -> ChatResult:
    try:
        result = await chat_complete(settings, skill, user, task_id=task_id)
    except LlmCallError as exc:
        raise TaskFailed(exc.code, exc.message) from exc
    try:
        result.content = normalize_output(result.content, skill)
    except TaskFailed:
        raise
    return result


async def _reduce(
    settings: Settings,
    skill: str,
    summaries: list[str],
    task_id: str,
) -> ChatResult:
    user = user_reduce(summaries)
    if fits(skill, user):
        return await _call_llm(settings, skill, user, task_id)
    if len(summaries) == 1:
        raise TaskFailed(ErrorCode.text_too_long, "reduce prompt does not fit")
    mid = max(1, len(summaries) // 2)
    left = await _reduce(settings, skill, summaries[:mid], task_id)
    right = await _reduce(settings, skill, summaries[mid:], task_id)
    merged = await _reduce(settings, skill, [left.content, right.content], task_id)
    merged.prompt_tokens = _add_tokens(
        _add_tokens(left.prompt_tokens, right.prompt_tokens), merged.prompt_tokens
    )
    merged.completion_tokens = _add_tokens(
        _add_tokens(left.completion_tokens, right.completion_tokens),
        merged.completion_tokens,
    )
    merged.latency_sec = left.latency_sec + right.latency_sec + merged.latency_sec
    return merged


async def run_pipeline(store: TaskStore, settings: Settings, task_id: str, slot: int = 0) -> None:
    del slot
    record = store.get(task_id)
    if record is None:
        return
    if not store.mark_running(task_id):
        return
    started = time.perf_counter()
    prepare_time: float | None = None
    llm_time: float | None = None
    total_time: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    chunk_count: int | None = None
    outcome: str | None = None
    error: ErrorDetail | None = None
    try:
        async with _task_deadline(settings):
            if not settings.llm_configured() and llm_mod.complete_override is None:
                raise TaskFailed(ErrorCode.llm_unconfigured)
            if not payload_exists(task_id, settings.DATA_DIR):
                raise TaskFailed(ErrorCode.missing_payload)
            text, skill = read_payload(task_id, settings.DATA_DIR)
            chunks = plan_chunks(skill, text)
            prepare_time = time.perf_counter() - started
            if chunks is not None and len(chunks) == 0:
                raise TaskFailed(ErrorCode.text_too_long, "skill + chunk does not fit")

            llm_started = time.perf_counter()
            if chunks is None:
                chunk_count = 1
                result = await _call_llm(settings, skill, user_single(text), task_id)
                prompt_tokens = _add_tokens(prompt_tokens, result.prompt_tokens)
                completion_tokens = _add_tokens(completion_tokens, result.completion_tokens)
                summary = result.content
            else:
                chunk_count = len(chunks)
                partials: list[str] = []
                for index, chunk in enumerate(chunks, start=1):
                    result = await _call_llm(
                        settings, skill, user_chunk(chunk, index, len(chunks)), task_id
                    )
                    prompt_tokens = _add_tokens(prompt_tokens, result.prompt_tokens)
                    completion_tokens = _add_tokens(
                        completion_tokens, result.completion_tokens
                    )
                    write_chunk_summary(task_id, settings.DATA_DIR, index, result.content)
                    partials.append(result.content)
                reduced = await _reduce(settings, skill, partials, task_id)
                prompt_tokens = _add_tokens(prompt_tokens, reduced.prompt_tokens)
                completion_tokens = _add_tokens(
                    completion_tokens, reduced.completion_tokens
                )
                write_reduce_summary(task_id, settings.DATA_DIR, reduced.content)
                summary = reduced.content
            llm_time = time.perf_counter() - llm_started
            total_time = time.perf_counter() - started
            store.mark_success(
                task_id,
                summary=summary,
                chunk_count=chunk_count,
                prepare_time_sec=prepare_time,
                llm_time_sec=llm_time,
                total_time_sec=total_time,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            outcome = "success"
    except Exception as exc:
        total_time = time.perf_counter() - started
        if isinstance(exc, TimeoutError):
            logger.warning(
                "task %s timed out after %s sec", task_id, settings.TASK_TIMEOUT_SEC
            )
            error = ErrorDetail(
                code=ErrorCode.task_timeout,
                message=f"exceeded TASK_TIMEOUT_SEC={settings.TASK_TIMEOUT_SEC}",
            )
        elif isinstance(exc, TaskFailed):
            error = ErrorDetail(code=exc.code, message=exc.message)
        elif isinstance(exc, LlmCallError):
            error = ErrorDetail(code=exc.code, message=exc.message)
        else:
            error = ErrorDetail(code=ErrorCode.pipeline_error, message=str(exc))
        store.mark_error(
            task_id,
            error,
            chunk_count=chunk_count,
            prepare_time_sec=prepare_time,
            llm_time_sec=llm_time,
            total_time_sec=total_time,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        outcome = "error"
    finally:
        try:
            current = store.get(task_id)
        except sqlite3.Error:
            pass
        else:
            if current is None or current.status in {TaskStatus.success, TaskStatus.error}:
                cleanup_tmp(task_id, settings.DATA_DIR)
        if outcome is not None:
            _emit_task_metrics(
                settings,
                record,
                task_id,
                status=outcome,
                prepare_time=prepare_time,
                llm_time=llm_time,
                total_time=total_time,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                chunk_count=chunk_count,
                error_code=None if error is None else error.code,
            )
