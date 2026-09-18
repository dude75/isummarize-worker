"""Prometheus registry and scrape-time gauges. Independent of CSV `metric_event`."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    Info,
    generate_latest,
)
from prometheus_client.gc_collector import GCCollector
from prometheus_client.metrics_core import GaugeMetricFamily
from prometheus_client.platform_collector import PlatformCollector
from prometheus_client.process_collector import ProcessCollector
from prometheus_client.registry import Collector
from starlette.requests import Request
from starlette.routing import Match

from app.schemas import LlmHealth, TaskStatus
from app.storage import tmp_root
from app.version import read_version

if TYPE_CHECKING:
    from app.config import Settings
    from app.queueing import TaskRunner

CONTENT_TYPE = CONTENT_TYPE_LATEST

STAGE_BUCKETS = (1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0, 1800.0)
QUEUE_WAIT_BUCKETS = (0.1, 0.5, 1.0, 2.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0)
HTTP_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
TOKEN_BUCKETS = (100.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 32000.0, 64000.0)

_active: Metrics | None = None


def queue_wait_sec(timestamp: str) -> float | None:
    try:
        queued_at = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    return max((datetime.now() - queued_at).total_seconds(), 0.0)


def http_path_template(request: Request) -> str:
    for route in request.app.router.routes:
        if not hasattr(route, "matches"):
            continue
        match, _child = route.matches(request.scope)
        if match == Match.FULL:
            path = getattr(route, "path", None)
            if isinstance(path, str) and path:
                return path
    return "unknown"


@dataclass
class RuntimeState:
    settings: Settings | None = None
    runner: TaskRunner | None = None
    runner_started: bool = False
    llm_status: LlmHealth = LlmHealth.unconfigured


class RuntimeCollector(Collector):
    def __init__(self, state: RuntimeState) -> None:
        self.state = state

    def collect(self):
        yield GaugeMetricFamily("isummarize_up", "Process is serving /metrics", value=1.0)
        yield GaugeMetricFamily(
            "isummarize_ready",
            "Runner started and LLM client is ready",
            value=1.0 if _ready(self.state) else 0.0,
        )
        yield from _llm_metrics(self.state)
        yield from _queue_metrics(self.state)
        yield from _disk_metrics(self.state.settings)


def _ready(state: RuntimeState) -> bool:
    return state.runner_started and state.llm_status is LlmHealth.ready


def _llm_metrics(state: RuntimeState):
    status_g = GaugeMetricFamily(
        "isummarize_llm_status",
        "1 for the current LLM client status",
        labels=["status"],
    )
    for name in LlmHealth:
        status_g.add_metric([name.value], 1.0 if state.llm_status is name else 0.0)
    yield status_g


def _queue_metrics(state: RuntimeState):
    settings = state.settings
    slots = float(settings.WORKERS) if settings is not None else 0.0
    limit = float(settings.WORKER_QUEUE_SIZE) if settings is not None else 0.0
    depth = 0.0
    running = 0.0
    age = 0.0
    store = state.runner.store if state.runner is not None else None
    if store is not None:
        try:
            depth = float(store.count_queued())
            running_rows = store.list_tasks(TaskStatus.running)
            running = float(len(running_rows))
            age = _max_running_age(running_rows)
        except (sqlite3.Error, OSError):
            pass
    yield GaugeMetricFamily("isummarize_queue_depth", "Tasks in queued", value=depth)
    yield GaugeMetricFamily("isummarize_tasks_running", "Tasks in running", value=running)
    yield GaugeMetricFamily("isummarize_worker_slots", "Configured WORKERS", value=slots)
    yield GaugeMetricFamily("isummarize_queue_limit", "Configured WORKER_QUEUE_SIZE", value=limit)
    yield GaugeMetricFamily(
        "isummarize_task_running_age_seconds",
        "Age of the oldest running task",
        value=age,
    )


def _max_running_age(records) -> float:
    now = datetime.now()
    max_age = 0.0
    for record in records:
        if not record.started_at:
            continue
        try:
            started = datetime.fromisoformat(record.started_at)
        except ValueError:
            continue
        max_age = max(max_age, (now - started).total_seconds())
    return max_age


def _disk_metrics(settings: Settings | None):
    tmp_bytes = 0
    tmp_dirs = 0
    sqlite_bytes = 0
    if settings is not None:
        tmp_bytes, tmp_dirs = _tmp_stats(Path(settings.DATA_DIR))
        sqlite_path = Path(settings.SQLITE_PATH)
        try:
            if sqlite_path.is_file():
                sqlite_bytes = sqlite_path.stat().st_size
        except OSError:
            sqlite_bytes = 0
    yield GaugeMetricFamily("isummarize_tmp_bytes", "Bytes under DATA_DIR/tmp", value=float(tmp_bytes))
    yield GaugeMetricFamily("isummarize_tmp_dirs", "Task directories under DATA_DIR/tmp", value=float(tmp_dirs))
    yield GaugeMetricFamily("isummarize_sqlite_bytes", "Size of SQLITE_PATH", value=float(sqlite_bytes))


def _tmp_stats(data_dir: Path) -> tuple[int, int]:
    root = tmp_root(data_dir)
    if not root.is_dir():
        return 0, 0
    total = 0
    dirs = 0
    try:
        children = list(root.iterdir())
    except OSError:
        return 0, 0
    for child in children:
        try:
            if child.is_dir():
                dirs += 1
                for file in child.rglob("*"):
                    if file.is_file():
                        try:
                            total += file.stat().st_size
                        except OSError:
                            pass
            elif child.is_file():
                total += child.stat().st_size
        except OSError:
            continue
    return total, dirs


class Metrics:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.registry = CollectorRegistry()
        ProcessCollector(registry=self.registry)
        PlatformCollector(registry=self.registry)
        GCCollector(registry=self.registry)
        self.runtime = RuntimeState()
        self._info: Info | None = None
        if not enabled:
            return
        self.registry.register(RuntimeCollector(self.runtime))
        self._info = Info("isummarize", "Process version and model", registry=self.registry)
        self.queue_rejected = Counter(
            "isummarize_queue_rejected_total",
            "POST /summarize rejected with queue_full",
            registry=self.registry,
        )
        self.tasks_submitted = Counter(
            "isummarize_tasks_submitted_total",
            "Tasks accepted into the queue",
            ["model"],
            registry=self.registry,
        )
        self.tasks_completed = Counter(
            "isummarize_tasks_completed_total",
            "Tasks that reached success or error",
            ["model", "status"],
            registry=self.registry,
        )
        self.task_errors = Counter(
            "isummarize_task_errors_total",
            "Failed tasks by error code",
            ["error_code", "model"],
            registry=self.registry,
        )
        self.pipeline_duration = Histogram(
            "isummarize_pipeline_duration_seconds",
            "Pipeline wall-clock time",
            ["model"],
            buckets=STAGE_BUCKETS,
            registry=self.registry,
        )
        self.llm_duration = Histogram(
            "isummarize_llm_duration_seconds",
            "LLM HTTP wall time for a task (sum of calls)",
            ["model"],
            buckets=STAGE_BUCKETS,
            registry=self.registry,
        )
        self.llm_call_duration = Histogram(
            "isummarize_llm_call_duration_seconds",
            "Single LLM HTTP call duration",
            ["model"],
            buckets=STAGE_BUCKETS,
            registry=self.registry,
        )
        self.queue_wait = Histogram(
            "isummarize_queue_wait_seconds",
            "Time from submit to running",
            ["model"],
            buckets=QUEUE_WAIT_BUCKETS,
            registry=self.registry,
        )
        self.prompt_tokens = Counter(
            "isummarize_prompt_tokens_total",
            "Prompt tokens reported by the provider",
            ["model"],
            registry=self.registry,
        )
        self.completion_tokens = Counter(
            "isummarize_completion_tokens_total",
            "Completion tokens reported by the provider",
            ["model"],
            registry=self.registry,
        )
        self.prompt_tokens_hist = Histogram(
            "isummarize_task_prompt_tokens",
            "Prompt tokens per finished task",
            ["model"],
            buckets=TOKEN_BUCKETS,
            registry=self.registry,
        )
        self.completion_tokens_hist = Histogram(
            "isummarize_task_completion_tokens",
            "Completion tokens per finished task",
            ["model"],
            buckets=TOKEN_BUCKETS,
            registry=self.registry,
        )
        self.restore_tasks = Counter(
            "isummarize_restore_tasks_total",
            "Unfinished tasks handled at process start",
            ["outcome"],
            registry=self.registry,
        )
        self.http_requests = Counter(
            "isummarize_http_requests_total",
            "HTTP requests",
            ["method", "path", "code"],
            registry=self.registry,
        )
        self.http_duration = Histogram(
            "isummarize_http_request_duration_seconds",
            "HTTP request duration",
            ["method", "path"],
            buckets=HTTP_BUCKETS,
            registry=self.registry,
        )

    def bind(self, *, settings: Settings, runner: TaskRunner, llm_status: LlmHealth) -> None:
        self.runtime.settings = settings
        self.runtime.runner = runner
        self.runtime.runner_started = True
        self.runtime.llm_status = llm_status
        if self._info is None:
            return
        self._info.info(
            {
                "version": read_version(),
                "model": settings.MODEL or "unconfigured",
            }
        )


def create_metrics(settings: Settings) -> Metrics:
    return Metrics(enabled=settings.METRICS_ENABLED)


def get_active() -> Metrics | None:
    return _active


def set_active(metrics: Metrics | None) -> None:
    global _active
    _active = metrics


def render() -> bytes:
    metrics = _active
    if metrics is None:
        return b""
    return generate_latest(metrics.registry)


def set_llm_status(status: LlmHealth) -> None:
    metrics = _active
    if metrics is None:
        return
    metrics.runtime.llm_status = status


def observe_queue_rejected() -> None:
    metrics = _active
    if metrics is None or not metrics.enabled:
        return
    metrics.queue_rejected.inc()


def observe_submitted(model: str) -> None:
    metrics = _active
    if metrics is None or not metrics.enabled:
        return
    metrics.tasks_submitted.labels(model=model or "unknown").inc()


def observe_restore(outcome: str) -> None:
    metrics = _active
    if metrics is None or not metrics.enabled:
        return
    metrics.restore_tasks.labels(outcome=outcome).inc()


def observe_http(method: str, path: str, status_code: int, duration_sec: float) -> None:
    metrics = _active
    if metrics is None or not metrics.enabled:
        return
    metrics.http_requests.labels(method=method, path=path, code=str(status_code)).inc()
    metrics.http_duration.labels(method=method, path=path).observe(duration_sec)


def observe_llm_call(model: str, duration_sec: float) -> None:
    metrics = _active
    if metrics is None or not metrics.enabled:
        return
    metrics.llm_call_duration.labels(model=model or "unknown").observe(duration_sec)


def observe_task_finished(
    *,
    model: str,
    status: str,
    error_code: str | None = None,
    llm_time_sec: float | None = None,
    total_time_sec: float | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    queue_wait: float | None = None,
) -> None:
    metrics = _active
    if metrics is None or not metrics.enabled:
        return
    label = model or "unknown"
    metrics.tasks_completed.labels(model=label, status=status).inc()
    if status == "error" and error_code:
        metrics.task_errors.labels(error_code=error_code, model=label).inc()
    if total_time_sec is not None:
        metrics.pipeline_duration.labels(model=label).observe(total_time_sec)
    if llm_time_sec is not None:
        metrics.llm_duration.labels(model=label).observe(llm_time_sec)
    if prompt_tokens is not None:
        metrics.prompt_tokens.labels(model=label).inc(prompt_tokens)
        metrics.prompt_tokens_hist.labels(model=label).observe(prompt_tokens)
    if completion_tokens is not None:
        metrics.completion_tokens.labels(model=label).inc(completion_tokens)
        metrics.completion_tokens_hist.labels(model=label).observe(completion_tokens)
    if queue_wait is not None:
        metrics.queue_wait.labels(model=label).observe(queue_wait)
