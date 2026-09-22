"""HTTP API задач. Uvicorn — один процесс."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.requests import ClientDisconnect, Request as StarletteRequest

from app import llm as llm_mod
from app.auth import api_token_is_valid, require_api_token
from app.config import get_settings
from app.llm import probe_llm, stub_chat_complete
from app.logging_setup import setup_logging
from app.prometheus_metrics import (
    CONTENT_TYPE,
    create_metrics,
    http_path_template,
    observe_http,
    observe_queue_rejected,
    render,
    set_active,
    set_llm_status,
)
from app.queueing import QueueFullError, TaskRunner, TaskRunningError
from app.schemas import (
    ErrorCode,
    ErrorDetail,
    HealthResponse,
    LlmHealth,
    WorkersHealth,
    PurgeResult,
    TaskListItem,
    TaskMeta,
    TaskResponse,
    TaskStatus,
    error_payload,
)
from app.summarize_body import parse_summarize_request
from app.tasks import TaskRecord
from app.version import read_version

_SWAGGER_STYLE = """
<style>
  .swagger-ui textarea.summarize-paste {
    min-height: 160px;
    width: 100%;
    font-family: monospace;
    line-height: 1.4;
  }
</style>
"""

_SWAGGER_PATCH = r"""
(function () {
  const fieldName = (input) => {
    const row = input.closest("tr");
    const raw = (
      input.getAttribute("name") ||
      row?.getAttribute("data-param-name") ||
      row?.querySelector(".parameter__name")?.textContent ||
      ""
    )
      .replace(/\*/g, "")
      .trim()
      .split(/\s+/)[0];
    return raw === "text" || raw === "skill" ? raw : "";
  };
  const toTextarea = (input) => {
    if (!input || input.dataset.asTextarea === "1") return;
    const ta = document.createElement("textarea");
    ta.className = (input.className || "") + " summarize-paste";
    ta.rows = 10;
    ta.value = input.value === "string" ? "" : input.value;
    const name = fieldName(input);
    if (name) ta.setAttribute("name", name);
    ["placeholder", "id"].forEach((attr) => {
      const value = input.getAttribute(attr);
      if (value) ta.setAttribute(attr, value);
    });
    input.dataset.asTextarea = "1";
    input.setAttribute("aria-hidden", "true");
    input.style.position = "absolute";
    input.style.opacity = "0";
    input.style.height = "0";
    input.style.width = "0";
    input.style.pointerEvents = "none";
    input.parentNode.insertBefore(ta, input);
  };
  const scan = () => {
    document.querySelectorAll(".opblock").forEach((block) => {
      const path = block.querySelector(".opblock-summary-path");
      if (!path || path.textContent.trim() !== "/summarize") return;
      [...block.querySelectorAll("input[type=text]")].forEach(toTextarea);
      const unnamed = [...block.querySelectorAll("textarea.summarize-paste")].filter((el) => !el.name);
      if (unnamed.length === 2 && !block.querySelector("textarea.summarize-paste[name]")) {
        unnamed[0].name = "text";
        unnamed[1].name = "skill";
      }
    });
  };
  new MutationObserver(scan).observe(document.documentElement, { childList: true, subtree: true });

  const summarizeFields = () => {
    const block = [...document.querySelectorAll(".opblock")].find((el) => {
      const path = el.querySelector(".opblock-summary-path");
      return path && path.textContent.trim() === "/summarize";
    });
    if (!block) return null;
    const text = block.querySelector("textarea.summarize-paste[name='text']");
    const skill = block.querySelector("textarea.summarize-paste[name='skill']");
    const all = [...block.querySelectorAll("textarea.summarize-paste")];
    const textEl = text || all[0];
    const skillEl = skill || all[1];
    if (!textEl || !skillEl) return null;
    return { text: textEl.value, skill: skillEl.value };
  };

  const withJsonHeaders = (headers) => {
    const next = {};
    if (headers && typeof headers.forEach === "function") {
      headers.forEach((value, key) => { next[key] = value; });
    } else if (Array.isArray(headers)) {
      headers.forEach((pair) => { next[pair[0]] = pair[1]; });
    } else if (headers && typeof headers === "object") {
      Object.assign(next, headers);
    }
    Object.keys(next).forEach((key) => {
      if (key.toLowerCase() === "content-type") delete next[key];
    });
    next["Content-Type"] = "application/json";
    return next;
  };

  const origBundle = window.SwaggerUIBundle;
  if (origBundle) {
    const wrapped = function (config) {
      const prev = config.requestInterceptor;
      config.requestInterceptor = (req) => {
        const fields = summarizeFields();
        if (fields && String(req.method || "GET").toUpperCase() === "POST") {
          const url = String(req.url || "").replace(/\/+$/, "");
          if (url.endsWith("/summarize")) {
            req.body = JSON.stringify(fields);
            req.headers = withJsonHeaders(req.headers);
          }
        }
        return prev ? prev(req) : req;
      };
      return origBundle.apply(this, arguments);
    };
    Object.assign(wrapped, origBundle);
    window.SwaggerUIBundle = wrapped;
  }

  const origFetch = window.fetch.bind(window);
  window.fetch = function (input, init) {
    init = init ? Object.assign({}, init) : {};
    const url = typeof input === "string" ? input : (input && input.url) || "";
    const method = String(init.method || (input && input.method) || "GET").toUpperCase();
    if (method === "POST" && String(url).replace(/\/+$/, "").endsWith("/summarize")) {
      const fields = summarizeFields();
      if (fields) {
        init.body = JSON.stringify(fields);
        init.headers = withJsonHeaders(init.headers);
      }
    }
    return origFetch(input, init);
  };
})();
"""

_SUMMARIZE_FORM_SCHEMA = {
    "type": "object",
    "required": ["text", "skill"],
    "additionalProperties": False,
    "properties": {
        "text": {"type": "string", "minLength": 1, "title": "text"},
        "skill": {"type": "string", "minLength": 1, "title": "skill"},
    },
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings)
    logging.getLogger("app").info("service start")
    metrics = create_metrics(settings)
    set_active(metrics)
    if os.environ.get("ISUMMARIZE_STUBS", "").lower() in {"1", "true", "yes"}:
        llm_mod.complete_override = stub_chat_complete
        llm_mod.status_override = (
            LlmHealth.ready if settings.llm_configured() else LlmHealth.unconfigured
        )
    llm_status = await probe_llm(settings)
    runner = TaskRunner(settings)
    await runner.start()
    metrics.bind(settings=settings, runner=runner, llm_status=llm_status)
    app.state.runner = runner
    app.state.llm_status = llm_status
    try:
        yield
    finally:
        await runner.stop()
        set_active(None)
        llm_mod.complete_override = None
        llm_mod.status_override = None


app = FastAPI(
    title="isummarize-worker",
    version=read_version(),
    lifespan=lifespan,
    docs_url=None,
)


def _api_error(status_code: int, code: ErrorCode) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=error_payload(code),
        headers={"Connection": "close"},
    )


def _http_error(status_code: int, code: ErrorCode) -> HTTPException:
    return HTTPException(status_code=status_code, detail=error_payload(code))


def _summarize_path(request: Request) -> bool:
    path = request.scope.get("path") or request.url.path
    return str(path).rstrip("/") == "/summarize"


async def _read_body_capped(request: Request, limit: int) -> bytes | None:
    """Читает тело чанками. None — поток превысил limit, не дожидаясь EOF.

    Кладёт результат в ``request._body``: BaseHTTPMiddleware после ``stream()``
    иначе прокидывает downstream пустое тело.
    """
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        if not chunk:
            continue
        next_size = size + len(chunk)
        if next_size > limit:
            return None
        chunks.append(chunk)
        size = next_size
    body = b"".join(chunks)
    request._body = body
    return body


@app.middleware("http")
async def gate_summarize_payload(request: Request, call_next):
    """Auth, LLM ready, размер тела и queue_full до разбора JSON на POST /summarize."""
    if request.method != "POST" or not _summarize_path(request):
        return await call_next(request)
    if not api_token_is_valid(request.headers.get("authorization")):
        return _api_error(status.HTTP_401_UNAUTHORIZED, ErrorCode.unauthorized)
    settings = get_settings()
    llm_status = await probe_llm(settings)
    if llm_status is not LlmHealth.ready:
        code = (
            ErrorCode.llm_unconfigured
            if llm_status is LlmHealth.unconfigured
            else ErrorCode.llm_unavailable
        )
        return _api_error(status.HTTP_503_SERVICE_UNAVAILABLE, code)
    raw_cl = request.headers.get("content-length")
    if raw_cl is not None:
        try:
            size = int(raw_cl)
        except ValueError:
            size = -1
        if size < 0 or size > settings.MAX_PAYLOAD_BYTES:
            return _api_error(status.HTTP_413_CONTENT_TOO_LARGE, ErrorCode.payload_too_large)
    runner = getattr(request.app.state, "runner", None)
    if runner is not None and runner.store.count_queued() >= settings.WORKER_QUEUE_SIZE:
        observe_queue_rejected()
        return _api_error(status.HTTP_503_SERVICE_UNAVAILABLE, ErrorCode.queue_full)
    body = await _read_body_capped(request, settings.MAX_PAYLOAD_BYTES)
    if body is None:
        return _api_error(status.HTTP_413_CONTENT_TOO_LARGE, ErrorCode.payload_too_large)

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": body, "more_body": False}

    request = StarletteRequest(request.scope, receive)
    return await call_next(request)


@app.middleware("http")
async def prometheus_http_middleware(request: Request, call_next):
    started = time.perf_counter()
    path = http_path_template(request)
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except ClientDisconnect:
        status_code = 499
        return Response(status_code=status_code)
    finally:
        observe_http(request.method, path, status_code, time.perf_counter() - started)


def get_runner() -> TaskRunner:
    return app.state.runner


@app.exception_handler(HTTPException)
async def http_exception_handler(_request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})


def _record_to_response(record: TaskRecord) -> TaskResponse:
    error = ErrorDetail.model_validate(record.error) if record.error is not None else None
    return TaskResponse(
        status=record.status,
        meta=TaskMeta(
            timestamp=record.timestamp,
            task_id=record.task_id,
            model=record.model,
            text_chars=record.text_chars,
            skill_chars=record.skill_chars,
            chunk_count=record.chunk_count,
            prepare_time_sec=record.prepare_time_sec,
            llm_time_sec=record.llm_time_sec,
            total_time_sec=record.total_time_sec,
            prompt_tokens=record.prompt_tokens,
            completion_tokens=record.completion_tokens,
        ),
        summary=record.summary,
        error=error,
    )


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/docs", include_in_schema=False)
def swagger_docs() -> HTMLResponse:
    page = get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="isummarize-worker",
        swagger_ui_parameters={"persistAuthorization": True},
    )
    html = bytes(page.body).decode("utf-8")
    marker = "const ui = SwaggerUIBundle({"
    if marker not in html:
        raise RuntimeError("swagger html: SwaggerUIBundle init not found")
    html = html.replace(marker, _SWAGGER_PATCH + marker, 1)
    html = html.replace("</head>", _SWAGGER_STYLE + "</head>", 1)
    return HTMLResponse(html)


async def _health_body(runner: TaskRunner | None = None) -> HealthResponse:
    settings = get_settings()
    llm_status = await probe_llm(settings)
    set_llm_status(llm_status)
    workers = (
        runner.worker_snapshot()
        if runner is not None
        else WorkersHealth(max=settings.WORKERS, active=0, available=settings.WORKERS)
    )
    return HealthResponse(
        status="ok",
        version=read_version(),
        model=settings.MODEL,
        llm=llm_status,
        workers=workers,
    )


@app.get("/health", response_model=HealthResponse)
async def health(runner: TaskRunner = Depends(get_runner)) -> HealthResponse:
    return await _health_body(runner)


@app.get(
    "/ready",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse}},
)
async def ready(runner: TaskRunner = Depends(get_runner)) -> HealthResponse | JSONResponse:
    body = await _health_body(runner)
    if body.llm is LlmHealth.ready:
        return body
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=body.model_dump(mode="json"),
    )


@app.get("/metrics")
def metrics(_: str = Depends(require_api_token)) -> Response:
    return Response(content=render(), media_type=CONTENT_TYPE)


@app.post(
    "/summarize",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=TaskResponse,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/x-www-form-urlencoded": {"schema": _SUMMARIZE_FORM_SCHEMA},
                "application/json": {"schema": _SUMMARIZE_FORM_SCHEMA},
            },
        }
    },
)
async def summarize(
    request: Request,
    _: str = Depends(require_api_token),
    runner: TaskRunner = Depends(get_runner),
) -> TaskResponse:
    payload = await parse_summarize_request(request)
    try:
        record = await runner.submit(payload.text, payload.skill)
    except QueueFullError as exc:
        observe_queue_rejected()
        raise _http_error(status.HTTP_503_SERVICE_UNAVAILABLE, exc.code) from None
    return _record_to_response(record)


@app.delete("/tasks", response_model=PurgeResult)
async def purge_tasks(
    _: str = Depends(require_api_token),
    runner: TaskRunner = Depends(get_runner),
) -> PurgeResult:
    return await runner.purge()


@app.get("/tasks", response_model=list[TaskListItem])
def list_tasks(
    status_filter: TaskStatus | None = Query(default=None, alias="status"),
    _: str = Depends(require_api_token),
    runner: TaskRunner = Depends(get_runner),
) -> list[TaskListItem]:
    records = runner.store.list_tasks(status_filter)
    return [
        TaskListItem(
            task_id=item.task_id,
            status=item.status,
            timestamp=item.timestamp,
            model=item.model,
            text_chars=item.text_chars,
            skill_chars=item.skill_chars,
        )
        for item in records
    ]


@app.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task(
    task_id: str,
    _: str = Depends(require_api_token),
    runner: TaskRunner = Depends(get_runner),
) -> TaskResponse:
    record = runner.store.get(task_id)
    if record is None:
        raise _http_error(status.HTTP_404_NOT_FOUND, ErrorCode.not_found)
    return _record_to_response(record)


@app.delete("/tasks/{task_id}")
async def delete_task(
    task_id: str,
    _: str = Depends(require_api_token),
    runner: TaskRunner = Depends(get_runner),
) -> dict[str, str]:
    try:
        await runner.delete(task_id)
    except KeyError:
        raise _http_error(status.HTTP_404_NOT_FOUND, ErrorCode.not_found) from None
    except TaskRunningError as exc:
        raise _http_error(status.HTTP_409_CONFLICT, exc.code) from None
    return {"status": "ok"}
