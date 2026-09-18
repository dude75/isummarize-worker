"""Разбор тела POST /summarize: JSON или form-поля text/skill."""

from __future__ import annotations

import json
from urllib.parse import parse_qs

from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.requests import Request

from app.schemas import SummarizeRequest


def _as_request_validation(exc: ValidationError) -> RequestValidationError:
    return RequestValidationError(exc.errors())


def _validate(data: object) -> SummarizeRequest:
    try:
        return SummarizeRequest.model_validate(data)
    except ValidationError as exc:
        raise _as_request_validation(exc) from exc


def parse_form_bytes(raw: bytes) -> dict[str, str]:
    parsed = parse_qs(raw.decode("utf-8"), keep_blank_values=True)
    return {key: (values[-1] if values else "") for key, values in parsed.items()}


async def parse_summarize_request(request: Request) -> SummarizeRequest:
    content_type = (request.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
    raw = await request.body()
    if content_type == "application/json" or (
        not content_type and raw.lstrip()[:1] == b"{"
    ):
        try:
            payload = json.loads(raw.decode("utf-8") or "null")
        except json.JSONDecodeError as exc:
            raise RequestValidationError(
                [
                    {
                        "type": "json_invalid",
                        "loc": ("body",),
                        "msg": f"JSON decode error: {exc.msg}",
                        "input": raw.decode("utf-8", errors="replace"),
                    }
                ]
            ) from exc
        return _validate(payload)
    form = parse_form_bytes(raw)
    return _validate(
        {
            "text": form.get("text", ""),
            "skill": form.get("skill", ""),
        }
    )
