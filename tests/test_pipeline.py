from __future__ import annotations

import pytest

from app.chunking import plan_chunks, split_text, user_single
from app.pipeline import normalize_output
from app.schemas import ErrorCode
from app.pipeline import TaskFailed


def test_normalize_strips_fence_when_skill_does_not_ask() -> None:
    raw = "```markdown\n# Title\nbody\n```"
    assert normalize_output(raw, "Summarize the meeting") == "# Title\nbody"


def test_normalize_keeps_fence_when_skill_asks() -> None:
    raw = "```json\n{\"a\": 1}\n```"
    skill = "Return a markdown code fence with JSON"
    assert normalize_output(raw, skill) == raw.strip()


def test_normalize_json_as_text() -> None:
    raw = '{"ok": true, "items": [1]}'
    assert normalize_output(raw, "Return JSON") == raw


def test_normalize_empty_raises() -> None:
    with pytest.raises(TaskFailed) as exc:
        normalize_output("   ", "skill")
    assert exc.value.code is ErrorCode.llm_bad_response


def test_split_text_paragraphs_and_overlap() -> None:
    text = "aaa\n\nbbb\n\nccc\n\nddd"
    chunks = split_text(text, 10)
    assert len(chunks) >= 2
    assert "".join(part.replace("\n\n", "") for part in chunks).find("aaa") >= 0 or any(
        "aaa" in chunk for chunk in chunks
    )


def test_plan_chunks_single_when_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.chunking as chunking

    monkeypatch.setattr(chunking, "MAX_PROMPT_CHARS", 10_000)
    assert plan_chunks("skill", "short text") is None


def test_plan_chunks_splits_when_over_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.chunking as chunking

    monkeypatch.setattr(chunking, "MAX_PROMPT_CHARS", 200)
    skill = "short"
    text = ("word " * 80) + "\n\n" + ("other " * 80)
    planned = plan_chunks(skill, text)
    assert planned is not None
    assert len(planned) > 1


def test_plan_chunks_too_long_skill(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.chunking as chunking

    monkeypatch.setattr(chunking, "MAX_PROMPT_CHARS", 50)
    planned = plan_chunks("S" * 80, "text")
    assert planned == []


def test_user_single_contains_transcript() -> None:
    user = user_single("hello world")
    assert "hello world" in user
    assert "<transcript>" in user
