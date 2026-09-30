from __future__ import annotations

import json

import pytest

from app.chunking import plan_chunks, user_chunk, user_reduce, user_single


def test_user_templates_neutral_wording() -> None:
    single = user_single("payload")
    chunk = user_chunk("part", 1, 3)
    reduce_prompt = user_reduce(["a", "b"])
    assert "<transcript>" in single and "payload" in single
    assert "часть 1 из 3" in chunk.lower() or "Часть 1 из 3" in chunk
    assert "промежуточные" in reduce_prompt.lower()
    assert "саммари" not in single.lower()
    assert "саммари" not in chunk.lower()
    assert "саммари" not in reduce_prompt.lower()


def test_plan_chunks_invalid_json_uses_text_path(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.chunking as chunking

    monkeypatch.setattr(chunking, "MAX_PROMPT_CHARS", 200)
    text = "{not-json\n\n" + ("word " * 80)
    planned = plan_chunks("s", text)
    assert planned is not None
    assert len(planned) > 1


def test_plan_chunks_json_transcript_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.chunking as chunking

    monkeypatch.setattr(chunking, "MAX_PROMPT_CHARS", 800)
    segments = [{"speaker": "a", "text": "x" * 40} for _ in range(20)]
    payload = json.dumps({"type": "lecture", "transcript": segments}, ensure_ascii=False)
    planned = plan_chunks("skill", payload)
    assert planned is not None
    assert len(planned) > 1
    for piece in planned:
        parsed = json.loads(piece)
        assert isinstance(parsed["transcript"], list)
        assert parsed["type"] == "lecture"
        assert len(parsed["transcript"]) >= 1


def test_plan_chunks_json_no_list_fallback_text(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.chunking as chunking

    monkeypatch.setattr(chunking, "MAX_PROMPT_CHARS", 200)
    payload = json.dumps({"ok": True, "note": "hello " * 50})
    planned = plan_chunks("s", payload)
    assert planned is not None
    assert len(planned) > 1
    with pytest.raises(json.JSONDecodeError):
        json.loads(planned[0])
