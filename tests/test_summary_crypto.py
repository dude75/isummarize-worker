from __future__ import annotations

from pathlib import Path

from app.tasks import TaskStore


def _seed_queued(store: TaskStore, task_id: str, tmp_path: Path) -> None:
    store.create(
        task_id,
        model="m",
        text_chars=10,
        skill_chars=4,
        payload_dir=str(tmp_path / "tmp" / task_id),
    )


def _mark_done(store: TaskStore, task_id: str, summary: str) -> None:
    store.mark_success(
        task_id,
        summary=summary,
        chunk_count=1,
        prepare_time_sec=0.1,
        llm_time_sec=0.1,
        total_time_sec=0.2,
        prompt_tokens=1,
        completion_tokens=1,
    )


def test_summary_encrypted_at_rest(tmp_path: Path) -> None:
    store = TaskStore(str(tmp_path / "tasks.db"))
    try:
        _seed_queued(store, "enc-1", tmp_path)
        _mark_done(store, "enc-1", "секретный текст")
        with store._lock:
            raw = store._conn.execute(
                "SELECT summary FROM tasks WHERE task_id = ?",
                ("enc-1",),
            ).fetchone()["summary"]
        assert raw
        assert raw.startswith("gAAAAA")
        assert "секретный текст" not in raw
        rec = store.get("enc-1")
        assert rec is not None
        assert rec.summary == "секретный текст"
    finally:
        store.close()


def test_summary_legacy_plaintext_still_reads(tmp_path: Path) -> None:
    store = TaskStore(str(tmp_path / "tasks.db"))
    try:
        _seed_queued(store, "legacy-plain", tmp_path)
        with store._lock:
            store._conn.execute(
                "UPDATE tasks SET summary = ? WHERE task_id = ?",
                ("старый plaintext", "legacy-plain"),
            )
            store._conn.commit()
        rec = store.get("legacy-plain")
        assert rec is not None
        assert rec.summary == "старый plaintext"
    finally:
        store.close()
