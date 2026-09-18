"""Payload на диске: `{DATA_DIR}/tmp/<task_id>/`."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

TMP_PREFIX = "tmp_"
TMP_SUBDIR = "tmp"
TEXT_NAME = "text.txt"
SKILL_NAME = "skill.md"
INPUT_NAME = "input.json"


def _resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    from app.config import get_settings

    return Path(get_settings().DATA_DIR)


def tmp_root(data_dir: str | Path | None = None) -> Path:
    return _resolve_data_dir(data_dir) / TMP_SUBDIR


def tmp_dir(task_id: str, data_dir: str | Path | None = None) -> Path:
    return tmp_root(data_dir) / task_id


def create_tmp(task_id: str, data_dir: str | Path | None = None) -> Path:
    path = tmp_dir(task_id, data_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_tmp(task_id: str, data_dir: str | Path | None = None) -> None:
    path = tmp_dir(task_id, data_dir)
    shutil.rmtree(path, ignore_errors=True)


def cleanup_tmp_except(keep_ids: set[str], data_dir: str | Path | None = None) -> int:
    """Снимает каталоги `{DATA_DIR}/tmp/<id>/`, кроме keep_ids. Возвращает число удалённых."""
    root = tmp_root(data_dir)
    if not root.is_dir():
        return 0
    removed = 0
    for path in root.iterdir():
        if path.is_dir() and path.name not in keep_ids:
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    return removed


def cleanup_legacy_cwd_tmp(root: str | Path | None = None) -> int:
    """Удаляет устаревшие `tmp_*` в CWD. Возвращает число каталогов."""
    base = Path.cwd() if root is None else Path(root)
    removed = 0
    for path in base.glob(f"{TMP_PREFIX}*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    return removed


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def write_payload(
    task_id: str,
    data_dir: str | Path,
    *,
    text: str,
    skill: str,
    model: str,
) -> Path:
    dest = create_tmp(task_id, data_dir)
    _atomic_write(dest / TEXT_NAME, text)
    _atomic_write(dest / SKILL_NAME, skill)
    _atomic_write(
        dest / INPUT_NAME,
        json.dumps(
            {
                "model": model,
                "text_chars": len(text),
                "skill_chars": len(skill),
            },
            ensure_ascii=False,
        ),
    )
    return dest


def payload_exists(task_id: str, data_dir: str | Path | None = None) -> bool:
    dest = tmp_dir(task_id, data_dir)
    return (dest / TEXT_NAME).is_file() and (dest / SKILL_NAME).is_file()


def read_payload(task_id: str, data_dir: str | Path | None = None) -> tuple[str, str]:
    dest = tmp_dir(task_id, data_dir)
    text_path = dest / TEXT_NAME
    skill_path = dest / SKILL_NAME
    if not text_path.is_file() or not skill_path.is_file():
        raise FileNotFoundError(str(dest))
    return text_path.read_text(encoding="utf-8"), skill_path.read_text(encoding="utf-8")


def write_chunk_summary(task_id: str, data_dir: str | Path, index: int, summary: str) -> None:
    dest = create_tmp(task_id, data_dir)
    _atomic_write(dest / f"chunk_{index:03d}.md", summary)


def write_reduce_summary(task_id: str, data_dir: str | Path, summary: str) -> None:
    dest = create_tmp(task_id, data_dir)
    _atomic_write(dest / "reduce.md", summary)
