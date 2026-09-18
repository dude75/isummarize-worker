from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from app.config import Settings, get_settings
from app.logging_setup import LOG_FILENAME, setup_logging
from app.metrics import CSV_FIELDS, CSV_HEADER, MetricEvent, write_metric

PROBE_MESSAGE = "log-flag-probe"


def _event(task_id: str = "t1") -> MetricEvent:
    return MetricEvent(
        timestamp="2026-08-20T10:00:00",
        task_id=task_id,
        model="stub-model",
        text_chars=10,
        skill_chars=4,
        chunk_count=1,
        prepare_time_sec=0.1,
        llm_time_sec=1.0,
        total_time_sec=1.2,
        prompt_tokens=11,
        completion_tokens=7,
        status="success",
    )


def test_header_once_and_append(tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERFORMANCE_LOG_ENABLED", "true")
    get_settings.cache_clear()
    log_path = tmp_path / "performance_log.csv"
    first = _event("t1")
    second = MetricEvent(
        timestamp="2026-08-20T10:01:00",
        task_id="t2",
        model="other",
        status="error",
    )
    write_metric(log_path, first)
    write_metric(log_path, second)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == CSV_HEADER
    assert lines[0].count(",") == len(CSV_FIELDS) - 1
    assert len(lines) == 3
    assert lines[1].startswith("2026-08-20T10:00:00,t1,stub-model,")
    assert "t2" in lines[2]
    assert lines[2].count(",") == len(CSV_FIELDS) - 1

    stdout = capsys.readouterr().out.strip().splitlines()
    assert len(stdout) == 2
    payload1 = json.loads(stdout[0])
    payload2 = json.loads(stdout[1])
    assert payload1["metric_event"] is True
    assert payload1["status"] == "success"
    assert payload1["task_id"] == "t1"
    assert payload2["metric_event"] is True
    assert payload2["status"] == "error"
    get_settings.cache_clear()


@pytest.mark.parametrize("raw", ["false", "0", "no", "FALSE", "No"])
def test_flag_env_false_aliases(raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_ENABLED", raw)
    monkeypatch.setenv("PERFORMANCE_LOG_ENABLED", raw)
    settings = Settings()
    assert settings.LOG_ENABLED is False
    assert settings.PERFORMANCE_LOG_ENABLED is False


@pytest.mark.parametrize(
    ("log_raw", "perf_raw", "log_on", "perf_on"),
    [
        ("true", "true", True, True),
        ("true", "false", True, False),
        ("false", "true", False, True),
        ("false", "false", False, False),
    ],
)
def test_four_flag_combinations(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
    log_raw: str,
    perf_raw: str,
    log_on: bool,
    perf_on: bool,
) -> None:
    log_dir = tmp_path / "logs"
    csv_path = log_dir / "performance_log.csv"
    app_log = log_dir / LOG_FILENAME
    monkeypatch.setenv("LOG_DIR", str(log_dir))
    monkeypatch.setenv("PERFORMANCE_LOG", str(csv_path))
    monkeypatch.setenv("LOG_ENABLED", log_raw)
    monkeypatch.setenv("PERFORMANCE_LOG_ENABLED", perf_raw)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.LOG_ENABLED is log_on
    assert settings.PERFORMANCE_LOG_ENABLED is perf_on

    setup_logging(settings)
    logging.getLogger("app.llm").warning(PROBE_MESSAGE)
    write_metric(csv_path, _event())

    captured = capsys.readouterr()
    stdout_lines = [line for line in captured.out.splitlines() if line.strip()]
    if perf_on:
        assert csv_path.is_file()
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == CSV_HEADER
        assert len(lines) == 2
        assert any('"metric_event": true' in line or '"metric_event":true' in line for line in stdout_lines)
        payload = json.loads(stdout_lines[-1])
        assert payload["metric_event"] is True
        assert payload["task_id"] == "t1"
    else:
        assert not csv_path.exists()
        assert all("metric_event" not in line for line in stdout_lines)

    if log_on:
        assert app_log.is_file()
        assert PROBE_MESSAGE in app_log.read_text(encoding="utf-8")
        assert PROBE_MESSAGE in captured.err
    else:
        assert not app_log.exists()
        assert PROBE_MESSAGE not in captured.err
        assert PROBE_MESSAGE not in captured.out
    get_settings.cache_clear()


def test_app_log_rotates_by_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_dir = tmp_path / "logs"
    monkeypatch.setenv("LOG_DIR", str(log_dir))
    monkeypatch.setenv("LOG_ENABLED", "true")
    monkeypatch.setenv("LOG_MAX_BYTES", "200")
    monkeypatch.setenv("LOG_BACKUP_COUNT", "2")
    get_settings.cache_clear()
    setup_logging(get_settings())
    logger = logging.getLogger("app")
    for i in range(40):
        logger.info("rotation-probe-%s-%s", i, "x" * 80)
    assert (log_dir / LOG_FILENAME).is_file()
    backups = {p.name for p in log_dir.glob("app.log.*")}
    assert "app.log.1" in backups
    get_settings.cache_clear()


def test_performance_off_does_not_grow_existing_csv(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "performance_log.csv"
    csv_path.write_text("pre-existing\n", encoding="utf-8")
    before = csv_path.read_bytes()
    monkeypatch.setenv("PERFORMANCE_LOG_ENABLED", "0")
    get_settings.cache_clear()
    write_metric(csv_path, _event())
    assert csv_path.read_bytes() == before
    assert "metric_event" not in capsys.readouterr().out
    get_settings.cache_clear()
