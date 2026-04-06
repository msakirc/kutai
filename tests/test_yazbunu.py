"""Tests for yazbunu logging library."""
import json
import logging
import sys
import os

# yazbunu is a sibling directory at repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "yazbunu"))

from yazbunu.formatter import YazFormatter


def test_formatter_required_fields():
    """Formatter output contains ts, level, src, msg."""
    fmt = YazFormatter()
    record = logging.LogRecord(
        name="kutai.core.orchestrator",
        level=logging.INFO,
        pathname="orchestrator.py",
        lineno=42,
        msg="task dispatched",
        args=(),
        exc_info=None,
    )
    line = fmt.format(record)
    doc = json.loads(line)
    assert "ts" in doc
    assert doc["level"] == "INFO"
    assert doc["src"] == "kutai.core.orchestrator"
    assert doc["msg"] == "task dispatched"
    # INFO should NOT have fn/ln
    assert "fn" not in doc
    assert "ln" not in doc


def test_formatter_warning_includes_fn_ln():
    """WARNING+ records include fn and ln fields."""
    fmt = YazFormatter()
    record = logging.LogRecord(
        name="kutai.agents.base",
        level=logging.WARNING,
        pathname="base.py",
        lineno=284,
        msg="tool exec failed",
        args=(),
        exc_info=None,
    )
    record.funcName = "_run_tool"
    line = fmt.format(record)
    doc = json.loads(line)
    assert doc["fn"] == "_run_tool"
    assert doc["ln"] == 284


def test_formatter_context_fields():
    """Extra context fields (task, mission, agent, model) appear in output."""
    fmt = YazFormatter()
    record = logging.LogRecord(
        name="kutai.core.orchestrator",
        level=logging.INFO,
        pathname="orchestrator.py",
        lineno=42,
        msg="task dispatched",
        args=(),
        exc_info=None,
    )
    record.task = "42"
    record.mission = "m-7"
    record.agent = "coder"
    record.model = "qwen-32b"
    line = fmt.format(record)
    doc = json.loads(line)
    assert doc["task"] == "42"
    assert doc["mission"] == "m-7"
    assert doc["agent"] == "coder"
    assert doc["model"] == "qwen-32b"


def test_formatter_exception():
    """Exception info is captured in exc field."""
    fmt = YazFormatter()
    try:
        raise ValueError("test error")
    except ValueError:
        import sys
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="kutai.core.orchestrator",
        level=logging.ERROR,
        pathname="orchestrator.py",
        lineno=42,
        msg="something failed",
        args=(),
        exc_info=exc_info,
    )
    line = fmt.format(record)
    doc = json.loads(line)
    assert "exc" in doc
    assert "ValueError" in doc["exc"]


import tempfile
from pathlib import Path
from yazbunu import get_logger, init_logging


def test_get_logger_info(tmp_path):
    """get_logger returns a logger that writes structured JSONL."""
    init_logging(log_dir=str(tmp_path), project="testproj", console=False)
    logger = get_logger("core.thing")
    logger.info("hello", task="1")

    log_file = tmp_path / "testproj.jsonl"
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    doc = json.loads(lines[-1])
    assert doc["src"] == "testproj.core.thing"
    assert doc["msg"] == "hello"
    assert doc["task"] == "1"


def test_get_logger_bind(tmp_path):
    """Bound loggers carry context across calls."""
    init_logging(log_dir=str(tmp_path), project="testproj2", console=False)
    logger = get_logger("agents.base").bind(task="99", mission="m-3")
    logger.info("step done")

    log_file = tmp_path / "testproj2.jsonl"
    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    doc = json.loads(lines[-1])
    assert doc["task"] == "99"
    assert doc["mission"] == "m-3"


def test_init_logging_rotation(tmp_path):
    """Rotating file handler is created with correct params."""
    init_logging(log_dir=str(tmp_path), project="rottest", console=False,
                 max_bytes=1000, backup_count=2)
    logger = get_logger("x")
    # Write enough to trigger rotation
    for i in range(200):
        logger.info(f"line {i}" + "x" * 100)

    files = list(tmp_path.glob("rottest.jsonl*"))
    assert len(files) >= 2  # at least one backup


def test_kutai_shim_reexports():
    """src.infra.logging_config re-exports yazbunu's public API."""
    from src.infra.logging_config import get_logger, init_logging
    # Verify they are the yazbunu versions
    import yazbunu
    assert get_logger is yazbunu.get_logger
    assert init_logging is yazbunu.init_logging
