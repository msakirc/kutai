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
