"""Verify llm_dispatcher uses _ContextLogger for kwargs-style logging."""
import importlib


def test_dispatcher_uses_context_logger():
    """Dispatcher must use _ContextLogger (get_logger) to support kwargs-style logging."""
    source = importlib.util.find_spec("src.core.llm_dispatcher").origin
    with open(source) as f:
        text = f.read()
    assert "from src.infra.logging_config import get_logger" in text
