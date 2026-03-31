"""Verify llm_dispatcher has no hard coupling to src.infra at top level."""
import importlib


def test_dispatcher_uses_stdlib_logging():
    source = importlib.util.find_spec("src.core.llm_dispatcher").origin
    with open(source) as f:
        text = f.read()
    assert "from src.infra.logging_config" not in text
    assert "import logging" in text
