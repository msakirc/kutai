"""Verify shell.py has no hard coupling to src.infra."""
import importlib


def test_shell_uses_stdlib_logging():
    source = importlib.util.find_spec("src.tools.shell").origin
    with open(source, encoding="utf-8") as f:
        text = f.read()
    assert "from src.infra.logging_config" not in text
    assert "import logging" in text
