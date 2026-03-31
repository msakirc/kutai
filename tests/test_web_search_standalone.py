"""Verify web_search.py has no hard coupling to src.infra or src.tools at top level."""
import importlib


def test_web_search_uses_stdlib_logging():
    source = importlib.util.find_spec("src.tools.web_search").origin
    with open(source) as f:
        text = f.read()
    assert "from src.infra.logging_config" not in text
    assert "import logging" in text


def test_web_search_no_toplevel_tools_import():
    """run_shell import must be lazy, not at module level."""
    source = importlib.util.find_spec("src.tools.web_search").origin
    with open(source) as f:
        lines = f.readlines()
    for i, line in enumerate(lines, 1):
        stripped = line.lstrip()
        if stripped != line:
            continue  # indented = inside function
        assert "from src.tools import" not in line, f"Top-level src.tools import at line {i}"
        assert "from src.tools." not in line, f"Top-level src.tools import at line {i}"
