"""Verify free_apis.py has no hard coupling to src.infra at import time."""
import importlib


def test_free_apis_uses_stdlib_logging():
    source = importlib.util.find_spec("src.tools.free_apis").origin
    with open(source) as f:
        text = f.read()
    assert "from src.infra.logging_config" not in text
    assert "import logging" in text


def test_free_apis_no_toplevel_db_import():
    """DB imports must be lazy (inside functions), not at module level."""
    source = importlib.util.find_spec("src.tools.free_apis").origin
    with open(source) as f:
        lines = f.readlines()
    for i, line in enumerate(lines, 1):
        stripped = line.lstrip()
        if stripped != line:
            continue  # indented = inside function, OK
        assert "from src.infra.db" not in line, f"Top-level DB import at line {i}"
