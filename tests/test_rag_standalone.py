"""Verify rag.py has no hard coupling to src.infra or src.memory at top level."""
import importlib


def test_rag_uses_stdlib_logging():
    source = importlib.util.find_spec("src.memory.rag").origin
    with open(source) as f:
        text = f.read()
    assert "from src.infra.logging_config" not in text
    assert "import logging" in text


def test_rag_no_toplevel_memory_imports():
    """vector_store and embeddings imports must be lazy."""
    source = importlib.util.find_spec("src.memory.rag").origin
    with open(source) as f:
        lines = f.readlines()
    for i, line in enumerate(lines, 1):
        stripped = line.lstrip()
        if stripped != line:
            continue  # indented
        assert "from src.memory.vector_store" not in line, f"Top-level vector_store import at line {i}"
        assert "from src.memory.embeddings" not in line, f"Top-level embeddings import at line {i}"
