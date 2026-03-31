"""Verify embeddings module has no hard coupling to src.infra."""
import importlib


def test_embeddings_uses_stdlib_logging():
    """embeddings.py must use stdlib logging, not src.infra.logging_config."""
    source = importlib.util.find_spec("src.memory.embeddings").origin
    with open(source) as f:
        text = f.read()
    assert "from src.infra.logging_config" not in text
    assert "import logging" in text
