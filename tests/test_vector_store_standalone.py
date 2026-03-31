"""Verify vector_store has no hard coupling to src.infra and accepts injectable deps."""
import importlib


def test_vector_store_uses_stdlib_logging():
    source = importlib.util.find_spec("src.memory.vector_store").origin
    with open(source) as f:
        text = f.read()
    assert "from src.infra.logging_config" not in text
    assert "import logging" in text


def test_init_store_accepts_embed_fn():
    """init_store should accept an optional embed_fn parameter."""
    import inspect
    from src.memory.vector_store import init_store
    sig = inspect.signature(init_store)
    assert "embed_fn" in sig.parameters, "init_store must accept embed_fn parameter"
