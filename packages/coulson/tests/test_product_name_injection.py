import pytest


def test_product_name_block_present():
    from coulson.context import _product_name_block
    block = _product_name_block("FlowState")
    assert block is not None
    assert "FlowState" in block
    assert "EXACTLY" in block


def test_product_name_block_empty_returns_none():
    from coulson.context import _product_name_block
    assert _product_name_block("   ") is None
    assert _product_name_block(None) is None


@pytest.mark.asyncio
async def test_load_product_name_from_store(monkeypatch):
    from coulson import context as ctx

    # The engine stores the model result as a JSON STRING (hooks.py:1641).
    class _Store:
        async def retrieve(self, mid, name):
            assert name == "product_name"
            return '{"product_name": "FlowState"}'

    monkeypatch.setattr(
        "src.workflows.engine.hooks.get_artifact_store", lambda: _Store(),
    )
    assert await ctx._load_product_name(42) == "FlowState"


@pytest.mark.asyncio
async def test_load_product_name_missing_returns_none(monkeypatch):
    from coulson import context as ctx

    class _Store:
        async def retrieve(self, mid, name):
            return None

    monkeypatch.setattr(
        "src.workflows.engine.hooks.get_artifact_store", lambda: _Store(),
    )
    assert await ctx._load_product_name(42) is None
    assert await ctx._load_product_name(None) is None
