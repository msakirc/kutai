"""Engine→app hook inversion (Phase B §5a). The engine calls injected
``dabidabi.hooks`` callables instead of importing ``src.*``; unset hooks
degrade to the same best-effort no-op the old try/except-import produced."""
import pytest
import dabidabi
from dabidabi import hooks


@pytest.fixture(autouse=True)
def _clean_hooks():
    hooks.reset()
    yield
    hooks.reset()


def test_register_rejects_unknown_name():
    with pytest.raises(KeyError):
        hooks.register(not_a_real_hook=lambda: None)


def test_register_and_reset():
    async def _fn(**kw):
        return None
    hooks.register(embed_and_store=_fn)
    assert hooks.embed_and_store is _fn
    hooks.reset()
    assert hooks.embed_and_store is None


@pytest.mark.asyncio
async def test_store_memory_invokes_embed_hook_when_set(tmp_path):
    dabidabi.configure(str(tmp_path / "sm.db"))
    await dabidabi.init_db()
    seen = {}

    async def _embed(**kw):
        seen.update(kw)

    hooks.register(embed_and_store=_embed)
    await dabidabi.store_memory("k1", "v1", category="note", mission_id=7)
    assert seen.get("text") == "k1: v1"
    assert seen["metadata"]["key"] == "k1"
    await dabidabi.close_db()


@pytest.mark.asyncio
async def test_store_memory_noop_when_embed_hook_unset(tmp_path):
    # Unset hook → memory row still written, embedding silently skipped.
    dabidabi.configure(str(tmp_path / "sm2.db"))
    await dabidabi.init_db()
    await dabidabi.store_memory("k2", "v2")  # must not raise
    rows = await dabidabi.recall_memory()
    assert any(r["key"] == "k2" for r in rows)
    await dabidabi.close_db()


@pytest.mark.asyncio
async def test_semantic_recall_uses_vector_query_hook(tmp_path):
    dabidabi.configure(str(tmp_path / "sr.db"))
    await dabidabi.init_db()

    async def _vquery(**kw):
        return [{"metadata": {"key": "kk", "category": "c", "mission_id": "7"},
                 "text": "vv", "distance": 0.1}]

    hooks.register(vector_query=_vquery)
    out = await dabidabi.semantic_recall("anything")
    assert out and out[0]["key"] == "kk" and out[0]["value"] == "vv"
    await dabidabi.close_db()


@pytest.mark.asyncio
async def test_semantic_recall_empty_when_hook_unset(tmp_path):
    dabidabi.configure(str(tmp_path / "sr2.db"))
    await dabidabi.init_db()
    assert await dabidabi.semantic_recall("anything") == []
    await dabidabi.close_db()


@pytest.mark.asyncio
async def test_add_mission_invokes_container_hook(tmp_path):
    dabidabi.configure(str(tmp_path / "am.db"))
    await dabidabi.init_db()
    got = []

    async def _ensure(mission_id):
        got.append(mission_id)

    hooks.register(ensure_mission_container=_ensure)
    mid = await dabidabi.add_mission("t", "d")
    assert got == [mid]
    await dabidabi.close_db()
