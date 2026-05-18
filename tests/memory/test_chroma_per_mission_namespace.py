"""Z10 T3C — Chroma per-mission namespace."""
from __future__ import annotations

import importlib.util

import pytest

_HAS_CHROMA = importlib.util.find_spec("chromadb") is not None
_chroma_skip = pytest.mark.skipif(
    not _HAS_CHROMA, reason="chromadb not installed in this venv"
)


def test_collection_name_helper():
    from src.memory.vector_store import mission_chroma_collection_name
    assert mission_chroma_collection_name(47, "memory") == "mission_47__memory"
    assert mission_chroma_collection_name(0, "x") == "mission_0__x"
    assert mission_chroma_collection_name(None, "memory") == "global__memory"
    # Sanity: stable for int-string conversion.
    assert mission_chroma_collection_name(47, "semantic").endswith("__semantic")


@_chroma_skip
@pytest.mark.asyncio
async def test_two_missions_write_to_distinct_collections(tmp_path, monkeypatch):
    from src.memory import vector_store as vs

    # Point persist_dir at a tmp dir + reset state.
    monkeypatch.setattr(vs, "_DB_DIR", str(tmp_path / "chroma"), raising=False)
    monkeypatch.setattr(vs, "_initialized", False, raising=False)
    monkeypatch.setattr(vs, "_client", None, raising=False)
    monkeypatch.setattr(vs, "_collections", {}, raising=False)
    monkeypatch.setattr(vs, "_namespaced_collections", {}, raising=False)

    # Stub embed to avoid sentence-transformers download.
    async def _fake_embed(text, is_query=False):
        # Deterministic 8-dim vector for tests.
        return [float((ord(c) % 13) / 13.0) for c in (text + "xxxxxxxx")[:8]]

    monkeypatch.setattr(vs, "_get_embed_fn", lambda: _fake_embed)
    monkeypatch.setattr(vs, "_get_dimension_fn", lambda: (lambda: 8))

    ok = await vs.init_store(persist_dir=str(tmp_path / "chroma"))
    assert ok

    await vs.embed_and_store_for_mission(
        "hello from M47", {"k": "v"}, "memory", 47
    )
    await vs.embed_and_store_for_mission(
        "hello from M48", {"k": "v"}, "memory", 48
    )

    col47 = await vs._get_or_create_namespaced_collection("mission_47__memory")
    col48 = await vs._get_or_create_namespaced_collection("mission_48__memory")
    assert col47.count() == 1
    assert col48.count() == 1

    # Global slot is independent.
    await vs.embed_and_store_for_mission(
        "shared seed", {}, "memory", None
    )
    glb = await vs._get_or_create_namespaced_collection("global__memory")
    assert glb.count() == 1


@_chroma_skip
@pytest.mark.asyncio
async def test_purge_mission_chroma_collections(tmp_path, monkeypatch):
    from src.memory import vector_store as vs

    monkeypatch.setattr(vs, "_DB_DIR", str(tmp_path / "chroma2"), raising=False)
    monkeypatch.setattr(vs, "_initialized", False, raising=False)
    monkeypatch.setattr(vs, "_client", None, raising=False)
    monkeypatch.setattr(vs, "_collections", {}, raising=False)
    monkeypatch.setattr(vs, "_namespaced_collections", {}, raising=False)

    async def _fake_embed(text, is_query=False):
        return [0.1] * 8

    monkeypatch.setattr(vs, "_get_embed_fn", lambda: _fake_embed)
    monkeypatch.setattr(vs, "_get_dimension_fn", lambda: (lambda: 8))

    await vs.init_store(persist_dir=str(tmp_path / "chroma2"))
    await vs.embed_and_store_for_mission("a", {}, "memory", 99)
    await vs.embed_and_store_for_mission("b", {}, "semantic", 99)
    await vs.embed_and_store_for_mission("c", {}, "memory", 100)

    n = await vs.purge_mission_chroma_collections(99)
    assert n == 2  # 99's two collections gone

    # 100 still alive.
    names = await vs.list_mission_chroma_collections(100)
    assert "mission_100__memory" in names

    # idempotent
    again = await vs.purge_mission_chroma_collections(99)
    assert again == 0
