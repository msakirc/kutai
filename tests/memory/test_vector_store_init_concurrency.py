"""TDD: init_store must serialize concurrent callers.

Regression guard for the 2026-07-25 cold-boot fix: the vector store is now
warmed by a background task, and a RAG query can race that warmup. Both call
init_store(); without an init lock both enter the body and each constructs a
ChromaDB PersistentClient against the same on-disk store (double-open — the
Windows file-lock / cache-race the module header warns about).
"""
import asyncio


class _FakeCol:
    def __init__(self):
        self.metadata = {}

    def count(self):
        return 0


class _FakeClient:
    def get_or_create_collection(self, name, embedding_function=None, metadata=None):
        return _FakeCol()


async def test_init_store_serializes_concurrent_callers(tmp_path, monkeypatch):
    import chromadb
    import src.memory.vector_store as vs

    # Clean module state; monkeypatch auto-reverts after the test.
    monkeypatch.setattr(vs, "_initialized", False, raising=False)
    monkeypatch.setattr(vs, "_init_lock", None, raising=False)
    monkeypatch.setattr(vs, "_client", None, raising=False)
    monkeypatch.setattr(vs, "_collections", {}, raising=False)

    constructions = {"n": 0}

    def _fake_persistent_client(*args, **kwargs):
        constructions["n"] += 1
        return _FakeClient()

    monkeypatch.setattr(chromadb, "PersistentClient", _fake_persistent_client)

    # Two concurrent inits = background warmup racing a first RAG query.
    r1, r2 = await asyncio.gather(
        vs.init_store(persist_dir=str(tmp_path)),
        vs.init_store(persist_dir=str(tmp_path)),
    )

    assert r1 is True and r2 is True
    assert constructions["n"] == 1, (
        f"expected exactly one Chroma client construction under concurrency, "
        f"got {constructions['n']}"
    )
