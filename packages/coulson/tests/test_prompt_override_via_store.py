import asyncio
import finch.store as store
import coulson


class _P:
    name = "summarizer"
    _prompt_version_override = None


def test_override_loaded_from_injected_store():
    class S:
        async def get_active(self, key): return "FROM STORE" if key == "summarizer" else None
        async def save_version(self, *a, **k): return 1
        async def record_quality(self, *a, **k): return None
        async def list_versions(self, k): return []

    store.set_store(S())
    p = _P()
    asyncio.run(coulson._load_db_prompt_override(p))
    assert p._prompt_version_override == "FROM STORE"
    store.set_store(None)
