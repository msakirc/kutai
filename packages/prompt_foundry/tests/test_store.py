import asyncio
import prompt_foundry.store as store

def test_get_active_returns_none_without_store():
    store.set_store(None)
    assert asyncio.run(store.get_active("summarizer")) is None

def test_injected_store_is_used():
    class FakeStore:
        async def get_active(self, key):
            return "DB PROMPT" if key == "summarizer" else None
        async def save_version(self, key, text, notes="", activate=False):
            return 1
        async def record_quality(self, key, score):
            return None
        async def list_versions(self, key):
            return []
    store.set_store(FakeStore())
    assert asyncio.run(store.get_active("summarizer")) == "DB PROMPT"
    assert asyncio.run(store.get_active("other")) is None
    store.set_store(None)  # reset
