import asyncio
import pytest
import finch.store as store


@pytest.fixture(autouse=True)
def reset_store():
    store.set_store(None)
    yield
    store.set_store(None)


def test_get_active_returns_none_without_store():
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


def test_record_quality_delegated():
    calls = []

    class FakeStore:
        async def get_active(self, key):
            return None
        async def save_version(self, key, text, notes="", activate=False):
            return 1
        async def record_quality(self, key, score):
            calls.append((key, score))
        async def list_versions(self, key):
            return []

    # With store set, call is delegated.
    store.set_store(FakeStore())
    asyncio.run(store.record_quality("summarizer", 9.0))
    assert calls == [("summarizer", 9.0)]

    # With no store, it is a no-op (no error raised).
    store.set_store(None)
    asyncio.run(store.record_quality("summarizer", 9.0))  # must not raise
