import pytest
import src.memory.vector_store as vs
import src.memory.episodic as episodic


class _FakeCollection:
    def __init__(self):
        self.upserts = []

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)


@pytest.fixture
def fake_store(monkeypatch):
    col = _FakeCollection()
    monkeypatch.setattr(vs, "_initialized", True)
    monkeypatch.setattr(vs, "_collections", {"semantic": col, "episodic": col})

    def _fake_get_embed_fn():
        async def _embed(text, is_query=False):
            return [0.0] * 8
        return _embed

    monkeypatch.setattr(vs, "_get_embed_fn", _fake_get_embed_fn)
    return col


@pytest.mark.asyncio
async def test_good_task_result_still_stored(fake_store):
    doc_id = await episodic.store_task_result(
        task={"id": 7, "title": "Build parser", "description": "parse YAML", "agent_type": "coder"},
        result="Implemented the YAML parser and all tests pass.",
        model="m", cost=0.0, duration=0.0, success=True,
    )
    assert doc_id is not None
    assert len(fake_store.upserts) == 1
    assert fake_store.upserts[0]["metadatas"][0]["type"] == "task_result"


@pytest.mark.asyncio
async def test_bulleted_markdown_task_result_still_stored(fake_store):
    """Guard for the Minor-1 fix: a legit multi-line, bulleted, keyword-y
    design-artifact result (the shape the old pollution regex wrongly dropped)
    must still be stored."""
    doc_id = await episodic.store_task_result(
        task={"id": 8, "title": "Design tokens", "description": "author tokens", "agent_type": "analyst"},
        result="## Analysis: design tokens\n---\n_schema_version: 1\n* STRATEGY: reuse palette\nObservation: complete",
        model="m", cost=0.0, duration=0.0, success=True,
    )
    assert doc_id is not None
    assert len(fake_store.upserts) == 1
