import pytest
import src.memory.vector_store as vs


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

    # _get_embed_fn is SYNC and returns an async callable (vector_store.py:37).
    def _fake_get_embed_fn():
        async def _embed(text, is_query=False):
            return [0.0] * 8
        return _embed

    monkeypatch.setattr(vs, "_get_embed_fn", _fake_get_embed_fn)
    return col


@pytest.mark.asyncio
async def test_implicit_accepted_feedback_is_dropped(fake_store):
    doc_id = await vs.embed_and_store(
        text="Feedback on task: X\nType: accepted\nAgent: coder",
        metadata={"type": "user_feedback", "feedback_type": "accepted", "task_id": "1"},
        collection="semantic",
        doc_id="feedback-1-123",
    )
    assert doc_id is None
    assert fake_store.upserts == []


@pytest.mark.asyncio
async def test_explicit_correction_goes_through(fake_store):
    doc_id = await vs.embed_and_store(
        text="Feedback on task: X\nType: modified\nUser correction: use snake_case",
        metadata={"type": "user_feedback", "feedback_type": "modified", "task_id": "1"},
        collection="semantic",
        doc_id="feedback-1-456",
    )
    assert doc_id == "feedback-1-456"
    assert len(fake_store.upserts) == 1


@pytest.mark.asyncio
async def test_normal_task_result_write_goes_through(fake_store):
    doc_id = await vs.embed_and_store(
        text="Task: Build parser\nDescription: parse YAML\nOutcome: success\nResult: done",
        metadata={"type": "task_result", "task_id": "2"},
        collection="episodic",
        doc_id="task-2-123",
    )
    assert doc_id == "task-2-123"
    assert len(fake_store.upserts) == 1


@pytest.mark.asyncio
async def test_curated_fact_not_filtered(fake_store):
    """Blocker-1 guard at the seam: a bulleted fact must still store."""
    doc_id = await vs.embed_and_store(
        text="* Observation: the founder prefers snake_case",
        metadata={"type": "fact", "task_id": "3"},
        collection="semantic",
        doc_id="fact-3",
    )
    assert doc_id == "fact-3"
    assert len(fake_store.upserts) == 1
