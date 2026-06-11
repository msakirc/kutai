"""classify_task is now a CPS kickoff (no await_inline). It enqueues the
classifier LLM child with an on_complete continuation and returns the child id.
The resume handler reconstructs the TaskClassification via parse_classification."""
import pytest
from src.core import task_classifier as tc


@pytest.mark.asyncio
async def test_classify_task_kicks_off_with_on_complete(monkeypatch):
    captured = {}

    async def fake_enqueue(spec, **kwargs):
        captured["spec"] = spec
        captured["kwargs"] = kwargs
        return 4242  # child task id

    monkeypatch.setattr(tc, "_enqueue", fake_enqueue, raising=False)
    child_id = await tc.classify_task("build a parser", "write a JSON parser")
    assert child_id == 4242
    assert captured["kwargs"]["on_complete"] == "task_classifier.classify.resume"
    # the kickoff must NOT pass await_inline
    assert "await_inline" not in captured["kwargs"]
    # cont_state carries title/description for the resume to rebuild classification
    st = captured["kwargs"]["cont_state"]
    assert st["title"] == "build a parser"
    assert "parser" in st["description"]
    # the enqueued LLM spec is a raw_dispatch classifier call
    assert captured["spec"]["context"]["llm_call"]["raw_dispatch"] is True


@pytest.mark.asyncio
async def test_classify_resume_builds_classification(monkeypatch):
    seen = {}
    monkeypatch.setattr(tc, "_on_classified", lambda cls, state: seen.update(cls=cls, state=state))
    await tc._classify_resume(
        99,
        {"content": '{"agent_type": "fixer", "difficulty": 6}'},
        {"title": "fix bug", "description": "auth error"},
    )
    assert seen["cls"].agent_type == "fixer"
    assert seen["cls"].difficulty == 6


@pytest.mark.asyncio
async def test_classify_resume_registered():
    from general_beckman.continuations import _HANDLERS
    tc.register_continuations()
    assert "task_classifier.classify.resume" in _HANDLERS
