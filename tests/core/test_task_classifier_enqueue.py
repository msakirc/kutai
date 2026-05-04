"""
TDD tests for Site 10 migration: pre-task classifier enqueues
with kind='classifier' via beckman.enqueue(await_inline=True).
"""
from __future__ import annotations

import pytest
from unittest.mock import patch


@pytest.mark.asyncio
async def test_classify_with_llm_enqueues_kind_classifier(tmp_path, monkeypatch):
    """_classify_with_llm must call beckman.enqueue with kind='classifier'."""
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    captured = {}

    async def fake_enqueue(spec, **kwargs):
        captured["spec"] = spec
        captured["kwargs"] = kwargs
        from general_beckman import TaskResult
        return TaskResult(
            status="completed",
            result={
                "content": '{"agent_type": "coder", "difficulty": 5, "needs_tools": true, "needs_vision": false, "needs_thinking": false, "local_only": false, "priority": "normal", "search_depth": "none"}'
            },
            error=None,
        )

    with patch("general_beckman.enqueue", fake_enqueue):
        from src.core.task_classifier import _classify_with_llm
        cls = await _classify_with_llm("Build login page", "Create a login page with React")

    spec = captured["spec"]
    kwargs = captured["kwargs"]

    assert spec["kind"] == "classifier", f"Expected kind='classifier', got {spec.get('kind')!r}"
    assert kwargs.get("await_inline") is True
    assert kwargs.get("parent_id") is None
    assert spec["context"]["llm_call"]["raw_dispatch"] is True
    assert cls.agent_type == "coder"
    assert cls.difficulty == 5


@pytest.mark.asyncio
async def test_classify_with_llm_await_inline_true(tmp_path, monkeypatch):
    """await_inline=True must be passed."""
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    captured_kwargs = {}

    async def fake_enqueue(spec, **kwargs):
        captured_kwargs.update(kwargs)
        from general_beckman import TaskResult
        return TaskResult(
            status="completed",
            result={
                "content": '{"agent_type": "researcher", "difficulty": 3, "needs_tools": false, "needs_vision": false, "needs_thinking": false, "local_only": false, "priority": "normal", "search_depth": "shallow"}'
            },
            error=None,
        )

    with patch("general_beckman.enqueue", fake_enqueue):
        from src.core.task_classifier import _classify_with_llm
        await _classify_with_llm("Research X", "Research about topic X")

    assert captured_kwargs.get("await_inline") is True


@pytest.mark.asyncio
async def test_classify_with_llm_parent_id_none(tmp_path, monkeypatch):
    """parent_id must be None — this runs BEFORE any task exists."""
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    captured_kwargs = {}

    async def fake_enqueue(spec, **kwargs):
        captured_kwargs.update(kwargs)
        from general_beckman import TaskResult
        return TaskResult(
            status="completed",
            result={
                "content": '{"agent_type": "executor", "difficulty": 2, "needs_tools": false, "needs_vision": false, "needs_thinking": false, "local_only": false, "priority": "normal", "search_depth": "none"}'
            },
            error=None,
        )

    with patch("general_beckman.enqueue", fake_enqueue):
        from src.core.task_classifier import _classify_with_llm
        await _classify_with_llm("Simple task", "Do a simple thing")

    assert captured_kwargs.get("parent_id") is None


@pytest.mark.asyncio
async def test_classify_with_llm_spec_context_has_messages(tmp_path, monkeypatch):
    """spec.context.llm_call must carry messages."""
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    captured = {}

    async def fake_enqueue(spec, **kwargs):
        captured["spec"] = spec
        from general_beckman import TaskResult
        return TaskResult(
            status="completed",
            result={
                "content": '{"agent_type": "coder", "difficulty": 7, "needs_tools": true, "needs_vision": false, "needs_thinking": true, "local_only": false, "priority": "high", "search_depth": "deep"}'
            },
            error=None,
        )

    with patch("general_beckman.enqueue", fake_enqueue):
        from src.core.task_classifier import _classify_with_llm
        await _classify_with_llm("Hard task", "Implement a distributed lock")

    llm_call = captured["spec"]["context"]["llm_call"]
    assert "messages" in llm_call
    assert isinstance(llm_call["messages"], list)
    assert len(llm_call["messages"]) > 0
    # Message content should include the task description
    all_content = " ".join(str(m) for m in llm_call["messages"])
    assert "distributed lock" in all_content or "Hard task" in all_content


@pytest.mark.asyncio
async def test_classify_with_llm_result_shape(tmp_path, monkeypatch):
    """Return value must be a TaskClassification with correct fields."""
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    async def fake_enqueue(spec, **kwargs):
        from general_beckman import TaskResult
        return TaskResult(
            status="completed",
            result={
                "content": '{"agent_type": "fixer", "difficulty": 4, "needs_tools": true, "needs_vision": false, "needs_thinking": false, "local_only": true, "priority": "high", "search_depth": "none"}'
            },
            error=None,
        )

    with patch("general_beckman.enqueue", fake_enqueue):
        from src.core.task_classifier import _classify_with_llm, TaskClassification
        cls = await _classify_with_llm("Fix bug", "Fix the login bug")

    assert isinstance(cls, TaskClassification)
    assert cls.agent_type == "fixer"
    assert cls.difficulty == 4
    assert cls.needs_tools is True
    assert cls.local_only is True
    assert cls.method == "llm"
