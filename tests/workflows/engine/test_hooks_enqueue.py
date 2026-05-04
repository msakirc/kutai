"""
Tests for Site 6 migration: _llm_summarize() in hooks.py calls beckman.enqueue
directly instead of dispatcher.request() alias.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_llm_summarize_enqueues_with_overhead_kind(tmp_path, monkeypatch):
    """_llm_summarize must enqueue with kind='overhead' and await_inline=True."""
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
            result={"content": "A sufficiently long summary of the artifact content preserving key facts and decisions for downstream use."},
            error=None,
        )

    with patch("general_beckman.enqueue", fake_enqueue), \
         patch("dogru_mu_samet.assess") as mock_assess:
        mock_assess.return_value = MagicMock(is_degenerate=False)
        from src.workflows.engine.hooks import _llm_summarize
        result = await _llm_summarize("some long text " * 100, "test_artifact")

    assert captured["kwargs"].get("await_inline") is True
    assert captured["spec"]["kind"] == "overhead"
    assert captured["spec"]["context"]["llm_call"]["raw_dispatch"] is True
    assert result is not None
    assert len(result) > 10


@pytest.mark.asyncio
async def test_llm_summarize_enqueue_has_overhead_call_category(tmp_path, monkeypatch):
    """The llm_call payload must set call_category='overhead'."""
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
            result={"content": "Concise summary with key facts here covering the material."},
            error=None,
        )

    with patch("general_beckman.enqueue", fake_enqueue), \
         patch("dogru_mu_samet.assess") as mock_assess:
        mock_assess.return_value = MagicMock(is_degenerate=False)
        from src.workflows.engine.hooks import _llm_summarize
        await _llm_summarize("artifact text " * 50, "my_artifact")

    llm_call = captured["spec"]["context"]["llm_call"]
    assert llm_call["call_category"] == "overhead"
    assert llm_call["task"] == "summarizer"


@pytest.mark.asyncio
async def test_llm_summarize_parent_id_from_current_task_id_contextvar(tmp_path, monkeypatch):
    """parent_id must be taken from current_task_id ContextVar when available."""
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
            result={"content": "Summary that is long enough to pass quality check here."},
            error=None,
        )

    from src.core.heartbeat import current_task_id as _ctid
    token = _ctid.set(123)
    try:
        with patch("general_beckman.enqueue", fake_enqueue), \
             patch("dogru_mu_samet.assess") as mock_assess:
            mock_assess.return_value = MagicMock(is_degenerate=False)
            from src.workflows.engine.hooks import _llm_summarize
            await _llm_summarize("text " * 100, "artifact_x")
    finally:
        _ctid.reset(token)

    assert captured["kwargs"].get("parent_id") == 123


@pytest.mark.asyncio
async def test_llm_summarize_parent_id_none_when_no_contextvar(tmp_path, monkeypatch):
    """When current_task_id ContextVar is not set, parent_id should be None (no error)."""
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
            result={"content": "Enough length summary text passes quality verification here."},
            error=None,
        )

    # Ensure contextvar is cleared
    from src.core.heartbeat import current_task_id as _ctid
    token = _ctid.set(None)
    try:
        with patch("general_beckman.enqueue", fake_enqueue), \
             patch("dogru_mu_samet.assess") as mock_assess:
            mock_assess.return_value = MagicMock(is_degenerate=False)
            from src.workflows.engine.hooks import _llm_summarize
            await _llm_summarize("long text " * 80, "art_y")
    finally:
        _ctid.reset(token)

    # parent_id should be None or absent
    assert captured["kwargs"].get("parent_id") is None


@pytest.mark.asyncio
async def test_llm_summarize_returns_none_on_short_summary(tmp_path, monkeypatch):
    """If the enqueued call returns content shorter than 50 chars, return None."""
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    async def fake_enqueue(spec, **kwargs):
        from general_beckman import TaskResult
        return TaskResult(
            status="completed",
            result={"content": "Too short"},
            error=None,
        )

    with patch("general_beckman.enqueue", fake_enqueue), \
         patch("dogru_mu_samet.assess") as mock_assess:
        mock_assess.return_value = MagicMock(is_degenerate=False)
        from src.workflows.engine.hooks import _llm_summarize
        result = await _llm_summarize("text " * 100, "artifact_z")

    assert result is None
