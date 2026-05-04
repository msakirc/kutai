"""
Tests for Site 5 migration: grade_task() calls beckman.enqueue directly
instead of dispatcher.request() alias.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch



@pytest.fixture(autouse=True)
async def _reset_db_singleton():
    import src.infra.db as _dbmod
    if _dbmod._db_connection is not None:
        try: await _dbmod._db_connection.close()
        except Exception: pass
    _dbmod._db_connection = None
    yield
    if _dbmod._db_connection is not None:
        try: await _dbmod._db_connection.close()
        except Exception: pass
    _dbmod._db_connection = None


@pytest.mark.asyncio
async def test_grade_task_enqueues_with_overhead_kind(tmp_path, monkeypatch):
    """grade_task must enqueue with kind='overhead' and await_inline=True."""
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
                "content": (
                    "RELEVANT: YES\nCOMPLETE: YES\nVERDICT: PASS\n"
                    "WELL_FORMED: PASS\nCOHERENT: PASS\n"
                    "SITUATION: test\nSTRATEGY: used tools\nTOOLS: shell\n"
                    "PREFERENCE: NONE\nINSIGHT: NONE"
                ),
                "model": "test-model",
            },
            error=None,
        )

    task = {
        "id": 42,
        "title": "Test task",
        "description": "Do something",
        "result": "x" * 20,
        "context": "{}",
    }

    with patch("general_beckman.enqueue", fake_enqueue), \
         patch("dogru_mu_samet.assess") as mock_assess:
        mock_assess.return_value = MagicMock(is_degenerate=False, summary="ok")
        from src.core.grading import grade_task
        result = await grade_task(task)

    assert captured["kwargs"].get("await_inline") is True
    assert captured["spec"]["kind"] == "overhead"
    assert captured["kwargs"].get("parent_id") == 42
    assert captured["spec"]["context"]["llm_call"]["raw_dispatch"] is True
    assert result.passed is True


@pytest.mark.asyncio
async def test_grade_task_enqueue_spec_carries_correct_call_category(tmp_path, monkeypatch):
    """The llm_call payload must set call_category='overhead'."""
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
                "content": (
                    "RELEVANT: YES\nCOMPLETE: YES\nVERDICT: PASS\n"
                    "WELL_FORMED: PASS\nCOHERENT: PASS\n"
                    "SITUATION: s\nSTRATEGY: t\nTOOLS: shell\n"
                    "PREFERENCE: NONE\nINSIGHT: NONE"
                ),
                "model": "test-model",
            },
            error=None,
        )

    task = {
        "id": 99,
        "title": "Another task",
        "description": "desc",
        "result": "a" * 30,
        "context": "{}",
    }

    with patch("general_beckman.enqueue", fake_enqueue), \
         patch("dogru_mu_samet.assess") as mock_assess:
        mock_assess.return_value = MagicMock(is_degenerate=False)
        from src.core.grading import grade_task
        await grade_task(task)

    llm_call = captured["spec"]["context"]["llm_call"]
    assert llm_call["call_category"] == "overhead"
    assert llm_call["task"] == "reviewer"


@pytest.mark.asyncio
async def test_grade_task_unwraps_task_result_to_graderesult(tmp_path, monkeypatch):
    """A TaskResult with failing grader output should produce GradeResult(passed=False)."""
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    async def fake_enqueue(spec, **kwargs):
        from general_beckman import TaskResult
        return TaskResult(
            status="completed",
            result={"content": "VERDICT: FAIL\nWELL_FORMED: PASS\nCOHERENT: PASS\nSITUATION: s\nSTRATEGY: t\nTOOLS: x\nPREFERENCE: NONE\nINSIGHT: NONE"},
            error=None,
        )

    task = {
        "id": 7,
        "title": "Fail task",
        "description": "desc",
        "result": "b" * 20,
        "context": "{}",
    }

    with patch("general_beckman.enqueue", fake_enqueue), \
         patch("dogru_mu_samet.assess") as mock_assess:
        mock_assess.return_value = MagicMock(is_degenerate=False)
        from src.core.grading import grade_task
        result = await grade_task(task)

    assert result.passed is False


@pytest.mark.asyncio
async def test_grade_task_enqueue_failed_status_returns_auto_fail(tmp_path, monkeypatch):
    """When Beckman returns status='failed', grade_task should return auto-fail GradeResult."""
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    async def fake_enqueue(spec, **kwargs):
        from general_beckman import TaskResult
        return TaskResult(status="failed", result=None, error="no model available")

    task = {
        "id": 5,
        "title": "T",
        "description": "D",
        "result": "c" * 20,
        "context": "{}",
    }

    with patch("general_beckman.enqueue", fake_enqueue), \
         patch("dogru_mu_samet.assess") as mock_assess:
        mock_assess.return_value = MagicMock(is_degenerate=False)
        from src.core.grading import grade_task
        result = await grade_task(task)

    # Should auto-fail (not raise) since grade_task catches errors gracefully
    assert result.passed is False
