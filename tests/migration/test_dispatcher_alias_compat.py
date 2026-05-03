"""
Tests for dispatcher.request() alias routing through beckman.enqueue(await_inline=True).

Task 4: Beckman admission migration — install the routing pipe.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch


@pytest.mark.asyncio
async def test_request_alias_routes_through_beckman_enqueue(tmp_path, monkeypatch):
    """request() must call beckman.enqueue with await_inline=True and return mapped response."""
    import src.infra.db as _dbmod
    monkeypatch.setattr(_dbmod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    fake_result_dict = {"content": "hello world", "tokens_used": 42}
    captured = {}

    async def fake_enqueue(spec, **kwargs):
        captured["spec"] = spec
        captured["kwargs"] = kwargs
        from general_beckman import TaskResult
        return TaskResult(status="completed", result=fake_result_dict, error=None)

    with patch("general_beckman.enqueue", fake_enqueue):
        from src.core.llm_dispatcher import get_dispatcher, CallCategory
        resp = await get_dispatcher().request(
            CallCategory.OVERHEAD,
            task="grader",
            agent_type="reviewer",
            difficulty=2,
            messages=[{"role": "user", "content": "x"}],
            prefer_speed=True,
            estimated_input_tokens=200,
            estimated_output_tokens=100,
        )

    assert captured["kwargs"].get("await_inline") is True
    assert captured["spec"]["kind"] == "overhead"
    assert "llm_call" in captured["spec"].get("context", {})
    assert captured["spec"]["context"]["llm_call"]["raw_dispatch"] is True
    assert resp["content"] == "hello world"


@pytest.mark.asyncio
async def test_request_main_work_category_maps_to_main_work_kind(tmp_path, monkeypatch):
    import src.infra.db as _dbmod
    monkeypatch.setattr(_dbmod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()
    captured = {}

    async def fake_enqueue(spec, **kwargs):
        captured["spec"] = spec
        from general_beckman import TaskResult
        return TaskResult(status="completed", result={"content": "ok"}, error=None)

    with patch("general_beckman.enqueue", fake_enqueue):
        from src.core.llm_dispatcher import get_dispatcher, CallCategory
        await get_dispatcher().request(CallCategory.MAIN_WORK, task="t", messages=[])
    assert captured["spec"]["kind"] == "main_work"


@pytest.mark.asyncio
async def test_request_propagates_error_on_failure(tmp_path, monkeypatch):
    """When TaskResult.status == 'failed', request() should raise ModelCallFailed."""
    import src.infra.db as _dbmod
    monkeypatch.setattr(_dbmod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    async def fake_enqueue(spec, **kwargs):
        from general_beckman import TaskResult
        return TaskResult(status="failed", result={}, error="something broke")

    from src.core.router import ModelCallFailed
    with patch("general_beckman.enqueue", fake_enqueue):
        from src.core.llm_dispatcher import get_dispatcher, CallCategory
        with pytest.raises((ModelCallFailed, RuntimeError)):
            await get_dispatcher().request(CallCategory.OVERHEAD, task="t", messages=[])


@pytest.mark.asyncio
async def test_request_kwargs_embedded_in_context(tmp_path, monkeypatch):
    """Ensure key kwargs are preserved inside context.llm_call."""
    import src.infra.db as _dbmod
    monkeypatch.setattr(_dbmod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    captured = {}

    async def fake_enqueue(spec, **kwargs):
        captured["spec"] = spec
        from general_beckman import TaskResult
        return TaskResult(status="completed", result={"content": "ok"}, error=None)

    messages = [{"role": "user", "content": "test"}]
    tools = [{"name": "my_tool"}]

    with patch("general_beckman.enqueue", fake_enqueue):
        from src.core.llm_dispatcher import get_dispatcher, CallCategory
        await get_dispatcher().request(
            CallCategory.MAIN_WORK,
            task="coder",
            agent_type="coder",
            difficulty=7,
            messages=messages,
            tools=tools,
            prefer_speed=False,
            prefer_local=True,
            needs_json_mode=True,
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
        )

    spec = captured["spec"]
    llm_call = spec["context"]["llm_call"]
    assert llm_call["messages"] == messages
    assert llm_call["tools"] == tools
    assert llm_call["difficulty"] == 7
    assert llm_call["agent_type"] == "coder"
    assert llm_call["prefer_local"] is True
    assert llm_call["needs_json_mode"] is True
    assert llm_call["estimated_input_tokens"] == 1000
    assert llm_call["estimated_output_tokens"] == 500


@pytest.mark.asyncio
async def test_request_title_derives_from_task_kwarg(tmp_path, monkeypatch):
    """spec.title should be derived from task kwarg."""
    import src.infra.db as _dbmod
    monkeypatch.setattr(_dbmod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    captured = {}

    async def fake_enqueue(spec, **kwargs):
        captured["spec"] = spec
        from general_beckman import TaskResult
        return TaskResult(status="completed", result={"content": "ok"}, error=None)

    with patch("general_beckman.enqueue", fake_enqueue):
        from src.core.llm_dispatcher import get_dispatcher, CallCategory
        await get_dispatcher().request(
            CallCategory.OVERHEAD,
            task="summarizer",
            messages=[],
        )

    assert "summarizer" in captured["spec"]["title"]


@pytest.mark.asyncio
async def test_overhead_failure_raises_runtime_error(tmp_path, monkeypatch):
    """OVERHEAD failure should also raise (ModelCallFailed or RuntimeError)."""
    import src.infra.db as _dbmod
    monkeypatch.setattr(_dbmod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    async def fake_enqueue(spec, **kwargs):
        from general_beckman import TaskResult
        return TaskResult(status="failed", result={}, error="overhead broken")

    with patch("general_beckman.enqueue", fake_enqueue):
        from src.core.llm_dispatcher import get_dispatcher, CallCategory
        from src.core.router import ModelCallFailed
        with pytest.raises((ModelCallFailed, RuntimeError)):
            await get_dispatcher().request(CallCategory.OVERHEAD, task="t", messages=[])
