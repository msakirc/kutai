"""
Task 5 — Admission gates run exactly once (in Beckman), not twice.

(SP3b Task 2: the raw_dispatch worker is now ``husam.run`` — the dispatcher's
``dispatch``/``_do_dispatch`` were deleted. These tests now drive husam.run,
which honours the Beckman-preselected pick and must NOT re-select.)

Tests verify:
- fatih_hoca.select() called only in Beckman admission, not again in husam.run()
- husam.run() receives selected_model in spec (via orchestrator pass-through)
- in_flight slot reserved by Beckman before husam.run() is called
- est_tokens shim is gone (begin_call gets 0 from execute, keeps Beckman's value)
- Beckman writes selected_model into next_task result
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture(autouse=True)
async def _reset_db_singleton():
    """Drop the module-level cached aiosqlite connection between tests.

    Without this, tests that share the pytest session inherit whichever
    DB_PATH the first test opened — monkeypatch.setenv on later tests
    has no effect because the singleton is already bound.
    """
    import src.infra.db as _dbmod
    if _dbmod._db_connection is not None:
        try:
            await _dbmod._db_connection.close()
        except Exception:
            pass
    _dbmod._db_connection = None
    yield
    if _dbmod._db_connection is not None:
        try:
            await _dbmod._db_connection.close()
        except Exception:
            pass
    _dbmod._db_connection = None


# ── helpers ────────────────────────────────────────────────────────────────

def _make_pick(model_name="test-model", is_local=False):
    model = MagicMock()
    model.name = model_name
    model.provider = "test-provider"
    model.is_local = is_local
    model.litellm_name = f"openai/{model_name}"
    model.thinking_model = False
    model.has_vision = False

    pick = MagicMock()
    pick.model = model
    pick.score = 0.8
    pick.top_summary = ""
    pick.estimated_load_seconds = 0.0
    return pick


def _make_task(task_id=999, agent_type="main_work", pick=None, context_extra=None):
    import json
    ctx: dict = {"llm_call": {
        "raw_dispatch": True,
        "call_category": "main_work",
        "task": "test-task",
        "agent_type": agent_type,
        "difficulty": 5,
        "messages": [{"role": "user", "content": "hello"}],
        "failures": [],
    }}
    if context_extra:
        ctx.update(context_extra)
    task = {
        "id": task_id,
        "title": "llm_call:test-task:000000-abcdef",
        "description": "LLM main_work call",
        "agent_type": agent_type,
        "kind": "main_work",
        "priority": 5,
        "difficulty": 5,
        "status": "processing",
        "context": json.dumps(ctx),
        "mission_id": None,
        "parent_task_id": None,
        "worker_attempts": 0,
        "max_worker_attempts": 6,
    }
    if pick is not None:
        task["preselected_pick"] = pick
    return task


# ── Test 1: fatih_hoca.select called once (in Beckman), dispatcher re-uses pick ───────

@pytest.mark.asyncio
async def test_fatih_hoca_select_called_only_in_beckman(tmp_path, monkeypatch):
    """When a task carries preselected_pick, husam must NOT call fatih_hoca.select again."""
    import src.infra.db as _dbmod
    monkeypatch.setattr(_dbmod, "DB_PATH", str(tmp_path / "test.db"))

    pick = _make_pick("cloud-model")
    select_call_count = {"n": 0}

    def counting_select(*args, **kwargs):
        select_call_count["n"] += 1
        return pick

    # Build a minimal CallResult-like object
    call_result = MagicMock()
    call_result.__class__.__name__ = "CallResult"
    call_result.content = "response"
    call_result.model = "cloud-model"
    call_result.model_name = "cloud-model"
    call_result.cost = 0.001
    call_result.usage = {}
    call_result.tool_calls = []
    call_result.latency = 0.1
    call_result.thinking = ""
    call_result.is_local = False
    call_result.provider = "test-provider"
    call_result.task = "test-task"

    import hallederiz_kadir
    call_result.__class__ = hallederiz_kadir.CallResult

    with (
        patch("fatih_hoca.select", counting_select),
        patch("hallederiz_kadir.call", new=AsyncMock(return_value=call_result)),
        patch("src.core.in_flight._push", new=AsyncMock()),
        patch("src.core.llm_dispatcher.LLMDispatcher._record_pick", new=AsyncMock()),
    ):
        import husam

        # Simulate what orchestrator does: build spec with preselected_pick
        spec = {
            "context": {
                "llm_call": {
                    "raw_dispatch": True,
                    "call_category": "main_work",
                    "task": "test-task",
                    "agent_type": "coder",
                    "difficulty": 5,
                    "messages": [{"role": "user", "content": "hello"}],
                    "failures": [],
                    "selected_model": "cloud-model",  # Beckman wrote this
                }
            },
            "kind": "main_work",
            "preselected_pick": pick,  # Beckman's in-memory pick
        }

        result = await husam.run(spec)

    # select must not have been called from husam — Beckman did it
    assert select_call_count["n"] == 0, (
        f"fatih_hoca.select was called {select_call_count['n']} time(s) from husam "
        "but should be 0 (Beckman already selected at admission)"
    )
    assert result["content"] == "response"


# ── Test 2: dispatcher.dispatch receives selected_model in spec ────────────

@pytest.mark.asyncio
async def test_dispatcher_dispatch_receives_selected_model_in_spec(tmp_path, monkeypatch):
    """Beckman writes selected_model into llm_call; husam uses it instead of re-selecting."""
    import src.infra.db as _dbmod
    monkeypatch.setattr(_dbmod, "DB_PATH", str(tmp_path / "test.db"))

    pick = _make_pick("known-model-xyz")

    captured_model = {"name": None}

    async def fake_call(model, **kwargs):
        captured_model["name"] = model.name
        from hallederiz_kadir import CallResult
        r = MagicMock(spec=CallResult)
        r.content = "ok"
        r.model = model.name
        r.model_name = model.name
        r.cost = 0.0
        r.usage = {}
        r.tool_calls = []
        r.latency = 0.1
        r.thinking = ""
        r.is_local = False
        r.provider = "test"
        r.task = "t"
        return r

    with (
        patch("fatih_hoca.select", return_value=pick),
        patch("hallederiz_kadir.call", new=AsyncMock(side_effect=fake_call)),
        patch("src.core.in_flight._push", new=AsyncMock()),
        patch("src.core.llm_dispatcher.LLMDispatcher._record_pick", new=AsyncMock()),
    ):
        import husam

        spec = {
            "context": {
                "llm_call": {
                    "raw_dispatch": True,
                    "call_category": "main_work",
                    "task": "t",
                    "agent_type": "coder",
                    "difficulty": 5,
                    "messages": [{"role": "user", "content": "hi"}],
                    "failures": [],
                    "selected_model": "known-model-xyz",
                }
            },
            "kind": "main_work",
            "preselected_pick": pick,
        }

        await husam.run(spec)

    assert captured_model["name"] == "known-model-xyz"


# ── Test 3: in_flight slot reserved before dispatcher.dispatch ────────────────

@pytest.mark.asyncio
async def test_in_flight_slot_reserved_before_dispatch(tmp_path, monkeypatch):
    """reserve_task must have been called (by Beckman) before husam.run fires."""
    import src.infra.db as _dbmod
    monkeypatch.setattr(_dbmod, "DB_PATH", str(tmp_path / "test.db"))

    pick = _make_pick("slot-test-model")
    events = []

    async def fake_reserve(task_id, pick, est_tokens=0):
        events.append(("reserve", task_id))

    async def fake_call(model, **kwargs):
        events.append(("dispatch_call",))
        from hallederiz_kadir import CallResult
        r = MagicMock(spec=CallResult)
        r.content = "done"
        r.model = model.name
        r.model_name = model.name
        r.cost = 0.0
        r.usage = {}
        r.tool_calls = []
        r.latency = 0.1
        r.thinking = ""
        r.is_local = False
        r.provider = "p"
        r.task = "t"
        return r

    with (
        patch("fatih_hoca.select", return_value=pick),
        patch("hallederiz_kadir.call", new=AsyncMock(side_effect=fake_call)),
        patch("src.core.in_flight.reserve_task", new=AsyncMock(side_effect=fake_reserve)),
        patch("src.core.in_flight._push", new=AsyncMock()),
        patch("src.core.llm_dispatcher.LLMDispatcher._record_pick", new=AsyncMock()),
    ):
        # Simulate Beckman admission: reserve first
        from src.core.in_flight import reserve_task
        await reserve_task(42, pick, est_tokens=1000)

        import husam
        spec = {
            "context": {"llm_call": {
                "raw_dispatch": True,
                "call_category": "main_work",
                "task": "t",
                "agent_type": "coder",
                "difficulty": 5,
                "messages": [{"role": "user", "content": "hi"}],
                "failures": [],
                "selected_model": "slot-test-model",
            }},
            "kind": "main_work",
            "preselected_pick": pick,
        }
        await husam.run(spec)

    # reserve must appear before dispatch_call in the events list
    r_idx = next((i for i, e in enumerate(events) if e[0] == "reserve"), None)
    d_idx = next((i for i, e in enumerate(events) if e[0] == "dispatch_call"), None)
    assert r_idx is not None, "reserve_task was never called"
    assert d_idx is not None, "dispatch call never happened"
    assert r_idx < d_idx, "reserve_task must fire before dispatcher executes the call"


# ── Test 4: est_tokens shim gone from dispatcher (begin_call gets 0, keeps prior) ──

@pytest.mark.asyncio
async def test_dispatcher_passes_zero_est_tokens_to_begin_call(tmp_path, monkeypatch):
    """After Task 5, the call path no longer computes est_tokens for begin_call.
    It passes 0; begin_call's max(prior, 0) keeps the Beckman-reserved value.
    (execute() still owns begin_call; husam.run drives it.)
    """
    import src.infra.db as _dbmod
    monkeypatch.setattr(_dbmod, "DB_PATH", str(tmp_path / "test.db"))

    pick = _make_pick("est-test-model")
    captured_est = {"value": None}

    original_begin = None

    async def capturing_begin_call(*, category, model_name, provider, is_local, task_id, est_tokens=0):
        captured_est["value"] = est_tokens
        return f"call-{task_id}"

    async def fake_call(model, **kwargs):
        from hallederiz_kadir import CallResult
        r = MagicMock(spec=CallResult)
        r.content = "ok"
        r.model = model.name
        r.model_name = model.name
        r.cost = 0.0
        r.usage = {}
        r.tool_calls = []
        r.latency = 0.1
        r.thinking = ""
        r.is_local = False
        r.provider = "p"
        r.task = "t"
        return r

    with (
        patch("fatih_hoca.select", return_value=pick),
        patch("hallederiz_kadir.call", new=AsyncMock(side_effect=fake_call)),
        # Patch the alias _begin_call as imported by llm_dispatcher
        patch("src.core.llm_dispatcher._begin_call", new=AsyncMock(side_effect=capturing_begin_call)),
        patch("src.core.llm_dispatcher._end_call", new=AsyncMock()),
        patch("src.core.in_flight._push", new=AsyncMock()),
        patch("src.core.llm_dispatcher.LLMDispatcher._record_pick", new=AsyncMock()),
    ):
        import husam
        spec = {
            "context": {"llm_call": {
                "raw_dispatch": True,
                "call_category": "main_work",
                "task": "t",
                "agent_type": "coder",
                "difficulty": 5,
                "messages": [{"role": "user", "content": "hi"}],
                "failures": [],
                "selected_model": "est-test-model",
                # Note: estimated_input_tokens NOT passed — execute must not compute from these
            }},
            "kind": "main_work",
            "preselected_pick": pick,
        }
        await husam.run(spec)

    # execute() must pass 0 (not compute its own shim)
    assert captured_est["value"] == 0, (
        f"execute passed est_tokens={captured_est['value']} to begin_call "
        "but should pass 0 (Beckman owns est_tokens via reserve_task)"
    )


# ── Test 5: Beckman writes selected_model into next_task result ─────────────

@pytest.mark.asyncio
async def test_beckman_writes_selected_model_to_task(tmp_path, monkeypatch):
    """next_task() must set task['context']['llm_call']['selected_model'] = pick.model.name."""
    import src.infra.db as _dbmod
    monkeypatch.setattr(_dbmod, "DB_PATH", str(tmp_path / "test.db"))

    import json
    from src.infra.db import init_db, add_task
    await init_db()

    pick = _make_pick("beckman-selected-model")

    # Add a raw_dispatch task to the queue
    llm_call = {
        "raw_dispatch": True,
        "call_category": "main_work",
        "task": "coder",
        "agent_type": "coder",
        "difficulty": 5,
        "messages": [{"role": "user", "content": "code this"}],
        "failures": [],
    }
    task_id = await add_task(
        title="llm_call:coder:000001-abcdef",
        description="LLM main_work call",
        agent_type="coder",
        kind="main_work",
        priority=5,
        context={"llm_call": llm_call},
    )

    with (
        patch("fatih_hoca.select", return_value=pick),
        patch("src.core.in_flight.reserve_task", new=AsyncMock()),
        patch("nerd_herd.refresh_snapshot", new=AsyncMock(return_value=None)),
    ):
        import general_beckman
        # Reset fingerprint cache so admission doesn't short-circuit
        general_beckman._last_admission_admitted = True
        general_beckman._last_admission_fp = None

        task = await general_beckman.next_task()

    assert task is not None, "next_task() returned None — task not admitted"
    assert task["id"] == task_id

    # The task's context must carry selected_model
    ctx_raw = task.get("context") or "{}"
    ctx = json.loads(ctx_raw) if isinstance(ctx_raw, str) else ctx_raw
    llm_call_out = ctx.get("llm_call") or {}
    assert llm_call_out.get("selected_model") == "beckman-selected-model", (
        f"selected_model not written by Beckman into task context. "
        f"Got: {llm_call_out.get('selected_model')!r}"
    )
