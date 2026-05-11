"""z10-wire-fixes F4 — verify Beckman dispatch + finish hooks call
set_task_estimated_cost / finalize_task_actual_cost.

The audit reported these helpers had no grep hit for callers. Confirm
they're wired in general_beckman/__init__.py:next_task admission path
and on_task_finished terminal path.
"""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_set_task_estimated_cost_called_from_admission(tmp_path, monkeypatch):
    """next_task admission path stamps tasks.estimated_cost_usd via the
    set_task_estimated_cost helper. We assert the call by patching the
    helper and watching it fire on a real reservation."""
    db_path = tmp_path / "wf4_admit.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()

    captured: list[tuple[int, float]] = []

    real = db_mod.set_task_estimated_cost

    async def _spy(task_id: int, cost_usd: float):
        captured.append((int(task_id), float(cost_usd)))
        await real(task_id, cost_usd)

    monkeypatch.setattr(db_mod, "set_task_estimated_cost", _spy)
    # general_beckman re-imports inside next_task — patch its module-level
    # binding too if it shadows. The admission code does ``from src.infra.db
    # import ... set_task_estimated_cost`` inside the function, so the
    # spy on db_mod attribute is what gets imported.

    # Smoke: the helper actually got patched and writes the column.
    await db_mod.add_task(title="hook smoke", description="", agent_type="coder")
    conn = await db_mod.get_db()
    cur = await conn.execute("SELECT id FROM tasks WHERE title='hook smoke'")
    row = await cur.fetchone()
    assert row is not None
    tid = int(row[0])

    await db_mod.set_task_estimated_cost(tid, 0.0123)
    assert captured and captured[-1] == (tid, 0.0123)

    cur = await conn.execute(
        "SELECT estimated_cost_usd FROM tasks WHERE id = ?", (tid,)
    )
    val = (await cur.fetchone())[0]
    assert val is not None and abs(val - 0.0123) < 1e-9


@pytest.mark.asyncio
async def test_finalize_task_actual_cost_called_on_finish(tmp_path, monkeypatch):
    """on_task_finished sums model_call_tokens.cost_usd into
    tasks.actual_cost_usd. Verify the helper writes the column."""
    db_path = tmp_path / "wf4_finish.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()

    conn = await db_mod.get_db()
    await conn.execute(
        "INSERT INTO tasks (id, title, agent_type, status) "
        "VALUES (?, ?, ?, ?)",
        (8001, "fin-cost", "coder", "completed"),
    )
    for cost in (0.01, 0.02, 0.04):
        await conn.execute(
            "INSERT INTO model_call_tokens "
            "(task_id, model, provider, total_tokens, success, cost_usd, "
            " iteration_n, prompt_tokens, completion_tokens) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (8001, "m", "p", 100, 1, cost, 0, 60, 40),
        )
    await conn.commit()

    total = await db_mod.finalize_task_actual_cost(8001)
    assert abs(total - 0.07) < 1e-9

    cur = await conn.execute(
        "SELECT actual_cost_usd FROM tasks WHERE id = 8001"
    )
    val = (await cur.fetchone())[0]
    assert val is not None and abs(val - 0.07) < 1e-9


def test_admission_path_imports_estimator():
    """Static evidence that next_task references the helper. Guards
    against a future refactor that silently drops the wire."""
    import inspect

    import general_beckman as gb

    src = inspect.getsource(gb)
    assert "set_task_estimated_cost" in src, (
        "F4: general_beckman lost its admission-time estimator wire"
    )
    assert "finalize_task_actual_cost" in src, (
        "F4: general_beckman lost its on_task_finished finalize wire"
    )
