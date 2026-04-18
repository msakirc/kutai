"""Pick telemetry: model_pick_log table must exist after init_db()."""
from __future__ import annotations

import pytest
import aiosqlite


@pytest.mark.asyncio
async def test_model_pick_log_table_exists(tmp_path, monkeypatch):
    """After init_db(), model_pick_log must exist with expected columns."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))

    from src.infra.db import init_db
    await init_db()

    async with aiosqlite.connect(tmp_path / "test.db") as db:
        cur = await db.execute("PRAGMA table_info(model_pick_log)")
        cols = {row[1] for row in await cur.fetchall()}

    expected = {
        "id", "timestamp", "task_name", "agent_type", "difficulty",
        "call_category", "picked_model", "picked_score", "picked_reasons",
        "candidates_json", "failures_json", "snapshot_summary",
    }
    assert expected.issubset(cols), f"missing columns: {expected - cols}"
    assert "pool" in cols, f"missing 'pool' column: {cols}"
    assert "urgency" in cols, f"missing 'urgency' column: {cols}"


@pytest.mark.asyncio
async def test_model_pick_log_indexes_exist(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db, get_db
    await init_db()

    # Use the same db connection that init_db() populated
    db = await get_db()
    cur = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='model_pick_log'"
    )
    idx_names = {row[0] for row in await cur.fetchall()}

    assert "idx_pick_log_task" in idx_names, f"missing task index, got: {idx_names}"
    assert "idx_pick_log_model" in idx_names, f"missing model index, got: {idx_names}"


@pytest.mark.asyncio
async def test_select_persists_pick_to_db(tmp_path, monkeypatch, caplog):
    """A successful select() must write one row to model_pick_log with top candidates."""
    import json
    import logging
    import asyncio

    db_file = tmp_path / "test.db"

    # Patch the module-level DB_PATH in src.infra.db (captured at import time)
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(db_file))
    # Reset connection singleton so init_db() opens the new path
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db
    await init_db()

    import fatih_hoca
    from fatih_hoca.registry import ModelInfo, ModelRegistry
    from fatih_hoca.selector import Selector

    # Reset singletons
    fatih_hoca._registry = None
    fatih_hoca._selector = None

    reg = ModelRegistry()
    reg._models["a"] = ModelInfo(
        name="a", location="local",
        provider="llama_cpp", litellm_name="openai/a",
        path="/fake/a.gguf",
        total_params_b=8.0, active_params_b=8.0,
        capabilities={c: 7.0 for c in ["reasoning", "code_generation", "analysis", "instruction_adherence"]},
    )
    reg._models["b"] = ModelInfo(
        name="b", location="local",
        provider="llama_cpp", litellm_name="openai/b",
        path="/fake/b.gguf",
        total_params_b=8.0, active_params_b=8.0,
        capabilities={c: 5.0 for c in ["reasoning", "code_generation", "analysis", "instruction_adherence"]},
    )

    class _Nh:
        def snapshot(self):
            from nerd_herd.types import SystemSnapshot
            return SystemSnapshot(vram_available_mb=24000)

    fatih_hoca._registry = reg
    fatih_hoca._selector = Selector(registry=reg, nerd_herd=_Nh())

    from fatih_hoca import selector as _sel_mod
    monkeypatch.setattr(_sel_mod, "_telemetry_db_path", str(tmp_path / "test.db"))

    with caplog.at_level(logging.INFO, logger="fatih_hoca.selector"):
        pick = fatih_hoca.select(
            task="coder", agent_type="coder", difficulty=5,
            estimated_input_tokens=500, estimated_output_tokens=500,
            call_category="main_work",
        )

    assert pick is not None, "selector should return a Pick for eligible local models"

    # Give the fire-and-forget task a moment to complete
    await asyncio.sleep(0.3)

    # Logger: top-candidates line
    top_logs = [r for r in caplog.records if "picked=" in r.message and "candidates=" in r.message]
    assert top_logs, f"selector must emit a top-candidates summary log line, got: {[r.message for r in caplog.records]}"

    # DB: one row in model_pick_log
    async with aiosqlite.connect(tmp_path / "test.db") as db:
        cur = await db.execute(
            "SELECT picked_model, picked_score, candidates_json FROM model_pick_log"
        )
        rows = await cur.fetchall()
    assert len(rows) == 1, f"expected 1 pick row, got {len(rows)}"
    picked_model, picked_score, cand_json = rows[0]
    assert picked_model == pick.model.name
    cands = json.loads(cand_json)
    assert len(cands) >= 2, "candidates_json must include all ranked candidates"
    assert all("name" in c and "composite" in c and "reasons" in c for c in cands)

    # Cleanup: close and reset shared connection so subsequent tests get a fresh DB
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None


@pytest.mark.asyncio
async def test_telemetry_does_not_leak_when_os_env_has_db_path(tmp_path, monkeypatch):
    """Repro for 2026-04-17 production pollution: OS-level DB_PATH must not
    drive telemetry writes. Requires explicit enable_telemetry() opt-in."""
    prod_db = tmp_path / "fake_production.db"
    monkeypatch.setenv("DB_PATH", str(prod_db))

    # Reset shared connection so init_db() opens the tmp path this test wants.
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(prod_db))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db
    await init_db()

    import fatih_hoca
    from fatih_hoca.registry import ModelInfo, ModelRegistry
    from fatih_hoca.selector import Selector

    fatih_hoca._registry = None
    fatih_hoca._selector = None
    reg = ModelRegistry()
    reg._models["leaky_fixture"] = ModelInfo(
        name="leaky_fixture", location="local",
        provider="llama_cpp", litellm_name="openai/leaky_fixture",
        path="/fake/leaky_fixture.gguf",
        total_params_b=8.0, active_params_b=8.0,
        capabilities={c: 7.0 for c in ["reasoning","code_generation","analysis","instruction_adherence"]},
    )

    class _Nh:
        def snapshot(self):
            from nerd_herd.types import SystemSnapshot
            return SystemSnapshot(vram_available_mb=24000)
    fatih_hoca._registry = reg
    fatih_hoca._selector = Selector(registry=reg, nerd_herd=_Nh())

    pick = fatih_hoca.select(
        task="coder", agent_type="coder", difficulty=5,
        estimated_input_tokens=500, estimated_output_tokens=500,
        call_category="main_work",
    )
    assert pick is not None, "precondition: selector must make a pick for this test to meaningfully probe leak"

    import asyncio
    await asyncio.sleep(0.2)

    import sqlite3
    if not prod_db.exists():
        count = 0
    else:
        conn = sqlite3.connect(prod_db)
        try:
            try:
                cur = conn.execute(
                    "SELECT COUNT(*) FROM model_pick_log WHERE picked_model='leaky_fixture'"
                )
                count = cur.fetchone()[0]
            except sqlite3.OperationalError:
                # Table missing → no telemetry write happened → no leak.
                count = 0
        finally:
            conn.close()

    assert count == 0, (
        f"telemetry leaked: {count} rows for leaky_fixture. "
        f"tests must not write without explicit enable_telemetry() opt-in."
    )


@pytest.mark.asyncio
async def test_pick_log_records_pool_and_urgency(tmp_path, monkeypatch):
    """model_pick_log rows must have non-null pool and urgency after a successful select()."""
    import asyncio

    db_file = tmp_path / "test_pool_urgency.db"

    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(db_file))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None

    from src.infra.db import init_db
    await init_db()

    import fatih_hoca
    from fatih_hoca.registry import ModelInfo, ModelRegistry
    from fatih_hoca.selector import Selector

    fatih_hoca._registry = None
    fatih_hoca._selector = None

    reg = ModelRegistry()
    reg._models["a"] = ModelInfo(
        name="a", location="local",
        provider="llama_cpp", litellm_name="openai/a",
        path="/fake/a.gguf",
        total_params_b=8.0, active_params_b=8.0,
        capabilities={c: 7.0 for c in ["reasoning", "code_generation", "analysis", "instruction_adherence"]},
    )
    reg._models["b"] = ModelInfo(
        name="b", location="local",
        provider="llama_cpp", litellm_name="openai/b",
        path="/fake/b.gguf",
        total_params_b=8.0, active_params_b=8.0,
        capabilities={c: 5.0 for c in ["reasoning", "code_generation", "analysis", "instruction_adherence"]},
    )

    class _Nh:
        def snapshot(self):
            from nerd_herd.types import SystemSnapshot
            return SystemSnapshot(vram_available_mb=24000)

    fatih_hoca._registry = reg
    fatih_hoca._selector = Selector(registry=reg, nerd_herd=_Nh())

    from fatih_hoca import selector as _sel_mod
    monkeypatch.setattr(_sel_mod, "_telemetry_db_path", str(db_file))

    pick = fatih_hoca.select(
        task="coder", agent_type="coder", difficulty=5,
        estimated_input_tokens=500, estimated_output_tokens=500,
        call_category="main_work",
    )
    assert pick is not None, "selector must return a Pick"

    await asyncio.sleep(0.3)

    import aiosqlite
    async with aiosqlite.connect(db_file) as db:
        cur = await db.execute("SELECT pool, urgency FROM model_pick_log LIMIT 1")
        row = await cur.fetchone()

    assert row is not None, "expected at least one row in model_pick_log"
    pool_val, urgency_val = row
    assert pool_val in {"local", "time_bucketed", "per_call"}, (
        f"pool must be one of local/time_bucketed/per_call, got: {pool_val!r}"
    )
    assert urgency_val is not None, "urgency must not be NULL"
    assert 0.0 <= urgency_val <= 1.0, f"urgency must be in [0, 1], got: {urgency_val}"

    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None


def test_select_in_sync_context_skips_telemetry_cleanly(tmp_path, monkeypatch):
    """When fatih_hoca.select() runs in a pure sync context with no event loop,
    pick telemetry must be silently skipped — no crash, no DeprecationWarning,
    no DB row. Python 3.12+ will break get_event_loop() in this path.
    """
    import warnings
    import asyncio

    # Set up DB schema (if telemetry accidentally writes, we'd see it)
    from src.infra.db import init_db
    import src.infra.db as _db_mod

    db_file = tmp_path / "sync_test.db"
    monkeypatch.setattr(_db_mod, "DB_PATH", str(db_file))
    if _db_mod._db_connection is not None:
        async def _close():
            await _db_mod._db_connection.close()
        asyncio.run(_close())
        _db_mod._db_connection = None
    asyncio.run(init_db())

    # Opt telemetry IN so we'd actually see a write if the sync path leaked
    from fatih_hoca import selector as _sel_mod
    monkeypatch.setattr(_sel_mod, "_telemetry_db_path", str(db_file))

    import fatih_hoca
    from fatih_hoca.registry import ModelInfo, ModelRegistry
    from fatih_hoca.selector import Selector

    fatih_hoca._registry = None
    fatih_hoca._selector = None
    reg = ModelRegistry()
    reg._models["a"] = ModelInfo(
        name="a", location="local",
        provider="llama_cpp", litellm_name="openai/a",
        path="/fake/a.gguf",
        total_params_b=8.0, active_params_b=8.0,
        capabilities={c: 7.0 for c in ["reasoning", "code_generation", "analysis", "instruction_adherence"]},
    )
    reg._models["b"] = ModelInfo(
        name="b", location="local",
        provider="llama_cpp", litellm_name="openai/b",
        path="/fake/b.gguf",
        total_params_b=8.0, active_params_b=8.0,
        capabilities={c: 5.0 for c in ["reasoning", "code_generation", "analysis", "instruction_adherence"]},
    )

    class _Nh:
        def snapshot(self):
            from nerd_herd.types import SystemSnapshot
            return SystemSnapshot(vram_available_mb=24000)

    fatih_hoca._registry = reg
    fatih_hoca._selector = Selector(registry=reg, nerd_herd=_Nh())

    # Sync call — no asyncio.run wrapper. Promote DeprecationWarning to error
    # so we catch the get_event_loop() call.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        pick = fatih_hoca.select(
            task="coder", agent_type="coder", difficulty=5,
            estimated_input_tokens=500, estimated_output_tokens=500,
            call_category="main_work",
        )

    assert pick is not None, "select() must return a Pick even when telemetry skipped"
    assert pick.model.name in ("a", "b"), f"expected a or b, got {pick.model.name}"

    # No row written (sync context must skip telemetry)
    import sqlite3
    conn = sqlite3.connect(db_file)
    try:
        cur = conn.execute("SELECT COUNT(*) FROM model_pick_log")
        count = cur.fetchone()[0]
    finally:
        conn.close()
    assert count == 0, f"sync-context select() must not write telemetry, got {count} rows"
