# Fatih Hoca Phase 2b Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land data-free selection-intelligence infrastructure (async GC audit, scheduled benchmark refresh, `/bench_picks` telemetry view, i2p dry-run simulator) that unblocks weight-tuning work.

**Architecture:** Four independent vertical slices: (1) `asyncio.create_task` call-site audit with module-level in-flight sets, (2) new `ScheduledJobs.tick_benchmark_refresh()` heartbeat hook, (3) new `/bench_picks` handler in `telegram_bot.py`, (4) new `packages/fatih_hoca/src/fatih_hoca/simulate_i2p.py` CLI that walks 182 i2p steps through a fresh `Selector` with a pinned fake snapshot. No DB schema changes.

**Tech Stack:** Python 3.10, asyncio, aiosqlite, python-telegram-bot v20+, pytest, existing fatih_hoca package.

---

## File Structure

**Create:**
- `packages/fatih_hoca/src/fatih_hoca/simulate_i2p.py` — simulator CLI module
- `packages/fatih_hoca/tests/test_simulate_i2p.py` — simulator tests
- `tests/unit/test_scheduled_benchmark_refresh.py` — tick tests
- `tests/unit/test_async_task_gc.py` — fire-and-forget audit tests
- `tests/unit/test_bench_picks_command.py` — Telegram command tests

**Modify:**
- `packages/fatih_hoca/src/fatih_hoca/selector.py` — fix unassigned `create_task` at line 314
- `src/tools/web_search.py` — fix unassigned `create_task` at line 252
- `src/app/scheduled_jobs.py` — add `tick_benchmark_refresh()`
- `src/core/orchestrator.py` — wire new tick into heartbeat (grep for existing `tick_*` call sites)
- `src/app/telegram_bot.py` — register `/bench_picks` handler + add `cmd_bench_picks` method

---

## Task 1: Fire-and-Forget Asyncio Audit

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/selector.py:312-318`
- Modify: `src/tools/web_search.py:~252`
- Test: `tests/unit/test_async_task_gc.py`

### Step 1.1: Grep for all unassigned `create_task` call sites

- [ ] Run: `rg -n "asyncio\.create_task\(" src/ packages/ --type py`
- [ ] For each hit, inspect context. An assigned call looks like `task = asyncio.create_task(...)` or `self._tasks.add(asyncio.create_task(...))`. An **unassigned** call is `asyncio.create_task(...)` as a standalone statement.
- [ ] Record the list. Known starting set: `selector.py:314`, `web_search.py:252`. Add any others found.

### Step 1.2: Write failing test for selector pick-telemetry task retention

- [ ] **Create** `tests/unit/test_async_task_gc.py`:

```python
"""Regression tests: fire-and-forget asyncio.create_task must retain strong refs."""
import asyncio
import gc
import logging
import pytest

from fatih_hoca import selector as selector_mod


@pytest.mark.asyncio
async def test_selector_pick_telemetry_task_not_gcd(caplog, tmp_path, monkeypatch):
    """Scheduling _write() must not produce 'Task was destroyed but it is pending'."""
    db_path = tmp_path / "telemetry.db"
    import aiosqlite
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute(
            """CREATE TABLE model_pick_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                task_name TEXT, agent_type TEXT, difficulty INTEGER,
                call_category TEXT, picked_model TEXT, picked_score REAL,
                picked_reasons TEXT, candidates_json TEXT,
                failures_json TEXT, snapshot_summary TEXT
            )"""
        )
        await db.commit()
    monkeypatch.setattr(selector_mod, "_telemetry_db_path", str(db_path))

    caplog.set_level(logging.WARNING, logger="asyncio")

    # Spawn many persist calls rapidly and force GC between them.
    from fatih_hoca.selector import Selector

    class _FakeModel:
        name = "fake-model"
        is_local = False
        is_loaded = True
        load_time_seconds = 0.0

    class _FakeRanked:
        def __init__(self):
            self.model = _FakeModel()
            self.score = 7.5
            self.reasons = ["test"]

    class _FakeReqs:
        effective_task = "test_task"
        agent_type = "coder"
        difficulty = 5

    class _FakeSnapshot:
        vram_available_mb = 7000
        local = None

    sel = Selector.__new__(Selector)  # bypass __init__
    for _ in range(20):
        sel._persist_pick_telemetry(
            scored=[_FakeRanked()],
            reqs=_FakeReqs(),
            task_name="test_task",
            call_category="main_work",
            failures=[],
            snapshot=_FakeSnapshot(),
        )
        gc.collect()

    # Drain tasks
    for _ in range(5):
        await asyncio.sleep(0.05)

    bad = [r for r in caplog.records if "Task was destroyed" in r.getMessage()]
    assert bad == [], f"GC reaped pending tasks: {bad!r}"
```

- [ ] **Step 1.3:** Run failing test (expect either failure or asyncio warnings):

  Run: `timeout 30 pytest tests/unit/test_async_task_gc.py::test_selector_pick_telemetry_task_not_gcd -v`

  Expected: FAIL (either assertion failure on `bad == []`, or the test may pass inconsistently because GC timing is non-deterministic — in which case still proceed; the fix is still valid preventive work).

### Step 1.4: Implement strong-reference pattern in selector

- [ ] **Edit** `packages/fatih_hoca/src/fatih_hoca/selector.py`: add module-level set near top (after `_telemetry_db_path`):

```python
# Holds strong references to fire-and-forget telemetry tasks so GC can't
# reap them mid-flight. Tasks remove themselves via done-callback.
_pending_telemetry_tasks: set[asyncio.Task] = set()
```

Note: add `import asyncio` at module top if not already there (it's currently inside the method — move it up).

- [ ] **Replace** the tail of `_persist_pick_telemetry` (current lines ~312-318):

```python
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — sync caller. Skip silently.
            return
        task = loop.create_task(_write())
        _pending_telemetry_tasks.add(task)
        task.add_done_callback(_pending_telemetry_tasks.discard)
```

### Step 1.5: Apply the same pattern to `src/tools/web_search.py`

- [ ] Read the current `_record_fetch_quality_fire_and_forget` call site and surrounding module. Add `_pending_fetch_quality_tasks: set[asyncio.Task] = set()` at module scope.
- [ ] Replace the unassigned `asyncio.create_task(_record_fetch_quality(...))` with the same pattern:

```python
task = asyncio.create_task(_record_fetch_quality(...))
_pending_fetch_quality_tasks.add(task)
task.add_done_callback(_pending_fetch_quality_tasks.discard)
```

- [ ] If the call site is inside a sync function that has no running loop, wrap in `try: asyncio.get_running_loop() except RuntimeError: return`.

### Step 1.6: Fix any additional sites found in Step 1.1

- [ ] For each additional unassigned `asyncio.create_task(` found in Step 1.1 (outside the two known sites), apply the same pattern with a descriptively-named module-level set.

### Step 1.7: Run tests

- [ ] Run: `timeout 30 pytest tests/unit/test_async_task_gc.py -v`

  Expected: PASS. No "Task was destroyed" warnings in captured logs.

- [ ] Run: `timeout 30 pytest packages/fatih_hoca/tests/ -v`

  Expected: PASS (sanity check no regression).

### Step 1.8: Commit

```bash
git add packages/fatih_hoca/src/fatih_hoca/selector.py src/tools/web_search.py tests/unit/test_async_task_gc.py
git commit -m "fix(async): retain strong refs for fire-and-forget create_task calls"
```

---

## Task 2: Scheduled Benchmark Refresh

**Files:**
- Modify: `src/app/scheduled_jobs.py` (add `tick_benchmark_refresh()`)
- Modify: `src/core/orchestrator.py` (wire into heartbeat)
- Test: `tests/unit/test_scheduled_benchmark_refresh.py`

### Step 2.1: Locate heartbeat wiring in orchestrator

- [ ] Run: `rg -n "tick_(todos|api_discovery|digest|price_watches|benchmark)\|scheduled_jobs\." src/core/orchestrator.py`
- [ ] Identify the existing call pattern where `scheduled_jobs.tick_*()` is invoked. This is where the new tick will be plugged in.

### Step 2.2: Write failing test

- [ ] **Create** `tests/unit/test_scheduled_benchmark_refresh.py`:

```python
"""Tests for ScheduledJobs.tick_benchmark_refresh()."""
import asyncio
import time
from pathlib import Path
import pytest

from src.app.scheduled_jobs import ScheduledJobs


@pytest.fixture
def jobs(tmp_path):
    return ScheduledJobs(telegram=None)


@pytest.mark.asyncio
async def test_skips_when_cache_fresh(tmp_path, monkeypatch, jobs):
    cache_dir = tmp_path / ".benchmark_cache"
    cache_dir.mkdir()
    fresh = cache_dir / "_bulk_artificialanalysis.json"
    fresh.write_text('{"timestamp": %d, "models": {}}' % int(time.time()))

    call_count = {"n": 0}

    def _fake_refresh():
        call_count["n"] += 1

    monkeypatch.setattr(
        "src.app.scheduled_jobs._benchmark_refresh_impl",
        _fake_refresh,
    )
    monkeypatch.setattr(
        "src.app.scheduled_jobs._benchmark_cache_dir",
        lambda: cache_dir,
    )

    await jobs.tick_benchmark_refresh()
    assert call_count["n"] == 0


@pytest.mark.asyncio
async def test_refreshes_when_cache_stale(tmp_path, monkeypatch, jobs):
    cache_dir = tmp_path / ".benchmark_cache"
    cache_dir.mkdir()
    stale = cache_dir / "_bulk_artificialanalysis.json"
    # 48h old
    stale.write_text(
        '{"timestamp": %d, "models": {}}' % int(time.time() - 48 * 3600)
    )

    call_count = {"n": 0}

    def _fake_refresh():
        call_count["n"] += 1

    monkeypatch.setattr(
        "src.app.scheduled_jobs._benchmark_refresh_impl",
        _fake_refresh,
    )
    monkeypatch.setattr(
        "src.app.scheduled_jobs._benchmark_cache_dir",
        lambda: cache_dir,
    )

    await jobs.tick_benchmark_refresh()
    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_noop_when_refresh_in_flight(tmp_path, monkeypatch, jobs):
    from src.app import scheduled_jobs as sj_mod

    cache_dir = tmp_path / ".benchmark_cache"
    cache_dir.mkdir()
    (cache_dir / "_bulk_artificialanalysis.json").write_text(
        '{"timestamp": %d, "models": {}}' % int(time.time() - 48 * 3600)
    )

    monkeypatch.setattr(sj_mod, "_benchmark_cache_dir", lambda: cache_dir)

    started = asyncio.Event()
    release = asyncio.Event()

    def _slow_refresh():
        started.set()
        # Busy-wait in thread until release is set (signalled from test)
        # Thread-safe via polling
        import time as _t
        while not release.is_set():
            _t.sleep(0.01)

    monkeypatch.setattr(sj_mod, "_benchmark_refresh_impl", _slow_refresh)

    call_count = {"n": 0}
    orig_impl = _slow_refresh

    def _counted():
        call_count["n"] += 1
        orig_impl()

    monkeypatch.setattr(sj_mod, "_benchmark_refresh_impl", _counted)

    first = asyncio.create_task(jobs.tick_benchmark_refresh())
    await started.wait()
    # Second call while first still running — must noop
    await jobs.tick_benchmark_refresh()
    release.set()
    await first
    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_exception_is_swallowed(tmp_path, monkeypatch, jobs, caplog):
    import logging
    from src.app import scheduled_jobs as sj_mod

    cache_dir = tmp_path / ".benchmark_cache"
    cache_dir.mkdir()
    (cache_dir / "_bulk_artificialanalysis.json").write_text(
        '{"timestamp": %d, "models": {}}' % int(time.time() - 48 * 3600)
    )
    monkeypatch.setattr(sj_mod, "_benchmark_cache_dir", lambda: cache_dir)

    def _boom():
        raise RuntimeError("network dead")

    monkeypatch.setattr(sj_mod, "_benchmark_refresh_impl", _boom)
    caplog.set_level(logging.WARNING, logger="app.scheduled_jobs")

    await jobs.tick_benchmark_refresh()  # must not raise
    assert any("benchmark refresh failed" in r.getMessage().lower()
               for r in caplog.records)
```

### Step 2.3: Run failing test

- [ ] Run: `timeout 30 pytest tests/unit/test_scheduled_benchmark_refresh.py -v`

  Expected: FAIL with `AttributeError: 'ScheduledJobs' object has no attribute 'tick_benchmark_refresh'`.

### Step 2.4: Implement `tick_benchmark_refresh`

- [ ] **Edit** `src/app/scheduled_jobs.py`: add at module scope (below `_VALID_SUGGESTION_AGENTS`):

```python
from pathlib import Path

_BENCHMARK_FRESHNESS_HOURS = 24
_benchmark_refresh_in_flight = False


def _benchmark_cache_dir() -> Path:
    """Return the benchmark cache dir. Monkeypatched in tests."""
    return Path(".benchmark_cache")


def _benchmark_refresh_impl() -> tuple[int, int]:
    """Sync refresh via BenchmarkFetcher. Returns (before_count, after_count).

    Monkeypatched in tests.
    """
    from src.models.benchmark.benchmark_fetcher import BenchmarkFetcher

    fetcher = BenchmarkFetcher()
    before = len(fetcher.fetch_all_bulk())  # may hit cache if still fresh
    fetcher.refresh_cache()
    after = len(fetcher.fetch_all_bulk())
    return before, after


def _benchmark_cache_is_fresh() -> bool:
    import time as _time
    cache_dir = _benchmark_cache_dir()
    if not cache_dir.exists():
        return False
    bulks = list(cache_dir.glob("_bulk_*.json"))
    if not bulks:
        return False
    newest = max(b.stat().st_mtime for b in bulks)
    age_hours = (_time.time() - newest) / 3600
    return age_hours < _BENCHMARK_FRESHNESS_HOURS
```

- [ ] **Add method** to `ScheduledJobs` class (after `tick_price_watches`):

```python
async def tick_benchmark_refresh(self):
    """Refresh benchmark cache when older than 24h. Fires on heartbeat."""
    global _benchmark_refresh_in_flight

    if _benchmark_refresh_in_flight:
        logger.debug("benchmark refresh already in flight — noop")
        return

    if _benchmark_cache_is_fresh():
        logger.debug("benchmark cache fresh — skip refresh")
        return

    _benchmark_refresh_in_flight = True
    try:
        before, after = await asyncio.to_thread(_benchmark_refresh_impl)
        delta = after - before
        logger.info(
            "benchmark refresh: matched %d→%d (%+d)",
            before, after, delta,
        )
    except Exception as exc:
        logger.warning("benchmark refresh failed: %s", exc, exc_info=True)
    finally:
        _benchmark_refresh_in_flight = False
```

### Step 2.5: Run tests

- [ ] Run: `timeout 30 pytest tests/unit/test_scheduled_benchmark_refresh.py -v`

  Expected: PASS.

### Step 2.6: Wire into orchestrator heartbeat

- [ ] In `src/core/orchestrator.py`, locate where `self.scheduled_jobs.tick_api_discovery()` is called (or similar per-cycle tick). Add a sibling call, gated by a frequency check if existing ticks are gated (e.g., only fire once per hour):

```python
# Benchmark cache refresh — freshness check is internal, safe to call each cycle
try:
    await self.scheduled_jobs.tick_benchmark_refresh()
except Exception as exc:
    logger.debug("benchmark refresh tick failed: %s", exc)
```

If the heartbeat fires every 60s and `tick_benchmark_refresh()` is itself cheap when the cache is fresh (just a directory glob + mtime check), calling it every cycle is acceptable. If the existing pattern runs heavier ticks on longer intervals, follow that pattern (e.g., gate on `time.time() - self._last_benchmark_refresh_tick > 3600`).

### Step 2.7: Smoke test orchestrator import

- [ ] Run: `timeout 30 python -c "from src.core.orchestrator import Orchestrator; print('ok')"`

  Expected: prints `ok`, no import errors.

### Step 2.8: Commit

```bash
git add src/app/scheduled_jobs.py src/core/orchestrator.py tests/unit/test_scheduled_benchmark_refresh.py
git commit -m "feat(scheduler): tick_benchmark_refresh keeps benchmark cache <24h"
```

---

## Task 3: `/bench_picks` Telegram Command

**Files:**
- Modify: `src/app/telegram_bot.py` (add handler registration + `cmd_bench_picks` method)
- Test: `tests/unit/test_bench_picks_command.py`

### Step 3.1: Write failing test

- [ ] **Create** `tests/unit/test_bench_picks_command.py`:

```python
"""Tests for the /bench_picks Telegram command."""
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_bench_picks_empty(tmp_path, monkeypatch):
    import aiosqlite
    db_path = tmp_path / "test.db"
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute(
            """CREATE TABLE model_pick_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                task_name TEXT, picked_model TEXT, picked_score REAL
            )"""
        )
        await db.commit()

    monkeypatch.setenv("DB_PATH", str(db_path))

    # Import lazily so env is set first
    from src.app.telegram_bot import TelegramInterface

    bot = TelegramInterface.__new__(TelegramInterface)
    bot._reply = AsyncMock()

    update = MagicMock()
    ctx = MagicMock()
    await bot.cmd_bench_picks(update, ctx)

    reply = bot._reply.call_args
    text = reply.args[1] if len(reply.args) > 1 else reply.kwargs.get("text", "")
    combined = " ".join([str(a) for a in reply.args]) + str(reply.kwargs)
    assert "no pick log" in combined.lower() or "empty" in combined.lower() \
        or "0 entries" in combined.lower()


@pytest.mark.asyncio
async def test_bench_picks_with_rows(tmp_path, monkeypatch):
    import aiosqlite
    db_path = tmp_path / "test.db"
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute(
            """CREATE TABLE model_pick_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                task_name TEXT, picked_model TEXT, picked_score REAL
            )"""
        )
        for (task, model, score) in [
            ("coder", "qwen3-coder", 9.1),
            ("coder", "qwen3-coder", 9.2),
            ("coder", "apriel", 7.5),
            ("researcher", "gpt-oss", 8.8),
        ]:
            await db.execute(
                "INSERT INTO model_pick_log (task_name, picked_model, picked_score) VALUES (?, ?, ?)",
                (task, model, score),
            )
        await db.commit()

    monkeypatch.setenv("DB_PATH", str(db_path))

    from src.app.telegram_bot import TelegramInterface

    bot = TelegramInterface.__new__(TelegramInterface)
    bot._reply = AsyncMock()

    update = MagicMock()
    ctx = MagicMock()
    await bot.cmd_bench_picks(update, ctx)

    combined = " ".join(str(a) for a in bot._reply.call_args.args) + \
        str(bot._reply.call_args.kwargs)
    assert "qwen3-coder" in combined
    assert "coder" in combined
    assert "2" in combined  # count for qwen3-coder+coder
```

### Step 3.2: Run failing test

- [ ] Run: `timeout 30 pytest tests/unit/test_bench_picks_command.py -v`

  Expected: FAIL with `AttributeError: 'TelegramInterface' object has no attribute 'cmd_bench_picks'`.

### Step 3.3: Implement handler

- [ ] **Edit** `src/app/telegram_bot.py`:

  1. In `_setup_handlers()` (line ~1763), add next to other `CommandHandler` registrations:

  ```python
  self.app.add_handler(CommandHandler("bench_picks", self.cmd_bench_picks))
  ```

  2. Add method (anywhere among the `cmd_*` methods; near `cmd_status` is a natural place):

  ```python
  async def cmd_bench_picks(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
      """Show 7-day model pick distribution from model_pick_log."""
      import os
      import aiosqlite

      db_path = os.getenv("DB_PATH", "kutai.db")
      query = """
          SELECT task_name, picked_model, COUNT(*) AS n,
                 ROUND(AVG(picked_score), 2) AS avg_score
          FROM model_pick_log
          WHERE timestamp > datetime('now', '-7 days')
          GROUP BY task_name, picked_model
          ORDER BY task_name, n DESC
      """
      try:
          async with aiosqlite.connect(db_path) as db:
              cursor = await db.execute(query)
              rows = await cursor.fetchall()
      except Exception as exc:
          await self._reply(update, f"❌ bench_picks query failed: {exc}")
          return

      if not rows:
          await self._reply(update, "📊 No pick log entries in last 7 days.")
          return

      MAX_ROWS = 40
      truncated = len(rows) > MAX_ROWS
      rows = rows[:MAX_ROWS]

      lines = [
          f"{'task':<20} {'model':<28} {'n':>4} {'avg':>5}",
          "─" * 60,
      ]
      for task, model, n, avg in rows:
          lines.append(
              f"{(task or '?')[:20]:<20} {(model or '?')[:28]:<28} "
              f"{n:>4} {avg:>5.2f}"
          )
      body = "\n".join(lines)
      footer = "\n\n… (truncated)" if truncated else ""
      await self._reply(
          update,
          f"📊 *Model picks — last 7 days*\n```\n{body}\n```{footer}",
          parse_mode="Markdown",
      )
  ```

### Step 3.4: Run tests

- [ ] Run: `timeout 30 pytest tests/unit/test_bench_picks_command.py -v`

  Expected: PASS.

### Step 3.5: Smoke test import

- [ ] Run: `timeout 30 python -c "from src.app.telegram_bot import TelegramInterface; print('ok')"`

  Expected: `ok`.

### Step 3.6: Commit

```bash
git add src/app/telegram_bot.py tests/unit/test_bench_picks_command.py
git commit -m "feat(telegram): /bench_picks shows 7-day model_pick_log distribution"
```

---

## Task 4: i2p Dry-Run Simulator

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/simulate_i2p.py`
- Create: `packages/fatih_hoca/tests/test_simulate_i2p.py`

### Step 4.1: Write failing unit test

- [ ] **Create** `packages/fatih_hoca/tests/test_simulate_i2p.py`:

```python
"""Tests for the i2p dry-run simulator."""
import json
from pathlib import Path

import pytest

from fatih_hoca.simulate_i2p import (
    DIFFICULTY_MAP,
    simulate,
    build_report,
    _FakeNerdHerd,
)


def test_difficulty_mapping():
    assert DIFFICULTY_MAP["easy"] == 3
    assert DIFFICULTY_MAP["medium"] == 5
    assert DIFFICULTY_MAP["hard"] == 8


def test_fake_nerd_herd_snapshot_is_stable():
    nh = _FakeNerdHerd()
    s1 = nh.snapshot()
    s2 = nh.snapshot()
    assert s1.vram_available_mb == s2.vram_available_mb
    assert s1.vram_available_mb == 7000  # pinned


def test_simulate_returns_one_record_per_step(tmp_path):
    """Given a minimal workflow of 3 steps, simulator produces 3 records."""
    workflow = {
        "steps": [
            {"id": "1.1", "name": "s1", "agent": "coder", "difficulty": "easy",
             "tools_hint": ["shell"]},
            {"id": "1.2", "name": "s2", "agent": "researcher", "difficulty": "medium",
             "tools_hint": []},
            {"id": "1.3", "name": "s3", "agent": "analyst", "difficulty": "hard",
             "tools_hint": []},
        ]
    }
    workflow_path = tmp_path / "wf.json"
    workflow_path.write_text(json.dumps(workflow))

    records = simulate(workflow_path)
    assert len(records) == 3
    assert {r["step_id"] for r in records} == {"1.1", "1.2", "1.3"}
    for r in records:
        assert "picked_model" in r
        assert "picked_score" in r
        assert "agent" in r
        assert "difficulty" in r


def test_build_report_aggregates_picks():
    records = [
        {"step_id": "1", "task_name": "a", "agent": "coder", "difficulty": 5,
         "picked_model": "m1", "picked_score": 8.0, "top3": []},
        {"step_id": "2", "task_name": "b", "agent": "coder", "difficulty": 5,
         "picked_model": "m1", "picked_score": 7.5, "top3": []},
        {"step_id": "3", "task_name": "c", "agent": "researcher", "difficulty": 5,
         "picked_model": "m2", "picked_score": 9.0, "top3": []},
    ]
    report = build_report(records)
    assert report["total_steps"] == 3
    assert report["coverage"] == 3
    # m1 picked twice, m2 once
    dist = {row[0]: row[1] for row in report["distribution"]}
    assert dist["m1"] == 2
    assert dist["m2"] == 1


def test_simulate_real_i2p_smoke():
    """Smoke: simulate real i2p_v3.json, assert non-empty, most steps pick something."""
    wf = Path(__file__).resolve().parents[3] / "src" / "workflows" / "i2p" / "i2p_v3.json"
    if not wf.exists():
        pytest.skip("i2p_v3.json not present")
    records = simulate(wf)
    assert len(records) > 100  # v3 has ~182 steps
    picked = [r for r in records if r["picked_model"] != "<none>"]
    # Registry may not have perfect coverage for every agent type; require ≥60%.
    assert len(picked) / len(records) > 0.6
```

### Step 4.2: Run failing test

- [ ] Run: `timeout 30 pytest packages/fatih_hoca/tests/test_simulate_i2p.py -v`

  Expected: FAIL with `ModuleNotFoundError: No module named 'fatih_hoca.simulate_i2p'`.

### Step 4.3: Implement simulator module

- [ ] **Create** `packages/fatih_hoca/src/fatih_hoca/simulate_i2p.py`:

```python
"""Dry-run simulator: walk i2p step profiles through Selector.select().

Outputs a task × model distribution report. No DB writes: telemetry is
opt-in via enable_telemetry() and the simulator never calls it.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from fatih_hoca.registry import ModelRegistry
from fatih_hoca.selector import Selector


DIFFICULTY_MAP = {"easy": 3, "medium": 5, "hard": 8}
_DEFAULT_WORKFLOW = (
    Path(__file__).resolve().parents[4] / "src" / "workflows" / "i2p" / "i2p_v3.json"
)


@dataclass
class _FakeLocalState:
    model_name: str | None = None


@dataclass
class _FakeSnapshot:
    vram_available_mb: int = 7000
    local: _FakeLocalState | None = None

    def __post_init__(self):
        if self.local is None:
            self.local = _FakeLocalState()


class _FakeNerdHerd:
    """Returns a pinned snapshot so simulation is reproducible."""

    def __init__(self, vram_mb: int = 7000, loaded: str | None = None):
        self._snapshot = _FakeSnapshot(
            vram_available_mb=vram_mb,
            local=_FakeLocalState(model_name=loaded),
        )

    def snapshot(self):
        return self._snapshot


def _load_steps(workflow_path: Path) -> list[dict]:
    data = json.loads(workflow_path.read_text(encoding="utf-8"))
    steps = data.get("steps") or data.get("phases") or []
    if not isinstance(steps, list):
        raise ValueError(f"workflow steps is not a list: {type(steps)}")
    return steps


def simulate(workflow_path: Path | str) -> list[dict]:
    """Run each step through Selector.select() and return per-step records."""
    steps = _load_steps(Path(workflow_path))

    registry = ModelRegistry()
    try:
        registry.wait_for_enrichment()
    except Exception:
        pass

    selector = Selector(registry=registry, nerd_herd=_FakeNerdHerd())

    records: list[dict] = []
    for step in steps:
        difficulty_raw = step.get("difficulty", "medium")
        difficulty = DIFFICULTY_MAP.get(
            difficulty_raw if isinstance(difficulty_raw, str) else "medium",
            5,
        )
        agent_type = step.get("agent", "") or ""
        task_name = step.get("name", step.get("id", "unknown"))

        pick = selector.select(
            task=task_name,
            agent_type=agent_type,
            difficulty=difficulty,
            call_category="main_work",
        )

        if pick is None:
            records.append({
                "step_id": step.get("id", ""),
                "task_name": task_name,
                "agent": agent_type,
                "difficulty": difficulty,
                "picked_model": "<none>",
                "picked_score": 0.0,
                "top3": [],
            })
        else:
            records.append({
                "step_id": step.get("id", ""),
                "task_name": task_name,
                "agent": agent_type,
                "difficulty": difficulty,
                "picked_model": pick.model.name,
                "picked_score": round(getattr(pick.model, "score", 0.0) or 0.0, 2),
                "top3": [],  # Pick doesn't carry scored list; leave empty for MVP
            })
    return records


def build_report(records: list[dict]) -> dict:
    total = len(records)
    covered = sum(1 for r in records if r["picked_model"] != "<none>")

    pick_counter: Counter[str] = Counter(r["picked_model"] for r in records)
    by_agent: dict[str, Counter] = {}
    by_difficulty: dict[int, Counter] = {}
    for r in records:
        by_agent.setdefault(r["agent"] or "?", Counter())[r["picked_model"]] += 1
        by_difficulty.setdefault(r["difficulty"], Counter())[r["picked_model"]] += 1

    distribution = sorted(pick_counter.items(), key=lambda x: -x[1])
    agent_top = sorted(
        (agent, c.most_common(1)[0][0] if c else "<none>", sum(c.values()))
        for agent, c in by_agent.items()
    )
    difficulty_top = sorted(
        (d, c.most_common(1)[0][0] if c else "<none>", sum(c.values()))
        for d, c in by_difficulty.items()
    )
    return {
        "total_steps": total,
        "coverage": covered,
        "distribution": distribution,
        "by_agent": agent_top,
        "by_difficulty": difficulty_top,
    }


def _format_report(report: dict) -> str:
    lines = [
        f"Total steps: {report['total_steps']}",
        f"Covered: {report['coverage']} "
        f"({report['coverage'] * 100 / max(report['total_steps'], 1):.0f}%)",
        "",
        "— Pick distribution —",
        f"{'model':<40} {'count':>6} {'pct':>6}",
    ]
    total = max(report['total_steps'], 1)
    for model, count in report["distribution"]:
        lines.append(f"{model[:40]:<40} {count:>6} {count * 100 / total:>5.1f}%")
    lines += ["", "— By agent —", f"{'agent':<20} {'top_model':<40} {'n':>4}"]
    for agent, top, n in report["by_agent"]:
        lines.append(f"{agent[:20]:<20} {top[:40]:<40} {n:>4}")
    lines += ["", "— By difficulty —", f"{'diff':>4} {'top_model':<40} {'n':>4}"]
    for d, top, n in report["by_difficulty"]:
        lines.append(f"{d:>4} {top[:40]:<40} {n:>4}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", default=str(_DEFAULT_WORKFLOW),
                        help="Path to workflow JSON (default: i2p_v3.json)")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="Write per-step records as JSON array to this path")
    args = parser.parse_args(argv)

    records = simulate(args.workflow)
    report = build_report(records)
    print(_format_report(report))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(records, indent=2))
        print(f"\nWrote {len(records)} records to {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Step 4.4: Run tests

- [ ] Run: `timeout 60 pytest packages/fatih_hoca/tests/test_simulate_i2p.py -v`

  Expected: PASS. The smoke test (`test_simulate_real_i2p_smoke`) may take longer because it touches the real registry.

### Step 4.5: Smoke the CLI manually

- [ ] Run: `timeout 60 python -m fatih_hoca.simulate_i2p`

  Expected: stdout prints a distribution report. No traceback. Exit 0.

- [ ] Run: `timeout 60 python -m fatih_hoca.simulate_i2p --json /tmp/i2p_sim.json && ls -la /tmp/i2p_sim.json`

  Expected: JSON file exists, contains ~182 records.

### Step 4.6: Commit

```bash
git add packages/fatih_hoca/src/fatih_hoca/simulate_i2p.py packages/fatih_hoca/tests/test_simulate_i2p.py
git commit -m "feat(fatih-hoca): i2p dry-run simulator for selection distribution previews"
```

---

## Final Verification

### Step F.1: Full fatih_hoca test suite

- [ ] Run: `timeout 120 pytest packages/fatih_hoca/tests/ -v`

  Expected: all pass.

### Step F.2: Targeted new-tests suite

- [ ] Run: `timeout 60 pytest tests/unit/test_async_task_gc.py tests/unit/test_scheduled_benchmark_refresh.py tests/unit/test_bench_picks_command.py -v`

  Expected: all pass.

### Step F.3: Import smoke test

- [ ] Run: `timeout 30 python -c "from src.core.orchestrator import Orchestrator; from src.app.telegram_bot import TelegramInterface; from fatih_hoca.simulate_i2p import simulate; print('ok')"`

  Expected: `ok`.

### Step F.4: Push worktree branch

```bash
git push -u origin feat/fatih-hoca-phase2b
```

(User will merge to main when ready.)

---

## Success Criteria Recap

- No unassigned `asyncio.create_task(` calls remain in `src/` or `packages/`.
- `.benchmark_cache/_bulk_*.json` stays under 24h old during normal orchestrator uptime.
- `/bench_picks` in Telegram returns a monospace table of last-7-day model picks.
- `python -m fatih_hoca.simulate_i2p` produces a reproducible report covering ≥60% of i2p_v3 steps.
- All existing tests still pass.
