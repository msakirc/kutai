# Yalayut Demand-Signal Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the 6 dead Yalayut demand-signal types to production firing sites and add an autonomous trigger that drains accumulated signals into on-demand discovery runs.

**Architecture:** The demand-signal subsystem (`packages/yalayut/.../discovery/demand.py`) is fully built and tested but only ever fires from one site (`/yalayut discover`, type `founder`). This plan adds a firing site for each remaining type and a periodic drain. Core-loop files (`general_beckman/apply.py`) never import `yalayut` directly — the `dlq` signal routes through a new mechanical executor. `intersect/flash.py` and `coulson/react.py` already lazy-import `yalayut`, so they fire directly. The drain folds into the existing `yalayut_discovery` daily executor (no new orchestrator method, no new cadence row).

**Tech Stack:** Python 3.10, async/await, aiosqlite, pytest. Packages: `yalayut`, `intersect`, `coulson`, `general_beckman`, `mr_roboto`.

---

## Deviations from the handoff (`docs/handoff/2026-05-16-yalayut-demand-signals-gap.md`)

The handoff is a sketch. Two deliberate departures, both grounded in code read during planning:

1. **`planning_miss` fires from `intersect/flash.py`, not the expander.** The expander's `expand_steps_to_tasks` is **synchronous**; calling the async `yalayut.query()` there would mean a second full catalog query per step for zero new information — `flash.py` already runs that exact query at dispatch. Instead, `flash.py` distinguishes the two proactive misses by whether the step declared a `recipe_hint` (planner explicitly expected catalog help → `planning_miss`) vs not (generic → `step_entry_miss`). Same site, mutually-exclusive signal type.

2. **`repeat_pattern` is derived inside the Unit B drain**, not a hot-path call. It scans `yalayut_demand_signals` for patterns that already accumulated ≥3 un-discovered signals of other types and records one amplifying `repeat_pattern` signal. This is the "cheapest to compute from existing rows" option the handoff itself names.

Net: 5 production call sites (flash fires 2 types) + 1 derived signal in the drain = all 7 wired.

---

## File Structure

**Created:**
- `packages/mr_roboto/src/mr_roboto/executors/yalayut_demand.py` — mechanical executor: records one demand signal so `apply.py` need not import `yalayut`.
- `packages/yalayut/src/yalayut/discovery/demand_drain.py` — the autonomous drain: `run_demand_drain()` + `_scan_repeat_patterns()`.
- `tests/yalayut/test_demand_wiring.py` — tests for every firing site + the drain.

**Modified:**
- `packages/yalayut/src/yalayut/discovery/demand.py` — add `DEMAND_DISCOVERY_THRESHOLD` constant + `record(...)` kwargs convenience.
- `packages/yalayut/src/yalayut/discovery/source_scout.py` — replace hardcoded `0.5` with the constant.
- `packages/yalayut/src/yalayut/__init__.py` — export `record_demand_signal`, `run_demand_drain`.
- `packages/intersect/src/intersect/flash.py` — fire `step_entry_miss` / `planning_miss` on empty `query()`.
- `packages/coulson/src/coulson/react.py` — fire `tool_call` on an unresolved tool request.
- `packages/yalayut/src/yalayut/capture.py` — fire `hint_miss` on a repeat (upsert) capture.
- `packages/general_beckman/src/general_beckman/apply.py` — enqueue a `yalayut_demand` mechanical task in `_dlq_write`.
- `packages/mr_roboto/src/mr_roboto/__init__.py` — route the `yalayut_demand` action.
- `packages/mr_roboto/src/mr_roboto/executors/yalayut_discovery.py` — daily mode also runs the drain.

---

## Testing notes (read before Task 1)

- **Always prefix pytest with a timeout** (CLAUDE.md rule): `timeout 60 pytest tests/yalayut/test_demand_wiring.py -v`.
- **DB fixture:** the `yalayut_demand_signals` table already exists in the schema. Tests need an initialized temp DB. Follow the fixture pattern already used in `tests/yalayut/test_phase4_cron_discovery.py` — `init_db()` against a temp `DB_PATH`. If that file uses a shared `conftest.py` fixture, reuse it; otherwise copy its setup.
- **Cooldown gotcha:** `record_signal()` dedupes the same `(source_step_pattern, signal_type)` within 7 days, returning `-1`. Tests that record twice must use distinct patterns or assert `-1` on the second call.
- **Async tests:** use `@pytest.mark.asyncio` consistent with the existing yalayut test files.

---

## Task 1: Demand convenience API + shared threshold constant

**Files:**
- Modify: `packages/yalayut/src/yalayut/discovery/demand.py`
- Modify: `packages/yalayut/src/yalayut/discovery/source_scout.py:107`
- Modify: `packages/yalayut/src/yalayut/__init__.py`
- Test: `tests/yalayut/test_demand_wiring.py`

- [ ] **Step 1: Write the failing test**

Create `tests/yalayut/test_demand_wiring.py`. Replicate the DB-init fixture from `tests/yalayut/test_phase4_cron_discovery.py` (a `db` fixture that calls `init_db()` against a temp path). Then:

```python
import pytest

from yalayut.discovery import demand as _demand


@pytest.mark.asyncio
async def test_record_helper_inserts_row(db):
    row_id = await _demand.record(
        source_step_pattern="test:helper-pattern",
        intent_keywords=["pdf", "extract"],
        signal_type="tool_call",
        confidence=0.4,
    )
    assert row_id > 0
    stacked = await _demand.stack_confidence("test:helper-pattern")
    assert stacked == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_threshold_constant_is_half():
    assert _demand.DEMAND_DISCOVERY_THRESHOLD == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 60 pytest tests/yalayut/test_demand_wiring.py -v`
Expected: FAIL — `AttributeError: module 'yalayut.discovery.demand' has no attribute 'record'` (and `DEMAND_DISCOVERY_THRESHOLD`).

- [ ] **Step 3: Add the constant and helper to `demand.py`**

In `packages/yalayut/src/yalayut/discovery/demand.py`, after the `COOLDOWN_SECONDS` line (line 28):

```python
#: Stacked-confidence threshold a pattern must cross before an autonomous
#: on-demand discovery run is triggered for it. Shared with source_scout's
#: demand web-search scan so both autonomy paths agree on "high confidence".
DEMAND_DISCOVERY_THRESHOLD: float = 0.5
```

At the end of the same file, after `mark_discovered`:

```python
async def record(
    *,
    source_step_pattern: str,
    intent_keywords: list[str],
    signal_type: str,
    confidence: float = 0.3,
) -> int:
    """Kwargs convenience over ``record_signal`` — build a ``DemandSignal``
    and insert it. Returns the new row id, or ``-1`` when deduped. Lets a
    firing site fire one signal without importing ``DemandSignal`` itself."""
    return await record_signal(DemandSignal(
        source_step_pattern=source_step_pattern,
        intent_keywords=list(intent_keywords),
        signal_type=signal_type,
        confidence=confidence,
    ))
```

- [ ] **Step 4: Refactor `source_scout.py` to use the constant**

In `packages/yalayut/src/yalayut/discovery/source_scout.py`, inside `_scan_demand_websearch` (line ~103-107), the import already pulls `demand as _demand`. Replace:

```python
        if sig["stacked_confidence"] < 0.5:
```

with:

```python
        if sig["stacked_confidence"] < _demand.DEMAND_DISCOVERY_THRESHOLD:
```

- [ ] **Step 5: Export `record_demand_signal` from the package API**

In `packages/yalayut/src/yalayut/__init__.py`, add `"record_demand_signal"` to `__all__`, then add this function (place it after `capture_hint`):

```python
async def record_demand_signal(
    *,
    source_step_pattern: str,
    intent_keywords: list[str],
    signal_type: str,
    confidence: float = 0.3,
) -> int:
    """Public API — record one demand signal. Firing sites call this."""
    from yalayut.discovery.demand import record as _impl
    return await _impl(
        source_step_pattern=source_step_pattern,
        intent_keywords=intent_keywords,
        signal_type=signal_type,
        confidence=confidence,
    )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `timeout 60 pytest tests/yalayut/test_demand_wiring.py -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Verify source_scout still imports and its tests pass**

Run: `python -c "from yalayut.discovery import source_scout"` then `timeout 60 pytest tests/yalayut/ -k "scout or demand" -v`
Expected: imports clean, no regressions.

- [ ] **Step 8: Commit**

```bash
git add packages/yalayut/src/yalayut/discovery/demand.py packages/yalayut/src/yalayut/discovery/source_scout.py packages/yalayut/src/yalayut/__init__.py tests/yalayut/test_demand_wiring.py
git commit -m "feat(yalayut): add demand record() helper + shared DEMAND_DISCOVERY_THRESHOLD"
```

---

## Task 2: Fire `step_entry_miss` / `planning_miss` from `intersect.flash`

**Files:**
- Modify: `packages/intersect/src/intersect/flash.py:134-136`
- Test: `tests/yalayut/test_demand_wiring.py`

`flash.py` already lazy-imports `yalayut` (line 132). The `if not candidates: return task` branch (lines 135-136) is the empty-catalog site. Fire `planning_miss` when the step declared a `recipe_hint` (planner expected a catalog match), else `step_entry_miss`.

- [ ] **Step 1: Write the failing test**

Append to `tests/yalayut/test_demand_wiring.py`:

```python
from unittest.mock import patch


@pytest.mark.asyncio
async def test_flash_empty_query_fires_step_entry_miss(db):
    from intersect import flash as _flash

    task = {"id": 901, "title": "Parse the invoice CSV", "description": "",
            "context": {}}
    with patch("yalayut.query", new=_async_return([])):
        await _flash.flash(task)

    rows = await _demand.pending_signals(limit=50)
    hit = [r for r in rows if "Parse the invoice CSV"[:40] in r["source_step_pattern"]]
    assert hit, "step_entry_miss signal not recorded"
    # type check via raw row
    dbc = await _get_db_for_test()
    cur = await dbc.execute(
        "SELECT signal_type FROM yalayut_demand_signals "
        "WHERE source_step_pattern LIKE ?", ("%Parse the invoice CSV%",))
    types = {r[0] for r in await cur.fetchall()}
    assert types == {"step_entry_miss"}


@pytest.mark.asyncio
async def test_flash_empty_query_with_recipe_hint_fires_planning_miss(db):
    from intersect import flash as _flash

    task = {"id": 902, "title": "Send a Slack notification",
            "description": "", "context": {"recipe_hint": "slack"}}
    with patch("yalayut.query", new=_async_return([])):
        await _flash.flash(task)

    dbc = await _get_db_for_test()
    cur = await dbc.execute(
        "SELECT signal_type FROM yalayut_demand_signals "
        "WHERE source_step_pattern LIKE ?", ("%Send a Slack notification%",))
    types = {r[0] for r in await cur.fetchall()}
    assert types == {"planning_miss"}
```

Add these helpers near the top of the test file (after imports):

```python
from src.infra.db import get_db as _get_db_for_test


def _async_return(value):
    async def _fn(*args, **kwargs):
        return value
    return _fn
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 60 pytest tests/yalayut/test_demand_wiring.py -k flash -v`
Expected: FAIL — no rows recorded (`assert hit`).

- [ ] **Step 3: Implement the firing site in `flash.py`**

In `packages/intersect/src/intersect/flash.py`, replace lines 134-136:

```python
        candidates = await yalayut.query(task)
        if not candidates:
            return task
```

with:

```python
        candidates = await yalayut.query(task)
        if not candidates:
            await _fire_miss_signal(task, ctx)
            return task
```

Then add this helper function above `flash` (after `_slot_key`, line ~117):

```python
async def _fire_miss_signal(task: dict, ctx: dict) -> None:
    """Record a proactive demand miss when the catalog returns nothing.

    ``planning_miss`` when the step declared a ``recipe_hint`` (the planner
    expected catalog help and got none); ``step_entry_miss`` otherwise.
    Best-effort — a signal failure must never disturb dispatch.
    """
    try:
        import yalayut
        title = (task.get("title") or "").strip()
        if not title:
            return
        sig_type = "planning_miss" if ctx.get("recipe_hint") else "step_entry_miss"
        keywords = [w for w in (title + " "
                    + (task.get("description") or "")).split() if len(w) > 2]
        await yalayut.record_demand_signal(
            source_step_pattern=f"{sig_type}:{title[:40]}",
            intent_keywords=keywords[:12],
            signal_type=sig_type,
            confidence=0.3,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("demand miss signal skipped: %s", exc)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 60 pytest tests/yalayut/test_demand_wiring.py -k flash -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Verify flash still degrades gracefully**

Run: `timeout 60 pytest tests/intersect/ -v`
Expected: no regressions in the intersect suite.

- [ ] **Step 6: Commit**

```bash
git add packages/intersect/src/intersect/flash.py tests/yalayut/test_demand_wiring.py
git commit -m "feat(intersect): fire step_entry_miss/planning_miss demand signal on empty catalog query"
```

---

## Task 3: Fire `tool_call` from `coulson.react` on an unresolved tool

**Files:**
- Modify: `packages/coulson/src/coulson/react.py:1120-1124`
- Test: `tests/yalayut/test_demand_wiring.py`

`react.py:1120` is the `else` branch reached when a tool is not in `TOOL_REGISTRY` and not a yalayut api/mcp tool — the agent asked for a capability with no backing. `react.py` already lazy-imports `yalayut` (line 1112), so a direct call is consistent.

- [ ] **Step 1: Write the failing test**

Append to `tests/yalayut/test_demand_wiring.py`:

```python
@pytest.mark.asyncio
async def test_react_unresolved_tool_fires_tool_call_signal(db):
    from coulson import react as _react

    await _react._fire_tool_call_signal("scrape_pdf_table", task_id=903)

    dbc = await _get_db_for_test()
    cur = await dbc.execute(
        "SELECT signal_type, intent_keywords_json FROM yalayut_demand_signals "
        "WHERE source_step_pattern = ?", ("tool_call:scrape_pdf_table",))
    rows = await cur.fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "tool_call"
    assert "scrape_pdf_table" in rows[0][1]
```

> Note: the test exercises the extracted helper directly because a full
> `react()` loop needs an LLM. Step 3 both creates the helper AND calls it
> from the unresolved-tool branch, so the production path is genuinely wired.

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 60 pytest tests/yalayut/test_demand_wiring.py -k react -v`
Expected: FAIL — `AttributeError: module 'coulson.react' has no attribute '_fire_tool_call_signal'`.

- [ ] **Step 3: Implement the helper and wire it into the unresolved-tool branch**

In `packages/coulson/src/coulson/react.py`, add this module-level helper (place it near the other module-level helpers, above the main loop):

```python
async def _fire_tool_call_signal(tool_name: str, *, task_id: int | None = None) -> None:
    """Record a ``tool_call`` demand signal — an agent requested a tool with
    no backing skill/tool in any registry. Best-effort: a signal failure must
    never affect the agent loop."""
    try:
        import yalayut
        await yalayut.record_demand_signal(
            source_step_pattern=f"tool_call:{tool_name}",
            intent_keywords=[tool_name],
            signal_type="tool_call",
            confidence=0.3,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("tool_call demand signal skipped (task %s): %s",
                     task_id, exc)
```

Then in the `else` branch at line ~1120-1124, change:

```python
                else:
                    tool_output = (
                        f"❌ Unknown tool '{tool_name}'. "
                        f"Available: {list(TOOL_REGISTRY.keys())}"
                    )
```

to:

```python
                else:
                    await _fire_tool_call_signal(tool_name, task_id=task_id)
                    tool_output = (
                        f"❌ Unknown tool '{tool_name}'. "
                        f"Available: {list(TOOL_REGISTRY.keys())}"
                    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 60 pytest tests/yalayut/test_demand_wiring.py -k react -v`
Expected: PASS.

- [ ] **Step 5: Verify react still imports**

Run: `python -c "from coulson import react"`
Expected: clean import, no error.

- [ ] **Step 6: Commit**

```bash
git add packages/coulson/src/coulson/react.py tests/yalayut/test_demand_wiring.py
git commit -m "feat(coulson): fire tool_call demand signal on unresolved tool request"
```

---

## Task 4: Fire `hint_miss` from `yalayut.capture` on a repeat capture

**Files:**
- Modify: `packages/yalayut/src/yalayut/capture.py:65-69`
- Test: `tests/yalayut/test_demand_wiring.py`

`capture_hint` upserts an `internal_hint` artifact. The `if existing:` branch (line 65) means the *same pattern was captured before* — a repeated internal derivation. That recurrence is exactly the `hint_miss` signal: a reusable *external* skill would beat re-deriving this. `capture.py` is inside the `yalayut` package, so it imports `demand` directly.

- [ ] **Step 1: Write the failing test**

Append to `tests/yalayut/test_demand_wiring.py`:

```python
@pytest.mark.asyncio
async def test_capture_repeat_fires_hint_miss(db):
    from yalayut.capture import capture_hint

    task = {"title": "Retry flaky HTTP with backoff",
            "description": "wrap requests in exponential backoff"}
    outcome = {"status": "completed", "iterations": 3}

    # First capture — inserts, no hint_miss.
    await capture_hint(task, outcome)
    dbc = await _get_db_for_test()
    cur = await dbc.execute("SELECT COUNT(*) FROM yalayut_demand_signals "
                            "WHERE signal_type = 'hint_miss'")
    assert (await cur.fetchone())[0] == 0

    # Second capture of the same task — upsert path → hint_miss fires.
    await capture_hint(task, outcome)
    cur = await dbc.execute("SELECT source_step_pattern FROM yalayut_demand_signals "
                            "WHERE signal_type = 'hint_miss'")
    rows = await cur.fetchall()
    assert len(rows) == 1
    assert rows[0][0].startswith("hint_miss:internal-")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 60 pytest tests/yalayut/test_demand_wiring.py -k capture -v`
Expected: FAIL — second assert finds 0 `hint_miss` rows.

- [ ] **Step 3: Implement the firing site in `capture.py`**

In `packages/yalayut/src/yalayut/capture.py`, the `if existing:` branch (lines 65-69). After the `UPDATE` statement, fire the signal. Replace:

```python
    if existing:
        await db.execute(
            "UPDATE yalayut_index SET body_excerpt = ?, embedding = ?, "
            "vetted_at = ? WHERE id = ?",
            (body[:500], embedding_blob, now, existing[0]))
```

with:

```python
    if existing:
        await db.execute(
            "UPDATE yalayut_index SET body_excerpt = ?, embedding = ?, "
            "vetted_at = ? WHERE id = ?",
            (body[:500], embedding_blob, now, existing[0]))
        # Repeat capture of the same pattern — a reusable EXTERNAL skill would
        # beat re-deriving this internally. Record a reactive hint_miss signal.
        try:
            from yalayut.discovery.demand import record as _record_demand
            await _record_demand(
                source_step_pattern=f"hint_miss:{name}",
                intent_keywords=[w for w in title.split() if len(w) > 2][:12],
                signal_type="hint_miss",
                confidence=0.3,
            )
        except Exception as exc:  # noqa: BLE001 — capture must never crash
            logger.debug("hint_miss demand signal skipped: %s", exc)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 60 pytest tests/yalayut/test_demand_wiring.py -k capture -v`
Expected: PASS.

- [ ] **Step 5: Verify existing capture tests still pass**

Run: `timeout 60 pytest tests/yalayut/ -k capture -v`
Expected: no regressions.

- [ ] **Step 6: Commit**

```bash
git add packages/yalayut/src/yalayut/capture.py tests/yalayut/test_demand_wiring.py
git commit -m "feat(yalayut): fire hint_miss demand signal on repeat internal_hint capture"
```

---

## Task 5: Fire `dlq` via a new `yalayut_demand` mechanical executor

**Files:**
- Create: `packages/mr_roboto/src/mr_roboto/executors/yalayut_demand.py`
- Modify: `packages/mr_roboto/src/mr_roboto/__init__.py:4744` (after the `capture_hint` action block)
- Modify: `packages/general_beckman/src/general_beckman/apply.py` (`_dlq_write`, line ~505-556)
- Test: `tests/yalayut/test_demand_wiring.py`

`apply.py` is a core-loop file and must NOT import `yalayut`. It already enqueues a mechanical `notify_user` task via `add_task(..., context=_mechanical_context(...))`. The `dlq` signal uses the same mechanism with a new `yalayut_demand` action.

- [ ] **Step 1: Write the failing test for the executor**

Append to `tests/yalayut/test_demand_wiring.py`:

```python
@pytest.mark.asyncio
async def test_yalayut_demand_executor_records_signal(db):
    from mr_roboto.executors.yalayut_demand import run

    task = {"payload": {
        "action": "yalayut_demand",
        "source_step_pattern": "dlq:task-555",
        "intent_keywords": ["migrate", "schema"],
        "signal_type": "dlq",
        "confidence": 0.3,
    }}
    res = await run(task)
    assert res["ok"] is True

    dbc = await _get_db_for_test()
    cur = await dbc.execute(
        "SELECT signal_type FROM yalayut_demand_signals "
        "WHERE source_step_pattern = ?", ("dlq:task-555",))
    rows = await cur.fetchall()
    assert len(rows) == 1 and rows[0][0] == "dlq"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 60 pytest tests/yalayut/test_demand_wiring.py -k demand_executor -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mr_roboto.executors.yalayut_demand'`.

- [ ] **Step 3: Create the executor**

Create `packages/mr_roboto/src/mr_roboto/executors/yalayut_demand.py`:

```python
"""Yalayut demand-signal mechanical executor.

Lets core-loop files (general_beckman/apply.py) record a demand signal
WITHOUT importing yalayut — they enqueue a mechanical task with
action "yalayut_demand" and this leaf shim does the import.
"""
from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.yalayut_demand")


async def run(task: dict[str, Any]) -> dict[str, Any]:
    payload = task.get("payload") or {}
    pattern = payload.get("source_step_pattern") or ""
    if not pattern:
        return {"ok": False, "reason": "yalayut_demand needs source_step_pattern"}
    import yalayut
    try:
        row_id = await yalayut.record_demand_signal(
            source_step_pattern=pattern,
            intent_keywords=payload.get("intent_keywords") or [],
            signal_type=payload.get("signal_type") or "dlq",
            confidence=float(payload.get("confidence", 0.3)),
        )
        return {"ok": True, "row_id": row_id}
    except Exception as e:  # noqa: BLE001 — a signal failure must not DLQ
        logger.warning("yalayut_demand executor failed: %s", e)
        return {"ok": True, "recorded": False, "reason": str(e)}
```

- [ ] **Step 4: Route the action in `mr_roboto/__init__.py`**

In `packages/mr_roboto/src/mr_roboto/__init__.py`, after the `capture_hint` action block (line ~4744, before the final `return Action(status="failed", ...)`):

```python
    if action == "yalayut_demand":
        from mr_roboto.executors.yalayut_demand import run as _yal_demand_run
        res = await _yal_demand_run(task)
        return Action(status="completed", result=res)
```

- [ ] **Step 5: Run executor test to verify it passes**

Run: `timeout 60 pytest tests/yalayut/test_demand_wiring.py -k demand_executor -v`
Expected: PASS.

- [ ] **Step 6: Write the failing test for the `apply.py` DLQ firing**

Append to `tests/yalayut/test_demand_wiring.py`:

```python
@pytest.mark.asyncio
async def test_dlq_write_enqueues_yalayut_demand_task(db, monkeypatch):
    from general_beckman import apply as _apply

    enqueued = []

    async def _fake_add_task(**kwargs):
        enqueued.append(kwargs)
        return 1

    async def _fake_update_task(*a, **k):
        return None

    async def _fake_quarantine(**k):
        return None

    monkeypatch.setattr("src.infra.db.add_task", _fake_add_task)
    monkeypatch.setattr("src.infra.db.update_task", _fake_update_task)
    monkeypatch.setattr("src.infra.dead_letter.quarantine_task", _fake_quarantine)

    task = {"id": 777, "title": "Convert HEIC images to PNG",
            "agent_type": "executor", "mission_id": None}
    await _apply._dlq_write(task, error="all attempts failed",
                            category="exhausted", attempts=5)

    demand_tasks = [
        e for e in enqueued
        if (e.get("context") or {}).get("payload", {}).get("action")
        == "yalayut_demand"
    ]
    assert len(demand_tasks) == 1
    p = demand_tasks[0]["context"]["payload"]
    assert p["signal_type"] == "dlq"
    assert p["source_step_pattern"] == "dlq:777"
```

- [ ] **Step 7: Run it to verify it fails**

Run: `timeout 60 pytest tests/yalayut/test_demand_wiring.py -k dlq_write -v`
Expected: FAIL — `assert len(demand_tasks) == 1` is 0.

- [ ] **Step 8: Fire the signal from `_dlq_write`**

In `packages/general_beckman/src/general_beckman/apply.py`, inside `_dlq_write`, after the Telegram notification `add_task(...)` block (the one titled `f"Notify: DLQ task #{task['id']}"`, ending line ~556), append:

```python
    # Yalayut demand signal: a DLQ'd task is unmet demand — its intent could
    # not be satisfied. Record it (reactive `dlq` signal) WITHOUT importing
    # yalayut into this core-loop file; route through a mechanical task.
    _title = (task.get("title") or "").strip()
    if _title:
        await add_task(
            title=f"Demand signal: DLQ #{task['id']}",
            description="",
            agent_type="mechanical",
            mission_id=task.get("mission_id"),
            context=_mechanical_context(
                "yalayut_demand",
                source_step_pattern=f"dlq:{task['id']}",
                intent_keywords=[w for w in _title.split() if len(w) > 2][:12],
                signal_type="dlq",
                confidence=0.3,
            ),
            depends_on=[],
        )
```

- [ ] **Step 9: Run the DLQ test to verify it passes**

Run: `timeout 60 pytest tests/yalayut/test_demand_wiring.py -k dlq_write -v`
Expected: PASS.

- [ ] **Step 10: Verify beckman + mr_roboto suites still pass**

Run: `timeout 120 pytest tests/general_beckman/ tests/mr_roboto/ -v`
Expected: no regressions. (If `tests/mr_roboto/` does not exist, run `tests/general_beckman/` only and `python -c "import mr_roboto"`.)

- [ ] **Step 11: Commit**

```bash
git add packages/mr_roboto/src/mr_roboto/executors/yalayut_demand.py packages/mr_roboto/src/mr_roboto/__init__.py packages/general_beckman/src/general_beckman/apply.py tests/yalayut/test_demand_wiring.py
git commit -m "feat(beckman): fire dlq demand signal via yalayut_demand mechanical executor"
```

---

## Task 6: The autonomous drain — `run_demand_drain` + `repeat_pattern`

**Files:**
- Create: `packages/yalayut/src/yalayut/discovery/demand_drain.py`
- Modify: `packages/yalayut/src/yalayut/__init__.py`
- Test: `tests/yalayut/test_demand_wiring.py`

The drain derives the `repeat_pattern` signal, then drains every pattern at or above `DEMAND_DISCOVERY_THRESHOLD` through `on_demand_discovery` (which itself calls `mark_discovered`).

- [ ] **Step 1: Write the failing test**

Append to `tests/yalayut/test_demand_wiring.py`:

```python
@pytest.mark.asyncio
async def test_repeat_pattern_scan_amplifies_recurring_pattern(db):
    from yalayut.discovery.demand_drain import _scan_repeat_patterns

    # Same pattern, 3 distinct signal types — a recurrence.
    for st in ("step_entry_miss", "tool_call", "dlq"):
        await _demand.record(source_step_pattern="recur:pdf-parse",
                             intent_keywords=["pdf"], signal_type=st,
                             confidence=0.3)
    added = await _scan_repeat_patterns()
    assert added == 1

    dbc = await _get_db_for_test()
    cur = await dbc.execute(
        "SELECT COUNT(*) FROM yalayut_demand_signals "
        "WHERE source_step_pattern = 'recur:pdf-parse' "
        "AND signal_type = 'repeat_pattern'")
    assert (await cur.fetchone())[0] == 1


@pytest.mark.asyncio
async def test_run_demand_drain_triggers_discovery_above_threshold(db, monkeypatch):
    from yalayut.discovery import demand_drain

    # Stack a pattern over 0.5: two 0.3 signals → 1-(0.7*0.7)=0.51.
    await _demand.record(source_step_pattern="drain:slack-bot",
                         intent_keywords=["slack"], signal_type="tool_call",
                         confidence=0.3)
    await _demand.record(source_step_pattern="drain:slack-bot",
                         intent_keywords=["slack"], signal_type="dlq",
                         confidence=0.3)

    discovered = []

    async def _fake_on_demand(demand):
        discovered.append(demand["source_step_pattern"])
        await _demand.mark_discovered(demand["source_step_pattern"])
        return {"pattern": demand["source_step_pattern"], "artifacts_ingested": 0}

    monkeypatch.setattr("yalayut.discovery.on_demand.on_demand_discovery",
                        _fake_on_demand)

    summary = await demand_drain.run_demand_drain()
    assert "drain:slack-bot" in discovered
    assert summary["patterns_discovered"] >= 1

    # Drained → no longer pending.
    pending = await _demand.pending_signals(limit=50)
    assert all(p["source_step_pattern"] != "drain:slack-bot" for p in pending)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `timeout 60 pytest tests/yalayut/test_demand_wiring.py -k "repeat_pattern or demand_drain" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'yalayut.discovery.demand_drain'`.

- [ ] **Step 3: Create `demand_drain.py`**

Create `packages/yalayut/src/yalayut/discovery/demand_drain.py`:

```python
"""Yalayut Phase 4 — autonomous demand-signal drain.

``run_demand_drain()`` is the autonomous trigger the demand subsystem was
missing: it derives the ``repeat_pattern`` signal, then for every pattern
whose stacked confidence crosses ``DEMAND_DISCOVERY_THRESHOLD`` it runs an
on-demand discovery pass. ``on_demand_discovery`` marks the pattern
discovered, so a drained pattern drops out of the next sweep.

Folded into the ``yalayut_discovery`` daily mechanical executor — no new
orchestrator method, no new cron cadence row.
"""
from __future__ import annotations

from src.infra.db import get_db
from src.infra.logging_config import get_logger
from yalayut.discovery import demand as _demand

logger = get_logger("yalayut.demand_drain")

#: A pattern with at least this many distinct un-discovered signal types is a
#: recurrence — worth one amplifying ``repeat_pattern`` signal.
REPEAT_PATTERN_MIN_TYPES: int = 3


async def _scan_repeat_patterns() -> int:
    """Reactive ``repeat_pattern`` derivation. For each un-discovered pattern
    with >= REPEAT_PATTERN_MIN_TYPES distinct *other* signal types, record one
    ``repeat_pattern`` signal. Returns the count recorded (deduped by the
    7-day cooldown). Best-effort — never raises into the drain."""
    added = 0
    try:
        db = await get_db()
        cur = await db.execute(
            "SELECT source_step_pattern, COUNT(DISTINCT signal_type) "
            "FROM yalayut_demand_signals "
            "WHERE resulted_in_discovery = 0 AND signal_type != 'repeat_pattern' "
            "GROUP BY source_step_pattern")
        rows = await cur.fetchall()
        await cur.close()
        for pattern, type_count in rows:
            if int(type_count) < REPEAT_PATTERN_MIN_TYPES:
                continue
            kw_cur = await db.execute(
                "SELECT intent_keywords_json FROM yalayut_demand_signals "
                "WHERE source_step_pattern = ? AND resulted_in_discovery = 0 "
                "LIMIT 1", (pattern,))
            kw_row = await kw_cur.fetchone()
            await kw_cur.close()
            import json
            try:
                keywords = json.loads(kw_row[0]) if kw_row and kw_row[0] else []
            except (json.JSONDecodeError, TypeError):
                keywords = []
            row_id = await _demand.record(
                source_step_pattern=pattern,
                intent_keywords=list(keywords),
                signal_type="repeat_pattern",
                confidence=0.3,
            )
            if row_id > 0:
                added += 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("repeat_pattern scan failed: %s", exc)
    return added


async def run_demand_drain() -> dict:
    """Derive repeat_pattern, then drain every pattern at/above the
    discovery threshold through on_demand_discovery. Returns a summary."""
    summary = {
        "repeat_patterns_added": 0,
        "patterns_considered": 0,
        "patterns_discovered": 0,
        "errors": [],
    }
    summary["repeat_patterns_added"] = await _scan_repeat_patterns()

    pending = await _demand.pending_signals(limit=20)
    summary["patterns_considered"] = len(pending)
    for sig in pending:
        if sig["stacked_confidence"] < _demand.DEMAND_DISCOVERY_THRESHOLD:
            continue
        try:
            # Imported here (not at module top) so tests can monkeypatch
            # yalayut.discovery.on_demand.on_demand_discovery.
            from yalayut.discovery import on_demand as _on_demand
            await _on_demand.on_demand_discovery({
                "source_step_pattern": sig["source_step_pattern"],
                "intent_keywords": sig["intent_keywords"],
            })
            summary["patterns_discovered"] += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("on-demand drain failed for %s: %s",
                           sig["source_step_pattern"], exc)
            summary["errors"].append(f"{sig['source_step_pattern']}: {exc}")
    logger.info("run_demand_drain complete", **{
        k: v for k, v in summary.items() if k != "errors"})
    return summary
```

- [ ] **Step 4: Export `run_demand_drain` from the package API**

In `packages/yalayut/src/yalayut/__init__.py`, add `"run_demand_drain"` to `__all__`, then add (after `source_scout_scan`):

```python
async def run_demand_drain() -> dict:
    """Autonomous drain: derive repeat_pattern + run on-demand discovery for
    every demand pattern above the discovery threshold."""
    from yalayut.discovery.demand_drain import run_demand_drain as _impl
    return await _impl()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `timeout 60 pytest tests/yalayut/test_demand_wiring.py -k "repeat_pattern or demand_drain" -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add packages/yalayut/src/yalayut/discovery/demand_drain.py packages/yalayut/src/yalayut/__init__.py tests/yalayut/test_demand_wiring.py
git commit -m "feat(yalayut): autonomous demand drain + repeat_pattern derivation"
```

---

## Task 7: Wire the drain into the `yalayut_discovery` daily executor

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/executors/yalayut_discovery.py:23-24`
- Test: `tests/yalayut/test_phase4_executors.py` (existing) + `tests/yalayut/test_demand_wiring.py`

The orchestrator's `_check_yalayut_discovery` already enqueues a daily `mode: "daily"` task. Folding the drain into that executor's `daily` branch means zero new wiring (handoff's recommended option 2). Existing top-level summary keys are preserved; the drain summary is nested under `demand_drain`.

- [ ] **Step 1: Write the failing test**

Append to `tests/yalayut/test_demand_wiring.py`:

```python
@pytest.mark.asyncio
async def test_daily_executor_runs_demand_drain(db, monkeypatch):
    from mr_roboto.executors import yalayut_discovery as _exec

    drained = {"called": False}

    async def _fake_daily():
        return {"sources_scanned": 0, "artifacts_ingested": 0, "errors": []}

    async def _fake_drain():
        drained["called"] = True
        return {"patterns_discovered": 2, "repeat_patterns_added": 0,
                "patterns_considered": 2, "errors": []}

    monkeypatch.setattr("yalayut.daily_discovery", _fake_daily)
    monkeypatch.setattr("yalayut.run_demand_drain", _fake_drain)

    res = await _exec.run({"payload": {"mode": "daily"}})

    assert drained["called"] is True
    assert res["demand_drain"]["patterns_discovered"] == 2
    # Existing top-level keys preserved for backward compat.
    assert "sources_scanned" in res
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 60 pytest tests/yalayut/test_demand_wiring.py -k daily_executor -v`
Expected: FAIL — `KeyError: 'demand_drain'`.

- [ ] **Step 3: Fold the drain into the `daily` branch**

In `packages/mr_roboto/src/mr_roboto/executors/yalayut_discovery.py`, change the `daily` branch:

```python
        if mode == "daily":
            return await yalayut.daily_discovery()
```

to:

```python
        if mode == "daily":
            result = await yalayut.daily_discovery()
            # Fold the autonomous demand drain into the daily run — no new
            # orchestrator method, no new cron cadence row (handoff option 2).
            try:
                result["demand_drain"] = await yalayut.run_demand_drain()
            except Exception as e:  # noqa: BLE001 — drain must not fail the run
                logger.warning("demand drain failed inside daily discovery: %s", e)
                result["demand_drain"] = {"error": str(e)}
            return result
```

Also update the module docstring's `mode` description (lines 4-5) to note that `daily` now also drains demand signals:

```python
  - ``daily``      → yalayut.daily_discovery() + yalayut.run_demand_drain()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 60 pytest tests/yalayut/test_demand_wiring.py -k daily_executor -v`
Expected: PASS.

- [ ] **Step 5: Run the existing executor suite — check for the return-shape regression**

Run: `timeout 60 pytest tests/yalayut/test_phase4_executors.py -v`
Expected: PASS. If a test asserts the exact `daily`-mode return dict by equality, relax it to assert the original keys are a subset (the new `demand_drain` key is additive). Fix any such test inline.

- [ ] **Step 6: Commit**

```bash
git add packages/mr_roboto/src/mr_roboto/executors/yalayut_discovery.py tests/yalayut/test_demand_wiring.py tests/yalayut/test_phase4_executors.py
git commit -m "feat(mr_roboto): fold autonomous demand drain into yalayut daily discovery"
```

---

## Task 8: Full-suite verification + acceptance gate

**Files:** none (verification only).

- [ ] **Step 1: Run the full demand-wiring suite**

Run: `timeout 120 pytest tests/yalayut/test_demand_wiring.py -v`
Expected: ALL PASS — covers the helper, all 5 firing sites (flash×2, react, capture, dlq executor + apply.py), the drain, repeat_pattern, and the daily-executor fold.

- [ ] **Step 2: Run the touched-package suites for regressions**

Run: `timeout 120 pytest tests/yalayut/ tests/intersect/ tests/general_beckman/ -v`
Expected: no regressions vs. baseline.

- [ ] **Step 3: Import-smoke every modified module**

Run:
```bash
python -c "import yalayut; from yalayut.discovery import demand, demand_drain, source_scout; from intersect import flash; from coulson import react; from general_beckman import apply; import mr_roboto; from mr_roboto.executors import yalayut_demand, yalayut_discovery; print('imports OK')"
```
Expected: `imports OK`.

- [ ] **Step 4: Confirm acceptance criteria from the handoff**

Verify each line below holds — they ARE the handoff's "Acceptance" section:

- Each of the 6 signal types (`step_entry_miss`, `planning_miss`, `tool_call`, `hint_miss`, `dlq`, `repeat_pattern`) has a production call site, proven by a test that exercises the host path and asserts a row lands in `yalayut_demand_signals`. → Tasks 2,3,4,5,6.
- A test proves the autonomous trigger drains `pending_signals()`, calls `on_demand_discovery` once a pattern crosses the threshold, and `mark_discovered` flips `resulted_in_discovery`. → Task 6 `test_run_demand_drain_triggers_discovery_above_threshold`.
- No core-loop file imports `yalayut` directly except `intersect`. → `apply.py` routes through the `yalayut_demand` mechanical executor (Task 5); `react.py` already lazy-imported `yalayut` pre-existing; `flash.py` is `intersect`.

- [ ] **Step 5: Final summary commit (if any test fixes from Step 5 of earlier tasks are uncommitted)**

```bash
git status
# commit any stragglers, otherwise this task is verification-only
```

---

## Self-Review

**Spec coverage** (vs. handoff Unit A + Unit B):
- Unit A — 6 dead signals: `step_entry_miss` (Task 2), `planning_miss` (Task 2), `tool_call` (Task 3), `hint_miss` (Task 4), `dlq` (Task 5), `repeat_pattern` (Task 6). ✅
- Unit B — autonomous trigger: `run_demand_drain` (Task 6) folded into the daily executor (Task 7), threshold constant created and reused by `source_scout` (Task 1). ✅
- Acceptance section: verified in Task 8. ✅

**Type consistency:**
- `demand.record(*, source_step_pattern, intent_keywords, signal_type, confidence)` — defined Task 1, called identically by `flash.py` (via `yalayut.record_demand_signal`), `capture.py` (direct `record`), `demand_drain.py` (direct `record`), `yalayut_demand` executor (via `record_demand_signal`).
- `yalayut.record_demand_signal(...)` — same kwargs, defined Task 1, used Tasks 2,3,5.
- `yalayut.run_demand_drain()` — defined Task 6, called Task 7.
- `DEMAND_DISCOVERY_THRESHOLD` — defined Task 1 in `demand.py`, read by `source_scout` (Task 1) and `demand_drain` (Task 6).
- Drain summary keys (`patterns_discovered`, `repeat_patterns_added`, `patterns_considered`, `errors`) — produced Task 6, asserted Tasks 6 & 7 consistently.

**Placeholder scan:** every code step contains complete code; every test step contains the actual test; every run step names the command and expected outcome. No TBD / "add error handling" / "similar to Task N".
