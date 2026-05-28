# CPS SP1 — Durable Continuation Substrate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a durable, restart-surviving continuation substrate by upgrading Beckman's existing `on_complete` from in-memory fire-and-forget to a DB-backed `continuations` table — so a task can enqueue a child and *return* (no held lane slot), and a named handler resumes with saved state when the child reaches a terminal state.

**Architecture:** A new `continuations(child_task_id PK, resume_name, on_error_name, state_json, status, created_at)` table. The row is written **inside `add_task`'s own transaction** (atomic with the child row — `add_task` runs on an isolated aux connection that commits before returning, so this is the only place atomicity is possible). `on_task_finished` does a **claim-then-fire** (atomic CAS to `'fired'`, then detached handler dispatch). Startup reconciles children that went terminal while down and expires genuinely-dead stale rows (alive-aware TTL). **Zero production call-site changes** — `await_inline` stays intact and coexists until SP5. A throwaway grading spike validates the handler shape before sign-off.

**Tech Stack:** Python 3.10 async, aiosqlite (single shared connection + isolated aux connections), pytest-asyncio. Package `general_beckman`, DB layer `src/infra/db.py`.

**Spec:** `docs/superpowers/specs/2026-05-27-cps-migration-design.md` (rev. 2).

**Conventions (project rules — non-negotiable):**
- Run tests with a timeout prefix: `timeout 60 pytest tests/... -v` (zombie pytest holds SQLite write locks and crash-loops KutAI).
- SQLite datetime is `strftime('%Y-%m-%d %H:%M:%S')` / `datetime('now')` — **never** `isoformat()` (the `T` separator breaks string comparisons).
- Prefix git commands with `rtk` (token-optimized passthrough).
- Lazy cross-module imports (inside functions) to avoid circular imports.

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `src/infra/db.py` | `init_db` (~337): create `continuations` table + index. `add_task` (~4331): new `on_complete`/`on_error`/`cont_state` params, dedup-skip for continuation children, atomic continuations INSERT. | Modify |
| `packages/general_beckman/src/general_beckman/continuations.py` | Registry + durable fire logic: `register_resume` (alias of `register`), 3-arg `dispatch_on_complete`, `claim_for_fire` (CAS), `fire_for_task`, `reconcile_continuations`, `expire`-via-TTL, `register_startup_handlers`. | Modify |
| `packages/general_beckman/src/general_beckman/__init__.py` | `enqueue` (~1030): `on_error`/`cont_state` params, mutual-exclusion guard, pass-through to `add_task`. `on_task_finished` (~860-896): replace the fire-and-forget block with `fire_for_task` claim-then-fire + legacy fallback; keep `next_task_spec` + `await_inline`. | Modify |
| `src/app/run.py` | `main()` (~378): run `register_startup_handlers()` + `reconcile_continuations()` after the critical-health gate. | Modify |
| `packages/mr_roboto/src/mr_roboto/executors/analytics_digest.py` | `_store_weekly_digest` → 3-arg; add module-level `register_continuations()`. | Modify |
| `packages/mr_roboto/src/mr_roboto/executors/classify_signals.py` | `_on_classifier_complete` → 3-arg; add module-level `register_continuations()` (registers at import, not only in `run()`). | Modify |
| `tests/beckman/test_continuations.py` | Update the 2 legacy handler fns to 3-arg signature. | Modify |
| `tests/beckman/test_continuations_durable.py` | New comprehensive SP1 test suite. | Create |

---

## Task 1: `continuations` table schema

**Files:**
- Modify: `src/infra/db.py` (inside `init_db`, after the `tasks` CREATE TABLE block, ~line 738+)
- Test: `tests/beckman/test_continuations_durable.py`

- [ ] **Step 1: Write the failing test**

Create `tests/beckman/test_continuations_durable.py`:

```python
"""SP1 durable continuation substrate — host-path, DB-isolated tests."""
import asyncio
import json
import pytest

import src.infra.db as _db_mod
from general_beckman import cron_seed as _cs
from general_beckman import paused_patterns as _pp


async def _fresh_db(tmp_path, monkeypatch):
    """Isolated temp DB per test (mirrors tests/beckman/test_continuations.py)."""
    db_file = tmp_path / "cps.db"
    monkeypatch.setattr(_db_mod, "DB_PATH", str(db_file))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
    monkeypatch.setattr(_db_mod, "_db_connection", None)
    monkeypatch.setattr(_cs, "_seeded", False)
    _pp._patterns.clear()
    await _db_mod.init_db()


async def _close_db():
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None


@pytest.mark.asyncio
async def test_continuations_table_and_index_created(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='continuations'"
        )
        assert await cur.fetchone() is not None, "continuations table missing"
        cur = await db.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND name='idx_continuations_pending'"
        )
        assert await cur.fetchone() is not None, "status index missing"
        # Column set matches the spec.
        cur = await db.execute("PRAGMA table_info(continuations)")
        cols = {row[1] for row in await cur.fetchall()}
        assert cols == {
            "child_task_id", "resume_name", "on_error_name",
            "state_json", "status", "created_at",
        }, f"unexpected columns: {cols}"
    finally:
        await _close_db()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 60 pytest tests/beckman/test_continuations_durable.py::test_continuations_table_and_index_created -v`
Expected: FAIL — `continuations table missing`.

- [ ] **Step 3: Add the table to `init_db`**

In `src/infra/db.py`, inside `init_db()`, after the `CREATE TABLE IF NOT EXISTS tasks (...)` block (~line 738+), add:

```python
    # Durable continuation substrate (CPS SP1). One row per child task; the
    # child's terminal state fires the registered resume/on_error handler.
    # child_task_id is the PK → exactly one continuation per child (load-bearing;
    # add_task skips dedup for continuation children to honor this).
    await db.execute("""
        CREATE TABLE IF NOT EXISTS continuations (
            child_task_id   INTEGER PRIMARY KEY,
            resume_name     TEXT NOT NULL,
            on_error_name   TEXT,
            state_json      TEXT NOT NULL DEFAULT '{}',
            status          TEXT NOT NULL DEFAULT 'pending',
            created_at      TEXT NOT NULL
        )
    """)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_continuations_pending "
        "ON continuations(status)"
    )
    await db.commit()
```

> Note: `resume_name` is `NOT NULL`. An `on_error`-only continuation (no resume) still needs a row; in that case `add_task` writes an empty-string `resume_name` and `fire_for_task` treats empty as "no resume". This keeps the schema simple and the PK invariant intact.

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 60 pytest tests/beckman/test_continuations_durable.py::test_continuations_table_and_index_created -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/infra/db.py tests/beckman/test_continuations_durable.py
rtk git commit -m "feat(beckman): add durable continuations table (CPS SP1 Task 1)"
```

---

## Task 2: `add_task` continuation params + atomic insert + dedup-skip

**Files:**
- Modify: `src/infra/db.py` — `add_task` signature (~4331) + body (~4346-4463)
- Test: `tests/beckman/test_continuations_durable.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/beckman/test_continuations_durable.py`:

```python
async def _add(**kw):
    from src.infra.db import add_task
    base = dict(title="t", description="d", agent_type="coder")
    base.update(kw)
    return await add_task(**base)


@pytest.mark.asyncio
async def test_add_task_writes_continuation_row_atomically(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        cid = await _add(on_complete="x.resume", cont_state={"k": 1})
        assert isinstance(cid, int)
        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT resume_name, on_error_name, state_json, status "
            "FROM continuations WHERE child_task_id = ?", (cid,)
        )
        row = await cur.fetchone()
        assert row is not None, "continuation row not written"
        assert row[0] == "x.resume"
        assert row[1] is None
        assert json.loads(row[2]) == {"k": 1}
        assert row[3] == "pending"
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_add_task_on_error_only_writes_empty_resume(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        cid = await _add(on_error="x.fail")
        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT resume_name, on_error_name FROM continuations WHERE child_task_id = ?",
            (cid,),
        )
        row = await cur.fetchone()
        assert row[0] == ""        # empty resume sentinel
        assert row[1] == "x.fail"
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_continuation_children_are_never_deduped(tmp_path, monkeypatch):
    """Two identical specs WITH a continuation yield two distinct children +
    two rows (no PK collision, no lost handler) — dedup is skipped."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        a = await _add(on_complete="x.resume")
        b = await _add(on_complete="x.resume")
        assert a != b, "continuation children were deduped (PK collision risk)"
        db = await _db_mod.get_db()
        cur = await db.execute("SELECT COUNT(*) FROM continuations")
        assert (await cur.fetchone())[0] == 2
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_plain_tasks_still_dedup(tmp_path, monkeypatch):
    """Regression: tasks WITHOUT a continuation keep the existing dedup behavior."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        a = await _add()
        b = await _add()        # identical → deduped to None
        assert isinstance(a, int)
        assert b is None, f"expected dedup (None), got {b}"
    finally:
        await _close_db()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `timeout 60 pytest tests/beckman/test_continuations_durable.py -k "add_task or dedup or on_error_only" -v`
Expected: FAIL — `add_task() got an unexpected keyword argument 'on_complete'`.

- [ ] **Step 3: Add params + dedup-skip + atomic insert to `add_task`**

In `src/infra/db.py`, extend the `add_task` signature (currently ends `lane=None):`):

```python
async def add_task(title, description, mission_id=None, parent_task_id=None,
                   agent_type="executor", tier="auto", priority=5,
                   requires_approval=False, depends_on=None, context=None,
                   kind="main_work", runner=None,
                   needs_real_tools=None, reversibility=None,
                   lane=None,
                   on_complete=None, on_error=None, cont_state=None):
```

Just before the dedup probe (`cursor = await db.execute("""SELECT id, title, status...`), gate it so continuation children skip dedup. Wrap the existing dedup block:

```python
            # Continuation children (on_complete/on_error set) are call-scoped
            # and never legitimately shared — skip the dedup probe so two
            # parents can't collapse to one child (PK collision on continuations
            # + lost handler). See CPS SP1 design "Enqueue atomicity & dedup".
            _has_cont = on_complete is not None or on_error is not None
            if not _has_cont:
                cursor = await db.execute(
                    """SELECT id, title, status, started_at FROM tasks
                       WHERE task_hash = ?
                         AND status IN ('pending', 'processing')
                       LIMIT 1""",
                    (task_hash,)
                )
                duplicate = await cursor.fetchone()
                if duplicate:
                    # ... EXISTING duplicate-handling block stays here verbatim ...
```

> The entire existing `if duplicate:` body (stuck-reset + skip-creation paths, ~lines 4376-4413) moves **inside** the new `if not _has_cont:` guard. Indent it one level. Nothing inside it changes.

Then, immediately after `row_id = cursor.lastrowid` (line ~4455) and **before** the `if db._conn.in_transaction: await db.execute("COMMIT")`, insert the continuation row in the same transaction:

```python
            row_id = cursor.lastrowid
            # CPS SP1: write the continuation row atomically with the child, on
            # the SAME aux connection / transaction. add_task commits before
            # returning, so this is the ONLY place the child can never exist
            # (claimable) without its continuation row — closing the missed-fire
            # window. resume_name is NOT NULL → empty string for on_error-only.
            if _has_cont:
                await db.execute(
                    "INSERT INTO continuations "
                    "(child_task_id, resume_name, on_error_name, state_json, "
                    " status, created_at) "
                    "VALUES (?, ?, ?, ?, 'pending', "
                    " strftime('%Y-%m-%d %H:%M:%S','now'))",
                    (row_id, on_complete or "", on_error,
                     json.dumps(cont_state or {})),
                )
            if db._conn.in_transaction:
                await db.execute("COMMIT")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `timeout 60 pytest tests/beckman/test_continuations_durable.py -k "add_task or dedup or on_error_only" -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the existing db/beckman suites for regressions**

Run: `timeout 120 pytest tests/beckman/ -v`
Expected: PASS (no regressions in dedup-dependent tests).

- [ ] **Step 6: Commit**

```bash
rtk git add src/infra/db.py tests/beckman/test_continuations_durable.py
rtk git commit -m "feat(beckman): add_task atomic continuation insert + dedup-skip (CPS SP1 Task 2)"
```

---

## Task 3: registry + claim-then-fire logic in `continuations.py`

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/continuations.py`
- Test: `tests/beckman/test_continuations_durable.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/beckman/test_continuations_durable.py`:

```python
@pytest.mark.asyncio
async def test_claim_for_fire_is_single_winner(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.continuations import claim_for_fire
        cid = await _add(on_complete="x.resume", cont_state={"v": 9})
        first = await claim_for_fire(cid)
        second = await claim_for_fire(cid)
        assert first is not None and first["resume_name"] == "x.resume"
        assert first["state"] == {"v": 9}
        assert second is None, "second claim must lose"
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_fire_for_task_success_dispatches_resume(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.continuations import (
            register_resume, fire_for_task, _HANDLERS,
        )
        seen = []

        async def resume(task_id, result, state):
            seen.append((task_id, result, state))

        register_resume("t.resume", resume)
        cid = await _add(on_complete="t.resume", cont_state={"s": 1})
        fired = await fire_for_task(cid, {"status": "completed", "result": "ok"},
                                    "completed")
        await asyncio.sleep(0.05)   # handler dispatched detached
        assert fired is True
        assert seen == [(cid, {"status": "completed", "result": "ok"}, {"s": 1})]
    finally:
        _HANDLERS.pop("t.resume", None)
        await _close_db()


@pytest.mark.asyncio
async def test_fire_failed_with_on_error_dispatches_on_error(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.continuations import (
            register_resume, fire_for_task, _HANDLERS,
        )
        errs = []

        async def on_err(task_id, result, state):
            errs.append((task_id, result.get("status"), state))

        register_resume("t.err", on_err)
        cid = await _add(on_error="t.err", cont_state={"p": 2})
        fired = await fire_for_task(cid, {"status": "failed", "error": "boom"},
                                    "failed")
        await asyncio.sleep(0.05)
        assert fired is True
        assert errs == [(cid, "failed", {"p": 2})]
    finally:
        _HANDLERS.pop("t.err", None)
        await _close_db()


@pytest.mark.asyncio
async def test_fire_failed_without_on_error_is_noop(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.continuations import fire_for_task
        cid = await _add(on_complete="t.resume")     # resume only, no on_error
        fired = await fire_for_task(cid, {"status": "failed"}, "failed")
        # Row is claimed (so it can't re-fire as success later) but nothing runs.
        assert fired is True
        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT status FROM continuations WHERE child_task_id = ?", (cid,)
        )
        assert (await cur.fetchone())[0] == "fired"
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_fire_needs_clarification_leaves_pending(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.continuations import fire_for_task
        cid = await _add(on_complete="t.resume")
        fired = await fire_for_task(cid, {"status": "needs_clarification"},
                                    "needs_clarification")
        assert fired is False, "needs_clarification must not fire"
        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT status FROM continuations WHERE child_task_id = ?", (cid,)
        )
        assert (await cur.fetchone())[0] == "pending", "row must stay pending"
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_fire_no_row_returns_false(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.continuations import fire_for_task
        plain = await _add()    # no continuation
        assert await fire_for_task(plain, {"status": "completed"}, "completed") is False
    finally:
        await _close_db()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `timeout 60 pytest tests/beckman/test_continuations_durable.py -k "claim or fire_" -v`
Expected: FAIL — `cannot import name 'claim_for_fire'` / `register_resume`.

- [ ] **Step 3: Rewrite `continuations.py`**

Replace the body of `packages/general_beckman/src/general_beckman/continuations.py` with:

```python
"""Durable continuation substrate for Beckman (CPS SP1).

A continuation = "when child task T reaches a terminal state, run a named
handler with saved parent state". The row lives in the `continuations` table
(written atomically with the child in add_task), so it survives restarts.

Handler signature: ``async (task_id: int, result: dict, state: dict) -> None``
where ``task_id`` is the CHILD id and ``state`` is the parent state passed as
``cont_state`` at enqueue time (``{}`` when none).
"""
from __future__ import annotations

import asyncio
import json
from typing import Awaitable, Callable

from src.infra.logging_config import get_logger

_log = get_logger("beckman.continuations")

# Default expiry for a pending continuation whose child never reaches terminal
# AND is no longer alive. Replaces the opaque 600s await_inline block.
CONTINUATION_TTL_SECONDS = 3600

# name → async callable(task_id: int, result: dict, state: dict) -> None
_HANDLERS: dict[str, Callable[[int, dict, dict], Awaitable[None]]] = {}


def register_resume(name: str, handler) -> None:
    """Register a resume / on_error handler. Idempotent — overrides existing."""
    _HANDLERS[name] = handler


# Back-compat alias: existing callers (analytics_digest, classify_signals) use
# ``register``. Keep it pointing at the same registry.
register = register_resume


async def dispatch_on_complete(name: str, task_id: int, result: dict,
                               state: dict | None = None) -> None:
    """Look up and invoke the named handler (3-arg). Swallows handler errors."""
    handler = _HANDLERS.get(name)
    if handler is None:
        _log.warning("dispatch: no handler registered", name=name, task_id=task_id)
        return
    try:
        await handler(task_id, result, state or {})
    except Exception as exc:  # noqa: BLE001
        _log.error("continuation handler raised", name=name, task_id=task_id,
                   error=str(exc))


async def claim_for_fire(child_task_id: int) -> dict | None:
    """Atomically claim the pending continuation for a child (CAS).

    Returns ``{resume_name, on_error_name, state}`` if THIS caller won the
    claim (flipped pending→fired), else ``None`` (no row, or already fired).
    Claim happens BEFORE handler dispatch so a re-entrant on_task_finished
    can never double-fire.
    """
    from src.infra.db import get_db
    db = await get_db()
    upd = await db.execute(
        "UPDATE continuations SET status='fired' "
        "WHERE child_task_id=? AND status='pending'",
        (child_task_id,),
    )
    await db.commit()
    if upd.rowcount != 1:
        return None
    cur = await db.execute(
        "SELECT resume_name, on_error_name, state_json "
        "FROM continuations WHERE child_task_id=?",
        (child_task_id,),
    )
    row = await cur.fetchone()
    if row is None:
        return None
    try:
        state = json.loads(row[2]) if row[2] else {}
    except Exception:
        state = {}
    return {"resume_name": row[0], "on_error_name": row[1], "state": state}


async def fire_for_task(child_task_id: int, result: dict, raw_status: str) -> bool:
    """Fire the continuation for a terminal child. Returns True if it claimed.

    Status mapping (raw agent status, BEFORE route_result rewrite):
      - 'needs_clarification' → leave pending, do NOT claim (not terminal).
      - 'failed'              → dispatch on_error if set, else claimed no-op.
      - anything else (incl. 'completed', and a grade that graded as
        {passed: false}) → dispatch resume.
    Handler dispatch is detached; the claim (CAS) is synchronous.
    """
    if raw_status == "needs_clarification":
        return False
    claim = await claim_for_fire(child_task_id)
    if claim is None:
        return False
    state = claim["state"]
    if raw_status == "failed":
        name = claim["on_error_name"]
        if name:
            asyncio.create_task(dispatch_on_complete(name, child_task_id, result, state))
        else:
            _log.info("continuation: failed child, no on_error — no-op",
                      child_task_id=child_task_id)
    else:
        name = claim["resume_name"]
        if name:
            asyncio.create_task(dispatch_on_complete(name, child_task_id, result, state))
    return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `timeout 60 pytest tests/beckman/test_continuations_durable.py -k "claim or fire_" -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
rtk git add packages/general_beckman/src/general_beckman/continuations.py tests/beckman/test_continuations_durable.py
rtk git commit -m "feat(beckman): durable claim-then-fire continuation logic (CPS SP1 Task 3)"
```

---

## Task 4: `enqueue` wiring (params + mutual-exclusion guard)

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/__init__.py` — `enqueue` (~1030-1108)
- Test: `tests/beckman/test_continuations_durable.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/beckman/test_continuations_durable.py`:

```python
@pytest.mark.asyncio
async def test_enqueue_with_continuation_returns_fresh_id_and_row(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman import enqueue
        cid = await enqueue(
            {"title": "child", "description": "d", "agent_type": "coder"},
            on_complete="t.resume", cont_state={"parent_id": 7},
        )
        assert isinstance(cid, int)
        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT resume_name, state_json FROM continuations WHERE child_task_id=?",
            (cid,),
        )
        row = await cur.fetchone()
        assert row[0] == "t.resume"
        assert json.loads(row[1]) == {"parent_id": 7}
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_enqueue_rejects_await_inline_plus_on_complete(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman import enqueue
        with pytest.raises(ValueError):
            await enqueue(
                {"title": "x", "description": "d", "agent_type": "coder"},
                await_inline=True, on_complete="t.resume",
            )
    finally:
        await _close_db()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `timeout 60 pytest tests/beckman/test_continuations_durable.py -k "enqueue_with_continuation or rejects_await_inline" -v`
Expected: FAIL — `enqueue() got an unexpected keyword argument 'cont_state'`.

- [ ] **Step 3: Update `enqueue`**

In `packages/general_beckman/src/general_beckman/__init__.py`, change the `enqueue` signature:

```python
async def enqueue(
    spec: dict,
    *,
    parent_id: int | None = None,
    await_inline: bool = False,
    on_complete: str | None = None,
    on_error: str | None = None,
    cont_state: dict | None = None,
    next_task_spec: dict | None = None,
    lane: str | None = None,
) -> "int | TaskResult":
```

Right after the docstring, add the mutual-exclusion guard:

```python
    if await_inline and (on_complete is not None or on_error is not None):
        raise ValueError(
            "enqueue: await_inline and on_complete/on_error are mutually "
            "exclusive (a blocking wait can't also fire a continuation)"
        )
```

Replace the existing continuation-envelope block (the `if on_complete is not None or next_task_spec is not None:` block, ~lines 1083-1102) so **only** `next_task_spec` is stored in context (`on_complete`/`on_error` now go to the table via `add_task`):

```python
    # ── next_task_spec envelope (fire-and-forget chain — NOT the durable
    # substrate; stays context-based and coexists). on_complete/on_error go
    # straight to add_task → continuations table.
    if next_task_spec is not None:
        raw_ctx = spec.get("context")
        if raw_ctx is None:
            ctx = {}
        elif isinstance(raw_ctx, str):
            try:
                ctx = _json.loads(raw_ctx)
            except Exception:
                ctx = {}
        else:
            ctx = dict(raw_ctx)
        beckman_sub = dict(ctx.get("beckman") or {})
        beckman_sub["next_task_spec"] = next_task_spec
        ctx["beckman"] = beckman_sub
        spec["context"] = ctx
```

Change the `add_task` call to pass the continuation params and assert a child came back:

```python
    task_id = await add_task(**spec, kind=kind, lane=_lane,
                             on_complete=on_complete, on_error=on_error,
                             cont_state=cont_state)
    if (on_complete is not None or on_error is not None) and task_id is None:
        raise RuntimeError(
            "enqueue: add_task returned no child id for a continuation task "
            "(dedup should be skipped for continuations — investigate add_task)"
        )
    await build_and_push()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `timeout 60 pytest tests/beckman/test_continuations_durable.py -k "enqueue_with_continuation or rejects_await_inline" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
rtk git add packages/general_beckman/src/general_beckman/__init__.py tests/beckman/test_continuations_durable.py
rtk git commit -m "feat(beckman): enqueue on_error/cont_state params + mutual-exclusion guard (CPS SP1 Task 4)"
```

---

## Task 5: `on_task_finished` durable fire (replace fire-and-forget)

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/__init__.py` — the continuation block in `on_task_finished` (~lines 860-896)
- Test: `tests/beckman/test_continuations_durable.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/beckman/test_continuations_durable.py`:

```python
@pytest.mark.asyncio
async def test_on_task_finished_fires_resume_with_state(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman import enqueue, on_task_finished
        from general_beckman.continuations import register_resume, _HANDLERS
        seen = []

        async def resume(task_id, result, state):
            seen.append((task_id, state))

        register_resume("t.resume", resume)
        cid = await enqueue(
            {"title": "c", "description": "d", "agent_type": "coder"},
            on_complete="t.resume", cont_state={"parent_id": 99},
        )
        await on_task_finished(cid, {"status": "completed", "result": "ok"})
        await asyncio.sleep(0.05)
        assert seen == [(cid, {"parent_id": 99})]
    finally:
        _HANDLERS.pop("t.resume", None)
        await _close_db()


@pytest.mark.asyncio
async def test_double_on_task_finished_fires_once(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman import enqueue, on_task_finished
        from general_beckman.continuations import register_resume, _HANDLERS
        calls = []

        async def resume(task_id, result, state):
            calls.append(task_id)

        register_resume("t.resume", resume)
        cid = await enqueue(
            {"title": "c", "description": "d", "agent_type": "coder"},
            on_complete="t.resume",
        )
        await on_task_finished(cid, {"status": "completed", "result": "ok"})
        await on_task_finished(cid, {"status": "completed", "result": "ok"})
        await asyncio.sleep(0.05)
        assert calls == [cid], f"expected single fire, got {calls}"
    finally:
        _HANDLERS.pop("t.resume", None)
        await _close_db()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `timeout 60 pytest tests/beckman/test_continuations_durable.py -k "on_task_finished_fires or double_on_task" -v`
Expected: FAIL — resume not invoked / invoked twice (old context path stores nothing now).

- [ ] **Step 3: Replace the continuation block in `on_task_finished`**

In `packages/general_beckman/src/general_beckman/__init__.py`, replace the entire `try:` block under the `# ── Continuation hooks ─` comment (~lines 873-896 — the one doing `dispatch_on_complete` via `create_task`, `next_task_spec` chain, and `resolve_inline`) with:

```python
    try:
        _raw_status = (result or {}).get("status") or "completed"

        # Durable continuation fire (claim-then-detach via the table). Replaces
        # the old fire-and-forget create_task(dispatch_on_complete). Idempotent.
        from general_beckman.continuations import fire_for_task, dispatch_on_complete
        _fired = await fire_for_task(task_id, dict(result or {}), _raw_status)

        # Legacy straggler shim (removable post-SP5): a task enqueued BEFORE this
        # upgrade carried on_complete in context.beckman, not the table. If no
        # row claimed and the status is terminal, fire that legacy handler once.
        if not _fired and _raw_status != "needs_clarification":
            _legacy = (task_ctx.get("beckman") or {}).get("on_complete")
            if _legacy:
                asyncio.create_task(
                    dispatch_on_complete(_legacy, task_id, dict(result or {}), {})
                )

        # next_task_spec fire-and-forget chain (unchanged, context-based).
        _next_spec = (task_ctx.get("beckman") or {}).get("next_task_spec")
        if _next_spec and isinstance(_next_spec, dict):
            asyncio.create_task(enqueue(_next_spec, parent_id=task_id))

        # await_inline resolve (coexists until SP5).
        if task_id in _inline_waiters:
            _tr = TaskResult(
                status=_raw_status,
                result=(result or {}).get("result"),
                error=(result or {}).get("error"),
            )
            resolve_inline(task_id, _tr)
    except Exception as _ce:
        log.debug("continuation hook failed", task_id=task_id, error=str(_ce))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `timeout 60 pytest tests/beckman/test_continuations_durable.py -k "on_task_finished_fires or double_on_task" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
rtk git add packages/general_beckman/src/general_beckman/__init__.py tests/beckman/test_continuations_durable.py
rtk git commit -m "feat(beckman): on_task_finished durable claim-then-fire (CPS SP1 Task 5)"
```

---

## Task 6: restart reconcile + alive-aware TTL + startup handler registration

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/continuations.py` (add `reconcile_continuations`, `register_startup_handlers`)
- Modify: `src/app/run.py` — `main()` (~after line 378)
- Test: `tests/beckman/test_continuations_durable.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/beckman/test_continuations_durable.py`:

```python
async def _set_task_status(task_id, status, result_json=None):
    db = await _db_mod.get_db()
    if result_json is None:
        await db.execute("UPDATE tasks SET status=? WHERE id=?", (status, task_id))
    else:
        await db.execute(
            "UPDATE tasks SET status=?, result=? WHERE id=?",
            (status, result_json, task_id),
        )
    await db.commit()


@pytest.mark.asyncio
async def test_reconcile_fires_terminal_child_with_reconstructed_result(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.continuations import (
            register_resume, reconcile_continuations, _HANDLERS,
        )
        seen = []

        async def resume(task_id, result, state):
            seen.append((task_id, result.get("verdict"), state))

        register_resume("t.resume", resume)
        cid = await _add(on_complete="t.resume", cont_state={"p": 5})
        # Child went terminal while we were "down" — result persisted on the row.
        await _set_task_status(cid, "completed", json.dumps({"verdict": "pass"}))
        await reconcile_continuations()
        await asyncio.sleep(0.05)
        assert seen == [(cid, "pass", {"p": 5})], (
            f"reconcile must fire with reconstructed result: {seen}"
        )
    finally:
        _HANDLERS.pop("t.resume", None)
        await _close_db()


@pytest.mark.asyncio
async def test_ttl_expires_dead_stale_child(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman.continuations import (
            register_resume, reconcile_continuations, _HANDLERS,
        )
        errs = []

        async def on_err(task_id, result, state):
            errs.append(task_id)

        register_resume("t.err", on_err)
        cid = await _add(on_error="t.err")
        # Child still 'processing' (not terminal), not in_flight (dead),
        # and the row is older than TTL.
        await _set_task_status(cid, "processing")
        db = await _db_mod.get_db()
        await db.execute(
            "UPDATE continuations SET created_at='2000-01-01 00:00:00' "
            "WHERE child_task_id=?", (cid,))
        await db.commit()
        await reconcile_continuations(ttl_seconds=3600)
        await asyncio.sleep(0.05)
        assert errs == [cid], "dead stale child must expire via on_error"
    finally:
        _HANDLERS.pop("t.err", None)
        await _close_db()


@pytest.mark.asyncio
async def test_ttl_leaves_alive_child_pending(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        import src.core.in_flight as _if
        from general_beckman.continuations import reconcile_continuations
        cid = await _add(on_complete="t.resume")
        await _set_task_status(cid, "processing")
        db = await _db_mod.get_db()
        await db.execute(
            "UPDATE continuations SET created_at='2000-01-01 00:00:00' "
            "WHERE child_task_id=?", (cid,))
        await db.commit()

        # Make the child appear alive in the in-flight registry.
        class _E:
            task_id = cid
        monkeypatch.setattr(_if, "in_flight_snapshot", lambda: [_E()])

        await reconcile_continuations(ttl_seconds=3600)
        cur = await db.execute(
            "SELECT status FROM continuations WHERE child_task_id=?", (cid,))
        assert (await cur.fetchone())[0] == "pending", "alive child must stay pending"
    finally:
        await _close_db()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `timeout 60 pytest tests/beckman/test_continuations_durable.py -k "reconcile or ttl" -v`
Expected: FAIL — `cannot import name 'reconcile_continuations'`.

- [ ] **Step 3: Add reconcile + startup registration to `continuations.py`**

Append to `packages/general_beckman/src/general_beckman/continuations.py`:

```python
# Modules that register continuation handlers as an import side-effect. After a
# restart the in-memory registry is empty until these are imported — reconcile
# must run AFTER this, or it finds no handler and drops the continuation.
_HANDLER_MODULES = (
    "mr_roboto.executors.analytics_digest",
    "mr_roboto.executors.classify_signals",
)


def register_startup_handlers() -> None:
    """Import known continuation-bearing modules so their register() fires."""
    import importlib
    for mod in _HANDLER_MODULES:
        try:
            m = importlib.import_module(mod)
            reg = getattr(m, "register_continuations", None)
            if callable(reg):
                reg()
        except Exception as exc:  # noqa: BLE001
            _log.debug("startup handler import failed", module=mod, error=str(exc))


async def reconcile_continuations(ttl_seconds: int = CONTINUATION_TTL_SECONDS) -> None:
    """Startup/periodic recovery pass over pending continuations.

    For each pending row:
      - child terminal (completed/failed) → reconstruct result from tasks.result
        and fire (closes the down-while-child-finished gap);
      - else, if past TTL AND child is not alive (no in_flight entry) → expire
        (fire on_error if set, else log). A still-alive long-runner is left
        pending — no premature abandon.
    """
    from src.infra.db import get_db
    db = await get_db()
    cur = await db.execute(
        "SELECT child_task_id FROM continuations WHERE status='pending'"
    )
    pending_ids = [r[0] for r in await cur.fetchall()]

    for cid in pending_ids:
        tcur = await db.execute("SELECT status, result FROM tasks WHERE id=?", (cid,))
        trow = await tcur.fetchone()
        if trow is None:
            continue
        tstatus, tresult = trow[0], trow[1]

        if tstatus in ("completed", "failed"):
            res: dict = {}
            if tresult:
                try:
                    parsed = json.loads(tresult) if isinstance(tresult, str) else tresult
                    res = dict(parsed) if isinstance(parsed, dict) else {"result": parsed}
                except Exception:
                    res = {"result": tresult}
            res.setdefault("status", tstatus)
            await fire_for_task(cid, res, tstatus)
            continue

        # Not terminal — TTL + alive check.
        ecur = await db.execute(
            "SELECT 1 FROM continuations WHERE child_task_id=? "
            "AND datetime(created_at, '+' || ? || ' seconds') < datetime('now')",
            (cid, ttl_seconds),
        )
        if await ecur.fetchone() is None:
            continue  # not yet expired

        alive = False
        try:
            from src.core.in_flight import in_flight_snapshot
            alive = any(getattr(e, "task_id", None) == cid for e in in_flight_snapshot())
        except Exception:
            alive = False
        if alive:
            continue  # long-runner — leave pending

        claim = await claim_for_fire(cid)
        if claim is None:
            continue
        name = claim["on_error_name"]
        if name:
            asyncio.create_task(dispatch_on_complete(
                name, cid,
                {"status": "failed", "error": "continuation TTL expired"},
                claim["state"],
            ))
        else:
            _log.warning("continuation expired (no on_error)", child_task_id=cid)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `timeout 60 pytest tests/beckman/test_continuations_durable.py -k "reconcile or ttl" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Wire the startup pass into `run.py`**

In `src/app/run.py`, `main()`, immediately after the critical-health gate (after the `if not critical_ok: ... sys.exit(1)` block at line ~378), before the Phase 2 `async def _docker_phase()`:

```python
    # Continuation recovery (CPS SP1): re-fire continuations whose child went
    # terminal while we were down; expire genuinely-dead stale ones. Handlers
    # must be registered first (in-memory registry is empty after restart).
    try:
        from general_beckman.continuations import (
            register_startup_handlers, reconcile_continuations,
        )
        register_startup_handlers()
        await reconcile_continuations()
        _log.info("Continuation recovery pass complete")
    except Exception as exc:
        _log.warning("Continuation reconcile failed (non-critical): %s", exc)
```

- [ ] **Step 6: Verify run.py imports cleanly**

Run: `.venv\Scripts\python -c "import ast; ast.parse(open('src/app/run.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 7: Commit**

```bash
rtk git add packages/general_beckman/src/general_beckman/continuations.py src/app/run.py tests/beckman/test_continuations_durable.py
rtk git commit -m "feat(beckman): restart reconcile + alive-aware TTL + startup handler reg (CPS SP1 Task 6)"
```

---

## Task 7: migrate the 2 existing handlers to 3-arg + fix legacy test

**Files:**
- Modify: `packages/mr_roboto/src/mr_roboto/executors/analytics_digest.py`
- Modify: `packages/mr_roboto/src/mr_roboto/executors/classify_signals.py`
- Modify: `tests/beckman/test_continuations.py` (legacy handler signatures)
- Test: `tests/beckman/test_continuations_durable.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/beckman/test_continuations_durable.py`:

```python
@pytest.mark.asyncio
async def test_existing_handlers_fire_with_empty_state(tmp_path, monkeypatch):
    """analytics_digest + classify_signals handlers run under the durable path
    with state={} (they don't use state). Proves the 3-arg migration."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from general_beckman import enqueue, on_task_finished
        from general_beckman.continuations import register_startup_handlers, _HANDLERS

        register_startup_handlers()
        assert "growth.store_weekly_digest" in _HANDLERS
        assert "growth.classify_signals_complete" in _HANDLERS

        # Calling each with 3 args must not raise (they early-return on empty
        # input — we only assert the signature accepts state).
        await _HANDLERS["growth.store_weekly_digest"](1, {"result": ""}, {})
        await _HANDLERS["growth.classify_signals_complete"](1, {"result": {}}, {})
    finally:
        await _close_db()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 60 pytest tests/beckman/test_continuations_durable.py -k "existing_handlers" -v`
Expected: FAIL — either handler not registered (classify_signals registers only in `run()`), or `takes 2 positional arguments but 3 were given`.

- [ ] **Step 3: Update `analytics_digest.py`**

Change `_store_weekly_digest` signature to accept `state`:

```python
async def _store_weekly_digest(task_id: int, result: dict, state: dict | None = None) -> None:
```

Add a public module-level registration alias and call it (rename for the startup importer). Replace the existing `_register_continuation()` definition + its module-level call with:

```python
def register_continuations() -> None:
    """Register the digest-persistence continuation (idempotent)."""
    try:
        from general_beckman.continuations import register
        register(_DIGEST_CONTINUATION, _store_weekly_digest)
    except Exception as exc:  # noqa: BLE001
        logger.debug("digest continuation registration deferred", error=str(exc))


# Back-compat name used internally elsewhere in this module.
_register_continuation = register_continuations

# Register at import so the handler is present for restart reconcile.
register_continuations()
```

> The internal calls to `_register_continuation()` (inside `_enqueue_synthesis_agent`) keep working via the alias.

- [ ] **Step 4: Update `classify_signals.py`**

Change `_on_classifier_complete` signature:

```python
async def _on_classifier_complete(task_id: int, result: dict, state: dict | None = None) -> None:
```

Add a module-level registration (it currently registers only inside `run()`). After the imports / `_log` definition near the top, add:

```python
def register_continuations() -> None:
    """Register the classify-signals continuation (idempotent). Called at import
    so the handler survives a restart for the reconcile pass."""
    try:
        from general_beckman.continuations import register
        register("growth.classify_signals_complete", _on_classifier_complete)
    except Exception as exc:  # noqa: BLE001
        _log.debug("classify_signals continuation registration deferred", error=str(exc))
```

At the **end** of the module (after `_on_classifier_complete` is defined), add:

```python
register_continuations()
```

> Keep the existing `register("growth.classify_signals_complete", _on_classifier_complete)` call inside `run()` — it's idempotent and harmless.

- [ ] **Step 5: Fix the legacy test's 2-arg handlers**

In `tests/beckman/test_continuations.py`, update the three local handler fns to accept `state`:

```python
        async def my_handler(task_id: int, result: dict, state: dict = None) -> None:
            calls.append((task_id, result))
```
```python
        async def bad_handler(task_id: int, result: dict, state: dict = None) -> None:
            raise RuntimeError("boom")
```
```python
        async def my_resume(task_id: int, result: dict, state: dict = None) -> None:
            invoked.append(task_id)
```

- [ ] **Step 6: Run the new + legacy continuation tests**

Run: `timeout 60 pytest tests/beckman/test_continuations.py tests/beckman/test_continuations_durable.py -v`
Expected: PASS (all). The legacy `test_terminal_router_fires_on_complete` now exercises the table path end-to-end.

- [ ] **Step 7: Verify the two executor modules import cleanly**

Run: `.venv\Scripts\python -c "import mr_roboto.executors.analytics_digest, mr_roboto.executors.classify_signals; print('ok')"`
Expected: `ok`.

- [ ] **Step 8: Commit**

```bash
rtk git add packages/mr_roboto/src/mr_roboto/executors/analytics_digest.py packages/mr_roboto/src/mr_roboto/executors/classify_signals.py tests/beckman/test_continuations.py tests/beckman/test_continuations_durable.py
rtk git commit -m "feat(beckman): migrate on_complete handlers to 3-arg durable signature (CPS SP1 Task 7)"
```

---

## Task 8: full-suite regression + acceptance gate

**Files:** none (verification only)

- [ ] **Step 1: Run the full beckman + mr_roboto + workflow-hook suites**

Run: `timeout 120 pytest tests/beckman/ packages/general_beckman/tests/ packages/mr_roboto/tests/ tests/workflows/engine/ -v`
Expected: PASS, zero regressions. Pay attention to any test that asserted on the old context-based `on_complete` fire path or on `dispatch_on_complete`'s 2-arg signature — those must now go green via the table path.

- [ ] **Step 2: Grep for stray callers of the old 2-arg `dispatch_on_complete`**

Run: `rtk grep "dispatch_on_complete(" --type py`
Expected: only `continuations.py` (definition + internal calls) and `__init__.py` (legacy shim). Any other 2-arg call site must be updated to pass `state`.

- [ ] **Step 3: Confirm no `await_inline` production call site was touched**

Run: `rtk git diff --stat main`
Expected: changed files are exactly the 7 in the File Structure table — no `grading.py`, `code_review.py`, `telegram_bot.py`, `hooks.py`, etc. (those are SP2/SP3).

- [ ] **Step 4: Acceptance checklist (from the spec)**

Confirm each, citing the test that proves it:
- `continuations` table + `idx_continuations_pending` created on init — `test_continuations_table_and_index_created`.
- `enqueue(on_complete=…, cont_state=…)` returns a fresh, never-deduped child id, never blocks — `test_enqueue_with_continuation_returns_fresh_id_and_row` + `test_continuation_children_are_never_deduped`.
- Resume runs with state on completion — `test_on_task_finished_fires_resume_with_state`.
- Claim-then-fire idempotent under double terminal — `test_double_on_task_finished_fires_once`.
- Restart-reconcile fires with result reconstructed from the task row — `test_reconcile_fires_terminal_child_with_reconstructed_result`.
- Alive-aware TTL — `test_ttl_expires_dead_stale_child` + `test_ttl_leaves_alive_child_pending`.
- `needs_clarification` no-fire — `test_fire_needs_clarification_leaves_pending`.
- Existing 2 callers fire under the durable path — `test_existing_handlers_fire_with_empty_state`.

- [ ] **Step 5: Commit (if any fixups were needed)**

```bash
rtk git add -A
rtk git commit -m "test(beckman): CPS SP1 full-suite regression green (Task 8)"
```

---

## Task 9: SP3 grading spike (THROWAWAY — gates sign-off, do NOT merge)

> **Purpose:** Prove the `(child_id, child_result, state)` handler shape is sufficient to migrate the hardest real consumer (grading) BEFORE declaring the substrate done. If the shape is wrong, fix the substrate now (Tasks 3-6), not in SP3. **This task's code is discarded** — it validates, it does not ship.

**Files (scratch branch only):**
- Read: `src/core/grading.py` (find the `await_inline=True` / `beckman.enqueue(... await_inline=True)` reviewer-child call site)
- Read: `packages/general_beckman/src/general_beckman/apply.py` (the grade-verdict apply path, e.g. `_apply_posthook_verdict` / the `verdict.source_task_id` handlers at apply.py:2019+)

- [ ] **Step 1: Create the scratch branch**

```bash
rtk git checkout -b spike/cps-sp1-grading-validation
```

- [ ] **Step 2: Locate the grading inline-wait site**

Run: `rtk grep "await_inline" src/core/grading.py`
Read the surrounding function. Identify: (a) what the grader passes to the reviewer child, (b) what it does with the returned `TaskResult` (the verdict-apply call), (c) what parent/source identifiers it needs afterward.

- [ ] **Step 3: Write a host-path spike test (the success criterion)**

Create `tests/beckman/test_spike_grading_cps.py` proving the verdict round-trips via CPS instead of blocking:
- enqueue the reviewer child with `on_complete="grade.resume"` + `cont_state={parent/source ids + anything the verdict-apply needs}`;
- register a `grade.resume(child_id, result, state)` that reconstructs the verdict from `result` + `state` and drives the SAME verdict-apply path the inline code used (per [[feedback_verify_verdict_roundtrip]]: the verdict must reach `_apply_posthook_verdict`);
- drive `enqueue → on_task_finished(child, reviewer_result)` against a temp DB;
- assert the source task's status/verdict landed exactly as the `await_inline` version produced.

- [ ] **Step 4: Run the spike test**

Run: `timeout 60 pytest tests/beckman/test_spike_grading_cps.py -v`

- [ ] **Step 5: Record the finding**

- **If green:** the handler shape is sufficient. Note in the spec's Acceptance ("SP3 grading spike green") and proceed.
- **If it needs more than `(child_id, result, state)`** (e.g. the handler can't re-enter routing, or needs a `parent_id`-keyed finish API): **stop**, amend Tasks 3/4/6 to provide what's missing (likely a documented `state` convention carrying `parent_id` + a thin `on_task_finished(parent_id, produced_result)` re-entry helper), re-run SP1 Tasks 3-8, then re-run this spike.

- [ ] **Step 6: Discard the spike**

```bash
rtk git checkout main
rtk git branch -D spike/cps-sp1-grading-validation
```

> Do not merge the spike. The real grading migration ships in SP3 with its own spec.

---

## Self-Review (completed by plan author)

**Spec coverage:** table+index (T1), atomic insert + dedup-skip (T2), CAS claim-then-fire + 3-status mapping + detached dispatch (T3), enqueue params + mutual-exclusion + non-None assert (T4), on_task_finished durable fire + legacy shim + next_task_spec/await_inline coexistence (T5), restart reconcile w/ result reconstruction + alive-aware TTL + startup handler registration (T6), 2 existing callers migrated to 3-arg (T7), full-suite + acceptance (T8), SP3 grading spike (T9). All spec sections mapped.

**Placeholder scan:** every code step shows full code; commands have expected output. The only intentionally-exploratory task is T9 (a throwaway spike, exploratory by definition) — it gives exact files to read, a concrete success criterion, and a branch-discard.

**Type consistency:** handler signature `(task_id, result, state)` consistent across `dispatch_on_complete`, `fire_for_task`, `register_resume`, both migrated handlers, all tests. `claim_for_fire` returns `{resume_name, on_error_name, state}` consumed identically in `fire_for_task` and `reconcile_continuations`. `add_task` params `on_complete/on_error/cont_state` match `enqueue`'s pass-through. Terminal status set `{completed, failed}` consistent (T6 reconcile + spec).
