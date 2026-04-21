# Pool Pressure Shared Primitive + Beckman Admission — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a shared pool-pressure primitive (derived per-cloud-pool with in-flight tracking) and rewire Beckman's `next_task()` to admit tasks based on Hoca's preselected model and that model's pool pressure, gated by task urgency.

**Architecture:**
- KDV tracks cloud calls between dispatch and response via `begin_call`/`end_call` (in_flight counter with TTL safety net). It pushes `CloudProviderState` to nerd_herd on each boundary.
- `NerdHerd.snapshot()` computes per-(provider, model) `PoolPressure` lazily from pushed state. Accessor `snapshot.pressure_for(model)` returns a signed float in [-1, +1].
- Fatih Hoca becomes pure: swap-budget state moves to nerd_herd (read by Hoca, written by dispatcher); `model_pick_log` writes move to dispatcher post-iteration.
- Beckman's `next_task()` scans top-K ready tasks by urgency. For each, asks Hoca for a Pick, looks up the Pick's model pressure, admits the first task whose pressure clears `threshold(urgency)`.

**Tech Stack:** Python 3.10, asyncio, aiosqlite, existing packages (`kuleden_donen_var`, `nerd_herd`, `fatih_hoca`, `general_beckman`), pytest.

---

## File Structure

Modifications clustered by responsibility:

| File | Responsibility | Action |
|---|---|---|
| `packages/nerd_herd/src/nerd_herd/swap_budget.py` | SwapBudget state owner | **Create** (move from `fatih_hoca.types`) |
| `packages/nerd_herd/src/nerd_herd/nerd_herd.py` | Expose swap + queue_profile + push APIs | Modify |
| `packages/nerd_herd/src/nerd_herd/pool_pressure.py` | Pure pool-pressure computation | **Create** |
| `packages/nerd_herd/src/nerd_herd/types.py` | Add `in_flight`, `PoolPressure`, `QueueProfile`, `pressure_for` | Modify |
| `packages/kuleden_donen_var/src/kuleden_donen_var/in_flight.py` | `begin_call`/`end_call` API with TTL prune | **Create** |
| `packages/kuleden_donen_var/src/kuleden_donen_var/__init__.py` | Re-export in_flight API; update push_cloud_state | Modify |
| `packages/fatih_hoca/src/fatih_hoca/selector.py` | Remove side effects (purity) | Modify |
| `packages/fatih_hoca/src/fatih_hoca/scarcity.py` | Consume `snapshot.pressure_for()`; keep queue arm | Modify |
| `packages/fatih_hoca/src/fatih_hoca/types.py` | Remove `SwapBudget` class (moved) | Modify |
| `packages/general_beckman/src/general_beckman/__init__.py` | Push queue_profile on queue-change events | Modify |
| `packages/general_beckman/src/general_beckman/admission.py` | New module — urgency, threshold, top-K loop | **Create** |
| `packages/general_beckman/src/general_beckman/types.py` | Add `preselected_pick`, `age_seconds`, `downstream_unblocks_count` properties | Modify |
| `src/core/llm_dispatcher.py` | Wrap cloud calls with `begin_call`/`end_call`; write pick_log; record swap; honour preselected_pick | Modify |

---

## Prerequisites

- Sandbox venv at `.venv/Scripts/python.exe` (Windows) / `.venv/bin/python` (POSIX).
- All commands run from repo root `C:\Users\sakir\Dropbox\Workspaces\kutay`.
- Run tests with timeout: `.venv/Scripts/python -m pytest <path> --timeout=30`.

---

## Phase 1 — Hoca purity precondition (Tasks 1–6)

This phase must ship before any pool-pressure work so that Beckman can call `hoca.select()` without polluting telemetry or swap-budget state.

### Task 1: Move `SwapBudget` class to nerd_herd

**Files:**
- Create: `packages/nerd_herd/src/nerd_herd/swap_budget.py`
- Modify: `packages/nerd_herd/src/nerd_herd/__init__.py`
- Test: `packages/nerd_herd/tests/test_swap_budget.py`

- [ ] **Step 1: Write failing test**

```python
# packages/nerd_herd/tests/test_swap_budget.py
import time
import pytest
from nerd_herd.swap_budget import SwapBudget


def test_record_swap_increments():
    sb = SwapBudget(max_swaps=3, window_seconds=300)
    assert sb.recent_count() == 0
    sb.record_swap("llama")
    assert sb.recent_count() == 1


def test_can_swap_false_after_limit():
    sb = SwapBudget(max_swaps=2, window_seconds=300)
    sb.record_swap("a")
    sb.record_swap("b")
    assert sb.can_swap() is False


def test_expired_swaps_pruned():
    sb = SwapBudget(max_swaps=3, window_seconds=1)
    sb.record_swap("a")
    time.sleep(1.1)
    sb.record_swap("b")
    assert sb.recent_count() == 1
```

- [ ] **Step 2: Run test → fail (module missing)**

Run: `.venv/Scripts/python -m pytest packages/nerd_herd/tests/test_swap_budget.py --timeout=10`
Expected: FAIL `ModuleNotFoundError: No module named 'nerd_herd.swap_budget'`

- [ ] **Step 3: Locate the existing `SwapBudget` class**

The class currently lives at `packages/fatih_hoca/src/fatih_hoca/types.py`. Read it and copy to the new module location, with no behavioural change.

- [ ] **Step 4: Create `swap_budget.py`**

```python
# packages/nerd_herd/src/nerd_herd/swap_budget.py
"""Swap-budget state owned by nerd_herd. Hoca reads; dispatcher writes."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class SwapBudget:
    max_swaps: int = 3
    window_seconds: int = 300
    _events: deque = field(default_factory=deque)

    def _prune(self) -> None:
        cutoff = time.time() - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def record_swap(self, model_name: str) -> None:
        self._prune()
        self._events.append((time.time(), model_name))

    def can_swap(self, local_only: bool = False, priority: int = 5) -> bool:
        self._prune()
        return len(self._events) < self.max_swaps

    def recent_count(self) -> int:
        self._prune()
        return len(self._events)

    @property
    def remaining(self) -> int:
        return max(0, self.max_swaps - self.recent_count())
```

- [ ] **Step 5: Run test → pass**

Run: `.venv/Scripts/python -m pytest packages/nerd_herd/tests/test_swap_budget.py --timeout=10 -v`
Expected: 3 passed.

- [ ] **Step 6: Export from nerd_herd `__init__.py`**

Add to `packages/nerd_herd/src/nerd_herd/__init__.py` imports section:
```python
from nerd_herd.swap_budget import SwapBudget
```
And add `"SwapBudget"` to `__all__`.

- [ ] **Step 7: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/swap_budget.py packages/nerd_herd/src/nerd_herd/__init__.py packages/nerd_herd/tests/test_swap_budget.py
git commit -m "feat(nerd_herd): add SwapBudget module (moved from fatih_hoca)

Hoca reads; dispatcher writes. State ownership moves out of selector
to enable Hoca purity refactor.
"
```

---

### Task 2: NerdHerd exposes swap read/write API

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/nerd_herd.py`
- Test: `packages/nerd_herd/tests/test_nerd_herd_swap_api.py`

- [ ] **Step 1: Write failing test**

```python
# packages/nerd_herd/tests/test_nerd_herd_swap_api.py
from nerd_herd.nerd_herd import NerdHerd


def test_nerd_herd_exposes_swap_api():
    nh = NerdHerd(metrics_port=0)    # port 0 = no-op, test-friendly
    assert nh.recent_swap_count() == 0
    nh.record_swap("model_a")
    assert nh.recent_swap_count() == 1
    assert nh.can_swap() is True


def test_nerd_herd_swap_budget_configurable():
    nh = NerdHerd(metrics_port=0)
    for i in range(3):
        nh.record_swap(f"m{i}")
    assert nh.can_swap() is False
```

- [ ] **Step 2: Run test → fail (attribute missing)**

Run: `.venv/Scripts/python -m pytest packages/nerd_herd/tests/test_nerd_herd_swap_api.py --timeout=10`
Expected: FAIL `AttributeError: 'NerdHerd' object has no attribute 'recent_swap_count'`

- [ ] **Step 3: Add SwapBudget to NerdHerd constructor + delegate methods**

Modify `packages/nerd_herd/src/nerd_herd/nerd_herd.py`:

Add at top with other imports:
```python
from nerd_herd.swap_budget import SwapBudget
```

Inside `__init__`, after existing `_local_state`/`_cloud_state` initialization:
```python
self._swap_budget = SwapBudget(max_swaps=3, window_seconds=300)
```

Add methods (anywhere after `push_cloud_state`):
```python
def recent_swap_count(self) -> int:
    return self._swap_budget.recent_count()

def can_swap(self, local_only: bool = False, priority: int = 5) -> bool:
    return self._swap_budget.can_swap(local_only=local_only, priority=priority)

def record_swap(self, model_name: str) -> None:
    self._swap_budget.record_swap(model_name)
```

- [ ] **Step 4: Run test → pass**

Run: `.venv/Scripts/python -m pytest packages/nerd_herd/tests/test_nerd_herd_swap_api.py --timeout=10 -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/nerd_herd.py packages/nerd_herd/tests/test_nerd_herd_swap_api.py
git commit -m "feat(nerd_herd): NerdHerd exposes recent_swap_count/can_swap/record_swap

Delegates to SwapBudget. Read + write surfaces for Hoca and dispatcher
respectively.
"
```

---

### Task 3: Selector reads swap count from nerd_herd (no state of its own)

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/selector.py`
- Modify: `packages/fatih_hoca/src/fatih_hoca/types.py` (remove `SwapBudget` re-export if present)
- Test: `packages/fatih_hoca/tests/test_selector_purity.py`

- [ ] **Step 1: Write failing test (selector does not mutate swap state)**

```python
# packages/fatih_hoca/tests/test_selector_purity.py
from unittest.mock import MagicMock
from fatih_hoca.selector import Selector
from fatih_hoca.registry import ModelRegistry


def test_select_does_not_record_swap():
    nh = MagicMock()
    nh.recent_swap_count.return_value = 0
    nh.can_swap.return_value = True
    nh.record_swap = MagicMock()

    registry = ModelRegistry()
    sel = Selector(registry=registry, nerd_herd=nh, available_providers=set())

    # Any call into select should NOT invoke record_swap.
    try:
        sel.select(task="coder", difficulty=5)
    except Exception:
        pass   # ok if no models; we only assert no record_swap call

    nh.record_swap.assert_not_called()
```

- [ ] **Step 2: Run test → fail (current selector calls `self._swap_budget.record_swap`)**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/test_selector_purity.py --timeout=10`
Expected: FAIL (swap-budget still mutates during select).

- [ ] **Step 3: Remove local `_swap_budget` from Selector; read via `nerd_herd`**

In `packages/fatih_hoca/src/fatih_hoca/selector.py`:

Remove the instance attribute (around line 67):
```python
# BEFORE:
self._swap_budget = SwapBudget(max_swaps=3, window_seconds=300)

# AFTER: delete this line
```

At line 176 (current `if not self._swap_budget.can_swap(...):`) change to:
```python
if not self._nerd_herd.can_swap(local_only=local_only, priority=priority):
```

At line 195 (`self._swap_budget.record_swap()`), DELETE the call entirely. Dispatcher will record post-swap.

At line 198 (`self._swap_budget.remaining`), change to:
```python
self._nerd_herd.recent_swap_count(),
```
(or expose `remaining` on nerd_herd if preferred — but `recent_count` is sufficient for logging)

Remove the `SwapBudget` import at the top of `selector.py` if no other usage remains.

- [ ] **Step 4: Run purity test → pass**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests/test_selector_purity.py --timeout=10 -v`
Expected: pass.

- [ ] **Step 5: Run existing fatih_hoca tests to confirm no regression**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests --timeout=60`
Expected: all pass. If any fail due to missing `SwapBudget` import or missing `self._swap_budget`, update them to use nerd_herd mock.

- [ ] **Step 6: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/selector.py packages/fatih_hoca/tests/test_selector_purity.py
git commit -m "refactor(fatih_hoca): Selector reads swap state from nerd_herd, no longer mutates

Selector is now pure wrt swap budget. Dispatcher will record_swap after
successful execution (next task).
"
```

---

### Task 4: Dispatcher records swap post-execution

**Files:**
- Modify: `src/core/llm_dispatcher.py`
- Test: `tests/core/test_dispatcher_records_swap.py`

- [ ] **Step 1: Write failing test**

```python
# tests/core/test_dispatcher_records_swap.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_dispatcher_records_swap_after_swap(monkeypatch):
    """After dispatcher triggers a local swap via ensure_local_model,
    nerd_herd.record_swap must be called with the loaded model name."""
    from src.core.llm_dispatcher import LLMDispatcher, CallCategory

    # Patch fatih_hoca.select to return a local model that triggers swap.
    fake_model = MagicMock(is_local=True, name="qwen3-8b", location="gguf",
                           provider="local", thinking_model=False)
    fake_pick = MagicMock(model=fake_model, min_time_seconds=30)

    with patch("fatih_hoca.select", return_value=fake_pick), \
         patch("nerd_herd.record_swap") as mock_record, \
         patch("hallederiz_kadir.call", new=AsyncMock(return_value={"ok": True})):
        d = LLMDispatcher()
        # Arrange: ensure_local_model returns True + reports swap_happened=True
        d._ensure_local_model = AsyncMock(return_value=(True, True))
        await d.request(
            category=CallCategory.MAIN_WORK,
            task="coder", difficulty=5, messages=[], tools=None,
        )
        mock_record.assert_called_once_with("qwen3-8b")
```

- [ ] **Step 2: Run → fail**

Run: `.venv/Scripts/python -m pytest tests/core/test_dispatcher_records_swap.py --timeout=20`
Expected: FAIL (dispatcher does not call record_swap today).

- [ ] **Step 3: Modify `_ensure_local_model` signature to return `(ok, swap_happened)`**

In `src/core/llm_dispatcher.py`, locate `_ensure_local_model`. Update to return a tuple signalling whether an actual swap occurred. Minimal change: detect if current loaded model changed from before → after the ensure call.

```python
async def _ensure_local_model(self, model, needs_thinking=False, **kwargs):
    # existing body ...
    before = dallama.currently_loaded_model_name()    # or equivalent
    ok = await dallama.ensure(model, needs_thinking=needs_thinking, **kwargs)
    after = dallama.currently_loaded_model_name()
    swap_happened = ok and (before != after)
    return ok, swap_happened
```

(Adjust to actual API shape of the dallama facade.)

- [ ] **Step 4: Call `nerd_herd.record_swap` when swap happens**

In the dispatcher's local-model branch (around the existing `ensure_local_model` call site), after receiving `(ok, swap_happened)`:

```python
ok, swap_happened = await self._ensure_local_model(model, needs_thinking=is_thinking, ...)
if not ok:
    # existing failure handling
    ...
if swap_happened:
    import nerd_herd
    nerd_herd.record_swap(model.name)
```

Update the rest of the dispatcher to destructure the tuple where `_ensure_local_model` is called.

- [ ] **Step 5: Run → pass**

Run: `.venv/Scripts/python -m pytest tests/core/test_dispatcher_records_swap.py --timeout=20 -v`
Expected: pass.

- [ ] **Step 6: Run wider dispatcher tests to confirm no regression**

Run: `.venv/Scripts/python -m pytest tests/core --timeout=60`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/core/llm_dispatcher.py tests/core/test_dispatcher_records_swap.py
git commit -m "feat(dispatcher): record swap to nerd_herd after successful ensure_local_model

Dispatcher is the only component that executes swaps; it now owns the
write side. Hoca reads the counter via nerd_herd.can_swap().
"
```

---

### Task 5: Move `model_pick_log` write from Selector to dispatcher

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/selector.py` (remove `_log_pick`)
- Modify: `src/core/llm_dispatcher.py` (add pick-log write)
- Create: `src/infra/pick_log.py` (thin helper)
- Test: `tests/infra/test_pick_log.py`, `tests/core/test_dispatcher_pick_log.py`

- [ ] **Step 1: Write failing test for pick_log helper**

```python
# tests/infra/test_pick_log.py
import aiosqlite
import pytest
from src.infra.pick_log import write_pick_log_row


@pytest.mark.asyncio
async def test_write_pick_log_inserts(tmp_path):
    db_path = tmp_path / "test.db"
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE model_pick_log (
                id INTEGER PRIMARY KEY, task_name TEXT, picked_model TEXT,
                picked_score REAL, category TEXT, success INTEGER, timestamp REAL
            )
        """)
        await db.commit()

    await write_pick_log_row(
        db_path=str(db_path),
        task_name="coder", picked_model="qwen3-8b", picked_score=0.72,
        category="main_work", success=True,
    )
    async with aiosqlite.connect(db_path) as db:
        async with db.execute("SELECT task_name, picked_model, success FROM model_pick_log") as cur:
            rows = await cur.fetchall()
    assert rows == [("coder", "qwen3-8b", 1)]
```

- [ ] **Step 2: Run → fail (module missing)**

- [ ] **Step 3: Create `src/infra/pick_log.py`**

```python
# src/infra/pick_log.py
"""Helper for writing model_pick_log rows from dispatcher post-iteration."""
from __future__ import annotations

import time
import aiosqlite

from src.infra.logging_config import get_logger

logger = get_logger("infra.pick_log")


async def write_pick_log_row(
    db_path: str,
    task_name: str,
    picked_model: str,
    picked_score: float,
    category: str,
    success: bool,
    error_category: str = "",
    snapshot_summary: str = "",
) -> None:
    """Fire-and-forget write. Never raises to the caller."""
    try:
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "INSERT INTO model_pick_log "
                "(task_name, picked_model, picked_score, category, success, "
                " error_category, snapshot_summary, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (task_name, picked_model, picked_score, category,
                 1 if success else 0, error_category, snapshot_summary, time.time()),
            )
            await db.commit()
    except Exception as e:
        logger.warning("pick_log write failed: %s", e)
```

Adjust column names to match the current `model_pick_log` schema — check `src/infra/db.py:485` for the exact CREATE TABLE. If there are more columns, fill them with sensible defaults.

- [ ] **Step 4: Run pick_log test → pass**

- [ ] **Step 5: Remove `_log_pick` from selector**

In `packages/fatih_hoca/src/fatih_hoca/selector.py`, delete the entire `_log_pick` method (around lines 244–299) and any call site that invokes it (search for `self._log_pick(`).

- [ ] **Step 6: Write failing test for dispatcher pick_log integration**

```python
# tests/core/test_dispatcher_pick_log.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_dispatcher_writes_pick_log_on_success():
    from src.core.llm_dispatcher import LLMDispatcher, CallCategory

    fake_model = MagicMock(is_local=False, name="claude-sonnet-4-6",
                           provider="anthropic")
    fake_pick = MagicMock(model=fake_model, composite=0.65, min_time_seconds=8)

    writes = []
    async def fake_write(**kw):
        writes.append(kw)

    with patch("fatih_hoca.select", return_value=fake_pick), \
         patch("hallederiz_kadir.call", new=AsyncMock(return_value={"ok": True})), \
         patch("src.infra.pick_log.write_pick_log_row", new=fake_write):
        d = LLMDispatcher()
        await d.request(
            category=CallCategory.MAIN_WORK,
            task="coder", difficulty=7, messages=[], tools=None,
        )

    assert len(writes) == 1
    assert writes[0]["picked_model"] == "claude-sonnet-4-6"
    assert writes[0]["success"] is True
```

- [ ] **Step 7: Run → fail**

- [ ] **Step 8: Wire pick_log write into dispatcher**

In `src/core/llm_dispatcher.py`, inside the iteration loop (around the call to `hallederiz.call(...)`) wrap it so success and failure both write a log row:

```python
from src.infra.pick_log import write_pick_log_row
from os import environ

db_path = environ.get("DB_PATH", "kutai.db")

# inside iteration try/except:
try:
    result = await hallederiz_kadir.call(model=model, ...)
    await write_pick_log_row(
        db_path=db_path,
        task_name=task or agent_type,
        picked_model=model.name,
        picked_score=getattr(pick, "composite", 0.0),
        category=category.value,
        success=True,
    )
    return result
except Exception as e:
    await write_pick_log_row(
        db_path=db_path,
        task_name=task or agent_type,
        picked_model=model.name,
        picked_score=getattr(pick, "composite", 0.0),
        category=category.value,
        success=False,
        error_category=type(e).__name__,
    )
    # existing failure handling
    ...
```

- [ ] **Step 9: Run both tests → pass**

Run: `.venv/Scripts/python -m pytest tests/infra/test_pick_log.py tests/core/test_dispatcher_pick_log.py --timeout=20 -v`
Expected: both pass.

- [ ] **Step 10: Confirm existing Hoca tests do not regress**

Run: `.venv/Scripts/python -m pytest packages/fatih_hoca/tests --timeout=60`

- [ ] **Step 11: Commit**

```bash
git add src/infra/pick_log.py tests/infra/test_pick_log.py \
        packages/fatih_hoca/src/fatih_hoca/selector.py \
        src/core/llm_dispatcher.py tests/core/test_dispatcher_pick_log.py
git commit -m "refactor: model_pick_log writes move from selector to dispatcher

Selector is now side-effect-free. Dispatcher writes the log row after
each iteration with the actual outcome (success/failure), capturing
what was dispatched rather than what was theoretically picked.
"
```

---

### Task 6: Phase 2d regression validation (no code change)

- [ ] **Step 1: Run Phase 2d scenario simulator**

Run: `.venv/Scripts/python packages/fatih_hoca/tests/sim/run_scenarios.py`
Expected: all scenarios green (hard_sat ≥ 90%, easy_waste < 10%, diverse_pool free_q > 70%).

- [ ] **Step 2: Run swap-storm check**

Run: `.venv/Scripts/python packages/fatih_hoca/tests/sim/run_swap_storm_check.py`
Expected: no swap storm.

- [ ] **Step 3: If any regression, pause and diagnose before continuing.** Purity refactor should not alter behaviour. Check whether swap counter is being read correctly via nerd_herd (Task 3) and whether test fixtures stub nerd_herd properly.

---

## Phase 2 — In-flight counter + pool pressure (Tasks 7–13)

### Task 7: Add `in_flight` field to `RateLimit`

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/types.py`
- Test: `packages/nerd_herd/tests/test_types_in_flight.py`

- [ ] **Step 1: Write failing test**

```python
# packages/nerd_herd/tests/test_types_in_flight.py
from nerd_herd.types import RateLimit


def test_rate_limit_default_in_flight_zero():
    rl = RateLimit(limit=100, remaining=50, reset_at=0)
    assert rl.in_flight == 0


def test_rate_limit_accepts_in_flight():
    rl = RateLimit(limit=100, remaining=50, reset_at=0, in_flight=3)
    assert rl.in_flight == 3
```

- [ ] **Step 2: Run → fail (field missing)**

- [ ] **Step 3: Add field to `RateLimit` dataclass**

In `packages/nerd_herd/src/nerd_herd/types.py`, modify `RateLimit`:

```python
@dataclass
class RateLimit:
    limit: int | None = None
    remaining: int | None = None
    reset_at: int | None = None        # absolute epoch seconds
    in_flight: int = 0                 # calls dispatched but not yet confirmed
```

- [ ] **Step 4: Run → pass.**

- [ ] **Step 5: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/types.py packages/nerd_herd/tests/test_types_in_flight.py
git commit -m "feat(nerd_herd): add in_flight field to RateLimit

Tracks cloud calls dispatched but not yet confirmed. Populated by kdv
push. Consumed by pool_pressure computation.
"
```

---

### Task 8: `kdv.begin_call` / `kdv.end_call` API

**Files:**
- Create: `packages/kuleden_donen_var/src/kuleden_donen_var/in_flight.py`
- Modify: `packages/kuleden_donen_var/src/kuleden_donen_var/__init__.py`
- Test: `packages/kuleden_donen_var/tests/test_in_flight.py`

- [ ] **Step 1: Write failing test**

```python
# packages/kuleden_donen_var/tests/test_in_flight.py
import time
import pytest
from kuleden_donen_var.in_flight import InFlightTracker, InFlightHandle


def test_begin_call_increments_count():
    t = InFlightTracker()
    h = t.begin_call("anthropic", "claude-sonnet-4-6", ttl_s=60)
    assert t.count("anthropic", "claude-sonnet-4-6") == 1


def test_end_call_decrements_count():
    t = InFlightTracker()
    h = t.begin_call("anthropic", "claude-sonnet-4-6")
    t.end_call(h)
    assert t.count("anthropic", "claude-sonnet-4-6") == 0


def test_ttl_prunes_expired():
    t = InFlightTracker()
    h = t.begin_call("anthropic", "claude-sonnet-4-6", ttl_s=0.01)
    time.sleep(0.05)
    t.begin_call("anthropic", "claude-sonnet-4-6")     # triggers prune
    assert t.count("anthropic", "claude-sonnet-4-6") == 1


def test_end_call_is_idempotent():
    t = InFlightTracker()
    h = t.begin_call("anthropic", "claude-sonnet-4-6")
    t.end_call(h)
    t.end_call(h)   # second call is a no-op
    assert t.count("anthropic", "claude-sonnet-4-6") == 0
```

- [ ] **Step 2: Run → fail (module missing)**

- [ ] **Step 3: Create `in_flight.py`**

```python
# packages/kuleden_donen_var/src/kuleden_donen_var/in_flight.py
"""In-flight tracker for cloud calls. TTL safety net for crash-leaked handles."""
from __future__ import annotations

import time
import uuid
import os
from dataclasses import dataclass, field

DEFAULT_TTL_S = float(os.environ.get("KDV_INFLIGHT_TTL_S", "180"))


@dataclass(frozen=True)
class InFlightHandle:
    provider: str
    model: str
    started_at: float
    ttl_s: float
    token: str


class InFlightTracker:
    def __init__(self) -> None:
        # (provider, model) -> list[InFlightHandle]
        self._handles: dict[tuple[str, str], list[InFlightHandle]] = {}

    def _prune(self, key: tuple[str, str]) -> None:
        now = time.time()
        bucket = self._handles.get(key)
        if not bucket:
            return
        self._handles[key] = [h for h in bucket if h.started_at + h.ttl_s > now]

    def begin_call(self, provider: str, model: str, ttl_s: float = DEFAULT_TTL_S) -> InFlightHandle:
        key = (provider, model)
        self._prune(key)
        h = InFlightHandle(
            provider=provider, model=model,
            started_at=time.time(), ttl_s=ttl_s, token=str(uuid.uuid4()),
        )
        self._handles.setdefault(key, []).append(h)
        return h

    def end_call(self, handle: InFlightHandle) -> None:
        key = (handle.provider, handle.model)
        bucket = self._handles.get(key, [])
        self._handles[key] = [h for h in bucket if h.token != handle.token]

    def count(self, provider: str, model: str) -> int:
        key = (provider, model)
        self._prune(key)
        return len(self._handles.get(key, []))
```

- [ ] **Step 4: Run → pass.**

- [ ] **Step 5: Export from package `__init__.py`**

Append to `packages/kuleden_donen_var/src/kuleden_donen_var/__init__.py`:
```python
from kuleden_donen_var.in_flight import InFlightTracker, InFlightHandle

# Module-level singleton used by dispatcher
_in_flight_tracker = InFlightTracker()

def begin_call(provider: str, model: str, ttl_s: float | None = None) -> InFlightHandle:
    kwargs = {}
    if ttl_s is not None:
        kwargs["ttl_s"] = ttl_s
    return _in_flight_tracker.begin_call(provider, model, **kwargs)

def end_call(handle: InFlightHandle) -> None:
    _in_flight_tracker.end_call(handle)

def in_flight_count(provider: str, model: str) -> int:
    return _in_flight_tracker.count(provider, model)
```

- [ ] **Step 6: Commit**

```bash
git add packages/kuleden_donen_var/src/kuleden_donen_var/in_flight.py \
        packages/kuleden_donen_var/src/kuleden_donen_var/__init__.py \
        packages/kuleden_donen_var/tests/test_in_flight.py
git commit -m "feat(kdv): begin_call/end_call API with TTL-backed in-flight tracker"
```

---

### Task 9: kdv pushes `in_flight` to nerd_herd on each boundary

**Files:**
- Modify: `packages/kuleden_donen_var/src/kuleden_donen_var/in_flight.py` (push on begin/end)
- Modify: `packages/kuleden_donen_var/src/kuleden_donen_var/__init__.py` (wire pushes)
- Test: `packages/kuleden_donen_var/tests/test_in_flight_pushes.py`

- [ ] **Step 1: Write failing test**

```python
# packages/kuleden_donen_var/tests/test_in_flight_pushes.py
from unittest.mock import MagicMock
from kuleden_donen_var.in_flight import InFlightTracker


def test_begin_call_pushes_to_nerd_herd():
    nh = MagicMock()
    t = InFlightTracker(nerd_herd=nh)
    t.begin_call("anthropic", "claude-sonnet-4-6")
    assert nh.push_cloud_state.called
    state = nh.push_cloud_state.call_args[0][0]
    # Expect anthropic provider state updated with in_flight=1 for the model
    assert state.provider == "anthropic"
    model_state = state.models.get("claude-sonnet-4-6")
    assert model_state is not None
    assert model_state.limits.rpd.in_flight == 1
```

- [ ] **Step 2: Run → fail**

- [ ] **Step 3: Wire push**

Modify `InFlightTracker.__init__` to accept `nerd_herd` and keep a reference to KDV's current provider state to reconstruct the push:

```python
def __init__(self, nerd_herd=None, state_getter=None):
    self._handles = {}
    self._nerd_herd = nerd_herd
    # state_getter(provider) -> current CloudProviderState (read from kdv)
    self._state_getter = state_getter

def _push(self, provider: str) -> None:
    if self._nerd_herd is None or self._state_getter is None:
        return
    state = self._state_getter(provider)
    if state is None:
        return
    # Overlay in_flight on each model's limits.rpd
    for model_name, model_state in state.models.items():
        model_state.limits.rpd.in_flight = self.count(provider, model_name)
    # Also update provider-level if applicable
    state.limits.rpd.in_flight = sum(
        self.count(provider, m) for m in state.models
    )
    self._nerd_herd.push_cloud_state(state)
```

Call `self._push(handle.provider)` from both `begin_call` and `end_call`.

- [ ] **Step 4: Wire into package singleton**

In `__init__.py`, construct the module-level tracker with references to nerd_herd and kdv's state store:
```python
# Lazy wiring — resolve nerd_herd and state_getter at first call to avoid import cycles
_in_flight_tracker: InFlightTracker | None = None

def _get_tracker() -> InFlightTracker:
    global _in_flight_tracker
    if _in_flight_tracker is None:
        from kuleden_donen_var.state_store import get_provider_state   # existing kdv module
        import nerd_herd
        _in_flight_tracker = InFlightTracker(
            nerd_herd=nerd_herd,
            state_getter=get_provider_state,
        )
    return _in_flight_tracker

def begin_call(provider, model, ttl_s=None):
    kwargs = {"ttl_s": ttl_s} if ttl_s is not None else {}
    return _get_tracker().begin_call(provider, model, **kwargs)

def end_call(handle):
    _get_tracker().end_call(handle)

def in_flight_count(provider, model):
    return _get_tracker().count(provider, model)
```

Adjust import paths to match real KDV internal module names.

- [ ] **Step 5: Run → pass.**

- [ ] **Step 6: Commit**

```bash
git add packages/kuleden_donen_var
git commit -m "feat(kdv): push updated in_flight to nerd_herd on begin/end_call

Nerd_herd receives live in-flight overlay alongside confirmed rate-limit
state. Enables pool_pressure computation to account for mid-flight calls.
"
```

---

### Task 10: `compute_pool_pressure` pure function in nerd_herd

**Files:**
- Create: `packages/nerd_herd/src/nerd_herd/pool_pressure.py`
- Test: `packages/nerd_herd/tests/test_pool_pressure.py`

- [ ] **Step 1: Write failing test (per arm)**

```python
# packages/nerd_herd/tests/test_pool_pressure.py
import time
from nerd_herd.pool_pressure import compute_pool_pressure


def test_depletion_dominates_below_15pct():
    now = int(time.time())
    p = compute_pool_pressure(remaining=5, limit=100, reset_at=now + 3600, in_flight_count=0)
    assert p.value < -0.5
    assert p.depletion < 0


def test_abundance_peaks_near_reset():
    now = int(time.time())
    p = compute_pool_pressure(remaining=90, limit=100, reset_at=now + 600, in_flight_count=0)
    assert p.value > 0.8
    assert p.abundance > 0.8


def test_no_reset_at_no_time_weight():
    p = compute_pool_pressure(remaining=90, limit=100, reset_at=None, in_flight_count=0)
    assert p.time_weight == 0.0
    assert p.abundance == 0.0


def test_in_flight_reduces_effective_remaining():
    now = int(time.time())
    p1 = compute_pool_pressure(remaining=30, limit=100, reset_at=now + 3600, in_flight_count=0)
    p2 = compute_pool_pressure(remaining=30, limit=100, reset_at=now + 3600, in_flight_count=20)
    # 10/100 = 10% → depletion activates → more negative value
    assert p2.value < p1.value


def test_zero_limit_returns_neutral():
    p = compute_pool_pressure(remaining=0, limit=0, reset_at=None, in_flight_count=0)
    assert p.value == 0.0
```

- [ ] **Step 2: Run → fail (module missing)**

- [ ] **Step 3: Create module**

```python
# packages/nerd_herd/src/nerd_herd/pool_pressure.py
"""Pure pool-pressure computation. Consumed by SystemSnapshot.pressure_for()."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

DEPLETION_THRESHOLD = 0.15
TIME_SCALE_SECS = 86400.0


@dataclass
class PoolPressure:
    value: float
    depletion: float
    abundance: float
    time_weight: float
    in_flight_count: int


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def compute_pool_pressure(
    remaining: int | None,
    limit: int | None,
    reset_at: int | None,
    in_flight_count: int,
) -> PoolPressure:
    """Derive signed pressure [-1, +1] from pool state.

    Inputs:
      remaining     confirmed remaining quota
      limit         total quota in current window
      reset_at      absolute epoch seconds of next reset; None → no time weight
      in_flight_count  calls dispatched but not yet confirmed
    """
    if not limit or limit <= 0:
        return PoolPressure(0.0, 0.0, 0.0, 0.0, in_flight_count)
    effective = max(0, (remaining or 0) - in_flight_count)
    remaining_frac = min(1.0, effective / limit)

    depletion = 0.0
    abundance = 0.0
    time_weight = 0.0

    if remaining_frac < DEPLETION_THRESHOLD:
        intensity = (DEPLETION_THRESHOLD - remaining_frac) / DEPLETION_THRESHOLD
        depletion = _clamp(-1.0 * intensity, -1.0, 0.0)
    elif reset_at is not None and reset_at > 0:
        reset_in = max(0.0, reset_at - time.time())
        time_weight = math.exp(-reset_in / TIME_SCALE_SECS)
        abundance = _clamp(remaining_frac * time_weight, 0.0, 1.0)

    value = _clamp(depletion + abundance)
    return PoolPressure(
        value=value,
        depletion=depletion,
        abundance=abundance,
        time_weight=time_weight,
        in_flight_count=in_flight_count,
    )
```

- [ ] **Step 4: Run tests → pass.**

- [ ] **Step 5: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/pool_pressure.py packages/nerd_herd/tests/test_pool_pressure.py
git commit -m "feat(nerd_herd): pure compute_pool_pressure function

Depletion (below 15% effective remaining) and abundance (scaled by
exp(-reset_in/24h)) arms. Returns PoolPressure dataclass for telemetry.
"
```

---

### Task 11: Attach `pool_pressure` to `CloudModelState` (lazy cache) + `pressure_for()` accessor

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/types.py`
- Test: `packages/nerd_herd/tests/test_pressure_for.py`

- [ ] **Step 1: Write failing test**

```python
# packages/nerd_herd/tests/test_pressure_for.py
import time
from unittest.mock import MagicMock
from nerd_herd.types import (
    SystemSnapshot, CloudProviderState, CloudModelState, RateLimit, RateLimits,
    LocalModelState,
)


def _snap_with_cloud(provider, model_name, remaining, limit, reset_at, in_flight=0):
    m = CloudModelState(model_id=model_name)
    m.limits.rpd = RateLimit(limit=limit, remaining=remaining, reset_at=reset_at, in_flight=in_flight)
    prov = CloudProviderState(provider=provider, models={model_name: m})
    return SystemSnapshot(cloud={provider: prov})


def test_pressure_for_cloud_model_depletion_negative():
    snap = _snap_with_cloud("anthropic", "claude-sonnet-4-6",
                            remaining=5, limit=100, reset_at=int(time.time()) + 3600)
    fake = MagicMock(is_local=False, provider="anthropic", name="claude-sonnet-4-6")
    assert snap.pressure_for(fake) < -0.5


def test_pressure_for_missing_model_returns_zero():
    snap = SystemSnapshot()
    fake = MagicMock(is_local=False, provider="unknown", name="x")
    assert snap.pressure_for(fake) == 0.0


def test_pressure_for_cached_after_first_read():
    snap = _snap_with_cloud("anthropic", "claude-sonnet-4-6",
                            remaining=50, limit=100, reset_at=int(time.time()) + 3600)
    fake = MagicMock(is_local=False, provider="anthropic", name="claude-sonnet-4-6")
    _ = snap.pressure_for(fake)
    first_obj = snap.cloud["anthropic"].models["claude-sonnet-4-6"].pool_pressure
    _ = snap.pressure_for(fake)
    second_obj = snap.cloud["anthropic"].models["claude-sonnet-4-6"].pool_pressure
    assert first_obj is second_obj   # same PoolPressure instance — cache hit


def test_pressure_for_local_busy_negative_or_zero():
    snap = SystemSnapshot(local=LocalModelState(model_name="qwen3-8b"))
    fake = MagicMock(is_local=True, name="qwen3-8b")
    val = snap.pressure_for(fake)
    assert -1.0 <= val <= 1.0
```

- [ ] **Step 2: Run → fail (field missing + method missing)**

- [ ] **Step 3: Extend `CloudModelState` with `pool_pressure` and add accessor**

Modify `packages/nerd_herd/src/nerd_herd/types.py`:

```python
# Add import at top:
from nerd_herd.pool_pressure import PoolPressure, compute_pool_pressure


@dataclass
class CloudModelState:
    model_id: str = ""
    utilization_pct: float = 0.0
    limits: RateLimits = field(default_factory=RateLimits)
    pool_pressure: PoolPressure | None = None   # lazy-cached by pressure_for()
```

Add method to `SystemSnapshot`:

```python
@dataclass
class SystemSnapshot:
    vram_available_mb: int = 0
    local: LocalModelState = field(default_factory=LocalModelState)
    cloud: dict[str, CloudProviderState] = field(default_factory=dict)

    def pressure_for(self, model) -> float:
        """Signed pool pressure [-1, +1] for model. 0.0 if data missing."""
        if getattr(model, "is_local", False):
            return self._local_pressure()
        provider = getattr(model, "provider", "")
        prov = self.cloud.get(provider)
        if prov is None:
            return 0.0
        model_id = getattr(model, "name", "")
        m = prov.models.get(model_id)
        if m is None:
            # Fall back to provider-level limits (anthropic case)
            if m is None and prov.limits.rpd.limit:
                return compute_pool_pressure(
                    remaining=prov.limits.rpd.remaining,
                    limit=prov.limits.rpd.limit,
                    reset_at=prov.limits.rpd.reset_at,
                    in_flight_count=prov.limits.rpd.in_flight,
                ).value
            return 0.0
        if m.pool_pressure is None:
            m.pool_pressure = compute_pool_pressure(
                remaining=m.limits.rpd.remaining,
                limit=m.limits.rpd.limit,
                reset_at=m.limits.rpd.reset_at,
                in_flight_count=m.limits.rpd.in_flight,
            )
        return m.pool_pressure.value

    def _local_pressure(self) -> float:
        """Derive local lane pressure from LocalModelState.

        idle + loaded → mildly positive (can handle work).
        busy or swapping → negative (don't dispatch more).
        """
        if self.local is None or self.local.model_name is None:
            return 0.0
        if self.local.is_swapping:
            return -0.5
        idle = self.local.idle_seconds or 0.0
        if idle <= 0:
            return -0.2    # actively processing
        return min(0.3, idle / 60.0 * 0.3)
```

- [ ] **Step 4: Run tests → pass.**

- [ ] **Step 5: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/types.py packages/nerd_herd/tests/test_pressure_for.py
git commit -m "feat(nerd_herd): SystemSnapshot.pressure_for(model) with lazy cache

Returns signed pool pressure in [-1, +1]. First read per snapshot instance
computes and caches; subsequent reads hit cache. Local model derives from
LocalModelState; cloud from rpd + in_flight.
"
```

---

### Task 12: Dispatcher wraps cloud calls with `begin_call`/`end_call`

**Files:**
- Modify: `src/core/llm_dispatcher.py`
- Test: `tests/core/test_dispatcher_in_flight.py`

- [ ] **Step 1: Write failing test**

```python
# tests/core/test_dispatcher_in_flight.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_dispatcher_calls_begin_end_for_cloud():
    from src.core.llm_dispatcher import LLMDispatcher, CallCategory

    fake_model = MagicMock(is_local=False, provider="anthropic",
                           name="claude-sonnet-4-6")
    fake_pick = MagicMock(model=fake_model, composite=0.7)

    with patch("fatih_hoca.select", return_value=fake_pick), \
         patch("kuleden_donen_var.begin_call") as mock_begin, \
         patch("kuleden_donen_var.end_call") as mock_end, \
         patch("hallederiz_kadir.call", new=AsyncMock(return_value={"ok": True})), \
         patch("src.infra.pick_log.write_pick_log_row", new=AsyncMock()):
        mock_begin.return_value = MagicMock()
        d = LLMDispatcher()
        await d.request(
            category=CallCategory.MAIN_WORK,
            task="coder", difficulty=7, messages=[], tools=None,
        )
    mock_begin.assert_called_once_with("anthropic", "claude-sonnet-4-6")
    mock_end.assert_called_once()


@pytest.mark.asyncio
async def test_dispatcher_ends_call_even_on_exception():
    from src.core.llm_dispatcher import LLMDispatcher, CallCategory

    fake_model = MagicMock(is_local=False, provider="anthropic",
                           name="claude-sonnet-4-6")
    fake_pick = MagicMock(model=fake_model, composite=0.7)

    async def raise_err(*a, **k):
        raise RuntimeError("boom")

    with patch("fatih_hoca.select", return_value=fake_pick), \
         patch("kuleden_donen_var.begin_call") as mock_begin, \
         patch("kuleden_donen_var.end_call") as mock_end, \
         patch("hallederiz_kadir.call", new=raise_err), \
         patch("src.infra.pick_log.write_pick_log_row", new=AsyncMock()):
        mock_begin.return_value = MagicMock()
        d = LLMDispatcher()
        with pytest.raises(Exception):
            await d.request(
                category=CallCategory.MAIN_WORK,
                task="coder", difficulty=7, messages=[], tools=None,
            )
    mock_end.assert_called()
```

- [ ] **Step 2: Run → fail**

- [ ] **Step 3: Add `begin_call`/`end_call` wrapping**

In `src/core/llm_dispatcher.py`, inside the iteration path after model selection:

```python
import kuleden_donen_var
...
model = pick.model

if model.is_local:
    # existing local path (ensure_local_model + record_swap)
    ok, swap_happened = await self._ensure_local_model(model, needs_thinking=is_thinking, ...)
    if not ok:
        ... # failure handling as before
    if swap_happened:
        nerd_herd.record_swap(model.name)
    try:
        result = await hallederiz_kadir.call(model=model, ...)
        await write_pick_log_row(..., success=True)
        return result
    except Exception as e:
        await write_pick_log_row(..., success=False, error_category=type(e).__name__)
        ... # failure + retry
else:
    # cloud path — wrap with in-flight tracking
    handle = kuleden_donen_var.begin_call(model.provider, model.name)
    try:
        result = await hallederiz_kadir.call(model=model, ...)
        await write_pick_log_row(..., success=True)
        return result
    except Exception as e:
        await write_pick_log_row(..., success=False, error_category=type(e).__name__)
        ... # failure + retry
    finally:
        kuleden_donen_var.end_call(handle)
```

- [ ] **Step 4: Run tests → pass.**

- [ ] **Step 5: Regression**

Run: `.venv/Scripts/python -m pytest tests/core --timeout=60`

- [ ] **Step 6: Commit**

```bash
git add src/core/llm_dispatcher.py tests/core/test_dispatcher_in_flight.py
git commit -m "feat(dispatcher): wrap cloud calls with kdv.begin_call/end_call

Every cloud iteration registers an in-flight handle before the wire
call and releases in try/finally. Prevents t=0 burn on bursty dispatch.
"
```

---

### Task 13: Fatih Hoca scarcity consumes `snapshot.pressure_for()`; queue arm stays

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/scarcity.py`
- Test: existing scarcity tests + Phase 2d regression

- [ ] **Step 1: Read current `scarcity.py` and identify replacement points**

`_time_bucketed_scarcity` and `_per_call_scarcity` both contain direct `rpd.remaining / limit / reset_at` reads. These should be replaced by a single call to `snapshot.pressure_for(model)`.

The **queue arm** inside `_per_call_scarcity` (the `hard_tasks_count` / `total_tasks` logic) stays — it needs queue_profile from snapshot.

- [ ] **Step 2: Refactor `pool_scarcity` entry point**

```python
# packages/fatih_hoca/src/fatih_hoca/scarcity.py

def pool_scarcity(
    model, snapshot, queue_state=None, task_difficulty: int = 0,
) -> float:
    pool = classify_pool(model)
    if pool is Pool.LOCAL:
        return _local_scarcity(model, snapshot)
    # Supply-side signal via shared primitive
    supply = snapshot.pressure_for(model)
    # Layer queue-aware modulation on top for per_call pool (demand-side)
    if pool is Pool.PER_CALL:
        queue_term = _queue_arm(queue_state, task_difficulty)
        # conservation only; cannot boost beyond supply
        return _clamp(supply + queue_term)
    # time_bucketed pools: supply signal alone
    return supply


def _queue_arm(queue_state, task_difficulty: int) -> float:
    """Demand-side conservation when easy task sees a hard-ready-queue ahead."""
    if queue_state is None or task_difficulty >= 7:
        return 0.0
    total = int(getattr(queue_state, "total_ready_count", 0)
                or getattr(queue_state, "total_tasks", 0) or 0)
    hard = int(getattr(queue_state, "hard_tasks_count", 0) or 0)
    if total <= 0 or hard <= 0:
        return 0.0
    hard_ratio = hard / total
    pressure = min(1.0, hard_ratio / 0.1)
    easiness = max(0.0, (7 - task_difficulty)) / 6.0
    return -1.0 * pressure * easiness
```

Delete or simplify `_time_bucketed_scarcity` and the old supply-side arms of `_per_call_scarcity` — they are now consumed from `snapshot.pressure_for()`. `_local_scarcity` stays.

- [ ] **Step 3: Run Phase 2d regression**

Run: `.venv/Scripts/python packages/fatih_hoca/tests/sim/run_scenarios.py`
Expected: green — this refactor is identity-preserving for supply-side arms, only relocates code.

- [ ] **Step 4: If scenarios regress, pause and diagnose.** Likely causes: queue_arm weight differs from old arm; compute_pool_pressure arm thresholds differ from old values. Adjust to match.

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/scarcity.py
git commit -m "refactor(fatih_hoca): scarcity supply-side arms move to nerd_herd

Hoca now consumes snapshot.pressure_for(model) for depletion/abundance
signals. Queue arm (demand-side conservation) stays in Hoca, feeding
off snapshot.queue_profile (wired in Task 15). Phase 2d scenarios
unchanged.
"
```

---

## Phase 3 — Queue profile + Beckman admission (Tasks 14–21)

### Task 14: `QueueProfile` dataclass + `NerdHerd.push_queue_profile` receiver

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/types.py`
- Modify: `packages/nerd_herd/src/nerd_herd/nerd_herd.py`
- Test: `packages/nerd_herd/tests/test_queue_profile.py`

- [ ] **Step 1: Write failing test**

```python
# packages/nerd_herd/tests/test_queue_profile.py
from nerd_herd.nerd_herd import NerdHerd
from nerd_herd.types import QueueProfile


def test_push_queue_profile_stored_and_exposed():
    nh = NerdHerd(metrics_port=0)
    nh.push_queue_profile(QueueProfile(hard_tasks_count=4, total_ready_count=12))
    snap = nh.snapshot()
    assert snap.queue_profile is not None
    assert snap.queue_profile.hard_tasks_count == 4
    assert snap.queue_profile.total_ready_count == 12


def test_queue_profile_none_by_default():
    nh = NerdHerd(metrics_port=0)
    assert nh.snapshot().queue_profile is None
```

- [ ] **Step 2: Run → fail**

- [ ] **Step 3: Add QueueProfile dataclass**

In `packages/nerd_herd/src/nerd_herd/types.py`:
```python
@dataclass
class QueueProfile:
    hard_tasks_count: int = 0
    total_ready_count: int = 0
```

Add to `SystemSnapshot`:
```python
@dataclass
class SystemSnapshot:
    vram_available_mb: int = 0
    local: LocalModelState = field(default_factory=LocalModelState)
    cloud: dict[str, CloudProviderState] = field(default_factory=dict)
    queue_profile: QueueProfile | None = None   # NEW
    # pressure_for method unchanged
```

In `nerd_herd.py`:
```python
self._queue_profile: QueueProfile | None = None

def push_queue_profile(self, profile: QueueProfile) -> None:
    self._queue_profile = profile

def snapshot(self) -> SystemSnapshot:
    gpu = self._gpu.gpu_state()
    return SystemSnapshot(
        vram_available_mb=self.get_vram_budget_mb() if gpu.available else 0,
        local=self._local_state,
        cloud=dict(self._cloud_state),
        queue_profile=self._queue_profile,
    )
```

Export from `__init__.py`: add `"QueueProfile"` to `__all__` + import.

- [ ] **Step 4: Run → pass.**

- [ ] **Step 5: Commit**

```bash
git add packages/nerd_herd
git commit -m "feat(nerd_herd): QueueProfile type + push_queue_profile receiver"
```

---

### Task 15: Beckman pushes queue_profile on queue-change events

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/__init__.py`
- Modify: `packages/general_beckman/src/general_beckman/apply.py` (hook into enqueue/on_task_finished)
- Create: `packages/general_beckman/src/general_beckman/queue_profile_push.py`
- Test: `packages/general_beckman/tests/test_queue_profile_push.py`

- [ ] **Step 1: Write failing test**

```python
# packages/general_beckman/tests/test_queue_profile_push.py
import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.asyncio
async def test_enqueue_triggers_queue_profile_push(monkeypatch):
    import general_beckman

    pushed = []
    with patch("nerd_herd.push_queue_profile", side_effect=lambda p: pushed.append(p)):
        # Simulate enqueue via public API
        await general_beckman.enqueue({"task_type": "test", "priority": 5, "difficulty": 3})

    assert len(pushed) >= 1
    assert hasattr(pushed[0], "hard_tasks_count")
    assert hasattr(pushed[0], "total_ready_count")
```

(Adjust enqueue args to the real Beckman signature; may require a temp DB fixture.)

- [ ] **Step 2: Run → fail**

- [ ] **Step 3: Create pusher helper**

```python
# packages/general_beckman/src/general_beckman/queue_profile_push.py
"""Derive current QueueProfile from queue tables and push to nerd_herd."""
from __future__ import annotations

import aiosqlite
from nerd_herd.types import QueueProfile


async def build_and_push(db_path: str, nerd_herd_module) -> None:
    """Called after queue-change events (enqueue/complete/sweep)."""
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT "
            " SUM(CASE WHEN difficulty >= 7 THEN 1 ELSE 0 END) AS hard,"
            " COUNT(*) AS total "
            "FROM task_queue "
            "WHERE status='ready'"
        ) as cur:
            row = await cur.fetchone()
    hard = int(row[0] or 0)
    total = int(row[1] or 0)
    profile = QueueProfile(hard_tasks_count=hard, total_ready_count=total)
    nerd_herd_module.push_queue_profile(profile)
```

(Adjust table + column names to match Beckman's actual schema — verify with `src/infra/db.py` CREATE TABLE for tasks.)

- [ ] **Step 4: Hook into `enqueue`, `on_task_finished`, and `sweep_queue`**

In `packages/general_beckman/src/general_beckman/__init__.py`:

```python
from general_beckman.queue_profile_push import build_and_push as _push_queue_profile
import nerd_herd
import os

_DB_PATH = os.environ.get("DB_PATH", "kutai.db")

async def enqueue(spec: dict) -> int:
    # existing body...
    new_id = ...
    await _push_queue_profile(_DB_PATH, nerd_herd)
    return new_id

async def on_task_finished(task_id: int, result: dict) -> None:
    # existing body...
    await _push_queue_profile(_DB_PATH, nerd_herd)
```

Do the same in `sweep_queue` entry point (see `sweep.py` for the public sweep function).

- [ ] **Step 5: Run → pass.**

- [ ] **Step 6: Commit**

```bash
git add packages/general_beckman
git commit -m "feat(beckman): push QueueProfile to nerd_herd on queue-change events

Pushed after enqueue, on_task_finished, and sweep. Feeds Hoca's queue
arm and Beckman's own admission threshold (next task).
"
```

---

### Task 16: Hoca scarcity queue arm reads from `snapshot.queue_profile`

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/scarcity.py`
- Modify: `packages/fatih_hoca/src/fatih_hoca/ranking.py` (remove queue_state argument)
- Test: existing Phase 2d tests

- [ ] **Step 1: Update `pool_scarcity` signature**

```python
def pool_scarcity(model, snapshot, task_difficulty: int = 0) -> float:
    pool = classify_pool(model)
    if pool is Pool.LOCAL:
        return _local_scarcity(model, snapshot)
    supply = snapshot.pressure_for(model)
    if pool is Pool.PER_CALL:
        queue_term = _queue_arm(snapshot.queue_profile, task_difficulty)
        return _clamp(supply + queue_term)
    return supply
```

- [ ] **Step 2: Remove `queue_state` plumbing in `ranking.py`**

Find callers of `pool_scarcity` in `ranking.py` and elsewhere. Remove `queue_state` kwarg; rely on `snapshot.queue_profile`.

- [ ] **Step 3: Run Phase 2d regression**

Run: `.venv/Scripts/python packages/fatih_hoca/tests/sim/run_scenarios.py`
Expected: green.

- [ ] **Step 4: Commit**

```bash
git add packages/fatih_hoca
git commit -m "refactor(fatih_hoca): queue arm reads from snapshot.queue_profile

Drops the queue_state kwarg through the ranking call chain; uses the
shared snapshot field populated by Beckman pushes.
"
```

---

### Task 17: `Task.preselected_pick` + derived urgency fields

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/types.py`
- Test: `packages/general_beckman/tests/test_task_urgency_fields.py`

- [ ] **Step 1: Write failing test**

```python
# packages/general_beckman/tests/test_task_urgency_fields.py
import time
from general_beckman.types import Task


def test_task_age_seconds_derived_from_created_at():
    t = Task(id=1, created_at=time.time() - 120, priority=5, difficulty=3)
    assert 110 < t.age_seconds < 130


def test_task_preselected_pick_defaults_none():
    t = Task(id=1, priority=5, difficulty=3)
    assert t.preselected_pick is None


def test_task_downstream_unblocks_count_default_zero():
    t = Task(id=1, priority=5, difficulty=3)
    assert t.downstream_unblocks_count == 0
```

- [ ] **Step 2: Run → fail**

- [ ] **Step 3: Extend Task dataclass**

In `packages/general_beckman/src/general_beckman/types.py`:

```python
from dataclasses import dataclass, field
from typing import Any
import time


@dataclass
class Task:
    id: int = 0
    priority: int = 5
    difficulty: int = 5
    created_at: float = field(default_factory=time.time)
    # ... existing fields ...
    preselected_pick: Any = None               # fatih_hoca.Pick attached by admission
    downstream_unblocks_count: int = 0         # populated by admission from dep graph

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - (self.created_at or time.time()))
```

(Adjust to real current Task shape — likely already has `id`, `created_at`, `priority`, `difficulty`. Keep all existing fields; only add new ones.)

- [ ] **Step 4: Run → pass.**

- [ ] **Step 5: Commit**

```bash
git add packages/general_beckman/src/general_beckman/types.py packages/general_beckman/tests/test_task_urgency_fields.py
git commit -m "feat(beckman): Task.preselected_pick + age_seconds + downstream_unblocks_count"
```

---

### Task 18: `_compute_urgency` and `_threshold` pure helpers

**Files:**
- Create: `packages/general_beckman/src/general_beckman/admission.py`
- Test: `packages/general_beckman/tests/test_admission_urgency.py`

- [ ] **Step 1: Write failing test**

```python
# packages/general_beckman/tests/test_admission_urgency.py
import time
from general_beckman.admission import compute_urgency, threshold


def _task(priority=5, difficulty=3, age_s=0, unblocks=0):
    from general_beckman.types import Task
    return Task(id=1, priority=priority, difficulty=difficulty,
                created_at=time.time() - age_s,
                downstream_unblocks_count=unblocks)


def test_priority_5_baseline():
    u = compute_urgency(_task(priority=5))
    assert abs(u - 0.5) < 0.01


def test_age_scales_over_24h():
    u0 = compute_urgency(_task(priority=5, age_s=0))
    u1 = compute_urgency(_task(priority=5, age_s=86400))
    assert u1 > u0
    assert u1 - u0 <= 0.05 + 1e-6


def test_blocker_bump_capped():
    u0 = compute_urgency(_task(priority=5, unblocks=0))
    u1 = compute_urgency(_task(priority=5, unblocks=5))
    u2 = compute_urgency(_task(priority=5, unblocks=50))
    assert u2 == u1     # capped at 5


def test_urgency_clamped_0_1():
    u = compute_urgency(_task(priority=10, age_s=10**9, unblocks=10**6))
    assert u <= 1.0


def test_threshold_linear():
    assert abs(threshold(0.0) - 0.5) < 1e-6
    assert abs(threshold(0.5) - 0.0) < 1e-6
    assert abs(threshold(1.0) - (-0.5)) < 1e-6
```

- [ ] **Step 2: Run → fail**

- [ ] **Step 3: Create module**

```python
# packages/general_beckman/src/general_beckman/admission.py
"""Admission policy: urgency composition and threshold function."""
from __future__ import annotations

from general_beckman.types import Task


AGE_SCALE_S = 86400.0
AGE_WEIGHT = 0.05
BLOCKER_CAP = 5
BLOCKER_WEIGHT = 0.05


def compute_urgency(task: Task) -> float:
    priority_term = (task.priority or 5) / 10.0
    age_term = min(1.0, task.age_seconds / AGE_SCALE_S) * AGE_WEIGHT
    unblocks = task.downstream_unblocks_count or 0
    blocker_term = min(1.0, unblocks / BLOCKER_CAP) * BLOCKER_WEIGHT
    return max(0.0, min(1.0, priority_term + age_term + blocker_term))


def threshold(urgency: float) -> float:
    return max(-1.0, min(1.0, 0.5 - urgency))
```

- [ ] **Step 4: Run → pass.**

- [ ] **Step 5: Commit**

```bash
git add packages/general_beckman/src/general_beckman/admission.py packages/general_beckman/tests/test_admission_urgency.py
git commit -m "feat(beckman): admission helpers — compute_urgency + threshold"
```

---

### Task 19: Beckman `next_task()` rewrite with top-K + admission check

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/__init__.py`
- Modify: `packages/general_beckman/src/general_beckman/admission.py` (add top-K loop)
- Test: `packages/general_beckman/tests/test_next_task_admission.py`

- [ ] **Step 1: Write failing test**

```python
# packages/general_beckman/tests/test_next_task_admission.py
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from general_beckman.types import Task


def _mock_pick(provider, model_name, is_local=False):
    fake_model = MagicMock(is_local=is_local, provider=provider, name=model_name)
    return MagicMock(model=fake_model, composite=0.6)


@pytest.mark.asyncio
async def test_admits_task_when_pool_abundant():
    import general_beckman

    snap = MagicMock()
    snap.pressure_for = MagicMock(return_value=0.7)
    nh = MagicMock(snapshot=MagicMock(return_value=snap))
    ready = [Task(id=1, priority=7, difficulty=5, created_at=time.time())]

    with patch("general_beckman.queue.pick_ready_top_k", new=AsyncMock(return_value=ready)), \
         patch("fatih_hoca.select", return_value=_mock_pick("anthropic", "claude-sonnet-4-6")), \
         patch("nerd_herd.snapshot", return_value=snap):
        out = await general_beckman.next_task()

    assert out is not None
    assert out.id == 1
    assert out.preselected_pick is not None


@pytest.mark.asyncio
async def test_holds_when_pool_depleted_and_urgency_low():
    import general_beckman

    snap = MagicMock()
    snap.pressure_for = MagicMock(return_value=-0.8)
    ready = [Task(id=1, priority=3, difficulty=3, created_at=time.time())]

    with patch("general_beckman.queue.pick_ready_top_k", new=AsyncMock(return_value=ready)), \
         patch("fatih_hoca.select", return_value=_mock_pick("anthropic", "claude-sonnet-4-6")), \
         patch("nerd_herd.snapshot", return_value=snap):
        out = await general_beckman.next_task()

    assert out is None


@pytest.mark.asyncio
async def test_skips_candidate_when_hoca_returns_none():
    import general_beckman

    snap = MagicMock()
    snap.pressure_for = MagicMock(return_value=0.7)
    task_a = Task(id=1, priority=7, difficulty=5, created_at=time.time())
    task_b = Task(id=2, priority=6, difficulty=4, created_at=time.time())

    picks = [None, _mock_pick("anthropic", "claude-sonnet-4-6")]
    def side_effect(*args, **kwargs):
        return picks.pop(0)

    with patch("general_beckman.queue.pick_ready_top_k", new=AsyncMock(return_value=[task_a, task_b])), \
         patch("fatih_hoca.select", side_effect=side_effect), \
         patch("nerd_herd.snapshot", return_value=snap):
        out = await general_beckman.next_task()

    assert out is not None
    assert out.id == 2
```

- [ ] **Step 2: Add `pick_ready_top_k` to `queue.py`**

In `packages/general_beckman/src/general_beckman/queue.py`, add a function that returns up to K ready tasks sorted by `compute_urgency` descending:

```python
from general_beckman.admission import compute_urgency


async def pick_ready_top_k(db_path: str, k: int = 5) -> list[Task]:
    # Use existing pick_ready_task SQL as base; extend to return up to K
    # rows ordered by urgency descending.
    tasks = await _fetch_ready_tasks(db_path)   # existing helper or write one
    return sorted(tasks, key=compute_urgency, reverse=True)[:k]
```

(Use whatever existing helper loads ready tasks. If none exists, write `_fetch_ready_tasks` using the same SQL as `pick_ready_task`.)

- [ ] **Step 3: Rewrite `next_task()`**

In `packages/general_beckman/src/general_beckman/__init__.py`:

```python
import os
import fatih_hoca
import nerd_herd
from general_beckman.queue import pick_ready_top_k
from general_beckman.admission import compute_urgency, threshold

BECKMAN_TOP_K = int(os.environ.get("BECKMAN_TOP_K", "5"))
BECKMAN_HARD_CAP = int(os.environ.get("BECKMAN_HARD_CAP", "4"))
_DB_PATH = os.environ.get("DB_PATH", "kutai.db")


async def next_task() -> Task | None:
    if await _currently_dispatched_count() >= BECKMAN_HARD_CAP:
        return None
    snapshot = nerd_herd.snapshot()

    candidates = await pick_ready_top_k(_DB_PATH, k=BECKMAN_TOP_K)
    for task in candidates:
        pick = fatih_hoca.select(
            task=task.profile if hasattr(task, "profile") else task.agent_type or "",
            agent_type=task.agent_type or "",
            difficulty=task.difficulty,
        )
        if pick is None:
            continue
        pressure = snapshot.pressure_for(pick.model)
        urgency = compute_urgency(task)
        if pressure >= threshold(urgency):
            task.preselected_pick = pick
            await _mark_task_dispatched(task)   # existing hook
            return task
    return None


async def _currently_dispatched_count() -> int:
    # Uses existing Beckman bookkeeping (task_queue.status = 'dispatched').
    ...
```

(Preserve existing side effects of `next_task` such as marking the chosen task as dispatched.)

- [ ] **Step 4: Run tests → pass.**

- [ ] **Step 5: Commit**

```bash
git add packages/general_beckman
git commit -m "feat(beckman): next_task top-K admission loop with pool-pressure gate

Iterates top-K ready tasks by urgency; asks Hoca for a Pick per candidate;
admits first task whose pool pressure clears threshold(urgency). Orchestrator
pump continues to call next_task() until None for multi-task dispatch.
"
```

---

### Task 20: Dispatcher honours `Task.preselected_pick` on iteration 0

**Files:**
- Modify: `src/core/llm_dispatcher.py`
- Modify: `src/core/orchestrator.py` (pass preselected_pick through)
- Test: `tests/core/test_dispatcher_preselected_pick.py`

- [ ] **Step 1: Write failing test**

```python
# tests/core/test_dispatcher_preselected_pick.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_preselected_pick_skips_first_hoca_call():
    from src.core.llm_dispatcher import LLMDispatcher, CallCategory

    fake_model = MagicMock(is_local=False, provider="anthropic",
                           name="claude-sonnet-4-6")
    preselected = MagicMock(model=fake_model, composite=0.7)

    select_calls = MagicMock()
    with patch("fatih_hoca.select", select_calls), \
         patch("kuleden_donen_var.begin_call", return_value=MagicMock()), \
         patch("kuleden_donen_var.end_call"), \
         patch("hallederiz_kadir.call", new=AsyncMock(return_value={"ok": True})), \
         patch("src.infra.pick_log.write_pick_log_row", new=AsyncMock()):
        d = LLMDispatcher()
        await d.request(
            category=CallCategory.MAIN_WORK,
            task="coder", difficulty=7, messages=[], tools=None,
            preselected_pick=preselected,
        )
    assert select_calls.call_count == 0, "Hoca.select should NOT be called when preselected_pick is provided"
```

- [ ] **Step 2: Run → fail**

- [ ] **Step 3: Add kwarg + short-circuit first iteration**

In `src/core/llm_dispatcher.py`, extend `request()` signature:

```python
async def request(
    self, category, task="", agent_type="", difficulty=5,
    messages=None, tools=None, failures=None,
    preselected_pick=None,
    **kwargs,
) -> dict:
    ...
```

Inside the iteration loop:
```python
for iteration in range(max_recursion):
    if iteration == 0 and preselected_pick is not None:
        pick = preselected_pick
        preselected_pick = None
    else:
        pick = fatih_hoca.select(task=task, ..., failures=failures)
    # ... rest of loop unchanged
```

Update orchestrator to forward `task.preselected_pick` when calling dispatcher:

```python
# src/core/orchestrator.py (around the dispatcher call)
result = await dispatcher.request(
    category=category, task=task.profile, ...,
    preselected_pick=getattr(task, "preselected_pick", None),
)
```

- [ ] **Step 4: Run tests → pass.**

- [ ] **Step 5: Regression**

Run: `.venv/Scripts/python -m pytest tests/core --timeout=60`

- [ ] **Step 6: Commit**

```bash
git add src/core/llm_dispatcher.py src/core/orchestrator.py tests/core/test_dispatcher_preselected_pick.py
git commit -m "feat(dispatcher): honour Task.preselected_pick on iteration 0

Beckman's admission-time Hoca query is reused by dispatcher's first
iteration. No redundant select(). Iterations 2+ call Hoca fresh as today.
"
```

---

### Task 21: Remove obsolete `system_busy` single-bit check if present

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/__init__.py` (or wherever `_system_busy` referenced)
- Test: existing Beckman tests

- [ ] **Step 1: Search for `system_busy` or `_system_busy` usage**

Run: `rg -n "system_busy" packages/general_beckman/`
Identify any gate in `next_task()` that pre-dates the new admission loop.

- [ ] **Step 2: Remove the gate**

If `next_task()` still has a pre-loop check like `if nerd_herd.system_busy(): return None`, delete it. New loop handles admission more expressively.

- [ ] **Step 3: Run Beckman tests**

Run: `.venv/Scripts/python -m pytest packages/general_beckman/tests --timeout=60`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add packages/general_beckman
git commit -m "cleanup(beckman): drop obsolete system_busy single-bit gate

Superseded by the top-K admission loop with pool-pressure threshold.
"
```

---

## Phase 4 — Simulator + regression (Tasks 22–23)

### Task 22: Beckman admission scenarios in simulator

**Files:**
- Create: `packages/general_beckman/tests/sim/run_admission_scenarios.py`
- Create: `packages/general_beckman/tests/sim/scenarios.py`
- Share: Phase 2d `SimState` infrastructure in `packages/fatih_hoca/tests/sim/`

- [ ] **Step 1: Write scenario 1 — cloud near reset + hot queue**

```python
# packages/general_beckman/tests/sim/scenarios.py
from dataclasses import dataclass
from typing import Callable


@dataclass
class Scenario:
    name: str
    setup: Callable        # builds initial SimState
    ticks: int
    expected: Callable     # takes final SimState, returns (passed: bool, reason: str)


def scenario_cloud_near_reset_hot_queue():
    def setup(s):
        s.cloud["anthropic"]["claude-sonnet-4-6"].remaining = 28
        s.cloud["anthropic"]["claude-sonnet-4-6"].limit = 30
        s.cloud["anthropic"]["claude-sonnet-4-6"].reset_in = 600
        s.queue.load(hard=6, easy=0)
        return s

    def expected(s):
        admitted = s.metrics["admissions"]
        if admitted < 5:
            return False, f"too few admitted: {admitted}"
        if s.cloud["anthropic"]["claude-sonnet-4-6"].remaining < 0:
            return False, "over-burn"
        return True, "ok"

    return Scenario("cloud_near_reset_hot_queue", setup, 20, expected)
```

Build 4 more scenarios as described in spec section 8.3: depleted+cold, mixed, i2p burst (180 correlated deps), starvation recovery.

- [ ] **Step 2: Write runner**

```python
# packages/general_beckman/tests/sim/run_admission_scenarios.py
import asyncio
from general_beckman.tests.sim.scenarios import (
    scenario_cloud_near_reset_hot_queue,
    # ... other scenarios
)


async def main():
    scenarios = [
        scenario_cloud_near_reset_hot_queue(),
        # ...
    ]
    failures = []
    for sc in scenarios:
        state = sc.setup(_fresh_sim_state())
        for _ in range(sc.ticks):
            await _run_tick(state)
        ok, reason = sc.expected(state)
        print(f"{sc.name}: {'PASS' if ok else 'FAIL'} — {reason}")
        if not ok:
            failures.append(sc.name)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
```

(The `_fresh_sim_state` + `_run_tick` helpers should share Phase 2d simulator infrastructure — import from `packages/fatih_hoca/tests/sim/`.)

- [ ] **Step 3: Run scenarios**

Run: `.venv/Scripts/python packages/general_beckman/tests/sim/run_admission_scenarios.py`
Expected: all scenarios pass.

- [ ] **Step 4: Tune constants if scenarios fail**

If scenario 4 (i2p burst) shows Beckman admitting too many too fast, reduce the admission cadence or tighten the threshold intercept. If scenario 5 (starvation) shows stuck tasks not admitting, raise `AGE_WEIGHT` slightly.

Document any tuning in commit message.

- [ ] **Step 5: Commit**

```bash
git add packages/general_beckman/tests/sim
git commit -m "test(beckman): admission scenarios (5) validating equilibrium

Scenarios: cloud-abundant burn, cloud-depleted hold, mixed pools,
i2p 180-task burst, starvation recovery. Shared SimState with Phase 2d.
"
```

---

### Task 23: Full regression sweep

- [ ] **Step 1: Targeted tests**

Run: `.venv/Scripts/python -m pytest packages/nerd_herd/tests packages/kuleden_donen_var/tests packages/fatih_hoca/tests packages/general_beckman/tests tests/core tests/infra --timeout=120`
Expected: all pass.

- [ ] **Step 2: Phase 2d scenarios**

Run: `.venv/Scripts/python packages/fatih_hoca/tests/sim/run_scenarios.py`
Run: `.venv/Scripts/python packages/fatih_hoca/tests/sim/run_swap_storm_check.py`
Expected: green.

- [ ] **Step 3: Beckman scenarios**

Run: `.venv/Scripts/python packages/general_beckman/tests/sim/run_admission_scenarios.py`
Expected: green.

- [ ] **Step 4: Docs update**

Edit `docs/architecture-modularization.md` (section on dispatcher + Hoca layering). Add a short paragraph describing the new pool-pressure flow:

> Pool pressure is derived in nerd_herd from KDV's pushed `CloudProviderState` (now including `in_flight`) and exposed via `snapshot.pressure_for(model)`. Hoca reads it for per-call scoring; Beckman reads it during `next_task()` admission. The admission threshold scales by per-task urgency (priority + age bump + blocker bump).

- [ ] **Step 5: Commit**

```bash
git add docs/architecture-modularization.md
git commit -m "docs: architecture-modularization note on pool-pressure flow"
```

- [ ] **Step 6: Final merge readiness check**

```bash
git log --oneline -30
git status
```

Confirm:
- 23 commits (one per task).
- Clean working tree.
- All tests green (step 1–3).
- Phase 2d scenarios not regressed.

---

## Self-Review Summary

1. **Spec coverage**: all 21 migration steps from spec section 10 map to tasks.
2. **Placeholder scan**: no TBD/TODO inside task steps. Values that require environment-specific checking (e.g. exact column names in schemas) are flagged with "adjust to match" notes rather than left blank.
3. **Type consistency**: `PoolPressure`, `QueueProfile`, `SwapBudget`, `InFlightHandle` used consistently across tasks that reference them.

---

## Out-of-Scope / Deferred Items

- Adaptive TTL tuning from observed latency (spec §11).
- Priority reshape (spec §11).
- Historical continuity of `model_pick_log` for counterfactual CLI (spec §11).
- Consolidation of `select_for_simulation` with the pure `select()` — marked in spec 10.1 step 4 as an audit; if the two can be unified, do so within Task 3; otherwise defer.
- Resolution of `BECKMAN_HARD_CAP` default — spec §12 open. Initial value `4` set via env; production tune after observing scenarios.
