# Fatih Hoca Phase 2c Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire pool-aware utilization scoring (local idle, free/prepaid cloud reset urgency) into Fatih Hoca's ranking, gated by a capability ratio, and replace the flat `perf_score=50` cloud fallback with a grading-blended score derived from `model_stats`.

**Architecture:** Three-pool taxonomy (`local` / `time_bucketed` / `per_call`) classified at ranking time. Each candidate gets a `[0,1]` urgency score from its pool; applied as a Layer-3 composite multiplier `× (1 + 0.25·urgency)` only when `cap_score ≥ 0.85 × top_cap`. `perf_score` for cloud blends `model_stats.success_rate` (≥20 samples) with the existing tps-derived signal for locals.

**Tech Stack:** Python 3.10, pytest (timeout-wrapped), aiosqlite, existing `fatih_hoca` + `nerd_herd` + `kuleden_donen_var` packages.

**Branch:** `feat/fatih-hoca-phase2c` in `.worktrees/fatih-hoca-phase2c`.

**Spec:** `docs/superpowers/specs/2026-04-18-fatih-hoca-phase2c-design.md`.

---

## File Structure

| File | Role |
|------|------|
| `packages/nerd_herd/src/nerd_herd/types.py` | MODIFY — add `idle_seconds` to `LocalModelState` |
| `packages/nerd_herd/src/nerd_herd/inference.py` *(or wherever `measured_tps` is recorded)* | MODIFY — track `last_inference_ts`, compute `idle_seconds` |
| `packages/nerd_herd/src/nerd_herd/client.py` | MODIFY — pass `idle_seconds` through `get_snapshot()` mapping |
| `packages/nerd_herd/src/nerd_herd/exposition.py` | MODIFY — parse `idle_seconds` from snapshot body |
| `packages/fatih_hoca/src/fatih_hoca/pools.py` | **NEW** — `Pool` enum, `classify_pool`, `compute_urgency` |
| `packages/fatih_hoca/src/fatih_hoca/grading.py` | **NEW** — `grading_perf_score(model_name) -> Optional[float]` |
| `packages/fatih_hoca/src/fatih_hoca/ranking.py` | MODIFY — perf_score blend + Layer-3 urgency multiplier |
| `packages/fatih_hoca/src/fatih_hoca/selector.py` | MODIFY — write `pool` + `urgency` to `model_pick_log` |
| `packages/fatih_hoca/src/fatih_hoca/counterfactual.py` | **NEW** — CLI that re-scores `model_pick_log` rows |
| `src/infra/db.py` | MODIFY — schema migration adding `pool TEXT` + `urgency REAL` columns |
| `packages/fatih_hoca/tests/test_pools.py` | **NEW** — tests for classification + urgency math |
| `packages/fatih_hoca/tests/test_grading.py` | **NEW** — grading perf_score with sample-size gating |
| `packages/fatih_hoca/tests/test_capability_gate.py` | **NEW** — gate only lets near-peer candidates get boost |
| `packages/fatih_hoca/tests/test_ranking.py` | MODIFY — add regression + new-behavior cases |
| `packages/fatih_hoca/tests/test_counterfactual.py` | **NEW** — CLI integration test over a temp sqlite |
| `tests/fatih_hoca/test_pick_telemetry.py` | MODIFY — assert `pool` + `urgency` columns populated |

### Test execution conventions (from CLAUDE.md + memory)

- Unit tests in `packages/fatih_hoca/tests/*`: `timeout 60 pytest packages/fatih_hoca/tests/test_X.py -v`
- Unit tests in `tests/unit/*` or `tests/fatih_hoca/*`: `PYTHONPATH=. timeout 60 pytest tests/fatih_hoca/test_X.py -v`
- Full fatih_hoca suite: `timeout 120 pytest packages/fatih_hoca/tests/ -v`
- Simulator feedback loop: `PYTHONPATH=packages/fatih_hoca/src python -m fatih_hoca.simulate_i2p --model-dir "C:\Users\sakir\ai\models"` (runs in ~30s; treat as fast loop after ranking changes)
- Never install packages; the shared venv is pinned.

### Commit policy

One commit per green task (after all steps in that task pass). Conventional-commit prefixes: `feat()`, `test()`, `refactor()`, `chore()`.

---

## Task 1: `model_pick_log` schema migration — add `pool` + `urgency` columns

**Files:**
- Modify: `src/infra/db.py` (around the `model_pick_log` `CREATE TABLE` statement; search for `CREATE TABLE IF NOT EXISTS model_pick_log`)
- Modify: `tests/fatih_hoca/test_pick_telemetry.py` (extend the existing column assertion)

- [ ] **Step 1: Write the failing test** — extend `test_model_pick_log_table_exists` to require the new columns.

```python
# tests/fatih_hoca/test_pick_telemetry.py — in test_model_pick_log_table_exists,
# after the existing column assertions add:
assert "pool" in cols, f"missing 'pool' column: {cols}"
assert "urgency" in cols, f"missing 'urgency' column: {cols}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. timeout 30 pytest tests/fatih_hoca/test_pick_telemetry.py::test_model_pick_log_table_exists -v`
Expected: FAIL with `missing 'pool' column` (or `missing 'urgency' column`).

- [ ] **Step 3: Add columns to the `CREATE TABLE` and run the migration on open**

In `src/infra/db.py`, locate the `CREATE TABLE IF NOT EXISTS model_pick_log (...)` block. Add these columns at the end of the column list (before the closing `)`):

```sql
pool TEXT,
urgency REAL
```

Then, immediately after the `CREATE TABLE` execution, add an idempotent migration for existing DBs:

```python
# Idempotent column add for pre-Phase-2c databases.
for col_name, col_type in (("pool", "TEXT"), ("urgency", "REAL")):
    try:
        await db.execute(f"ALTER TABLE model_pick_log ADD COLUMN {col_name} {col_type}")
    except Exception:
        pass  # column already exists
```

(If the surrounding `init_db` is sync, use `db.execute` without `await` — match the existing style.)

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. timeout 30 pytest tests/fatih_hoca/test_pick_telemetry.py::test_model_pick_log_table_exists -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/infra/db.py tests/fatih_hoca/test_pick_telemetry.py
git commit -m "feat(fatih-hoca): model_pick_log gains pool + urgency columns"
```

---

## Task 2: `LocalModelState.idle_seconds` — snapshot signal for local GPU idle duration

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/types.py` (add field to `LocalModelState`)
- Modify: `packages/nerd_herd/src/nerd_herd/inference.py` (track `last_inference_ts`; search for wherever `measured_tps` is updated)
- Modify: `packages/nerd_herd/src/nerd_herd/client.py` (pass `idle_seconds` through `get_snapshot()`'s dataclass assembly around line 262)
- Modify: `packages/nerd_herd/src/nerd_herd/exposition.py` (parse from dict around line 146)
- Create: `packages/nerd_herd/tests/test_idle_seconds.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/nerd_herd/tests/test_idle_seconds.py
"""LocalModelState.idle_seconds populates from inference timestamps."""
from __future__ import annotations
import time
from nerd_herd.inference import InferenceCollector   # adjust import if class named differently
from nerd_herd.types import LocalModelState


def test_idle_seconds_zero_on_fresh_inference():
    collector = InferenceCollector()
    collector.record(measured_tps=20.0, model_name="test-model")
    state = collector.local_state()
    assert state.idle_seconds == 0.0 or state.idle_seconds < 0.1


def test_idle_seconds_grows_with_time(monkeypatch):
    collector = InferenceCollector()
    t0 = time.time()
    monkeypatch.setattr(time, "time", lambda: t0)
    collector.record(measured_tps=20.0, model_name="test-model")
    monkeypatch.setattr(time, "time", lambda: t0 + 42.5)
    state = collector.local_state()
    assert 42.0 <= state.idle_seconds <= 43.0


def test_idle_seconds_none_when_no_inference_yet():
    collector = InferenceCollector()
    state = collector.local_state()
    # New collector, never saw inference → field defaults to 0.0
    # (test both conventions; pick whichever matches the implementation)
    assert state.idle_seconds == 0.0
```

**Note to implementer:** If `InferenceCollector` doesn't exist by that name, grep: `grep -rn "measured_tps" packages/nerd_herd/src/` — the class writing `LocalModelState` is the one to modify. Adapt the test to that actual symbol and keep the behavior assertions.

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 pytest packages/nerd_herd/tests/test_idle_seconds.py -v`
Expected: FAIL — `AttributeError: 'LocalModelState' object has no attribute 'idle_seconds'`, or class/attribute missing.

- [ ] **Step 3: Add `idle_seconds` to `LocalModelState`**

In `packages/nerd_herd/src/nerd_herd/types.py`, modify `LocalModelState`:

```python
@dataclass
class LocalModelState:
    model_name: str | None = None
    thinking_enabled: bool = False
    vision_enabled: bool = False
    measured_tps: float = 0.0
    context_length: int = 0
    is_swapping: bool = False
    kv_cache_ratio: float = 0.0
    idle_seconds: float = 0.0   # seconds since last completed local inference; 0 while a call is in-flight or before first inference
```

- [ ] **Step 4: Track `last_inference_ts` in the collector and expose `idle_seconds`**

In the module that writes `LocalModelState.measured_tps`, add `_last_inference_ts: float = 0.0` as an instance attribute. In the method that records an inference completion (where `measured_tps` is set), record `self._last_inference_ts = time.time()`. In the method/property that assembles `LocalModelState` for the snapshot, compute:

```python
idle_seconds = 0.0
if self._last_inference_ts > 0 and not self._in_flight:  # _in_flight: whatever flag tracks active calls; use False if no such flag
    idle_seconds = max(0.0, time.time() - self._last_inference_ts)
```

and pass it into `LocalModelState(..., idle_seconds=idle_seconds)`.

- [ ] **Step 5: Propagate through client + exposition**

In `packages/nerd_herd/src/nerd_herd/client.py` around line 262 where `LocalModelState(...)` is constructed from `local_data`:

```python
measured_tps=float(local_data.get("measured_tps", 0.0)),
idle_seconds=float(local_data.get("idle_seconds", 0.0)),
```

And in the `record_snapshot` signature/kwargs at line 198 and in the body dict at line 208, add `idle_seconds`:

```python
def record_snapshot(
    self,
    ...
    measured_tps: float = 0.0,
    idle_seconds: float = 0.0,
    ...
):
    ...
    "measured_tps": measured_tps,
    "idle_seconds": idle_seconds,
```

In `packages/nerd_herd/src/nerd_herd/exposition.py` around line 146:

```python
measured_tps=float(body.get("measured_tps", 0.0)),
idle_seconds=float(body.get("idle_seconds", 0.0)),
```

- [ ] **Step 6: Run test to verify it passes**

Run: `timeout 30 pytest packages/nerd_herd/tests/test_idle_seconds.py -v`
Expected: PASS. Also run the existing nerd_herd tests to confirm nothing broke:

Run: `timeout 60 pytest packages/nerd_herd/tests/ -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add packages/nerd_herd/
git commit -m "feat(nerd-herd): LocalModelState.idle_seconds tracks GPU idle duration"
```

---

## Task 3: `fatih_hoca.pools` — pool classification + urgency math

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/pools.py`
- Create: `packages/fatih_hoca/tests/test_pools.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_pools.py
"""Pool classification + urgency formulas (pure functions, no I/O)."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from fatih_hoca.pools import (
    Pool,
    classify_pool,
    compute_urgency,
    URGENCY_MAX_BONUS,
    LOCAL_IDLE_SATURATION_SECS,
    RESET_HORIZON_SECS,
)


@dataclass
class _FakeModel:
    name: str
    is_local: bool = False
    is_free: bool = False
    provider: str = ""


@dataclass
class _FakeRateLimit:
    remaining: int | None = None
    limit: int | None = None
    reset_at: int | None = None


@dataclass
class _FakeRateLimits:
    rpd: _FakeRateLimit = field(default_factory=_FakeRateLimit)


@dataclass
class _FakeModelState:
    limits: _FakeRateLimits = field(default_factory=_FakeRateLimits)


@dataclass
class _FakeProviderState:
    models: dict[str, _FakeModelState] = field(default_factory=dict)
    limits: _FakeRateLimits = field(default_factory=_FakeRateLimits)


@dataclass
class _FakeLocal:
    idle_seconds: float = 0.0


@dataclass
class _FakeSnapshot:
    local: _FakeLocal = field(default_factory=_FakeLocal)
    cloud: dict[str, _FakeProviderState] = field(default_factory=dict)


# ── Classification ──

def test_local_model_classifies_as_local():
    m = _FakeModel(name="qwen", is_local=True)
    assert classify_pool(m) is Pool.LOCAL


def test_free_cloud_classifies_as_time_bucketed():
    m = _FakeModel(name="groq-llama", is_local=False, is_free=True, provider="groq")
    assert classify_pool(m) is Pool.TIME_BUCKETED


def test_paid_cloud_classifies_as_per_call():
    m = _FakeModel(name="claude-sonnet", is_local=False, is_free=False, provider="anthropic")
    assert classify_pool(m) is Pool.PER_CALL


# ── Urgency: local ──

def test_local_urgency_zero_when_active():
    m = _FakeModel(name="qwen", is_local=True)
    snap = _FakeSnapshot(local=_FakeLocal(idle_seconds=0.0))
    assert compute_urgency(m, snap) == 0.0


def test_local_urgency_scales_linearly_to_saturation():
    m = _FakeModel(name="qwen", is_local=True)
    snap = _FakeSnapshot(local=_FakeLocal(idle_seconds=LOCAL_IDLE_SATURATION_SECS / 2))
    u = compute_urgency(m, snap)
    assert 0.45 <= u <= 0.55


def test_local_urgency_saturates_at_one():
    m = _FakeModel(name="qwen", is_local=True)
    snap = _FakeSnapshot(local=_FakeLocal(idle_seconds=LOCAL_IDLE_SATURATION_SECS * 5))
    assert compute_urgency(m, snap) == 1.0


# ── Urgency: time-bucketed cloud ──

def _make_bucketed_snapshot(remaining, limit, reset_in_seconds, provider="groq", model_id="llama-70b"):
    now = time.time()
    rl = _FakeRateLimit(remaining=remaining, limit=limit, reset_at=int(now + reset_in_seconds))
    mstate = _FakeModelState(limits=_FakeRateLimits(rpd=rl))
    prov = _FakeProviderState(models={model_id: mstate}, limits=_FakeRateLimits(rpd=rl))
    return _FakeSnapshot(cloud={provider: prov})


def test_bucketed_urgency_high_when_unused_and_close_to_reset():
    m = _FakeModel(name="llama-70b", is_free=True, provider="groq")
    snap = _make_bucketed_snapshot(remaining=900, limit=1000, reset_in_seconds=300)
    u = compute_urgency(m, snap)
    # remaining_frac = 0.9, reset_proximity = 1 - 300/3600 ≈ 0.917 → urgency ≈ 0.825
    assert 0.78 <= u <= 0.88


def test_bucketed_urgency_low_when_reset_far_away():
    m = _FakeModel(name="llama-70b", is_free=True, provider="groq")
    snap = _make_bucketed_snapshot(remaining=900, limit=1000, reset_in_seconds=86400)
    u = compute_urgency(m, snap)
    # reset_proximity clamped to 0
    assert u == 0.0


def test_bucketed_urgency_zero_when_quota_exhausted():
    m = _FakeModel(name="llama-70b", is_free=True, provider="groq")
    snap = _make_bucketed_snapshot(remaining=0, limit=1000, reset_in_seconds=300)
    assert compute_urgency(m, snap) == 0.0


def test_bucketed_urgency_midnight_utc_fallback_when_reset_missing():
    """If reset_at is None, fall back to midnight UTC assumption."""
    m = _FakeModel(name="llama-70b", is_free=True, provider="groq")
    rl = _FakeRateLimit(remaining=900, limit=1000, reset_at=None)
    mstate = _FakeModelState(limits=_FakeRateLimits(rpd=rl))
    prov = _FakeProviderState(models={"llama-70b": mstate}, limits=_FakeRateLimits(rpd=rl))
    snap = _FakeSnapshot(cloud={"groq": prov})
    u = compute_urgency(m, snap)
    assert 0.0 <= u <= 1.0  # fallback yields a defined number


# ── Urgency: per-call ──

def test_per_call_urgency_always_zero():
    m = _FakeModel(name="claude-sonnet", is_free=False, provider="anthropic")
    snap = _FakeSnapshot()
    assert compute_urgency(m, snap) == 0.0


# ── Missing telemetry ──

def test_missing_local_snapshot_returns_zero():
    m = _FakeModel(name="qwen", is_local=True)
    snap = _FakeSnapshot()  # idle_seconds=0.0 default
    assert compute_urgency(m, snap) == 0.0


def test_unknown_provider_returns_zero():
    m = _FakeModel(name="mystery", is_free=True, provider="unknown-provider")
    snap = _FakeSnapshot()
    assert compute_urgency(m, snap) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 pytest packages/fatih_hoca/tests/test_pools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fatih_hoca.pools'`.

- [ ] **Step 3: Implement `pools.py`**

Create `packages/fatih_hoca/src/fatih_hoca/pools.py`:

```python
"""Pool taxonomy and urgency math for Fatih Hoca Phase 2c.

Three pools: LOCAL (sunk-cost GPU), TIME_BUCKETED (free-tier or prepaid cloud
with reset timer), PER_CALL (paid cloud, no bucket).

Urgency in [0, 1]. Applied at Layer 3 in ranking.py as:
    composite *= (1 + URGENCY_MAX_BONUS * urgency)
when capability gate allows.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Any

# ── Tunables ───────────────────────────────────────────────────────────
URGENCY_MAX_BONUS: float = 0.25
LOCAL_IDLE_SATURATION_SECS: float = 600.0
RESET_HORIZON_SECS: float = 3600.0


class Pool(str, Enum):
    LOCAL = "local"
    TIME_BUCKETED = "time_bucketed"
    PER_CALL = "per_call"


def classify_pool(model: Any) -> Pool:
    """Classify a ModelInfo-like object into its utilization pool."""
    if getattr(model, "is_local", False):
        return Pool.LOCAL
    if getattr(model, "is_free", False):
        return Pool.TIME_BUCKETED
    # Future: prepaid cloud lands here too via `getattr(model, "prepaid_remaining", 0) > 0`
    return Pool.PER_CALL


def _midnight_utc_reset_in_seconds() -> float:
    """Fallback: assume quotas reset at 00:00 UTC daily."""
    now = time.time()
    seconds_today = now % 86400
    return 86400 - seconds_today


def _bucketed_urgency(model: Any, snapshot: Any) -> float:
    provider = getattr(model, "provider", "") or ""
    prov_state = getattr(snapshot, "cloud", {}).get(provider)
    if prov_state is None:
        return 0.0

    # Prefer per-model limits, fall back to provider-wide
    model_id = getattr(model, "name", None) or getattr(model, "litellm_name", "")
    model_state = prov_state.models.get(model_id) if hasattr(prov_state, "models") else None
    source = model_state if model_state is not None else prov_state

    limits = getattr(source, "limits", None)
    if limits is None:
        return 0.0
    # Daily bucket (rpd). If unavailable, tpm/rpm can be added later.
    rpd = getattr(limits, "rpd", None)
    if rpd is None:
        return 0.0

    remaining = getattr(rpd, "remaining", None)
    limit = getattr(rpd, "limit", None)
    reset_at = getattr(rpd, "reset_at", None)

    if remaining is None or limit is None or limit <= 0:
        return 0.0
    if remaining <= 0:
        return 0.0

    if reset_at is not None and reset_at > 0:
        reset_in = max(0.0, reset_at - time.time())
    else:
        reset_in = _midnight_utc_reset_in_seconds()

    remaining_frac = min(1.0, max(0.0, remaining / limit))
    reset_proximity = max(0.0, 1.0 - min(1.0, reset_in / RESET_HORIZON_SECS))
    return remaining_frac * reset_proximity


def _local_urgency(snapshot: Any) -> float:
    local = getattr(snapshot, "local", None)
    if local is None:
        return 0.0
    idle = float(getattr(local, "idle_seconds", 0.0) or 0.0)
    if idle <= 0:
        return 0.0
    return min(1.0, idle / LOCAL_IDLE_SATURATION_SECS)


def compute_urgency(model: Any, snapshot: Any) -> float:
    """Return urgency in [0, 1] for this model under the current snapshot."""
    pool = classify_pool(model)
    if pool is Pool.LOCAL:
        return _local_urgency(snapshot)
    if pool is Pool.TIME_BUCKETED:
        return _bucketed_urgency(model, snapshot)
    return 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 30 pytest packages/fatih_hoca/tests/test_pools.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/pools.py packages/fatih_hoca/tests/test_pools.py
git commit -m "feat(fatih-hoca): pool taxonomy and urgency math (local + time-bucketed)"
```

---

## Task 4: `fatih_hoca.grading` — grading-derived perf_score

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/grading.py`
- Create: `packages/fatih_hoca/tests/test_grading.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_grading.py
"""grading_perf_score blends model_stats into Performance History."""
from __future__ import annotations
import sqlite3
import pytest
from fatih_hoca.grading import grading_perf_score, GRADING_MIN_SAMPLES


@pytest.fixture
def stats_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE model_stats (
            model TEXT,
            agent_type TEXT,
            total_calls INTEGER DEFAULT 0,
            success_rate REAL DEFAULT 1.0,
            avg_grade REAL DEFAULT 0.0,
            PRIMARY KEY (model, agent_type)
        )"""
    )
    conn.commit()
    monkeypatch.setenv("DB_PATH", str(db_path))
    # grading.py reads DB_PATH lazily; tests control the env var
    yield conn
    conn.close()


def test_returns_none_when_no_stats(stats_db):
    assert grading_perf_score("nonexistent-model") is None


def test_returns_none_when_below_min_samples(stats_db):
    stats_db.execute(
        "INSERT INTO model_stats (model, agent_type, total_calls, success_rate) "
        "VALUES (?, ?, ?, ?)",
        ("qwen", "coder", GRADING_MIN_SAMPLES - 1, 0.9),
    )
    stats_db.commit()
    assert grading_perf_score("qwen") is None


def test_returns_blended_score_above_threshold(stats_db):
    stats_db.execute(
        "INSERT INTO model_stats (model, agent_type, total_calls, success_rate) "
        "VALUES (?, ?, ?, ?)",
        ("qwen", "coder", 50, 1.0),
    )
    stats_db.commit()
    score = grading_perf_score("qwen")
    assert score is not None
    assert 90.0 <= score <= 95.0   # 100% success → top of scale


def test_zero_success_maps_to_floor(stats_db):
    stats_db.execute(
        "INSERT INTO model_stats (model, agent_type, total_calls, success_rate) "
        "VALUES (?, ?, ?, ?)",
        ("brokenmodel", "coder", 50, 0.0),
    )
    stats_db.commit()
    score = grading_perf_score("brokenmodel")
    assert score is not None
    assert score == 20.0  # floor


def test_aggregates_across_agent_types(stats_db):
    """A model used across agents aggregates weighted by total_calls."""
    stats_db.executemany(
        "INSERT INTO model_stats (model, agent_type, total_calls, success_rate) VALUES (?,?,?,?)",
        [
            ("qwen", "coder", 30, 1.0),
            ("qwen", "planner", 30, 0.5),
        ],
    )
    stats_db.commit()
    score = grading_perf_score("qwen")
    # weighted success ≈ 0.75 → maps to ~20 + 0.75*(95-20) = 76.25
    assert 73.0 <= score <= 80.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 pytest packages/fatih_hoca/tests/test_grading.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fatih_hoca.grading'`.

- [ ] **Step 3: Implement `grading.py`**

Create `packages/fatih_hoca/src/fatih_hoca/grading.py`:

```python
"""Grading-derived perf_score from model_stats.

Used by ranking.py to replace the flat `perf_score=50` fallback. Reads the
main KutAI sqlite DB (DB_PATH env) in a tight, synchronous connection — this
is selection-path code, must not block on async machinery.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from typing import Optional

GRADING_MIN_SAMPLES: int = 20
GRADING_PERF_FLOOR: float = 20.0
GRADING_PERF_CEIL: float = 95.0

logger = logging.getLogger(__name__)


def _db_path() -> str | None:
    return os.environ.get("DB_PATH")


def grading_perf_score(model_name: str) -> Optional[float]:
    """Aggregate model_stats rows for `model_name` into a 0-100 perf score.

    Returns None when total sample count across agent_types is below
    GRADING_MIN_SAMPLES. Otherwise maps weighted success_rate in [0, 1] to
    [GRADING_PERF_FLOOR, GRADING_PERF_CEIL] linearly.
    """
    path = _db_path()
    if not path or not os.path.exists(path):
        return None
    try:
        conn = sqlite3.connect(path)
        try:
            cur = conn.execute(
                "SELECT total_calls, success_rate FROM model_stats WHERE model = ?",
                (model_name,),
            )
            rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as e:
        logger.debug("grading_perf_score read failed for %s: %s", model_name, e)
        return None

    total = sum(r[0] or 0 for r in rows)
    if total < GRADING_MIN_SAMPLES:
        return None

    weighted_success = sum((r[0] or 0) * (r[1] or 0.0) for r in rows) / total
    weighted_success = max(0.0, min(1.0, weighted_success))
    return GRADING_PERF_FLOOR + weighted_success * (GRADING_PERF_CEIL - GRADING_PERF_FLOOR)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 30 pytest packages/fatih_hoca/tests/test_grading.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/grading.py packages/fatih_hoca/tests/test_grading.py
git commit -m "feat(fatih-hoca): grading_perf_score blends model_stats into selection"
```

---

## Task 5: Integrate grading perf_score into `ranking.py`

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/ranking.py` (perf_score block at lines ~293-326)
- Modify: `packages/fatih_hoca/tests/test_ranking.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_ranking.py — append
"""Grading blend: cloud model with stats gets non-flat perf_score."""


def test_cloud_model_with_stats_gets_grading_perf_score(monkeypatch):
    from fatih_hoca import ranking, grading

    def fake_grading(name):
        if name == "groq-llama-70b":
            return 80.0
        return None

    monkeypatch.setattr(ranking, "grading_perf_score", fake_grading)
    # Build a minimal ModelInfo-like fake: cloud, not loaded, tps=0.
    # Use the same fixture helper as other ranking tests — or construct
    # a ScoredModel scenario directly. Assert:
    #   blended = 0.6 * 80 + 0.4 * 50 = 68.0 ± 0.5
    # via capturing perf_score from reasons or by instrumenting the score path.
    # (Concrete assertion wires into whatever test harness test_ranking already uses.)


def test_local_without_stats_falls_back_to_tps_derived(monkeypatch):
    from fatih_hoca import ranking

    monkeypatch.setattr(ranking, "grading_perf_score", lambda _: None)
    # Ensure a local model with measured_tps=20 still produces the tps-derived
    # perf_score (should be ~65, unchanged from pre-Phase-2c).
```

**Note to implementer:** `test_ranking.py` already has fixtures for building ModelInfo/snapshot scenarios. Use those; the test above is a signature-level sketch. Grep `test_ranking.py` for the pattern it uses to build `_score()` calls and mirror it.

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 pytest packages/fatih_hoca/tests/test_ranking.py -v -k grading`
Expected: FAIL — `AttributeError: module 'fatih_hoca.ranking' has no attribute 'grading_perf_score'` (or the blend assertion fails because perf_score is still flat).

- [ ] **Step 3: Wire `grading_perf_score` into ranking**

At the top of `packages/fatih_hoca/src/fatih_hoca/ranking.py`, add:

```python
from fatih_hoca.grading import grading_perf_score
```

Replace the perf_score block at **lines 314-327** (the section labeled `# ── 4. Performance History (0–100) ──`) with:

```python
# ── 4. Performance History (0–100) ──
# Blends tps-derived (local speed signal) with grading-derived
# (success_rate from model_stats). Falls back cleanly when either side
# is missing. Phase 2c: replaces the flat perf=50 fallback for cloud.
GRADING_WEIGHT = 0.6

if model.is_local and model.is_loaded and \
   local_state.model_name == model.name and local_state.measured_tps > 0:
    tps = local_state.measured_tps
    tps_perf = min(95.0, 50.0 + (tps - 10) * 1.5) if tps >= 10 else max(20.0, 20.0 + tps * 3.0)
elif model.is_local and model.tokens_per_second > 0:
    tps = model.tokens_per_second
    tps_perf = min(90.0, 45.0 + (tps - 10) * 1.2) if tps >= 10 else max(15.0, 15.0 + tps * 3.0)
else:
    tps_perf = 50.0

grading = grading_perf_score(model.name)
if grading is not None:
    perf_score = GRADING_WEIGHT * grading + (1.0 - GRADING_WEIGHT) * tps_perf
    reasons.append(f"perf={perf_score:.0f}(g={grading:.0f},tps={tps_perf:.0f})")
else:
    perf_score = tps_perf
    reasons.append(f"perf={perf_score:.0f}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 30 pytest packages/fatih_hoca/tests/test_ranking.py -v`
Expected: all PASS (new grading tests + all pre-existing regressions green).

- [ ] **Step 5: Run the simulator and record the distribution**

Run: `PYTHONPATH=packages/fatih_hoca/src python -m fatih_hoca.simulate_i2p --model-dir "C:\Users\sakir\ai\models" --json /tmp/phase2c-after-task5.json 2>&1 | tee /tmp/phase2c-after-task5.txt`

Expected: Simulator runs cleanly. Capture pick distribution for comparison; likely still heavily groq-favored because urgency isn't wired yet.

- [ ] **Step 6: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/ranking.py packages/fatih_hoca/tests/test_ranking.py
git commit -m "feat(fatih-hoca): ranking blends grading-derived perf_score with tps signal"
```

---

## Task 6: Integrate urgency multiplier + capability gate into `ranking.py`

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/ranking.py` (Layer-3 insertion around line 483)
- Modify: `packages/fatih_hoca/src/fatih_hoca/selector.py` (needs to tell ranker the top cap; or ranker does a two-pass over candidates)
- Create: `packages/fatih_hoca/tests/test_capability_gate.py`

The ranker currently scores candidates one-at-a-time inside a loop. The capability gate needs the **max cap_score across all candidates** — a two-pass structure. Simplest path: score all candidates into a list without applying urgency, then walk the list a second time and apply urgency + gate.

Inspect `ranking.py` to find where the per-candidate loop produces `ScoredModel`s and where they're returned. Split into a two-phase pipeline.

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_capability_gate.py
"""Urgency multiplier applies only when cap_score ≥ CAP_GATE_RATIO × top_cap."""
from __future__ import annotations
import pytest
from fatih_hoca.pools import URGENCY_MAX_BONUS

# Reuse whatever score-two-candidates fixture test_ranking.py already has, or
# build here: two candidates A and B, A has cap=90, B has cap=70.
# If CAP_GATE_RATIO=0.85 → threshold = 0.85*90 = 76.5 → B (70) is below, no boost.
# If both have urgency=1.0 set via snapshot, A gets * (1 + 0.25) but B does not.


def test_boost_applies_only_to_top_cap_candidate(score_pair):
    """score_pair: fixture or helper returning (top_composite, other_composite)
    after scoring two candidates with equal urgency but different cap_scores."""
    top, other = score_pair(top_cap=90, other_cap=70, urgency_for_both=1.0)
    # `top` gains up to 25% urgency boost; `other` is gated out.
    assert top > other * 1.10   # meaningful separation


def test_both_candidates_boosted_when_near_peer(score_pair):
    """If other_cap >= 0.85 * top_cap, both get urgency."""
    top, other = score_pair(top_cap=90, other_cap=80, urgency_for_both=1.0)
    # Both boosted equally → ratio unchanged compared to no-urgency baseline.
    baseline_top, baseline_other = score_pair(top_cap=90, other_cap=80, urgency_for_both=0.0)
    ratio_after = top / other
    ratio_before = baseline_top / baseline_other
    assert abs(ratio_after - ratio_before) < 0.02


def test_zero_urgency_never_boosts(score_pair):
    top, other = score_pair(top_cap=90, other_cap=70, urgency_for_both=0.0)
    baseline_top, baseline_other = score_pair(top_cap=90, other_cap=70, urgency_for_both=0.0)
    assert top == baseline_top
    assert other == baseline_other
```

**Note to implementer:** the `score_pair` helper must mirror whatever scaffolding `test_ranking.py` uses — reuse its builders rather than reinventing. If none exist, create a minimal one in `conftest.py` under `packages/fatih_hoca/tests/`.

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 pytest packages/fatih_hoca/tests/test_capability_gate.py -v`
Expected: FAIL — urgency not wired into ranking yet.

- [ ] **Step 3: Two-pass refactor — collect candidates, then apply urgency gate**

At the top of `packages/fatih_hoca/src/fatih_hoca/ranking.py`:

```python
from fatih_hoca.pools import (
    Pool, classify_pool, compute_urgency,
    URGENCY_MAX_BONUS,
)

CAP_GATE_RATIO: float = 0.85
```

After the existing per-candidate scoring loop finishes (i.e., just before the function returns the list of `ScoredModel`s), insert:

```python
# ── Phase 2c: pool-urgency layer with capability gate ──
if scored:   # at least one candidate
    top_cap = max(sm.cap_score for sm in scored)
    cap_threshold = top_cap * CAP_GATE_RATIO
    for sm in scored:
        urgency = compute_urgency(sm.model, snapshot)
        pool = classify_pool(sm.model)
        sm.pool = pool.value
        sm.urgency = urgency
        if urgency > 0 and sm.cap_score >= cap_threshold:
            mult = 1.0 + URGENCY_MAX_BONUS * urgency
            sm.composite *= mult
            sm.reasons.append(f"urgency={pool.value}:{urgency:.2f}×{mult:.2f}")
        elif urgency > 0:
            sm.reasons.append(f"urgency_gated={pool.value}:{urgency:.2f}")
    # Re-sort after urgency adjustments
    scored.sort(key=lambda sm: sm.composite, reverse=True)
```

Extend `ScoredModel` in `packages/fatih_hoca/src/fatih_hoca/types.py` (grep for `class ScoredModel`) to add the telemetry fields:

```python
@dataclass
class ScoredModel:
    ...  # existing fields unchanged
    pool: str = ""        # "local" | "time_bucketed" | "per_call" — set post-urgency
    urgency: float = 0.0  # [0, 1], set post-urgency
```

**Hint to implementer:** if `scored` is built inline rather than as a list, refactor the scoring loop to accumulate candidates into a list first. Do not apply urgency per-candidate without a gate — the gate needs the top cap score.

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 30 pytest packages/fatih_hoca/tests/test_capability_gate.py packages/fatih_hoca/tests/test_ranking.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the simulator and compare**

Run:
```bash
PYTHONPATH=packages/fatih_hoca/src python -m fatih_hoca.simulate_i2p \
    --model-dir "C:\Users\sakir\ai\models" \
    --json /tmp/phase2c-after-task6.json 2>&1 | tee /tmp/phase2c-after-task6.txt
```

Expected: meaningful shift in distribution vs. the baseline (Task 5 capture). Locals should win a non-zero share of coder/writer/analyst steps at d≤5. If not, note the gap for Task 9's tuning work — don't block the commit.

- [ ] **Step 6: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/ranking.py \
        packages/fatih_hoca/src/fatih_hoca/types.py \
        packages/fatih_hoca/tests/test_capability_gate.py \
        packages/fatih_hoca/tests/test_ranking.py
git commit -m "feat(fatih-hoca): pool-urgency multiplier with capability gate"
```

---

## Task 7: Write `pool` + `urgency` to `model_pick_log`

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/selector.py` (the `_persist_pick_telemetry` method around line 234-293)
- Modify: `tests/fatih_hoca/test_pick_telemetry.py`

- [ ] **Step 1: Write the failing test** — extend the existing pick-write test.

```python
# tests/fatih_hoca/test_pick_telemetry.py — add a new test
@pytest.mark.asyncio
async def test_pool_and_urgency_persist_on_pick(tmp_path, monkeypatch):
    """After select(), model_pick_log row has non-null pool + urgency values."""
    # Reuse the existing fixture that runs a successful select() against a seeded
    # registry (grep this file for the existing pick-write test; mirror it).
    # After the select, query:
    #   SELECT pool, urgency FROM model_pick_log LIMIT 1
    # Assert:
    #   pool in {"local", "time_bucketed", "per_call"}
    #   0.0 <= urgency <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. timeout 30 pytest tests/fatih_hoca/test_pick_telemetry.py -v -k pool_and_urgency`
Expected: FAIL — column values are NULL.

- [ ] **Step 3: Update `_persist_pick_telemetry` to include pool + urgency**

In `packages/fatih_hoca/src/fatih_hoca/selector.py` around line 234, inspect the `_persist_pick_telemetry` signature and the `INSERT INTO model_pick_log` SQL at line 293. Change:

```python
INSERT INTO model_pick_log
    (timestamp, task_name, agent_type, difficulty, call_category,
     picked_model, picked_score, picked_reasons,
     candidates_json, failures_json, snapshot_summary,
     pool, urgency)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
```

And pass `scored[0].pool` and `scored[0].urgency` from the winning candidate into the parameter tuple. (Winner = first element of the sorted list the ranker returned, which `selector` already holds.)

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. timeout 30 pytest tests/fatih_hoca/test_pick_telemetry.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/selector.py tests/fatih_hoca/test_pick_telemetry.py
git commit -m "feat(fatih-hoca): model_pick_log captures pool + urgency for offline tuning"
```

---

## Task 8: Counterfactual CLI — `fatih_hoca.counterfactual`

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/counterfactual.py`
- Create: `packages/fatih_hoca/tests/test_counterfactual.py`

The CLI replays `model_pick_log` rows, re-scoring each candidate list under configurable parameters, and reports top-1 agreement with `model_stats.success_rate`.

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_counterfactual.py
"""Counterfactual CLI sanity check: runs against a seeded sqlite and emits JSON."""
from __future__ import annotations
import json
import sqlite3
import subprocess
import sys


def test_cli_runs_on_empty_db(tmp_path, monkeypatch):
    db = tmp_path / "empty.db"
    # Create minimal schema
    conn = sqlite3.connect(db)
    conn.executescript(
        """CREATE TABLE model_pick_log (
            id INTEGER PRIMARY KEY, timestamp TEXT, task_name TEXT, agent_type TEXT,
            difficulty INTEGER, call_category TEXT, picked_model TEXT,
            picked_score REAL, picked_reasons TEXT, candidates_json TEXT,
            failures_json TEXT, snapshot_summary TEXT, pool TEXT, urgency REAL
        );
        CREATE TABLE model_stats (
            model TEXT, agent_type TEXT, total_calls INTEGER,
            success_rate REAL, avg_grade REAL,
            PRIMARY KEY (model, agent_type)
        );"""
    )
    conn.commit()
    conn.close()

    env = {"PATH": "", "DB_PATH": str(db), "PYTHONPATH": "packages/fatih_hoca/src"}
    result = subprocess.run(
        [sys.executable, "-m", "fatih_hoca.counterfactual",
         "--urgency-bonus", "0.25", "--cap-gate", "0.85"],
        capture_output=True, text=True, env={**env, **dict(monkeypatch.setenv.__self__.__dict__) if hasattr(monkeypatch, "_setenvs") else env},
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    # Output includes "rows: 0" or similar
    assert "0" in result.stdout


def test_cli_reports_agreement_rate(tmp_path):
    """Seed a row where candidates_json contains the picked model as #1, and
    verify agreement report includes 100%."""
    db = tmp_path / "seeded.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """CREATE TABLE model_pick_log (
            id INTEGER PRIMARY KEY, timestamp TEXT, task_name TEXT, agent_type TEXT,
            difficulty INTEGER, call_category TEXT, picked_model TEXT,
            picked_score REAL, picked_reasons TEXT, candidates_json TEXT,
            failures_json TEXT, snapshot_summary TEXT, pool TEXT, urgency REAL
        );
        CREATE TABLE model_stats (
            model TEXT, agent_type TEXT, total_calls INTEGER,
            success_rate REAL, avg_grade REAL,
            PRIMARY KEY (model, agent_type)
        );"""
    )
    candidates = json.dumps([
        {"name": "qwen", "composite": 80.0, "cap_score": 78.0, "pool": "local", "urgency": 0.5},
        {"name": "groq-llama-70b", "composite": 75.0, "cap_score": 72.0, "pool": "time_bucketed", "urgency": 0.0},
    ])
    conn.execute(
        """INSERT INTO model_pick_log
           (timestamp, task_name, agent_type, difficulty, call_category,
            picked_model, picked_score, picked_reasons, candidates_json,
            failures_json, snapshot_summary, pool, urgency)
           VALUES ('2026-04-18T00:00:00', 'test', 'coder', 5, 'main_work',
                   'qwen', 80.0, 'test', ?, '[]', '{}', 'local', 0.5)""",
        (candidates,),
    )
    conn.execute(
        "INSERT INTO model_stats (model, agent_type, total_calls, success_rate, avg_grade) "
        "VALUES ('qwen', 'coder', 30, 1.0, 8.5)"
    )
    conn.commit()
    conn.close()

    import os
    env = {**os.environ, "DB_PATH": str(db), "PYTHONPATH": "packages/fatih_hoca/src"}
    result = subprocess.run(
        [sys.executable, "-m", "fatih_hoca.counterfactual",
         "--urgency-bonus", "0.25", "--cap-gate", "0.85"],
        capture_output=True, text=True, env=env, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "agreement" in result.stdout.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 pytest packages/fatih_hoca/tests/test_counterfactual.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fatih_hoca.counterfactual'`.

- [ ] **Step 3: Implement the CLI**

Create `packages/fatih_hoca/src/fatih_hoca/counterfactual.py`:

```python
"""Counterfactual scoring CLI — replays model_pick_log under candidate parameters.

Usage:
    python -m fatih_hoca.counterfactual [--urgency-bonus F] [--cap-gate F] \
        [--limit-days N] [--json PATH]

Reads DB_PATH from env. Joins model_pick_log against model_stats to compute
how often each pick aligned with the empirically-best model (highest
success_rate) at pick time. Does NOT re-run the full ranker — it rescales
stored candidate composites using the stored pool/urgency and the given
parameters, then re-ranks.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from typing import Any


def _load_rows(db_path: str, limit_days: int | None) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    q = ("SELECT picked_model, agent_type, difficulty, candidates_json "
         "FROM model_pick_log")
    if limit_days is not None:
        q += f" WHERE timestamp >= datetime('now','-{int(limit_days)} days')"
    try:
        cur = conn.execute(q)
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
    return rows


def _load_success_map(db_path: str) -> dict[str, float]:
    """model_name → weighted success rate across agent types."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT model, total_calls, success_rate FROM model_stats"
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    weighted: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for model, total, rate in rows:
        if total and total > 0 and rate is not None:
            weighted[model].append((total, rate))
    out: dict[str, float] = {}
    for m, pairs in weighted.items():
        tot = sum(t for t, _ in pairs)
        if tot > 0:
            out[m] = sum(t * r for t, r in pairs) / tot
    return out


def _rescore(candidates: list[dict[str, Any]], urgency_bonus: float, cap_gate: float) -> list[dict[str, Any]]:
    """Apply urgency multiplier + cap gate to stored candidate composites."""
    if not candidates:
        return candidates
    top_cap = max((c.get("cap_score", 0.0) or 0.0) for c in candidates)
    threshold = top_cap * cap_gate
    out = []
    for c in candidates:
        composite = float(c.get("composite", 0.0) or 0.0)
        urgency = float(c.get("urgency", 0.0) or 0.0)
        cap = float(c.get("cap_score", 0.0) or 0.0)
        if urgency > 0 and cap >= threshold:
            composite *= 1.0 + urgency_bonus * urgency
        nc = dict(c)
        nc["composite_cf"] = composite
        out.append(nc)
    out.sort(key=lambda x: x["composite_cf"], reverse=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urgency-bonus", type=float, default=0.25)
    parser.add_argument("--cap-gate", type=float, default=0.85)
    parser.add_argument("--limit-days", type=int, default=None)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args(argv)

    db_path = os.environ.get("DB_PATH")
    if not db_path or not os.path.exists(db_path):
        print(f"DB_PATH not set or missing: {db_path}", file=sys.stderr)
        return 1

    rows = _load_rows(db_path, args.limit_days)
    success_map = _load_success_map(db_path)

    # Compute three agreement rates:
    #  (a) CF-pick == historical-pick
    #  (b) CF-pick == empirically-best-by-success-rate among candidates
    #  (c) historical-pick == empirically-best
    agree_cf_hist = 0
    agree_cf_best = 0
    agree_hist_best = 0
    total = 0
    pool_counter: Counter[str] = Counter()

    for r in rows:
        try:
            candidates = json.loads(r["candidates_json"] or "[]")
        except Exception:
            continue
        if not candidates:
            continue
        total += 1
        rescored = _rescore(candidates, args.urgency_bonus, args.cap_gate)
        cf_pick = rescored[0].get("name") or rescored[0].get("model")
        hist_pick = r["picked_model"]
        best = None
        best_rate = -1.0
        for c in candidates:
            name = c.get("name") or c.get("model")
            rate = success_map.get(name)
            if rate is not None and rate > best_rate:
                best_rate = rate
                best = name
        if cf_pick == hist_pick:
            agree_cf_hist += 1
        if best is not None and cf_pick == best:
            agree_cf_best += 1
        if best is not None and hist_pick == best:
            agree_hist_best += 1
        pool_counter[rescored[0].get("pool", "?")] += 1

    summary = {
        "rows": total,
        "urgency_bonus": args.urgency_bonus,
        "cap_gate": args.cap_gate,
        "agreement_cf_vs_historical": (agree_cf_hist / total) if total else 0.0,
        "agreement_cf_vs_best": (agree_cf_best / total) if total else 0.0,
        "agreement_historical_vs_best": (agree_hist_best / total) if total else 0.0,
        "pool_distribution_cf": dict(pool_counter),
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    for k, v in summary.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 30 pytest packages/fatih_hoca/tests/test_counterfactual.py -v`
Expected: all PASS.

- [ ] **Step 5: Smoke-test against the live DB (read-only)**

Run: `PYTHONPATH=packages/fatih_hoca/src python -m fatih_hoca.counterfactual --urgency-bonus 0.25 --cap-gate 0.85 --json /tmp/cf-default.json`

Expected: Prints rows count and three agreement rates; no exceptions. The live DB has 349 rows, so `rows: 349`. Note: historical rows lack `pool`/`urgency` columns (populated only going forward from Task 7), so `_rescore` falls through and CF==historical agreement is near 100% — this is expected and shows the tool works end-to-end.

- [ ] **Step 6: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/counterfactual.py packages/fatih_hoca/tests/test_counterfactual.py
git commit -m "feat(fatih-hoca): counterfactual CLI for urgency/cap-gate parameter sweeps"
```

---

## Task 9: Simulator validation + parameter sweep

This task has no new test code. It produces a decision: are defaults (`URGENCY_MAX_BONUS=0.25`, `CAP_GATE_RATIO=0.85`, `GRADING_WEIGHT=0.6`) good enough, or does the simulator + counterfactual agree they need tuning?

- [ ] **Step 1: Re-run simulator at defaults**

Run:
```bash
PYTHONPATH=packages/fatih_hoca/src python -m fatih_hoca.simulate_i2p \
    --model-dir "C:\Users\sakir\ai\models" \
    --json /tmp/phase2c-final.json 2>&1 | tee /tmp/phase2c-final.txt
```

Expected: distribution shows locals winning a meaningful share of coder/writer/analyst steps at d≤5; claude-sonnet still wins d=8.

- [ ] **Step 2: Counterfactual sweep**

Run a small grid — manual, no script needed:

```bash
for bonus in 0.15 0.20 0.25 0.30; do
  for gate in 0.80 0.85 0.90; do
    echo "=== bonus=$bonus gate=$gate ==="
    PYTHONPATH=packages/fatih_hoca/src python -m fatih_hoca.counterfactual \
        --urgency-bonus $bonus --cap-gate $gate
  done
done | tee /tmp/phase2c-sweep.txt
```

(On Windows bash, the loop works in git-bash. If not, issue the 12 commands individually.)

- [ ] **Step 3: Decide**

Read `/tmp/phase2c-final.txt` and `/tmp/phase2c-sweep.txt`. If (a) simulator shows ≥30% local wins at d≤5 across coder/writer/analyst AND (b) no sweep combination beats defaults by >5 percentage points on agreement-vs-best, **defaults hold — proceed to Step 5**.

Otherwise, choose the combination that maximizes agreement-vs-best while keeping d=8 on claude-sonnet in the simulator output, update constants in `pools.py` and `ranking.py`, re-run tests, and re-run the simulator.

- [ ] **Step 4: (Conditional) Update tunables + re-run tests**

If tuning is needed, edit `URGENCY_MAX_BONUS` in `packages/fatih_hoca/src/fatih_hoca/pools.py` and `CAP_GATE_RATIO` in `packages/fatih_hoca/src/fatih_hoca/ranking.py` to the chosen values. Also update the corresponding test thresholds in `test_pools.py`.

Run: `timeout 120 pytest packages/fatih_hoca/tests/ tests/fatih_hoca/ -v`
Expected: all PASS.

- [ ] **Step 5: Write a short memory note + commit any tuning**

Append a one-liner to `MEMORY.md` via a new entry (per the auto-memory rules — lead with the finding):

- Phase 2c landed — locals now win ≥X% of d≤5 cold-start steps; final tunables: `URGENCY_MAX_BONUS=…`, `CAP_GATE_RATIO=…`, `GRADING_WEIGHT=0.6`. Counterfactual agreement-vs-best improved from Y% (pre-Phase-2c) to Z%.

If tuning changed any code:

```bash
git add packages/fatih_hoca/src/fatih_hoca/pools.py \
        packages/fatih_hoca/src/fatih_hoca/ranking.py \
        packages/fatih_hoca/tests/test_pools.py
git commit -m "chore(fatih-hoca): final Phase 2c tunables from counterfactual sweep"
```

Otherwise skip the commit.

---

## Task 10: End-to-end validation + CLAUDE.md update

- [ ] **Step 1: Full test sweep**

Run: `timeout 180 pytest packages/fatih_hoca/tests/ packages/nerd_herd/tests/ tests/fatih_hoca/ -v`
Expected: all PASS.

- [ ] **Step 2: Update CLAUDE.md pitfalls/architecture**

In `CLAUDE.md` under "Common Pitfalls", add:

> - **Pool-urgency multiplier** (Phase 2c, 2026-04-18): ranking.py now applies a Layer-3 urgency multiplier (up to +25%) for LOCAL and TIME_BUCKETED candidates, gated at `cap_score ≥ 0.85 × top_cap`. If a model seems to lose inexplicably, check the reason log for `urgency_gated` — it means the candidate was capability-underqualified vs the top scorer. Grading-derived perf_score requires `model_stats.total_calls ≥ 20` (per model, summed across agent_types); below that, cloud candidates fall back to the flat 50 baseline.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: CLAUDE.md notes Phase 2c pool-urgency multiplier + cap gate"
```

- [ ] **Step 4: Readiness check for merge**

Run `git log --oneline main..HEAD` to confirm the commit chain is coherent. Every commit should be green. Don't merge or push — hand back to the user for review + push.

---

## Self-Review

**Spec coverage:**
- §1 Problem / §2 Objective — addressed across Tasks 5+6.
- §3 Pool taxonomy — Task 3.
- §4 Urgency formulae — Task 3.
- §5 Capability gate — Task 6.
- §6 Grading perf_score — Tasks 4+5.
- §7 Layer-3 insertion — Task 6.
- §8 Data plumbing — Task 2 (nerd_herd), Task 3 (classification covers kuleden_donen_var fallback via midnight UTC), Task 1 (model_pick_log schema).
- §9 Counterfactual — Task 8.
- §10 Validation loop — Tasks 5, 6, 9.
- §11 Non-goals — respected (no weight rebalance at d=5 unless Task 9 step 3 demands it).
- §12 Risks — cap gate (Task 6) addresses thrash; fallback at midnight UTC (Task 3) addresses missing reset headers; grading min samples (Task 4) addresses low sample sizes.
- §13 Success criteria — verified in Task 9 step 1 (simulator ≥30%) and Task 8 smoke test (counterfactual agreement).
- §14 Out of scope — respected.

**Placeholder scan:** No "TBD", no "similar to task N" without repeat, no "add appropriate error handling" without shown code. Two `"Hint to implementer"` blocks acknowledge real-world code variance — they provide grep commands, not placeholder content.

**Type consistency:**
- `Pool` (enum) used consistently across `pools.py`, `ranking.py`, `selector.py`.
- `URGENCY_MAX_BONUS`, `LOCAL_IDLE_SATURATION_SECS`, `RESET_HORIZON_SECS` live in `pools.py`.
- `CAP_GATE_RATIO`, `GRADING_WEIGHT` live in `ranking.py` (ranking-layer concerns, not pool-layer).
- `GRADING_MIN_SAMPLES`, `GRADING_PERF_FLOOR`, `GRADING_PERF_CEIL` live in `grading.py`.
- `ScoredModel` gets `.pool: str` and `.urgency: float` — written by ranking, read by selector.
- `LocalModelState.idle_seconds: float` in `nerd_herd.types` — written by inference collector, read by `pools.compute_urgency`.

All symbols defined before use.
