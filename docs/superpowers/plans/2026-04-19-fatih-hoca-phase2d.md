# Fatih Hoca Phase 2d Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Phase 2c's gated urgency layer with a unified utilization equation driven by pool scarcity + capability fit-excess, validated by a stateful pytest simulator across 7 scenarios.

**Architecture:** Two new runtime modules under `packages/fatih_hoca/src/fatih_hoca/` (`capability_curve.py`, `scarcity.py`); `ranking.py` gets `_apply_utilization_layer` replacing `_apply_urgency_layer`; gate constant `CAP_GATE_RATIO` and module constant `URGENCY_MAX_BONUS` deleted/renamed. All simulator/scenario code lives under `packages/fatih_hoca/tests/sim/` — test infrastructure, not shipped.

**Tech Stack:** Python 3.10, dataclasses, pytest (with `timeout`), existing fatih_hoca + nerd_herd packages.

**Spec:** `docs/superpowers/specs/2026-04-19-fatih-hoca-phase2d-finalized-design.md`

---

## File Structure

**New runtime files:**
- `packages/fatih_hoca/src/fatih_hoca/capability_curve.py` — `CAP_NEEDED_BY_DIFFICULTY` dict + `cap_needed_for_difficulty(d)` lookup
- `packages/fatih_hoca/src/fatih_hoca/scarcity.py` — `pool_scarcity(model, snapshot, queue_state)` dispatching per pool

**Modified runtime files:**
- `packages/fatih_hoca/src/fatih_hoca/ranking.py` — replace `_apply_urgency_layer` with `_apply_utilization_layer`; delete `CAP_GATE_RATIO`; rename `URGENCY_MAX_BONUS` usage
- `packages/fatih_hoca/src/fatih_hoca/pools.py` — rename constant `URGENCY_MAX_BONUS` → `UTILIZATION_K`, value `0.25`
- `packages/fatih_hoca/src/fatih_hoca/requirements.py` — add public `queue_profile` property to `QuotaPlanner`
- `packages/fatih_hoca/src/fatih_hoca/counterfactual.py` — rewrite internal `_rescore` to call new equation

**New test files (test infrastructure + scenarios):**
- `packages/fatih_hoca/tests/test_capability_curve.py`
- `packages/fatih_hoca/tests/test_scarcity.py`
- `packages/fatih_hoca/tests/test_utilization_layer.py`
- `packages/fatih_hoca/tests/sim/__init__.py`
- `packages/fatih_hoca/tests/sim/state.py`
- `packages/fatih_hoca/tests/sim/runner.py`
- `packages/fatih_hoca/tests/sim/report.py`
- `packages/fatih_hoca/tests/sim/scenarios.py`
- `packages/fatih_hoca/tests/test_scenarios.py`

**Modified test files:**
- `packages/fatih_hoca/tests/test_capability_gate.py` — update expectations (gate is gone; rewrite as utilization-layer regression tests or delete obsolete cases)
- `packages/fatih_hoca/tests/test_ranking.py` — update any assertions that reference urgency bonus directly

**Docs:**
- `CLAUDE.md` — replace Phase 2c "machinery, not balanced" note with Phase 2d description
- `docs/architecture-modularization.md` — add Phase 2d section

---

## Commit Policy

Every task ends with a green `pytest` run for that task's scope, then a single commit. Commit messages follow conventional-commits (`feat(fatih-hoca): …`, `test(fatih-hoca): …`, `refactor(fatih-hoca): …`).

All pytest commands use `timeout` (never unbounded). All imports go through the shared venv at `../../.venv/Scripts/python.exe`. **Never** `pip install -e` from a worktree path.

---

## Task 1: Capability Curve Module

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/capability_curve.py`
- Test: `packages/fatih_hoca/tests/test_capability_curve.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_capability_curve.py
"""Tests for capability_curve module (Phase 2d)."""
from fatih_hoca.capability_curve import (
    CAP_NEEDED_BY_DIFFICULTY,
    cap_needed_for_difficulty,
)


def test_dict_has_all_difficulties_1_through_10():
    for d in range(1, 11):
        assert d in CAP_NEEDED_BY_DIFFICULTY
        assert 0 <= CAP_NEEDED_BY_DIFFICULTY[d] <= 100


def test_monotonic_non_decreasing():
    prev = -1.0
    for d in range(1, 11):
        v = CAP_NEEDED_BY_DIFFICULTY[d]
        assert v >= prev, f"d={d} ({v}) < d={d-1} ({prev})"
        prev = v


def test_lookup_returns_dict_value():
    assert cap_needed_for_difficulty(1) == CAP_NEEDED_BY_DIFFICULTY[1]
    assert cap_needed_for_difficulty(5) == CAP_NEEDED_BY_DIFFICULTY[5]
    assert cap_needed_for_difficulty(10) == CAP_NEEDED_BY_DIFFICULTY[10]


def test_lookup_clamps_below_range():
    assert cap_needed_for_difficulty(0) == CAP_NEEDED_BY_DIFFICULTY[1]
    assert cap_needed_for_difficulty(-5) == CAP_NEEDED_BY_DIFFICULTY[1]


def test_lookup_clamps_above_range():
    assert cap_needed_for_difficulty(11) == CAP_NEEDED_BY_DIFFICULTY[10]
    assert cap_needed_for_difficulty(99) == CAP_NEEDED_BY_DIFFICULTY[10]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 15 ../../.venv/Scripts/python.exe -m pytest packages/fatih_hoca/tests/test_capability_curve.py -v`
Expected: FAIL with `ModuleNotFoundError: fatih_hoca.capability_curve`

- [ ] **Step 3: Write minimal implementation**

```python
# packages/fatih_hoca/src/fatih_hoca/capability_curve.py
"""Capability-needed-per-difficulty curve (Phase 2d).

Plain dict mapping task difficulty (1-10) to the minimum `cap_score_100`
a model should have to serve that difficulty without over-qualification.

Used by ranking._apply_utilization_layer via:
    fit_excess = (cap_score_100 - cap_needed_for_difficulty(d)) / 100

Hand-tuned starting curve; graduation to empirical derivation from
`model_stats` is deferred until sample counts per (model, d) warrant it.
"""
from __future__ import annotations


CAP_NEEDED_BY_DIFFICULTY: dict[int, float] = {
    1: 30.0, 2: 30.0, 3: 30.0,
    4: 45.0, 5: 45.0,
    6: 60.0, 7: 60.0,
    8: 75.0,
    9: 88.0, 10: 88.0,
}


def cap_needed_for_difficulty(d: int) -> float:
    """Return the cap_score_100 floor for difficulty `d`.

    Clamps `d` to [1, 10] so out-of-range inputs degrade gracefully.
    """
    d_clamped = max(1, min(10, int(d)))
    return CAP_NEEDED_BY_DIFFICULTY[d_clamped]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 15 ../../.venv/Scripts/python.exe -m pytest packages/fatih_hoca/tests/test_capability_curve.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/capability_curve.py packages/fatih_hoca/tests/test_capability_curve.py
git commit -m "feat(fatih-hoca): capability curve for Phase 2d utilization equation"
```

---

## Task 2: Expose QuotaPlanner.queue_profile

The utilization equation reads `queue_state.hard_tasks_count` / `total_tasks`. `QuotaPlanner` stores these in a private `_queue_profile` — add a public accessor.

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/requirements.py` (add property in `QuotaPlanner` class, near `expensive_threshold`)
- Test: `packages/fatih_hoca/tests/test_requirements.py` (append test case)

- [ ] **Step 1: Write the failing test**

Append to `packages/fatih_hoca/tests/test_requirements.py`:

```python
def test_queue_profile_property_returns_default_on_init():
    from fatih_hoca.requirements import QuotaPlanner, QueueProfile
    planner = QuotaPlanner()
    profile = planner.queue_profile
    assert isinstance(profile, QueueProfile)
    assert profile.total_tasks == 0
    assert profile.hard_tasks_count == 0


def test_queue_profile_property_reflects_set_value():
    from fatih_hoca.requirements import QuotaPlanner, QueueProfile
    planner = QuotaPlanner()
    profile = QueueProfile(total_tasks=182, hard_tasks_count=18, max_difficulty=9)
    planner.set_queue_profile(profile)
    assert planner.queue_profile.total_tasks == 182
    assert planner.queue_profile.hard_tasks_count == 18
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 15 ../../.venv/Scripts/python.exe -m pytest packages/fatih_hoca/tests/test_requirements.py::test_queue_profile_property_returns_default_on_init packages/fatih_hoca/tests/test_requirements.py::test_queue_profile_property_reflects_set_value -v`
Expected: FAIL with `AttributeError: 'QuotaPlanner' object has no attribute 'queue_profile'`

- [ ] **Step 3: Write minimal implementation**

In `packages/fatih_hoca/src/fatih_hoca/requirements.py`, inside `class QuotaPlanner`, directly after the `expensive_threshold` property:

```python
    @property
    def queue_profile(self) -> QueueProfile:
        """Current queue profile — defaults to empty until `set_queue_profile` called."""
        return self._queue_profile
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 15 ../../.venv/Scripts/python.exe -m pytest packages/fatih_hoca/tests/test_requirements.py -v`
Expected: PASS (all prior tests + 2 new tests)

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/requirements.py packages/fatih_hoca/tests/test_requirements.py
git commit -m "feat(fatih-hoca): expose QuotaPlanner.queue_profile property"
```

---

## Task 3: Scarcity Module — Local Pool

`scarcity.py` dispatches per pool. Build it incrementally: local first, time_bucketed next (Task 4), per_call last (Task 5).

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/scarcity.py`
- Test: `packages/fatih_hoca/tests/test_scarcity.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_scarcity.py
"""Tests for scarcity module (Phase 2d)."""
from types import SimpleNamespace

from fatih_hoca.scarcity import pool_scarcity


def _local_model(is_loaded=False, requests_processing=0):
    return SimpleNamespace(
        name="test-local",
        is_local=True,
        is_free=False,
        is_loaded=is_loaded,
        provider="local",
    )


def _snapshot_with_local(idle_seconds=0.0, loaded_name="other", requests_processing=0):
    local = SimpleNamespace(
        model_name=loaded_name,
        idle_seconds=idle_seconds,
        measured_tps=20.0,
        thinking_enabled=False,
        requests_processing=requests_processing,
    )
    return SimpleNamespace(local=local, cloud={})


# ── Local pool ──────────────────────────────────────────────────────────

def test_local_busy_returns_negative_small():
    model = _local_model(is_loaded=True)
    snap = _snapshot_with_local(idle_seconds=0.0, loaded_name="test-local", requests_processing=1)
    s = pool_scarcity(model, snap, queue_state=None)
    assert -0.2 <= s < 0  # busy → mild negative


def test_local_cold_idle_returns_zero():
    model = _local_model(is_loaded=False)
    snap = _snapshot_with_local(idle_seconds=0.0, loaded_name="something_else")
    s = pool_scarcity(model, snap, queue_state=None)
    assert s == 0.0  # not loaded, no idle info → neutral


def test_local_loaded_and_saturated_idle_returns_strong_positive():
    model = _local_model(is_loaded=True)
    snap = _snapshot_with_local(idle_seconds=600.0, loaded_name="test-local")
    s = pool_scarcity(model, snap, queue_state=None)
    assert 0.4 <= s <= 0.5


def test_local_loaded_partial_idle_scales_linearly():
    model = _local_model(is_loaded=True)
    snap = _snapshot_with_local(idle_seconds=300.0, loaded_name="test-local")
    s = pool_scarcity(model, snap, queue_state=None)
    assert 0.20 <= s <= 0.30


def test_local_scarcity_clamped_to_plus_one():
    model = _local_model(is_loaded=True)
    snap = _snapshot_with_local(idle_seconds=999999.0, loaded_name="test-local")
    s = pool_scarcity(model, snap, queue_state=None)
    assert s <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 15 ../../.venv/Scripts/python.exe -m pytest packages/fatih_hoca/tests/test_scarcity.py -v`
Expected: FAIL with `ModuleNotFoundError: fatih_hoca.scarcity`

- [ ] **Step 3: Write minimal implementation (local branch only)**

```python
# packages/fatih_hoca/src/fatih_hoca/scarcity.py
"""Pool scarcity signal for Phase 2d unified utilization equation.

Returns a float in [-1, +1] describing the opportunity cost of using
a given model right now:

    +1   "use it or lose it" — time_bucketed pool with reset imminent
     0   neutral — no preference
    -1   "conserve" — per_call pool with hard tasks queued

Consumed by ranking._apply_utilization_layer as:
    composite *= 1 + UTILIZATION_K * scarcity * (1 - max(0, fit_excess))
"""
from __future__ import annotations

from typing import Any

from fatih_hoca.pools import (
    LOCAL_IDLE_SATURATION_SECS,
    Pool,
    classify_pool,
)

# Soft cap on local-idle scarcity (matches spec §4 range 0.3-0.5)
LOCAL_IDLE_SCARCITY_MAX: float = 0.5
# Penalty when a loaded local is actively processing another request
LOCAL_BUSY_PENALTY: float = -0.10


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _local_scarcity(model: Any, snapshot: Any) -> float:
    local = getattr(snapshot, "local", None)
    if local is None:
        return 0.0

    loaded_name = getattr(local, "model_name", "") or ""
    is_this_model_loaded = (
        getattr(model, "is_loaded", False)
        and loaded_name == getattr(model, "name", None)
    )

    if is_this_model_loaded:
        requests_processing = int(getattr(local, "requests_processing", 0) or 0)
        if requests_processing > 0:
            return LOCAL_BUSY_PENALTY

        idle = float(getattr(local, "idle_seconds", 0.0) or 0.0)
        if idle <= 0:
            return 0.0
        frac = min(1.0, idle / LOCAL_IDLE_SATURATION_SECS)
        return _clamp(frac * LOCAL_IDLE_SCARCITY_MAX)

    # Not loaded — no idle signal, neutral
    return 0.0


def pool_scarcity(model: Any, snapshot: Any, queue_state: Any = None) -> float:
    """Compute signed scarcity in [-1, +1] for (model, snapshot, queue_state)."""
    pool = classify_pool(model)
    if pool is Pool.LOCAL:
        return _local_scarcity(model, snapshot)
    # Time-bucketed + per_call added in Tasks 4 + 5
    return 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 15 ../../.venv/Scripts/python.exe -m pytest packages/fatih_hoca/tests/test_scarcity.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/scarcity.py packages/fatih_hoca/tests/test_scarcity.py
git commit -m "feat(fatih-hoca): scarcity module — local pool branch"
```

---

## Task 4: Scarcity Module — Time-Bucketed Pool

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/scarcity.py` (add `_time_bucketed_scarcity`)
- Modify: `packages/fatih_hoca/tests/test_scarcity.py` (append tests)

- [ ] **Step 1: Write the failing test**

Append to `packages/fatih_hoca/tests/test_scarcity.py`:

```python
import time as _time


def _free_cloud_model(provider="groq", model_id="groq/llama-70b"):
    return SimpleNamespace(
        name=model_id,
        litellm_name=model_id,
        is_local=False,
        is_free=True,
        is_loaded=False,
        provider=provider,
    )


def _snapshot_with_cloud(provider, model_id, remaining, limit, reset_in_secs):
    reset_at = _time.time() + reset_in_secs
    rpd = SimpleNamespace(remaining=remaining, limit=limit, reset_at=reset_at)
    limits = SimpleNamespace(rpd=rpd)
    model_state = SimpleNamespace(limits=limits, utilization_pct=0.0, daily_exhausted=False)
    prov_state = SimpleNamespace(
        models={model_id: model_state},
        limits=limits,
        utilization_pct=0.0,
        consecutive_failures=0,
    )
    return SimpleNamespace(local=None, cloud={provider: prov_state})


def test_time_bucketed_reset_imminent_high_remaining_returns_strong_positive():
    model = _free_cloud_model()
    # 30 min to reset, 85% remaining → strong positive
    snap = _snapshot_with_cloud("groq", "groq/llama-70b", remaining=850, limit=1000, reset_in_secs=1800)
    s = pool_scarcity(model, snap, queue_state=None)
    assert 0.6 <= s <= 1.0


def test_time_bucketed_reset_far_low_remaining_returns_negative():
    model = _free_cloud_model()
    # 5h to reset, 10% remaining → conserve (negative)
    snap = _snapshot_with_cloud("groq", "groq/llama-70b", remaining=100, limit=1000, reset_in_secs=18000)
    s = pool_scarcity(model, snap, queue_state=None)
    assert -0.5 <= s <= -0.2


def test_time_bucketed_balanced_returns_near_zero():
    model = _free_cloud_model()
    # 4h to reset, 50% remaining → neutral
    snap = _snapshot_with_cloud("groq", "groq/llama-70b", remaining=500, limit=1000, reset_in_secs=14400)
    s = pool_scarcity(model, snap, queue_state=None)
    assert -0.2 <= s <= 0.2


def test_time_bucketed_exhausted_returns_zero():
    model = _free_cloud_model()
    snap = _snapshot_with_cloud("groq", "groq/llama-70b", remaining=0, limit=1000, reset_in_secs=3600)
    s = pool_scarcity(model, snap, queue_state=None)
    assert s == 0.0


def test_time_bucketed_missing_provider_returns_zero():
    model = _free_cloud_model(provider="missing-provider")
    snap = SimpleNamespace(local=None, cloud={})
    s = pool_scarcity(model, snap, queue_state=None)
    assert s == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 15 ../../.venv/Scripts/python.exe -m pytest packages/fatih_hoca/tests/test_scarcity.py -v`
Expected: FAIL on the 4 positive/negative/balanced tests (returns 0.0 because Pool.TIME_BUCKETED branch is stubbed)

- [ ] **Step 3: Add the time_bucketed branch**

In `packages/fatih_hoca/src/fatih_hoca/scarcity.py`:

Add imports at top:
```python
import time
```

Add constants after `LOCAL_BUSY_PENALTY`:
```python
# Time-bucketed pool tunables
RESET_IMMINENT_SECS: float = 3600.0   # "imminent" threshold (1h)
RESET_FAR_SECS: float = 14400.0        # "far" threshold (4h)
TIME_BUCKETED_BOOST_MAX: float = 1.0   # max positive when burning
TIME_BUCKETED_CONSERVE_MAX: float = -0.5  # max negative when saving
```

Add helper above `pool_scarcity`:

```python
def _time_bucketed_scarcity(model: Any, snapshot: Any) -> float:
    provider = getattr(model, "provider", "") or ""
    prov_state = getattr(snapshot, "cloud", {}).get(provider)
    if prov_state is None:
        return 0.0

    model_id = getattr(model, "name", None) or getattr(model, "litellm_name", "")
    model_state = prov_state.models.get(model_id) if hasattr(prov_state, "models") else None
    source = model_state if model_state is not None else prov_state

    limits = getattr(source, "limits", None)
    if limits is None:
        return 0.0
    rpd = getattr(limits, "rpd", None)
    if rpd is None:
        return 0.0

    remaining = getattr(rpd, "remaining", None)
    limit = getattr(rpd, "limit", None)
    reset_at = getattr(rpd, "reset_at", None)
    if remaining is None or limit is None or limit <= 0 or remaining <= 0:
        return 0.0

    remaining_frac = min(1.0, remaining / limit)

    if reset_at is not None and reset_at > 0:
        reset_in = max(0.0, reset_at - time.time())
    else:
        return 0.0

    if reset_in <= RESET_IMMINENT_SECS:
        # Reset imminent: burn proportional to remaining fraction
        proximity = 1.0 - (reset_in / RESET_IMMINENT_SECS)  # 0..1
        return _clamp(TIME_BUCKETED_BOOST_MAX * proximity * remaining_frac)

    if reset_in >= RESET_FAR_SECS:
        # Reset far: conserve when low remaining
        # scarcity negative when remaining < 0.3, approaching 0 as remaining rises
        if remaining_frac < 0.3:
            depletion = (0.3 - remaining_frac) / 0.3  # 0..1
            return _clamp(TIME_BUCKETED_CONSERVE_MAX * depletion)
        return 0.0

    # Between imminent and far: neutral
    return 0.0
```

Update dispatcher `pool_scarcity`:

```python
def pool_scarcity(model: Any, snapshot: Any, queue_state: Any = None) -> float:
    pool = classify_pool(model)
    if pool is Pool.LOCAL:
        return _local_scarcity(model, snapshot)
    if pool is Pool.TIME_BUCKETED:
        return _time_bucketed_scarcity(model, snapshot)
    return 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 15 ../../.venv/Scripts/python.exe -m pytest packages/fatih_hoca/tests/test_scarcity.py -v`
Expected: PASS (10 tests total — 5 from Task 3 + 5 new)

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/scarcity.py packages/fatih_hoca/tests/test_scarcity.py
git commit -m "feat(fatih-hoca): scarcity time-bucketed branch — burn imminent, conserve far"
```

---

## Task 5: Scarcity Module — Per-Call Pool

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/scarcity.py` (add `_per_call_scarcity`)
- Modify: `packages/fatih_hoca/tests/test_scarcity.py` (append tests)

Per-call scarcity reads `queue_state.hard_tasks_count` + `queue_state.total_tasks`. When hard tasks are queued and this task is easy, return strong negative. With no queue pressure, return near zero. **Never positive** — per-call costs never encourage spending.

The `queue_state` passed in is a `QueueProfile` (from requirements.py). The **current task's difficulty** is not in `queue_state` — it must be passed separately. We'll extend the dispatcher signature.

- [ ] **Step 1: Write the failing test**

Append to `packages/fatih_hoca/tests/test_scarcity.py`:

```python
def _paid_cloud_model(provider="anthropic", model_id="anthropic/claude-sonnet"):
    return SimpleNamespace(
        name=model_id,
        litellm_name=model_id,
        is_local=False,
        is_free=False,
        is_loaded=False,
        provider=provider,
    )


def _queue_profile(total=0, hard=0, max_d=0):
    return SimpleNamespace(
        total_tasks=total,
        hard_tasks_count=hard,
        max_difficulty=max_d,
        needs_vision_count=0,
        needs_tools_count=0,
        needs_thinking_count=0,
        cloud_only_count=0,
    )


def test_per_call_easy_task_with_hard_queue_returns_strong_negative():
    model = _paid_cloud_model()
    snap = SimpleNamespace(local=None, cloud={})
    qp = _queue_profile(total=20, hard=5, max_d=8)
    s = pool_scarcity(model, snap, queue_state=qp, task_difficulty=3)
    assert -1.0 <= s <= -0.6


def test_per_call_hard_task_with_hard_queue_returns_near_zero():
    # Current task is itself hard → no reason to conserve from it
    model = _paid_cloud_model()
    snap = SimpleNamespace(local=None, cloud={})
    qp = _queue_profile(total=20, hard=5, max_d=8)
    s = pool_scarcity(model, snap, queue_state=qp, task_difficulty=8)
    assert -0.2 <= s <= 0.0


def test_per_call_no_queue_pressure_returns_zero():
    model = _paid_cloud_model()
    snap = SimpleNamespace(local=None, cloud={})
    qp = _queue_profile(total=10, hard=0, max_d=4)
    s = pool_scarcity(model, snap, queue_state=qp, task_difficulty=3)
    assert s == 0.0


def test_per_call_no_queue_state_returns_zero():
    model = _paid_cloud_model()
    snap = SimpleNamespace(local=None, cloud={})
    s = pool_scarcity(model, snap, queue_state=None, task_difficulty=3)
    assert s == 0.0


def test_per_call_never_positive():
    model = _paid_cloud_model()
    snap = SimpleNamespace(local=None, cloud={})
    for qp in [_queue_profile(), _queue_profile(total=50, hard=20, max_d=10)]:
        for d in range(1, 11):
            s = pool_scarcity(model, snap, queue_state=qp, task_difficulty=d)
            assert s <= 0.0, f"per_call positive for d={d} qp.hard={qp.hard_tasks_count}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 15 ../../.venv/Scripts/python.exe -m pytest packages/fatih_hoca/tests/test_scarcity.py -v`
Expected: FAIL on signature (`task_difficulty` not accepted) / TypeError

- [ ] **Step 3: Extend dispatcher signature + add per_call branch**

In `packages/fatih_hoca/src/fatih_hoca/scarcity.py`:

Add constants after `TIME_BUCKETED_CONSERVE_MAX`:

```python
# Per-call pool tunables
PER_CALL_RESERVE_MAX: float = -1.0   # strongest conservation signal
PER_CALL_HARD_QUEUE_RATIO: float = 0.1  # 10% hard tasks in queue → strong pressure
```

Add helper above `pool_scarcity`:

```python
def _per_call_scarcity(queue_state: Any, task_difficulty: int) -> float:
    if queue_state is None:
        return 0.0
    total = int(getattr(queue_state, "total_tasks", 0) or 0)
    hard = int(getattr(queue_state, "hard_tasks_count", 0) or 0)
    if total <= 0 or hard <= 0:
        return 0.0

    # If the CURRENT task is itself hard, no reason for it to be rationed
    if task_difficulty >= 7:
        return 0.0

    hard_ratio = hard / total
    # Saturate pressure at PER_CALL_HARD_QUEUE_RATIO
    pressure = min(1.0, hard_ratio / PER_CALL_HARD_QUEUE_RATIO)
    # Scale by how far below "hard" the current task is (d=1 → full, d=6 → partial)
    easiness = max(0.0, (7 - task_difficulty)) / 6.0  # d=1→1.0, d=7→0
    return _clamp(PER_CALL_RESERVE_MAX * pressure * easiness)
```

Replace the dispatcher signature + routing:

```python
def pool_scarcity(
    model: Any,
    snapshot: Any,
    queue_state: Any = None,
    task_difficulty: int = 0,
) -> float:
    """Compute signed scarcity in [-1, +1].

    Parameters
    ----------
    model : ModelInfo-like
        Must expose `is_local`, `is_free`, `provider`, `name`.
    snapshot : SystemSnapshot-like
        Has `.local` and `.cloud` attrs.
    queue_state : QueueProfile or None
        Optional; used by per_call branch.
    task_difficulty : int
        Current task difficulty (1-10); used by per_call branch.
    """
    pool = classify_pool(model)
    if pool is Pool.LOCAL:
        return _local_scarcity(model, snapshot)
    if pool is Pool.TIME_BUCKETED:
        return _time_bucketed_scarcity(model, snapshot)
    if pool is Pool.PER_CALL:
        return _per_call_scarcity(queue_state, task_difficulty)
    return 0.0
```

Also update Task-3 and Task-4 test calls that previously used the old signature — add `task_difficulty=0` where needed (the earlier local/time_bucketed tests use positional `queue_state=None` which still works since `task_difficulty` defaults to 0).

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 15 ../../.venv/Scripts/python.exe -m pytest packages/fatih_hoca/tests/test_scarcity.py -v`
Expected: PASS (15 tests total)

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/scarcity.py packages/fatih_hoca/tests/test_scarcity.py
git commit -m "feat(fatih-hoca): scarcity per-call branch — queue-aware, never positive"
```

---

## Task 6: Rename `URGENCY_MAX_BONUS` → `UTILIZATION_K`

The Phase 2c constant name becomes misleading. Rename across `pools.py` and any caller.

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/pools.py`
- Modify: `packages/fatih_hoca/src/fatih_hoca/ranking.py` (import + usage)
- Modify: `packages/fatih_hoca/tests/test_pools.py` (if it references the constant)
- Modify: `packages/fatih_hoca/tests/test_capability_gate.py` (if it references the constant)

- [ ] **Step 1: Find all references**

Run: `timeout 10 grep -rn "URGENCY_MAX_BONUS" packages/fatih_hoca/`
Expected: at least `pools.py` (definition), `ranking.py` (import + usage), possibly tests.

- [ ] **Step 2: Rename constant in `pools.py`**

In `packages/fatih_hoca/src/fatih_hoca/pools.py`, replace line `URGENCY_MAX_BONUS: float = 0.25` with:

```python
UTILIZATION_K: float = 0.25
```

And update the module docstring line mentioning `URGENCY_MAX_BONUS` to reference `UTILIZATION_K`.

- [ ] **Step 3: Update `ranking.py` import and usage**

In `packages/fatih_hoca/src/fatih_hoca/ranking.py`:

Change the import line:
```python
from fatih_hoca.pools import (
    Pool, classify_pool, compute_urgency,
    UTILIZATION_K,
)
```

Find the one use of `URGENCY_MAX_BONUS` inside `_apply_urgency_layer` and replace with `UTILIZATION_K`. (This function will be fully replaced in Task 7; for now just make the rename clean so tests still pass.)

- [ ] **Step 4: Update any test references**

Run: `timeout 10 grep -rn "URGENCY_MAX_BONUS" packages/fatih_hoca/tests/` and replace each hit with `UTILIZATION_K`.

- [ ] **Step 5: Run full fatih_hoca test suite**

Run: `timeout 60 ../../.venv/Scripts/python.exe -m pytest packages/fatih_hoca/tests/ -v`
Expected: PASS (all existing tests, nothing regressed)

- [ ] **Step 6: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/pools.py packages/fatih_hoca/src/fatih_hoca/ranking.py packages/fatih_hoca/tests/
git commit -m "refactor(fatih-hoca): rename URGENCY_MAX_BONUS → UTILIZATION_K"
```

---

## Task 7: Replace `_apply_urgency_layer` with `_apply_utilization_layer`

Delete the gated urgency layer; replace with the unified equation. Update `ranking.py` to thread task_difficulty + queue_state through.

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/ranking.py`
- Create: `packages/fatih_hoca/tests/test_utilization_layer.py`
- Modify: `packages/fatih_hoca/tests/test_capability_gate.py` (rewrite or delete obsolete tests)

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_utilization_layer.py
"""Tests for the Phase 2d unified utilization layer."""
from types import SimpleNamespace

from fatih_hoca.ranking import ScoredModel, _apply_utilization_layer


def _sm(name, cap_score_1_to_10, score, is_local=False, is_free=False, is_loaded=False,
        provider=""):
    model = SimpleNamespace(
        name=name,
        litellm_name=name,
        is_local=is_local,
        is_free=is_free,
        is_loaded=is_loaded,
        provider=provider,
    )
    return ScoredModel(
        model=model,
        score=score,
        capability_score=cap_score_1_to_10,
        composite_score=score,
    )


def _blank_snapshot():
    local = SimpleNamespace(
        model_name="",
        idle_seconds=0.0,
        measured_tps=0.0,
        thinking_enabled=False,
        requests_processing=0,
    )
    return SimpleNamespace(local=local, cloud={})


def test_zero_scarcity_leaves_score_unchanged():
    sm = _sm("x", cap_score_1_to_10=6.0, score=100.0, is_local=True)
    # local with no idle and not-loaded → scarcity 0
    snap = _blank_snapshot()
    _apply_utilization_layer([sm], snap, task_difficulty=5, queue_state=None)
    assert sm.score == 100.0


def test_positive_scarcity_under_qualified_model_gets_full_boost():
    # local, loaded, saturated idle → +0.5 scarcity
    # cap_score_100=50, d=5 → cap_needed=45, fit_excess=0.05; (1-0.05)=0.95
    # composite *= 1 + 0.25 * 0.5 * 0.95 = 1.11875
    sm = _sm("loaded-local", cap_score_1_to_10=5.0, score=100.0, is_local=True, is_loaded=True)
    snap = SimpleNamespace(
        local=SimpleNamespace(
            model_name="loaded-local", idle_seconds=600.0,
            measured_tps=20.0, thinking_enabled=False, requests_processing=0,
        ),
        cloud={},
    )
    _apply_utilization_layer([sm], snap, task_difficulty=5, queue_state=None)
    assert 110 < sm.score < 113


def test_over_qualified_model_ignores_positive_scarcity():
    # cap_score_100=95, d=3 → cap_needed=30, fit_excess=0.65
    # (1 - 0.65) = 0.35 → adjustment only 35% of K*scarcity
    # with scarcity +0.5: composite *= 1 + 0.25 * 0.5 * 0.35 = 1.04375
    sm = _sm("overq-local", cap_score_1_to_10=9.5, score=100.0, is_local=True, is_loaded=True)
    snap = SimpleNamespace(
        local=SimpleNamespace(
            model_name="overq-local", idle_seconds=600.0,
            measured_tps=20.0, thinking_enabled=False, requests_processing=0,
        ),
        cloud={},
    )
    _apply_utilization_layer([sm], snap, task_difficulty=3, queue_state=None)
    assert 103 < sm.score < 105


def test_under_qualified_model_feels_full_scarcity_magnitude():
    # cap_score_100=25, d=5 → cap_needed=45, fit_excess=-0.2 → clamped to 0
    # (1 - 0) = 1.0; with scarcity +0.5: composite *= 1 + 0.25 * 0.5 * 1.0 = 1.125
    sm = _sm("weak-local", cap_score_1_to_10=2.5, score=100.0, is_local=True, is_loaded=True)
    snap = SimpleNamespace(
        local=SimpleNamespace(
            model_name="weak-local", idle_seconds=600.0,
            measured_tps=20.0, thinking_enabled=False, requests_processing=0,
        ),
        cloud={},
    )
    _apply_utilization_layer([sm], snap, task_difficulty=5, queue_state=None)
    assert 112 < sm.score < 113


def test_pool_and_urgency_fields_populated():
    sm = _sm("x", cap_score_1_to_10=5.0, score=100.0, is_local=True, is_loaded=True)
    snap = SimpleNamespace(
        local=SimpleNamespace(
            model_name="x", idle_seconds=600.0,
            measured_tps=20.0, thinking_enabled=False, requests_processing=0,
        ),
        cloud={},
    )
    _apply_utilization_layer([sm], snap, task_difficulty=5, queue_state=None)
    assert sm.pool == "local"
    # urgency field is repurposed to store the scalar scarcity for telemetry continuity
    assert 0.4 <= sm.urgency <= 0.5


def test_empty_list_is_no_op():
    _apply_utilization_layer([], _blank_snapshot(), task_difficulty=5, queue_state=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 15 ../../.venv/Scripts/python.exe -m pytest packages/fatih_hoca/tests/test_utilization_layer.py -v`
Expected: FAIL with `ImportError: cannot import name '_apply_utilization_layer'`

- [ ] **Step 3: Replace the layer function in `ranking.py`**

In `packages/fatih_hoca/src/fatih_hoca/ranking.py`:

Update the top imports:
```python
from fatih_hoca.capabilities import TaskRequirements, score_model_for_task
from fatih_hoca.capability_curve import cap_needed_for_difficulty
from fatih_hoca.grading import grading_perf_score
from fatih_hoca.pools import (
    Pool, classify_pool,
    UTILIZATION_K,
)
from fatih_hoca.requirements import get_quota_planner
from fatih_hoca.scarcity import pool_scarcity
```

(Remove `compute_urgency` import; delete the `URGENCY_MAX_BONUS` line if still present.)

Delete the constant `CAP_GATE_RATIO` (and its comment block).

Replace the body of `_apply_urgency_layer` with a rename + new implementation. Delete the old function entirely; add:

```python
def _apply_utilization_layer(
    scored: list[ScoredModel],
    snapshot: SystemSnapshot,
    task_difficulty: int,
    queue_state,
) -> None:
    """Apply Phase 2d unified utilization equation.

    For each ScoredModel:
        fit_excess = (cap_score_100 - cap_needed_for_difficulty(d)) / 100
        scarcity   = pool_scarcity(model, snapshot, queue_state, d)
        composite *= 1 + UTILIZATION_K * scarcity * (1 - max(0, fit_excess))

    Mutates each .score/.composite_score/.pool/.urgency in place.
    Does NOT re-sort — caller is responsible.
    """
    if not scored:
        return
    cap_needed = cap_needed_for_difficulty(task_difficulty)
    for sm in scored:
        cap_score_100 = sm.capability_score * 10.0
        fit_excess = (cap_score_100 - cap_needed) / 100.0
        scarcity = pool_scarcity(sm.model, snapshot, queue_state, task_difficulty)
        pool = classify_pool(sm.model)
        sm.pool = pool.value
        # Reuse `urgency` column for scarcity scalar — telemetry schema continuity
        sm.urgency = scarcity

        if scarcity == 0.0:
            continue
        over_qual_dampener = 1.0 - max(0.0, fit_excess)
        adjustment = 1.0 + UTILIZATION_K * scarcity * over_qual_dampener
        if adjustment == 1.0:
            continue
        sm.score *= adjustment
        sm.composite_score = sm.score
        sign = "+" if scarcity > 0 else "-"
        sm.reasons.append(
            f"util={pool.value}:s={scarcity:+.2f}×({over_qual_dampener:.2f})→{adjustment:.3f}"
        )
```

- [ ] **Step 4: Update the call site in `rank_candidates`**

Find the block:
```python
    # ── Phase 2c: Pool-urgency layer with capability gate ──
    _apply_urgency_layer(scored, snapshot)
    # Re-sort after urgency adjustments (gate may shift ordering)
    scored.sort(key=lambda c: -c.score)
```

Replace with:
```python
    # ── Phase 2d: Unified utilization layer ──
    planner = get_quota_planner()
    _apply_utilization_layer(
        scored,
        snapshot,
        task_difficulty=reqs.difficulty,
        queue_state=planner.queue_profile,
    )
    scored.sort(key=lambda c: -c.score)
```

- [ ] **Step 5: Obsolete-test cleanup**

Open `packages/fatih_hoca/tests/test_capability_gate.py`. The cap-gate tests assert the 0.85× threshold behavior — this is now deleted. Delete any test that specifically checks the gate (e.g. "gated_candidate does not get bonus"). Keep any test that checks bonus math — but since the bonus formula changed, rewrite it or move to `test_utilization_layer.py` and delete the test file if it becomes empty.

Concretely: skim the file, delete tests whose setup is specifically about `CAP_GATE_RATIO`. If the file has 4+ tests and all fail, just delete the file entirely — Phase 2d's utilization tests cover the new behavior.

Run: `timeout 10 grep -l "CAP_GATE_RATIO\|cap_threshold\|_apply_urgency_layer" packages/fatih_hoca/tests/` and manually prune each file.

- [ ] **Step 6: Run full fatih_hoca test suite**

Run: `timeout 120 ../../.venv/Scripts/python.exe -m pytest packages/fatih_hoca/tests/ -v`
Expected: PASS — new `test_utilization_layer.py` passes (6 tests), no regressions.

If `test_ranking.py` has failures specifically around urgency multiplier math, update those assertions to match the new equation.

- [ ] **Step 7: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/ranking.py packages/fatih_hoca/tests/
git commit -m "feat(fatih-hoca): unified utilization layer replaces gated urgency"
```

---

## Task 8: Simulator — SimState

The simulator lives under `packages/fatih_hoca/tests/sim/`. Build it in four chunks: state (Task 8), runner (Task 9), report (Task 10), scenarios + scenario tests (Tasks 11-12).

**Files:**
- Create: `packages/fatih_hoca/tests/sim/__init__.py`
- Create: `packages/fatih_hoca/tests/sim/state.py`
- Create: `packages/fatih_hoca/tests/sim/test_state.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/sim/test_state.py
"""Tests for stateful simulator state (Phase 2d)."""
from fatih_hoca.tests.sim.state import (  # type: ignore
    SimState, SimPoolCounter, SimLocalModel,
)


def test_simstate_init_defaults():
    s = SimState()
    assert s.virtual_clock == 0.0
    assert s.time_bucketed == {}
    assert s.per_call == {}
    assert s.locals == {}


def test_advance_clock():
    s = SimState()
    s.advance_clock(30.5)
    assert s.virtual_clock == 30.5
    s.advance_clock(10.0)
    assert s.virtual_clock == 40.5


def test_time_bucketed_decrement_and_reset():
    s = SimState()
    s.time_bucketed["groq"] = SimPoolCounter(remaining=1000, limit=1000, reset_at=3600.0)
    s.time_bucketed["groq"].remaining -= 1
    assert s.time_bucketed["groq"].remaining == 999

    # Reset fires once clock crosses reset_at
    s.virtual_clock = 3601.0
    s.maybe_reset_buckets()
    assert s.time_bucketed["groq"].remaining == 1000
    # reset_at rolls forward by 86400 (daily)
    assert s.time_bucketed["groq"].reset_at == 3600.0 + 86400.0


def test_per_call_spend_accumulates():
    s = SimState()
    s.per_call["anthropic"] = SimPoolCounter(remaining=30, limit=30, reset_at=86400.0)
    s.per_call["anthropic"].remaining -= 1
    assert s.per_call["anthropic"].remaining == 29


def test_local_idle_increments_when_unused():
    s = SimState()
    s.locals["llama-3"] = SimLocalModel(is_loaded=True, idle_seconds=0.0)
    s.tick_locals(delta_seconds=30.0, used_local_name=None)
    assert s.locals["llama-3"].idle_seconds == 30.0


def test_local_idle_resets_when_used():
    s = SimState()
    s.locals["llama-3"] = SimLocalModel(is_loaded=True, idle_seconds=120.0)
    s.tick_locals(delta_seconds=5.0, used_local_name="llama-3")
    assert s.locals["llama-3"].idle_seconds == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 15 ../../.venv/Scripts/python.exe -m pytest packages/fatih_hoca/tests/sim/test_state.py -v`
Expected: FAIL `ModuleNotFoundError: fatih_hoca.tests.sim.state`

- [ ] **Step 3: Write the module**

```python
# packages/fatih_hoca/tests/sim/__init__.py
```

(empty init)

```python
# packages/fatih_hoca/tests/sim/state.py
"""Stateful simulator state (Phase 2d test infrastructure)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SimPoolCounter:
    remaining: int
    limit: int
    reset_at: float  # virtual-clock seconds


@dataclass
class SimLocalModel:
    is_loaded: bool = False
    idle_seconds: float = 0.0
    tokens_per_second: float = 20.0


@dataclass
class SimState:
    virtual_clock: float = 0.0
    time_bucketed: dict[str, SimPoolCounter] = field(default_factory=dict)
    per_call: dict[str, SimPoolCounter] = field(default_factory=dict)
    locals: dict[str, SimLocalModel] = field(default_factory=dict)

    def advance_clock(self, delta_seconds: float) -> None:
        self.virtual_clock += delta_seconds

    def maybe_reset_buckets(self) -> None:
        """Reset any bucket whose reset_at has elapsed; roll reset_at forward by 24h."""
        for counter in self.time_bucketed.values():
            while counter.reset_at <= self.virtual_clock:
                counter.remaining = counter.limit
                counter.reset_at += 86400.0

    def tick_locals(self, delta_seconds: float, used_local_name: str | None) -> None:
        """Increment idle for all loaded locals; zero the one that was used."""
        for name, local in self.locals.items():
            if not local.is_loaded:
                continue
            if name == used_local_name:
                local.idle_seconds = 0.0
            else:
                local.idle_seconds += delta_seconds
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 15 ../../.venv/Scripts/python.exe -m pytest packages/fatih_hoca/tests/sim/test_state.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/tests/sim/
git commit -m "test(fatih-hoca): simulator SimState — counters, clock, idle tracking"
```

---

## Task 9: Simulator — Runner

The runner evolves `SimState` across a task sequence by calling `fatih_hoca.select()` for each task, recording the pick, decrementing counters, advancing the clock, and handling resets. To keep scope tight, the runner depends on a **snapshot factory function** provided by the scenario, which turns `SimState` into the `SystemSnapshot` the selector expects.

**Files:**
- Create: `packages/fatih_hoca/tests/sim/runner.py`
- Create: `packages/fatih_hoca/tests/sim/test_runner.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/sim/test_runner.py
"""Tests for simulator runner (Phase 2d)."""
from dataclasses import dataclass
from types import SimpleNamespace

from fatih_hoca.tests.sim.state import SimState, SimLocalModel, SimPoolCounter
from fatih_hoca.tests.sim.runner import SimTask, SimRun, run_simulation


def _fake_select(state, task):
    """Always pick 'loaded-local'. Deterministic stub for testing runner mechanics."""
    return SimpleNamespace(
        model_name="loaded-local",
        pool="local",
        estimated_output_tokens=1000,
        tokens_per_second=20.0,
    )


def _fake_snapshot_factory(state):
    return SimpleNamespace(
        local=SimpleNamespace(
            model_name="loaded-local",
            idle_seconds=state.locals.get("loaded-local", SimLocalModel()).idle_seconds,
            measured_tps=20.0,
            requests_processing=0,
            thinking_enabled=False,
        ),
        cloud={},
    )


def test_runner_records_pick_per_task():
    state = SimState()
    state.locals["loaded-local"] = SimLocalModel(is_loaded=True, idle_seconds=100.0)
    tasks = [SimTask(idx=i, difficulty=3) for i in range(5)]
    run: SimRun = run_simulation(
        tasks=tasks,
        initial_state=state,
        select_fn=_fake_select,
        snapshot_factory=_fake_snapshot_factory,
    )
    assert len(run.picks) == 5
    assert all(p.model_name == "loaded-local" for p in run.picks)


def test_runner_advances_clock_per_pick():
    state = SimState()
    state.locals["loaded-local"] = SimLocalModel(is_loaded=True)
    tasks = [SimTask(idx=i, difficulty=3, estimated_output_tokens=1000) for i in range(3)]
    run: SimRun = run_simulation(
        tasks=tasks,
        initial_state=state,
        select_fn=_fake_select,
        snapshot_factory=_fake_snapshot_factory,
    )
    # 1000 tokens / 20 tps = 50s per task, 3 tasks = 150s
    assert run.final_state.virtual_clock == 150.0


def test_runner_resets_used_local_idle():
    state = SimState()
    state.locals["loaded-local"] = SimLocalModel(is_loaded=True, idle_seconds=300.0)
    tasks = [SimTask(idx=0, difficulty=3, estimated_output_tokens=1000)]
    run: SimRun = run_simulation(
        tasks=tasks,
        initial_state=state,
        select_fn=_fake_select,
        snapshot_factory=_fake_snapshot_factory,
    )
    # Used → idle resets
    assert run.final_state.locals["loaded-local"].idle_seconds == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 15 ../../.venv/Scripts/python.exe -m pytest packages/fatih_hoca/tests/sim/test_runner.py -v`
Expected: FAIL `ModuleNotFoundError: fatih_hoca.tests.sim.runner`

- [ ] **Step 3: Write the module**

```python
# packages/fatih_hoca/tests/sim/runner.py
"""Stateful simulator runner (Phase 2d test infrastructure).

Evolves a SimState through a sequence of SimTasks by calling a
caller-provided select function + snapshot factory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from fatih_hoca.tests.sim.state import SimState


@dataclass
class SimTask:
    idx: int
    difficulty: int
    estimated_output_tokens: int = 1000
    task_name: str = "generic"


@dataclass
class SimPick:
    task_idx: int
    task_difficulty: int
    model_name: str
    pool: str
    cap_score_100: float = 0.0
    elapsed_seconds: float = 0.0


@dataclass
class SimRun:
    picks: list[SimPick] = field(default_factory=list)
    final_state: SimState = field(default_factory=SimState)


def run_simulation(
    tasks: list[SimTask],
    initial_state: SimState,
    select_fn: Callable[[SimState, SimTask], Any],
    snapshot_factory: Callable[[SimState], Any],
) -> SimRun:
    """Run the simulator and return SimRun with per-task picks + final state.

    `select_fn(state, task)` must return an object with:
        .model_name, .pool, .estimated_output_tokens, .tokens_per_second
        (optionally .cap_score_100 for reporting)

    `snapshot_factory(state)` builds the SystemSnapshot-like object passed
    into the selector. This is scenario-specific because each scenario wires
    its own pool state into a SystemSnapshot shape.
    """
    state = initial_state
    picks: list[SimPick] = []

    for task in tasks:
        state.maybe_reset_buckets()
        pick = select_fn(state, task)

        elapsed = (
            task.estimated_output_tokens / pick.tokens_per_second
            if pick.tokens_per_second > 0 else 0.0
        )

        used_local = pick.model_name if pick.pool == "local" else None
        state.tick_locals(delta_seconds=elapsed, used_local_name=used_local)
        state.advance_clock(elapsed)

        if pick.pool == "time_bucketed":
            counter = state.time_bucketed.get(pick.model_name)
            if counter is not None and counter.remaining > 0:
                counter.remaining -= 1
        elif pick.pool == "per_call":
            counter = state.per_call.get(pick.model_name)
            if counter is not None and counter.remaining > 0:
                counter.remaining -= 1

        picks.append(SimPick(
            task_idx=task.idx,
            task_difficulty=task.difficulty,
            model_name=pick.model_name,
            pool=pick.pool,
            cap_score_100=getattr(pick, "cap_score_100", 0.0),
            elapsed_seconds=elapsed,
        ))

    return SimRun(picks=picks, final_state=state)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 15 ../../.venv/Scripts/python.exe -m pytest packages/fatih_hoca/tests/sim/test_runner.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/tests/sim/runner.py packages/fatih_hoca/tests/sim/test_runner.py
git commit -m "test(fatih-hoca): simulator runner evolves SimState across task sequence"
```

---

## Task 10: Simulator — Report

Computes the §7 metrics from a `SimRun`.

**Files:**
- Create: `packages/fatih_hoca/tests/sim/report.py`
- Create: `packages/fatih_hoca/tests/sim/test_report.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/sim/test_report.py
"""Tests for simulator report (Phase 2d)."""
from fatih_hoca.tests.sim.runner import SimPick, SimRun
from fatih_hoca.tests.sim.state import SimState, SimPoolCounter
from fatih_hoca.tests.sim.report import compute_metrics


def _picks(*specs):
    out = []
    for i, (d, pool, cap) in enumerate(specs):
        out.append(SimPick(
            task_idx=i, task_difficulty=d, model_name=f"m{i}",
            pool=pool, cap_score_100=cap, elapsed_seconds=10.0,
        ))
    return out


def test_hard_task_satisfaction_100pct():
    picks = _picks((8, "local", 80), (9, "local", 90))
    run = SimRun(picks=picks, final_state=SimState())
    m = compute_metrics(run)
    assert m.hard_task_satisfaction == 1.0


def test_hard_task_satisfaction_50pct():
    # d=8 needs 75; one meets, one doesn't
    picks = _picks((8, "local", 80), (8, "local", 50))
    run = SimRun(picks=picks, final_state=SimState())
    m = compute_metrics(run)
    assert m.hard_task_satisfaction == 0.5


def test_easy_task_waste_rate():
    # d=2 needs 30; cap 95 → fit_excess = 0.65 → > 0.4 → waste
    # d=2 cap 40 → fit_excess = 0.10 → not waste
    picks = _picks((2, "local", 95), (2, "local", 40), (2, "local", 35))
    run = SimRun(picks=picks, final_state=SimState())
    m = compute_metrics(run)
    assert m.easy_task_waste == 1.0 / 3.0


def test_free_quota_utilization():
    final = SimState()
    final.time_bucketed["groq"] = SimPoolCounter(remaining=200, limit=1000, reset_at=0.0)
    run = SimRun(picks=[], final_state=final)
    m = compute_metrics(run)
    # 800 used / 1000 limit = 80%
    assert m.free_quota_utilization == 0.8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 15 ../../.venv/Scripts/python.exe -m pytest packages/fatih_hoca/tests/sim/test_report.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write the module**

```python
# packages/fatih_hoca/tests/sim/report.py
"""Metrics for simulator runs (Phase 2d spec §7)."""
from __future__ import annotations

from dataclasses import dataclass

from fatih_hoca.capability_curve import cap_needed_for_difficulty
from fatih_hoca.tests.sim.runner import SimRun


@dataclass
class SimMetrics:
    hard_task_satisfaction: float = 0.0     # fraction of d>=7 picks meeting cap_needed
    easy_task_waste: float = 0.0            # fraction of d<=4 picks with fit_excess>0.4
    free_quota_utilization: float = 0.0     # avg fraction of time_bucketed capacity consumed
    max_local_idle: float = 0.0             # max idle_seconds across run (future)
    exhaustion_crashes: int = 0             # runner exceptions (0 if clean)


def compute_metrics(run: SimRun) -> SimMetrics:
    m = SimMetrics()

    hard = [p for p in run.picks if p.task_difficulty >= 7]
    if hard:
        passed = sum(
            1 for p in hard
            if p.cap_score_100 >= cap_needed_for_difficulty(p.task_difficulty)
        )
        m.hard_task_satisfaction = passed / len(hard)

    easy = [p for p in run.picks if p.task_difficulty <= 4]
    if easy:
        wasted = 0
        for p in easy:
            fit_excess = (p.cap_score_100 - cap_needed_for_difficulty(p.task_difficulty)) / 100.0
            if fit_excess > 0.4:
                wasted += 1
        m.easy_task_waste = wasted / len(easy)

    tb = run.final_state.time_bucketed
    if tb:
        ratios = []
        for counter in tb.values():
            if counter.limit > 0:
                used = counter.limit - counter.remaining
                ratios.append(max(0.0, used / counter.limit))
        if ratios:
            m.free_quota_utilization = sum(ratios) / len(ratios)

    locals_idle = [l.idle_seconds for l in run.final_state.locals.values()]
    m.max_local_idle = max(locals_idle) if locals_idle else 0.0

    return m
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 15 ../../.venv/Scripts/python.exe -m pytest packages/fatih_hoca/tests/sim/test_report.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/tests/sim/report.py packages/fatih_hoca/tests/sim/test_report.py
git commit -m "test(fatih-hoca): simulator report — spec §7 metrics"
```

---

## Task 11: Simulator — Scenarios

Each scenario is a factory function returning: initial `SimState`, a task sequence, a `snapshot_factory`, and a `select_fn` (which wires through the real `fatih_hoca.select()` against the simulated state).

**Files:**
- Create: `packages/fatih_hoca/tests/sim/scenarios.py`
- No dedicated test file — scenario correctness is asserted by Task 12's pytest cases.

- [ ] **Step 1: Write the module**

```python
# packages/fatih_hoca/tests/sim/scenarios.py
"""Scenario factories for Phase 2d simulator.

Each scenario returns a dict with keys: `tasks`, `initial_state`,
`snapshot_factory`, `select_fn`. Wire through the real fatih_hoca.select()
in `select_fn` so scenarios exercise the live code path — the equation and
scarcity logic being validated.

Keep scenarios focused on the state + demand profile; let the shared
`_build_select_fn` + `_build_snapshot_factory` helpers translate SimState
into the shapes the selector expects.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

from fatih_hoca.tests.sim.runner import SimTask
from fatih_hoca.tests.sim.state import (
    SimLocalModel,
    SimPoolCounter,
    SimState,
)


RNG = random.Random(42)


@dataclass
class Scenario:
    name: str
    tasks: list[SimTask]
    initial_state: SimState
    snapshot_factory: Callable[[SimState], Any]
    select_fn: Callable[[SimState, SimTask], Any]


def _standard_i2p_task_mix(count: int = 182, seed: int = 42) -> list[SimTask]:
    """Approximate i2p v3's task-difficulty distribution.

    Roughly 50% d<=3, 30% d=4-6, 15% d=7-8, 5% d=9-10.
    Uses a fixed-seed RNG for determinism.
    """
    rng = random.Random(seed)
    tasks = []
    for i in range(count):
        r = rng.random()
        if r < 0.50:
            d = rng.choice([1, 2, 3])
        elif r < 0.80:
            d = rng.choice([4, 5, 6])
        elif r < 0.95:
            d = rng.choice([7, 8])
        else:
            d = rng.choice([9, 10])
        est_out = 1000 if d <= 5 else (2500 if d <= 7 else 5000)
        tasks.append(SimTask(idx=i, difficulty=d, estimated_output_tokens=est_out))
    return tasks


def _build_snapshot_factory(scenario_providers: dict[str, Any]):
    """Builds a closure that turns SimState → SystemSnapshot-like object.

    `scenario_providers` maps provider names to static config (limits, etc.)
    so the factory can render per-model cloud state from current SimState counters.
    """
    from types import SimpleNamespace

    def factory(state: SimState) -> Any:
        # Local
        if state.locals:
            loaded_name = next(
                (n for n, l in state.locals.items() if l.is_loaded), ""
            )
            loaded = state.locals.get(loaded_name)
            local_snap = SimpleNamespace(
                model_name=loaded_name,
                idle_seconds=loaded.idle_seconds if loaded else 0.0,
                measured_tps=loaded.tokens_per_second if loaded else 0.0,
                thinking_enabled=False,
                requests_processing=0,
            )
        else:
            local_snap = SimpleNamespace(
                model_name="", idle_seconds=0.0, measured_tps=0.0,
                thinking_enabled=False, requests_processing=0,
            )

        cloud = {}
        for provider, prov_cfg in scenario_providers.items():
            models = {}
            for model_id, model_cfg in prov_cfg.get("models", {}).items():
                if prov_cfg.get("is_free"):
                    counter = state.time_bucketed.get(model_id)
                else:
                    counter = state.per_call.get(model_id)
                if counter is None:
                    models[model_id] = SimpleNamespace(
                        limits=None, utilization_pct=0.0, daily_exhausted=True,
                    )
                    continue
                rpd = SimpleNamespace(
                    remaining=counter.remaining,
                    limit=counter.limit,
                    reset_at=counter.reset_at,
                )
                util = 100.0 * (1.0 - counter.remaining / max(1, counter.limit))
                models[model_id] = SimpleNamespace(
                    limits=SimpleNamespace(rpd=rpd),
                    utilization_pct=util,
                    daily_exhausted=(counter.remaining <= 0),
                )
            cloud[provider] = SimpleNamespace(
                models=models,
                limits=None,
                utilization_pct=0.0,
                consecutive_failures=0,
            )
        return SimpleNamespace(local=local_snap, cloud=cloud)

    return factory


def _build_select_fn(scenario_providers: dict[str, Any]):
    """Wires through the real fatih_hoca.select() against the SimState."""
    from types import SimpleNamespace
    # Deferred imports avoid heavy load at module import time
    from fatih_hoca import selector as _selector

    def select(state: SimState, task: SimTask) -> Any:
        # For Phase 2d we call selector.select_model directly. The selector
        # is responsible for producing a ScoredModel-like object from which we
        # extract name/pool/cap/output-tokens. A lightweight adapter keeps this
        # test-only code decoupled from the selector's private API.
        picked = _selector.select_for_simulation(
            task_name=task.task_name,
            difficulty=task.difficulty,
            estimated_output_tokens=task.estimated_output_tokens,
            snapshot=_build_snapshot_factory(scenario_providers)(state),
            providers_cfg=scenario_providers,
        )
        return SimpleNamespace(
            model_name=picked.model_name,
            pool=picked.pool,
            cap_score_100=picked.cap_score_100,
            estimated_output_tokens=task.estimated_output_tokens,
            tokens_per_second=picked.tokens_per_second,
        )

    return select


# ── Scenario factories ───────────────────────────────────────────────────

def baseline() -> Scenario:
    providers = {
        "groq": {
            "is_free": True,
            "models": {"groq/llama-3.1-70b": {"cap_score_100": 72}},
        },
        "anthropic": {
            "is_free": False,
            "models": {"anthropic/claude-sonnet": {"cap_score_100": 93}},
        },
    }
    state = SimState()
    state.locals["loaded-local"] = SimLocalModel(is_loaded=True, idle_seconds=300.0, tokens_per_second=20.0)
    state.time_bucketed["groq/llama-3.1-70b"] = SimPoolCounter(remaining=1000, limit=1000, reset_at=86400.0)
    state.per_call["anthropic/claude-sonnet"] = SimPoolCounter(remaining=1000, limit=1000, reset_at=86400.0)
    return Scenario(
        name="baseline",
        tasks=_standard_i2p_task_mix(),
        initial_state=state,
        snapshot_factory=_build_snapshot_factory(providers),
        select_fn=_build_select_fn(providers),
    )


def claude_constrained() -> Scenario:
    providers = {
        "anthropic": {
            "is_free": False,
            "models": {"anthropic/claude-sonnet": {"cap_score_100": 93}},
        },
        "groq": {
            "is_free": True,
            "models": {"groq/llama-3.1-70b": {"cap_score_100": 72}},
        },
    }
    state = SimState()
    state.locals["loaded-local"] = SimLocalModel(is_loaded=True, idle_seconds=300.0)
    state.per_call["anthropic/claude-sonnet"] = SimPoolCounter(remaining=30, limit=30, reset_at=86400.0)
    state.time_bucketed["groq/llama-3.1-70b"] = SimPoolCounter(remaining=1000, limit=1000, reset_at=86400.0)
    return Scenario(
        name="claude_constrained",
        tasks=_standard_i2p_task_mix(),
        initial_state=state,
        snapshot_factory=_build_snapshot_factory(providers),
        select_fn=_build_select_fn(providers),
    )


def groq_near_reset() -> Scenario:
    providers = {
        "groq": {"is_free": True, "models": {"groq/llama-3.1-70b": {"cap_score_100": 72}}},
        "anthropic": {"is_free": False, "models": {"anthropic/claude-sonnet": {"cap_score_100": 93}}},
    }
    state = SimState()
    state.locals["loaded-local"] = SimLocalModel(is_loaded=True, idle_seconds=300.0)
    # 30min to reset, 85% remaining
    state.time_bucketed["groq/llama-3.1-70b"] = SimPoolCounter(remaining=850, limit=1000, reset_at=1800.0)
    state.per_call["anthropic/claude-sonnet"] = SimPoolCounter(remaining=1000, limit=1000, reset_at=86400.0)
    return Scenario(
        name="groq_near_reset",
        tasks=_standard_i2p_task_mix(),
        initial_state=state,
        snapshot_factory=_build_snapshot_factory(providers),
        select_fn=_build_select_fn(providers),
    )


def diverse_pool() -> Scenario:
    providers = {
        "groq": {"is_free": True, "models": {"groq/llama-3.1-70b": {"cap_score_100": 72}}},
        "gemini": {"is_free": True, "models": {"gemini/gemini-1.5-flash": {"cap_score_100": 68}}},
        "openrouter": {"is_free": True, "models": {"openrouter/free-mistral": {"cap_score_100": 55}}},
        "anthropic": {"is_free": False, "models": {"anthropic/claude-sonnet": {"cap_score_100": 93}}},
    }
    state = SimState()
    state.locals["loaded-local"] = SimLocalModel(is_loaded=True, idle_seconds=300.0)
    state.time_bucketed["groq/llama-3.1-70b"] = SimPoolCounter(remaining=1000, limit=1000, reset_at=86400.0)
    state.time_bucketed["gemini/gemini-1.5-flash"] = SimPoolCounter(remaining=1500, limit=1500, reset_at=86400.0)
    state.time_bucketed["openrouter/free-mistral"] = SimPoolCounter(remaining=500, limit=500, reset_at=86400.0)
    state.per_call["anthropic/claude-sonnet"] = SimPoolCounter(remaining=1000, limit=1000, reset_at=86400.0)
    return Scenario(
        name="diverse_pool",
        tasks=_standard_i2p_task_mix(),
        initial_state=state,
        snapshot_factory=_build_snapshot_factory(providers),
        select_fn=_build_select_fn(providers),
    )


def exhaustion_sequence() -> Scenario:
    # Tight free-tier budgets; sequence exhausts them progressively
    providers = {
        "groq": {"is_free": True, "models": {"groq/llama-3.1-70b": {"cap_score_100": 72}}},
        "gemini": {"is_free": True, "models": {"gemini/gemini-1.5-flash": {"cap_score_100": 68}}},
        "anthropic": {"is_free": False, "models": {"anthropic/claude-sonnet": {"cap_score_100": 93}}},
    }
    state = SimState()
    state.locals["loaded-local"] = SimLocalModel(is_loaded=True, idle_seconds=300.0)
    state.time_bucketed["groq/llama-3.1-70b"] = SimPoolCounter(remaining=40, limit=40, reset_at=86400.0)
    state.time_bucketed["gemini/gemini-1.5-flash"] = SimPoolCounter(remaining=40, limit=40, reset_at=86400.0)
    state.per_call["anthropic/claude-sonnet"] = SimPoolCounter(remaining=1000, limit=1000, reset_at=86400.0)
    return Scenario(
        name="exhaustion_sequence",
        tasks=_standard_i2p_task_mix(count=182),
        initial_state=state,
        snapshot_factory=_build_snapshot_factory(providers),
        select_fn=_build_select_fn(providers),
    )


def back_to_back_i2p() -> Scenario:
    providers = diverse_pool().snapshot_factory  # placeholder; overwritten below
    # Rebuild providers cleanly
    providers_cfg = {
        "groq": {"is_free": True, "models": {"groq/llama-3.1-70b": {"cap_score_100": 72}}},
        "anthropic": {"is_free": False, "models": {"anthropic/claude-sonnet": {"cap_score_100": 93}}},
    }
    state = SimState()
    state.locals["loaded-local"] = SimLocalModel(is_loaded=True, idle_seconds=300.0)
    state.time_bucketed["groq/llama-3.1-70b"] = SimPoolCounter(remaining=1500, limit=1500, reset_at=86400.0)
    state.per_call["anthropic/claude-sonnet"] = SimPoolCounter(remaining=1000, limit=1000, reset_at=86400.0)
    # Three i2p runs back to back with fresh difficulty distributions, distinct seeds
    tasks = []
    for run_idx, seed in enumerate([42, 43, 44]):
        batch = _standard_i2p_task_mix(count=182, seed=seed)
        for t in batch:
            tasks.append(SimTask(
                idx=len(tasks), difficulty=t.difficulty,
                estimated_output_tokens=t.estimated_output_tokens,
            ))
    return Scenario(
        name="back_to_back_i2p",
        tasks=tasks,
        initial_state=state,
        snapshot_factory=_build_snapshot_factory(providers_cfg),
        select_fn=_build_select_fn(providers_cfg),
    )


def staggered_i2p() -> Scenario:
    providers_cfg = {
        "groq": {"is_free": True, "models": {"groq/llama-3.1-70b": {"cap_score_100": 72}}},
        "anthropic": {"is_free": False, "models": {"anthropic/claude-sonnet": {"cap_score_100": 93}}},
    }
    state = SimState()
    state.locals["loaded-local"] = SimLocalModel(is_loaded=True, idle_seconds=300.0)
    state.time_bucketed["groq/llama-3.1-70b"] = SimPoolCounter(remaining=1000, limit=1000, reset_at=86400.0)
    state.per_call["anthropic/claude-sonnet"] = SimPoolCounter(remaining=1000, limit=1000, reset_at=86400.0)

    # First i2p: 91 tasks (mid-flight equivalent), skewed easy in last half
    first = _standard_i2p_task_mix(count=91, seed=100)
    # Second i2p: fresh 182, full distribution
    second = _standard_i2p_task_mix(count=182, seed=101)
    tasks = []
    for t in first + second:
        tasks.append(SimTask(
            idx=len(tasks), difficulty=t.difficulty,
            estimated_output_tokens=t.estimated_output_tokens,
        ))
    return Scenario(
        name="staggered_i2p",
        tasks=tasks,
        initial_state=state,
        snapshot_factory=_build_snapshot_factory(providers_cfg),
        select_fn=_build_select_fn(providers_cfg),
    )
```

- [ ] **Step 2: Add `selector.select_for_simulation` adapter**

The scenarios call `selector.select_for_simulation(...)` which is a test-only convenience. Add it to `packages/fatih_hoca/src/fatih_hoca/selector.py`:

First check what's currently in selector — the real entry point is already `select(...)`. Add a new thin function at the bottom of that file:

```python
# ─── Test-only helper (Phase 2d scenarios) ───────────────────────────────

@dataclass
class _SimPickResult:
    model_name: str
    pool: str
    cap_score_100: float
    tokens_per_second: float


def select_for_simulation(
    *,
    task_name: str,
    difficulty: int,
    estimated_output_tokens: int,
    snapshot: Any,
    providers_cfg: dict,
) -> "_SimPickResult":
    """Test-only adapter: build a minimal ModelRequirements + candidate list
    from providers_cfg, call rank_candidates, return a light result.

    This keeps the simulator independent of fatih_hoca's real init (catalog,
    benchmark cache, etc.) which is too heavy for fast pytest runs.
    """
    # Light-weight in-memory model registry from providers_cfg
    # Implementation detail: delegate to a helper that wires ModelInfo stubs
    # and invokes rank_candidates directly.
    raise NotImplementedError("Task 11 step 2 — wire in Task 12")
```

(Placeholder raises so tests force us to complete wiring in Task 12. This is the one acceptable stub because the glue is substantial and belongs in its own task.)

- [ ] **Step 3: Run module import sanity**

Run: `timeout 15 ../../.venv/Scripts/python.exe -c "from fatih_hoca.tests.sim import scenarios; print(scenarios.baseline().name)"`
Expected: prints `baseline` (no runtime error — `select_for_simulation` isn't called yet).

- [ ] **Step 4: Commit**

```bash
git add packages/fatih_hoca/tests/sim/scenarios.py packages/fatih_hoca/src/fatih_hoca/selector.py
git commit -m "test(fatih-hoca): scenario factories for Phase 2d simulator"
```

---

## Task 12: Wire `select_for_simulation` + Pass All 7 Scenarios

This is the tuning + integration task. Implement the selector adapter, then run the 7 scenarios and iterate on `K`, scarcity piecewise values until §7 metrics pass.

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/selector.py` (complete `select_for_simulation`)
- Create: `packages/fatih_hoca/tests/test_scenarios.py`

- [ ] **Step 1: Write the scenario pytest cases**

```python
# packages/fatih_hoca/tests/test_scenarios.py
"""Scenario validation for Phase 2d (spec §6, §7)."""
import pytest

from fatih_hoca.tests.sim.report import compute_metrics
from fatih_hoca.tests.sim.runner import run_simulation
from fatih_hoca.tests.sim.scenarios import (
    baseline, claude_constrained, groq_near_reset, diverse_pool,
    exhaustion_sequence, back_to_back_i2p, staggered_i2p,
)


SCENARIOS = [
    ("baseline", baseline),
    ("claude_constrained", claude_constrained),
    ("groq_near_reset", groq_near_reset),
    ("diverse_pool", diverse_pool),
    ("exhaustion_sequence", exhaustion_sequence),
    ("back_to_back_i2p", back_to_back_i2p),
    ("staggered_i2p", staggered_i2p),
]


@pytest.mark.parametrize("name,factory", SCENARIOS)
def test_scenario_hard_task_satisfaction(name, factory):
    scenario = factory()
    run = run_simulation(
        tasks=scenario.tasks,
        initial_state=scenario.initial_state,
        select_fn=scenario.select_fn,
        snapshot_factory=scenario.snapshot_factory,
    )
    m = compute_metrics(run)
    assert m.hard_task_satisfaction >= 0.90, (
        f"{name}: hard-task satisfaction {m.hard_task_satisfaction:.2%} < 90%"
    )


@pytest.mark.parametrize("name,factory", SCENARIOS)
def test_scenario_easy_task_waste(name, factory):
    scenario = factory()
    run = run_simulation(
        tasks=scenario.tasks,
        initial_state=scenario.initial_state,
        select_fn=scenario.select_fn,
        snapshot_factory=scenario.snapshot_factory,
    )
    m = compute_metrics(run)
    assert m.easy_task_waste < 0.10, (
        f"{name}: easy-task waste {m.easy_task_waste:.2%} >= 10%"
    )


def test_diverse_pool_free_quota_utilization():
    scenario = diverse_pool()
    run = run_simulation(
        tasks=scenario.tasks,
        initial_state=scenario.initial_state,
        select_fn=scenario.select_fn,
        snapshot_factory=scenario.snapshot_factory,
    )
    m = compute_metrics(run)
    assert m.free_quota_utilization > 0.70, (
        f"diverse_pool: free_quota_utilization {m.free_quota_utilization:.2%} <= 70%"
    )


def test_exhaustion_sequence_no_crashes():
    scenario = exhaustion_sequence()
    # Must complete without raising
    run = run_simulation(
        tasks=scenario.tasks,
        initial_state=scenario.initial_state,
        select_fn=scenario.select_fn,
        snapshot_factory=scenario.snapshot_factory,
    )
    assert len(run.picks) == len(scenario.tasks)
```

- [ ] **Step 2: Complete `select_for_simulation`**

Replace the `raise NotImplementedError` in `packages/fatih_hoca/src/fatih_hoca/selector.py::select_for_simulation` with a real implementation that:

1. Builds `ModelInfo` stubs from `providers_cfg` + a fixed local model (`loaded-local` with `cap_score_100=55`).
2. Constructs a `ModelRequirements` from `task_name` + `difficulty` + `estimated_output_tokens`.
3. Calls `ranking.rank_candidates(...)` directly.
4. Returns the top scored model as a `_SimPickResult`.

Concrete implementation:

```python
def select_for_simulation(
    *,
    task_name: str,
    difficulty: int,
    estimated_output_tokens: int,
    snapshot: Any,
    providers_cfg: dict,
) -> "_SimPickResult":
    from types import SimpleNamespace
    from fatih_hoca.ranking import rank_candidates
    from fatih_hoca.requirements import ModelRequirements

    candidates: list[Any] = []

    # Local stub — one loaded local
    local_model = SimpleNamespace(
        name="loaded-local",
        litellm_name="loaded-local",
        is_local=True,
        is_loaded=True,
        is_free=False,
        provider="local",
        capabilities=SimpleNamespace(as_weights=lambda: {}),
        tokens_per_second=20.0,
        load_time_seconds=0.0,
        total_params_b=7,
        active_params_b=7,
        specialty=None,
        thinking_model=False,
        operational_dict=lambda: {"context_window": 32000},
        estimated_cost=lambda inp, out: 0.0,
        location="local",
    )
    # Shim cap_score by wiring a fake score_model_for_task? Too invasive.
    # Instead, we add a capabilities.raw_score_override in providers_cfg and
    # monkeypatch score_model_for_task in tests. For Phase 2d simulator, give
    # each stub a fixed cap_score_100 via a special attribute read by a
    # thin scoring shim here:

    def _score(m, cap_100):
        return cap_100 / 10.0  # raw 0-10

    from fatih_hoca import ranking as _ranking_mod
    # Monkey-patch score_model_for_task for the duration of this call so stubs
    # report their declared cap_score_100 directly.
    real_score = _ranking_mod.score_model_for_task
    cap_overrides: dict[str, float] = {"loaded-local": 55.0}

    # Cloud stubs
    for provider, cfg in providers_cfg.items():
        is_free = cfg.get("is_free", False)
        for model_id, model_cfg in cfg.get("models", {}).items():
            cap_overrides[model_id] = float(model_cfg.get("cap_score_100", 50.0))
            candidates.append(SimpleNamespace(
                name=model_id,
                litellm_name=model_id,
                is_local=False,
                is_loaded=False,
                is_free=is_free,
                provider=provider,
                capabilities=SimpleNamespace(as_weights=lambda: {}),
                tokens_per_second=0.0,
                load_time_seconds=0.0,
                total_params_b=0,
                active_params_b=0,
                specialty=None,
                thinking_model=False,
                operational_dict=lambda: {"context_window": 128000},
                estimated_cost=lambda inp, out, _m=model_id: 0.005 if not is_free else 0.0,
                location="cloud",
            ))
    candidates.append(local_model)

    def _fake_score(model_capabilities, model_operational, requirements):
        # Find the cap_override by model name via closure
        for c in candidates:
            if c.capabilities is model_capabilities:
                return _score(c, cap_overrides.get(c.name, 50.0)) * 10.0  # returns raw 0-100? rank expects 0-10
        return 5.0
    # Note: score_model_for_task returns a 0-10 weighted mean per ranking.py line 289.
    # Adjust: we want raw 0-10, so divide cap_100 by 10.
    def _fake_score_0_10(model_capabilities, model_operational, requirements):
        for c in candidates:
            if c.capabilities is model_capabilities:
                return cap_overrides.get(c.name, 50.0) / 10.0
        return 5.0

    _ranking_mod.score_model_for_task = _fake_score_0_10
    try:
        reqs = ModelRequirements(
            task=task_name or "generic",
            difficulty=difficulty,
            estimated_output_tokens=estimated_output_tokens,
        )
        scored = rank_candidates(
            candidates=candidates,
            reqs=reqs,
            snapshot=snapshot,
            failures=[],
            remaining_budget=300.0,
        )
    finally:
        _ranking_mod.score_model_for_task = real_score

    if not scored:
        # Fallback — should not happen with non-empty candidates
        return _SimPickResult(
            model_name="loaded-local", pool="local",
            cap_score_100=55.0, tokens_per_second=20.0,
        )

    top = scored[0]
    return _SimPickResult(
        model_name=top.model.name,
        pool=top.pool or "local",
        cap_score_100=top.capability_score * 10.0,
        tokens_per_second=top.model.tokens_per_second or 20.0,
    )
```

- [ ] **Step 3: Run the scenarios**

Run: `timeout 180 ../../.venv/Scripts/python.exe -m pytest packages/fatih_hoca/tests/test_scenarios.py -v`
Expected outcome: **Some scenarios may fail initially.** This is the tuning phase.

- [ ] **Step 4: Tune until all pass**

For each failing scenario:

1. Read the failure message (which metric, which scenario).
2. Suspect: `UTILIZATION_K` too small/large, `LOCAL_IDLE_SCARCITY_MAX`, `TIME_BUCKETED_BOOST_MAX`, `PER_CALL_RESERVE_MAX`, or `CAP_NEEDED_BY_DIFFICULTY` curve values.
3. Adjust one constant at a time (small steps: ±0.05 on K, ±5 on curve entries).
4. Re-run. Stop once all 7 × 2 parametrized tests + 2 standalone tests are green.

Budget: no more than 8 iterations. If stuck after that, review the scenario task distributions — the simulator may be generating unrealistic sequences.

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/selector.py packages/fatih_hoca/tests/test_scenarios.py packages/fatih_hoca/src/fatih_hoca/scarcity.py packages/fatih_hoca/src/fatih_hoca/pools.py packages/fatih_hoca/src/fatih_hoca/capability_curve.py
git commit -m "feat(fatih-hoca): Phase 2d — tune utilization constants until all 7 scenarios pass"
```

---

## Task 13: Counterfactual CLI Rewrite

`counterfactual.py::_rescore` must invoke the new equation.

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/counterfactual.py`
- Modify: `packages/fatih_hoca/tests/test_counterfactual.py`

- [ ] **Step 1: Inspect current `_rescore`**

Run: `timeout 10 ../../.venv/Scripts/python.exe -c "import inspect, fatih_hoca.counterfactual as c; print(inspect.getsource(c._rescore))"`

- [ ] **Step 2: Write the failing test**

Append to `packages/fatih_hoca/tests/test_counterfactual.py`:

```python
def test_rescore_uses_utilization_equation(monkeypatch):
    """_rescore must invoke pool_scarcity + cap_needed_for_difficulty, not gate logic."""
    from fatih_hoca.counterfactual import _rescore
    # Feed a candidate row with cap_score=50, pool=local, task_difficulty=5
    # With scarcity=+0.5 and fit_excess=0.05:
    # expected = orig * (1 + 0.25 * 0.5 * 0.95) = orig * 1.11875
    result = _rescore(
        original_score=100.0,
        cap_score_100=50.0,
        task_difficulty=5,
        scarcity=0.5,
        K=0.25,
    )
    assert 111 < result < 113
```

- [ ] **Step 3: Run to confirm failure**

Run: `timeout 15 ../../.venv/Scripts/python.exe -m pytest packages/fatih_hoca/tests/test_counterfactual.py::test_rescore_uses_utilization_equation -v`
Expected: FAIL — either missing parameter or old gate-based logic.

- [ ] **Step 4: Rewrite `_rescore`**

In `packages/fatih_hoca/src/fatih_hoca/counterfactual.py`, replace `_rescore` body with:

```python
def _rescore(
    original_score: float,
    cap_score_100: float,
    task_difficulty: int,
    scarcity: float,
    K: float = 0.25,
) -> float:
    """Re-score a historical candidate under the Phase 2d utilization equation."""
    from fatih_hoca.capability_curve import cap_needed_for_difficulty
    cap_needed = cap_needed_for_difficulty(task_difficulty)
    fit_excess = (cap_score_100 - cap_needed) / 100.0
    over_qual_dampener = 1.0 - max(0.0, fit_excess)
    adjustment = 1.0 + K * scarcity * over_qual_dampener
    return original_score * adjustment
```

If the existing `_rescore` is called with different arguments elsewhere, update call sites in `counterfactual.py` to pass the new shape. The CLI sweep loop should sweep `K` (e.g. 0.15, 0.20, 0.25, 0.30, 0.35) rather than `cap_gate_ratio`.

- [ ] **Step 5: Update any broken tests in test_counterfactual.py**

Old sweep tests that referenced `cap_gate_ratio` must be rewritten to sweep `K`. Delete obsolete gate-specific cases.

- [ ] **Step 6: Run full counterfactual test suite**

Run: `timeout 30 ../../.venv/Scripts/python.exe -m pytest packages/fatih_hoca/tests/test_counterfactual.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/counterfactual.py packages/fatih_hoca/tests/test_counterfactual.py
git commit -m "feat(fatih-hoca): counterfactual CLI sweeps K under new utilization equation"
```

---

## Task 14: Full Suite Regression + Integration Sweep

Run every fatih_hoca + nerd_herd test to confirm no regressions.

- [ ] **Step 1: Run fatih_hoca package tests**

Run: `timeout 180 ../../.venv/Scripts/python.exe -m pytest packages/fatih_hoca/tests/ -v`
Expected: PASS (all tests including sim/ + scenarios)

- [ ] **Step 2: Run cross-package tests**

Run: `timeout 120 ../../.venv/Scripts/python.exe -m pytest tests/fatih_hoca/ -v`
Expected: PASS

- [ ] **Step 3: Run nerd_herd tests**

Run: `timeout 60 ../../.venv/Scripts/python.exe -m pytest packages/nerd_herd/tests/ -v`
Expected: PASS

- [ ] **Step 4: Integration smoke test**

Run: `timeout 30 ../../.venv/Scripts/python.exe -c "from fatih_hoca import selector; from fatih_hoca.ranking import _apply_utilization_layer; from fatih_hoca.scarcity import pool_scarcity; from fatih_hoca.capability_curve import cap_needed_for_difficulty; print('imports ok')"`
Expected: `imports ok`

- [ ] **Step 5: If all green, commit nothing (no new changes) — proceed to Task 15.**

---

## Task 15: Docs Update

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/architecture-modularization.md` (add Phase 2d section)

- [ ] **Step 1: Update CLAUDE.md Phase 2c note**

In `CLAUDE.md`, find the "Phase 2c machinery" bullet in "Common Pitfalls". Replace with:

```markdown
- **Phase 2d unified utilization (2026-04-19)**: `packages/fatih_hoca/src/fatih_hoca/scarcity.py` computes signed scarcity ∈ [-1, +1] per pool. `capability_curve.py` holds `CAP_NEEDED_BY_DIFFICULTY` (dict d→cap_100). `ranking.py::_apply_utilization_layer` applies: `composite *= 1 + K * scarcity * (1 - max(0, fit_excess))` where `K = UTILIZATION_K = 0.25`. No gate — the over-qualification dampener handles "don't waste strong models on easy tasks" naturally. Phase 2c's urgency+gate retired. Validated by 7 stateful-simulator scenarios in `packages/fatih_hoca/tests/test_scenarios.py`. To change tuning constants, sweep via simulator first — never eyeball.
```

- [ ] **Step 2: Add Phase 2d section to architecture doc**

Append to `docs/architecture-modularization.md` in the appropriate Fatih Hoca section (find where Phase 2c was mentioned and add after):

```markdown
### Phase 2d — Unified Utilization Equation (2026-04-19)

Replaces Phase 2c's gated urgency layer with one multiplier derived from pool scarcity × capability fit-excess. Validated by a stateful pytest simulator that evolves per-pool counters and a virtual clock across 7 scenarios. Key files:

- `packages/fatih_hoca/src/fatih_hoca/capability_curve.py` — `CAP_NEEDED_BY_DIFFICULTY` lookup
- `packages/fatih_hoca/src/fatih_hoca/scarcity.py` — signed scarcity per pool, queue-aware for per_call
- `packages/fatih_hoca/src/fatih_hoca/ranking.py::_apply_utilization_layer` — the equation
- `packages/fatih_hoca/tests/sim/` — stateful simulator (test infrastructure, not shipped)
- `packages/fatih_hoca/tests/test_scenarios.py` — 7 scenarios asserting §7 pass criteria

The over-qualification dampener `(1 - max(0, fit_excess))` replaces the old cap-gate: strong models get smaller utilization adjustments on easy tasks (naturally discouraging "Claude on d=3") without hard-blocking anyone.
```

- [ ] **Step 3: Commit docs**

```bash
git add CLAUDE.md docs/architecture-modularization.md
git commit -m "docs: Phase 2d unified utilization equation"
```

---

## Task 16: Final Suite + Merge Prep

- [ ] **Step 1: Rerun the full test suite**

Run: `timeout 300 ../../.venv/Scripts/python.exe -m pytest packages/fatih_hoca/ tests/fatih_hoca/ packages/nerd_herd/ -v`
Expected: PASS

- [ ] **Step 2: List commits on the branch**

Run: `git log origin/main..HEAD --oneline`
Expected: 15-16 commits, one per task.

- [ ] **Step 3: Report out**

Summarize to the user:
- Scenarios passing: 7/7
- Final tuned constants: `K=?`, local max `?`, time_bucketed `?/?`, per_call `?`
- Total commits, final test count

Ask whether to merge into main (same `--no-ff` merge pattern as Phase 2c).

---

## Appendix: Spec Coverage Self-Review

| Spec section | Task(s) |
|---|---|
| §2 Core equation | 7 |
| §3 `cap_needed_for_difficulty` | 1 |
| §4 `pool_scarcity` per pool | 3, 4, 5 |
| §5 Simulator state/runner/report/scenarios | 8, 9, 10, 11, 12 |
| §6 Seven scenarios | 11, 12 |
| §7 Pass criteria asserted | 12 |
| §8 Phase 2c foundation preserved | untouched — no task touches pools.py pool classification, grading.py, idle_seconds |
| §9 Deletions (gate + `URGENCY_MAX_BONUS`) | 6, 7 |
| §10 Scope guardrails | enforced by task list — no tasks for empirical curve / nerd_herd extension / cloud wiring |
| §11 TDD per task | every task follows test-first pattern |
| §12 Worktree + never-install-e | noted in header |
| §13 Implementation order | matches plan order |

No spec requirement is without a task. No task introduces work outside the spec.
