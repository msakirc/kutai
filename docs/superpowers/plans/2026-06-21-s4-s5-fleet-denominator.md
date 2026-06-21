# S4/S5 Fleet-Capacity Denominator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reshape queue-conservation signals S4/S5 to divide queue demand by the *fleet's* cycle-remaining (Σ all cloud models, per rate-limit axis, minus per-model in-flight) instead of a single model's window — killing the aggregate-÷-single over-conservation leak while keeping genuine fleet-exhaustion conservation (pp11).

**Architecture:** S4 (`s4_queue_tokens`) and S5 (`s5_queue_calls`) gain a `fleet_remaining: dict[str,int] | None` param. `SystemSnapshot.pressure_for` builds `fleet_remaining` from `self.cloud` when not passed (pressure-only path) and forwards it; the ranking layer precomputes it once per pick (prod perf path) and passes it in. Direct bare-matrix calls fall back to per-model remaining (== today's behavior).

**Tech Stack:** Python 3.10, pytest. Packages `nerd_herd` (signals, types) and `fatih_hoca` (ranking, sim).

**Spec:** `docs/superpowers/specs/2026-06-21-s4-s5-fleet-denominator-design.md`

**Invariants that MUST hold (do not break):**
- S4/S5 ∈ `QUEUE_BUCKET` — rank multiplier only, NEVER the supply veto (`48e4cee8`). Do not move them to `OTHER_BUCKET`.
- pp11 (`assert_pp11_daily_overshoot_still_conserves`) stays green.
- Per-minute axes stay excluded (`cycle_*_cells()` already drops them).
- No constant change (THRESHOLD=0.70, SLOPE=2.0, W_QUEUE=0.7) without a sim delta justifying it.

**Windows testing rule:** run pytest FOREGROUND with `timeout`, NEVER `run_in_background` (orphans hold the prod SQLite lock). Use `python -m pytest ... -o addopts="" -p no:aiohttp` for package tests.

---

### Task 1: S4 fleet-capacity denominator

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/signals/s4_queue_tokens.py`
- Test: `packages/nerd_herd/tests/signals/test_s4.py`

- [ ] **Step 1: Write the failing tests**

Append to `packages/nerd_herd/tests/signals/test_s4.py`:

```python
# ── Fleet-capacity denominator (2026-06-21) ───────────────────────────────────

def test_s4_fleet_denominator_dilutes_small_window():
    # This model's own tpd window is tiny (20k), but the FLEET has 1.02M on the
    # tpd axis. A 400k queue projection is 5% of the fleet -> no conservation,
    # even though it is 20x this model's own window.
    m = _matrix(tpd=RateLimit(limit=20_000, remaining=20_000))
    qp = QueueProfile(projected_tokens=400_000)
    assert s4_queue_tokens(m, queue=qp, fleet_remaining={"tpd": 1_020_000}) == 0.0


def test_s4_per_model_fallback_when_no_fleet():
    # No fleet_remaining -> falls back to this model's own remaining (old behavior).
    m = _matrix(tpd=RateLimit(limit=20_000, remaining=20_000))
    qp = QueueProfile(projected_tokens=400_000)  # 20x this model's window
    assert s4_queue_tokens(m, queue=qp) == pytest.approx(-1.0, abs=0.05)


def test_s4_fleet_of_one_equals_per_model():
    # Fleet sum of a fleet-of-one == this model's remaining -> still conserves
    # (the genuine "no escape hatch" case, e.g. pp11).
    m = _matrix(tpd=RateLimit(limit=20_000, remaining=20_000))
    qp = QueueProfile(projected_tokens=40_000)  # 2x the only window
    assert s4_queue_tokens(m, queue=qp, fleet_remaining={"tpd": 20_000}) == pytest.approx(-1.0, abs=0.05)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest packages/nerd_herd/tests/signals/test_s4.py -v -o addopts="" -p no:aiohttp`
Expected: the 3 new tests FAIL with `TypeError: s4_queue_tokens() got an unexpected keyword argument 'fleet_remaining'`.

- [ ] **Step 3: Add the fleet_remaining param**

Replace the body of `s4_queue_tokens` in `packages/nerd_herd/src/nerd_herd/signals/s4_queue_tokens.py`:

```python
def s4_queue_tokens(
    matrix: RateLimitMatrix, *, queue: QueueProfile,
    fleet_remaining: dict[str, int] | None = None,
) -> float:
    projected = queue.projected_tokens
    if projected <= 0:
        return 0.0
    worst = 0.0
    # Cycle axes only: a per-minute token window paces (refills ~60s), it does
    # not conserve. Denominator is the FLEET's cycle-remaining on this axis
    # (passed in / built by pressure_for) so a small-window model is not floored
    # by the whole queue when other capacity can absorb it. fleet_remaining=None
    # or axis-absent -> fall back to this model's own remaining (fleet-of-one /
    # bare-matrix unit tests == today's behavior).
    for name, rl in matrix.cycle_token_cells():
        if fleet_remaining is not None and name in fleet_remaining:
            remaining = fleet_remaining[name]
        else:
            remaining = max(0, (rl.remaining or 0) - rl.in_flight)
        if remaining <= 0:
            continue
        ratio = projected / remaining
        excess = max(0.0, ratio - THRESHOLD)
        if excess <= 0:
            continue
        pressure = -min(1.0, excess * SLOPE)
        if pressure < worst:
            worst = pressure
    return worst
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `python -m pytest packages/nerd_herd/tests/signals/test_s4.py -v -o addopts="" -p no:aiohttp`
Expected: all tests PASS (3 new + 6 existing — existing pass via the per-model fallback).

- [ ] **Step 5: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/signals/s4_queue_tokens.py packages/nerd_herd/tests/signals/test_s4.py
git commit -m "feat(nerd_herd): S4 fleet-capacity denominator (Residual 2)"
```

---

### Task 2: S5 fleet-capacity denominator (same shape)

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/signals/s5_queue_calls.py`
- Test: `packages/nerd_herd/tests/signals/test_s5.py`

- [ ] **Step 1: Write the failing tests**

Append to `packages/nerd_herd/tests/signals/test_s5.py`:

```python
# ── Fleet-capacity denominator (2026-06-21) ───────────────────────────────────

def test_s5_fleet_denominator_dilutes_small_window():
    # The leak case: free model rpd=20, but fleet rpd = 1020 (free20 + premium1000).
    # A 40-call queue is 4% of the fleet -> no conservation -> free stays serviceable.
    m = _matrix(rpd=RateLimit(limit=20, remaining=20))
    qp = QueueProfile(projected_calls=40)
    assert s5_queue_calls(m, queue=qp, fleet_remaining={"rpd": 1020}) == 0.0


def test_s5_per_model_fallback_when_no_fleet():
    # No fleet view -> per-model remaining (old behavior): 40/20 = 2x -> floor.
    m = _matrix(rpd=RateLimit(limit=20, remaining=20))
    qp = QueueProfile(projected_calls=40)
    assert s5_queue_calls(m, queue=qp) == pytest.approx(-1.0, abs=0.05)


def test_s5_fleet_of_one_equals_per_model():
    # Fleet-of-one (only this daily-budgeted model) -> still conserves (pp11 shape).
    m = _matrix(rpd=RateLimit(limit=20, remaining=20))
    qp = QueueProfile(projected_calls=40)
    assert s5_queue_calls(m, queue=qp, fleet_remaining={"rpd": 20}) == pytest.approx(-1.0, abs=0.05)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest packages/nerd_herd/tests/signals/test_s5.py -v -o addopts="" -p no:aiohttp`
Expected: 3 new tests FAIL with `TypeError: ... unexpected keyword argument 'fleet_remaining'`.

- [ ] **Step 3: Add the fleet_remaining param**

Replace the body of `s5_queue_calls` in `packages/nerd_herd/src/nerd_herd/signals/s5_queue_calls.py`:

```python
def s5_queue_calls(
    matrix: RateLimitMatrix, *, queue: QueueProfile,
    fleet_remaining: dict[str, int] | None = None,
) -> float:
    projected = queue.projected_calls
    if projected <= 0:
        return 0.0
    worst = 0.0
    # Cycle axes only (excludes rpm). Denominator is the FLEET's cycle-remaining
    # on this request axis (see s4_queue_tokens) — fleet_remaining=None / axis
    # absent -> per-model fallback (fleet-of-one / unit tests == old behavior).
    for name, rl in matrix.cycle_request_cells():
        if fleet_remaining is not None and name in fleet_remaining:
            remaining = fleet_remaining[name]
        else:
            remaining = max(0, (rl.remaining or 0) - rl.in_flight)
        if remaining <= 0:
            continue
        ratio = projected / remaining
        excess = max(0.0, ratio - THRESHOLD)
        if excess <= 0:
            continue
        pressure = -min(1.0, excess * SLOPE)
        if pressure < worst:
            worst = pressure
    return worst
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `python -m pytest packages/nerd_herd/tests/signals/test_s5.py -v -o addopts="" -p no:aiohttp`
Expected: all PASS (3 new + 4 existing).

- [ ] **Step 5: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/signals/s5_queue_calls.py packages/nerd_herd/tests/signals/test_s5.py
git commit -m "feat(nerd_herd): S5 fleet-capacity denominator (Residual 2)"
```

---

### Task 3: pressure_for builds + forwards fleet_remaining

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/types.py` (`SystemSnapshot.pressure_for` + new helper `_build_fleet_cycle_remaining`)
- Test: `packages/nerd_herd/tests/test_pressure_for.py`

- [ ] **Step 1: Write the failing test**

Append to `packages/nerd_herd/tests/test_pressure_for.py`:

```python
def test_pressure_for_fleet_denominator_unfloors_small_free():
    # Leak reproduction at the pressure_for level: free model rpd=20 alongside a
    # premium rpd=1000. A 40-call queue should NOT floor the free model's queue
    # bucket, because the fleet (1020) absorbs it. Pre-fix (per-model 40/20=2x)
    # this floored to ~ -1.4.
    from nerd_herd.types import (
        CloudModelState, CloudProviderState, QueueProfile,
        RateLimit, RateLimitMatrix, SystemSnapshot,
    )
    from types import SimpleNamespace
    import time

    now = time.time()
    free_m = CloudModelState(
        model_id="free/m", limits=RateLimitMatrix(
            rpd=RateLimit(limit=20, remaining=20, reset_at=int(now + 86400))),
    )
    prem_m = CloudModelState(
        model_id="prem/m", limits=RateLimitMatrix(
            rpd=RateLimit(limit=1000, remaining=1000, reset_at=int(now + 86400))),
    )
    snap = SystemSnapshot(cloud={
        "free_prov": CloudProviderState(provider="free_prov", models={"free/m": free_m}),
        "prem_prov": CloudProviderState(provider="prem_prov", models={"prem/m": prem_m}),
    })
    snap.queue_profile = QueueProfile(total_ready_count=40, projected_calls=40)
    model = SimpleNamespace(name="free/m", provider="free_prov", is_free=True,
                            is_local=False, cap_score=7.0)

    bd = snap.pressure_for(model, task_difficulty=3, est_per_task_tokens=2_000)
    assert bd.bucket_totals.get("queue", 0.0) > -0.3  # serviceable, not floored


def test_pressure_for_fleet_of_one_still_conserves():
    # Single daily-budgeted model, queue 2x its window -> fleet-of-one -> floors
    # (the pp11 invariant at the pressure_for level).
    from nerd_herd.types import (
        CloudModelState, CloudProviderState, QueueProfile,
        RateLimit, RateLimitMatrix, SystemSnapshot,
    )
    from types import SimpleNamespace
    import time

    now = time.time()
    only_m = CloudModelState(
        model_id="free/m", limits=RateLimitMatrix(
            rpd=RateLimit(limit=20, remaining=20, reset_at=int(now + 86400))),
    )
    snap = SystemSnapshot(cloud={
        "free_prov": CloudProviderState(provider="free_prov", models={"free/m": only_m}),
    })
    snap.queue_profile = QueueProfile(total_ready_count=40, projected_calls=40)
    model = SimpleNamespace(name="free/m", provider="free_prov", is_free=True,
                            is_local=False, cap_score=7.0)

    bd = snap.pressure_for(model, task_difficulty=5, est_per_task_tokens=1_000)
    assert bd.bucket_totals.get("queue", 0.0) <= -0.3  # conserves


def test_pressure_for_precomputed_matches_internal_build():
    # Passing a precomputed fleet_remaining must yield the SAME scalar as letting
    # pressure_for build it from self.cloud (the ranking perf path == internal
    # path). Thread a shared `now` so the S9 free-cloud proximity term is
    # identical across the two calls — otherwise each call reads time.time()
    # microseconds apart and the scalars differ by ~1e-9 (flaky on bare `==`).
    import pytest
    from nerd_herd.types import (
        CloudModelState, CloudProviderState, QueueProfile,
        RateLimit, RateLimitMatrix, SystemSnapshot,
    )
    from types import SimpleNamespace
    import time

    now = time.time()
    free_m = CloudModelState(model_id="free/m", limits=RateLimitMatrix(
        rpd=RateLimit(limit=20, remaining=20, reset_at=int(now + 86400))))
    prem_m = CloudModelState(model_id="prem/m", limits=RateLimitMatrix(
        rpd=RateLimit(limit=1000, remaining=1000, reset_at=int(now + 86400))))
    snap = SystemSnapshot(cloud={
        "free_prov": CloudProviderState(provider="free_prov", models={"free/m": free_m}),
        "prem_prov": CloudProviderState(provider="prem_prov", models={"prem/m": prem_m}),
    })
    snap.queue_profile = QueueProfile(total_ready_count=40, projected_calls=40)
    model = SimpleNamespace(name="free/m", provider="free_prov", is_free=True,
                            is_local=False, cap_score=7.0)

    internal = snap.pressure_for(model, task_difficulty=3, now=now).scalar
    precomputed = snap.pressure_for(
        model, task_difficulty=3, now=now, fleet_remaining={"rpd": 1020}).scalar
    assert internal == pytest.approx(precomputed, abs=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest packages/nerd_herd/tests/test_pressure_for.py -v -o addopts="" -p no:aiohttp`
Expected: FAIL — `test_pressure_for_fleet_denominator_unfloors_small_free` floors (~ -1.4, < -0.3) because S5 still uses per-model; the precomputed test fails with `unexpected keyword argument 'fleet_remaining'`.

- [ ] **Step 3: Add helper + wire into pressure_for**

In `packages/nerd_herd/src/nerd_herd/types.py`, add a method to `SystemSnapshot` (place it just above `pressure_for`):

```python
    def _build_fleet_cycle_remaining(self) -> dict[str, int]:
        """Σ over ALL cloud models of max(0, remaining − per-model in-flight) per
        CYCLE axis (rpd/tpd/…; per-minute excluded). The denominator S4/S5 divide
        the whole-queue projection by — so a small-window model conserves only
        when the FLEET would be exhausted (no escape hatch), not when the whole
        queue merely dwarfs its own window. In-flight is per-MODEL here (cycle
        budgets are per-model), distinct from the per-PROVIDER rpm/tpm
        subtraction in pressure_for (those free-tier minute limits are shared)."""
        inflight_calls: dict[str, int] = {}
        inflight_tokens: dict[str, int] = {}
        for c in self.in_flight_calls:
            if getattr(c, "is_local", False):
                continue
            m = getattr(c, "model", "") or ""
            inflight_calls[m] = inflight_calls.get(m, 0) + 1
            inflight_tokens[m] = inflight_tokens.get(m, 0) + int(getattr(c, "est_tokens", 0) or 0)
        fleet: dict[str, int] = {}
        for ps in self.cloud.values():
            for mname, ms in ps.models.items():
                for axis, rl in ms.limits.cycle_request_cells():
                    rem = max(0, (rl.remaining or 0) - inflight_calls.get(mname, 0))
                    fleet[axis] = fleet.get(axis, 0) + rem
                for axis, rl in ms.limits.cycle_token_cells():
                    rem = max(0, (rl.remaining or 0) - inflight_tokens.get(mname, 0))
                    fleet[axis] = fleet.get(axis, 0) + rem
        return fleet
```

Add the parameter to `pressure_for`'s signature (after `eligible_models`):

```python
        eligible_models: list | None = None,
        fleet_remaining: dict[str, int] | None = None,
    ):
```

Inside `pressure_for`, just before the `sig = { ... }` dict is built, add:

```python
        # Fleet cycle-remaining for S4/S5 denominator. Ranking passes a
        # precomputed map (perf); the pressure-only path (pp11/pp13 call this
        # directly) builds it here from self.cloud.
        if fleet_remaining is None:
            fleet_remaining = self._build_fleet_cycle_remaining()
```

Then change the S4 and S5 entries in the `sig` dict from:

```python
            "S4": s4_queue_tokens(matrix, queue=self.queue_profile or QueueProfile()),
            "S5": s5_queue_calls(matrix, queue=self.queue_profile or QueueProfile()),
```

to:

```python
            "S4": s4_queue_tokens(matrix, queue=self.queue_profile or QueueProfile(),
                                  fleet_remaining=fleet_remaining),
            "S5": s5_queue_calls(matrix, queue=self.queue_profile or QueueProfile(),
                                 fleet_remaining=fleet_remaining),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest packages/nerd_herd/tests/test_pressure_for.py -v -o addopts="" -p no:aiohttp`
Expected: all PASS — leak test serviceable (> -0.3), fleet-of-one conserves (<= -0.3), precomputed == internal.

- [ ] **Step 5: Run the full nerd_herd signal suite (no regressions)**

Run: `python -m pytest packages/nerd_herd/tests/ -q -o addopts="" -p no:aiohttp`
Expected: all PASS (existing pressure/signal tests unaffected — single-model or no-cloud snapshots fall back to per-model).

- [ ] **Step 6: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/types.py packages/nerd_herd/tests/test_pressure_for.py
git commit -m "feat(nerd_herd): pressure_for builds+forwards fleet_remaining to S4/S5"
```

---

### Task 4: Ranking precomputes fleet_remaining (perf path)

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/ranking.py` (`_apply_utilization_layer`)
- Test: `packages/fatih_hoca/tests/test_ranking_fleet_remaining.py` (create)

This is behavior-identical to Task 3 (same value); it just avoids rebuilding the fleet map inside every `pressure_for` call (O(models) once vs O(models²) per pick).

- [ ] **Step 1: Write the failing test**

Create `packages/fatih_hoca/tests/test_ranking_fleet_remaining.py`:

```python
"""Ranking precomputes fleet_remaining and passes it to pressure_for; the result
must equal the snapshot's own internal build (perf path == correctness path)."""
import time
from types import SimpleNamespace

from nerd_herd.types import (
    CloudModelState, CloudProviderState, QueueProfile,
    RateLimit, RateLimitMatrix, SystemSnapshot,
)


def _snap():
    now = time.time()
    free_m = CloudModelState(model_id="free/m", limits=RateLimitMatrix(
        rpd=RateLimit(limit=20, remaining=20, reset_at=int(now + 86400))))
    prem_m = CloudModelState(model_id="prem/m", limits=RateLimitMatrix(
        rpd=RateLimit(limit=1000, remaining=1000, reset_at=int(now + 86400))))
    snap = SystemSnapshot(cloud={
        "free_prov": CloudProviderState(provider="free_prov", models={"free/m": free_m}),
        "prem_prov": CloudProviderState(provider="prem_prov", models={"prem/m": prem_m}),
    })
    snap.queue_profile = QueueProfile(total_ready_count=40, projected_calls=40)
    return snap


def test_ranking_passes_fleet_remaining_that_unfloors_small_free():
    # A captured pressure_for must receive a non-None fleet_remaining whose value
    # matches the snapshot's internal build, and the small free model's queue
    # bucket must be serviceable (not floored).
    from fatih_hoca import ranking

    snap = _snap()
    expected_fleet = snap._build_fleet_cycle_remaining()
    captured = {}
    orig = SystemSnapshot.pressure_for

    def spy(self, model, **kw):
        captured[getattr(model, "name", "")] = kw.get("fleet_remaining")
        return orig(self, model, **kw)

    free_model = SimpleNamespace(
        name="free/m", provider="free_prov", is_free=True, is_local=False,
        cap_score=7.0, agent_type="researcher", context={},
        estimated_cost=lambda *_: 0.0,
    )
    scored = [ranking.ScoredModel(model=free_model, score=1.0, composite_score=1.0)]

    SystemSnapshot.pressure_for = spy
    try:
        ranking._apply_utilization_layer(
            scored, snap, task_difficulty=3, reqs=free_model, now=time.time(),
            burn_log=None,
        )
    finally:
        SystemSnapshot.pressure_for = orig

    assert captured.get("free/m") == expected_fleet
```

Signatures verified against source (`ranking.py:58-83`, `ranking.py:140-148`):
`ScoredModel(model, score, capability_score=0.0, composite_score=0.0, ...)` and
`_apply_utilization_layer(scored, snapshot, task_difficulty, reqs=None, *, now=None, burn_log=None)`.
`reqs` is the `estimate_for` proxy — it reads `.agent_type` and `.context`, which
the `free_model` stub provides. No further adjustment needed.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest packages/fatih_hoca/tests/test_ranking_fleet_remaining.py -v -o addopts="" -p no:aiohttp`
Expected: FAIL — `captured["free/m"]` is `None` (ranking does not yet pass fleet_remaining).

- [ ] **Step 3: Build + pass fleet_remaining in ranking**

In `packages/fatih_hoca/src/fatih_hoca/ranking.py`, immediately AFTER the `fleet_consumed` build block (ends at `ranking.py:192`, `fleet_consumed[_prov] = _consumed`), add:

```python
    # Fleet cycle-remaining for the S4/S5 denominator. Built once here (perf:
    # O(models)) and passed into every pressure_for so it does not rebuild the
    # fleet map per candidate (O(models²)). Identical value to pressure_for's
    # own internal build — see SystemSnapshot._build_fleet_cycle_remaining.
    fleet_remaining = snapshot._build_fleet_cycle_remaining()
```

Then add `fleet_remaining=fleet_remaining,` to the `pressure_for(...)` call (alongside `fleet_consumed=fleet_consumed,` at `ranking.py:243`):

```python
            fleet_consumed=fleet_consumed,
            fleet_remaining=fleet_remaining,
            now=now,
            burn_log=burn_log,
            eligible_models=eligible_models,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest packages/fatih_hoca/tests/test_ranking_fleet_remaining.py -v -o addopts="" -p no:aiohttp`
Expected: PASS — captured fleet_remaining == internal build.

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/ranking.py packages/fatih_hoca/tests/test_ranking_fleet_remaining.py
git commit -m "perf(fatih_hoca): ranking precomputes fleet_remaining for S4/S5"
```

---

### Task 5: pp13 sim anchor (leak — fails when reverted)

**Files:**
- Modify: `packages/fatih_hoca/tests/sim/scenarios.py` (add `assert_pp13_aggregate_vs_single_leak` + dispatch entries)

- [ ] **Step 1: Add the assertion function**

In `packages/fatih_hoca/tests/sim/scenarios.py`, immediately after the END of the `assert_pp12_btable_demand_reduces_floor` function (~line 1573, just before the `POOL_PRESSURE_SCENARIOS` list at ~1578 — anchor on the function's end, do NOT land inside the registry block), add:

```python
# ── Scenario PP13: aggregate-queue ÷ single-model leak (Residual 2) ──────────

def assert_pp13_aggregate_vs_single_leak() -> list[str]:
    """The over-conservation leak the fleet-capacity denominator removes: a small
    daily-window free model (rpd=20) sitting beside an abundant premium (rpd=1000)
    must NOT be conserve-floored by the WHOLE queue's projected demand — only a
    fraction routes to it; the fleet (1020) absorbs the rest. Pre-fix (per-model
    denominator: 40/20 = 2x) this floored the free model and leaked the easy task
    to paid premium = waste. Reverting S4/S5 to the per-model denominator makes
    this assertion FAIL (the free model floors again).
    """
    from nerd_herd.types import (
        CloudModelState, CloudProviderState, QueueProfile,
        RateLimit, RateLimitMatrix, SystemSnapshot,
    )
    from types import SimpleNamespace

    now = _time.time()
    free_m = CloudModelState(model_id="free/m", limits=RateLimitMatrix(
        rpd=RateLimit(limit=20, remaining=20, reset_at=int(now + 86400))))
    prem_m = CloudModelState(model_id="prem/m", limits=RateLimitMatrix(
        rpd=RateLimit(limit=1000, remaining=1000, reset_at=int(now + 86400))))
    snap = SystemSnapshot(cloud={
        "free_prov": CloudProviderState(provider="free_prov", models={"free/m": free_m}),
        "prem_prov": CloudProviderState(provider="prem_prov", models={"prem/m": prem_m}),
    })
    snap.queue_profile = QueueProfile(total_ready_count=40, projected_calls=40)
    model = SimpleNamespace(name="free/m", provider="free_prov", is_free=True,
                            is_local=False, cap_score=7.0)
    bd = snap.pressure_for(model, task_difficulty=3, est_per_task_tokens=2_000)
    queue = bd.bucket_totals.get("queue", 0.0)
    failures = []
    if not (queue > -0.3):
        failures.append(
            f"pp13: small free model beside abundant premium must stay serviceable "
            f"(queue bucket ~0), got queue={queue:.3f} — aggregate-÷-single leak"
        )
    return failures
```

- [ ] **Step 2: Register pp13 in the scenario + dispatch tables**

In `POOL_PRESSURE_SCENARIOS` (~line 1589), add after the pp12 entry:

```python
    ("pp13_aggregate_vs_single_leak", pp1_fat_vs_tiny),  # pressure-only
```

In the assertion dispatch dict (~line 1619), add after the pp12 entry:

```python
    "pp13_aggregate_vs_single_leak": lambda sc: assert_pp13_aggregate_vs_single_leak(),
```

(Match the exact surrounding syntax — pp11/pp12 use `pp1_fat_vs_tiny` as a throwaway scenario factory because the assertion builds its own snapshot. Mirror them exactly.)

- [ ] **Step 3: Run the scenario runner — pp13 PASSES, pp11 still green**

Run: `python packages/fatih_hoca/tests/sim/run_scenarios.py`
Expected: pp1–pp13 PASS; pp11 (anchor) green; pp13 green. Report any pp/rp delta (expect NONE — S4/S5 dormant in full-flow).

- [ ] **Step 4: Prove pp13 fails when reverted (fails-when-reverted check)**

Temporarily revert ONLY S5's denominator: in `s5_queue_calls.py` Step 3 body, change the denominator line back to always per-model:

```python
        remaining = max(0, (rl.remaining or 0) - rl.in_flight)  # TEMP revert
```

Run: `python packages/fatih_hoca/tests/sim/run_scenarios.py`
Expected: pp13 FAILS (free model floors, `queue ~ -1.4`). This proves the anchor is real. **Restore the fleet-denominator line immediately after** and re-run — pp13 green again.

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/tests/sim/scenarios.py
git commit -m "test(fatih_hoca): pp13 aggregate-vs-single leak anchor (Residual 2)"
```

---

### Task 6: Full sim gate + swap storm + cross-package suite

**Files:** none (verification only)

- [ ] **Step 1: Run the full scenario gate**

Run: `python packages/fatih_hoca/tests/sim/run_scenarios.py`
Expected: pp1–pp13 PASS, rp1–rp5 unchanged (no delta — document the output). pp11 + pp12 green.

- [ ] **Step 2: Run the swap-storm check**

Run: `python packages/fatih_hoca/tests/sim/run_swap_storm_check.py`
Expected: swap rate ≤ 0.5%.

- [ ] **Step 3: Run the nerd_herd + fatih_hoca suites**

Run: `python -m pytest packages/nerd_herd/ packages/fatih_hoca/ -q -o addopts="" -p no:aiohttp`
Expected: all PASS. If any pre-existing failure appears, confirm it fails on `main` too (not introduced here) before proceeding.

- [ ] **Step 4: Import smoke**

Run: `python -c "import nerd_herd, fatih_hoca; from nerd_herd.signals.s4_queue_tokens import s4_queue_tokens; from nerd_herd.signals.s5_queue_calls import s5_queue_calls; print('ok')"`
Expected: `ok`.

- [ ] **Step 5: Request code review**

Use `superpowers:requesting-code-review` against the full diff (S4/S5 + pressure_for + ranking + pp13). Confirm: (a) the `48e4cee8` invariant intact (S4/S5 still QUEUE_BUCKET, no supply-veto path), (b) fleet sum's per-model in-flight is correct, (c) no scenario silently retuned.

---

## Deploy / handoff notes

- **Restart-gated:** the signal/combine layer loads at process start. After merge, the user must `/restart` (under **minimal** GPU mode to expose cloud-only) to live-verify.
- **Backlog:** `origin/main` is ~10 commits behind (kdv, FC-gate, phantom chain, registry-decouple, btable-wiring). `/restart` + verify + push the backlog before/with this work.
- **Merge mechanic:** worktree + 3-way merge (concurrent sessions cross `main`). NEVER `run_in_background` pytest on Windows (orphans hold the prod SQLite lock).
- **Memory:** on completion, record under `project_phantom_veto_*` lineage (link `[[project_phantom_veto_architecture_20260617]]`).
