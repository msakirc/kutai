# S7/S6 Continuity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** De-blind the S7 burn-rate signal and reactivate the dead S6 capable-supply signal by (a) teaching the Phase 2d simulator to exercise both, then (b) replacing their `0.70` dead-bands with continuous smoothstep ramps — all sim-validated and tuned, prod behavior unchanged until restart.

**Architecture:** Signal-layer only. Three new optional `pressure_for` kwargs (`now`, `burn_log`, `eligible_models`) default to current prod behavior; the ranking layer and the sim thread real values in. S7/S6 swap their gated `excess = max(0, ratio−0.70)` for a shared `_smoothstep(min(1, ratio/SAT))` ramp from a new `signals/_curves.py`. The simulator gains a consistent virtual clock (`now = wall_anchor + virtual_clock`) so the burn-rate window evicts correctly, plus per-pick `burn_log` recording and a capability-demand queue so S6 has supply/demand to compute against.

**Tech Stack:** Python 3.10, pytest (with timeouts per CLAUDE.md), the in-memory Phase 2d simulator (`packages/fatih_hoca/tests/sim/`), nerd_herd signal modules (`packages/nerd_herd/src/nerd_herd/signals/`).

---

## File Structure

**Modify:**
- `packages/nerd_herd/src/nerd_herd/types.py` — `pressure_for` gains `now` / `burn_log` / `eligible_models` kwargs (default-preserving), threads them into S7/S9/S6.
- `packages/fatih_hoca/src/fatih_hoca/ranking.py` — `rank_candidates` + `_apply_utilization_layer` gain `now` / `burn_log`; `_apply_utilization_layer` builds the `eligible_models` rollup and threads all three into `pressure_for`.
- `packages/fatih_hoca/src/fatih_hoca/selector.py` — `select_for_simulation` gains `now` / `burn_log`, threads to `rank_candidates`; `_SimPickResult` gains `provider`; cloud stubs carry real `capabilities` (set) + `rpd_remaining`.
- `packages/fatih_hoca/tests/sim/scenarios.py` — `_build_snapshot_factory` reset_at → absolute; `_build_select_fn` owns a shared `wall_anchor` + per-run `BurnLog`, passes `now`, records burn per pick, builds `by_capability`; new scenarios + assertions.
- `packages/fatih_hoca/tests/sim/run_scenarios.py` — register + print new scenarios.
- `packages/nerd_herd/src/nerd_herd/signals/s7_burn_rate.py` — smoothstep ramp.
- `packages/nerd_herd/src/nerd_herd/signals/s6_capable_supply.py` — smoothstep ramp.
- `packages/nerd_herd/src/nerd_herd/signals/s12_pool_balance.py` — import `_smoothstep` from `_curves`.

**Create:**
- `packages/nerd_herd/src/nerd_herd/signals/_curves.py` — shared `smoothstep`.
- `packages/nerd_herd/tests/signals/test_curves.py`
- `packages/nerd_herd/tests/signals/test_s7_continuity.py`
- `packages/nerd_herd/tests/signals/test_s6_continuity.py`
- `packages/nerd_herd/tests/test_pressure_for_threading.py`

**Conventions (read before starting):**
- Python exe: `/c/Users/sakir/Dropbox/Workspaces/kutay/.venv/Scripts/python.exe` (call it `PY` below).
- Always run pytest with a timeout: `timeout 60 $PY -m pytest <path> -v`.
- nerd_herd tests run from `packages/nerd_herd/` (its own `pyproject`); the signal imports are `from nerd_herd...`. Run: `cd packages/nerd_herd && timeout 60 $PY -m pytest tests/ -q`.
- The sim harness is run as a script: `$PY packages/fatih_hoca/tests/sim/run_scenarios.py`.
- Use `rtk git ...` for git (CLAUDE.md golden rule).
- Commit message footer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## Part A — Prereq: make the sim exercise S7 + S6

### Task 1: Shared smoothstep curve (extract, no behavior change)

**Files:**
- Create: `packages/nerd_herd/src/nerd_herd/signals/_curves.py`
- Create: `packages/nerd_herd/tests/signals/test_curves.py`
- Modify: `packages/nerd_herd/src/nerd_herd/signals/s12_pool_balance.py`

- [ ] **Step 1: Write the failing test**

Create `packages/nerd_herd/tests/signals/test_curves.py`:
```python
from nerd_herd.signals._curves import smoothstep


def test_smoothstep_endpoints():
    assert smoothstep(-1.0) == 0.0
    assert smoothstep(0.0) == 0.0
    assert smoothstep(1.0) == 1.0
    assert smoothstep(2.0) == 1.0


def test_smoothstep_midpoint_and_shape():
    # Hermite 3x^2 - 2x^3: midpoint 0.5, quiet near 0, steep in the middle.
    assert smoothstep(0.5) == 0.5
    assert smoothstep(0.1) < 0.05          # near-zero stays quiet
    assert smoothstep(0.9) > 0.95          # saturates near 1
    # Strictly monotonic on (0, 1)
    xs = [i / 20 for i in range(21)]
    ys = [smoothstep(x) for x in xs]
    assert all(b > a for a, b in zip(ys, ys[1:]))


def test_s12_still_uses_shared_curve():
    # S12's private _smoothstep is now the shared one (no drift).
    from nerd_herd.signals import s12_pool_balance as s12
    assert s12._smoothstep(0.5) == 0.5
    assert s12._smoothstep(0.1) == smoothstep(0.1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/nerd_herd && timeout 60 $PY -m pytest tests/signals/test_curves.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nerd_herd.signals._curves'`.

- [ ] **Step 3: Create the shared curve**

Create `packages/nerd_herd/src/nerd_herd/signals/_curves.py`:
```python
"""Shared continuous shaping curves for signals (no gates, no kinks).

Single source of truth so S7 / S6 / S12 cannot drift apart.
"""
from __future__ import annotations


def smoothstep(x: float) -> float:
    """Hermite 3x^2 - 2x^3 clamped to [0, 1]. Zero slope at both ends."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x * x * (3.0 - 2.0 * x)
```

- [ ] **Step 4: Point S12 at the shared curve**

In `packages/nerd_herd/src/nerd_herd/signals/s12_pool_balance.py`, replace the local
`_smoothstep` definition:
```python
def _smoothstep(x: float) -> float:
    """Continuous 0→1 ramp with zero slope at both ends (no kink)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x * x * (3.0 - 2.0 * x)
```
with an import-backed alias (keep the `_smoothstep` name so the rest of the module + its tests
are untouched). Add near the top imports:
```python
from nerd_herd.signals._curves import smoothstep as _smoothstep
```
and delete the old local `def _smoothstep` block.

- [ ] **Step 5: Run curve + S12 tests**

Run: `cd packages/nerd_herd && timeout 60 $PY -m pytest tests/signals/test_curves.py tests/signals/test_s12.py -v`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```bash
rtk git add packages/nerd_herd/src/nerd_herd/signals/_curves.py packages/nerd_herd/src/nerd_herd/signals/s12_pool_balance.py packages/nerd_herd/tests/signals/test_curves.py
rtk git commit -m "refactor(nerd-herd): extract shared smoothstep to signals/_curves

Single source of truth so S7/S6/S12 ramps cannot drift. S12 behavior
unchanged (aliases the shared fn).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `pressure_for` gains `now` / `burn_log` / `eligible_models` (default-preserving)

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/types.py:257-385`
- Create: `packages/nerd_herd/tests/test_pressure_for_threading.py`

- [ ] **Step 1: Write the failing test**

Create `packages/nerd_herd/tests/test_pressure_for_threading.py`:
```python
import time
from types import SimpleNamespace

from nerd_herd.types import (
    SystemSnapshot, CloudProviderState, CloudModelState,
    RateLimit, RateLimitMatrix, QueueProfile,
)
from nerd_herd.burn_log import BurnLog


def _free_snapshot(remaining=18, limit=20, reset_in=3600):
    now = time.time()
    rpd = RateLimit(limit=limit, remaining=remaining, reset_at=int(now + reset_in))
    cms = CloudModelState(model_id="gem/flash", utilization_pct=0.0,
                          limits=RateLimitMatrix(rpd=rpd))
    prov = CloudProviderState(provider="gem", models={"gem/flash": cms})
    return SystemSnapshot(cloud={"gem": prov}), now


def _free_model():
    return SimpleNamespace(name="gem/flash", provider="gem", is_local=False,
                           is_free=True, cap_score=7.0, capabilities=set(),
                           rpd_remaining=18)


def test_now_kwarg_defaults_to_walltime_and_is_threaded():
    snap, now = _free_snapshot()
    m = _free_model()
    # Passing an explicit now equal to wall time reproduces the default path.
    bd_default = snap.pressure_for(m, task_difficulty=5)
    bd_now = snap.pressure_for(m, task_difficulty=5, now=now)
    assert abs(bd_default.signals["S9"] - bd_now.signals["S9"]) < 0.05


def test_burn_log_kwarg_drives_s7():
    snap, now = _free_snapshot(remaining=18, limit=20, reset_in=3600)
    m = _free_model()
    # No burn → S7 == 0.
    assert snap.pressure_for(m, task_difficulty=5, now=now).signals["S7"] == 0.0
    # Heavy recent burn on a tiny tank → S7 fires negative.
    bl = BurnLog(window_secs=300.0)
    for i in range(20):
        bl.record(provider="gem", model="gem/flash", tokens=1000, calls=1, now=now - i)
    s7 = snap.pressure_for(m, task_difficulty=5, now=now, burn_log=bl).signals["S7"]
    assert s7 < 0.0


def test_eligible_models_kwarg_drives_s6():
    snap, now = _free_snapshot()
    m = SimpleNamespace(name="gem/flash", provider="gem", is_local=False,
                        is_free=True, cap_score=7.0, capabilities={"vision"},
                        rpd_remaining=2)
    snap.queue_profile = QueueProfile(by_capability={"vision": 50},
                                      total_ready_count=50)
    # Empty eligible list (prod default) → S6 == 0.
    assert snap.pressure_for(m, task_difficulty=5, now=now).signals["S6"] == 0.0
    # Fed a thin capable supply vs heavy demand → S6 fires negative.
    s6 = snap.pressure_for(m, task_difficulty=5, now=now,
                           eligible_models=[m]).signals["S6"]
    assert s6 < 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/nerd_herd && timeout 60 $PY -m pytest tests/test_pressure_for_threading.py -v`
Expected: FAIL — `test_burn_log_kwarg_drives_s7` and `test_eligible_models_kwarg_drives_s6` fail
(`pressure_for` has no `burn_log` / `eligible_models` params → TypeError).

- [ ] **Step 3: Add the kwargs and thread them**

In `packages/nerd_herd/src/nerd_herd/types.py`, change the `pressure_for` signature (currently
ends at `fleet_consumed: dict | None = None,`) to add three params:
```python
    def pressure_for(
        self,
        model,
        *,
        task_difficulty: int = 5,
        est_per_call_tokens: int = 0,
        est_per_task_tokens: int = 0,
        est_iterations: int = 1,
        est_call_cost: float = 0.0,
        cap_needed: float = 5.0,
        consecutive_failures: int = 0,
        fleet_consumed: dict | None = None,
        now: float | None = None,
        burn_log=None,
        eligible_models: list | None = None,
    ):
```
Replace the body's clock acquisition (currently `import time as _time; now = _time.time()`) with:
```python
        import time as _time
        now = now if now is not None else _time.time()
```
Replace the burn_log source in the S7 line. Current:
```python
            "S7": s7_burn_rate(matrix, provider=provider, model=getattr(model, "name", ""),
                               burn_log=get_burn_log(), now=now),
```
becomes:
```python
            "S7": s7_burn_rate(matrix, provider=provider, model=getattr(model, "name", ""),
                               burn_log=(burn_log if burn_log is not None else get_burn_log()),
                               now=now),
```
Replace the S6 line. Current:
```python
            "S6": s6_capable_supply(model, queue=self.queue_profile or QueueProfile(),
                                    eligible_models=[], iter_avg=float(est_iterations or 8)),
```
becomes:
```python
            "S6": s6_capable_supply(model, queue=self.queue_profile or QueueProfile(),
                                    eligible_models=eligible_models or [],
                                    iter_avg=float(est_iterations or 8)),
```
(S9 already receives `now=now` — no change there.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/nerd_herd && timeout 60 $PY -m pytest tests/test_pressure_for_threading.py -v`
Expected: PASS (3).

- [ ] **Step 5: Regression — full nerd_herd suite (defaults must be unchanged)**

Run: `cd packages/nerd_herd && timeout 120 $PY -m pytest tests/ -q`
Expected: PASS (all prior tests green — the new kwargs default to old behavior).

- [ ] **Step 6: Commit**

```bash
rtk git add packages/nerd_herd/src/nerd_herd/types.py packages/nerd_herd/tests/test_pressure_for_threading.py
rtk git commit -m "feat(nerd-herd): pressure_for accepts now/burn_log/eligible_models

Default-preserving seams so the ranking layer + Phase 2d sim can drive
S7 (rolling burn) and S6 (capable supply). None defaults reproduce the
exact current prod path.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Ranking threads `now`/`burn_log` and builds the `eligible_models` rollup

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/ranking.py:117-216` (`_apply_utilization_layer`), `:220-246` + `:686-691` (`rank_candidates`)
- Test: covered end-to-end by the sim in Tasks 5–6; add a focused unit test here.
- Create: `packages/fatih_hoca/tests/test_ranking_s6_rollup.py`

- [ ] **Step 1: Write the failing test**

Create `packages/fatih_hoca/tests/test_ranking_s6_rollup.py`:
```python
"""_apply_utilization_layer must build a capable-supply rollup and thread
now/burn_log into pressure_for. We capture the kwargs pressure_for receives."""
from types import SimpleNamespace

from fatih_hoca import ranking
from fatih_hoca.ranking import ScoredModel


def test_eligible_models_and_now_threaded(monkeypatch):
    captured = {}

    def fake_pressure_for(model, **kwargs):
        captured.setdefault("calls", []).append((getattr(model, "name", "?"), kwargs))
        return SimpleNamespace(scalar=0.0, signals={}, modifiers={},
                               bucket_totals={}, positive_total=0.0, negative_total=0.0)

    snap = SimpleNamespace(
        cloud={}, queue_profile=None, local=SimpleNamespace(model_name=None),
        pressure_for=fake_pressure_for,
    )
    m1 = SimpleNamespace(name="a/m", provider="a", is_local=False, is_free=True,
                         capabilities={"vision"}, is_loaded=False)
    m2 = SimpleNamespace(name="b/m", provider="b", is_local=False, is_free=True,
                         capabilities=set(), is_loaded=False)
    scored = [ScoredModel(model=m1, score=10.0), ScoredModel(model=m2, score=9.0)]

    ranking._apply_utilization_layer(scored, snap, task_difficulty=5, reqs=None,
                                     now=12345.0, burn_log="BL")

    first = captured["calls"][0][1]
    assert first["now"] == 12345.0
    assert first["burn_log"] == "BL"
    # eligible_models must contain both candidates (capable-supply rollup).
    names = {getattr(m, "name", "?") for m in first["eligible_models"]}
    assert names == {"a/m", "b/m"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/fatih_hoca && timeout 60 $PY -m pytest tests/test_ranking_s6_rollup.py -v`
Expected: FAIL — `_apply_utilization_layer` has no `now` / `burn_log` params (TypeError).

- [ ] **Step 3: Thread now/burn_log + build eligible_models**

In `packages/fatih_hoca/src/fatih_hoca/ranking.py`, change `_apply_utilization_layer`'s signature:
```python
def _apply_utilization_layer(
    scored: list[ScoredModel],
    snapshot: SystemSnapshot,
    task_difficulty: int,
    reqs: "ModelRequirements | None" = None,
    *,
    now: float | None = None,
    burn_log=None,
) -> None:
```
Immediately after the `fleet_consumed` rollup block (ends at the `if _has_capacity:
fleet_consumed[_prov] = _consumed` lines), add the capable-supply rollup:
```python
    # Capable-supply rollup for S6: the set of candidates with their remaining
    # daily capacity, so a capability-shortage produces conserve-pressure. Built
    # once here; rpd_remaining is read from the snapshot (authoritative) and
    # attached so s6._supply_for can sum real capacity in prod AND sim.
    eligible_models: list = []
    for _sm in scored:
        _mdl = _sm.model
        _prov_name = getattr(_mdl, "provider", "")
        _ps = snapshot.cloud.get(_prov_name)
        _ms = _ps.models.get(getattr(_mdl, "name", "")) if _ps else None
        _rpd_rem = 0
        if _ms is not None and _ms.limits.rpd is not None:
            _rpd_rem = _ms.limits.rpd.remaining or 0
        eligible_models.append(SimpleNamespace(
            name=getattr(_mdl, "name", ""),
            capabilities=getattr(_mdl, "capabilities", set()),
            rpd_remaining=_rpd_rem,
        ))
```
Add `from types import SimpleNamespace` to the file's imports if not present (it is not —
add it near the top with the other stdlib imports).

In the `for sm in scored:` loop, extend the `snapshot.pressure_for(...)` call to pass the three:
```python
        breakdown = snapshot.pressure_for(
            sm.model,
            task_difficulty=task_difficulty,
            est_per_call_tokens=estimates.per_call_tokens,
            est_per_task_tokens=estimates.total_tokens,
            est_iterations=estimates.iterations,
            est_call_cost=getattr(sm.model, "estimated_cost",
                                  lambda *_: 0.0)(estimates.in_tokens, estimates.out_tokens),
            cap_needed=CAP_NEEDED_BY_DIFFICULTY.get(task_difficulty, 5.0),
            fleet_consumed=fleet_consumed,
            now=now,
            burn_log=burn_log,
            eligible_models=eligible_models,
        )
```

- [ ] **Step 4: Thread now/burn_log through `rank_candidates`**

Change `rank_candidates` signature (add two kwargs after `remaining_budget`):
```python
def rank_candidates(
    candidates: list[ModelInfo],
    reqs: ModelRequirements,
    snapshot: SystemSnapshot,
    failures: list[Failure],
    remaining_budget: float = 0.0,
    *,
    now: float | None = None,
    burn_log=None,
) -> list[ScoredModel]:
```
Update the `_apply_utilization_layer(...)` call near the bottom (currently passes
`scored, snapshot, task_difficulty=reqs.difficulty, reqs=reqs`):
```python
    _apply_utilization_layer(
        scored,
        snapshot,
        task_difficulty=reqs.difficulty,
        reqs=reqs,
        now=now,
        burn_log=burn_log,
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd packages/fatih_hoca && timeout 60 $PY -m pytest tests/test_ranking_s6_rollup.py -v`
Expected: PASS.

- [ ] **Step 6: Regression — fatih_hoca ranking tests**

Run: `cd packages/fatih_hoca && timeout 120 $PY -m pytest tests/test_ranking.py -q`
Expected: PASS (defaults preserve behavior).

- [ ] **Step 7: Commit**

```bash
rtk git add packages/fatih_hoca/src/fatih_hoca/ranking.py packages/fatih_hoca/tests/test_ranking_s6_rollup.py
rtk git commit -m "feat(fatih-hoca): ranking builds S6 capable-supply rollup + threads now/burn_log

_apply_utilization_layer now assembles {capabilities, rpd_remaining} per
candidate from the snapshot and threads it (plus now/burn_log) into
pressure_for, reviving the dead S6 wiring. Defaults unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Sim selector threads `now`/`burn_log`; pick carries provider; stubs carry caps + rpd

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/selector.py` (`select_for_simulation` + `_SimPickResult` + cloud stubs)

- [ ] **Step 1: Locate `_SimPickResult`**

Run: `cd packages/fatih_hoca && rtk grep "_SimPickResult" src/fatih_hoca/selector.py`
Read its dataclass definition (a few lines above `select_for_simulation`).

- [ ] **Step 2: Add `provider` to `_SimPickResult`**

Add a `provider: str = ""` field to the `_SimPickResult` dataclass.

- [ ] **Step 3: Thread now/burn_log + enrich stubs + return provider**

In `select_for_simulation`, change the signature to add two kwargs:
```python
def select_for_simulation(
    *,
    task_name: str,
    difficulty: int,
    estimated_output_tokens: int,
    snapshot: Any,
    providers_cfg: dict,
    queue_profile: Any = None,
    now: float | None = None,
    burn_log=None,
) -> "_SimPickResult":
```
In the cloud-stub construction loop, replace `capabilities=SimpleNamespace(),` with a real set
sourced from the provider config (default empty), and add `rpd_remaining` from the snapshot:
```python
            _caps = set(model_cfg.get("capabilities", []))
            _ms = None
            _ps = snapshot.cloud.get(provider) if hasattr(snapshot, "cloud") else None
            if _ps is not None:
                _ms = _ps.models.get(model_id)
            _rpd_rem = 0
            if _ms is not None and _ms.limits.rpd is not None:
                _rpd_rem = _ms.limits.rpd.remaining or 0
            candidates.append(SimpleNamespace(
                name=model_id,
                litellm_name=model_id,
                is_local=False,
                is_loaded=False,
                is_free=is_free,
                provider=provider,
                capabilities=_caps,
                rpd_remaining=_rpd_rem,
                tokens_per_second=0.0,
                load_time_seconds=0.0,
                total_params_b=0,
                active_params_b=0,
                specialty=None,
                thinking_model=False,
                operational_dict=lambda: {"context_window": 128000},
                estimated_cost=(lambda inp, out, _free=is_free: 0.0 if _free else 0.005),
                location="cloud",
            ))
```
Also give the **local** stub `capabilities=set()` (it currently uses `SimpleNamespace()`); the
local model is never in the S6 capable pool but the rollup reads `.capabilities` defensively.
Update the `rank_candidates(...)` call to forward the clock + burn log:
```python
        scored = rank_candidates(
            candidates=candidates,
            reqs=reqs,
            snapshot=snapshot,
            failures=[],
            remaining_budget=300.0,
            now=now,
            burn_log=burn_log,
        )
```
Update the two `return _SimPickResult(...)` sites to set `provider`:
```python
    if not scored:
        return _SimPickResult(
            model_name="loaded-local", pool="local",
            cap_score_100=55.0, tokens_per_second=20.0, provider="local",
        )

    top = scored[0]
    return _SimPickResult(
        model_name=top.model.name,
        pool=top.pool or "local",
        cap_score_100=top.capability_score * 10.0,
        tokens_per_second=top.model.tokens_per_second or 20.0,
        provider=getattr(top.model, "provider", "") or "local",
    )
```

- [ ] **Step 4: Smoke-check the selector still imports + runs**

Run: `cd packages/fatih_hoca && timeout 60 $PY -c "from fatih_hoca.selector import select_for_simulation; print('ok')"`
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
rtk git add packages/fatih_hoca/src/fatih_hoca/selector.py
rtk git commit -m "feat(fatih-hoca): sim selector threads now/burn_log, pick carries provider, stubs carry caps+rpd

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Sim wiring — consistent virtual clock, per-pick burn, capability demand

**Files:**
- Modify: `packages/fatih_hoca/tests/sim/scenarios.py` (`_build_snapshot_factory`, `_build_select_fn`)

This task makes the sim faithfully drive S7. Key correctness point: `pressure_for` will now receive
`now = wall_anchor + virtual_clock`. So the snapshot's `reset_at` must be the **absolute** projected
reset (`wall_anchor + counter.reset_at`), NOT `wall_anchor + reset_in_secs` — otherwise `reset_in`
inside `pressure_for` double-subtracts the virtual clock and S9 breaks.

- [ ] **Step 1: Share one `wall_anchor`; make reset_at absolute**

Change `_build_snapshot_factory(scenario_providers)` to accept a shared anchor:
```python
def _build_snapshot_factory(scenario_providers: dict[str, Any], wall_anchor: float | None = None):
```
Replace its internal `wall_anchor = _time.time()` with:
```python
    if wall_anchor is None:
        wall_anchor = _time.time()
```
Replace the reset_at projection. Current:
```python
                reset_in_secs = max(0.0, counter.reset_at - state.virtual_clock)
                rpd = RateLimit(
                    limit=counter.limit,
                    remaining=counter.remaining,
                    reset_at=int(wall_anchor + reset_in_secs),
                )
```
becomes (absolute — reset projected onto the wall clock independent of the current virtual time;
`pressure_for` subtracts `now = wall_anchor + virtual_clock` itself):
```python
                rpd = RateLimit(
                    limit=counter.limit,
                    remaining=counter.remaining,
                    reset_at=int(wall_anchor + counter.reset_at),
                )
```

- [ ] **Step 2: `_build_select_fn` owns the anchor + burn log, records per pick, builds capability demand**

Replace `_build_select_fn` with the version below. Changes: creates one `wall_anchor` + one
per-run `BurnLog`; builds its snapshot_factory with that anchor; passes `now` + `burn_log` to
`select_for_simulation`; records each pick's burn at `now`; adds `by_capability` to the queue
profile via a task_name→capability map.
```python
def _build_select_fn(scenario_providers: dict[str, Any], tasks: list[SimTask] | None = None):
    """Wires through the real fatih_hoca.select() against the SimState.

    Owns a single ``wall_anchor`` and a per-run ``BurnLog`` so the burn-rate
    window (S7) and reset-proximity (S9) share one clock: ``now = wall_anchor
    + state.virtual_clock``. Each pick is recorded into the burn log so the
    NEXT tick's S7 sees a real rolling rate.
    """
    from types import SimpleNamespace
    from fatih_hoca import selector as _selector
    from fatih_hoca.requirements import QueueProfile
    from nerd_herd.burn_log import BurnLog

    wall_anchor = _time.time()
    burn_log = BurnLog(window_secs=300.0)
    snapshot_factory = _build_snapshot_factory(scenario_providers, wall_anchor=wall_anchor)

    # task_name → required capability (only names that imply a hard capability;
    # everything else implies none, so S6 stays 0 on generic workloads).
    _CAP_BY_TASK = {"visual_reviewer": "vision"}

    def select(state: SimState, task: SimTask) -> Any:
        now = wall_anchor + state.virtual_clock
        queue_profile = None
        if tasks is not None:
            remaining = tasks[task.idx:]
            total = len(remaining)
            hard = sum(1 for t in remaining if t.difficulty >= 7)
            by_difficulty: dict[int, int] = {}
            by_capability: dict[str, int] = {}
            for t in remaining:
                by_difficulty[t.difficulty] = by_difficulty.get(t.difficulty, 0) + 1
                cap = _CAP_BY_TASK.get(t.task_name)
                if cap:
                    by_capability[cap] = by_capability.get(cap, 0) + 1
            queue_profile = QueueProfile(
                total_ready_count=total,
                hard_tasks_count=hard,
                by_difficulty=by_difficulty,
                by_capability=by_capability,
            )

        picked = _selector.select_for_simulation(
            task_name=task.task_name,
            difficulty=task.difficulty,
            estimated_output_tokens=task.estimated_output_tokens,
            snapshot=snapshot_factory(state),
            providers_cfg=scenario_providers,
            queue_profile=queue_profile,
            now=now,
            burn_log=burn_log,
        )
        # Record this pick's consumption so the next tick's S7 sees it.
        if picked.pool in ("time_bucketed", "per_call") and picked.provider:
            burn_log.record(
                provider=picked.provider,
                model=picked.model_name,
                tokens=task.estimated_output_tokens,
                calls=1,
                now=now,
            )
        return SimpleNamespace(
            model_name=picked.model_name,
            pool=picked.pool,
            cap_score_100=picked.cap_score_100,
            estimated_output_tokens=task.estimated_output_tokens,
            tokens_per_second=picked.tokens_per_second,
        )

    return select
```
Note: `QueueProfile` is imported from `fatih_hoca.requirements` (matches the existing import in
the original `_build_select_fn`) — it re-exports the nerd_herd `QueueProfile` shape with
`by_capability`.

- [ ] **Step 3: Verify the sim still runs and S9-era scenarios are unchanged**

Run: `$PY packages/fatih_hoca/tests/sim/run_scenarios.py`
Expected: table prints; **rp1_realistic free_q ≈ 28%**, hard 100%, PP1–PP9 all PASS (the clock
change is mathematically equivalent for S9; S7 is still ~0 because no scenario hammers a tank yet).
If rp1 free_q moved materially from ~28%, STOP — the reset_at/now math is inconsistent; re-check Step 1.

- [ ] **Step 4: Commit**

```bash
rtk git add packages/fatih_hoca/tests/sim/scenarios.py
rtk git commit -m "feat(sim): consistent virtual clock + per-pick burn_log + capability demand

now = wall_anchor + virtual_clock threaded into pressure_for; reset_at made
absolute so S9 reset-proximity is unchanged while S7's 300s window now evicts
correctly. Each cloud pick records into a per-run BurnLog; queue gains
by_capability so S6 has demand. S9/S12 scenarios verified unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: New sim scenarios — overdraw early-warning, S7 continuity probe, S6 capability-conserve

**Files:**
- Modify: `packages/fatih_hoca/tests/sim/scenarios.py` (append scenarios + assertions + registry entries)
- Modify: `packages/fatih_hoca/tests/sim/run_scenarios.py` (register)

- [ ] **Step 1: Add the S7 continuity probe assertion (pressure-only, deterministic)**

Append to `scenarios.py`:
```python
# ── S7 continuity probe — burn-rate signal must be continuous + monotonic ─────

def assert_s7_continuity() -> list[str]:
    """Sample S7 across rising burn on a small tank: strictly increasing
    magnitude, no flat-zero region once burn is nonzero, saturating at -1."""
    import time as _t
    from nerd_herd.types import (
        SystemSnapshot, CloudProviderState, CloudModelState,
        RateLimit, RateLimitMatrix,
    )
    from nerd_herd.burn_log import BurnLog

    now = _t.time()
    rpd = RateLimit(limit=20, remaining=10, reset_at=int(now + 3600))
    cms = CloudModelState(model_id="gem/flash", utilization_pct=0.0,
                          limits=RateLimitMatrix(rpd=rpd))
    snap = SystemSnapshot(cloud={"gem": CloudProviderState(
        provider="gem", models={"gem/flash": cms})})
    from types import SimpleNamespace
    m = SimpleNamespace(name="gem/flash", provider="gem", is_local=False,
                        is_free=True, cap_score=7.0, capabilities=set(), rpd_remaining=10)

    mags = []
    for n_calls in (0, 1, 2, 4, 8, 16):
        bl = BurnLog(window_secs=300.0)
        for i in range(n_calls):
            bl.record(provider="gem", model="gem/flash", tokens=500, calls=1, now=now - i)
        s7 = snap.pressure_for(m, task_difficulty=5, now=now, burn_log=bl).signals["S7"]
        mags.append(-s7)  # magnitude (S7 <= 0)

    failures = []
    if mags[0] != 0.0:
        failures.append(f"S7-cont: zero burn must give 0, got {mags[0]}")
    # Monotonic non-decreasing in burn, strictly increasing somewhere before sat.
    if any(b < a - 1e-9 for a, b in zip(mags, mags[1:])):
        failures.append(f"S7-cont: magnitude not monotonic in burn: {mags}")
    if not (mags[1] > 0.0):
        failures.append(f"S7-cont: any nonzero burn must lift S7 off zero, got {mags[1]}")
    if not (mags[-1] >= mags[1]):
        failures.append("S7-cont: heavy burn must not be weaker than light burn")
    return failures
```

- [ ] **Step 2: Add the S6 capability-conserve assertion (pressure-only)**

Append to `scenarios.py`:
```python
# ── S6 capability-conserve — graded conserve-pressure, fed via rollup ─────────

def assert_s6_conserve() -> list[str]:
    """Vision demand >> vision supply → S6 fires negative and is graded
    (heavier shortage = stronger), not a single bang-bang step."""
    from nerd_herd.types import QueueProfile
    from nerd_herd.signals.s6_capable_supply import s6_capable_supply
    from types import SimpleNamespace

    vm = SimpleNamespace(name="v/m", provider="p", is_local=False, is_free=False,
                         cap_score=8.5, capabilities={"vision"}, rpd_remaining=20)
    light = QueueProfile(by_capability={"vision": 25}, total_ready_count=25)
    heavy = QueueProfile(by_capability={"vision": 200}, total_ready_count=200)
    s6_light = s6_capable_supply(vm, queue=light, eligible_models=[vm], iter_avg=8.0)
    s6_heavy = s6_capable_supply(vm, queue=heavy, eligible_models=[vm], iter_avg=8.0)

    failures = []
    if not (s6_heavy < 0):
        failures.append(f"S6: heavy vision shortage must be negative, got {s6_heavy}")
    if not (s6_heavy <= s6_light):
        failures.append(f"S6: heavier shortage must be >= magnitude (heavy {s6_heavy} "
                        f"!<= light {s6_light}) — graded, not bang-bang")
    return failures
```

- [ ] **Step 3: Add the overdraw early-warning full-sim scenario + assertion**

Append to `scenarios.py`:
```python
# ── rp5: overdraw early-warning — hammer a tank, load must shift before 0 ─────

def rp5_overdraw_early_warning() -> Scenario:
    """One mid-cap free giant + one comparable-cap free peer + paid fallback.
    A steady stream of medium tasks. With S7 de-blinded, sustained burn on the
    first free pool must raise conserve-pressure and shift share to the peer
    BEFORE the hammered pool hits 0 (no exhaustion → no 'no candidates')."""
    providers = {
        "hot": {"is_free": True, "models": {"hot/m": {"cap_score_100": 78}}},
        "cool": {"is_free": True, "models": {"cool/m": {"cap_score_100": 77}}},
        "anthropic": {"is_free": False, "models": {"anthropic/claude": {"cap_score_100": 92}}},
    }
    state = SimState()
    state.locals["loaded-local"] = SimLocalModel(is_loaded=True, idle_seconds=300.0, tokens_per_second=15.0)
    # Small-ish equal tanks, reset 2h out so burn extrapolation is meaningful.
    state.time_bucketed["hot/m"] = SimPoolCounter(remaining=60, limit=60, reset_at=7200.0)
    state.time_bucketed["cool/m"] = SimPoolCounter(remaining=60, limit=60, reset_at=7200.0)
    state.per_call["anthropic/claude"] = SimPoolCounter(remaining=1000, limit=1000, reset_at=86400.0)
    tasks = [SimTask(idx=i, difficulty=6, estimated_output_tokens=1500) for i in range(100)]
    return Scenario(
        name="rp5_overdraw_early_warning",
        tasks=tasks,
        initial_state=state,
        snapshot_factory=_build_snapshot_factory(providers),
        select_fn=_build_select_fn(providers, tasks=tasks),
    )


def assert_rp5(scenario: "Scenario") -> list[str]:
    from sim.runner import run_simulation
    run = run_simulation(
        tasks=scenario.tasks, initial_state=scenario.initial_state,
        select_fn=scenario.select_fn, snapshot_factory=scenario.snapshot_factory,
    )
    hot = run.final_state.time_bucketed.get("hot/m")
    cool = run.final_state.time_bucketed.get("cool/m")
    failures = []
    # Neither free pool may be driven to exhaustion (the whole point of early warning).
    if hot and hot.remaining <= 0:
        failures.append(f"rp5: hot pool exhausted (remaining={hot.remaining}) — S7 did not warn early")
    # Load must actually be shared across the two free peers (not all on one).
    hot_used = (hot.limit - hot.remaining) if hot else 0
    cool_used = (cool.limit - cool.remaining) if cool else 0
    if cool_used == 0 and hot_used > 0:
        failures.append("rp5: cool peer never used — load did not spread off the hammered pool")
    return failures
```

- [ ] **Step 4: Register new scenarios/assertions in `scenarios.py`**

Add to `POOL_PRESSURE_ASSERTIONS` (these take no scenario arg — wrap to ignore it):
```python
    "s7_continuity": lambda sc: assert_s7_continuity(),
    "s6_conserve": lambda sc: assert_s6_conserve(),
    "rp5_overdraw_early_warning": lambda sc: assert_rp5(sc),
```
Add to `POOL_PRESSURE_SCENARIOS` (the two pressure-only probes need a placeholder scenario so the
loop has something to pass; reuse a trivial factory):
```python
    ("s7_continuity", pp1_fat_vs_tiny),   # scenario arg unused by the assertion
    ("s6_conserve", pp1_fat_vs_tiny),     # scenario arg unused by the assertion
    ("rp5_overdraw_early_warning", rp5_overdraw_early_warning),
```

- [ ] **Step 5: Surface rp5 in the realistic table too (optional eyeball)**

In `scenarios.py`, add to `REALISTIC_POOL_SCENARIOS`:
```python
    ("rp5_overdraw_early_warning", rp5_overdraw_early_warning),
```

- [ ] **Step 6: Run the harness — new probes must PASS, S7 must actually fire in rp5**

Run: `$PY packages/fatih_hoca/tests/sim/run_scenarios.py`
Expected: `s7_continuity` PASS, `s6_conserve` PASS, `rp5_overdraw_early_warning` PASS; rp1 free_q
still ≈28%, hard 100%. If rp5 FAILs on exhaustion, that is the expected pre-C4 state (S7 still
gated at 0.70) — proceed to Part B which de-blinds it, then this assertion becomes the gate.

> Note: rp5 may legitimately FAIL here (before C4) because the 0.70 gate keeps S7 silent until
> nearly exhausted. That is the whole point — record the failure, move to Part B, and rp5 must flip
> to PASS after the ramp + SAT tune. Do not weaken the assertion to make it pass early.

- [ ] **Step 7: Commit**

```bash
rtk git add packages/fatih_hoca/tests/sim/scenarios.py packages/fatih_hoca/tests/sim/run_scenarios.py
rtk git commit -m "test(sim): add S7 continuity probe, S6 conserve probe, rp5 overdraw early-warning

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Part B — C4: soften S7 / S6 dead-bands to ramp-from-0

### Task 7: S7 smoothstep ramp + continuity unit tests

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/signals/s7_burn_rate.py`
- Create: `packages/nerd_herd/tests/signals/test_s7_continuity.py`

- [ ] **Step 1: Write the failing test**

Create `packages/nerd_herd/tests/signals/test_s7_continuity.py`:
```python
import time
from nerd_herd.types import RateLimit, RateLimitMatrix
from nerd_herd.burn_log import BurnLog
from nerd_herd.signals.s7_burn_rate import s7_burn_rate, SAT


def _matrix(remaining, limit, reset_in):
    now = time.time()
    return RateLimitMatrix(rpd=RateLimit(limit=limit, remaining=remaining,
                                         reset_at=int(now + reset_in))), now


def test_cold_start_zero():
    mtx, now = _matrix(10, 20, 3600)
    bl = BurnLog(300.0)
    assert s7_burn_rate(mtx, provider="p", model="m", burn_log=bl, now=now) == 0.0


def test_ramp_is_continuous_from_zero_no_deadband():
    # Light burn that yields ratio well under the OLD 0.70 gate must now be
    # nonzero (de-blinded) — proves the dead-band is gone.
    mtx, now = _matrix(remaining=10_000, limit=14_400, reset_in=3600)
    bl = BurnLog(300.0)
    # ~12 calls in window → calls_per_min = 12*60/300 = 2.4; extrapolated over
    # 60min = 144; ratio = 144/10000 = 0.0144 → old gate gave exactly 0.
    for i in range(12):
        bl.record(provider="p", model="m", tokens=100, calls=1, now=now - i)
    s7 = s7_burn_rate(mtx, provider="p", model="m", burn_log=bl, now=now)
    assert s7 < 0.0          # de-blinded: was 0 under the 0.70 gate
    assert s7 > -0.2         # but still a whisper, not a shout


def test_monotonic_in_burn_and_saturates():
    mtx, now = _matrix(remaining=20, limit=20, reset_in=3600)
    mags = []
    for n in (1, 3, 6, 12, 30):
        bl = BurnLog(300.0)
        for i in range(n):
            bl.record(provider="p", model="m", tokens=100, calls=1, now=now - i)
        mags.append(-s7_burn_rate(mtx, provider="p", model="m", burn_log=bl, now=now))
    assert all(b >= a - 1e-9 for a, b in zip(mags, mags[1:]))
    assert mags[-1] == 1.0   # heavy burn on a tiny tank saturates at -1


def test_sat_constant_exists():
    assert SAT > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/nerd_herd && timeout 60 $PY -m pytest tests/signals/test_s7_continuity.py -v`
Expected: FAIL — `ImportError: cannot import name 'SAT'` and the de-band assertion fails (old code
returns 0 under 0.70).

- [ ] **Step 3: Replace the dead-band with a smoothstep ramp**

In `packages/nerd_herd/src/nerd_herd/signals/s7_burn_rate.py`:
- Add import: `from nerd_herd.signals._curves import smoothstep`
- Replace the module constants `THRESHOLD = 0.70` / `SLOPE = 2.0` with:
```python
# Continuous ramp-from-0 (2026-06-05, replaces the 0.70 dead-band). SAT is the
# ratio at which conserve-pressure saturates to -1; tuned by sim SAT-sweep
# (run_scenarios.py rp5). Lower SAT = earlier/stronger warning. smoothstep keeps
# light burn quiet (3x^2-2x^3 ≈ 3·ratio^2 near 0) while de-blinding overdraw
# before exhaustion. See docs/superpowers/specs/2026-06-05-s7-s6-continuity-design.md.
SAT = 1.0
```
- Replace the per-axis pressure computation. Current:
```python
        excess = max(0.0, ratio - THRESHOLD)
        if excess <= 0:
            continue
        pressure = -min(1.0, excess * SLOPE)
        if pressure < worst:
            worst = pressure
```
with:
```python
        pressure = -smoothstep(min(1.0, ratio / SAT))
        if pressure < worst:
            worst = pressure
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/nerd_herd && timeout 60 $PY -m pytest tests/signals/test_s7_continuity.py -v`
Expected: PASS (4).

- [ ] **Step 5: Regression — nerd_herd suite**

Run: `cd packages/nerd_herd && timeout 120 $PY -m pytest tests/ -q`
Expected: PASS. If a prior S7 test asserted the 0.70 gate, update it to the ramp (the gate is gone
by design) — show the diff and keep the assertion meaningful (monotonic + saturation), don't delete.

- [ ] **Step 6: Commit**

```bash
rtk git add packages/nerd_herd/src/nerd_herd/signals/s7_burn_rate.py packages/nerd_herd/tests/signals/test_s7_continuity.py
rtk git commit -m "fix(nerd-herd): S7 burn-rate continuous smoothstep ramp (de-blind overdraw)

Replaces the 0.70 dead-band (flat-zero until hot, then ramp) with
-smoothstep(ratio/SAT) from ratio=0. Big-tank overdraw now whispers early
instead of shouting at exhaustion. SAT=1.0 pending sim sweep.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: S6 smoothstep ramp + continuity unit tests

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/signals/s6_capable_supply.py`
- Create: `packages/nerd_herd/tests/signals/test_s6_continuity.py`

- [ ] **Step 1: Write the failing test**

Create `packages/nerd_herd/tests/signals/test_s6_continuity.py`:
```python
from types import SimpleNamespace
from nerd_herd.types import QueueProfile
from nerd_herd.signals.s6_capable_supply import s6_capable_supply, SAT


def _vm(rpd=20):
    return SimpleNamespace(name="v/m", provider="p", is_local=False, is_free=False,
                           cap_score=8.5, capabilities={"vision"}, rpd_remaining=rpd)


def test_empty_demand_zero():
    assert s6_capable_supply(_vm(), queue=QueueProfile(), eligible_models=[_vm()]) == 0.0


def test_no_eligible_models_zero():
    vm = _vm()
    q = QueueProfile(by_capability={"vision": 100}, total_ready_count=100)
    assert s6_capable_supply(vm, queue=q, eligible_models=[]) == 0.0


def test_ramp_continuous_from_low_shortage():
    vm = _vm(rpd=20)
    # Light demand below the OLD 0.70 ratio must now be nonzero (de-blinded).
    # demand = 12*8 = 96; supply = 20*8 = 160; ratio = 0.6 < 0.70 → old gave 0.
    q = QueueProfile(by_capability={"vision": 12}, total_ready_count=12)
    s6 = s6_capable_supply(vm, queue=q, eligible_models=[vm], iter_avg=8.0)
    assert s6 < 0.0


def test_monotonic_in_shortage():
    vm = _vm(rpd=20)
    mags = []
    for demand in (12, 25, 50, 100, 400):
        q = QueueProfile(by_capability={"vision": demand}, total_ready_count=demand)
        mags.append(-s6_capable_supply(vm, queue=q, eligible_models=[vm], iter_avg=8.0))
    assert all(b >= a - 1e-9 for a, b in zip(mags, mags[1:]))
    assert mags[-1] == 1.0


def test_sat_constant_exists():
    assert SAT > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/nerd_herd && timeout 60 $PY -m pytest tests/signals/test_s6_continuity.py -v`
Expected: FAIL — `ImportError: cannot import name 'SAT'`; `test_ramp_continuous_from_low_shortage`
fails (old 0.70 gate returns 0 at ratio 0.6).

- [ ] **Step 3: Replace the dead-band with a smoothstep ramp**

In `packages/nerd_herd/src/nerd_herd/signals/s6_capable_supply.py`:
- Add import: `from nerd_herd.signals._curves import smoothstep`
- Replace `THRESHOLD = 0.70` / `SLOPE = 2.0` with:
```python
# Continuous ramp-from-0 (2026-06-05, replaces the 0.70 dead-band) — same shape
# as S7. SAT = demand/supply ratio at which conserve-pressure saturates.
SAT = 1.0
```
- Replace the pressure computation. Current:
```python
        ratio = demand / supply
        excess = max(0.0, ratio - THRESHOLD)
        if excess <= 0:
            continue
        pressure = -min(1.0, excess * SLOPE)
        if pressure < worst:
            worst = pressure
```
with:
```python
        ratio = demand / supply
        pressure = -smoothstep(min(1.0, ratio / SAT))
        if pressure < worst:
            worst = pressure
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/nerd_herd && timeout 60 $PY -m pytest tests/signals/test_s6_continuity.py -v`
Expected: PASS (5).

- [ ] **Step 5: Regression — nerd_herd suite + the PP6 assertion**

Run: `cd packages/nerd_herd && timeout 120 $PY -m pytest tests/ -q`
Then: `$PY packages/fatih_hoca/tests/sim/run_scenarios.py` and confirm `pp6_capability_shortage`
still PASS (it asserts only `s6 < 0`, which the ramp preserves).
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add packages/nerd_herd/src/nerd_herd/signals/s6_capable_supply.py packages/nerd_herd/tests/signals/test_s6_continuity.py
rtk git commit -m "fix(nerd-herd): S6 capable-supply continuous smoothstep ramp + reactivation

Same de-band as S7. With the ranking rollup now feeding eligible_models,
S6 emits graded conserve-pressure on capability shortage instead of being
structurally dead.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: SAT sweep + tune, full sim validation, swap-storm, full suites

**Files:**
- Modify (if sweep dictates): `packages/nerd_herd/src/nerd_herd/signals/s7_burn_rate.py` (`SAT`), `s6_capable_supply.py` (`SAT`)

- [ ] **Step 1: Run the full sim harness and read the table**

Run: `$PY packages/fatih_hoca/tests/sim/run_scenarios.py`
Record: rp1 free_q (target ≈28%, must NOT regress below ~25%), hard_task_satisfaction (must stay
100% on all rp*), rp5 result (must now PASS — load spread, hot pool not exhausted), all PP1–PP9 +
s7_continuity + s6_conserve PASS.

- [ ] **Step 2: SAT sweep for S7 (manual, three values)**

For `SAT` ∈ {1.0, 1.2, 1.4} in `s7_burn_rate.py`, run the harness each time and record rp1 free_q,
rp5 (exhaustion avoided?), and whether rp1/rp3 hard_task_satisfaction stays 100% and easy `waste`
stays ~0 (no easy→premium drift). Pick the smallest SAT that:
  (a) flips rp5 to PASS (hot pool not exhausted, cool peer used), AND
  (b) keeps rp1 free_q ≥ ~25% and all hard satisfaction 100%, AND
  (c) does not raise easy `waste` above its pre-change value.
Set S7 `SAT` to that value. Keep S6 `SAT` = the same value unless the s6_conserve probe or pp6
needs otherwise (document if they diverge).

- [ ] **Step 3: If no SAT in {1.0,1.2,1.4} satisfies (a)+(b)+(c) — apply the documented fallback**

Swap the S7/S6 curve from `smoothstep(min(1, ratio/SAT))` to the gentler `min(1.0, (ratio/SAT)**2)`
(slower low-end rise → less idle-burn penalty), re-run the sweep. Record which curve+SAT won and
why in the commit message. (This is the spec's documented fallback, not a silent change.)

- [ ] **Step 4: Swap-storm check**

Run: `$PY packages/fatih_hoca/tests/sim/run_swap_storm_check.py`
Expected: clean (0–~0.5% swaps; local stickiness intact). If swaps spike, S7/S6 are leaking into
local selection — STOP and investigate (they should be 0 for local models).

- [ ] **Step 5: Full package suites**

Run each (CLAUDE.md timeouts):
```
cd packages/nerd_herd && timeout 120 $PY -m pytest tests/ -q
cd packages/fatih_hoca && timeout 120 $PY -m pytest tests/ -q
cd packages/kuleden_donen_var && timeout 120 $PY -m pytest tests/ -q
```
Expected: all PASS. Fix any test that encoded the old 0.70 gate by updating it to the ramp
contract (monotonic + saturation), not by deleting coverage.

- [ ] **Step 6: Commit the tuned knobs**

```bash
rtk git add packages/nerd_herd/src/nerd_herd/signals/s7_burn_rate.py packages/nerd_herd/src/nerd_herd/signals/s6_capable_supply.py
rtk git commit -m "tune(nerd-herd): S7/S6 SAT=<value> by sim sweep (rp5 early-warning + rp1 free_q)

SAT-sweep {1.0,1.2,1.4}: chose <value> — flips rp5 (no exhaustion, peer
used), holds rp1 free_q ~28%, hard_task_satisfaction 100%, easy waste flat.
swap-storm clean. <fallback note if ratio**2 used>.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Spec outcome note + memory + handoff

**Files:**
- Modify: `docs/superpowers/specs/2026-06-05-s7-s6-continuity-design.md` (append outcome)
- Create: `C:\Users\sakir\.claude\projects\C--Users-sakir-Dropbox-Workspaces-kutay\memory\project_s7_s6_continuity_20260605.md` + MEMORY.md pointer
- Create: `docs/handoff/2026-06-05-s7-s6-continuity-handoff.md`

- [ ] **Step 1: Append an "Implementation outcome" section to the spec**

Record: chosen SAT (+ curve if fallback used), final rp1 free_q / rp5 result / hard satisfaction,
test counts per suite, and "Not live until KutAI restart." Mirror the parent spec's §8b style.

- [ ] **Step 2: Write the memory file**

`metadata.type: project`. One fact: what shipped (S7/S6 de-band + S6 reactivation + sim
clock/burn/eligible wiring), the chosen SAT, that it is restart-gated, and link
`[[project_cloud_diversity_collapse_20260604]]`. Add a one-line pointer to MEMORY.md.

- [ ] **Step 3: Write the handoff** mirroring `2026-06-04-cloud-utilization-equilibrium-handoff.md`:
why, what shipped (table), validation evidence, first-action-next-session (restart-gated verify:
`model_pick_log` should show free-provider diversity holding + no "No model candidates" tail).

- [ ] **Step 4: Commit**

```bash
rtk git add docs/superpowers/specs/2026-06-05-s7-s6-continuity-design.md docs/handoff/2026-06-05-s7-s6-continuity-handoff.md
rtk git commit -m "docs(fatih-hoca): S7/S6 continuity outcome + handoff

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Part A (prereq) → Tasks 1–6. A1 clock = Tasks 2/3/4/5; A2 burn_log = Tasks 2/5; A3 eligible+demand = Tasks 2/3/4/5.
- Part B (C4 ramps) → Tasks 7 (S7), 8 (S6), shared curve = Task 1.
- Validation §4 → unit tests in Tasks 1,2,3,7,8; sim scenarios in Task 6; sweep+suites+swap-storm in Task 9.
- Sequencing §5 (Part A observable before curve change) → Task 6 Step 6 explicitly records rp5's pre-C4 failure, Part B flips it.
- Risk §6 (over-penalty fallback) → Task 9 Step 3 (`ratio**2`).
- Out-of-scope §7 → nothing in the plan touches cap-parity/K, M1, discovery, catalog. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; SAT value resolved by the Task 9 sweep (a real procedure, not a placeholder).

**Type consistency:** `pressure_for(now, burn_log, eligible_models)` defined Task 2, consumed Tasks 3/5; `rank_candidates(now, burn_log)` defined Task 3, called Task 4; `_SimPickResult.provider` defined Task 4, read Task 5; `smoothstep` defined Task 1, imported Tasks 7/8; `SAT` defined Tasks 7/8, asserted by their tests + tuned Task 9; `by_capability` produced Task 5, consumed by S6 (Task 8) via the existing `s6_capable_supply` queue read.
