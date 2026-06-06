# Mid-task Urgency Asymmetry Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop "No model candidates available" from firing on already-running tasks by giving mid-task ReAct re-selection the task's real admission urgency plus a finish-bias, instead of a flat 0.5 that is stricter than admission.

**Architecture:** A started task should finish on the same alive pool-pressure band it was admitted under. Beckman stamps the admission urgency on the task dict. A single pure helper in `fatih_hoca` computes mid-task urgency = `admission_urgency + FINISH_BIAS` (`+ FAILURE_BUMP` when failures present). Both ReAct callers (coulson agent loop, husam raw_dispatch loop) use the helper. The pool-pressure gate, the −1.0 hard veto, and admission-wait-on-None are unchanged.

**Tech Stack:** Python 3.10, pytest, venv at `.venv/Scripts/python.exe`. Packages: `fatih_hoca`, `general_beckman`, `coulson`, `husam`. Tests run separately per package (the `tests/` vs `packages/*/tests/` conftest collision), always with a timeout and `-p no:warnings`.

**Spec:** `docs/superpowers/specs/2026-06-04-mid-task-urgency-asymmetry-design.md`

---

### Task 1: Pure helper `mid_task_urgency` in fatih_hoca

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/urgency.py`
- Modify: `packages/fatih_hoca/src/fatih_hoca/__init__.py` (export the helper)
- Test: `packages/fatih_hoca/tests/test_mid_task_urgency.py`

- [ ] **Step 1: Write the failing test**

Create `packages/fatih_hoca/tests/test_mid_task_urgency.py`:

```python
"""Mid-task urgency helper — a started task finishes on (at least) its
admission urgency plus a finish-bias, so it is never judged stricter than
a fresh task. Design: docs/superpowers/specs/2026-06-04-mid-task-urgency-asymmetry-design.md
"""
from fatih_hoca.urgency import mid_task_urgency, FINISH_BIAS, FAILURE_BUMP


def test_baseline_no_failures_adds_finish_bias():
    # admission urgency 0.5 (priority-5 baseline) → 0.5 + 0.1 finish-bias
    assert mid_task_urgency(0.5, has_failures=False) == 0.6


def test_baseline_with_failures_stacks_failure_bump():
    # 0.5 + 0.1 finish + 0.1 failure
    assert abs(mid_task_urgency(0.5, has_failures=True) - 0.7) < 1e-9


def test_high_admission_urgency_is_honored():
    # a priority>5 / aged task admitted at 0.8 keeps that band mid-task
    assert abs(mid_task_urgency(0.8, has_failures=False) - 0.9) < 1e-9


def test_clamped_at_one():
    assert mid_task_urgency(0.95, has_failures=True) == 1.0
    assert mid_task_urgency(1.0, has_failures=False) == 1.0


def test_none_base_falls_back_to_half():
    assert mid_task_urgency(None, has_failures=False) == 0.6


def test_constants_are_tenths():
    assert FINISH_BIAS == 0.1
    assert FAILURE_BUMP == 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest packages/fatih_hoca/tests/test_mid_task_urgency.py -p no:warnings -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'fatih_hoca.urgency'`

- [ ] **Step 3: Write the helper**

Create `packages/fatih_hoca/src/fatih_hoca/urgency.py`:

```python
"""Mid-task urgency policy (single source of truth).

A task admitted under a given pool-pressure urgency must not be re-judged
*stricter* mid-flight than it was at admission — otherwise a started task
gets vetoed off the alive band (scalar in -0.75..-1.0) that admitted it and
raises "No model candidates available". The mid-task urgency is therefore
the admission urgency plus a small finish-bias ("a little higher than
pre-dispatch", user design 2026-05-03), with an extra bump while a retry
failure is being adapted around.

Does NOT change the pool-pressure gate formula
(``selector.py``: threshold = -0.5 - 0.5*urgency) or the -1.0 hard veto.
"""
from __future__ import annotations

FINISH_BIAS = 0.1   # mid-task urgency sits a little above pre-dispatch urgency
FAILURE_BUMP = 0.1  # extra escalation while adapting around a transport failure


def mid_task_urgency(admission_urgency: float | None, *, has_failures: bool) -> float:
    """Urgency for a mid-task (re-)selection.

    ``admission_urgency`` is the value Beckman computed at admission
    (``compute_urgency``), stamped on the task as ``_admission_urgency``.
    Falls back to 0.5 when unknown.
    """
    base = 0.5 if admission_urgency is None else float(admission_urgency)
    urgency = min(1.0, base + FINISH_BIAS)
    if has_failures:
        urgency = min(1.0, urgency + FAILURE_BUMP)
    return urgency
```

- [ ] **Step 4: Export from the package**

In `packages/fatih_hoca/src/fatih_hoca/__init__.py`, add to the imports/exports (next to where `select` is exported):

```python
from fatih_hoca.urgency import mid_task_urgency  # noqa: F401
```

If `__init__.py` defines `__all__`, append `"mid_task_urgency"` to it.

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest packages/fatih_hoca/tests/test_mid_task_urgency.py -p no:warnings -q`
Expected: PASS (6 passed)

- [ ] **Step 6: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/urgency.py packages/fatih_hoca/src/fatih_hoca/__init__.py packages/fatih_hoca/tests/test_mid_task_urgency.py
git commit -m "feat(fatih_hoca): mid_task_urgency helper (admission urgency + finish-bias)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Beckman stamps `_admission_urgency` on the task

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/__init__.py:692`
- Test: `packages/general_beckman/tests/test_admission_stamps_urgency.py`

**Context:** At admission, `urgency = compute_urgency(task)` is computed (~line 584) and `task["preselected_pick"] = pick` is set (line 692). The task dict flows in-memory to the coulson ReAct loop (same path `preselected_pick` already rides). We add a sibling stamp. Because the function (`next_task`) is a long async DB loop with no unit seam, the test asserts the *invariant the stamp must satisfy* — `_admission_urgency` equals `compute_urgency(task)` — via a tiny check on the same inputs, guarding against the stamp being dropped or computed differently.

- [ ] **Step 1: Write the failing test**

Create `packages/general_beckman/tests/test_admission_stamps_urgency.py`:

```python
"""Admission must stamp the urgency it used onto the task dict so mid-task
re-selection (coulson/husam) can reuse it. Guards the contract between
admission and fatih_hoca.mid_task_urgency.
"""
import time

from general_beckman.admission import compute_urgency
from general_beckman import _stamp_admission_urgency


def _task(priority=8, age_s=0, unblocks=0):
    return {
        "id": 1,
        "priority": priority,
        "created_at": time.time() - age_s,
        "downstream_unblocks_count": unblocks,
    }


def test_stamp_matches_compute_urgency():
    task = _task(priority=8)
    expected = compute_urgency(task)
    _stamp_admission_urgency(task)
    assert task["_admission_urgency"] == expected


def test_stamp_baseline_is_half():
    task = _task(priority=5)
    _stamp_admission_urgency(task)
    assert abs(task["_admission_urgency"] - 0.5) < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest packages/general_beckman/tests/test_admission_stamps_urgency.py -p no:warnings -q`
Expected: FAIL — `ImportError: cannot import name '_stamp_admission_urgency'`

- [ ] **Step 3: Add the helper and use it at the stamp site**

In `packages/general_beckman/src/general_beckman/__init__.py`, add a small module-level helper (near the top-level functions, after the imports):

```python
def _stamp_admission_urgency(task: dict) -> float:
    """Compute the admission urgency and stamp it on the task dict so
    mid-task re-selection can reuse it (one source of truth)."""
    from general_beckman.admission import compute_urgency
    u = compute_urgency(task)
    task["_admission_urgency"] = u
    return u
```

Then change the admission body. Where it currently computes urgency (~line 584):

```python
        urgency = compute_urgency(task)
```

replace with:

```python
        urgency = _stamp_admission_urgency(task)
```

(This keeps `urgency` in scope for the existing `select(..., urgency=urgency, ...)` call and the ADMIT log line, and guarantees the stamp uses the exact same value passed to the selector.)

Confirm `compute_urgency` is still imported where line 584 used it (it is imported at the top of the function per the existing `from general_beckman.admission import compute_urgency` at ~line 279); the helper imports it locally too, so both paths work.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest packages/general_beckman/tests/test_admission_stamps_urgency.py -p no:warnings -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add packages/general_beckman/src/general_beckman/__init__.py packages/general_beckman/tests/test_admission_stamps_urgency.py
git commit -m "feat(general_beckman): stamp _admission_urgency on task at admission

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: coulson `pick_for_iter` uses the admission urgency (PRIMARY)

**Files:**
- Modify: `packages/coulson/src/coulson/dispatch_helpers.py:61-63`
- Test: `packages/coulson/tests/test_pick_for_iter_urgency.py` (new)
- Test: `packages/coulson/tests/test_pick_for_iter_reuse.py:108-109` (update existing assertion)

**Context (current code, `dispatch_helpers.py:56-90`):** after the reuse short-circuit, it sets `urgency = 0.5`, then `if failures: urgency = min(1.0, urgency + 0.1)`, and passes `urgency=urgency` to `fatih_hoca.select(...)`. The flat 0.5 ignores the task's admission urgency. `task` here IS the Beckman task dict (it already reads `task.get("preselected_pick")`), so `task.get("_admission_urgency")` is available after Task 2.

- [ ] **Step 1: Write the failing test**

Create `packages/coulson/tests/test_pick_for_iter_urgency.py`:

```python
"""Mid-task re-selection must use the task's admission urgency + finish-bias,
not a flat 0.5 — a started task is never judged stricter than a fresh one.
Drives the REAL pick_for_iter; only the fatih_hoca selector boundary is stubbed.
"""
from __future__ import annotations

from types import SimpleNamespace

import fatih_hoca
import coulson.dispatch_helpers as dh
from fatih_hoca.types import Pick, Failure
from fatih_hoca.requirements import ModelRequirements


def _pick(name: str) -> Pick:
    model = SimpleNamespace(name=name, litellm_name=name, is_local=True, is_loaded=True)
    return Pick(model=model, min_time_seconds=1.0)


def _reqs() -> ModelRequirements:
    return ModelRequirements(task="coder", agent_type="coder", difficulty=5)


def _tracking_select(monkeypatch, returns: Pick):
    calls: list[dict] = []

    def _sel(**kwargs):
        calls.append(kwargs)
        return returns

    monkeypatch.setattr(fatih_hoca, "select", _sel)
    return calls


def test_reselect_uses_admission_urgency_plus_finish_bias(monkeypatch):
    # held no longer servable → re-select; admission urgency 0.8 → 0.8 + 0.1
    monkeypatch.setattr(fatih_hoca, "is_servable", lambda **kw: False)
    fresh = _pick("gemini/gemma-4-26b")
    calls = _tracking_select(monkeypatch, fresh)

    task = {"id": 1, "preselected_pick": _pick("held"), "_admission_urgency": 0.8}
    result = dh.pick_for_iter(
        reqs=_reqs(), task=task, failures=[], iteration=2, remaining_budget=5.0,
    )
    assert result is fresh
    assert abs(calls[0]["urgency"] - 0.9) < 1e-9


def test_reselect_default_when_unstamped(monkeypatch):
    # no _admission_urgency → base 0.5 + 0.1 finish-bias = 0.6
    monkeypatch.setattr(fatih_hoca, "is_servable", lambda **kw: False)
    calls = _tracking_select(monkeypatch, _pick("x"))
    task = {"id": 1, "preselected_pick": _pick("held")}
    dh.pick_for_iter(reqs=_reqs(), task=task, failures=[], iteration=2, remaining_budget=5.0)
    assert abs(calls[0]["urgency"] - 0.6) < 1e-9


def test_failures_stack_failure_bump(monkeypatch):
    # admission 0.5 + finish 0.1 + failure 0.1 = 0.7
    monkeypatch.setattr(fatih_hoca, "is_servable", lambda **kw: True)
    calls = _tracking_select(monkeypatch, _pick("fallback"))
    task = {"id": 1, "preselected_pick": _pick("flaky"), "_admission_urgency": 0.5}
    dh.pick_for_iter(
        reqs=_reqs(), task=task,
        failures=[Failure(model="flaky", reason="timeout")],
        iteration=1, remaining_budget=5.0,
    )
    assert abs(calls[0]["urgency"] - 0.7) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest packages/coulson/tests/test_pick_for_iter_urgency.py -p no:warnings -q`
Expected: FAIL — `test_reselect_uses_admission_urgency_plus_finish_bias` asserts 0.9 but gets 0.5 (flat).

- [ ] **Step 3: Implement — read admission urgency via the helper**

In `packages/coulson/src/coulson/dispatch_helpers.py`, replace lines 61-63:

```python
    urgency = 0.5
    if failures:
        urgency = min(1.0, urgency + 0.1)
```

with:

```python
    from fatih_hoca.urgency import mid_task_urgency
    urgency = mid_task_urgency(
        task.get("_admission_urgency"), has_failures=bool(failures),
    )
```

- [ ] **Step 4: Update the existing reuse test's stale assertion**

In `packages/coulson/tests/test_pick_for_iter_reuse.py`, the test `test_reselects_when_failures_present` (lines ~108-109) asserts the old flat behavior:

```python
    # urgency bumped +0.1 when failures present
    assert calls[0]["urgency"] == 0.6
```

Replace with the new formula (no `_admission_urgency` stamped → base 0.5 + finish 0.1 + failure 0.1):

```python
    # mid-task urgency = base 0.5 + finish-bias 0.1 + failure-bump 0.1
    assert abs(calls[0]["urgency"] - 0.7) < 1e-9
```

- [ ] **Step 5: Run both coulson tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest packages/coulson/tests/test_pick_for_iter_urgency.py packages/coulson/tests/test_pick_for_iter_reuse.py -p no:warnings -q`
Expected: PASS (3 + 5 = 8 passed)

- [ ] **Step 6: Commit**

```bash
git add packages/coulson/src/coulson/dispatch_helpers.py packages/coulson/tests/test_pick_for_iter_urgency.py packages/coulson/tests/test_pick_for_iter_reuse.py
git commit -m "fix(coulson): mid-task re-selection uses admission urgency + finish-bias

A started ReAct task was re-judged at a flat urgency 0.5 (stricter than its
admission urgency), getting vetoed off the alive pool-pressure band that
admitted it -> spurious 'No model candidates available'. Now reuses the
stamped _admission_urgency via fatih_hoca.mid_task_urgency.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: husam raw_dispatch loop uses the helper

**Files:**
- Modify: `packages/husam/src/husam/worker.py:187-191`
- Test: `packages/husam/tests/test_worker_mid_task_urgency.py` (new, formula-boundary)

**Context (current code, `worker.py:186-202`):** in the `else` (mid-task / failures) branch it does `_urgency = urgency_in; if failures: _urgency = min(1.0, float(_urgency or 0.5) + 0.1)`, then `select(..., urgency=_urgency, ...)`. `urgency_in` comes from `llm_call.get("urgency")` defaulting to 0.5 (lines 123-125). This is the raw_dispatch/overhead path (graders etc.), a different dict shape than the agent task — so it does NOT carry `_admission_urgency`; it uses `urgency_in` as the base. We route it through the same helper so the finish-bias applies consistently. (Full admission-urgency threading into the raw_dispatch spec is out of scope per the design.)

- [ ] **Step 1: Write the failing test**

Create `packages/husam/tests/test_worker_mid_task_urgency.py`:

```python
"""husam mid-task re-selection routes urgency through the shared
fatih_hoca.mid_task_urgency helper, so the finish-bias applies on the
raw_dispatch path too. Boundary check on the helper contract husam relies on.
"""
from fatih_hoca.urgency import mid_task_urgency


def test_husam_base_gets_finish_bias():
    # urgency_in 0.5 (default) → 0.6 mid-task
    assert mid_task_urgency(0.5, has_failures=False) == 0.6


def test_husam_failures_stack():
    assert abs(mid_task_urgency(0.5, has_failures=True) - 0.7) < 1e-9
```

(The worker body is a long async function with no unit seam; this guards the helper contract husam calls. Behavioral coverage of the worker path comes from the existing husam suite in Task 5.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest packages/husam/tests/test_worker_mid_task_urgency.py -p no:warnings -q`
Expected: PASS already for the helper import IF Task 1 is merged — so instead verify the *worker still uses the old inline math* before editing by reading `worker.py:187-191`. If it still shows `min(1.0, float(_urgency or 0.5) + 0.1)`, proceed to Step 3. (This task's real change is wiring the worker to the helper; the test locks the helper values.)

- [ ] **Step 3: Implement — route the worker through the helper**

In `packages/husam/src/husam/worker.py`, replace lines 187-191:

```python
        # Mid-task urgency bump on retry recursion (see _do_dispatch rationale).
        _urgency = urgency_in
        if failures:
            _u = float(_urgency or 0.5) + 0.1
            _urgency = min(1.0, _u)
```

with:

```python
        # Mid-task urgency: admission urgency (urgency_in) + finish-bias, with
        # an extra bump while adapting around failures. Single source of truth
        # = fatih_hoca.mid_task_urgency (shared with the coulson ReAct loop).
        from fatih_hoca.urgency import mid_task_urgency
        _urgency = mid_task_urgency(urgency_in, has_failures=bool(failures))
```

- [ ] **Step 4: Run the husam test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest packages/husam/tests/test_worker_mid_task_urgency.py -p no:warnings -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add packages/husam/src/husam/worker.py packages/husam/tests/test_worker_mid_task_urgency.py
git commit -m "fix(husam): mid-task re-selection uses shared mid_task_urgency helper

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Regression suites + mandatory Phase 2d simulator

**Files:** none modified — verification only.

- [ ] **Step 1: Run the three package suites (separately, with timeout)**

Run each (Windows: use the venv python; do not run `tests/` and `packages/*/tests/` together):

```
.venv/Scripts/python.exe -m pytest packages/fatih_hoca/tests -p no:warnings -q
.venv/Scripts/python.exe -m pytest packages/coulson/tests -p no:warnings -q
.venv/Scripts/python.exe -m pytest packages/general_beckman/tests -p no:warnings -q
.venv/Scripts/python.exe -m pytest packages/husam/tests -p no:warnings -q
```

Expected: all PASS (no regressions). If a DB-integration test in beckman is slow (~19s embedding-model load), that is expected, not a hang.

- [ ] **Step 2: Run the mandatory Phase 2d equilibrium simulator (CLAUDE.md rule)**

Run:

```
.venv/Scripts/python.exe packages/fatih_hoca/tests/sim/run_scenarios.py
.venv/Scripts/python.exe packages/fatih_hoca/tests/sim/run_swap_storm_check.py
```

Expected: both report their scenarios passing with no equilibrium regression. This change alters only the urgency *passed in* by mid-task callers, not the gate formula or signal scalars, so scenario outcomes should be unchanged. If any scenario drifts, STOP — investigate before proceeding (do not retune signals to mask it).

- [ ] **Step 3: Final import smoke check**

Run:

```
.venv/Scripts/python.exe -c "import fatih_hoca; print(fatih_hoca.mid_task_urgency(0.8, has_failures=False))"
```

Expected: prints `0.9`.

- [ ] **Step 4: Commit any nothing-to-commit note**

No code changes here. If all green, the feature is complete and ready for the next KutAI restart (user-managed via Telegram).

---

## Self-Review

**Spec coverage:**
- Change 1 (Beckman stamps `_admission_urgency`) → Task 2. ✓
- Change 2 (coulson uses admission urgency + finish-bias) → Task 3. ✓
- Change 3 (husam alignment) → Task 4. ✓
- Constants `FINISH_BIAS=0.1` / `FAILURE_BUMP=0.1`, home decided (fatih_hoca/urgency.py) → Task 1. ✓
- Hard veto / admission-wait / gate formula unchanged → no task touches selector.py or s9_perishability.py. ✓
- TDD coverage → Tasks 1-4 each test-first. ✓
- Mandatory Phase 2d sim → Task 5 Step 2. ✓
- Accepted limitation (baseline priority-5 still waits) → encoded by FINISH_BIAS=0.1 (0.5→0.6, threshold −0.80, does not force −0.83); no task forces it. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; commands have expected output. ✓

**Type/name consistency:** `mid_task_urgency(admission_urgency, *, has_failures)` and constants `FINISH_BIAS`/`FAILURE_BUMP` used identically in Tasks 1, 3, 4. `_stamp_admission_urgency(task)` defined and tested in Task 2. Task dict key `_admission_urgency` written in Task 2, read in Task 3. ✓
