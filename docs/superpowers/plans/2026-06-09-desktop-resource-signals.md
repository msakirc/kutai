# Desktop-Aware Resource Signals — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Teach KutAI's existing pressure engine to sense the desktop (user presence + machine contention) so local-vs-cloud placement and task deferral yield to the human automatically, and move context-sizing out of the dispatcher into Fatih's pick.

**Architecture:** No new package, no "governor". Two new local-only pressure signals (S12 presence, S13 contention) feed the existing Nerd-Herd `pressure_for` fold; a new M4 modifier re-expresses "load mode" as per-signal weights on S12/S13. Placement (Fatih ranking) and deferral (Beckman admission) fall out of the existing scalar with zero new policy. `need_ctx` becomes a `Pick` field computed by Fatih and consumed by the dispatcher.

**Tech Stack:** Python 3.10, `dataclasses`, `pynvml`+`psutil` (already deps), Windows `ctypes` (`GetLastInputInfo`, `GetForegroundWindow`), pytest.

**Spec:** `docs/superpowers/specs/2026-05-31-resource-signals-design.md` (read §0 verifications first).

**Project rules:** async throughout; lazy cross-module imports; `python -c "import ..."` smoke after each module; pytest ALWAYS with a timeout (`timeout 30 pytest <targeted>`); never run the full suite without `timeout 120`.

---

## File Structure

**New files**
- `packages/nerd_herd/src/nerd_herd/presence.py` — `PresenceCollector`: input-idle seconds + foreground-fullscreen. Windows ctypes, degrades to "away / not-fullscreen" off-Windows or on error.
- `packages/nerd_herd/src/nerd_herd/signals/s12_presence.py` — `s12_presence(...)`, local-only, negative-only.
- `packages/nerd_herd/src/nerd_herd/signals/s13_contention.py` — `s13_contention(...)`, local-only, negative-only.
- `packages/fatih_hoca/src/fatih_hoca/need_ctx.py` — `compute_need_ctx(...)` (the single owner of context sizing).
- Tests alongside: `packages/nerd_herd/tests/test_presence.py`, `test_s12_presence.py`, `test_s13_contention.py`, `test_m4_load_mode.py`, `test_pressure_desktop_integration.py`; `packages/fatih_hoca/tests/test_need_ctx.py`, `test_minimal_mode_eligibility.py`.

**Modified files**
- `packages/nerd_herd/src/nerd_herd/types.py` — `SystemSnapshot` gains desktop fields; `pressure_for` computes S12/S13 + applies M4.
- `packages/nerd_herd/src/nerd_herd/combine.py` — S12/S13 join `OTHER_BUCKET`.
- `packages/nerd_herd/src/nerd_herd/modifiers.py` — add `M4_load_mode_weights`.
- `packages/nerd_herd/src/nerd_herd/load.py` — cache last external-GPU fraction; raw VRAM (drop cap).
- `packages/nerd_herd/src/nerd_herd/nerd_herd.py` — `snapshot()` populates desktop fields; register `PresenceCollector`.
- `packages/fatih_hoca/src/fatih_hoca/types.py` — `Pick.need_ctx`.
- `packages/fatih_hoca/src/fatih_hoca/selector.py` — set `need_ctx` on Pick; Minimal-mode local eligibility gate.
- `src/core/llm_dispatcher.py:362-364` — consume `pick.need_ctx`.

**Deletions (Phase 6)**
- `src/infra/load_manager.py` sync stubs; `src/core/router.py:233-247,317`; `ranking.py:309-310` dead comment; P1 dead symbols + their tests/re-exports.

---

## Phase 1 — Sensing substrate

### Task 1: Add desktop fields to `SystemSnapshot`

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/types.py` (the `SystemSnapshot` dataclass, ~242-256)
- Test: `packages/nerd_herd/tests/test_snapshot_fields.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/nerd_herd/tests/test_snapshot_fields.py
from nerd_herd.types import SystemSnapshot


def test_snapshot_has_desktop_fields_with_safe_defaults():
    s = SystemSnapshot()
    # "away" defaults: huge idle, not fullscreen, no contention, full mode
    assert s.load_mode == "full"
    assert s.user_idle_s >= 1e8
    assert s.foreground_fullscreen is False
    assert s.ram_available_mb == 0
    assert s.ram_total_mb == 0
    assert s.external_gpu_fraction == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 pytest packages/nerd_herd/tests/test_snapshot_fields.py -v`
Expected: FAIL — `AttributeError`/`TypeError` (fields absent).

- [ ] **Step 3: Add the fields**

In `packages/nerd_herd/src/nerd_herd/types.py`, inside `@dataclass class SystemSnapshot`, after `recent_swap_count: int = 0`, add:

```python
    # ── Desktop-awareness fields (2026-06-09 resource-signals) ──────
    # Populated by NerdHerd.snapshot(). Defaults describe an absent user
    # on an idle machine in "full" mode, so a manually-built snapshot
    # (tests, sims) reads as "no desktop pressure" — identical to today.
    load_mode: str = "full"
    user_idle_s: float = 1e9          # seconds since last user input; large = away
    foreground_fullscreen: bool = False
    ram_available_mb: int = 0
    ram_total_mb: int = 0
    external_gpu_fraction: float = 0.0  # cached from the 30s auto-detect loop
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 30 pytest packages/nerd_herd/tests/test_snapshot_fields.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/types.py packages/nerd_herd/tests/test_snapshot_fields.py
git commit -m "feat(nerd_herd): add desktop-awareness fields to SystemSnapshot"
```

---

### Task 2: `PresenceCollector` (input-idle + fullscreen)

**Files:**
- Create: `packages/nerd_herd/src/nerd_herd/presence.py`
- Test: `packages/nerd_herd/tests/test_presence.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/nerd_herd/tests/test_presence.py
from nerd_herd.presence import PresenceCollector


def test_degrades_to_away_when_apis_unavailable(monkeypatch):
    c = PresenceCollector()
    # Force the platform probes to raise — collector must never throw.
    monkeypatch.setattr(c, "_idle_seconds_impl", lambda: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(c, "_fullscreen_impl", lambda: (_ for _ in ()).throw(OSError()))
    state = c.collect()
    assert state["user_idle_s"] >= 1e8     # treated as away
    assert state["foreground_fullscreen"] is False


def test_collect_returns_floats_and_bool(monkeypatch):
    c = PresenceCollector()
    monkeypatch.setattr(c, "_idle_seconds_impl", lambda: 12.5)
    monkeypatch.setattr(c, "_fullscreen_impl", lambda: True)
    state = c.collect()
    assert state["user_idle_s"] == 12.5
    assert state["foreground_fullscreen"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 pytest packages/nerd_herd/tests/test_presence.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the collector**

```python
# packages/nerd_herd/src/nerd_herd/presence.py
"""User-presence collector — input-idle time + foreground-fullscreen.

Windows-only signals via ctypes; degrades to "away, not fullscreen" on any
other platform or any error. NEVER raises — presence sensing must not break
the snapshot hot path.
"""
from __future__ import annotations

import sys

from yazbunu import get_logger

logger = get_logger("nerd_herd.presence")

AWAY_IDLE_S = 1e9  # sentinel "no user" idle value used on failure/off-Windows


class PresenceCollector:
    name = "presence"

    def __init__(self) -> None:
        self._is_windows = sys.platform == "win32"

    # ── Public ──────────────────────────────────────────────────────
    def collect(self) -> dict:
        """Return {'user_idle_s': float, 'foreground_fullscreen': bool}.

        Cheap (~1-2ms): two WinAPI calls. Safe to call per-snapshot.
        """
        try:
            idle = float(self._idle_seconds_impl())
        except Exception as e:
            logger.debug("presence idle probe failed", error=str(e))
            idle = AWAY_IDLE_S
        try:
            full = bool(self._fullscreen_impl())
        except Exception as e:
            logger.debug("presence fullscreen probe failed", error=str(e))
            full = False
        return {"user_idle_s": idle, "foreground_fullscreen": full}

    # ── Platform impls (patched in tests) ───────────────────────────
    def _idle_seconds_impl(self) -> float:
        if not self._is_windows:
            return AWAY_IDLE_S
        import ctypes
        from ctypes import wintypes

        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

        info = LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(LASTINPUTINFO)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            raise OSError("GetLastInputInfo failed")
        millis_now = ctypes.windll.kernel32.GetTickCount()
        # GetTickCount wraps every ~49.7 days; clamp negatives to 0.
        return max(0.0, (millis_now - info.dwTime) / 1000.0)

    def _fullscreen_impl(self) -> bool:
        if not self._is_windows:
            return False
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return False
        # Ignore the desktop / shell window (no app is fullscreen).
        if hwnd in (user32.GetDesktopWindow(), user32.GetShellWindow()):
            return False
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return False
        # Compare against the primary screen metrics (SM_CXSCREEN=0, SM_CYSCREEN=1).
        screen_w = user32.GetSystemMetrics(0)
        screen_h = user32.GetSystemMetrics(1)
        win_w = rect.right - rect.left
        win_h = rect.bottom - rect.top
        return win_w >= screen_w and win_h >= screen_h
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 30 pytest packages/nerd_herd/tests/test_presence.py -v`
Expected: PASS.

- [ ] **Step 5: Smoke import**

Run: `python -c "from nerd_herd.presence import PresenceCollector; print(PresenceCollector().collect())"`
Expected: prints a dict (real values on Windows, away-defaults elsewhere). No exception.

- [ ] **Step 6: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/presence.py packages/nerd_herd/tests/test_presence.py
git commit -m "feat(nerd_herd): PresenceCollector (input-idle + foreground-fullscreen)"
```

---

### Task 3: Cache external-GPU fraction in `LoadManager`; expose raw VRAM

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/load.py`
- Test: `packages/nerd_herd/tests/test_load_external_cache.py`

Rationale: `detect_external_gpu_usage()` (pynvml ~10-20ms) already runs every 30s in `_auto_detect_loop` but the result is discarded. Cache it so `snapshot()` can read it without re-probing on the per-admission hot path.

- [ ] **Step 1: Write the failing test**

```python
# packages/nerd_herd/tests/test_load_external_cache.py
from nerd_herd.load import LoadManager


class _FakeGPU:
    def __init__(self, frac): self._frac = frac
    def gpu_state(self):
        from nerd_herd.types import GPUState
        return GPUState(available=True, vram_total_mb=8000, vram_free_mb=8000)
    def detect_external_gpu_usage(self):
        from nerd_herd.types import ExternalGPUUsage
        return ExternalGPUUsage(detected=True, external_vram_mb=int(8000*self._frac),
                                total_vram_mb=8000)


def test_external_fraction_defaults_zero_before_first_detect():
    lm = LoadManager(gpu_collector=_FakeGPU(0.4))
    assert lm.get_external_gpu_fraction() == 0.0


def test_record_external_updates_cache():
    lm = LoadManager(gpu_collector=_FakeGPU(0.4))
    lm._record_external(_FakeGPU(0.4).detect_external_gpu_usage())
    assert abs(lm.get_external_gpu_fraction() - 0.4) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 pytest packages/nerd_herd/tests/test_load_external_cache.py -v`
Expected: FAIL — `get_external_gpu_fraction`/`_record_external` absent.

- [ ] **Step 3: Implement cache + accessor; wire into the loop**

In `packages/nerd_herd/src/nerd_herd/load.py`:

(a) In `__init__`, after `self._detect_task = None`, add:
```python
        self._last_external_fraction: float = 0.0
```

(b) Add methods after `get_vram_budget_mb`:
```python
    def _record_external(self, ext) -> None:
        """Cache the latest external-GPU fraction from the auto-detect loop."""
        try:
            self._last_external_fraction = float(ext.external_vram_fraction)
        except Exception:
            self._last_external_fraction = 0.0

    def get_external_gpu_fraction(self) -> float:
        """Last external-GPU fraction seen by the 30s auto-detect loop.

        Read by snapshot() / S13 — cheap, no pynvml probe on the hot path.
        """
        return self._last_external_fraction
```

(c) In `_auto_detect_loop`, immediately after `ext = self._gpu.detect_external_gpu_usage()`:
```python
                self._record_external(ext)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 30 pytest packages/nerd_herd/tests/test_load_external_cache.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/load.py packages/nerd_herd/tests/test_load_external_cache.py
git commit -m "feat(nerd_herd): cache external-GPU fraction from auto-detect loop"
```

---

### Task 4: Populate desktop fields in `NerdHerd.snapshot()`

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/nerd_herd.py` (`__init__` + `snapshot`)
- Test: `packages/nerd_herd/tests/test_snapshot_populates_desktop.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/nerd_herd/tests/test_snapshot_populates_desktop.py
from nerd_herd.nerd_herd import NerdHerd


def test_snapshot_carries_desktop_signals(monkeypatch):
    nh = NerdHerd(metrics_port=0)
    # Stub the collectors so the test is hardware-independent.
    monkeypatch.setattr(nh._presence, "collect",
                        lambda: {"user_idle_s": 5.0, "foreground_fullscreen": True})
    from nerd_herd.types import SystemState, GPUState
    monkeypatch.setattr(nh._gpu, "system_state",
                        lambda: SystemState(ram_total_mb=32000, ram_available_mb=4000))
    monkeypatch.setattr(nh._load, "get_external_gpu_fraction", lambda: 0.7)
    nh._load.set_load_mode("shared", source="user")

    snap = nh.snapshot()
    assert snap.load_mode == "shared"
    assert snap.user_idle_s == 5.0
    assert snap.foreground_fullscreen is True
    assert snap.ram_available_mb == 4000
    assert snap.ram_total_mb == 32000
    assert snap.external_gpu_fraction == 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 pytest packages/nerd_herd/tests/test_snapshot_populates_desktop.py -v`
Expected: FAIL — `nh._presence` missing and/or snapshot fields unset.

- [ ] **Step 3: Register the collector and populate the snapshot**

In `packages/nerd_herd/src/nerd_herd/nerd_herd.py`:

(a) Add import at top:
```python
from nerd_herd.presence import PresenceCollector
```

(b) In `__init__`, after the GPU collector registration (`self.registry.register("gpu", self._gpu)`):
```python
        self._presence = PresenceCollector()
        self.registry.register("presence", self._presence)
```

(c) In `snapshot()`, replace the `return SystemSnapshot(...)` block with:
```python
        presence = self._presence.collect()
        sysstate = self._gpu.system_state()
        return SystemSnapshot(
            vram_available_mb=self.get_vram_budget_mb() if gpu.available else 0,
            local=local,
            cloud=dict(self._cloud_state),
            queue_profile=self._queue_profile,
            in_flight_calls=list(self._in_flight_calls),
            recent_swap_count=self._swap_budget.recent_count(),
            load_mode=self._load.get_load_mode(),
            user_idle_s=float(presence.get("user_idle_s", 1e9)),
            foreground_fullscreen=bool(presence.get("foreground_fullscreen", False)),
            ram_available_mb=int(sysstate.ram_available_mb),
            ram_total_mb=int(sysstate.ram_total_mb),
            external_gpu_fraction=float(self._load.get_external_gpu_fraction()),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 30 pytest packages/nerd_herd/tests/test_snapshot_populates_desktop.py -v`
Expected: PASS.

- [ ] **Step 5: Smoke + regression**

Run: `python -c "from nerd_herd.nerd_herd import NerdHerd; print(NerdHerd(metrics_port=0).snapshot().load_mode)"`
Expected: prints `full`.
Run: `timeout 60 pytest packages/nerd_herd/tests/ -q`
Expected: PASS (no regression in existing snapshot/client tests).

- [ ] **Step 6: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/nerd_herd.py packages/nerd_herd/tests/test_snapshot_populates_desktop.py
git commit -m "feat(nerd_herd): populate desktop signals in snapshot()"
```

---

## Phase 2 — The two signals

### Task 5: S12 user-presence signal

**Files:**
- Create: `packages/nerd_herd/src/nerd_herd/signals/s12_presence.py`
- Test: `packages/nerd_herd/tests/test_s12_presence.py`

Contract: local-only, negative-only. Fullscreen present → −10 sentinel. Present-normal → graded −0.3…−0.6 by recency. Away (idle ≥ AWAY_IDLE_S) → 0.0. Cloud → 0.0.

- [ ] **Step 1: Write the failing test**

```python
# packages/nerd_herd/tests/test_s12_presence.py
from types import SimpleNamespace
from nerd_herd.signals.s12_presence import s12_presence, PRESENT_IDLE_S

LOCAL = SimpleNamespace(is_local=True)
CLOUD = SimpleNamespace(is_local=False)


def test_cloud_always_zero():
    assert s12_presence(CLOUD, user_idle_s=0.0, foreground_fullscreen=True) == 0.0


def test_away_is_zero():
    assert s12_presence(LOCAL, user_idle_s=10_000.0, foreground_fullscreen=False) == 0.0


def test_fullscreen_hard_veto():
    assert s12_presence(LOCAL, user_idle_s=1.0, foreground_fullscreen=True) == -10.0


def test_present_normal_is_graded_negative():
    v = s12_presence(LOCAL, user_idle_s=1.0, foreground_fullscreen=False)
    assert -0.6 <= v <= -0.3


def test_just_past_present_threshold_fades_to_zero():
    # At exactly PRESENT_IDLE_S the penalty has decayed to ~0.
    v = s12_presence(LOCAL, user_idle_s=PRESENT_IDLE_S, foreground_fullscreen=False)
    assert -0.05 <= v <= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 pytest packages/nerd_herd/tests/test_s12_presence.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement S12**

```python
# packages/nerd_herd/src/nerd_herd/signals/s12_presence.py
"""S12 — user-presence pressure (local pool only).

Negative-only "yield the machine to the human" signal. Cloud models are
unaffected (zero desktop impact). Tuning constants are starting guesses —
revisit against real kutai.jsonl idle distributions (spec §7).
"""
from __future__ import annotations

from typing import Any

# Below this idle, the user is "actively present" and gets the full penalty;
# the penalty decays linearly to 0 as idle approaches PRESENT_IDLE_S ("away").
ACTIVE_IDLE_S = 30.0
PRESENT_IDLE_S = 300.0          # >= this → away → 0.0
PRESENT_PENALTY = -0.6          # strongest graded penalty while actively present
PRESENT_PENALTY_FLOOR = -0.3    # penalty at the active/away boundary's near edge
FULLSCREEN_VETO = -10.0         # sentinel — survives M3/M4 weights, pegs scalar -1.0


def s12_presence(model: Any, *, user_idle_s: float, foreground_fullscreen: bool) -> float:
    if not getattr(model, "is_local", False):
        return 0.0
    if user_idle_s >= PRESENT_IDLE_S:
        return 0.0
    if foreground_fullscreen:
        return FULLSCREEN_VETO
    # Graded: PRESENT_PENALTY when fully active, decaying linearly to ~0 at
    # PRESENT_IDLE_S. Clamp the "fully active" plateau below ACTIVE_IDLE_S.
    if user_idle_s <= ACTIVE_IDLE_S:
        return PRESENT_PENALTY
    span = PRESENT_IDLE_S - ACTIVE_IDLE_S
    decayed = PRESENT_PENALTY * (1.0 - (user_idle_s - ACTIVE_IDLE_S) / span)
    return max(PRESENT_PENALTY, min(0.0, decayed))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 30 pytest packages/nerd_herd/tests/test_s12_presence.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/signals/s12_presence.py packages/nerd_herd/tests/test_s12_presence.py
git commit -m "feat(nerd_herd): S12 user-presence pressure signal"
```

---

### Task 6: S13 machine-contention signal

**Files:**
- Create: `packages/nerd_herd/src/nerd_herd/signals/s13_contention.py`
- Test: `packages/nerd_herd/tests/test_s13_contention.py`

Contract: local-only, negative-only. external-GPU heavy → −10 sentinel. RAM pressure → graded negative on `% used` above a floor. Cloud → 0.0.

- [ ] **Step 1: Write the failing test**

```python
# packages/nerd_herd/tests/test_s13_contention.py
from types import SimpleNamespace
from nerd_herd.signals.s13_contention import s13_contention

LOCAL = SimpleNamespace(is_local=True)
CLOUD = SimpleNamespace(is_local=False)


def test_cloud_always_zero():
    assert s13_contention(CLOUD, ram_available_mb=10, ram_total_mb=32000,
                          external_gpu_fraction=0.9) == 0.0


def test_external_gpu_heavy_hard_veto():
    assert s13_contention(LOCAL, ram_available_mb=16000, ram_total_mb=32000,
                          external_gpu_fraction=0.7) == -10.0


def test_low_ram_pressure_is_zero():
    # 50% used (16000/32000 avail) → below the 0.80 floor → 0.0
    assert s13_contention(LOCAL, ram_available_mb=16000, ram_total_mb=32000,
                          external_gpu_fraction=0.0) == 0.0


def test_high_ram_pressure_graded_negative():
    # ~94% used (2000/32000 avail) → between floor and cap → graded negative
    v = s13_contention(LOCAL, ram_available_mb=2000, ram_total_mb=32000,
                       external_gpu_fraction=0.0)
    assert -1.0 < v < 0.0


def test_critical_ram_pegs_minus_one():
    # ~98% used → at/above cap → -1.0
    v = s13_contention(LOCAL, ram_available_mb=300, ram_total_mb=32000,
                       external_gpu_fraction=0.0)
    assert v == -1.0


def test_zero_total_ram_is_safe():
    assert s13_contention(LOCAL, ram_available_mb=0, ram_total_mb=0,
                          external_gpu_fraction=0.0) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 pytest packages/nerd_herd/tests/test_s13_contention.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement S13**

```python
# packages/nerd_herd/src/nerd_herd/signals/s13_contention.py
"""S13 — machine-contention pressure (local pool only).

Negative-only. Two inputs, both already collected elsewhere:
  - external-GPU fraction (cached from the 30s auto-detect loop)
  - RAM pressure (psutil via SystemState)
Can fire while the user is away (e.g. an overnight render owns the GPU).
Tuning constants are starting guesses (spec §7).
"""
from __future__ import annotations

from typing import Any

EXTERNAL_GPU_VETO_FRACTION = 0.30   # another process owns >=30% VRAM → veto local
EXTERNAL_GPU_VETO = -10.0           # sentinel
RAM_USED_FLOOR = 0.80               # below this used-fraction → no pressure
RAM_USED_CAP = 0.95                 # at/above → full -1.0


def s13_contention(
    model: Any,
    *,
    ram_available_mb: int,
    ram_total_mb: int,
    external_gpu_fraction: float,
) -> float:
    if not getattr(model, "is_local", False):
        return 0.0
    if external_gpu_fraction >= EXTERNAL_GPU_VETO_FRACTION:
        return EXTERNAL_GPU_VETO
    if ram_total_mb <= 0:
        return 0.0
    used_frac = 1.0 - (ram_available_mb / ram_total_mb)
    if used_frac < RAM_USED_FLOOR:
        return 0.0
    if used_frac >= RAM_USED_CAP:
        return -1.0
    # Linear ramp floor→cap mapped to 0→-1.
    intensity = (used_frac - RAM_USED_FLOOR) / (RAM_USED_CAP - RAM_USED_FLOOR)
    return max(-1.0, min(0.0, -intensity))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 30 pytest packages/nerd_herd/tests/test_s13_contention.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/signals/s13_contention.py packages/nerd_herd/tests/test_s13_contention.py
git commit -m "feat(nerd_herd): S13 machine-contention pressure signal"
```

---

### Task 7: Add S12/S13 to the combine fold

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/combine.py:17`
- Test: `packages/nerd_herd/tests/test_combine_desktop.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/nerd_herd/tests/test_combine_desktop.py
from nerd_herd.combine import combine_signals, OTHER_BUCKET


def test_s12_s13_in_other_bucket():
    assert "S12" in OTHER_BUCKET
    assert "S13" in OTHER_BUCKET


def test_s12_graded_negative_flows_through():
    sig = {k: 0.0 for k in ("S1","S2","S3","S4","S5","S6","S7","S9","S10","S11","S12","S13")}
    sig["S12"] = -0.6
    out = combine_signals(signals=sig, weights={})
    assert out.scalar < 0.0


def test_s13_sentinel_pegs_minus_one():
    sig = {k: 0.0 for k in ("S1","S2","S3","S4","S5","S6","S7","S9","S10","S11","S12","S13")}
    sig["S13"] = -10.0
    out = combine_signals(signals=sig, weights={})
    assert out.scalar == -1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 pytest packages/nerd_herd/tests/test_combine_desktop.py -v`
Expected: FAIL — S12/S13 not in `OTHER_BUCKET`.

- [ ] **Step 3: Add S12/S13 to OTHER_BUCKET**

In `packages/nerd_herd/src/nerd_herd/combine.py`, change line 17:
```python
OTHER_BUCKET = ("S1", "S7", "S9", "S10", "S11")
```
to:
```python
OTHER_BUCKET = ("S1", "S7", "S9", "S10", "S11", "S12", "S13")
```

(No other change: `combine_signals` already iterates `OTHER_BUCKET + ...`, takes worst-of-negatives per bucket, and `weights.get(k, 1.0)` defaults missing weights to 1.0. S12/S13 are negative-only so they never enter the `POSITIVE_ARM_SIGNALS` noisy-OR.)

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 30 pytest packages/nerd_herd/tests/test_combine_desktop.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/combine.py packages/nerd_herd/tests/test_combine_desktop.py
git commit -m "feat(nerd_herd): fold S12/S13 into OTHER_BUCKET"
```

---

## Phase 3 — M4 load-mode modifier + pressure_for integration

### Task 8: `M4_load_mode_weights`

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/modifiers.py`
- Test: `packages/nerd_herd/tests/test_m4_load_mode.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/nerd_herd/tests/test_m4_load_mode.py
from nerd_herd.modifiers import M4_load_mode_weights


def test_full_silences_desktop_signals():
    w = M4_load_mode_weights(mode="full")
    assert w["S12"] == 0.0 and w["S13"] == 0.0


def test_otomatik_passthrough():
    # "full" is the auto-managed default; "heavy"/"shared" express strength.
    w = M4_load_mode_weights(mode="heavy")
    assert w["S12"] >= 1.0 and w["S13"] >= 1.0


def test_shared_amplifies_more_than_heavy():
    assert M4_load_mode_weights(mode="shared")["S12"] > M4_load_mode_weights(mode="heavy")["S12"]


def test_unknown_mode_is_passthrough():
    w = M4_load_mode_weights(mode="bogus")
    assert w["S12"] == 1.0 and w["S13"] == 1.0


def test_only_touches_s12_s13():
    w = M4_load_mode_weights(mode="shared")
    assert set(w.keys()) == {"S12", "S13"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 pytest packages/nerd_herd/tests/test_m4_load_mode.py -v`
Expected: FAIL — function missing.

- [ ] **Step 3: Implement M4**

In `packages/nerd_herd/src/nerd_herd/modifiers.py`, append:
```python
# ── M4: Load-mode weights on desktop signals (S12/S13) ─────────────
# Re-expresses "yük modu" as per-signal weights, replacing the dead
# VRAM-% cap. Minimal is handled upstream by selector eligibility
# (load_mode_minimal), so it doesn't need a veto weight here — passthrough.
_M4_BY_MODE: dict[str, float] = {
    "full": 0.0,      # ignore the user — desktop signals silenced
    "heavy": 1.5,     # cloud-bias strength: amplify desktop penalty
    "shared": 2.0,    # stronger cloud bias
    "minimal": 1.0,   # local already vetoed at eligibility; passthrough
}


def M4_load_mode_weights(*, mode: str) -> dict[str, float]:
    """Per-signal weights for S12/S13 driven by load mode. Multiplied into
    the M3 weight dict before the fold. Unknown mode → passthrough (1.0)."""
    factor = _M4_BY_MODE.get(mode, 1.0)
    return {"S12": factor, "S13": factor}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 30 pytest packages/nerd_herd/tests/test_m4_load_mode.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/modifiers.py packages/nerd_herd/tests/test_m4_load_mode.py
git commit -m "feat(nerd_herd): M4 load-mode weights for S12/S13"
```

---

### Task 9: Wire S12/S13 + M4 into `pressure_for`

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/types.py` (`pressure_for`, ~257-404)
- Test: `packages/nerd_herd/tests/test_pressure_desktop_integration.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/nerd_herd/tests/test_pressure_desktop_integration.py
from types import SimpleNamespace
from nerd_herd.types import SystemSnapshot, LocalModelState

LOCAL = SimpleNamespace(name="loc", is_local=True, is_loaded=True, is_free=False,
                        provider="", cap_score=5.0, size_mb=4000)


def _snap(**kw):
    base = dict(vram_available_mb=8000,
                local=LocalModelState(model_name="loc", idle_seconds=120.0))
    base.update(kw)
    return SystemSnapshot(**base)


def test_away_full_mode_no_desktop_pressure():
    # away (idle huge) + full mode → S12/S13 contribute nothing
    snap = _snap(user_idle_s=1e9, load_mode="full")
    assert snap.pressure_for(LOCAL).scalar >= 0.0


def test_present_otomatik_pushes_local_negative():
    snap = _snap(user_idle_s=1.0, load_mode="heavy")  # present + amplified
    assert snap.pressure_for(LOCAL).scalar < 0.0


def test_full_mode_silences_presence_even_when_present():
    snap = _snap(user_idle_s=1.0, load_mode="full")
    # S12 would be -0.6 but M4 weight 0.0 zeroes it
    assert snap.pressure_for(LOCAL).scalar >= 0.0


def test_fullscreen_pegs_minus_one_in_otomatik():
    snap = _snap(user_idle_s=1.0, foreground_fullscreen=True, load_mode="heavy")
    assert snap.pressure_for(LOCAL).scalar == -1.0


def test_external_gpu_veto_in_otomatik():
    snap = _snap(user_idle_s=1e9, external_gpu_fraction=0.7, load_mode="heavy")
    assert snap.pressure_for(LOCAL).scalar == -1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 pytest packages/nerd_herd/tests/test_pressure_desktop_integration.py -v`
Expected: FAIL — S12/S13 not computed; full mode doesn't silence.

- [ ] **Step 3: Compute S12/S13 and apply M4 in `pressure_for`**

In `packages/nerd_herd/src/nerd_herd/types.py`, inside `pressure_for`:

(a) Add imports next to the other signal imports:
```python
        from nerd_herd.signals.s12_presence import s12_presence
        from nerd_herd.signals.s13_contention import s13_contention
        from nerd_herd.modifiers import M4_load_mode_weights
```

(b) In the `sig = { ... }` dict, after the `"S11": ...` entry, add:
```python
            "S12": s12_presence(
                model,
                user_idle_s=self.user_idle_s,
                foreground_fullscreen=self.foreground_fullscreen,
            ),
            "S13": s13_contention(
                model,
                ram_available_mb=self.ram_available_mb,
                ram_total_mb=self.ram_total_mb,
                external_gpu_fraction=self.external_gpu_fraction,
            ),
```

(c) After `weights = M3_difficulty_weights(...)` (the existing M3 call), merge M4:
```python
        # M4: load-mode weights on the desktop signals only. Merge over M3
        # (which never defines S12/S13), so the multiply is a clean overlay.
        weights.update(M4_load_mode_weights(mode=self.load_mode))
```

(d) Update the docstring count line `"""Compute pressure breakdown via 10 signals + 4 modifiers."""` → `"""Compute pressure breakdown via 12 signals + 4 modifiers."""`.

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 30 pytest packages/nerd_herd/tests/test_pressure_desktop_integration.py -v`
Expected: PASS.

- [ ] **Step 5: Regression — existing pressure tests unchanged**

Run: `timeout 120 pytest packages/nerd_herd/tests/ -q`
Expected: PASS. (Manually-built `SystemSnapshot`s default to away/full → S12/S13 = 0 → identical scalars to before.)

- [ ] **Step 6: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/types.py packages/nerd_herd/tests/test_pressure_desktop_integration.py
git commit -m "feat(nerd_herd): wire S12/S13 + M4 into pressure_for"
```

---

### Task 10: Re-run the Fatih simulators (no-regression gate)

**Files:**
- Run-only: `packages/fatih_hoca/tests/sim/run_scenarios.py`, `packages/fatih_hoca/tests/sim/run_swap_storm_check.py`

- [ ] **Step 1: Run the scenario simulator**

Run: `timeout 120 python packages/fatih_hoca/tests/sim/run_scenarios.py`
Expected: same pass/scenario summary as before this branch (desktop fields default to away/full in the sim snapshots, so picks must be unchanged). If any scenario flips, STOP — a default leaked non-zero desktop pressure; re-check Task 1 defaults.

- [ ] **Step 2: Run the swap-storm check**

Run: `timeout 120 python packages/fatih_hoca/tests/sim/run_swap_storm_check.py`
Expected: unchanged result.

- [ ] **Step 3: Commit (if either harness writes a golden/snapshot artifact)**

```bash
git add -A packages/fatih_hoca/tests/sim/
git commit -m "test(fatih_hoca): re-run sims after desktop signals — no pick regressions" || echo "nothing to commit"
```

---

## Phase 4 — need_ctx ownership moves to Fatih

### Task 11: `compute_need_ctx` helper

**Files:**
- Create: `packages/fatih_hoca/src/fatih_hoca/need_ctx.py`
- Test: `packages/fatih_hoca/tests/test_need_ctx.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_need_ctx.py
from fatih_hoca.need_ctx import compute_need_ctx, MIN_CTX


def test_floors_to_min_ctx_when_unknown():
    assert compute_need_ctx(min_context=0, est_in=0, est_out=0, model_ctx=128000) == MIN_CTX


def test_small_need_floored():
    # 4412 genuine need → still floored to MIN_CTX (8192)
    assert compute_need_ctx(min_context=4412, est_in=0, est_out=0, model_ctx=128000) == MIN_CTX


def test_ceils_to_2048_block():
    # 18000 → ceil to 18432
    assert compute_need_ctx(min_context=18000, est_in=0, est_out=0, model_ctx=128000) == 18432


def test_estimate_used_when_no_min_context():
    # (est_in+est_out)*1.3+512 = (4000+4000)*1.3+512 = 10912 → ceil2048 → 12288
    assert compute_need_ctx(min_context=0, est_in=4000, est_out=4000, model_ctx=128000) == 12288


def test_clamped_to_model_ceiling():
    assert compute_need_ctx(min_context=40000, est_in=0, est_out=0, model_ctx=8192) == 8192
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 pytest packages/fatih_hoca/tests/test_need_ctx.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the helper**

```python
# packages/fatih_hoca/src/fatih_hoca/need_ctx.py
"""Single owner of local-model context sizing (moved out of the dispatcher).

need_ctx = clamp(ceil_2048(min_context or estimate or MIN_CTX), MIN_CTX, model_ctx)

MIN_CTX=8192 is evidence-backed (kutai.jsonl 05-29/30: smallest genuine task
need 4412; bottom cluster 4412-10207; 8192 covers it with margin). Override
via env LLAMA_MIN_CTX.
"""
from __future__ import annotations

import os

MIN_CTX = int(os.environ.get("LLAMA_MIN_CTX", "8192"))
_BLOCK = 2048


def _ceil_block(n: int) -> int:
    if n <= 0:
        return 0
    return ((n + _BLOCK - 1) // _BLOCK) * _BLOCK


def compute_need_ctx(*, min_context: int, est_in: int, est_out: int, model_ctx: int) -> int:
    """Return the exact context window to load for a local model."""
    need = min_context
    if need <= 0 and (est_in or est_out):
        need = int((est_in + est_out) * 1.3) + 512
    need = _ceil_block(need) if need > 0 else MIN_CTX
    need = max(MIN_CTX, need)
    if model_ctx > 0:
        need = min(need, model_ctx)
    return need
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 30 pytest packages/fatih_hoca/tests/test_need_ctx.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/need_ctx.py packages/fatih_hoca/tests/test_need_ctx.py
git commit -m "feat(fatih_hoca): compute_need_ctx — single owner of context sizing"
```

---

### Task 12: Add `need_ctx` to `Pick` and set it in `select()`

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/types.py` (`Pick`)
- Modify: `packages/fatih_hoca/src/fatih_hoca/selector.py` (the `return Pick(...)` at ~426)
- Test: `packages/fatih_hoca/tests/test_pick_need_ctx.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_pick_need_ctx.py
from fatih_hoca.types import Pick


def test_pick_has_need_ctx_default_zero():
    p = Pick(model=object(), min_time_seconds=1.0)
    assert p.need_ctx == 0
```

(Selector-level wiring is covered by integration in Task 13; this unit pins the field.)

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 pytest packages/fatih_hoca/tests/test_pick_need_ctx.py -v`
Expected: FAIL — `need_ctx` absent.

- [ ] **Step 3a: Add the field to `Pick`**

In `packages/fatih_hoca/src/fatih_hoca/types.py`, inside `@dataclass class Pick`, after `top_summary: str = ""`:
```python
    # Exact local context window to load (ceil-2048, floored MIN_CTX, capped
    # to the model's trained window). Computed by select() from the task
    # estimates + chosen model. The dispatcher loads at this value instead
    # of deriving ctx itself. 0 for cloud picks / legacy callers.
    need_ctx: int = 0
```

- [ ] **Step 3b: Set `need_ctx` in `select()`**

In `packages/fatih_hoca/src/fatih_hoca/selector.py`, add the import near the top:
```python
from fatih_hoca.need_ctx import compute_need_ctx
```
Then change the final `return Pick(...)` (~426) to compute and pass `need_ctx`:
```python
        need_ctx = 0
        if getattr(best.model, "is_local", False):
            need_ctx = compute_need_ctx(
                min_context=reqs.effective_context_needed or min_context_length,
                est_in=estimated_input_tokens,
                est_out=estimated_output_tokens,
                model_ctx=getattr(best.model, "context_length", 0),
            )
        return Pick(
            model=best.model,
            min_time_seconds=min_time,
            estimated_load_seconds=load_time,
            score=float(best.score),
            top_summary=top_summary,
            need_ctx=need_ctx,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 30 pytest packages/fatih_hoca/tests/test_pick_need_ctx.py -v`
Expected: PASS.
Run: `python -c "from fatih_hoca.selector import Selector"`  (import smoke)
Expected: no error.

- [ ] **Step 5: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/types.py packages/fatih_hoca/src/fatih_hoca/selector.py packages/fatih_hoca/tests/test_pick_need_ctx.py
git commit -m "feat(fatih_hoca): Pick.need_ctx computed in select()"
```

---

### Task 13: Dispatcher consumes `pick.need_ctx`

**Files:**
- Modify: `src/core/llm_dispatcher.py:362-364`
- Test: `tests/core/test_dispatcher_need_ctx.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_dispatcher_need_ctx.py
# Verifies the dispatcher prefers pick.need_ctx over the legacy heuristic.
# Pure-logic extraction test: the resolution rule is `pick.need_ctx if >0
# else heuristic`. We test that rule against the dispatcher's helper.
from src.core.llm_dispatcher import _resolve_load_ctx


def test_prefers_pick_need_ctx():
    assert _resolve_load_ctx(need_ctx=18432, min_context=0, est_in=0, est_out=0) == 18432


def test_falls_back_to_min_context():
    assert _resolve_load_ctx(need_ctx=0, min_context=16384, est_in=0, est_out=0) == 16384


def test_falls_back_to_heuristic():
    # (1000+1000)*1.3+512 = 3112
    assert _resolve_load_ctx(need_ctx=0, min_context=0, est_in=1000, est_out=1000) == 3112
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 pytest tests/core/test_dispatcher_need_ctx.py -v`
Expected: FAIL — `_resolve_load_ctx` missing.

- [ ] **Step 3: Extract the resolver and call it**

In `src/core/llm_dispatcher.py`, add a module-level helper (near the top, after imports):
```python
def _resolve_load_ctx(*, need_ctx: int, min_context: int, est_in: int, est_out: int) -> int:
    """Pick's need_ctx wins; fall back to min_context, then the legacy
    token heuristic. Kept as a thin fallback for cloud picks / tests that
    don't carry need_ctx."""
    if need_ctx and need_ctx > 0:
        return need_ctx
    if min_context and min_context > 0:
        return min_context
    if est_in or est_out:
        return int((est_in + est_out) * 1.3) + 512
    return 0
```
Then replace the body at lines 362-364:
```python
                _min_ctx = min_context
                if _min_ctx <= 0 and (estimated_input_tokens or estimated_output_tokens):
                    _min_ctx = int((estimated_input_tokens + estimated_output_tokens) * 1.3) + 512
```
with:
```python
                _min_ctx = _resolve_load_ctx(
                    need_ctx=int(getattr(pick, "need_ctx", 0) or 0),
                    min_context=min_context,
                    est_in=estimated_input_tokens,
                    est_out=estimated_output_tokens,
                )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 30 pytest tests/core/test_dispatcher_need_ctx.py -v`
Expected: PASS.
Run: `python -c "import src.core.llm_dispatcher"`  (import smoke)
Expected: no error.

- [ ] **Step 5: Commit**

```bash
git add src/core/llm_dispatcher.py tests/core/test_dispatcher_need_ctx.py
git commit -m "feat(dispatcher): consume pick.need_ctx for local load sizing"
```

---

## Phase 5 — Minimal-mode eligibility

### Task 14: `load_mode_minimal` local eligibility gate

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/selector.py` (`_check_eligibility`, the local block ~627-634)
- Test: `packages/fatih_hoca/tests/test_minimal_mode_eligibility.py`

- [ ] **Step 1: Write the failing test**

```python
# packages/fatih_hoca/tests/test_minimal_mode_eligibility.py
from types import SimpleNamespace
from fatih_hoca.selector import Selector
from fatih_hoca.requirements import ModelRequirements


def _local_model():
    return SimpleNamespace(
        name="loc", litellm_name="loc", is_local=True, is_loaded=True,
        demoted=False, provider="", specialty=None, context_length=32000,
        supports_function_calling=True, supports_json_mode=True, has_vision=False,
        variant_flags=set(), max_input_tokens=None, rate_limit_tpm=0,
    )


def _selector():
    reg = SimpleNamespace(is_dead=lambda *_: False, is_provider_dead=lambda *_: False)
    return Selector(registry=reg, nerd_herd=SimpleNamespace())


def test_minimal_mode_rejects_local():
    sel = _selector()
    snap = SimpleNamespace(load_mode="minimal", vram_available_mb=8000, cloud={})
    reqs = ModelRequirements(task="coder", difficulty=5)
    reason = sel._check_eligibility(model=_local_model(), reqs=reqs,
                                    failed_models=set(), snapshot=snap)
    assert reason == "load_mode_minimal"


def test_full_mode_allows_local():
    sel = _selector()
    snap = SimpleNamespace(load_mode="full", vram_available_mb=8000, cloud={})
    reqs = ModelRequirements(task="coder", difficulty=5)
    reason = sel._check_eligibility(model=_local_model(), reqs=reqs,
                                    failed_models=set(), snapshot=snap)
    assert reason is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 pytest packages/fatih_hoca/tests/test_minimal_mode_eligibility.py -v`
Expected: FAIL — `test_minimal_mode_rejects_local` gets `None`.

- [ ] **Step 3: Add the gate**

In `packages/fatih_hoca/src/fatih_hoca/selector.py`, inside `_check_eligibility`, in the local block (the `if model.is_local:` near line 627), BEFORE the `vram_available` check:
```python
        # Local inference allowed check
        if model.is_local:
            # Minimal load mode = cloud-only. Local is structurally
            # ineligible — clearer than a pressure veto and gives a
            # named diag reason. (resource-signals 2026-06-09)
            if getattr(snapshot, "load_mode", "full") == "minimal":
                return "load_mode_minimal"
            vram_available = getattr(snapshot, "vram_available_mb", 0)
            if vram_available == 0:
                return "no_vram_available"
```
(Replace the existing local block so the two checks live together; keep the existing `vram_available == 0` behavior.)

- [ ] **Step 4: Run test to verify it passes**

Run: `timeout 30 pytest packages/fatih_hoca/tests/test_minimal_mode_eligibility.py -v`
Expected: PASS.

- [ ] **Step 5: Regression — selector suite**

Run: `timeout 60 pytest packages/fatih_hoca/tests/ -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add packages/fatih_hoca/src/fatih_hoca/selector.py packages/fatih_hoca/tests/test_minimal_mode_eligibility.py
git commit -m "feat(fatih_hoca): minimal load mode rejects local at eligibility"
```

---

## Phase 6 — Delete the dead VRAM-cap machinery

> Each deletion is its own commit. Before each, grep to confirm no NEW live caller appeared since the 2026-05-31 audit. Audit call sites, not docstrings.

### Task 15: Stop capping snapshot VRAM (use raw free)

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/load.py` (`get_vram_budget_mb`)
- Test: `packages/nerd_herd/tests/test_load.py` (update existing expectations)

- [ ] **Step 1: Update/confirm the test for raw free VRAM**

```python
# add to packages/nerd_herd/tests/test_load.py
def test_vram_budget_mb_is_raw_free_regardless_of_mode():
    class _G:
        def gpu_state(self):
            from nerd_herd.types import GPUState
            return GPUState(available=True, vram_total_mb=8000, vram_free_mb=8000)
    lm = LoadManager(gpu_collector=_G())
    lm.set_load_mode("shared", source="user")   # would have been 0.5× before
    assert lm.get_vram_budget_mb() == 8000       # no cap now — placement, not capping
```

- [ ] **Step 2: Run test to verify it fails**

Run: `timeout 30 pytest packages/nerd_herd/tests/test_load.py::test_vram_budget_mb_is_raw_free_regardless_of_mode -v`
Expected: FAIL — returns 4000 (capped).

- [ ] **Step 3: Drop the cap from `get_vram_budget_mb`**

In `packages/nerd_herd/src/nerd_herd/load.py`, change:
```python
    def get_vram_budget_mb(self) -> int:
        gpu = self._gpu.gpu_state()
        return int(gpu.vram_free_mb * self.get_vram_budget_fraction())
```
to:
```python
    def get_vram_budget_mb(self) -> int:
        # Placement, not capping (resource-signals 2026-06-09): the
        # desktop signals + --fit own contention now. snapshot.vram_available_mb
        # reflects the true free VRAM, never a mode-scaled fraction.
        gpu = self._gpu.gpu_state()
        return int(gpu.vram_free_mb)
```
Leave `VRAM_BUDGETS`, `get_vram_budget_fraction`, `suggest_mode_for_external_usage` in place — still used by the auto-detect loop + Prometheus exposition.

- [ ] **Step 4: Run tests**

Run: `timeout 60 pytest packages/nerd_herd/tests/test_load.py -v`
Expected: PASS (fix any now-stale fraction-cap assertions in the same file to expect raw free).

- [ ] **Step 5: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/load.py packages/nerd_herd/tests/test_load.py
git commit -m "refactor(nerd_herd): snapshot VRAM is raw free, not a mode-scaled cap"
```

---

### Task 16: Delete `load_manager.py` sync stubs + router enforcement

**Files:**
- Modify: `src/infra/load_manager.py` (remove sync `is_local_inference_allowed` / `get_vram_budget_fraction`)
- Modify: `src/core/router.py:233-247,317` (remove the no-op enforcement block)

- [ ] **Step 1: Confirm no live callers**

Run: `rtk grep -n "is_local_inference_allowed\b" src/ packages/ | grep -v "_async" | grep -v test`
Run: `rtk grep -n "get_vram_budget_fraction\b" src/ | grep -v "_async" | grep -v test`
Expected: only `src/core/router.py` (the dead path) + the definitions. If anything else appears, STOP and reassess.

- [ ] **Step 2: Remove the router enforcement block**

In `src/core/router.py`, delete the load-mode enforcement at lines 233-247 and the `_vb = get_vram_budget_fraction()` usage at 317 (and any now-unused local var). Remove the corresponding imports from `src.infra.load_manager`.

- [ ] **Step 3: Remove the sync stubs**

In `src/infra/load_manager.py`, delete the sync `def is_local_inference_allowed()` and `def get_vram_budget_fraction()` (keep the `_async` versions and the `VRAM_BUDGETS` re-export used elsewhere).

- [ ] **Step 4: Smoke + targeted tests**

Run: `python -c "import src.core.router; import src.infra.load_manager"`
Expected: no error.
Run: `timeout 60 pytest tests/ -k "router or load_manager" -q`
Expected: PASS (delete/adjust tests that asserted the dead sync stubs).

- [ ] **Step 5: Commit**

```bash
git add src/core/router.py src/infra/load_manager.py tests/
git commit -m "refactor: delete dead load-mode sync stubs + router enforcement (no-op)"
```

---

### Task 17: Delete P1 dead context symbols

**Files:**
- Modify: `packages/fatih_hoca/src/fatih_hoca/registry.py` (`calculate_dynamic_context`, `vram_context_ceiling`)
- Modify: `src/models/local_model_manager.py` (`BASELINE_LOCAL_CTX`, `_floored_baseline_ctx`)
- Modify: `src/models/model_registry.py` (re-exports)
- Delete: `tests/test_local_ctx_floor.py`; ctx tests in `packages/fatih_hoca/tests/test_registry.py`
- Modify: `ranking.py:309-310` (delete the false dead comment)

- [ ] **Step 1: Confirm no live callers**

Run: `rtk grep -n "calculate_dynamic_context\|vram_context_ceiling\|BASELINE_LOCAL_CTX\|_floored_baseline_ctx" src/ packages/ | grep -v test`
Expected: only definitions + back-compat re-exports in `model_registry.py`. If a live caller appears, STOP.

- [ ] **Step 2: Delete the symbols + re-exports + their tests**

Remove the four symbols from their definition sites, the re-export lines in `src/models/model_registry.py`, delete `tests/test_local_ctx_floor.py`, and delete the ctx test functions in `packages/fatih_hoca/tests/test_registry.py` (the `test_calculate_dynamic_context*` / `test_vram_context_ceiling*` set). In `packages/fatih_hoca/src/fatih_hoca/ranking.py`, delete the stale comment at 309-310 ("Skip load mode penalty — snapshot doesn't carry vram_budget_fraction; the selector already filtered models that exceed the VRAM budget.").

- [ ] **Step 3: Smoke + suites**

Run: `python -c "import src.models.local_model_manager, src.models.model_registry, fatih_hoca.registry, fatih_hoca.ranking"`
Expected: no ImportError.
Run: `timeout 120 pytest packages/fatih_hoca/tests/ -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: delete dead P1 context-sizing symbols + stale ranking comment"
```

---

## Phase 7 — End-to-end behavior + cleanup

### Task 18: Integration — placement & deferral fall out of pressure

**Files:**
- Test: `packages/fatih_hoca/tests/test_desktop_placement_integration.py`

Proves the headline claim: with the same task, flipping desktop state shifts the pick local↔cloud and triggers the WAIT (None) path — with no new policy code in Fatih/Beckman.

- [ ] **Step 1: Write the integration test**

```python
# packages/fatih_hoca/tests/test_desktop_placement_integration.py
# Uses the Phase-2d sim adapter to drive a real rank+gate with a stubbed snapshot.
from nerd_herd.types import SystemSnapshot, LocalModelState
from fatih_hoca.selector import select_for_simulation

PROVIDERS = {"groq": {"is_free": True, "models": {"groq/big": {"cap_score_100": 70}}}}


def _snap(**kw):
    base = dict(vram_available_mb=8000,
                local=LocalModelState(model_name="loaded-local", idle_seconds=120.0))
    base.update(kw)
    return SystemSnapshot(**base)


def test_away_keeps_local():
    # difficulty 3 (easy): loaded-local stickiness + no-swap + cost should
    # win when no desktop pressure. If this flakes on sim cap-weighting,
    # the robust claim is the CONTRAST below — lower difficulty or raise the
    # local cap_override; do NOT weaken test_user_gaming_forces_cloud.
    snap = _snap(user_idle_s=1e9, load_mode="full")
    pick = select_for_simulation(task_name="coder", difficulty=3,
                                 estimated_output_tokens=500, snapshot=snap,
                                 providers_cfg=PROVIDERS)
    assert pick.pool == "local"


def test_user_gaming_forces_cloud():
    # fullscreen present + heavy mode → local S12 veto → cloud wins
    snap = _snap(user_idle_s=1.0, foreground_fullscreen=True, load_mode="heavy")
    pick = select_for_simulation(task_name="coder", difficulty=4,
                                 estimated_output_tokens=500, snapshot=snap,
                                 providers_cfg=PROVIDERS)
    assert pick.pool != "local"
```

- [ ] **Step 2: Run it**

Run: `timeout 60 pytest packages/fatih_hoca/tests/test_desktop_placement_integration.py -v`
Expected: PASS. If `test_user_gaming_forces_cloud` fails with local still chosen, verify the sim's local stub carries `is_local=True` (it does, `selector.py:753`) and that the snapshot reached `pressure_for` (Task 9).

- [ ] **Step 3: Commit**

```bash
git add packages/fatih_hoca/tests/test_desktop_placement_integration.py
git commit -m "test(fatih_hoca): desktop state flips placement local<->cloud"
```

---

### Task 19: Full-suite gate + spec doc closeout

**Files:**
- Modify: `docs/superpowers/specs/2026-05-31-resource-signals-design.md` (mark Status: implemented; link this plan)
- Modify: `CLAUDE.md` (one line under the Phase 2d note: S12/S13 + M4 desktop signals live)

- [ ] **Step 1: Run the affected suites with timeouts**

Run: `timeout 120 pytest packages/nerd_herd/tests/ packages/fatih_hoca/tests/ -q`
Run: `timeout 60 pytest tests/core/ -k "dispatcher" -q`
Expected: PASS. Record the counts.

- [ ] **Step 2: Update the spec status + CLAUDE.md**

In the spec header, change `**Status:** Approved design, ready for implementation plan.` → `**Status:** Implemented (plan `docs/superpowers/plans/2026-06-09-desktop-resource-signals.md`).`
In `CLAUDE.md`, append to the Phase 2d pitfall note: "S12 user-presence + S13 machine-contention are local-only pressure signals (OTHER_BUCKET); load mode = M4 weights on them (Full silences / Heavy·Shared amplify / Minimal = eligibility veto). Desktop fields on `SystemSnapshot`; presence sensed in `nerd_herd/presence.py`; external-GPU read from the 30s auto-detect cache; need_ctx now on `Pick`."

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-05-31-resource-signals-design.md CLAUDE.md
git commit -m "docs: mark resource-signals implemented; note S12/S13/M4 in CLAUDE.md"
```

---

## Tuning follow-ups (NOT this plan — log after a live mission)

- S12/S13 thresholds (`ACTIVE_IDLE_S`, `PRESENT_IDLE_S`, presence penalty band, `RAM_USED_FLOOR/CAP`, `EXTERNAL_GPU_VETO_FRACTION`) are starting guesses. Tune against real `kutai.jsonl` presence/RAM distributions, the way S1/S9 thresholds were tuned.
- Confirm the auto-detect loop's mode downgrades now visibly bias selection (they were dead before M4). Watch a real game/render session.
- Telegram button labels still say full/heavy/shared/minimal; consider relabeling to the spec's Full/Otomatik/Heavy-Shared/Minimal preset language (out of scope here).
- **Spec §3.5 VRAM safety margin (~0.5-1 GB) — consciously DEFERRED, not implemented.** P1 (need-ctx) + small load windows already removed the OOM, so the margin is belt-and-suspenders hardening on `--fit`, not a fix. Add it to DaLLaMa's load flags only if a spike-OOM is observed post-launch. Flagged here so it isn't mistaken for covered.
```
