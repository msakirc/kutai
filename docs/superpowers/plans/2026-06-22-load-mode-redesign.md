# Load Mode Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the 4 manual GPU load modes to 3 (full=Local-free, balanced, minimal=Cloud-only) + an Auto flag, fix the broken "Otomatik" Telegram button, make Auto presence-aware, and rekey the dead VRAM cruft to observability-only.

**Architecture:** Load mode weights the desktop presence/contention signals (S13/S14) that bias work cloud↔local — it is not a VRAM cap (since 2026-06-09). `balanced` replaces both `heavy`(1.5×) and `shared`(2.0×) at weight 2.0. Auto's mode picker gains presence (foreground-fullscreen + idle) by wiring `PresenceCollector` into `LoadManager`. The broken Otomatik button (a label collision with workflow-auto) is fixed by disambiguating on keyboard state in the dispatcher.

**Tech Stack:** Python 3.10, asyncio, pytest, python-telegram-bot, prometheus_client, aiosqlite. Packages: `nerd_herd` (load mode + signals), `fatih_hoca` (selection consumers), `src/app/telegram_bot.py` (UI).

**Spec:** `docs/superpowers/specs/2026-06-22-load-mode-redesign-design.md`

---

## File Structure

- `packages/nerd_herd/src/nerd_herd/load.py` — mode set, normalization, VRAM rekey, presence-aware auto. **Core.**
- `packages/nerd_herd/src/nerd_herd/modifiers.py` — `_M4_BY_MODE` rekey.
- `packages/nerd_herd/src/nerd_herd/nerd_herd.py` — pass `PresenceCollector` into `LoadManager`.
- `packages/nerd_herd/src/nerd_herd/__main__.py` — normalize persisted mode at boot.
- `src/app/telegram_bot.py` — relabel buttons, dispatcher disambiguation, `cmd_load` help.
- Tests: `packages/nerd_herd/tests/test_load.py`, `test_m4_load_mode.py`, new `test_main_normalize.py`; `packages/fatih_hoca/tests/test_image_select_eviction.py`, `test_image_select_effective_snapshot.py`, `test_desktop_placement_integration.py`.

**Working directory for all `pytest`:** `packages/nerd_herd` and `packages/fatih_hoca` use their own installed packages. Run from repo root with the venv active. Always foreground + `timeout` (per CLAUDE.md — never background pytest).

---

## Task 1: load.py — 3-mode core, normalization, VRAM/gauge rekey

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/load.py`
- Test: `packages/nerd_herd/tests/test_load.py`

- [ ] **Step 1: Update the existing tests to the new 3-mode world (these will fail first)**

Replace the bodies of these tests in `packages/nerd_herd/tests/test_load.py`:

```python
def test_set_mode():
    lm = LoadManager(gpu_collector=MagicMock())
    lm.set_load_mode("balanced", source="user")
    assert lm.get_load_mode() == "balanced"
    assert not lm.is_auto_managed()


def test_vram_budget_fraction():
    lm = LoadManager(gpu_collector=MagicMock())
    assert lm.get_vram_budget_fraction() == 1.0
    lm.set_load_mode("balanced")
    assert lm.get_vram_budget_fraction() == 0.5
    lm.set_load_mode("minimal")
    assert lm.get_vram_budget_fraction() == 0.0


def test_vram_budget_mb(gpu_collector):
    lm = LoadManager(gpu_collector=gpu_collector)
    assert lm.get_vram_budget_mb() == 8000  # full mode — raw free
    lm.set_load_mode("balanced")
    assert lm.get_vram_budget_mb() == 8000  # balanced — still raw free (no cap)


def test_on_mode_change_callback():
    lm = LoadManager(gpu_collector=MagicMock())
    calls = []
    lm.on_mode_change(lambda old, new, src: calls.append((old, new, src)))
    lm.set_load_mode("balanced", source="auto")
    assert calls == [("full", "balanced", "auto")]


def test_suggest_mode():
    lm = LoadManager(gpu_collector=MagicMock())
    assert lm.suggest_mode_for_external_usage(0.05) == "full"
    assert lm.suggest_mode_for_external_usage(0.30) == "balanced"
    assert lm.suggest_mode_for_external_usage(0.70) == "minimal"


def test_enable_auto_management():
    lm = LoadManager(gpu_collector=MagicMock())
    lm.set_load_mode("balanced", source="user")
    assert not lm.is_auto_managed()
    lm.enable_auto_management()
    assert lm.is_auto_managed()


def test_vram_budget_mb_is_raw_free_regardless_of_mode():
    class _G:
        def gpu_state(self):
            from nerd_herd.types import GPUState
            return GPUState(available=True, vram_total_mb=8000, vram_free_mb=8000)
    lm = LoadManager(gpu_collector=_G())
    lm.set_load_mode("balanced", source="user")
    assert lm.get_vram_budget_mb() == 8000
```

Add these new tests at the end of the file:

```python
def test_normalize_legacy_modes():
    from nerd_herd.load import _normalize_mode
    assert _normalize_mode("heavy") == "balanced"
    assert _normalize_mode("shared") == "balanced"
    assert _normalize_mode("full") == "full"
    assert _normalize_mode("balanced") == "balanced"
    assert _normalize_mode("minimal") == "minimal"
    assert _normalize_mode("turbo") == "full"   # unknown → full


def test_set_legacy_mode_normalizes():
    lm = LoadManager(gpu_collector=MagicMock())
    msg = lm.set_load_mode("shared", source="user")
    assert lm.get_load_mode() == "balanced"
    assert "Unknown" not in msg


def test_init_normalizes_legacy_initial_mode():
    lm = LoadManager(gpu_collector=MagicMock(), initial_mode="heavy")
    assert lm.get_load_mode() == "balanced"


def test_load_modes_set():
    from nerd_herd.load import LOAD_MODES
    assert LOAD_MODES == ("full", "balanced", "minimal")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `timeout 60 python -m pytest packages/nerd_herd/tests/test_load.py -q`
Expected: FAIL — `_normalize_mode` import error, `suggest_mode_for_external_usage(0.30)` returns `"heavy"`, etc.

- [ ] **Step 3: Edit `load.py` constants and add `_normalize_mode`**

Replace lines 16-49 region. New `LOAD_MODES`, `MODE_ORDER`, `VRAM_BUDGETS`, `DESCRIPTIONS`, normalize helper, gauge HELP, `_mode_index`:

```python
LOAD_MODES = ("full", "balanced", "minimal")
MODE_ORDER = ("minimal", "balanced", "full")

# Legacy modes (pre-2026-06-22 4-mode set) normalize to the closest survivor.
_LEGACY_ALIASES = {"heavy": "balanced", "shared": "balanced"}


def _normalize_mode(mode: str) -> str:
    """Map any input to a canonical mode. Legacy heavy/shared → balanced;
    unknown → full. Idempotent on canonical values."""
    if mode in LOAD_MODES:
        return mode
    return _LEGACY_ALIASES.get(mode, "full")


# Observability only (feeds Prometheus nerd_herd_vram_budget_fraction → Grafana).
# NOT a VRAM cap — placement is owned by S13/S14 + --fit since 2026-06-09.
VRAM_BUDGETS: dict[str, float] = {
    "full": 1.0,
    "balanced": 0.5,
    "minimal": 0.0,
}

# User-facing mode descriptions. Load mode is placement, not a VRAM cap: it
# weights the desktop presence/contention signals (S13/S14) that bias work
# cloud↔local. VRAM_BUDGETS fractions above are advisory/observability only.
DESCRIPTIONS: dict[str, str] = {
    "full": "Yerel Serbest — masaüstü sinyallerini yoksay; yerele serbest gönder",
    "balanced": "Dengeli — sen aktifken güçlü bulut yönelimi (2×)",
    "minimal": "Sadece Bulut — yerel çıkarım kapalı, yalnızca bulut",
}

_g_mode = Gauge("nerd_herd_load_mode", "Current GPU load mode (0=minimal,1=balanced,2=full)")
```

Keep `_g_mode_info`, `_g_budget`, `_g_auto`, `_ALL_GAUGES` as-is.

Update `_mode_index` fallback:

```python
def _mode_index(mode: str) -> int:
    try:
        return MODE_ORDER.index(mode)
    except ValueError:
        return len(MODE_ORDER) - 1   # treat unknown as least-restrictive (full)
```

- [ ] **Step 4: Normalize in `__init__`, `set_load_mode`, and rekey `suggest`/`prometheus_metrics`**

In `LoadManager.__init__`, change the mode line:

```python
        self._mode = _normalize_mode(initial_mode)
```

In `set_load_mode`, guard genuinely-unknown input BEFORE normalizing (so
`_normalize_mode`'s unknown-to-full mapping does NOT swallow the
`test_invalid_mode` "turbo" -> "Unknown" expectation), THEN normalize legacy
aliases. This is the ONLY correct form — use exactly this:

```python
    def set_load_mode(self, mode: str, source: str = "user") -> str:
        if mode not in LOAD_MODES and mode not in _LEGACY_ALIASES:
            return f"Unknown mode '{mode}'. Choose: {', '.join(LOAD_MODES)}"
        mode = _normalize_mode(mode)
        prev = self._mode
        if prev == mode:
            return f"Already in *{mode}* mode"
        ...  # rest of the method body unchanged
```

Trace: `"turbo"` is not in LOAD_MODES and not in `_LEGACY_ALIASES` -> "Unknown".
`"shared"` is in `_LEGACY_ALIASES` -> normalizes to `"balanced"`. Do NOT
normalize before the guard — that path lets `_normalize_mode("turbo")` -> `"full"`
slip through and breaks `test_invalid_mode`.

Replace `suggest_mode_for_external_usage` (static, arity unchanged — the
`src/infra/load_manager.py` shim depends on it):

```python
    @staticmethod
    def suggest_mode_for_external_usage(external_vram_fraction: float) -> str:
        # External-GPU-only fallback (no presence). 3-mode set.
        if external_vram_fraction < 0.10:
            return "full"
        elif external_vram_fraction < 0.60:
            return "balanced"
        else:
            return "minimal"
```

Replace the `prometheus_metrics` int map:

```python
    def prometheus_metrics(self) -> list:
        mode_val = {"minimal": 0, "balanced": 1, "full": 2}.get(self._mode, 2)
        _g_mode.set(mode_val)
        for m in LOAD_MODES:
            _g_mode_info.labels(mode=m).set(1 if m == self._mode else 0)
        _g_budget.set(self.get_vram_budget_fraction())
        _g_auto.set(1 if self._auto_managed else 0)
        return list(_ALL_GAUGES)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `timeout 60 python -m pytest packages/nerd_herd/tests/test_load.py -q`
Expected: PASS (all, including the new normalize tests).

- [ ] **Step 6: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/load.py packages/nerd_herd/tests/test_load.py
git commit -m "feat(load): collapse to 3 modes (full/balanced/minimal) + legacy normalize"
```

---

## Task 2: load.py — presence-aware Auto

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/load.py`
- Test: `packages/nerd_herd/tests/test_load.py`

- [ ] **Step 1: Write the failing tests**

Add to `packages/nerd_herd/tests/test_load.py`:

```python
class _Presence:
    def __init__(self, idle_s, fullscreen):
        self._d = {"user_idle_s": idle_s, "foreground_fullscreen": fullscreen}
    def collect(self):
        return self._d


def test_suggest_mode_fullscreen_forces_minimal():
    lm = LoadManager(gpu_collector=MagicMock(),
                     presence_collector=_Presence(idle_s=1.0, fullscreen=True))
    # fullscreen wins even at zero external GPU
    assert lm._suggest_mode(0.0, lm._presence.collect()) == "minimal"


def test_suggest_mode_high_external_minimal():
    lm = LoadManager(gpu_collector=MagicMock(),
                     presence_collector=_Presence(idle_s=1e9, fullscreen=False))
    assert lm._suggest_mode(0.70, lm._presence.collect()) == "minimal"


def test_suggest_mode_present_balanced():
    lm = LoadManager(gpu_collector=MagicMock(),
                     presence_collector=_Presence(idle_s=5.0, fullscreen=False))
    # present (idle < 300) with low external GPU → balanced
    assert lm._suggest_mode(0.0, lm._presence.collect()) == "balanced"


def test_suggest_mode_away_idle_full():
    lm = LoadManager(gpu_collector=MagicMock(),
                     presence_collector=_Presence(idle_s=1e9, fullscreen=False))
    # away + low external GPU → full (local-free)
    assert lm._suggest_mode(0.05, lm._presence.collect()) == "full"


def test_suggest_mode_mid_external_balanced_even_if_away():
    lm = LoadManager(gpu_collector=MagicMock(),
                     presence_collector=_Presence(idle_s=1e9, fullscreen=False))
    assert lm._suggest_mode(0.20, lm._presence.collect()) == "balanced"


def test_suggest_mode_no_presence_degrades_to_external_only():
    lm = LoadManager(gpu_collector=MagicMock())  # no presence_collector
    assert lm._presence is None
    assert lm._suggest_mode(0.05, None) == "full"
    assert lm._suggest_mode(0.30, None) == "balanced"
    assert lm._suggest_mode(0.70, None) == "minimal"
```

- [ ] **Step 2: Run to verify they fail**

Run: `timeout 60 python -m pytest packages/nerd_herd/tests/test_load.py -k suggest_mode -q`
Expected: FAIL — `LoadManager.__init__` has no `presence_collector` kwarg; `_suggest_mode` undefined.

- [ ] **Step 3: Add `presence_collector` to `__init__` and the `_suggest_mode` method**

In `LoadManager.__init__` signature and body:

```python
    def __init__(
        self,
        gpu_collector,
        initial_mode: str = "full",
        detect_interval: int = 30,
        upgrade_delay: int = 300,
        presence_collector=None,
    ) -> None:
        self._gpu = gpu_collector
        self._presence = presence_collector
        self._mode = _normalize_mode(initial_mode)
        ...  # rest unchanged
```

Add the method (near `suggest_mode_for_external_usage`):

```python
    # Presence-aware auto mode picker. presence is the dict from
    # PresenceCollector.collect() or None (degrade to external-GPU only).
    def _suggest_mode(self, external_vram_fraction: float, presence: dict | None) -> str:
        from nerd_herd.signals.s13_presence import PRESENT_IDLE_S
        if presence:
            if presence.get("foreground_fullscreen"):
                return "minimal"
        if external_vram_fraction >= 0.60:
            return "minimal"
        present = bool(presence) and float(
            presence.get("user_idle_s", PRESENT_IDLE_S)
        ) < PRESENT_IDLE_S
        if external_vram_fraction >= 0.10 or present:
            return "balanced"
        return "full"
```

- [ ] **Step 4: Wire `_suggest_mode` into the auto-detect loop**

In `_auto_detect_loop`, replace the `suggested = ...` line (currently
`suggested = self.suggest_mode_for_external_usage(ext.external_vram_fraction)`):

```python
                ext = self._gpu.detect_external_gpu_usage()
                self._record_external(ext)
                presence = None
                if self._presence is not None:
                    try:
                        presence = self._presence.collect()
                    except Exception:
                        presence = None
                suggested = self._suggest_mode(ext.external_vram_fraction, presence)
```

- [ ] **Step 5: Run to verify they pass**

Run: `timeout 60 python -m pytest packages/nerd_herd/tests/test_load.py -q`
Expected: PASS (full file).

- [ ] **Step 6: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/load.py packages/nerd_herd/tests/test_load.py
git commit -m "feat(load): presence-aware auto mode (fullscreen→cloud, present→balanced)"
```

---

## Task 3: Wire PresenceCollector into LoadManager

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/nerd_herd.py:38-43`
- Test: `packages/nerd_herd/tests/test_nerd_herd.py`

- [ ] **Step 1: Write the failing test**

Add to `packages/nerd_herd/tests/test_nerd_herd.py`:

```python
def test_load_manager_gets_presence_collector():
    from nerd_herd.nerd_herd import NerdHerd
    nh = NerdHerd(metrics_port=0, llama_server_url="")
    assert nh._load._presence is nh._presence
```

- [ ] **Step 2: Run to verify it fails**

Run: `timeout 60 python -m pytest packages/nerd_herd/tests/test_nerd_herd.py::test_load_manager_gets_presence_collector -q`
Expected: FAIL — `nh._load._presence is None`, not `nh._presence`.

- [ ] **Step 3: Pass the collector in**

In `nerd_herd.py`, the `LoadManager(...)` construction:

```python
        self._load = LoadManager(
            gpu_collector=self._gpu,
            initial_mode=initial_load_mode,
            detect_interval=detect_interval,
            upgrade_delay=upgrade_delay,
            presence_collector=self._presence,
        )
```

(`self._presence` is already created at line 35, before this block — no ordering change needed.)

- [ ] **Step 4: Run to verify it passes**

Run: `timeout 60 python -m pytest packages/nerd_herd/tests/test_nerd_herd.py::test_load_manager_gets_presence_collector -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/nerd_herd.py packages/nerd_herd/tests/test_nerd_herd.py
git commit -m "feat(nerd_herd): wire PresenceCollector into LoadManager auto loop"
```

---

## Task 4: M4 weights rekey

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/modifiers.py:81-86`
- Test: `packages/nerd_herd/tests/test_m4_load_mode.py`

- [ ] **Step 1: Rewrite the M4 tests**

Replace `packages/nerd_herd/tests/test_m4_load_mode.py` body. The old
`test_*` referencing `heavy`/`shared` must go. Target tests:

```python
from nerd_herd.modifiers import M4_load_mode_weights


def test_full_silences_desktop_signals():
    w = M4_load_mode_weights(mode="full")
    assert w["S13"] == 0.0 and w["S14"] == 0.0


def test_balanced_amplifies():
    w = M4_load_mode_weights(mode="balanced")
    assert w["S13"] == 2.0 and w["S14"] == 2.0


def test_minimal_passthrough():
    w = M4_load_mode_weights(mode="minimal")
    assert w["S13"] == 1.0 and w["S14"] == 1.0


def test_unknown_mode_passthrough():
    w = M4_load_mode_weights(mode="bogus")
    assert w["S13"] == 1.0 and w["S14"] == 1.0


def test_only_touches_s13_s14():
    w = M4_load_mode_weights(mode="balanced")
    assert set(w.keys()) == {"S13", "S14"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `timeout 60 python -m pytest packages/nerd_herd/tests/test_m4_load_mode.py -q`
Expected: FAIL — `M4_load_mode_weights(mode="balanced")` returns 1.0 (unknown), not 2.0.

- [ ] **Step 3: Rekey `_M4_BY_MODE`**

In `modifiers.py`:

```python
_M4_BY_MODE: dict[str, float] = {
    "full": 0.0,       # ignore the user — desktop signals silenced
    "balanced": 2.0,   # strong cloud bias when active (was shared)
    "minimal": 1.0,    # local already vetoed at eligibility; passthrough
}
```

- [ ] **Step 4: Run to verify M4 tests pass**

Run: `timeout 60 python -m pytest packages/nerd_herd/tests/test_m4_load_mode.py -q`
Expected: PASS

- [ ] **Step 5: Fix the one dependent fatih_hoca test in the SAME commit (no cross-package red window)**

The M4 rekey makes `M4_load_mode_weights(mode="heavy")` return 1.0 instead of
1.5, so `packages/fatih_hoca/tests/test_desktop_placement_integration.py:42`
(`load_mode="heavy"`, expecting amplified cloud-ward placement) can go red. Edit
it in THIS commit — the source change and its dependent test must be atomic.

In `test_desktop_placement_integration.py`: change `load_mode="heavy"` (line ~42)
to `load_mode="balanced"`; update the module docstring (lines ~5, ~9) from
"×1.5" / "heavy" to "×2.0" / "balanced". The assertion (`pick.pool != "local"`
under fullscreen + amplified S13) stays valid — `balanced` (M4=2.0) still
amplifies S13/S14.

Run: `timeout 90 python -m pytest packages/fatih_hoca/tests/test_desktop_placement_integration.py -q`
Expected: PASS

- [ ] **Step 6: Commit (modifiers + both dependent tests together)**

```bash
git add packages/nerd_herd/src/nerd_herd/modifiers.py packages/nerd_herd/tests/test_m4_load_mode.py packages/fatih_hoca/tests/test_desktop_placement_integration.py
git commit -m "feat(modifiers): M4 balanced=2.0, drop heavy/shared"
```

---

## Task 5: Boot normalize of persisted mode

**Files:**
- Modify: `packages/nerd_herd/src/nerd_herd/__main__.py:63-78`
- Test: Create `packages/nerd_herd/tests/test_main_normalize.py`

- [ ] **Step 1: Write the failing test**

Create `packages/nerd_herd/tests/test_main_normalize.py`:

```python
import asyncio
import aiosqlite
import pytest


@pytest.mark.asyncio
async def test_load_mode_from_db_normalizes_legacy(tmp_path):
    from nerd_herd.__main__ import _load_mode_from_db
    db = str(tmp_path / "t.db")
    async with aiosqlite.connect(db, isolation_level=None) as c:
        await c.execute(
            "CREATE TABLE load_mode (id INTEGER PRIMARY KEY, mode TEXT NOT NULL, "
            "auto_managed INTEGER NOT NULL DEFAULT 1, updated_at TEXT)"
        )
        await c.execute("INSERT INTO load_mode (id, mode) VALUES (1, 'shared')")
        await c.commit()
    assert await _load_mode_from_db(db) == "balanced"


@pytest.mark.asyncio
async def test_load_mode_from_db_missing_returns_full(tmp_path):
    from nerd_herd.__main__ import _load_mode_from_db
    assert await _load_mode_from_db(str(tmp_path / "none.db")) == "full"
```

(If `pytest-asyncio` is not configured for this package, mark with the existing
async pattern used in `test_nerd_herd.py` — check that file's top for `asyncio`
mode. If tests there use `asyncio.run`, rewrite these two as plain functions
calling `asyncio.run(_load_mode_from_db(...))`.)

- [ ] **Step 2: Run to verify it fails**

Run: `timeout 60 python -m pytest packages/nerd_herd/tests/test_main_normalize.py -q`
Expected: FAIL — returns `"shared"`, not `"balanced"`.

- [ ] **Step 3: Normalize in `_load_mode_from_db`**

In `__main__.py`, change the return inside the cursor block:

```python
                row = await cursor.fetchone()
                if row:
                    from nerd_herd.load import _normalize_mode
                    return _normalize_mode(row[0])
```

- [ ] **Step 4: Run to verify it passes**

Run: `timeout 60 python -m pytest packages/nerd_herd/tests/test_main_normalize.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/nerd_herd/src/nerd_herd/__main__.py packages/nerd_herd/tests/test_main_normalize.py
git commit -m "fix(load): normalize persisted legacy mode at boot (heavy/shared→balanced)"
```

---

## Task 6: Cosmetic heavy/shared → balanced in remaining fatih_hoca tests

**Files:**
- Modify: `packages/fatih_hoca/tests/test_image_select_eviction.py` (lines ~211, ~232, ~251)
- Modify: `packages/fatih_hoca/tests/test_image_select_effective_snapshot.py` (lines ~74, ~89, ~106)

> `test_desktop_placement_integration.py` was already handled in Task 4 (it is
> the only fatih_hoca test the M4 rekey turns red — fixed atomically with the
> source change to avoid a cross-package red commit). The two files in THIS task
> use `load_mode` only via the string vetoes (`!= "full"` / `== "minimal"`),
> NOT via M4 weights, so they are GREEN both before and after Task 4 — this is a
> forward-compat cleanup (kill the dead `heavy`/`shared` literals), never red.

No production `fatih_hoca` code changes — `selector.py:756 == "minimal"` and
`image_select.py:204 != "full"` are correct for the new strings (verified in
spec "Unaffected"; grep of `packages/fatih_hoca/src` confirms zero `heavy`/`shared`
load-mode literals in prod code).

- [ ] **Step 1: Confirm green baseline**

Run: `timeout 90 python -m pytest packages/fatih_hoca/tests/test_image_select_eviction.py packages/fatih_hoca/tests/test_image_select_effective_snapshot.py -q`
Expected: PASS (these never went red — `heavy`/`shared` still satisfy `!= "full"`).

- [ ] **Step 2: Replace `heavy`/`shared` with `balanced`**

In `test_image_select_eviction.py`: change `mode="shared"` (lines ~211, ~251) and `mode="heavy"` (line ~232) all to `mode="balanced"`. These tests assert local is de-prioritized vs cloud under desktop pressure when `mode != "full"`; `balanced` satisfies that.

In `test_image_select_effective_snapshot.py`: change `load_mode="shared"` (line ~74) → `"balanced"` and the matching assertion `merged.load_mode == "shared"` (line ~89) → `== "balanced"`; change `SystemSnapshot(load_mode="heavy")` (line ~106) → `"balanced"`.

- [ ] **Step 3: Run to verify they still pass**

Run: `timeout 90 python -m pytest packages/fatih_hoca/tests/test_image_select_eviction.py packages/fatih_hoca/tests/test_image_select_effective_snapshot.py -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add packages/fatih_hoca/tests/test_image_select_eviction.py packages/fatih_hoca/tests/test_image_select_effective_snapshot.py
git commit -m "test(fatih_hoca): heavy/shared → balanced in image-select tests"
```

---

## Task 7: Telegram UI — relabel buttons, fix Otomatik, cmd_load help

**Files:**
- Modify: `src/app/telegram_bot.py` — `KB_YUK_MODU` (~177-181), `_BUTTON_ACTIONS` (~245-248), dispatcher (~6961), `cmd_load` (~6760-6794)
- Test: Create `tests/app/test_load_buttons.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/app/test_load_buttons.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
import src.app.telegram_bot as tb


def test_button_actions_have_new_load_labels():
    a = tb._BUTTON_ACTIONS
    assert a["🖥 Yerel Serbest"] == ("special", "load_full")
    assert a["⚖️ Dengeli"] == ("special", "load_balanced")
    assert a["☁️ Sadece Bulut"] == ("special", "load_minimal")
    # old labels gone
    assert "🔋 Heavy" not in a
    assert "⚖️ Shared" not in a


def test_otomatik_still_maps_to_workflow_auto():
    # The flat dict entry remains workflow-auto; load-menu routing is by state.
    assert tb._BUTTON_ACTIONS["🤖 Otomatik"] == ("special", "wf_auto")


@pytest.mark.asyncio
async def test_load_balanced_button_sets_balanced_mode(monkeypatch):
    iface = tb.TelegramInterface.__new__(tb.TelegramInterface)
    iface._reply = AsyncMock()
    called = {}

    async def _fake_set(mode, source="user"):
        called["mode"] = mode
        return f"set {mode}"
    monkeypatch.setattr("src.infra.load_manager.set_load_mode", _fake_set)

    update = MagicMock()
    update.effective_chat.id = 1
    context = MagicMock()
    await iface._handle_special_button(update, context, "load_balanced")
    assert called["mode"] == "balanced"
```

(If `TelegramInterface.__new__` skips required attributes that
`_handle_special_button`'s `load_` branch touches, the test only needs
`_reply` and `context.args` — the `load_` branch sets `context.args` then calls
`cmd_load`, which calls `set_load_mode`. Confirm by reading the branch at
telegram_bot.py:1036-1043 and cmd_load at 6789-6793; stub only what's hit.)

- [ ] **Step 2: Run to verify it fails**

Run: `timeout 60 python -m pytest tests/app/test_load_buttons.py -q`
Expected: FAIL — new labels not in `_BUTTON_ACTIONS`; `load_balanced` not handled.

- [ ] **Step 3: Relabel `KB_YUK_MODU` and `_BUTTON_ACTIONS`**

In `telegram_bot.py`, replace `KB_YUK_MODU` (lines ~177-181):

```python
KB_YUK_MODU = _make_keyboard([
    ["🤖 Otomatik", "🖥 Yerel Serbest"],
    ["⚖️ Dengeli", "☁️ Sadece Bulut"],
    ["🔙 Geri"],
])
```

Replace the four "Yük Modu sub-buttons" entries in `_BUTTON_ACTIONS` (lines ~244-248):

```python
    # ── Yük Modu sub-buttons ──
    "🖥 Yerel Serbest": ("special", "load_full"),
    "⚖️ Dengeli": ("special", "load_balanced"),
    "☁️ Sadece Bulut": ("special", "load_minimal"),
```

(Leave `"🤖 Otomatik": ("special", "wf_auto")` at line ~234 untouched.)

- [ ] **Step 4: Add the dispatcher disambiguation pre-check**

In the message handler, immediately BEFORE `btn_action = _BUTTON_ACTIONS.get(text.strip())` (line ~6961):

```python
        # Otomatik label is shared by the workflow picker and the Yük Modu menu.
        # In the load menu it means "auto-manage GPU load", not workflow-auto.
        if text.strip() == "🤖 Otomatik" and self._kb_state.get(chat_id) == "yuk_modu":
            self._pending_action.pop(chat_id, None)
            await self._handle_special_button(update, context, "load_auto")
            return
```

(`load_auto` is already handled at telegram_bot.py:1036-1043 → `context.args=["auto"]` → `cmd_load` → `enable_auto_management`.)

- [ ] **Step 5: Update `cmd_load` help text + accept legacy aliases**

In `cmd_load` (lines ~6767-6776), replace the usage block:

```python
            await self._reply(update,
                f"Current load mode: *{current}*{auto_str}\n\n"
                "Usage: `/load full|balanced|minimal|auto`\n"
                "• *full* (Yerel Serbest) — ignore desktop signals; send to local freely\n"
                "• *balanced* (Dengeli) — strong cloud bias when you're active (2×)\n"
                "• *minimal* (Sadece Bulut) — cloud only; pause local\n"
                "• *auto* (Otomatik) — auto-pick mode from external GPU + presence",
                parse_mode="Markdown",
            )
```

No change needed to the `set_load_mode(choice, ...)` call — `set_load_mode`
normalizes legacy `heavy`/`shared` (Task 1). `/load minimal` still triggers the
`_free_local_for_minimal` suffix (line ~6791) unchanged.

- [ ] **Step 6: Run to verify it passes**

Run: `timeout 60 python -m pytest tests/app/test_load_buttons.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/app/telegram_bot.py tests/app/test_load_buttons.py
git commit -m "fix(telegram): 3-mode load buttons + fix Otomatik state collision"
```

---

## Task 8: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Import smoke (both packages + shim)**

Run:
```bash
python -c "from nerd_herd.load import LOAD_MODES, VRAM_BUDGETS, DESCRIPTIONS, _normalize_mode; print(LOAD_MODES, VRAM_BUDGETS)"
python -c "from src.infra.load_manager import suggest_mode_for_external_usage as s; print(s(0.05), s(0.3), s(0.7))"
python -c "import src.app.telegram_bot as t; print('🖥 Yerel Serbest' in t._BUTTON_ACTIONS)"
```
Expected: `('full', 'balanced', 'minimal') {...}`; `full balanced minimal`; `True`. No ImportError.

- [ ] **Step 2: Targeted suites green**

Run: `timeout 120 python -m pytest packages/nerd_herd/tests/ -q`
Expected: PASS.

Run: `timeout 120 python -m pytest packages/fatih_hoca/tests/test_image_select_eviction.py packages/fatih_hoca/tests/test_image_select_effective_snapshot.py packages/fatih_hoca/tests/test_desktop_placement_integration.py packages/fatih_hoca/tests/test_minimal_mode_eligibility.py -q`
Expected: PASS.

Run: `timeout 60 python -m pytest tests/app/test_load_buttons.py -q`
Expected: PASS.

- [ ] **Step 3: Grep for stray dead-mode references**

Run: `rtk grep -rn '"heavy"\|"shared"\|load_heavy\|load_shared\|🔋 Heavy\|⚖️ Shared' packages/nerd_herd/src src/app/telegram_bot.py src/infra/load_manager.py`
Expected: no GPU-load-mode hits (cost-band `"heavy"` in beckman/mr_roboto posthooks is a DIFFERENT domain — ignore those if grep widened).

- [ ] **Step 4: Final commit if any cleanup was needed**

```bash
git add -A && git commit -m "test(load): full verification pass for 3-mode redesign"
```

(Skip if Steps 1-3 needed no edits.)

---

## Notes for the executor

- **Do NOT push.** Changes are restart-gated — the running KutAI must `/restart` (or full Yaşar Usta relaunch for the nerd_herd sidecar to pick up new `load.py`) to load them. The user pushes + restarts.
- **Never background pytest** (CLAUDE.md): foreground + `timeout`, reap by exact PID.
- **nerd_herd is a separate sidecar process** — `load.py`/`__main__.py` changes only take effect on sidecar restart, not KutAI `/restart`. Flag this to the user at handoff.
- Telegram label strings are user-facing TR; if the user wants different wording, only the `KB_YUK_MODU` + `_BUTTON_ACTIONS` keys + `DESCRIPTIONS` change.
