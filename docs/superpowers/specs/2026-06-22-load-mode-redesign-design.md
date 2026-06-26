# Load Mode Redesign — Design

**Date:** 2026-06-22
**Status:** Approved (pending spec review)

## Problem

Two user-reported defects in the Telegram "Yük Modu" (load mode) buttons:

1. **"Otomatik" button is broken.** The label `"🤖 Otomatik"` appears in *two* keyboards — `KB_WORKFLOW_SELECT` and `KB_YUK_MODU`. The flat `_BUTTON_ACTIONS` dict maps one label to one action, set to `("special", "wf_auto")` (workflow auto-select). So tapping Otomatik inside the Yük Modu menu routes to the `wf_` handler (telegram_bot.py:971), finds no `_pending_mission`, replies "❌ Görev açıklaması bulunamadı", and bounces to the Görevler keyboard. Auto load-management never fires. A `load_auto` code path exists (telegram_bot.py:1038) but **no button reaches it** — it is dead.

2. **The modes no longer make sense.** Since the 2026-06-09 resource-signals change, load mode is no longer a VRAM cap — it weights the desktop presence/contention signals (S13/S14) that bias work cloud↔local. Leftovers cause confusion:
   - Button text still implies VRAM throttle, not cloud↔local placement bias.
   - `VRAM_BUDGETS` fractions (`load.py:19`) are advisory-only; `get_vram_budget_mb` returns raw free VRAM.
   - Auto (`suggest_mode_for_external_usage`) reads only external-GPU fraction; it ignores presence (S13) despite mode now weighting presence.
   - Four manual modes (Full/Heavy/Shared/Minimal) is more than needed.

## Decisions (from brainstorming)

- Mode set: **Auto + 3 manual** (Local-free, Balanced, Cloud-only).
- Balanced weight on S13/S14 = **2.0** (= old Shared).
- Auto when foreground-fullscreen → **Cloud-only**.
- VRAM gauge: **keep**, rekey to 3 modes, mark observability-only.

## Design

### Mode model

Internal canonical mode strings collapse to three: `full`, `balanced`, `minimal`.
`heavy` and `shared` are removed. `auto` is **not** a mode string — it remains the
existing `_auto_managed` boolean flag on `LoadManager`.

| Button (TR) | English | Internal | M4 weight (S13/S14) | Behavior |
|---|---|---|---|---|
| 🤖 Otomatik | Auto | `_auto_managed=True` | — | Auto-detect loop picks full/balanced/minimal |
| 🖥 Yerel Serbest | Local-free | `full` | 0.0 | Desktop signals silenced; send to local freely |
| ⚖️ Dengeli | Balanced | `balanced` | 2.0 | Strong cloud bias when you're active; local allowed if cloud exhausted |
| ☁️ Sadece Bulut | Cloud-only | `minimal` | passthrough (1.0) | Local vetoed at eligibility; cloud only |

`MODE_ORDER = ("minimal", "balanced", "full")` (most → least restrictive) for the
auto-detect hysteresis.

### M4 weights (`nerd_herd/modifiers.py`)

```python
_M4_BY_MODE = {
    "full": 0.0,       # ignore the user — desktop signals silenced
    "balanced": 2.0,   # strong cloud bias when active
    "minimal": 1.0,    # local vetoed at eligibility; passthrough
}
```

### Auto logic (`nerd_herd/load.py`)

**Presence must first be made reachable from the loop** (review blocker). Today
`LoadManager.__init__(gpu_collector, ...)` has no `PresenceCollector`; presence
lives on the `NerdHerd` orchestrator (`nerd_herd.py:35` `self._presence`) and is
only folded into `snapshot()`. The auto-detect loop calls only
`self._gpu.detect_external_gpu_usage()`. `GPUCollector` exposes no idle/fullscreen.

Wiring change:
- `LoadManager.__init__` gains `presence_collector=None` and stores `self._presence`.
- `NerdHerd` construction of `LoadManager` (`nerd_herd.py:~38-43`) passes its
  `PresenceCollector`.
- The loop reads presence via `self._presence` (foreground-fullscreen + idle
  from `nerd_herd/presence.py`); if `self._presence is None`, it degrades to the
  external-GPU-only mapping (back-compat / unit tests without presence).

**Signature back-compat:** the existing `@staticmethod
suggest_mode_for_external_usage(external_vram_fraction)` is **kept and unchanged
in arity** so the `src/infra/load_manager.py:69-71` shim and the
`.claude/settings.local.json` smoke command keep working. Its body is rekeyed to
the 3-mode set: `<0.10 → full`, `<0.60 → balanced`, `else → minimal`. The
presence-aware decision is a **new** instance method
`_suggest_mode(ext_frac, presence)` that the loop calls; the static remains the
external-GPU-only fallback.

`_suggest_mode` mapping:
```
foreground_fullscreen            → minimal
external_gpu_fraction >= 0.60    → minimal
external_gpu_fraction >= 0.10
    OR user present (not away)   → balanced
else (idle AND low ext GPU)      → full
```

Hysteresis is unchanged: switch to a *more restrictive* mode immediately, switch
to a *less restrictive* mode only after `upgrade_delay` (300s) stable. The
`_auto_managed` guard, the `set_load_mode(..., source="auto")` path, and the
notify callback are all retained. `_mode_index` (`load.py:45-49`) `except`
fallback changes from hardcoded `3` to `len(MODE_ORDER)-1` (now 2).

### Otomatik button fix (`telegram_bot.py`)

Keep the identical label `"🤖 Otomatik"` in both menus (consistent UX) and
disambiguate by keyboard state at the dispatcher (telegram_bot.py:~6961, before
the `_BUTTON_ACTIONS.get` lookup):

```python
if text.strip() == "🤖 Otomatik" and self._kb_state.get(chat_id) == "yuk_modu":
    self._pending_action.pop(chat_id, None)
    await self._handle_special_button(update, context, "load_auto")
    return
```

The existing `load_` handler (telegram_bot.py:1038) already turns `load_auto`
into `context.args=["auto"]` → `cmd_load` → `enable_auto_management()`. No new
handler logic needed.

`KB_YUK_MODU` row gains the Otomatik button (already present) and the three
manual buttons are relabeled. `_BUTTON_ACTIONS` updates:

```python
"🖥 Yerel Serbest": ("special", "load_full"),
"⚖️ Dengeli":       ("special", "load_balanced"),
"☁️ Sadece Bulut":  ("special", "load_minimal"),
```

(`load_balanced` flows through the same `action.startswith("load_")` handler →
`set_load_mode("balanced")`.) Old `⚡ Full / 🔋 Heavy / ⚖️ Shared / 🔻 Minimal`
labels and their `_BUTTON_ACTIONS` entries are removed.

> **Alternative considered (review):** a distinct label (e.g. `🤖 Yük Otomatik`)
> in the load menu would self-route through `_BUTTON_ACTIONS` with no special-case
> and no ordering dependency on line 6961 — lower maintenance risk than coupling a
> label-string + state-string in the hot path. Chosen approach keeps the identical
> label for UX consistency and is guarded by the both-states routing test below.
> Final label call deferred to user at spec review.

### Migration (persisted mode)

Mode **is** DB-persisted: `__main__.py:81-107` `_persist_mode` writes a
`load_mode` table; boot restore reads `_load_mode_from_db` (`__main__.py:63-78`)
→ `initial_load_mode` → `LoadManager.__init__:65`
`self._mode = initial_mode if initial_mode in LOAD_MODES else "full"`.

This boot path **bypasses** `set_load_mode`/`get_load_mode`, so normalizing only
there would (review blocker) silently downgrade a persisted `heavy`/`shared` to
`full` on first restart after deploy. Fix:
- Add a `_normalize_mode(m)` helper in `load.py`: maps `heavy`/`shared` →
  `balanced`, passes through `full`/`balanced`/`minimal`, unknown → `full`.
- Apply it in **`_load_mode_from_db`** (or `LoadManager.__init__` before the
  membership test) — the seam that actually runs at boot.
- Also apply in `set_load_mode`/`get_load_mode` for any in-flight legacy value.

### cmd_load help + DESCRIPTIONS

Rewrite to placement language. New `DESCRIPTIONS`:

```python
DESCRIPTIONS = {
    "full":     "Yerel Serbest — masaüstü sinyallerini yoksay; yerele serbest gönder",
    "balanced": "Dengeli — sen aktifken güçlü bulut yönelimi (2×)",
    "minimal":  "Sadece Bulut — yerel çıkarım kapalı, yalnızca bulut",
}
```

`cmd_load` usage text updated to `/load full|balanced|minimal|auto` with the new
descriptions. `auto` and legacy `heavy`/`shared` accepted as input (heavy/shared
normalized to balanced) for back-compat.

### VRAM gauge (kept, observability-only)

`VRAM_BUDGETS` rekeyed to the 3 modes and explicitly marked non-policy:

```python
# Observability only (feeds Prometheus nerd_herd_vram_budget_fraction → Grafana).
# NOT a VRAM cap — placement is owned by S13/S14 + --fit since 2026-06-09.
VRAM_BUDGETS = {"full": 1.0, "balanced": 0.5, "minimal": 0.0}
```

`get_vram_budget_fraction` / `_async` and the `_g_budget` gauge are retained.
`prometheus_metrics` mode→int mapping rekeyed: `{"minimal":0,"balanced":1,"full":2}`.
The `_g_mode` Gauge **HELP string** (`load.py:37`,
`"...(0=minimal,1=shared,2=heavy,3=full)"`) is rewritten to
`"...(0=minimal,1=balanced,2=full)"` to match. Stale `heavy`/`shared`
`_g_mode_info` label series clear on next process restart (accepted; minor).

## Unaffected (verified correct under new strings)

- `selector.py:756` `load_mode == "minimal"` veto — unchanged.
- `image_select.py:194` `== "minimal"`, `:204` `!= "full"` — `balanced`/`minimal`
  both `!= "full"`, so the desktop signal still applies; correct.
- `load.py` `is_local_inference_allowed` `!= "minimal"` — `balanced` allows local;
  correct.

## Testing

- `nerd_herd/tests/test_load.py`: new mode set, M4 weights, presence-aware auto
  mapping via `_suggest_mode` (fullscreen→minimal, ext≥0.6→minimal, present→balanced,
  idle+low→full), `_suggest_mode` with `presence=None`/`self._presence is None`
  degrades to external-only, `_normalize_mode` (heavy/shared→balanced, unknown→full),
  `_mode_index` fallback = `len(MODE_ORDER)-1`, hysteresis with 3 modes,
  `suggest_mode_for_external_usage` static still single-arg and rekeyed to 3 modes.
- `nerd_herd/__main__.py`: boot restore of persisted `heavy`/`shared` normalizes to
  `balanced` (not silently reset to `full`).
- `fatih_hoca/tests/test_image_select_*`, `test_desktop_placement_integration.py`,
  `test_image_select_effective_snapshot.py`: replace `heavy`/`shared` with
  `balanced`; assert M4 weight 2.0.
- Telegram: button dispatch test — Otomatik in `yuk_modu` state → `enable_auto_management`;
  Otomatik in `workflow_select` state → `wf_auto`. Relabeled buttons → correct mode;
  `☁️ Sadece Bulut` still triggers `_free_local_for_minimal` unload suffix.
- Smoke: `python -c "from nerd_herd.load import LOAD_MODES, VRAM_BUDGETS, DESCRIPTIONS"`
  and `python -c "from src.infra.load_manager import suggest_mode_for_external_usage"`.

## Out of scope

- No change to placement/ranking pipeline beyond the M4 weight value.
- No change to eligibility veto semantics (`load_mode_minimal`).
- No Grafana dashboard edits (gauge contract preserved).
