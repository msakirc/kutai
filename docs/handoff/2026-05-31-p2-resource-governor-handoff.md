# Handoff — P2: Resource Governor (clean-slate kickoff)

**Date:** 2026-05-31
**Prereq shipped:** P1 need-ctx — commit `2330f1ab`. The OOM-storm fire is out.
**Read first:** `docs/handoff/2026-05-31-load-mode-redesign-ideas.md` (full rationale, crown insight, evidence). This doc is the *accelerated start* — what's already true, what to do next, what to delete.

---

## What P1 shipped (the ground you stand on)

- Local models now load at **need-ctx**, not VRAM-derived-then-floored. `_need_ctx(task_min_ctx, model_ctx_ceiling)` in `src/models/local_model_manager.py`:
  `need = clamp(ceil_2048(task_min_ctx or MIN_CTX), MIN_CTX, model.context_length)`, `MIN_CTX=8192` (env `LLAMA_MIN_CTX`).
- `--fit` owns GPU-layer fitting. **No VRAM math in the load path anymore.** Small need fits even under a VRAM spike → OOM cannot recur.
- `models.yaml` `context_length` override still wins (pinned, no resize). Reload-on-expansion guard kept for 18k–28k tasks.
- Verified: `_need_ctx` 4412→8192 / 0→8192 / 18000→18432 / 40000-cap→8192; 8/8 ctx tests pass.

## Clean-slate state — what is DEAD and removable in P2

These are RETIRED (uncalled), kept only so stale imports resolve. **P2 should delete them once it confirms nothing imports them** (audit call sites, not docstrings):

| Symbol | File | Status |
|---|---|---|
| `calculate_dynamic_context` | `packages/fatih_hoca/src/fatih_hoca/registry.py:440` | DEPRECATED, dead |
| `vram_context_ceiling` | `packages/fatih_hoca/src/fatih_hoca/registry.py:533` | DEPRECATED, dead (was the P1 stopgap) |
| `BASELINE_LOCAL_CTX`, `_floored_baseline_ctx` | `src/models/local_model_manager.py` | DEPRECATED, dead |
| re-exports of the above | `src/models/model_registry.py` | remove with the originals |
| `tests/test_local_ctx_floor.py`, ctx tests in `packages/fatih_hoca/tests/test_registry.py` | — | delete with the funcs they cover |

Severed load-mode enforcement (also dead, P2 replaces it — do NOT revive as-is):
- `src/core/router.py:233-247,317` calls **sync stubs** `is_local_inference_allowed()` / `get_vram_budget_fraction()` in `src/infra/load_manager.py:43,67` that hard-return `True`/`1.0`. No-op. (And `router.py` selection is itself dead per the root-debt-map — live selection is `fatih_hoca/ranking.py::rank_candidates`.)
- `fatih_hoca.select()` / `ranking.py` have **zero** load-mode references — the live selection path is mode-blind.
- The loader (`ensure_model`) never receives a budget.

## ⚠️ Founder has "many objections" to the P1 design

The founder explicitly said they have objections but chose to ship the fire fix first. **P2 brainstorming MUST surface and resolve these before designing** — do not assume the P1 shape (MIN_CTX=8192, ceil_2048, 2048-block, "trust --fit / accept rare OOM") is settled. Open the P2 session by asking the founder to enumerate the objections.

## P2 = Resource Governor — the direction (not yet specced)

**Crown insight (the whole point):** VRAM-cap is the WRONG "back off" lever for a desktop. Capping KutAI's VRAM pushes weights/KV onto CPU+RAM — the exact resources that make the desktop lag — while idling spare VRAM nobody uses. For a non-gamer, **VRAM is the spare resource.** So the lever is **placement, not capping.**

**Genuine "yield to user" levers, in order:** prefer cloud → keep local **on GPU** at need-ctx → unload when idle. CPU-offload (`-ngl` cap) only last-resort (local is the *only* option AND model doesn't fully fit).

**Signals (sense more, cap less):**
- VRAM free + external GPU usage ✓ already have (`nerd_herd`, `detect_external_gpu_usage`).
- **RAM pressure** — `psutil.virtual_memory`. Easy, missing.
- **User presence** — input-idle (`GetLastInputInfo` via ctypes) + fullscreen/game detection (nvidia-smi util / foreground window). Missing; highest-leverage new signal.
- Cloud availability ✓ (KDV).

**Placement policy (replaces VRAM_BUDGETS):**
| User state | Action |
|---|---|
| Away / idle (most missions run overnight) | Full send — max local GPU, big models |
| Active, light | keep local on GPU + need-ctx + throttle concurrency + cloud for bursts. **No CPU-offload.** |
| Active, GPU-heavy (game/render) | cloud; if cloud dead → unload + defer local-only (WAIT, founder principle) |
| Active, RAM-heavy | keep on GPU, smaller model, no CPU-offload |

**Manual buttons → presets over the governor:** Full=ignore user / Otomatik=governor decides / Heavy-Shared=cloud-bias strength / Minimal=cloud-only-pause. Load mode is NOT killed — its *mechanism* flips from VRAM-cap to placement.

## Open questions P2 must answer (beyond founder objections)

1. Where does placement decision live — bias inside `fatih_hoca/ranking.py` (cloud↔local), or a pre-selection governor gate? `ranking.py` is the only live selection path; wire there, not router.
2. Presence detection cost/reliability on Windows — pick cheapest signals (`GetLastInputInfo` idle, foreground-fullscreen, `psutil`).
3. Concurrency throttle + idle-unload/resume — idle-unload already exists in DaLLaMa; what's the resume trigger?
4. Small VRAM safety margin (~0.5–1 GB) so spikes never OOM even with need-ctx — bake into `--fit` flags or governor?
5. Reconcile budget basis only if any VRAM-budget survives: `min(frac×total, live_free)` vs nerd_herd `free×frac`. (Likely moot — placement, not budget.)

## Where to start P2 (fast path)

1. Ask founder for the objections list. Resolve.
2. `brainstorming` skill → fresh spec for the governor (the parked moon shot). The superseded `docs/superpowers/specs/2026-05-31-vram-aware-load-sizing-design.md` §B/§C/§D encode the WRONG (VRAM-cap) mechanism — mine its evidence/current-state section only.
3. Delete the dead symbols above as the first concrete commit (clean slate).
4. Add RAM + presence sensing to `nerd_herd` as new collectors.
5. Add placement bias to `fatih_hoca/ranking.py`; idle-unload/resume + concurrency throttle as mechanisms.
