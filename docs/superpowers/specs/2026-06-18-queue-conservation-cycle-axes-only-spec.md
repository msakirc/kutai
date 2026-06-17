# Spec — Queue conservation (S4/S5) applies to reset-cycle axes only

**Date:** 2026-06-18
**Status:** implementing (autonomous, sim-gated)
**Owner frame:** the utilization equilibrium — "don't waste the scarce thing, don't waste the perishing thing; balance or fail."
**Related:** `docs/architecture/fatih-hoca-phase2d-equilibrium.md`, `2026-06-17-phantom-veto-architecture-spec.md`, memory `project_phantom_veto_architecture_20260617`.

---

## 1. Problem (the waste pan)

Under a deep queue (high demand), free models get **erased** from ranking while a premium model wins **easy** tasks — the exact "don't waste Claude on a d3" failure.

Mechanism:
- Demand pressure = `S4` (queue tokens) / `S5` (queue calls) = **whole-queue projection ÷ this model's rate window**, folded worst-axis-wins.
- A free model's binding window is **per-minute** (rpm 5–15, tpm small). Deep queue → ratio huge → S4/S5 saturate → scalar → −1 → `composite *= (1 + K·−1) = ×0` → **erased**.
- A premium model has a huge / untracked window → S4/S5 ≈ 0 → **survives**.
- Base composite already scores cheap > premium on easy (cost weighting). So the *only* reason premium wins is that demand **erased the cheap model**. Asymmetric: dent the cheap, spare the expensive — backwards.

This also caused the **total stall** under minimal (cloud-only): every free floored on its per-minute window → `select=None` (addressed defensively by the supply/demand gate split `48e4cee8`; this spec removes the root cause).

## 2. Root cause

**A per-minute window is a *pacing* constraint, not a *conservation* one.** It refills every 60s — a deep queue drains over many minutes; it never "exhausts" a per-minute bucket. Conserving against it (erasing a model because the whole queue can't fit in one minute) is a category error.

The genuine conservation case S4/S5 were built for — *5 planners projected at 40 reqs vs gemini's 20/**day*** — is a **reset-cycle** (daily) axis: unused daily quota perishes at reset, and the queue genuinely can exhaust it before then.

So S4/S5 conflate two axes:
- **cycle** (rpd/tpd/cpd/…): queue can exhaust before reset → conserve (legit).
- **per-minute** (rpm/tpm/itpm/otpm): refills continuously → pace, don't conserve. Owned by lane caps + in-flight reservation, and per-*task* fit is already `S2`/`S3`.

## 3. Change

`s4_queue_tokens` and `s5_queue_calls`: iterate **cycle axes only** — exclude the per-minute axes `{rpm, tpm, itpm, otpm}`. Everything else unchanged (threshold, slope, M1, M3 weights, fold).

- Per-task overshoot ("does THIS task fit a minute's tokens") stays with `S2`/`S3` (unchanged — they still read tpm).
- Per-minute pacing stays with Beckman lane caps + `src.core.in_flight` reservation (unchanged).
- Daily/cycle overshoot ("the queue will exhaust the day's budget") still fires via the cycle axes (rpd/tpd) — overshoot protection retained.

Axis sets are explicit (note `rpm`≠`rpmonth`): `PER_MINUTE = {"rpm","tpm","itpm","otpm"}`.

## 4. Why this fixes it without a chop

- Live waste: free has **full rpd** + small **rpm/tpm** → cycle axes healthy → S4/S5 = 0 → not erased → base composite picks cheap. ✅
- Live stall: same — frees no longer floor on their per-minute window. ✅
- Overshoot (gemini 20/day): rpd is a cycle axis → S5 still fires → conserve. ✅
- No tuned constant touched; supply signals, modifiers, abundance gate, fallback all unchanged.

## 5. Known residual (smaller, deferred)

The aggregate-queue-vs-single-model comparison still exists on the **cycle** axes: a free model with a *small daily* budget + a premium with a *large daily* budget + easy task + deep queue could still erase the free and leak to premium. Rare (most frees have small per-minute but ample/ full daily). The complete fix (queue as pool-level pacing vs fleet capacity, never a per-model erasure — relying on S1-actual + in-flight for overshoot) is a larger change; defer unless sims/live show the residual bites.

## 6. Validation (mandatory — sims are the arbiter)

- Full `run_scenarios.py` + `run_swap_storm_check.py` must stay green — **especially any daily-overshoot scenario** (the S4/S5 raison d'être).
- `pp10` reframed: a free model constrained only by a **per-minute** window under a deep queue must **NOT** floor (scalar > −0.99) — the core of this change.
- New `pp11`: a queue that projects to exhaust a **daily** budget **must** still floor (S4/S5 fire on the cycle axis) — overshoot retained.
- Easy-task-under-demand: cheap free beats premium (the waste check) — covered by the realistic `waste=0%` metric + the gate tests.

## 7. Files

- `packages/nerd_herd/src/nerd_herd/signals/s4_queue_tokens.py` — cycle-axis filter.
- `packages/nerd_herd/src/nerd_herd/signals/s5_queue_calls.py` — cycle-axis filter.
- `packages/fatih_hoca/tests/sim/scenarios.py` — pp10 reframe + pp11.
- (+ unit tests for the per-minute-excluded / daily-included behavior.)
