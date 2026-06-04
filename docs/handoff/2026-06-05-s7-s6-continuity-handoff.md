# Handoff — S7/S6 Continuity (de-blind burn-rate, reactivate capable-supply)

**Date:** 2026-06-05
**Branch/commit:** merged to `main` @ `3566f595` (linear, 12 commits). Not pushed.
**Status:** SHIPPED to tree, **NOT live until KutAI restart** (signal layer loads at process start).
**Spec:** `docs/superpowers/specs/2026-06-05-s7-s6-continuity-design.md` (§8 outcome).
**Plan:** `docs/superpowers/plans/2026-06-05-s7-s6-continuity.md`.
**Parent:** `docs/handoff/2026-06-04-cloud-utilization-equilibrium-handoff.md` (b6b627bb shipped C1/C2/C3/C6/C7; deferred C4 + S6 on the sim-blocker — this closes that).

---

## 1. Why
Parent work (`b6b627bb`) deferred **C4** (soften S7/S6 `0.70` dead-bands) and **S6 reactivation**
on ONE shared blocker: the Phase 2d sim could not exercise S7 (no `burn_log`, real-clock-vs-virtual
mismatch) or S6 (`eligible_models=[]` hardcoded). We don't ship signal changes the sim can't
validate. This session built the prereq, then de-banded + reactivated, sim-tuned.

## 2. What shipped (12 commits `76a78c3a`→`3566f595`)
**Prereq (sim can now drive S7+S6):**
- `pressure_for(now, burn_log, eligible_models)` default-preserving kwargs (`ee72cdf4`).
- `ranking._apply_utilization_layer` builds a `{capabilities, rpd_remaining}` capable-supply rollup
  from the snapshot + threads `now`/`burn_log` (`cbe2cdeb`); sim selector threads them, pick carries
  `provider`, stubs carry caps+rpd (`4aaa677d`).
- Sim clock made consistent: `now = wall_anchor + virtual_clock` threaded in; `reset_at` made
  ABSOLUTE so S9 reset-proximity is unchanged while S7's 300s window evicts correctly; per-pick
  `BurnLog`; `by_capability` demand (`7cb09074`).

**C4 + reactivation:**
- Shared `smoothstep` → `signals/_curves.py` (`76a78c3a`); S12 aliases it.
- S7 `0.70` dead-band → `-smoothstep(min(1, ratio/SAT))`, `SAT=1.0` (`63b11459`).
- S6 same ramp + reactivated now that it's fed (`cd51894a`).

**C7 modifier fix (the real tuning finding) — `3566f595`:**
- Wiring `burn_log` made S7 LIVE in the sim for the first time. It exposed that on **hard** (d≥7)
  tasks S7 conserved the depleting **paid** pool and diverted hard work to under-qual models →
  `claude_constrained` hard satisfaction 100%→70%. Fix: `M3_difficulty_weights` sets `S7 = 0.0` for
  `d≥7 and model_is_paid` (free/local keep 1.0). Mirrors existing S6(0.7)/S12(0.5) hard
  down-weighting — "right tool must win on hard." 0.5 was insufficient (worst-wins); must be 0.0.

## 3. Knob: SAT = 1.0 (sweep-insensitive)
SAT-sweep {0.8,1.0,1.2,1.4}: rp1 free_q invariant 28.1%, hard 100%, waste 0%, rp5 PASS everywhere —
rp1's gemini tanks are tiny (S7 saturates regardless), groq huge (S7≈0), rp5 balance is S12-driven.
SAT=1.0 kept; de-band validated non-regressive; `ratio**2` fallback not needed.

## 4. Validation (evidence)
- `run_scenarios.py`: rp1 free_q **28.1%** (parent 28.4%, noise), hard **100%** all rp*, waste 0%;
  diverse_pool free_q 93.3%; ALL PP1–PP9 + s7_continuity + s6_conserve + rp5 PASS.
- `run_swap_storm_check.py`: clean (0–0.5% swaps, stickiness intact).
- Suites: **nerd_herd 274**, **fatih_hoca 375 (+1 skip)**, **kuleden 168** — green. 6 new test files
  (curves, pressure_for_threading, ranking_s6_rollup, s7_continuity, s6_continuity, m3 paid-hard).

## 5. Honest scope / caveats
- **rp5 is NOT a pure S7-overdraw scenario.** `select_for_simulation` always injects an attractive
  loaded-local, so free-vs-free can't be forced; rp5 was redesigned as a fleet-balance +
  anti-exhaustion + liveness guard (pick-count measured, reset-proof). **S7 de-blinding is proven at
  the UNIT level** (`test_s7_continuity`). Free-vs-free spread is covered by rp1 + pp9.
- **Original rp5 was mis-designed** (mine): `reset_at=7200s` was crossed by ~7500s virtual runtime →
  `maybe_reset_buckets` refilled mid-run, and the assertion read post-reset `remaining` (artifact).
  Fixed: 45-call tanks, reset 2h, measured by pick counts.
- Utilization is still a **bounded** ±K=1.0 layer; gemini's lead is mostly cap_score (parent §6).
  This work removes the exhaustion tail + de-blinds overdraw; it does not dethrone higher-cap models.

## 6. Latent / deferred (noted, not fixed)
- **`reset_at = int(wall_anchor + ...)` truncation** in the sim is sub-second sensitive in principle;
  not biting (table runs deterministic across invocations). A fixed module-constant `wall_anchor`
  would make the harness fully reproducible — cheap future hardening if flakiness ever appears.
- Parent §7 latents still open: M1 small-tank bias; discovery `status=ok + models=[]` overwrite guard.

## 7. First action next session
Restart-gated: confirm live — next mission's `model_pick_log` should show free-provider diversity
holding (gemini/groq/openrouter regaining share) and **no "No model candidates available" tail**.
The §6 sim prereq is now DONE, so any future S6/S7 tuning is sim-tunable: edit `SAT` (or M3), re-run
`run_scenarios.py` + `run_swap_storm_check.py`, like S12/S7 were.
