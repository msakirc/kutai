# Handoff — Cloud Utilization Equilibrium (continuity + fleet balance)

**Date:** 2026-06-04
**Branch/commit:** merged to `main` @ `b6b627bb` (fast-forward, linear). Not pushed.
**Status:** SHIPPED to tree, **NOT live until KutAI restart** (signal layer loads at process start).
**Spec:** `docs/superpowers/specs/2026-06-04-cloud-utilization-continuity-design.md`
**Memory:** `project_cloud_diversity_collapse_20260604`

---

## 1. Why (the symptom)

Cloud model diversity collapsed onto one provider (gemini in prod, groq in the
sim's cap numbers). Free quota on under-used providers rotted until reset while
the incumbent burned to exhaustion → "No model candidates available" with no
warm fallback. Both pans of the utilization "medallion" had fallen to one side
at once: idle quota wasted AND premium over-drawn.

Evidence: avg `picked_score` gemini 152.6 vs openrouter 49.8 (3:1); openrouter
distinct models/week 35→24→0→0→1→1; weekly cloud picks ~10k (W17, a retry-storm
week) → <900 normal — most of the *absolute* drop was workload normalization,
but the **provider-diversity collapse** is the real regression.

## 2. Root cause (verified in code, not guessed)

The abundance / positive arm was **scale-invariant**:
- `s1_remaining.py` free-cloud abundance = frac-of-OWN-limit → a giant-tank
  provider sits near frac=1.0 forever and pinned the noisy-OR positive arm,
  leaving no headroom to steer toward idle providers.
- `s9_perishability.py` had been changed (2026-05-03) from continuous
  `exp(-reset_in/24h)` decay to a **hard 1h window** → perishability invisible
  all day, cliff in the final hour = the "until too late" overshoot (violates
  the founder's continuous-signal principle).
- **No signal carried absolute / fleet-comparative wasted capacity at all** —
  S9 had delegated "magnitude" to S1, but S1's magnitude is frac, so it fell
  through the seam.

## 3. What shipped (in `b6b627bb`)

| Change | File | Effect |
|---|---|---|
| C1 | `signals/s9_perishability.py` | Continuous `exp(-reset_in/τ)`, **τ=6h** (`FREE_CLOUD_DECAY_TAU_SECS`). Reverts the 1h gate. |
| C3 | `signals/s1_remaining.py` | `time_bucketed abundance_max 1.0→0.0` — frac is conservation-only; depletion arm unchanged. |
| C2 | `signals/s12_pool_balance.py` (new) | Fleet-relative ABSOLUTE-consumption balance: pull the free provider that took fewest absolute calls toward equal share; smoothstep, no gate; paid/local=0. |
| C2-wire | `fatih_hoca/ranking.py` + `types.py` | Fleet rollup built once in `_apply_utilization_layer` (only layer that knows `is_free`), threaded via `pressure_for(fleet_consumed=...)`. Default None → S12=0 (pressure-only tests unaffected). |
| C6 | `combine.py` | Positive arm `(S1,S9)`→`(S9,S12)`; S12 added to `OTHER_BUCKET`. |
| C7 | `modifiers.py` | S12 M3 weights easy 1.5 / mid 1.0 / hard 0.5. |

**Knob tuning (by sim τ-sweep, per mandate):** τ=6h. {3,6,9,12}h tested — rp1
free_q peaks 28.4% at τ=6–9; τ=3 under-uses; τ=12 over-drains pools 24h from
reset (wastes the local sunk-cost GPU). τ=6 = peak rp1 + steepest genuine
perishability gradient (.85@1h .37@6h .02@24h).

## 4. Validation (evidence)

- `packages/fatih_hoca/tests/sim/run_scenarios.py`: rp1_realistic **gemini
  0→22 picks, free_q 2.5%→28.4%**, spread local/groq/gemini/claude;
  hard_task_satisfaction **100%** everywhere; rp2 (gemini burned) keeps gemini
  at 0 (S1 depletion correct); rp3/rp4 degrade to local+claude.
- **PP1–PP9 all PASS** (PP9 added: pressure-only anti-monopoly — idle small tank
  outpulls over-used giant tank of identical cap).
- `run_swap_storm_check.py`: clean (0–0.5% swaps; local stickiness intact).
- Unit: **632 pass** (13 new — `tests/signals/test_s12.py`,
  `test_s9_continuity.py`; updated 7 stale tests). kuleden **168 pass**.

## 5. Honest scope / caveats

- Utilization is a **bounded** ±K=1.0 (≤2×) multiplier. The gemini lead is
  mostly **cap_score** (3:1). So this redistributes **comparable-cap**
  cross-provider work + warms cold S10 priors + kills the exhaustion tail. It
  will **not** dethrone genuinely-higher-cap models (gemini-3.x) on hard tasks
  — nor should it. Harder redistribution = a K-raise or cap-parity lever
  (separate decision, out of scope).
- Free catalogs rotate upstream (openrouter dropped `minimax-m2.5:free`, its
  former #1 workhorse). No signal change recovers a model the provider removed.

## 6. Deferred follow-ups — ONE shared blocker

Both items below are real, both flagged in §4/§8b of the spec, and **both gated
on the same prerequisite**, which is why this is one handoff:

> **PREREQ (do first): teach the sim to exercise burn-rate + capability supply.**
> `packages/fatih_hoca/tests/sim/runner.py` does not populate `burn_log`, and
> `types.py:pressure_for` is called with `eligible_models=[]`. Until the sim
> feeds a rolling burn rate + a live eligible-model list, neither follow-up is
> tunable — and we do NOT ship signal changes the sim can't validate.

- **C4 — soften S7 (and S6) `0.70` dead-bands → continuous ramp-from-0.**
  `s7_burn_rate.py` / `s6_capable_supply.py` use `excess = max(0, ratio-0.70)`
  → flat-zero until hot, then ramp = the same bang-bang anti-pattern. Softening
  S7 de-blinds gemini-overdraw *early* (exhaustion-tail fix). **Blocked:** S7 is
  always 0 in the sim (no burn_log) → untunable today.
- **S6 reactivation.** `s6_capable_supply` is DEAD: `types.py:356` passes
  `eligible_models=[]` → always returns 0. It already builds a fleet rollup
  (`_supply_for`) — reusable machinery — but turning it on introduces a
  conserve-pressure that's been absent. Needs its own validation. **Blocked:**
  same — needs the eligible-models list fed in sim.

**Sequencing:** (1) build the sim burn_log + eligible-models fixture; (2) then
C4 and S6 both become sim-tunable; tune + ship like S12 was.

## 7. Latent findings (noted, not fixed)

- **M1 small-tank bias** (`modifiers.py M1_capacity_amplifier`): amplifies
  small-pool *negatives* (limit=10→1.5×) and dampens big-pool ones (1000→0.5×).
  On the negative arm this makes small-tank providers look more distressed and
  the giant safer — another thumb on the incumbent's side. Re-tune only if a
  future sim shows it dominating; S12 currently offsets it on the positive arm.
- **Discovery empty-cache overwrite** (`fatih_hoca/cloud/discovery.py:63-68`):
  a `status=ok` fetch with `models=[]` (e.g. `OPENROUTER_FREE_ONLY=1` filters
  all) overwrites the cache empty with no alert/fallback (fallback only on
  failure status). Not biting now (catalogs healthy); cheap guard if it recurs.

## 8. First action next session
Restart-gated: confirm the change is live (next mission's `model_pick_log`
should show gemini/openrouter regaining a share, esp. early-day). Then, if
pursuing C4/S6, start with the sim burn_log/eligible-models fixture (§6 prereq).
