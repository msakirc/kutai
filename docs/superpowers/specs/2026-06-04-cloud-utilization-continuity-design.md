# Cloud Utilization Equilibrium — Continuity Restoration & Pool-Total Signal

**Date:** 2026-06-04
**Status:** Design (pre-implementation)
**Owner:** Fatih Hoca / Nerd Herd signal layer
**Related:** `docs/architecture/fatih-hoca-phase2d-equilibrium.md`, memory `project_cloud_diversity_collapse_20260604`, `project_no_models_available_gate_20260603`

---

## 1. Problem

Cloud model diversity has collapsed onto gemini. Measured (last 14d, `model_pick_log`):
avg `picked_score` gemini **152.6** vs openrouter **49.8**; openrouter distinct
models/week 35 → 24 → 0 → 0 → 1 → 1; groq 12 → 3. Two symmetric failures of the
utilization equilibrium — the "medallion" — both firing at once:

- **Idle-rot pan:** openrouter / groq / cerebras sit on free quota that resets unused
  while gemini carries everything.
- **Exhaustion pan:** gemini is ridden to depletion → "No model candidates available"
  with no warmed fallback (peers' S10 priors stayed cold from starvation).

This is **not** a discovery/catalog problem (openrouter cache held 25 free models on
2026-06-04) and **not** rate-blocking (openrouter recent picks 8/8, 10/10 success).
It is a **selection** failure: the abundance side of the equilibrium cannot tilt the
fleet toward idle capacity.

### Design intent being violated (founder's frame)

> "core frame is still utilization. 'don't waste Claude on d3 in the first minute' is
> the same as 'don't bother with 9b for d3 if we're wasting quota that resets in 20
> minutes' — two sides of one equilibrium. If we don't balance it perfectly it falls
> to one side = we failed."

> "I tried to build everything as continuous signals instead of '<1h'-like gates, so
> we wouldn't see 'until too late' cases."

Both principles are currently broken in code.

---

## 2. Root cause (verified in code)

### 2a. Continuous → windowed regression (the "too late" mechanism)
`nerd_herd/signals/s9_perishability.py` (the docstring's "equilibrium core") free-cloud arm:
```python
FREE_CLOUD_PROXIMITY_WINDOW_SECS = 3600.0
proximity = 1.0 - min(1.0, soonest_reset / FREE_CLOUD_PROXIMITY_WINDOW_SECS)
```
Past 1h to reset → `proximity = 0`. The 2026-05-03 refactor **replaced the original
continuous `exp(-reset_in / 24h)` decay with a hard 1h window**, justified by *"0.96 at
1h-out was barely a signal."* That is a **tuning** complaint (τ too long → too flat),
wrongly fixed with a **gate**. Result: free-cloud perishability is invisible all day,
then ramps only in the final hour — exactly the overshoot the founder engineered out.
A bang-bang controller overshoots by construction.

### 2b. Same anti-pattern, two more signals (dead-bands)
- `s7_burn_rate.py`: `THRESHOLD = 0.70`; `excess = max(0, ratio - 0.70)` → **flat 0
  below 70%**, ramp above. Overdraw is silent until already hot. Also **big-tank-blind**:
  gemini's ~1500/day tank keeps `extrapolated/remaining` under 0.70 until very late, so
  S7 never warns in time → feeds the exhaustion pan.
- `s6_capable_supply.py`: identical `THRESHOLD = 0.70` dead-band.

### 2c. Scale-invariant frac owns abundance it didn't earn
`s1_remaining.py` `time_bucketed` profile: `abundance_max = 1.0`, `abundance_mode =
"proportional"` → abundance `= frac × 1.0`, `frac = remaining ÷ own_limit`.
Scale-invariant: rewards "own tank full" identically regardless of tank **size**.
gemini's huge tank sits near frac=1.0 → near-max abundance permanently; small-tank
openrouter loses abundance the instant it is used (self-penalizes). The combine
positive arm is **noisy-OR over S1 + S9 only**, so a full-tank gemini pins the positive
total at ~1.0 → **no headroom** for any fleet/pool-total term to express "openrouter is
more under-utilized."

### 2d. Missing axis: absolute / fleet-comparative amount
S9's 2026-05-03 refactor dropped amount-weight with comment *"magnitude lives in S1."*
But S1's magnitude is **frac**, not absolute. So absolute wasted-capacity fell through
the S9 → S1 seam and is represented by **no signal**.

### 2e. The one fleet-aware signal is dead
`s6_capable_supply` already builds a fleet rollup (`_supply_for` sums `rpd_remaining`
across capable models) — but `types.py:356` calls it with **`eligible_models=[]`
hardcoded** → S6 always returns `0.0`. The rollup machinery exists; it is not fed.

### 2f. Compounding biases (note, not primary)
- `M1_capacity_amplifier`: `2.0 − 0.5·log10(limit)` → small pools amplify negatives
  (limit=10 → 1.5×), big pools dampen (limit=1000 → 0.5×). On the **negative** arm this
  makes small-tank openrouter look more distressed and gemini safer — another thumb on
  gemini's side.
- `M3_difficulty_weights`: easy tasks (d≤3, bulk of volume) up-weight S9 **1.5×** and S6
  **1.5×** — landing on a windowed-to-0 S9 and a dead S6. Hard tasks (d≥7) down-weight
  free-cloud S9 to **0.7×** and up-weight paid S9 to 1.5× (leans premium).

---

## 3. Design principles (restore)

1. **Continuous, always-on.** No signal may be flat-zero until a boundary then ramp.
   Every signal emits a graded value at all times, monotonic in its driver. Tune slope
   via time-constant τ, never via a gate.
2. **Three orthogonal axes on the positive arm, one job each** (preserve the clean
   signal contract from the 05-03 split, but complete it):
   - **frac / stock** (S1) — conservation. Per-provider fullness. Negative arm only.
   - **timing** (S9) — how soon unused capacity vanishes. Continuous reset-proximity.
   - **pool-total / amount** (NEW S12) — how much absolute capacity is wasted,
     fleet-comparative, and whether *this* provider is under-drained vs its schedule.
3. **Survives being used.** The allocation pull must not collapse the moment a small-tank
   provider is touched (the current frac bug). Under-drain is measured against a drain
   schedule, so using a provider a little still leaves it under-drained until it catches up.
4. **Fleet-comparative, not self-normalized.** The signal must answer "where should the
   *next* call go across the fleet," not "is my own tank full."

---

## 4. Changes

### C1 — Restore S9 continuity (de-window)
`s9_perishability.py` free-cloud arm: replace the 1h linear window with continuous decay.
```python
# τ tuned for slope the 05-03 author wanted WITHOUT a cliff:
#   exp(-reset_in / τ):  τ=6h → 0.85 @1h, 0.37 @6h, 0.14 @12h, 0.06 @18h (still > 0)
TAU_SECS = 6 * 3600.0
proximity = math.exp(-soonest_reset / TAU_SECS)
```
- Keep the existence check (`has_remaining`) — nothing to flush if remaining ≤ 0.
- τ is the single tuning knob; default 6h, swept in sim (§7). Steeper than 24h (fixes
  "barely a signal"), continuous (fixes "too late").
- Local + paid arms unchanged.

### C2 — New S12: pool-total absolute-idle, continuous, fleet-comparative
New `signals/s12_pool_idle.py`. Purpose: the allocation pull. Recommended form:
```
under_drain_p = frac_remaining_p − frac_expected_p(t)        # >0 ⇒ behind schedule
frac_expected_p(t) = max(0, 1 − elapsed_in_cycle / cycle_len) # linear drain schedule
stake_p = idle_p / max(1, task_call_need)                     # enough to serve task?
S12_p = clamp( smoothstep(under_drain_p) × min(1, stake_p) , 0, 1)
```
- **Direction** from `under_drain_p`: gemini (hammered) is over-drained vs schedule →
  ~0; openrouter (untouched) is under-drained → high. This is what survives being used.
- **Stake** uses absolute idle **relative to one task's call need**, not idle-share —
  so a small tank with plenty for the task still scores full (avoids "biggest tank wins"
  AND avoids "small tank ignored"). `min(1, stake)` saturates once there's clearly enough.
- `smoothstep` (continuous, no threshold) maps under_drain ∈ [0, ~0.3] → [0, 1].
- **Data:** needs fleet/per-provider cycle + idle. Reuse the S6 rollup approach; see C5.

> **Open decision (sim arbitrates):** keep S12 separate (recommended — honors the
> founder's "separate signals for frac and pool-total" and the one-job contract) vs
> fold the amount back into S9. Separate is preferred; folding re-conflates timing and
> magnitude that 05-03 cleanly split. Spec assumes separate S12.

### C3 — Demote S1 positive arm (stop saturating noisy-OR)
`s1_remaining.py` `time_bucketed`: `abundance_max: 1.0 → 0.0` (mirror the existing
`per_call` profile, which already does this and documents why). Keep `depletion_max =
-1.0` and `depletion_threshold = 0.30` unchanged — **S1 stays full-strength on the
negative/conservation arm.** Effect: frac no longer claims abundance; the positive arm
becomes **S12 + S9**, both continuous, leaving headroom for S12 to steer.

### C4 — Soften S7 / S6 dead-bands to ramp-from-0
Replace `excess = max(0, ratio − 0.70)` with a continuous ramp from 0:
```python
pressure = -smoothstep(ratio / SAT)        # SAT≈1.0; graded from ratio=0 upward
```
- De-blinds gemini-overdraw: S7 whispers as burn rises, instead of shouting near exhaustion.
- Apply identically to S6's capability-shortage ramp.
- Keep these on the **negative** arm (conservation) — they are not abundance.

### C5 — Wire the fleet rollup (fix dead S6 + feed S12)
`types.py` `pressure_for` currently passes `eligible_models=[]` to S6. Build a
per-snapshot fleet view once (not per-model) and thread it in:
- Add `SystemSnapshot.fleet_idle: dict[provider, ProviderIdle]` populated where the
  snapshot is assembled (nerd_herd adapter already builds per-provider `CloudProviderState`
  — add a rollup: `idle`, `limit`, `cycle_len`, `reset_at` per free provider).
- Pass the real eligible-model list to S6 (revives it) and the fleet idle map to S12.
- One rollup per snapshot tick; signals read it. No new network calls.

### C6 — combine.py
- Add `S12` to `OTHER_BUCKET` **and** `POSITIVE_ARM_SIGNALS` → `("S1"→removed-from-pos via
  C3, "S9", "S12")`. Net positive arm = **noisy-OR(S9, S12)**.
- Leave noisy-OR composition as-is (continuous, reinforcing).

### C7 — Modifier interactions to re-verify (no change yet; assert in sim)
- **M2** (`M2_perishability_dampener`) gates on `s9_value` thresholds (0.5 / 0.2). With S9
  now continuous and nonzero most of the day, overqualified-model damping will relax more
  often. Confirm this doesn't let overqualified models win easy tasks (should be checked
  by the d≤3 sim scenarios). If needed, switch M2's gate to a continuous function of S9
  too (same principle) — flagged, not mandated.
- **M3** weights: leave numeric weights; just confirm the now-live S9/S6/S12 don't
  over-fire on easy tasks given the 1.5× up-weight.
- **M1**: leave for now; note its small-pool negative amplification slightly opposes
  redistribution. Re-tune only if sim shows it dominates.

---

## 5. What this fixes, per pan
- **Idle-rot:** S12 (continuous, fleet-comparative, survives-use) + de-windowed S9 emit a
  graded "use openrouter/groq now" pull **all day**, not just pre-reset. S1 no longer
  drowns it. Cross-provider comparable-cap models (e.g. openrouter `gpt-oss-120b:free` vs
  gemini-flash) start winning their share → peer S10 priors warm.
- **Exhaustion:** softened, de-blinded S7 raises gemini-overdraw pressure **early**,
  shifting load before depletion → warmed peers exist as fallback → kills the "No model
  candidates available" tail (see `project_no_models_available_gate_20260603`).

---

## 6. Honest caveats (do not oversell)
- Utilization is a **bounded** layer: `composite *= 1 + UTILIZATION_K · scalar`, K=1.0 →
  at most ~2× / down to 0. The measured gemini lead (152.6 vs 49.8 ≈ 3:1) is **mostly
  cap_score** (gemini-3.x genuinely out-caps obscure openrouter free models). So this work
  redistributes primarily among **comparable-cap** cross-provider models, warms cold
  priors, and removes the exhaustion tail. It will **not** dethrone gemini-3.x on hard
  tasks where capability legitimately wins — nor should it.
- Harder redistribution would need a **K raise** or **cap-parity** change — a separate
  lever, explicitly out of scope here.
- Free catalogs rotate upstream (openrouter dropped `minimax-m2.5:free`, its former #1
  workhorse). No signal change recovers a model the provider removed.

---

## 7. Validation (mandatory before merge)
Per CLAUDE.md, re-run after every tuning change:
- `packages/fatih_hoca/tests/sim/run_scenarios.py`
- `packages/fatih_hoca/tests/sim/run_swap_storm_check.py` (`run_scenarios.py` neighbor)

New stateful scenarios to add:
1. **Giant-tank vs many-small-idle-tanks.** 1 provider limit≈1500/day at high frac +
   3 providers limit≈50–1000/day fully idle. Drive a steady stream of comparable-cap
   tasks. **Assert load spreads across all four, monotonically, from the start of the
   day — not concentrated in the final pre-reset hour.** (Directly tests continuity.)
2. **Overdraw early-warning.** Hammer the giant tank. **Assert S7 pressure rises
   smoothly and load shifts to peers before remaining hits 0** (no "no candidates").
3. **Continuity probe.** Sample S9 + S12 hourly across a full reset cycle for an idle
   provider. **Assert both are strictly > 0 and monotonic at all hours** (no flat-zero
   region, no cliff).
4. **τ sweep.** Run scenario 1 across τ ∈ {3h, 6h, 9h, 12h}; pick the smallest τ that
   spreads load all day without starving the high-cap provider on hard tasks.
5. **Regression guard.** d≤3 overhead tasks must not start preferring overqualified
   premium models (M2 interaction, C7).

Also: unit tests for `exp` continuity in S9, `smoothstep` ramps in S7/S6/S12, and S12
under-drain direction (over-drained → ~0, under-drained → high).

---

## 8. Rollout / risk
- All changes are signal-layer; no dispatcher, KDV, or registry changes. Reversible per
  signal (each is an isolated module + combine wiring).
- Ship behind the existing Phase 2d sim gate; do **not** rely on live observation alone —
  the live effect is masked by cap dominance (§6).
- Sequence: C5 (data path) → C1 (S9 continuity) → C3 (S1 demote) → C2 (S12) → C4 (S7/S6)
  → C6 (combine wiring) → C7 (verify modifiers). Land C1+C3 first; they are the highest-
  leverage, lowest-risk continuity fixes and can be validated independently of S12.

---

## 8b. Implementation outcome (2026-06-04, branch `fix/cloud-utilization-continuity`)

Built C1 + C2 + C3 + C6 + C7. **C4 (soften S7/S6 0.70 dead-bands) deferred** —
C1+C2+C3 already fixed the rp1 collapse; S7/S6 softening is the exhaustion-tail
hardening and can land separately.

**Knobs tuned by sim (not guessed):**
- **τ = 6h** (`FREE_CLOUD_DECAY_TAU_SECS`). τ-sweep {3,6,9,12}h: rp1 free_q peaks
  28.4% at τ=6–9; τ=3 under-uses (19%), τ=12 over-drains pools 24h from reset
  (exhaustion_seq 17%→61%) — wasting the local sunk-cost GPU. τ=6 keeps the
  steepest genuine perishability gradient (.85@1h .37@6h .02@24h) at peak rp1.
- **S12 M3 weights** easy 1.5 / mid 1.0 / hard 0.5; smoothstep ramp; equal-share
  fair allocation (S1 depletion caps small tanks → water-fill emerges).

**S12 data path simpler than §4/C5 anticipated:** `pressure_for` is a
`SystemSnapshot` method, so it can see the whole fleet via `self.cloud`. The
free-provider absolute-consumption rollup is built once in
`ranking._apply_utilization_layer` (the only layer that knows each candidate's
`is_free`) and threaded via a new optional `pressure_for(fleet_consumed=...)`
kwarg — default None ⇒ S12=0, so all pressure-only unit tests are unaffected.
No `SystemSnapshot.fleet_idle` field was needed.

**Results:** rp1_realistic gemini 0→22 picks, free_q 2.5%→28.4%, distribution
spread across local/groq/gemini/claude; hard_task_satisfaction 100% everywhere;
rp2 (gemini burned) keeps gemini at 0 (S1 depletion correct); PP1–PP9 all PASS
(PP9 added — pressure-only anti-monopoly: idle small tank outpulls over-used
giant); swap-storm check clean. 632 unit tests pass (13 new), kuleden 168 pass.

**Not live until KutAI restart** (signal layer loaded at process start).

## 9. Out of scope
- Cap-parity / UTILIZATION_K changes.
- Discovery `status=ok + models=[]` empty-cache-overwrite guard
  (`cloud/discovery.py:63-68`) — separate latent bug, tracked in memory, not biting now.
- OpenRouter free-catalog rotation (upstream).
