# Spec — Phantom −1.0 fleet-stall: root architecture + fix options

**Date:** 2026-06-17
**Status:** investigation + spec (no code yet — owner chose "investigate deeper first")
**Author context:** triggered by recurring model-selection stalls across dozens of sessions; latest handoffs `2026-06-16-daily-exhausted-readmit`, `2026-06-17-phantom-pressure-and-port`.
**Memory:** `project_phantom_s10_and_port_20260617`, `project_phantom_remaining_pressure_20260617`, `project_daily_exhausted_selector_blind_20260615`.

---

## 1. The smell, restated

Every few sessions a *new* "phantom" stalls the whole fleet: `selector: all candidates below pressure threshold ... scalars=[gemini=-1.00, groq=-1.00, cerebras=-1.00]` → `select=None` → silent livelock, or `All models failed: No model candidates available`. Each time it's traced to a different signal feeding a bad value, point-fixed, and recurs elsewhere next session.

Recent instances, **all the same mechanism**:
- `01dba42f` — S1: unknown `remaining` → frac 0 → `depletion_max` −1.0
- (memory `phantom_remaining`) — S1: `remaining=None` → `(remaining or 0)` → frac 0 → −1.0
- `e3cc46b3` — S10: capacity 429s fed the reliability window → success rate craters → −1.0, and `provider_prior_rate` spread it to healthy full-quota siblings
- `bb86b00e` — selector read a stale snapshot that didn't carry KDV's daily-exhausted truth

This is the systematic-debugging **Phase 4.5 signal**: ≥3 fixes, each revealing the same coupling in a new place → the architecture is the bug, not any single signal.

## 2. The pipeline, traced end to end

```
KDV state ──► nerd_herd_adapter ──► 8 signals ──► combine_signals ──► gate ──► beckman.next_task
 (truth)      (builds inputs)       (per-model)    (worst-of-neg)     (>-1.0)   (None → livelock)
```

1. **Adapter** `packages/kuleden_donen_var/.../nerd_herd_adapter.py`
   - `_rl` (`:84`) builds each rate-limit cell. `remaining=None` (providers send no rpd/tpd header — gemini/groq/cerebras) → was read as 0 by `(remaining or 0)` downstream. **The recurring S1 phantom origin.** Now patched to `None→limit`, but only for None; the class of "input lies uniformly across the fleet" remains.
   - `provider_prior_rate` (`:162-177`, `:271`) feeds S10's cold-start fallback by **aggregating outcomes across all provider siblings**. A fleet-wide amplifier: one model's dip → every sibling reads the same low prior → all −1.0 together.
   - **Genuine keep-out is surfaced here as eligibility flags**: `daily_exhausted` (`:253`), `rpm_cooldown` (`:263`), `circuit_breaker_open` (`:289`). These are classification-driven and already correct.

2. **8 signals** `packages/nerd_herd/src/nerd_herd/signals/`

   | Signal | −1.0 source | Class | Can stall whole cloud fleet? |
   |---|---|---|---|
   | S1 stock | `frac < depletion_threshold` | **phantom-prone** (adapter inputs) | **yes** |
   | S7 burn | extrapolated burn > remaining | phantom-prone (rate spike) | per-model |
   | S10 reliability | `rate ≤ 0.20` | **phantom-prone** (capacity outcomes, sibling prior) | **yes (siblings)** |
   | S11 cost | bite > daily cost remaining | budget guard | paid only |
   | S9 perish | `LOCAL_BUSY_PENALTY −10` | **intentional** hard veto | local only |
   | S12 balance | never negative (smoothstep ≥0) | n/a | no |
   | S13 presence | `FULLSCREEN_VETO −10` / graded −0.6 | **intentional** | local only |
   | S14 contention | ext-GPU −10 / RAM −1 | **intentional** | local only |

   **Two distinct kinds of −1.0 are conflated:**
   - **Intentional sentinels** (`−10` family: S9 busy, S13 fullscreen, S14 ext-GPU) — deliberate "never admit," local-only, designed to survive weights and peg the scalar.
   - **Accumulated soft pressure** (S1/S7/S10/S11 hitting their negative cap) — "this model is stressed/degraded," a *ranking* signal that gets *pinned* to the floor by edge-case inputs.

3. **combine** `packages/nerd_herd/src/nerd_herd/combine.py`
   - `OTHER_BUCKET = (S1,S7,S9,S10,S11,S12,S13,S14)`, `W_OTHER=1.0`, `other_neg = min(...)` (`:17,34`). **Any single signal at −1.0 pins the whole model scalar to −1.0.** No quorum, no distinction between sentinel and soft.

4. **gate** `packages/fatih_hoca/src/fatih_hoca/selector.py:370-404`
   - `threshold = max(-1.0, -0.5 - 0.5*urgency)`; keep `urgency >= threshold AND urgency > -1.0`.
   - If none survive → `return None`. Comment explicitly forbids relaxing the threshold (that's how dead models got re-admitted). **No least-bad fallback, no anomaly detection.**

5. **beckman** `packages/general_beckman/src/general_beckman/__init__.py:906-911`
   - `select=None` → `_log.debug(... select=None ...)` → `continue`. Task stays pending, **no attempt increment, no DLQ, no urgency escalation, no alarm.** Fingerprint cache (`:620`) then skips the whole scan while state is unchanged. **Silent total stall** until a human reads logs.

## 3. Root structural defect

> **`−1.0` is an overloaded sentinel.** It means BOTH "genuinely depleted, must never admit" AND is the accidental output of any of 4 cloud signals fed a bad input. Because combine is worst-of-negatives at weight 1.0, a single signal pins the scalar; because the gate trusts −1.0 absolutely and has no floor, a uniform bad input becomes a silent fleet-wide outage. And the genuine keep-out it's *supposed* to protect is **already enforced upstream at eligibility** — so the pressure veto is a redundant, drift-prone second gate that contributes nothing but phantoms.

Point-fixing each signal's input edge case will never converge: 4 cloud signals × unbounded inputs (None, missing headers, empty matrix, restart races, sibling aggregation) = endless supply.

## 4. Fix options (tradeoffs)

### Option A — Anomaly floor + detector (safety net, cheap)
At the gate, when *every eligible* candidate is pressure-vetoed: that is the phantom signature (real depletion is staggered, not uniform). Instead of `return None`:
- pick the **least-pressured eligible** model,
- emit a loud diagnostic + Telegram alert (`fleet-wide pressure veto — admitting least-bad, likely phantom`),
- record an `admission_violations`-style fingerprint for offline audit.

Eligibility hard-excludes (daily_exhausted/rpm_cooldown/circuit-open/local sentinels) are honoured *before* this fallback, so it can never re-admit a genuinely dead model.

- **Pros:** small, localized to `selector.py`; catches *every* future phantom regardless of which signal; turns silent outage into loud-but-serving. Doesn't touch signal/combine math.
- **Cons:** masks the underlying signal bug (still want it found via the alert); needs a crisp "eligible but pressured" vs "ineligible" split so the fallback never admits a flagged-dead model.

### Option B — De-overload the sentinel (structural fix)
Make pressure **rank, not veto**:
- Pressure scalar orders candidates and sets a *backpressure* threshold for deferring low-priority work — but can **never empty the fleet**.
- Move the genuine hard keep-outs to **eligibility** where they already mostly live: cloud = daily_exhausted/rpm_cooldown/circuit-open (already there); local = S9-busy/S13-fullscreen/S14-ext-GPU promoted from `−10`-laundered-through-combine to explicit eligibility excludes (local-only, so cloud is never emptied by them).
- The soft caps of S1/S7/S10/S11 stop being −1.0 hard vetoes — they just push a model down the ranking.

- **Pros:** removes the phantom *vector* entirely; one keep-out mechanism (eligibility, classification-driven) instead of two drifting ones — exactly the "singular selection mechanism" the owner asked for, finished properly. Genuine exhaustion still enforced.
- **Cons:** biggest change; touches the gate semantics consolidated in the 2026-04-30 triage; needs the local sentinels carefully re-homed so single-GPU serial safety (S9) and minimal-mode behaviour are preserved; full sim re-run (`run_scenarios.py` + `run_swap_storm_check.py`) mandatory.

### Option C — Both (A as net, B as fix)
B is the real cure; A is the seatbelt that also catches *new* signals added later. Ship A first (stops bleeding immediately, low risk), then B.

### Adapter-layer hardening (orthogonal, do regardless)
- **Input invariant at the adapter boundary:** a cloud model that is `status=active ∧ ¬daily_exhausted ∧ ¬rpm_cooldown ∧ ¬circuit_open` must never emit signal inputs that yield −1.0. Assert/clamp at `build_cloud_provider_state` and log when violated — turns "found by hand weeks later" into "caught at the source on the first tick."
- **provider_prior amplifier:** cap how far a sibling-aggregated prior can pull an *individual* model that has its own (healthy) headroom, so one bad sibling can't −1.0 the group.

## 5. Recommendation

**Option C, A-first.** A is a few lines in `selector.py`, near-zero risk, and converts the entire class of failure from "silent multi-hour outage" to "loud alert + degraded-but-serving" — immediately, for every phantom including ones not yet discovered. B is the structural cure (kill the redundant veto, one keep-out at eligibility) but is a deliberate, sim-gated change to the consolidated gate and should be its own reviewed effort. Adapter hardening rides along with whichever lands first.

This holds the owner's invariants: *no band-aids* (A is a safety net, not a band-aid on a specific signal; B is the principled fix), *one selection mechanism* (B finishes the consolidation by removing the second gate), and *never admit a model that 429'd minutes ago* (eligibility flags still gate that — untouched).

## 6. Open decisions for owner

1. A-first then B, or go straight to B?
2. For A's fallback: alert via Telegram every time, or rate-limited summary? (phantom can fire many ticks)
3. For B: are the three local sentinels (S9/S13/S14) acceptable to re-home as explicit eligibility excludes, or is there a reason they must stay in the pressure scalar?

## 7. Key files

- `packages/kuleden_donen_var/src/kuleden_donen_var/nerd_herd_adapter.py` — `_rl` (:84), `provider_prior_rate` (:162-177,:271), eligibility flags (:253,:263,:289)
- `packages/nerd_herd/src/nerd_herd/combine.py` — `OTHER_BUCKET`, `other_neg` worst-of-neg (:17,34)
- `packages/nerd_herd/src/nerd_herd/signals/s1_remaining.py`, `s10_failure.py`, `s7_burn_rate.py`, `s11_cost.py` (cloud-fleet vectors); `s9_perishability.py`, `s13_presence.py`, `s14_contention.py` (local sentinels)
- `packages/fatih_hoca/src/fatih_hoca/selector.py:370-404` — pressure gate, `return None`
- `packages/general_beckman/src/general_beckman/__init__.py:906-911` — `select=None` → silent `continue`; fingerprint cache `:620`
- sims: `packages/fatih_hoca/tests/sim/run_scenarios.py`, `run_swap_storm_check.py`
```
