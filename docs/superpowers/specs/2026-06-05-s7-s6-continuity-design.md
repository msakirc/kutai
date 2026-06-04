# S7/S6 Continuity — De-Blind Burn-Rate & Reactivate Capable-Supply

**Date:** 2026-06-05
**Status:** Design (pre-implementation)
**Owner:** Fatih Hoca / Nerd Herd signal layer
**Related:** `docs/superpowers/specs/2026-06-04-cloud-utilization-continuity-design.md` (parent;
ships C1/C2/C3/C6/C7), `docs/handoff/2026-06-04-cloud-utilization-equilibrium-handoff.md` §6,
memory `project_cloud_diversity_collapse_20260604`

---

## 1. Problem

The 2026-06-04 cloud-utilization work (commit `b6b627bb`) shipped C1+C2+C3+C6+C7 and **deferred
C4 + S6 reactivation** on one shared blocker: the Phase 2d simulator cannot exercise two signals,
so neither is tunable, and we do **not** ship signal changes the sim can't validate.

Two signals remain the same bang-bang anti-pattern the parent spec set out to kill:

- **S7 burn-rate** (`s7_burn_rate.py`): `excess = max(0, ratio − 0.70)` → flat-zero below 70%,
  ramp above. Overdraw is **silent until already hot**. Big-tank-blind: gemini's large daily tank
  keeps `extrapolated/remaining` under 0.70 until very late → S7 never warns in time → feeds the
  exhaustion pan ("No model candidates available", `project_no_models_available_gate_20260603`).
- **S6 capable-supply** (`s6_capable_supply.py`): identical `0.70` dead-band **and structurally
  dead** — `types.py:pressure_for` passes `eligible_models=[]` hardcoded → S6 always returns 0.0.

### The shared blocker (must clear first)

`packages/fatih_hoca/tests/sim/runner.py`:
- Does **not** populate `burn_log` → S7's `burn_log.rate()` returns `BurnRate(0,0)` → S7≡0 in sim.
- `pressure_for` reads real `_time.time()` for `now`; a 182-task sim runs in ~ms of real wall-time,
  so every `burn_log` entry would cluster at one instant — the 300s window never evicts, and S7's
  rate grows unbounded (cumulative, not rolling). The virtual clock never reaches the signals.
- `eligible_models=[]` is hardcoded in `pressure_for` → even a sim-built capability demand can't
  reach S6.

---

## 2. Design principles (inherited from parent spec §3)

1. **Continuous, always-on.** No signal flat-zero until a boundary then ramp. Tune slope via a
   constant (SAT), never a gate.
2. **S7/S6 stay on the negative / conservation arm** — they are not abundance.
3. **Prod behavior unchanged until fed.** New `pressure_for` params default to current behavior;
   only the ranking layer (and the sim) thread real values in.

---

## 3. Changes

### Part A — Prereq: sim exercises S7 + S6

**A1 — Clock injection (S7/S9 fidelity).**
- `SystemSnapshot.pressure_for(..., now: float | None = None)`. Default `None` → `_time.time()`
  (prod unchanged). Thread `now` into: `s7_burn_rate(now=now)`, `s9_perishability(now=now)`, and
  the `reset_in` computation (all already accept/derive from `now`).
- `ranking.rank_candidates(..., now: float | None = None)` and
  `ranking._apply_utilization_layer(..., now=None)` — thread `now` through to `pressure_for`.
- Sim passes `now = wall_anchor + virtual_clock`; prod passes nothing.

**A2 — burn_log in the sim runner.**
- `runner.run_simulation` constructs a **fresh** `BurnLog(window_secs=300.0)` per run (not the
  process singleton — must reset between scenarios) and threads it + `wall_anchor` into `select_fn`.
- After each pick: `burn_log.record(provider=<prov>, model=<model>, tokens=task.estimated_output_tokens,
  calls=1, now=wall_anchor + state.virtual_clock)`.
- `SystemSnapshot.pressure_for(..., burn_log: BurnLog | None = None)`. Default `None` →
  `get_burn_log()` (prod singleton, unchanged). The sim threads its **fresh** per-run BurnLog so
  scenarios never leak burn history into each other. Same kwarg pattern as `now` / `eligible_models`
  — no mutation of the global singleton.
- `scenarios._build_select_fn` accepts the burn_log + wall_anchor and passes them through
  `select_for_simulation` → `rank_candidates` → `_apply_utilization_layer` → `pressure_for`.

> Wall-clock consistency: `_build_snapshot_factory` already projects `reset_at = wall_anchor +
> reset_in_secs`. Recording burn at `wall_anchor + virtual_clock` and passing the same `now` into
> `pressure_for` keeps reset-proximity (S9) and the burn window (S7) on **one** clock. As elapsed
> accrues (`elapsed = tokens/tps`), entries older than 300s evict → true rolling rate.

**A3 — S6 eligible-models + demand.**
- `pressure_for(..., eligible_models: list | None = None)`. Default `None` → `[]` (current
  behavior; all pressure-only unit tests + PP1–PP9 unaffected). Passed to
  `s6_capable_supply(eligible_models=...)`.
- `_apply_utilization_layer` builds the eligible list **once** (the `.model` of each `scored`
  entry — same locus as the existing `fleet_consumed` rollup) and threads it into every
  `pressure_for` call.
- Sim `select_fn` extends its `QueueProfile` build to populate `by_capability` from the remaining
  task tail via a `task_name → capability` map (e.g. `visual_reviewer → vision`; coder/most → none).
  Without `by_capability`, S6 legitimately stays 0 (no shortage).

### Part B — C4: soften S7 / S6 dead-bands to ramp-from-0

**B1 — shared curve.** Extract `_smoothstep` (currently private in `s12_pool_balance.py`) to a new
`signals/_curves.py`; `s12` imports it (behavior identical — assert S12 tests green). S7/S6 import
the same function (no copy-paste drift).

**B2 — S7.** Replace
```python
excess = max(0.0, ratio - THRESHOLD)        # 0.70 gate
pressure = -min(1.0, excess * SLOPE)         # SLOPE 2.0
```
with
```python
pressure = -_smoothstep(min(1.0, ratio / SAT))   # SAT tuning knob, default swept
```
- Continuous from `ratio = 0`; big-tank de-blinded (fires before remaining→0).
- **Risk acknowledged:** smoothstep is nonzero everywhere and steeper than the old curve above
  0.70 → could shave every cloud composite a hair. SAT is swept {1.0, 1.2, 1.4} (like the parent's
  τ-sweep) to keep light idle-burn `< ~0.1`. **Documented fallback** if the sweep shows
  over-penalty: `ratio**2` (gentler low-end than smoothstep) — fallback only, not default.

**B3 — S6.** Same `excess`→`_smoothstep(ratio/SAT)` swap on the capability-shortage ramp. Stays in
`QUEUE_BUCKET` (unchanged), negative-arm conserve-pressure. Reactivation = it is now *fed* (A3).

---

## 4. Validation (mandatory before commit — per CLAUDE.md)

### Unit (TDD — write first, watch fail, then implement)
- `tests/signals/test_curves.py` — extracted `_smoothstep`; S12 import unchanged.
- `tests/signals/test_s7_continuity.py` — monotonic from ratio=0, no flat-zero region, cold-start
  (no burn) → 0, big-tank de-blinding (low remaining frac fires earlier than the old 0.70 gate).
- `tests/signals/test_s6_continuity.py` — graded ramp; dead on empty `by_capability`; fires on
  shortage; `eligible_models=None` → 0 (prod-default guard).
- `pressure_for` threading: `now=` and `eligible_models=` defaults preserve current values across
  all existing PP1–PP9 + pressure-only assertions.

### Sim (per CLAUDE.md — re-run after every tuning change)
- `packages/fatih_hoca/tests/sim/run_scenarios.py`:
  - rp1 free_q holds ~28% (no regression vs b6b627bb), hard_task_satisfaction 100% everywhere.
  - **New** `rp5_overdraw_early_warning`: hammer a giant tank; assert S7 pressure rises smoothly
    and load shifts to peers **before** remaining hits 0 (no "no candidates").
  - **New** `cont_probe_s7`: sample S7 hourly across a burn ramp; assert strictly monotonic, no
    flat-zero region.
  - **New/extended** S6 capability-conserve scenario: S6 fires graded (not bang-bang) and does not
    starve a capability-needed task.
  - Regression guard (parent §7 #5): d≤3 tasks must not start preferring overqualified premium from
    now-live S7/S6.
- `packages/fatih_hoca/tests/sim/run_swap_storm_check.py` — clean (local stickiness intact).
- SAT-sweep {1.0,1.2,1.4} + re-confirm τ=6h still optimal under live S7/S6.

### Full suites
- `nerd_herd`, `fatih_hoca`, `kuleden_donen_var` test suites green (timeouts per CLAUDE.md).

---

## 5. Sequencing
A1+A3 clock/eligible wiring → A2 burn_log in runner → unit tests (fail) → B1 extract curve →
B2 S7 ramp → B3 S6 ramp → new sim scenarios → SAT sweep + tune → full suites → swap storm → commit.

Land Part A independently verifiable (S7/S6 become *observable* in sim before any curve change —
confirm S7≠0 and S6≠0 in a hammered/shortage scenario first).

---

## 6. Risk / rollout
- Signal-layer only; no dispatcher, KDV, registry, or selector-eligibility changes. Reversible per
  signal (isolated module + the `pressure_for` kwarg seam).
- S7/S6 now nonzero almost always (negative arm). Bounded by `UTILIZATION_K = 1.0` and worst-wins
  bucketing (only the single worst negative in each bucket survives). SAT tuned so idle-burn stays
  quiet; `ratio**2` fallback documented.
- **Not live until KutAI restart** (signal layer loaded at process start) — same as `b6b627bb`.

## 7. Out of scope
- Cap-parity / `UTILIZATION_K` changes (parent §6 — separate lever).
- M1 small-tank-bias re-tune (parent §7 latent; re-tune only if a sim shows it dominating).
- Discovery `status=ok + models=[]` empty-cache-overwrite guard (`cloud/discovery.py:63-68`).
- OpenRouter free-catalog upstream rotation.

---

## 8. Implementation outcome (2026-06-05, shipped to `main`)

**Status:** SHIPPED (12 commits `76a78c3a`→`3566f595`, linear on `main`, not pushed).
**NOT live until KutAI restart** (signal layer loads at process start), same as parent `b6b627bb`.

### What shipped
| Part | Commits | Effect |
|---|---|---|
| Shared curve | `76a78c3a` | `_smoothstep` extracted to `signals/_curves.py`; S7/S6/S12 share one impl. |
| Prereq seams | `ee72cdf4` | `pressure_for(now, burn_log, eligible_models)` — default-preserving kwargs. |
| S6 rollup | `cbe2cdeb` | `ranking._apply_utilization_layer` builds `{capabilities, rpd_remaining}` per candidate from the snapshot + threads `now`/`burn_log`. |
| Sim selector | `4aaa677d` | `select_for_simulation` threads `now`/`burn_log`; pick carries `provider`; stubs carry real caps + `rpd_remaining`. |
| Sim clock+burn | `7cb09074` | `now = wall_anchor + virtual_clock` threaded in; `reset_at` made absolute (S9 unchanged); per-pick `BurnLog`; `by_capability` demand. |
| New probes | `3a472a49`, `25a638c2` | S7-continuity, S6-conserve (pressure-only) + rp5 (full-sim fleet-balance/anti-exhaustion/liveness). |
| **C4 S7** | `63b11459` | S7 `0.70` dead-band → `-smoothstep(min(1, ratio/SAT))`, `SAT=1.0`. |
| **C4 S6** | `cd51894a` | S6 `0.70` dead-band → same ramp; S6 reactivated (now fed). |
| **C7 M3** | `3566f595` | **M3 zeroes S7 conserve-weight for PAID models on hard tasks** (see below). |

### Knob: SAT = 1.0 (sweep-insensitive)
SAT-sweep {0.8, 1.0, 1.2, 1.4}: rp1 free_q invariant at **28.1%**, hard 100%, waste 0%, rp5 PASS
at every value — because rp1's gemini tanks are tiny (S7 saturates regardless of SAT) and groq is
huge (S7≈0 regardless), and rp5 balance is S12-driven. SAT=1.0 (spec default) kept; the de-band is
validated as **non-regressive**, its early-warning value proven at the unit level. The `ratio**2`
fallback was not needed (no over-penalty observed).

### C7 modifier interaction discovered (the real tuning finding)
Wiring `burn_log` into the sim made S7 **live in the simulator for the first time** (pre-this-work
the sim never populated `burn_log`, so S7≡0 there). That exposed a latent gap: on **hard** (d≥7)
tasks, S7's conserve-pressure fired on the depleting **paid** pool and diverted hard work to
under-qualified cheaper models → `claude_constrained` hard-task satisfaction fell 100%→70%. Fix
(`3566f595`): `M3_difficulty_weights` now sets `S7 = 0.0` for `difficulty ≥ 7 and model_is_paid`
(free/local keep S7=1.0). This mirrors the existing hard-task down-weighting of S6 (0.7) and S12
(0.5) — "on hard tasks the right tool must win; don't conserve it by failing the task." Halving to
0.5 was insufficient (worst-wins bucketing still diverted); it must be 0.0.

### rp5 honest scope (downgraded from spec §7 #2)
A pure free-vs-free **overdraw isolation** scenario is **not achievable** in this harness:
`select_for_simulation` always injects an attractive loaded-local candidate, so medium work can
always fall to local — free-vs-free can't be forced. rp5 was therefore redesigned as an end-to-end
**fleet-balance + anti-exhaustion + liveness** guard (45-call tanks, reset 2h, measured by PICK
COUNTS — reset-proof): liveness, no free pool at its cap, free peers balanced, free cloud not
abandoned. The **S7 de-blinding itself is proven at the unit level** (`test_s7_continuity`:
de-banded ramp nonzero below the old 0.70 gate, monotonic, saturating). Free-vs-free spreading at
the fleet level is already covered by rp1 (gemini 0→share) and pp9 (anti-monopoly).

### Validation evidence
- `run_scenarios.py`: rp1 free_q **28.1%** (vs parent 28.4% — within noise), hard 100% on ALL rp*,
  waste 0%; diverse_pool free_q 93.3%; all PP1–PP9 + s7_continuity + s6_conserve + rp5 PASS.
- `run_swap_storm_check.py`: clean (0–0.5% swaps; local stickiness intact).
- Unit suites: **nerd_herd 274**, **fatih_hoca 375 (+1 skipped)**, **kuleden 168** — all green.
  New tests: `test_curves`, `test_pressure_for_threading`, `test_ranking_s6_rollup`,
  `test_s7_continuity`, `test_s6_continuity`, `test_m3_zeroes_s7_for_paid_on_hard_tasks`.

### First action next session (restart-gated)
Confirm live: next mission's `model_pick_log` should show free-provider diversity holding and no
"No model candidates available" tail. C4/S6 are now sim-tunable (the §6 prereq is done) — future
S6/S7 tuning re-runs `run_scenarios.py` + `run_swap_storm_check.py` like S12 was.
