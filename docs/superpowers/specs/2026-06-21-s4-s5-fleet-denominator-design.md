# S4/S5 reshape — fleet-capacity denominator (phantom-veto Residual 2)

**Date:** 2026-06-21
**Predecessor:** `docs/handoff/2026-06-21-phantom-veto-residuals-handoff.md` (Residual 2),
`docs/architecture/fatih-hoca-phase2d-equilibrium.md`.
**Status:** design — approved to proceed conditional on the mandatory sim gate passing.

## Problem

S4 (`s4_queue_tokens.py`) and S5 (`s5_queue_calls.py`) compute queue-conservation
pressure as:

```python
projected = queue.projected_tokens   # AGGREGATE over the WHOLE ready queue
for _, rl in matrix.cycle_*_cells(): # THIS one model's cycle window
    remaining = max(0, (rl.remaining or 0) - rl.in_flight)
    ratio = projected / remaining    # whole queue ÷ one model's window
    pressure = -min(1.0, max(0, ratio - THRESHOLD) * SLOPE)
```

The numerator is the demand of the **entire** ready queue; the denominator is a
**single** model's cycle remaining. A model with a small daily window (e.g.
gemini ~20/day) is conserve-penalised by the *whole* queue's projected demand —
even though only a fraction of those tasks would ever route to it.

**Leak (owner's words):** *small-daily free + big-daily premium + easy task +
deep queue* → the small-daily free gets floored on S4/S5 → the easy task leaks
to the big-daily premium = waste (paying money for an easy task a free model
should have served). The B-table wiring (`28a08aeb`) reduced the *magnitude* of
`projected_*` but not the **shape** (aggregate ÷ single-model).

## Root cause

Queue-overshoot conservation is fundamentally a **fleet** property: "will the
queue exhaust the budget that can actually absorb it?" The current code asks the
wrong question — "will the whole queue exhaust *this one* model?" — which floors
small-window models whenever the queue is large, regardless of how much other
capacity exists to absorb the overflow.

## Design — fleet-capacity denominator

Replace the per-model denominator with the **fleet's** total cycle remaining on
the matching axis:

```python
# pressure_for computes once, passes to S4/S5:
#   fleet_remaining[axis_name] = Σ over all cloud models of
#       max(0, rl.remaining - in_flight_for_that_model)   # cycle axes only
projected = queue.projected_calls
for axis_name, rl in matrix.cycle_request_cells():   # axes THIS model populates
    fleet_rem = fleet_remaining.get(axis_name)
    if not fleet_rem or fleet_rem <= 0:
        continue
    ratio = projected / fleet_rem
    pressure = -min(1.0, max(0, ratio - THRESHOLD) * SLOPE)
    if pressure < worst:
        worst = pressure
```

**Semantics:** a model conserves on an axis only when the **whole fleet's**
budget on that axis would be exhausted by the queue — i.e. there is no escape
hatch. When abundant capacity exists elsewhere, no floor fires, and small free
models stay serviceable for the easy/cheap work they should absorb.

### Invariants preserved

- **Worst-axis-wins** within a model — unchanged (loop over this model's cells).
- **Per-minute exclusion** — unchanged; `cycle_*_cells()` already excludes
  rpm/tpm/itpm/otpm. The fleet sum is built only from cycle cells.
- **Demand never regains veto power** (`48e4cee8` invariant) — S4/S5 stay in
  `QUEUE_BUCKET` (rank multiplier only), never the supply veto. Untouched.
- **M1 amplifier** — still applied per-model in `pressure_for` (basis = this
  model's smallest limit). The signal is fleet-relative; its amplification is
  per-model. Unchanged.
- **pp11 (daily-overshoot conserves)** — pp11's snapshot populates ONE model, so
  fleet remaining == that model's remaining → `40/20 = 2.0` → still floors. The
  anchor holds because a fleet-of-one is genuinely "no escape."

### Fleet scope = all cloud (free + paid)

The denominator sums across **all** cloud models, not just the free pool. The
leak is free-floored → leaks to paid; if the denominator excluded paid, the leak
would survive. Conserving free while paid is abundant is the over-conservation we
are removing. Capability is **not** mixed in — S4/S5 stay capability-blind
(budget signals); the hard-task→premium reservation remains M3's job
(down-weights S4/S5 to 0.7 on hard, up-weights S9 right-tool).

### What this opens, and why it is acceptable (S6 + intended behavior)

Fleet-denominator stops S4/S5 from conserving a model whose own window is small
when the *fleet's* budget is large. Two sub-cases:

- **Capability-tagged shortage** (lone vision/thinking/function_calling/cloud_only
  model, deep matching queue): **S6 owns this** with the same demand÷capable-supply
  shape (`s6_capable_supply.py:63-75`): `demand = count·iter_avg`,
  `supply = Σ rpd_remaining·iter_avg over capable models` → ratio → floor. S6
  fires even on a single capable model (supply = that one model). Confirmed: S6
  iterates `by_capability`, whose only keys are the four above
  (`queue_profile_push.py:110-132`).
- **Generic / difficulty-driven overshoot** (deep queue of hard d≥7 tasks whose
  only capable cloud model has a small daily window, NO capability tag): S6 does
  **NOT** cover this — `by_capability` carries no difficulty key (the
  `s6_capable_supply.py` docstring's "hard difficulty tier" is **stale/wrong** —
  the code never reads `by_difficulty`). Post-fix S5 goes quiet here. **This is
  the fix working, not a regression**: today the per-model shape floors that sole
  capable model on its own window, pushing hard work *away* from the only model
  that can do it — the exact over-conservation we are removing. Post-fix, hard
  work routes to its capable model (desired; M3 also down-weights S4/S5 to 0.7
  and up-weights S9 right-tool on hard), and **S1 still conserves** the model as
  it genuinely depletes. Demand never vetoes (`48e4cee8`), so worst case is a
  ranking nudge, never a stall.

S4/S5 stay the capability-blind generic-budget axis; capability shortage stays
in S6. The only behavior they lose — flooring a sole capable model by the whole
queue — is behavior we *want* gone.

### Fleet-of-one fallback (default when no fleet view)

`s4/s5` gain a parameter `fleet_remaining: dict[str, int] | None`. When `None`
or an axis is absent (direct bare-matrix unit-test calls), that axis falls back
to **this model's own `max(0, remaining − in_flight)`** — mathematically the
fleet sum of a fleet-of-one, == today's behavior. Existing `test_s4`/`test_s5`
unit tests (call the signal directly, no fleet view) stay green by construction.

### Threading — three call paths

`fleet_remaining[axis] = Σ over all cloud models of max(0, remaining −
in_flight_for_that_model)`, cycle axes only. Three paths set it:

1. **Prod / ranking** (perf path): build once per pick in the ranking layer
   (mirror `fleet_consumed`: `ranking.py:172` builds, `pressure_for` consumes),
   pass into each `pressure_for(... fleet_remaining=...)`. O(models), not the
   O(models²) of building inside the candidate loop.
2. **Pressure-only anchors** (pp11/pp13 call `snap.pressure_for` directly, never
   through ranking): `pressure_for` **builds `fleet_remaining` itself from
   `self.cloud` + `self.in_flight_calls` when the arg is None**, then forwards to
   S4/S5. One O(models) pass on a single-pick path — fine; prod skips it via #1.
3. **Direct signal unit tests**: call `s4_queue_tokens(matrix, queue=...)` with
   no fleet arg → per-model fallback (old behavior); new tests pass it explicitly.

Owner explicitly named "S1-actual remaining + in-flight" as the basis.

### In-flight attribution — note the asymmetry

The fleet sum subtracts in-flight **per model** (`c.model`, `types.py:236`) on
cycle axes (rpd/tpd) — those are per-model budgets. This is deliberately
**different** from the existing effective-matrix subtraction in `pressure_for`,
which subtracts in-flight **per provider** on rpm/tpm (`types.py:377-385`)
because free-tier per-minute limits are provider-aggregate (shared API key).
Do NOT reuse the provider filter for the fleet sum.

### Reset-window heterogeneity — accepted looseness

`cycle_*_cells()` yield the axis *field name* (`rpd`/`rpw`/`rpmonth`),
provider-independent (`types.py:121-141`). So `fleet_remaining["rpd"]` sums rpd
across providers whose daily windows reset at different clock times (gemini
rolling vs openrouter midnight). Accepted: the signal is an "does any escape
hatch exist for this queue" test, and the numerator (whole-queue projection) is
equally reset-agnostic. Precise per-reset-window accounting is out of scope —
noted, not silently introduced as if exact.

## Validation reality (no full-flow sim delta — rest on pressure-only anchors)

The full-flow sim factory builds `queue_profile` **without** `projected_tokens`/
`projected_calls` (`scenarios.py:179-184`; they default to 0), and S4/S5
early-return 0 on `projected <= 0`. So **S4/S5 are already dormant (==0) in every
full-flow scenario** (pp8, rp1–rp5, baseline…) — pre- AND post-fix. There is
**no rp2–rp5 distribution delta to report**; those scenarios do not exercise this
code path at all. (Pre-existing: the full S4/S5 mechanism has only ever been
validated pressure-only.)

Consequently validation rests entirely on:
- **pp11** (anchor) — daily-overshoot still conserves (fleet-of-one).
- **pp13** (new) — leak floor removed, FAILS when reverted.
- **`test_s4.py` / `test_s5.py`** — extended with a multi-model fleet case + the
  fleet-of-one fallback case.

Out of scope (logged as follow-up): wiring `projected_*` into the full-flow sim
factory so rp-scenarios genuinely exercise S4/S5 — that would light up the signal
across all scenarios and shift many distributions, a separate validated change.

pp1, pp10, pp12 — unaffected (S1 / per-minute / S2-S3 paths). Verify no drift.

## New sim anchor (must FAIL pre-fix, PASS post-fix — mirrors pp12)

`pp13_aggregate_vs_single_leak` (pressure-only):

- Fleet: `free/model` rpd=20 + `premium/model` rpd=1000 (both populated in the
  snapshot so the fleet sum sees both).
- `queue_profile = QueueProfile(total_ready_count=40, projected_calls=40)`.
- Easy task (d=3), small per-task estimate.
- Assert on `free/model`: **post-fix** `queue bucket > -0.3` (serviceable);
  **pre-fix** (revert to per-model denominator) the same model floors
  (`queue bucket <= -0.5`) → assertion fails when reverted. This proves the
  reshape, not a dampened magnitude.

Add to `POOL_PRESSURE_SCENARIOS` + the assertion dispatch table.

## Mandatory sim gate (every change)

- `python packages/fatih_hoca/tests/sim/run_scenarios.py` — pp1–pp13 PASS; pp11
  (anchor) green; new pp13 green. Expect **no rp/pp delta** (S4/S5 dormant in
  full-flow — see "Validation reality"); any shift = unintended, investigate.
- `python packages/fatih_hoca/tests/sim/run_swap_storm_check.py` — swap rate ≤0.5%.
- `pytest packages/nerd_herd/tests/signals/test_s4.py test_s5.py` — extend with
  a multi-model fleet case + the fleet-of-one fallback case (foreground +
  `timeout`, never `run_in_background` on Windows).

## Files touched

| file | change |
|---|---|
| `packages/nerd_herd/src/nerd_herd/signals/s4_queue_tokens.py` | add `fleet_remaining` param; divide by fleet, fleet-of-one fallback |
| `packages/nerd_herd/src/nerd_herd/signals/s5_queue_calls.py` | same shape (imports SLOPE/THRESHOLD from S4) |
| `packages/fatih_hoca/src/fatih_hoca/ranking.py` | build `fleet_remaining` per cycle axis (Σ `self.cloud` remaining − per-model in-flight) once per pick; pass into `pressure_for` (mirror `fleet_consumed`) |
| `packages/nerd_herd/src/nerd_herd/types.py` | `pressure_for`: accept `fleet_remaining`, forward to S4/S5 |
| `packages/fatih_hoca/tests/sim/scenarios.py` | add `pp13_aggregate_vs_single_leak` + assertion + dispatch |
| `packages/nerd_herd/tests/signals/test_s4.py`, `test_s5.py` | multi-model fleet + fallback cases |

## Out of scope

- Residual 1 (abundance gate) and Residual 3 (M4 desktop veto) — independent,
  separate specs.
- Routable-share numerator (the circular alternative) — rejected; fleet-capacity
  is the non-circular form of the same intent.
- Changing THRESHOLD/SLOPE/bucket weights — no constant change without a sim
  delta justifying it.

## Deploy notes

Restart-gated (signal layer loads at process start). `origin/main` is ~10
commits behind (kdv, FC-gate, phantom chain, registry-decouple, btable-wiring) —
`/restart` + verify + push the backlog before/with this work. Worktree + 3-way
merge (concurrent sessions cross `main`). NEVER `run_in_background` pytest on
Windows (orphans hold the prod SQLite lock).

## Post-restart watch-items (production red-team, 2026-06-21)

Verdict SHIP-WITH-WATCH-ITEMS — no blockers (every worst case is a rank nudge,
never a stall, because S4/S5 stay rank-only). After `/restart`, watch:

1. **Sim-vs-prod validation gap.** S4/S5 are dormant (==0) in every full-flow
   sim (the factory leaves `projected_*` unset), but `queue_profile_push.build_profile`
   DOES populate them in prod — so the reshape's emergent *ranking* effect is
   validated only by pressure-only anchors, never end-to-end. Confirm via
   `model_pick_log` that small free models stay serviceable on easy tasks under
   deep queues (the headline behavior) and that free-vs-paid distribution on easy
   work didn't regress. Follow-up (deferred to avoid mass scenario shift): wire
   `projected_*` into the sim factory for a genuine full-flow exercise.
2. **Long-context S6 gap.** The fleet sum counts an rpd-populating model even when
   it can't serve a capability-constrained queue. S6 covers vision/thinking/
   function_calling/cloud_only but has **no context-size key** — a lone
   large-context model (e.g. gemini 1M) under a long-context-only deep queue is
   conserved by neither S6 nor the fleet denominator. Bounded (S1 backstops as it
   depletes). Watch for long-context models exhausting faster; if seen, add an S6
   context-size capability key.
3. **`name==litellm_name` in-flight match.** No live mismatch today (dynamic
   discovery sets them equal; YAML cloud catalog empty). Guarded by a comment in
   `_build_fleet_cycle_remaining`. If a YAML cloud model with a friendly name is
   ever added, its in-flight stops subtracting → fleet over-counts → mild
   under-conserve (safe direction). Normalize on `litellm_name` if that path goes
   live.
