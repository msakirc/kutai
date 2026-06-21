# Handoff: Phantom-veto residuals — abundance-gate mutual-exclusivity + aggregate-vs-single-model cycle leak

**Date:** 2026-06-21
**Predecessor saga:** `docs/superpowers/specs/2026-06-17-phantom-veto-architecture-spec.md`, memory `project_phantom_veto_architecture_20260617`.
**Status:** two known residuals, deferred by owner. NOT bugs with a clear chop — both are in the **delicately-tuned utilization equilibrium**. Owner rule (verbatim): *"designed delicately, deep-dive semantics BEFORE any chop."* Do the deep-dive (modifiers.py + `docs/architecture/fatih-hoca-phase2d-equilibrium.md` + `docs/architecture/2026-06-04-cloud-utilization-continuity-design.md`) before touching either.

## What is already shipped (the phantom-veto chain — context)
The recurring `-1.0` fleet-stall was NOT one bug; it was a chain, each fixed + (mostly) pushed:
- `de94bbe6` stale rpd `remaining=0` never rolled over after reset elapsed → S1 phantom.
- `b854bdf7` openrouter `:free` misclassified as `paid` → positive arm amputated.
- `48e4cee8` **supply/demand gate split** — the gate now vetoes (`select=None`) ONLY when all candidates are SUPPLY-exhausted (`supply_pressure <= -1.0`); demand-floored-but-supply-healthy models are re-admitted ranked by `base_score`. (`selector.py` ~377-440, `ranking.py` stamps `sm.supply_pressure = bucket_totals['other']`, `base_score`.)
- `fd65e7f0` S4/S5 read **cycle axes only** (per-minute = pacing not conservation).
- `28a08aeb` (2026-06-20) **B-table wired into demand projection** (`queue_profile_push` + `ranking._apply_utilization_layer`) — killed chronic worst-case over-projection. Cold cache == prior behavior; live once the `btable_rollup` cron populates.

**Load-bearing invariant established by `48e4cee8`:** demand pressure (BURDEN S2/S3 + QUEUE S4/S5/S6) feeds only the **rank multiplier**; the admission veto fires only on **supply** (`other` bucket: S1/S7/S9/S10/S11/S12/S13/S14). So neither residual below can empty the fleet — both are **ranking-quality / waste-vs-conserve balance** issues, not stall risks. Keep it that way: any fix must NOT let demand re-acquire veto power.

---

## Residual 1 — abundance gate makes the two equilibrium arms mutually exclusive

**File:** `packages/nerd_herd/src/nerd_herd/combine.py:44-45` (`ABUNDANCE_GATE = -0.2`, line 13).

```python
positive_total = 0.0
if negative_total > ABUNDANCE_GATE:   # -0.2
    ... noisy-OR over POSITIVE_ARM_SIGNALS = (S9, S12) ...
```

**Symptom / mechanism.** The POSITIVE arm — "use perishable quota before it resets" (S9 free-perishability timing) noisy-OR "fleet under-utilized, balance the pool" (S12) — is computed **only when `negative_total > -0.2`**. So the moment *any* meaningful conserve/burden/queue pressure accumulates (`negative_total <= -0.2`), the positive pull is **zeroed**. The two arms become a **switch, not a balance**: under load the "burn-it-now" signal is silenced exactly when it should counter-weight conservation → systematic **collapse-to-conserve bias** (free/perishable quota hoarded, resets unused = waste; in the limit, contributed to the minimal-mode stalls before `48e4cee8`).

**Why it exists (reconstructed, owner "unsure why originally"):** intent ≈ "don't pull work onto a stressed model." It conflates ACTUAL depletion (S1 real-remaining-low → suppressing use-it is CORRECT) with PROJECTED demand (S3/S4/S5 burden+queue → suppressing use-it is WRONG: perishable quota isn't gone; queue + perishability both point the same way = burn now).

**Owner's directional idea (NOT yet designed/built):** gate the positive arm on **actual-remaining depletion (S1/supply) only**, not on projective demand — so demand + perishability *net* (free model pulled INTO serving the backlog) instead of cancelling (hoard → waste). Caveat the owner flagged: S9-free proximity is weak when reset is far (`exp(-reset/6h)`), so even an un-gated positive arm may not fully offset a far-from-reset model — genuinely subtle, this is the owner's equilibrium domain.

**Risk:** this is the most central equilibrium knob. Changing the gate condition shifts every pick under load. MUST re-run the full sim gate (below) AND eyeball the realistic-pool distributions (`rp1`–`rp5`) for waste-vs-hard balance, not just pass/fail.

---

## Residual 2 — aggregate-queue-vs-single-model comparison over-conserves small-window models

**Files:** `packages/nerd_herd/src/nerd_herd/signals/s4_queue_tokens.py` + `s5_queue_calls.py` (S5 imports `SLOPE`, `THRESHOLD` from S4; same shape).

```python
# s5_queue_calls.py
projected = queue.projected_calls          # AGGREGATE over ALL ready tasks
for _, rl in matrix.cycle_request_cells(): # this ONE model's cycle window
    remaining = max(0, (rl.remaining or 0) - rl.in_flight)
    ratio = projected / remaining          # whole queue vs one model's window
    ... pressure = -min(1.0, max(0, ratio - THRESHOLD) * SLOPE)
```

**Symptom / mechanism.** `queue.projected_tokens/calls` is the demand of the **entire** ready queue, but it's divided by **each single model's** cycle remaining. A model with a small daily window (e.g. gemini ~20/day) is conserve-penalized by the *whole* queue's projected demand — even though only a fraction of those tasks would ever route to it. Net leak (owner's words): *small-daily free + big-daily premium + easy task + deep queue* → the small-daily free gets floored, the easy task leaks to the big-daily premium = waste. The B-table wiring (`28a08aeb`) reduced the magnitude of `projected_*` but did not fix the **shape** (aggregate ÷ single-model).

**Owner's noted full fix (bigger change, deferred):** model queue conservation as **pool-level-vs-fleet pacing** — compare projected demand against the *fleet's* relevant capacity (or the share that actually routes to a pool), relying on **S1-actual remaining + in-flight** rather than aggregate-queue ÷ per-model-window. This is a structural reshape of S4/S5, not a constant tweak.

**Risk:** S4/S5 are the genuine daily-overshoot conservation guard (`pp11` proves it must still fire when the queue truly exhausts a daily budget). Any reshape must KEEP that (don't blind real overshoot) while stopping the aggregate-vs-single over-penalty. `pp11` is the regression anchor.

---

## Residual 3 — M4 amplifies graded S13/S14 desktop signals into a hard SUPPLY veto (NEW, found in 2026-06-21 review pass)

**Files:** `packages/nerd_herd/src/nerd_herd/modifiers.py:81-93` (`_M4_BY_MODE` = full 0.0 / heavy 1.5 / shared 2.0 / minimal 1.0), `signals/s13_presence.py` (`PRESENT_PENALTY=-0.6` graded, `FULLSCREEN_VETO=-10` sentinel), `signals/s14_contention.py` (RAM penalty graded to `-1.0`, ext-GPU `-10` sentinel).

**Symptom / mechanism.** S13/S14 are designed so only the **−10 sentinels** (fullscreen / external-GPU) peg the scalar to a veto; the rest is an intentionally **soft gradient** (present-user −0.6, RAM up to −1.0) meant to *down-rank* local, not exclude it. But S13/S14 live in `OTHER_BUCKET` (= supply), and M4 multiplies them by **1.5 (heavy) / 2.0 (shared)** in `combine` before the fold. So:
- **shared** + present user (no fullscreen): S13 = −0.6 × 2.0 = **−1.2** → `supply_pressure ≤ −1.0` → local **vetoed**.
- **heavy/shared** + RAM near cap: S14 = −1.0 × 1.5/2.0 = **−1.5 / −2.0** → local **vetoed**.

A graded "yield the machine" hint has become a hard supply veto without the sentinel. Because it's a *supply* signal it does **not** break the `48e4cee8` invariant, and in heavy/shared a cloud model is normally the escape — **except** the genuine fleet-empty edge: a **`local_only` task** (cloud excluded at eligibility) in heavy/shared mode under a present user or RAM pressure → every local shares the same machine-level S13/S14 value → entire local fleet vetoed at once → **`select=None` silent stall, no fallback.** Same silent-stall pathology the saga was about, just sourced from the desktop axis instead of the cloud axis.

**Directional idea (NOT designed/built):** keep M4-weighted S13/S14 **above** the veto floor — i.e. cap the graded (non-sentinel) contribution so only the −10 sentinels can reach `supply_pressure ≤ −1.0`, preserving the soft gradient M4 was meant to strengthen. OR: treat `local_only + all-local-vetoed-by-desktop` as a **defer** (wait for user-idle / RAM relief) with a clear reason, not a silent `select=None` loop. Either path must be sim-gated.

**Risk / reachability:** needs heavy/shared mode (user-selected) + a present user or real RAM pressure + a `local_only` task. Plausible in normal desktop use. Confirm reachability (is `local_only` ever issued while shared/heavy is set?) before sizing the fix; if unreachable in practice, downgrade to LOW.

## Review-pass findings (2026-06-21, independent adversarial review — non-residual)
The phantom-veto chain (Residual-context fixes above) was re-reviewed end-to-end; the supply/demand invariant holds, base_score/supply_pressure are stamped on every path (incl. `scalar==0` continue and time-gate rescue), cycle-axis exclusion is correct (no rpm/rpmonth confusion), and `_rl` rollover does not mask genuine depletion. Two LOW doc/claim drifts to clean up opportunistically (not bugs):
- **B-table "only decreases" claim is not enforced.** `estimate_for` uses learned **p90**; a step whose real p90 exceeds the curated static default would *increase* the projection. Only inflates demand pressure (rank-only, cannot veto), so no stall risk — but correct the "can only decrease" comment or add a `min(learned, static_default)` guard if the invariant is relied on.
- **`ScoredModel.supply_pressure` docstring drift** (`ranking.py:67-71`, mirrored `selector.py:391`): enumerates S1/S7/S9/S10/S11 but the value is `bucket_totals["other"]` = the FULL `OTHER_BUCKET` incl S12/S13/S14. Code correct; comment stale (and it's exactly the S13/S14 that Residual 3 rides).

## Interaction between the two
They compound: Residual 2 inflates `negative_total` (over-conserve on small-window models), which then trips Residual 1's gate (`negative_total <= -0.2`) and silences the positive arm — double-suppressing exactly the free/perishable models that should serve the backlog. Fixing Residual 2 alone reduces how often Residual 1 misfires, but doesn't remove the switch semantics. Consider sequencing: Residual 1 (gate condition) is the smaller, more central change; Residual 2 (S4/S5 reshape) is larger. Either order, sim-gate each independently. **Residual 3 is independent** (desktop/supply axis, not demand) — fixable on its own; cap-above-veto-floor is the smallest of the three.

## Mandatory sim gate (every change, both residuals)
- `python packages/fatih_hoca/tests/sim/run_scenarios.py` — all pool-pressure (`pp1`–`pp12`) PASS, **no existing scenario shifts** unless intended (report deltas, never silently retune). `pp11` (daily overshoot conserves) + `pp12` (B-table demand reduces S2/S3, fails-when-reverted) are the key anchors.
- `python packages/fatih_hoca/tests/sim/run_swap_storm_check.py` — swap rate ≤0.5%.
- Eyeball realistic-pool (`rp1`–`rp5`) waste% / hard% distributions.
- Add a NEW scenario that captures the residual's *intended* behavior change and FAILS pre-fix (mirror how `pp12` was made fail-when-reverted — a sim that's green both ways proves nothing).
- Restart-gated: signal/combine layer loads at process start → `/restart` (under **minimal** GPU mode to expose cloud-only) to live-verify.

## Carry-forward gotchas
- Demand must NEVER regain veto power (the `48e4cee8` invariant) — fixes here are rank/balance only.
- NO tuned constant change without a sim delta justifying it. `W_BURDEN/W_QUEUE/W_OTHER`, `ABUNDANCE_GATE`, `SLOPE/THRESHOLD`, M1/M3 weights are all load-bearing.
- Worktree + 3-way merge (concurrent sessions cross `main`); NEVER `run_in_background` pytest on Windows (orphans hold the prod SQLite lock); keep pytest.ini `--import-mode=importlib`, add `-p no:aiohttp` only; raw `python -c` loads MAIN not the worktree.
- Big restart-gated backlog: `origin/main` is ~10 commits behind (kdv, FC-gate, phantom chain, registry-decouple, btable-wiring). `/restart` + verify + push before/with this work.

## Key files
- `packages/nerd_herd/src/nerd_herd/combine.py` — buckets, weights, `ABUNDANCE_GATE`, positive-arm noisy-OR (Residual 1).
- `packages/nerd_herd/src/nerd_herd/signals/s4_queue_tokens.py`, `s5_queue_calls.py` — aggregate-vs-single cycle comparison (Residual 2).
- `packages/nerd_herd/src/nerd_herd/signals/s9_perishability.py`, `s12_pool_balance.py` — the positive arm Residual 1 gates.
- `packages/fatih_hoca/src/fatih_hoca/selector.py` ~377-440 — supply-only veto (the invariant to preserve).
- `packages/fatih_hoca/src/fatih_hoca/ranking.py` — `_apply_utilization_layer`, `supply_pressure`/`base_score` stamping.
- Design docs: `docs/architecture/fatih-hoca-phase2d-equilibrium.md`, `docs/architecture/2026-06-04-cloud-utilization-continuity-design.md`.
- Sims: `packages/fatih_hoca/tests/sim/run_scenarios.py`, `run_swap_storm_check.py`, `scenarios.py`.
