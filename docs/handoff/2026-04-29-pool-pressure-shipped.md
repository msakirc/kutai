# Handoff — Pool Pressure Utilization Equilibrium

**Date:** 2026-04-29
**Branch:** main
**Status:** Implementation complete (28/28 tasks). Live verification pending. Calibration deferred to soak window.

**Prior context:** brainstorm + spec + plan executed in this session. Spec at `docs/superpowers/specs/2026-04-29-pool-pressure-utilization-equilibrium-design.md`. Plan at `docs/superpowers/plans/2026-04-29-pool-pressure-utilization-equilibrium.md`. Telemetry that informed estimates at `docs/research/2026-04-28-token-distribution.md`.

---

## What shipped

32 commits, range `f352cb5..cdb3c35`. New architecture:

- **10 signals** (`packages/nerd_herd/src/nerd_herd/signals/`): S1 per-axis remaining, S2 per-call burden, S3 per-task burden, S4 queue-tokens, S5 queue-calls, S6 capable-supply, S7 burn-rate, S9 perishability, S10 failure, S11 cost. (S8 acceleration intentionally dropped.)
- **4 modifiers** (`packages/nerd_herd/src/nerd_herd/modifiers.py`): M1 capacity amplifier, M2 perishability-conditional fit-excess dampener, M3 difficulty re-weights. M4 urgency threshold lives in `general_beckman/admission.py` (unchanged).
- **Combination logic** (`packages/nerd_herd/src/nerd_herd/combine.py`): bucket worst-wins for negatives, weighted sum across buckets (W_burden=0.5, W_queue=0.7, W_other=1.0), gated abundance (S1+S9 positive arms only fire when `negative_total > -0.2`).
- **`SystemSnapshot.pressure_for`** rewritten to orchestrate signals + modifiers + combine. Returns `PressureBreakdown` with `.scalar` ∈ [-1, +1] + per-signal contribution dict for telemetry.
- **`RateLimitMatrix`** (renamed from `RateLimits`) with 16 axis cells (request × time × granularity). Today only `rpd` is populated end-to-end; other cells stay None until per-provider header parsers land (deferred follow-up).
- **`QueueProfile`** widened: `by_difficulty`, `by_capability`, `projected_tokens`, `projected_calls`. Built by `general_beckman/queue_profile_push.py` with dep-resolution + 30s-TTL completed_ids cache.
- **Token estimates** via three-level chain (`fatih_hoca/estimates.py`): B-table (`step_token_stats`, n≥5) → static A (`STEP_TOKEN_OVERRIDES`, 10 known-heavy steps) → `AGENT_REQUIREMENTS` recalibrated to 2026-04-28 telemetry p90 + new `AVG_ITERATIONS_BY_AGENT`.
- **Per-call telemetry**: new `model_call_tokens` table (90-day retention via cron), populated from `caller.py` after every LLM call. Hourly rollup `btable_rollup.py` aggregates into `step_token_stats` (14-day window) + refreshes in-memory btable_cache.
- **8 simulator scenarios** (`packages/fatih_hoca/tests/sim/scenarios.py`): scenario 8 is the merge-acceptance gate (full-mission equilibrium). All passed at merge.

---

## Known interim states (debugger watch list)

### 1. `model_call_tokens` collects NULL-keyed rows until later plumbing

**Why:** `caller.py` instrumentation was wired in Task 3, but in `caller.py` the variable `task` is a string label (e.g. `"main_work"`), not a Task object with `.id`/`.agent_type`/`.context`. So per-call rows have `task_id=None, agent_type=None, workflow_step_id=None, workflow_phase=None`.

**Effect:** B-table rollup filters `WHERE agent_type IS NOT NULL AND workflow_step_id IS NOT NULL`, so it sees zero rows. Btable stays empty. `estimate_for` always falls through to A-table or AGENT_REQUIREMENTS. System runs but doesn't learn from telemetry yet.

**Fix:** plumb the real Task object through `LLMDispatcher.request()` → `hallederiz_kadir.caller.py` via a new parameter on `call_model()`. Caller accesses `task.id`, `task.agent_type`, `task.context.get("workflow_step_id")`, `task.context.get("workflow_phase")`.

**Where to look:** `packages/hallederiz_kadir/src/hallederiz_kadir/caller.py:~520` (the `record_call_tokens` block — see the `getattr(task, "id", None) if task else None` defensive form). Trace upstream from there.

### 2. `iteration_n` always 0

Same root cause. `iteration_n=0` placeholder set in caller.py. Fix together with #1: pass the ReAct loop iteration index from `agents/base.py` through dispatcher to caller.

### 3. RateLimitMatrix only `rpd` populated

KDV adapter (`nerd_herd_adapter.py`) was generalized in Task 8 to forward all axes via `getattr(state, f"{axis}_limit", ...)`. The axis stub fields were added to `ModelLimits` in `rate_limiter.py`. But **no provider's header parser writes to those fields yet** — only `rpd_*` fields get populated by the existing parsers.

**Effect:** S1 / S2 / S3 / S4 / S5 / S7 / S9 / S11 only see the `rpd` cell on cloud models. TPM checks, monthly checks, cost-axis pressure all stay zero. System runs correctly with reduced signal until per-provider parser PRs land.

**Where to look:** `packages/kuleden_donen_var/src/kuleden_donen_var/header_parser.py`. Per spec §9 the deferred PRs are: Anthropic (input/output token daily headers), Groq (rpm/tpm/rpd minute+day), Gemini (rpm/tpm/rpd), OpenAI (rpm/tpm minute). Each is its own follow-up PR.

### 4. PER_CALL positive abundance gate (subtle)

Task 23 added a guard in `ranking.py::_apply_utilization_layer`: for `Pool.PER_CALL`, positive scalar from `pressure_for` is suppressed when `task_difficulty < 7`. This was a pragmatic fix for paid cloud winning easy tasks via S1's flat abundance arm.

**Architectural note:** the cleaner fix would be to set `abundance_max=0.0` for `per_call` profile in `signals/s1_remaining.py::PROFILE_PARAMS`, so S1 doesn't even emit positive abundance for paid pools — paid abundance lives exclusively in S9 (right-tool-perishability). The ranking-layer gate is band-aid, not root cause. Move when convenient.

### 5. Two `QueueProfile` types coexist

- `nerd_herd.types.QueueProfile` (the new widened one — used by signals)
- `fatih_hoca.requirements.QueueProfile` (the old one — used by `QuotaPlanner`)

Task 23's ranking change added a `hasattr(planner.queue_profile, "projected_tokens")` guard before mirroring fields. Long-term: collapse to one type. Short-term: works as-is.

### 6. S1 abundance fold uses max-of-positives

If multiple cells are abundant, S1 takes the max-positive (intentional, per spec §4 self-review fix). If you observe weird abundance behavior on multi-axis pools, look at `s1_remaining.py:cell_pressures` collection logic and the negs-vs-positives branch.

### 7. BurnLog uses fixed-window rate

S7's BurnLog records `total_tokens / window_secs * 60` (e.g. 5-min window, conservative on cold-start). An earlier implementation used span-based rate which was too aggressive on bursty workloads — fixed in commit `3d13f16`. If S7 ever fires too eagerly during low-traffic periods, the fixed-window may be too sensitive — but that's calibration territory.

### 8. Cron registration uses existing `INTERNAL_CADENCES` API

B-table rollup wired as marker `"btable_rollup"`, interval 3600s (hourly), in `packages/general_beckman/src/general_beckman/cron_seed.py`. Fires via `seed_internal_cadences()` on every `beckman.next_task()` invocation — first call after process start seeds the row in `scheduled_tasks`, then `cron.fire_due()` dispatches when due. Verify: query `SELECT * FROM scheduled_tasks WHERE marker='btable_rollup'` after a normal startup.

### 9. Pre-existing unrelated test failures

`packages/fatih_hoca/tests/test_counterfactual.py::test_cli_runs_on_empty_db` and `::test_cli_reports_agreement_rate` fail with `ModuleNotFoundError: nerd_herd` when invoked as subprocess. **Pre-existing.** Not introduced by this work. Don't touch.

---

## Calibration values (tune-by-eye risk)

The S9 weights, M3 difficulty matrix, and combination bucket weights are seeded by analytical reasoning + simulator scenario 8 acceptance. They have NOT been tuned against real-mission telemetry yet.

**Calibration surface:**
- S1 PROFILE_PARAMS thresholds (per_call: 0.15, time_bucketed: 0.30; depletion_max: -1.0/-0.5)
- S2/S3 BITE_THRESHOLD = 0.30, BITE_RANGE = 0.70
- S4/S5/S6/S7 THRESHOLD = 0.70, SLOPE = 2.0
- S9 LOCAL_IDLE_MAX = 0.5, COLD_LOCAL_VRAM_OK = 0.4, COLD_LOCAL_NO_VRAM = -0.5, FLUSH_THRESHOLD = 0.7, PAID_RIGHT_TOOL_DIFFICULTY_THRESHOLD = 7
- S10 step values (0/-0.2/-0.5)
- S11 THRESHOLD = 0.30
- M1 amplifier curve: `clip(2.0 - 0.5 * log10(limit), 0.5, 2.0)`
- M2 perishability triggers: 0.5 (no damp), 0.2 (partial damp)
- M3 difficulty matrix (10 signals × 3 difficulty buckets)
- combine W_burden=0.5, W_queue=0.7, W_other=1.0; ABUNDANCE_GATE=-0.2

**When to tune:** during soak window (Phase 1 of calibration per spec §14). Pull `model_pick_log.snapshot_summary` rows post-soak, look for signals saturating ±1 (clamp masks data) or never exceeding ±0.1 (dead signal). Adjust constants, re-run `packages/fatih_hoca/tests/sim/run_scenarios.py`, ship as a small tuning PR.

---

## How to debug if something fails in production

### Bot won't dispatch tasks (admission gate too strict)

1. Check `model_pick_log.snapshot_summary` for the latest pick: is `breakdown.scalar` very negative?
2. Look at `breakdown.bucket_totals` — which bucket dominates? burden, queue, or other?
3. Look at `breakdown.signals` — which specific signal is firing strongest negative?
4. If queue_neg dominates: check `queue_profile.projected_tokens / projected_calls` — is the projection wildly high? Check `estimate_for` — is it returning sane values for the queue's tasks?
5. If burden dominates: check the per-call estimate against the matrix's most-stressed cell. Is RPD remaining tiny? M1 amplifier kicks in heavily for small pools.
6. If other dominates and S10 is firing: there are recent failures on this provider — check `consecutive_failures` count on `CloudProviderState`.

### 429s climbing

1. Check S2/S3 burden values per pick — they should warn negative on calls that come close to TPM. If they don't, est_per_call_tokens is under-estimating the actual usage. Check the B-table rollup — is the per-(agent,step,phase) p90 too low?
2. Check S7 burn rate — if recent burn extrapolation > remaining, S7 should fire negative. If it doesn't, the BurnLog may not be receiving records — verify `caller.py` is calling `_kdv_post_call(...)` (existing) AND a future task plumbs `BurnLog.record(...)` from caller.py too. (Check whether that's wired.)

### Local model never gets used despite VRAM available

1. S9 cold local + VRAM-fit returns +0.4. Verify in test: snapshot has `vram_available_mb >= model.size_mb`.
2. Check `LocalModelState.idle_seconds` — if you see `idle_seconds = 0` while local is actually idle, the snapshot pump (`nerd_herd.refresh_snapshot`) isn't updating idle correctly.
3. Check ranking.py: does the loaded model's stickiness (1.10 main / 1.50 overhead) overwhelm the cold model's S9 boost?

### B-table never populates

1. Query `SELECT COUNT(*) FROM model_call_tokens` — should be growing. If zero or stuck, `caller.py::record_call_tokens(...)` block isn't firing. Check the try/except wrapping (best-effort — silent failures).
2. Even if rows exist, check `WHERE agent_type IS NOT NULL`. Per known issue #1 above, all rows currently have NULL keys — rollup filters them out. Plumb the Task object first.
3. Check `scheduled_tasks WHERE marker='btable_rollup'` exists with `next_run` set. If missing, `cron_seed.py::INTERNAL_CADENCES` change didn't propagate.

---

## Quick verification commands

```bash
# Full test suite (excludes pre-existing test_counterfactual.py failures)
timeout 300 pytest packages/nerd_herd/ packages/fatih_hoca/ packages/general_beckman/ packages/kuleden_donen_var/ -q

# Acceptance gate
timeout 300 python packages/fatih_hoca/tests/sim/run_scenarios.py

# Inspect telemetry
sqlite3 "$DB_PATH" "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM model_call_tokens"
sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM step_token_stats"
sqlite3 "$DB_PATH" "SELECT marker, next_run FROM scheduled_tasks WHERE marker='btable_rollup'"

# Inspect a recent pick's breakdown
sqlite3 "$DB_PATH" "SELECT picked_model, picked_score, snapshot_summary FROM model_pick_log ORDER BY id DESC LIMIT 5"
```

---

## Pending follow-ups (not in this PR)

1. **Plumb Task object through `LLMDispatcher.request()` → `caller.py`.** Unblocks the entire telemetry chain (B-table, calibration loop). Highest priority.
2. **Per-provider header parsers** (Groq TPM/RPM/RPD, Anthropic input/output token daily, Gemini, OpenAI) — populates more of the RateLimitMatrix, makes signals richer.
3. **Move per_call abundance suppression** from `ranking.py` band-aid to `s1_remaining.py` PROFILE_PARAMS root.
4. **Collapse two `QueueProfile` types** (nerd_herd vs fatih_hoca.requirements) into one.
5. **Calibration loop** (Phase 2 per spec §14) — weekly auto-tuner reading `model_pick_log.snapshot_summary`.
6. **Sibling demand projection** (graders + summarizers in queue projection) — deferred until measure shows projection ≥20% short.

---

## Files changed (commit-level)

```
f352cb5 db: step_token_stats + model_pick_log.outcome
b805c31 db: consolidate outcome into existing ALTER loop
9d35b04 telemetry: record_call_tokens helper + caller.py
702ad27 fatih_hoca: Estimates dataclass + STEP_TOKEN_OVERRIDES
1eb87e2 fatih_hoca: AGENT_REQUIREMENTS p90 + AVG_ITERATIONS_BY_AGENT
7a4ea4a fatih_hoca: estimate_for lookup chain
bc1f121 nerd_herd: RateLimits → RateLimitMatrix
fc84193 nerd_herd: drop alias
ade6535 kdv: adapter forwards all populated cells
51747ec nerd_herd: QueueProfile widening + PressureBreakdown
83884bd nerd_herd: S1
1d7f627 nerd_herd: S2
0a13bed nerd_herd: S3
9c26f90 nerd_herd: S4
2da4d8d nerd_herd: S5
babe970 nerd_herd: S6
0cdd2b2 nerd_herd: S7 + BurnLog
3d13f16 nerd_herd: S7 fixed-window + THRESHOLD=0.70
b936e1f nerd_herd: S9
cfda83c nerd_herd: S10
cd79e61 nerd_herd: S11
fff786e nerd_herd: M1/M2/M3
0dc20a6 nerd_herd: combine_signals
3b242fd nerd_herd: PressureBreakdown.modifiers Any
5ee2cba nerd_herd: pressure_for orchestrator
09ad904 fatih_hoca: ranking.py + delete scarcity.py
62bc710 beckman: admission gate + btable_cache
a78bc03 beckman: queue_profile_push dep-resolution + projections
c03a4d4 beckman: B-table rollup + cron
3165299 nerd_herd: remove legacy pool_pressure
cdb3c35 sim: 8 pool-pressure scenarios
```
