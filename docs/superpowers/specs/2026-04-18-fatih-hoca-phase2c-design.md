# Fatih Hoca Phase 2c — Pool-Urgency Selection & Grading-Derived Perf Score

**Status**: Design approved for spec review (2026-04-18)
**Branch**: `feat/fatih-hoca-phase2c`
**Worktree**: `.worktrees/fatih-hoca-phase2c`
**Predecessor**: Phase 2b merged at `2c12e73` (2026-04-18) — async audit, benchmark refresh tick, `/bench_picks`, i2p dry-run simulator.

---

## 1. Problem

The `simulate_i2p` dry-run on the current cold-start state picks **90.7% `groq-llama-70b`, 9.3% `claude-sonnet`, 0% locals** across 182 i2p steps. Live production telemetry shows the opposite — 349 rows of `model_pick_log` all on locals — because the cloud execution path is not yet wired; once wiring lands, cold-start selection will drain all work to cloud.

The user's GPU is VRAM-constrained, so local candidates are small models. They lose on raw capability against groq/claude on many medium-difficulty steps. Without an explicit boost for **wasted capacity** (idle GPU, unused free-tier quotas, unused prepaid credits), cloud wins every cold-start decision.

Two mechanical causes, verified at `ranking.py`:

1. **`perf_score` fallback flat 50 for cloud** (lines 293-326). Cloud candidates get an unearned 50 on Performance History since there is no local `measured_tps`. Grading-derived signal exists in `model_stats` but is not consulted.
2. **No pool-utilization signal** anywhere in Layer-3 ranking (lines 435-488). Layer 3 has `thinking_bonus`, `specialty`, stickiness/swap-penalty, failure-penalty. It has nothing that says "this resource is sitting idle / about to be burned."

The existing `QuotaPlanner.expensive_threshold` mechanism addresses *paid* cloud utilization only, via a `cost_score` penalty below the threshold. It does not model free-tier urgency, prepaid urgency, or local idle capacity.

## 2. Objective

Introduce a **pool-aware utilization layer** that boosts candidates whose underlying resource pool is idle or burning down, gated by capability so quality is preserved. Simultaneously replace the flat `perf_score` fallback with a grading-blended score so cloud models earn their Performance History points instead of getting them for free.

Target behavior:
- Cold-start medium steps (d=3, d=5) route meaningfully to capable locals whenever cap scores are near-peer.
- Hard steps (d=7+) still route to groq/claude when no local is capability-competitive.
- When locals are warm (loaded, producing output), stickiness already handles retention — urgency stays dormant.
- Time-bucketed cloud (free daily quotas, prepaid) gets ramped urgency as its reset approaches with unused quota remaining.
- Per-call cloud receives no urgency boost; existing `expensive_threshold` penalty remains its gate.

## 3. Pool Taxonomy

Three pools, classified per model in the registry.

| Pool | Members | Marginal cost | Urgency semantics |
|------|---------|---------------|-------------------|
| `local` | Any `ModelInfo.is_local == True` | ~0 (electricity sunk) | Grows with GPU idle duration since last inference |
| `time_bucketed` | Free-tier cloud (`is_free == True`) AND prepaid cloud (future: `prepaid_credits_remaining > 0`) | 0 until depleted/reset | Grows with (remaining % × proximity to reset) |
| `per_call` | Paid cloud without a bucket | >0 per call | Always 0 |

Pool 3 (prepaid) is collapsed into `time_bucketed` because bookkeeping is identical: remaining units plus a reset/expiry horizon.

## 4. Urgency Formulae

Each pool emits an urgency value in `[0.0, 1.0]`. The multiplier applied at Layer 3 is `1.0 + URGENCY_MAX_BONUS × urgency`, where `URGENCY_MAX_BONUS` defaults to `0.25` (max +25% composite boost).

**Local urgency**:
```
idle_s = snapshot.local.idle_seconds
local_urgency = min(1.0, idle_s / LOCAL_IDLE_SATURATION_SECS)   # default 600s
```
Rationale: GPU sitting idle for 10 minutes is maximally wasted; any longer adds no information.

**Time-bucketed urgency**:
```
remaining_frac = provider_state.remaining_quota_pct / 100.0
reset_proximity = 1.0 - min(1.0, reset_in_seconds / RESET_HORIZON_SECS)   # default 3600s
bucketed_urgency = remaining_frac × reset_proximity
```
Rationale: urgency peaks when quota is mostly unused *and* reset is close. A full quota one hour before midnight scores near 1.0; a full quota 12 hours before midnight scores ~0.08.

**Per-call urgency**: always `0.0`.

Missing telemetry → `0.0` (conservative — no boost rather than guessed boost).

## 5. Capability Gate

Urgency multiplier applies only when the candidate is capability-competitive with the top scorer:

```
top_cap = max(c.cap_score for c in candidates)
if candidate.cap_score >= top_cap × CAP_GATE_RATIO:     # default 0.85
    composite *= (1.0 + URGENCY_MAX_BONUS × urgency)
```

Prevents a 9B local with cap=55 from beating a 70B cloud with cap=85 via a +25% boost. On hard steps where only cloud is capability-competitive, gate closes for locals and urgency is irrelevant.

## 6. Grading-Derived Perf Score

Replace the flat-50 fallback at `ranking.py:293-326`:

```
grading = grading_perf_score(model.name)   # None when samples < GRADING_MIN_SAMPLES
if model.is_local and model.is_loaded and local_state.measured_tps > 0:
    tps_perf = tps_to_perf(local_state.measured_tps)
elif model.is_local and model.tokens_per_second > 0:
    tps_perf = tps_to_perf(model.tokens_per_second)
else:
    tps_perf = 50.0

if grading is not None:
    perf_score = GRADING_WEIGHT × grading + (1 - GRADING_WEIGHT) × tps_perf
else:
    perf_score = tps_perf
```

Defaults: `GRADING_WEIGHT = 0.6`, `GRADING_MIN_SAMPLES = 20`. `grading_perf_score` reads `model_stats.success_rate` as the primary signal (and optionally a quality score column if present in the schema during implementation — auditable at build time). Maps [0.0, 1.0] success to [20, 95] perf_score linearly, with a 20 floor so a single failure doesn't zero the score.

Cloud models with samples get a real `perf_score`; those without keep 50.

## 7. Layer-3 Insertion Point

Applied inside `ranking.py` after the existing stickiness / failure-penalty block (currently ending around line 488), before `ScoredModel` construction. Implementation keeps urgency orthogonal to stickiness: a loaded local with `idle_seconds=0` gets no urgency boost but still gets stickiness; a cold local with high idle seconds gets urgency but still pays the swap penalty.

Reason logs append `urgency={pool}:{value:.2f}` when non-zero so `/bench_picks` and counterfactual analysis can attribute boosts.

## 8. Data Plumbing

- **`packages/nerd_herd/.../snapshot.py`** — `LocalState` gains `idle_seconds: float`. Populated by tracking the last-observed-inference timestamp; snapshot computes `now - last_inference_ts`, clamped to 0 while a call is in-flight.
- **`packages/kuleden_donen_var/`** — audit each provider's `RateLimitState` for `remaining_quota_pct` and `reset_in_seconds`. Providers without reset headers fall back to a conservative model: assume daily reset at 00:00 UTC, compute `reset_in_seconds` from wall clock.
- **`packages/fatih_hoca/.../pools.py`** (new) — `classify_pool(ModelInfo) -> Pool` + `compute_urgency(model, snapshot) -> float`. Pure functions, testable without fixtures.
- **`packages/fatih_hoca/.../grading.py`** (new or extension) — `grading_perf_score(model_name) -> Optional[float]`. Reads `model_stats`, returns None below sample threshold.
- **`model_pick_log`** — add `pool TEXT` and `urgency REAL` columns via a light migration in `src/infra/db.py`. Ranking writes both.

## 9. Counterfactual Validation

A CLI at `packages/fatih_hoca/.../counterfactual.py`:

```
python -m fatih_hoca.counterfactual \
    --urgency-bonus 0.25 \
    --cap-gate 0.85 \
    --grading-weight 0.6 \
    --limit 7d
```

Replays `model_pick_log` rows under the given parameters, re-scoring each pick's candidate list. Outputs:
- Top-1 agreement rate with the originally-picked model.
- Top-1 agreement rate with the empirically best model per `model_stats.success_rate`.
- Per-agent distribution shift vs. recorded picks.
- Per-pool win-rate shift.

The intent is **parameter tuning** — sweep `URGENCY_MAX_BONUS ∈ {0.15, 0.20, 0.25, 0.30}` and `CAP_GATE_RATIO ∈ {0.80, 0.85, 0.90}`, pick the combination that maximizes agreement with `success_rate` while keeping the pool distribution roughly matched to available capacity.

Out of scope for this phase: running counterfactuals inside the orchestrator (stays a manual/CLI tool).

## 10. Validation Loop

Per-change feedback, in order of cost:

1. **Unit tests** — `test_pools.py`, `test_urgency.py`, `test_grading_perf.py`, `test_capability_gate.py`, plus ranking regression tests. TDD.
2. **Simulator** — `PYTHONPATH=packages/fatih_hoca/src python -m fatih_hoca.simulate_i2p --model-dir "C:\Users\sakir\ai\models"` after each ranking-affecting change. Target distribution: locals winning a meaningful share of coder/writer/analyst steps at d≤5, claude-sonnet still winning d=8 steps.
3. **Counterfactual** — post-implementation, sweep parameters against 349 `model_pick_log` rows.

## 11. Non-Goals

- Rebalancing `d=5` weights in `ranking.py:400-409`. Simulator will tell us whether urgency + grading perf_score closes the gap. Weight tuning is a follow-up within the phase only if those two land and locals still lose cold-start d=3/5 by a wide margin.
- Moving `expensive_threshold` logic. It stays as-is — orthogonal to urgency, concerned with pay-per-call cost control.
- Orchestrator-driven counterfactual scoring (CLI only for this phase).
- Prepaid-credit balance tracking as new infrastructure. `time_bucketed` pool semantics accommodate prepaid when data arrives; infrastructure work deferred.

## 12. Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| `model_stats` sample sizes are low across the board | Grading perf_score rarely triggers, cloud stays at 50 | Document as expected; success rates build as pipeline runs post-deploy |
| Free-tier providers omit reset headers | `time_bucketed` urgency silently stuck at 0 | Midnight-UTC fallback per provider; logged once on miss |
| Counterfactual tool finds no better parameter combo than defaults | Wasted tuning effort | Defaults were chosen to be directionally correct; "no change" is a valid signal |
| Urgency + stickiness compound, causing thrash | Rapid model swaps | Urgency gates on `cap_gate`, swap-penalty still applies to cold locals; simulator catches distributions |
| `LocalState.idle_seconds` races with in-flight call | False idle during warmup | Clamp to 0 when any call active; track last-completed-inference only |

## 13. Success Criteria

1. `simulate_i2p` cold-start shows locals winning ≥30% of d≤5 steps across coder/writer/analyst agents (current: 0%).
2. Hard steps (d=8) continue routing to claude-sonnet or better (current: 100%).
3. Counterfactual agreement with `model_stats.success_rate` strictly improves vs. current weights under at least one parameter combination.
4. All unit tests green; ranking regression tests unchanged behaviors preserved.
5. `model_pick_log` shows non-zero `urgency` and populated `pool` for all post-deploy picks.

## 14. Out of Scope, Adjacent Streams

- Cloud execution wiring itself — separate workstream. Phase 2c lands the selection layer so wiring, when it happens, doesn't immediately drain work to cloud.
- ChromaDB noise and phantom "a" model (see kickoff memory — unrelated).
- Uncommitted shopping test helpers on main (`tests/shopping/verify_reviews_live.py`) — leave alone.
