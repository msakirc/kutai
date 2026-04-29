# Pool Pressure — Utilization Equilibrium

**Date:** 2026-04-29
**Status:** Design — pending implementation plan
**Successor to:** Phase 2d (`docs/superpowers/specs/2026-04-19-fatih-hoca-phase2d-finalized-design.md`), `docs/superpowers/specs/2026-04-21-pool-pressure-shared-design.md`
**Touchpoints:** `packages/nerd_herd/`, `packages/fatih_hoca/`, `packages/general_beckman/`, `packages/kuleden_donen_var/`, `packages/hallederiz_kadir/`, `src/infra/db.py`

---

## 1. Problem

Phase 2d shipped a unified utilization signal (`pool_scarcity` / `pressure_for`) consumed by both Fatih Hoca's selection and Beckman's admission gate. Two gaps remain:

1. **Capacity magnitude is invisible.** `compute_pool_pressure` operates entirely on `remaining_frac = effective / limit`. A pool with 5/10 RPD and a pool with 500/1000 RPD both report 50% remaining and score identically. Tiny pools and fat pools compete equally on availability; the system can't tell the difference between "running out of a small pool" and "still flush on a big one".

2. **Estimated tokens never enter pool decisions.** `estimated_input_tokens` / `estimated_output_tokens` feed only cost calculation, local time-gates, and ctx-fit filtering. They never reach pool pressure or per-axis availability. A 200-token classifier and a 30k-token coder hit the same per-call pool identically. TPM headroom isn't checked anywhere — KDV's adapter explicitly drops `rpm`/`tpm`, only `rpd` reaches the snapshot.

### Why this matters

These aren't independent bugs — they're two faces of one principle: **utilization equilibrium**.

> "Don't waste claude on d=3 while quota is plentiful and unstressed" and "Don't waste 9b on d=3 while claude has 95% RPD resetting in 20 minutes" are the **same crime** at different tiers.

The system has a portfolio of capacity (local idle time, free cloud reset cliffs, paid cloud right-tool optionality). Equilibrium = no tier wasted on the wrong side. Falling either way means failure — under-utilized cloud (perishable burn) or over-burned cloud (no quota for hard work tomorrow).

Phase 2d covers part of this with per-pool scarcity branches. This design completes it: every observable signal becomes a first-class component, signals fuse with rules that respect what each one means, capacity magnitude and token cost enter as full-citizen factors.

## 2. Goal

Make `pressure_for(model, task, snapshot) → scalar ∈ [-1, +1]` an intelligent multi-signal fusion. Same scalar consumed by both Beckman (admission gate) and Hoca (utilization layer). Equilibrium emerges from the calibrated portfolio of signals, not from special-case carveouts.

Success criteria:

- A 10-RPD pool at 50% remaining produces stronger negative pressure than a 1000-RPD pool at 50% remaining.
- A 30k-token call against a 25k TPM headroom is filtered (not just penalized).
- Cold local with free VRAM admits a default-urgency task without special-casing.
- Free cloud near reset with flush quota wins admissions for easy work — burn perishable budget.
- Paid cloud with no perishability pressure does NOT admit easy work — reserve for hard tasks.
- Heavy queue ahead suppresses cloud admissions for routine work.

## 3. Architecture

Three layers, top-down:

### 3.1 Tier 1 — Eligibility filters

Run before pressure scoring. Each is a yes/no exclusion gate. Order matters — cheapest first.

1. **Capability** — model lacks vision / thinking / function_calling / required ctx_length → exclude.
2. **Health / cooldown** — model in cooldown OR `consecutive_failures ≥ N` → exclude. Threshold per pool profile (free: 5, paid: 3).
3. **Cost cap** — `est_call_cost > task.max_cost` (when set) → exclude.
4. **Token-fit (per-call)** — for any token-axis cell with limit data: `est_per_call_input + est_per_call_output > remaining` → exclude.
5. **Token-fit (per-task)** — for any token-axis cell with reset window ≥ task duration: `est_per_task_total > remaining` → exclude.

No RPD floor filter. S1 + M1 already produce strong negative pressure when remaining is critical; high-urgency tasks still get a chance through the threshold gate.

Filter exclusion reasons emit to `model_pick_log.candidates_json` for diagnostics.

### 3.2 Tier 2 — Pressure scoring

For each surviving candidate, compute 10 independent signals + 4 modifiers, fuse via category-bucketed worst-signal-wins + gated abundance. Output: scalar ∈ [-1, +1] per candidate plus a diagnostic `PressureBreakdown` struct.

### 3.3 Tier 3 — Existing scoring layers

Capability fit, cost, availability, performance, speed (already in `ranking.py`). Pressure scalar feeds the utilization layer at the top:

```python
composite *= 1 + UTILIZATION_K * pressure
```

`UTILIZATION_K = 1.0` preserved. Stickiness multipliers (1.10 main / 1.50 overhead) preserved.

## 4. Signal Definitions

Ten signals. Each is a pure function `(model, task, snapshot) → float ∈ [-1, +1]`. 0 = no opinion / data missing. All compute cheaply (no I/O, snapshot-derived).

### S1. Per-axis remaining pressure

For every populated cell in the model's and provider's `RateLimitMatrix`:

- `effective = max(0, remaining - in_flight)`
- `frac = effective / limit`
- Map to pressure via two-arm curve (per-cell pool profile selects depletion threshold + abundance shape — same logic as today's free/paid split, just per cell).

**Fold across cells:**
- If any cell is negative → take min (most negative). Worst-axis-wins for depletion: if RPM is fine but TPM is exhausted, the constraint is real.
- Else (all cells positive or zero) → take max. Best-axis-wins for abundance: the strongest "use it" signal across axes is what matters when no axis is depleted.

### S2. Per-call burden

For each token-axis cell with `remaining > 0`:

- `bite_frac = est_per_call_tokens / remaining`
- 30% bite or less → 0 (neutral)
- 100% bite → -1.0 (max negative)
- Curve: `pressure = -clip(bite_frac - 0.3, 0, 0.7) / 0.7`

**Fold:** largest bite across windows. A 50k call dents not just TPM but TPD/TPW/TPMonth too; biggest bite is the signal.

### S3. Per-task burden

Same as S2 but uses `est_per_task_tokens = est_per_call_tokens × est_iterations` against the budget projected over the reset horizon. Captures "this whole task across all its iterations will eat X% before reset."

### S4. Queue token pressure

Sum of `est_per_task_tokens` over **unblocked + pending + dep-resolved** queued tasks → vs token budget across windows.

- 70% of budget projected → 0
- 95% projected → -0.5
- 120% projected → -1.0

Per-task estimate via the lookup chain in §6 (B-table → static A → `AGENT_REQUIREMENTS`).

### S5. Queue request pressure

Same shape as S4 but on request axis. Per-task value: `est_iterations` (a researcher task with 23 iterations counts as 23 RPM ticks, not 1). Without this multiplier, S5 would systematically under-estimate request demand by 6-23×.

### S6. Capable-capacity overlap

For each capability requirement appearing in the queue (vision, thinking, function_calling, difficulty tier):

- Count queue's specific demand for that capability (count of unblocked tasks needing it).
- Sum total remaining budget across **eligible** models (models that can serve it).
- `pressure = -clip((demand / capable_supply) - 0.7, 0, 0.5) * 2`

**Per-model attribution:** if the current model is eligible for that capability AND demand exceeds supply, the model gets pressure proportional to its share of capable supply.

If model isn't eligible for any constrained capability → S6 = 0.

**Example:** queue has 50 d=9 tasks. Eligible = `claude_opus` (12 RPD) + `gpt5` (200 RPD). Combined eligible call-capacity = 212 × est_iterations ≈ 2300 calls. Demand = 50 × 11 ≈ 550. Ratio 0.24 → S6 = 0 (sufficient supply). If supply shrank or demand grew, both eligible models would carry conserve-pressure for non-d=9 tasks.

### S7. Burn rate extrapolation

From recent rolling history (KDV `_token_log` + a new request-rate counter):

- Last 5 min: actual `tokens_consumed`, `calls_consumed`
- Project to reset: `extrapolated_demand = recent_rate × seconds_until_reset / 60`
- Compare to remaining: `pressure = -clip((extrapolated / remaining) - 0.7, 0, 0.5) * 2`

Independent of S4/S5 — captures off-queue demand (cron tasks, external dispatch) and validates queue projections against historical truth. Cold-start (no history) → 0.

### S9. Perishability — universal abundance signal

S9 is the equilibrium core. Same signal across pool types — what differs is the per-pool computation:

- **Loaded local + idle**: linear perishable. `pressure = clip(idle_seconds / 60 * 0.5, 0, 0.5)`.
- **Loaded local + processing**: -0.10 (busy penalty).
- **Cold local + VRAM available + can fit**: +0.4 (cold-but-ready perishable).
- **Cold local + VRAM not available**: -0.5 (would need eviction; expensive).
- **Free cloud**: `pressure = remaining_frac × time_decay(reset_in) × 1.0`. Time decay = `exp(-reset_in / 86400)`. Strong positive only when remaining is high AND reset is imminent.
- **Paid cloud + budget flush + task_difficulty matches model tier**: +1.0 (right-tool perishability — wasted optionality if not used now).
- **Paid cloud + no perishability trigger**: 0.

### S10. Failure state

From `cloud.consecutive_failures` and recent timeout/degenerate counts:

- 0 failures → 0
- 1-2 recent → -0.2
- 3+ recent → -0.5

Cooldown is a Tier 1 filter; S10 covers near-cooldown states.

### S11. Cost burden

For paid models with cost-gate active:

- `est_call_cost = (est_in × cost_per_in) + (est_out × cost_per_out)`
- Compare to daily/monthly cost budget remaining → ratio → pressure curve

If no cost cap configured → 0.

### Signal numbering

S8 (acceleration) considered and dropped — marginal value over S7's flat extrapolation. If burn-storms become a real failure mode, revisit.

## 5. Modifiers

Modifiers reshape signals. Four of them.

### M1. Capacity amplifier

Small absolute pool sizes amplify the negative components of S1–S5; abundance untouched.

```
factor = clip(2.0 - 0.5 * log10(limit), 0.5, 2.0)
# limit=10  → 1.5
# limit=100 → 1.0
# limit=1000→ 0.5
```

Applied per signal that consumes a `limit`. S1's per-cell pressure gets per-cell amplification.

### M2. Perishability-conditional fit-excess dampener

When model is overqualified for the task, dampen the positive arm — UNLESS perishability is screaming.

```
def M2(model, task, S9_value):
    fit_excess = max(0, model.cap_score - cap_needed_by_difficulty[task.difficulty])
    if S9_value > 0.5:                    # strong perishability — burn it regardless
        return 1.0
    elif S9_value > 0.2:                  # mild — partial dampening
        return clip(1.0 - fit_excess * 0.25, 0.5, 1.0)
    else:                                  # no perishability — full dampening
        return clip(1.0 - fit_excess * 0.5, 0.0, 1.0)
```

This is the equilibrium key: M2 makes "burn perishable capacity even on easy work" the default behavior whenever the perishability is real, and "reserve for right-tier work" the default when no perishability pressure exists.

### M3. Difficulty re-weights

Difficulty changes which signals carry weight:

| Signal | Easy (d≤3) | Mid (d=4-6) | Hard (d≥7) |
|---|---|---|---|
| S1 (pool remaining) | 1.0 | 1.0 | 1.0 |
| S2 (call burden) | 0.5 | 1.0 | 1.5 |
| S3 (task burden) | 0.5 | 1.0 | 1.5 |
| S4 (queue tokens) | 1.5 | 1.0 | 0.7 |
| S5 (queue calls) | 1.5 | 1.0 | 0.7 |
| S6 (capable supply) | 1.5 | 1.0 | 0.7 |
| S7 (burn rate) | 1.0 | 1.0 | 1.0 |
| S9 (perishability) | 1.5 (free); 0.7 (paid) | 1.0 | 0.7 (free); 1.5 (paid) |
| S10 (failure) | 1.0 | 1.0 | 1.0 |
| S11 (cost) | 1.5 | 1.0 | 0.7 |

**Reading:** easy tasks weight queue pressure heavier (don't waste pool when hard queue ahead); hard tasks weight burden heavier (a big call deserves the model). S9 splits direction by pool type for d≤3 (burn perishable free quota; conserve paid for hard). Hard task on paid pool inverts — perishability-of-right-tool fires.

### M4. Urgency threshold shift (Beckman side)

Already exists in `admission.py::threshold(urgency) = 0.5 - urgency`. High-urgency task accepts pressure ≥ -0.5; idle task requires ≥ +0.5. No change. Hoca's selection doesn't apply M4 — only Beckman's admission gate does.

## 6. Token Estimate Sources

Each `pressure_for` call needs four estimates per task:

- `est_per_call_input_tokens`
- `est_per_call_output_tokens`
- `est_iterations`
- `est_per_task_total_tokens` = `(in + out) × iterations`

### Lookup chain

```python
def estimate_for(task) -> Estimates:
    key = (task.agent_type, task.workflow_step_id, task.workflow_phase)

    # Level 1 — learned table (B)
    if (e := step_token_stats.get(key)) and e.samples_n >= MIN_SAMPLES:
        return e.estimates_p90  # p90 not p50 — under-reserve = 429

    # Level 2 — static curated overrides (A)
    if e := STEP_TOKEN_OVERRIDES.get(task.workflow_step_id):
        return e

    # Level 3 — AGENT_REQUIREMENTS default
    r = AGENT_REQUIREMENTS.get(task.agent_type) or AGENT_REQUIREMENTS["assistant"]
    return Estimates(
        in_tokens=r.estimated_input_tokens,
        out_tokens=r.estimated_output_tokens,
        iterations=AVG_ITERATIONS_BY_AGENT.get(task.agent_type, 6),
    )
```

`MIN_SAMPLES = 5`. Below that, B's percentiles aren't trustworthy.

### Level 1 — Learned (B)

Table `step_token_stats`, refreshed hourly by Beckman's scheduled_jobs cron. Rolls up from `model_call_tokens` per-call log (§7). 14-day rolling window.

### Level 2 — Static (A)

Curated dict in `requirements.py` — `STEP_TOKEN_OVERRIDES`. Hand-seeded from 2026-04-28 telemetry sweep. Only known-heavy steps that significantly deviate from agent defaults get an entry:

```python
STEP_TOKEN_OVERRIDES = {
    "4.5b":   Estimates(in=10_000, out=100_000, iters=12),  # openapi_spec
    "5.4b":   Estimates(in=6_000,  out=92_000,  iters=8),   # forms_and_states
    "3.5":    Estimates(in=10_000, out=58_000,  iters=24),  # integration_requirements
    "4.15a1": Estimates(in=20_000, out=44_000,  iters=6),   # backend_core_design
    "5.11b":  Estimates(in=28_000, out=43_000,  iters=8),   # design_handoff_document
    # ~10-15 entries total
}
```

### Level 3 — Agent defaults

Existing `AGENT_REQUIREMENTS` recalibrated to **p90** from telemetry. Today's values are 3-8× short on p90 across i2p agents. Ship the recalibration in same PR.

New constant `AVG_ITERATIONS_BY_AGENT` from telemetry: `analyst=8, architect=12, researcher=24, writer=6, reviewer=12, ...`

### Cold-start

For agents with zero telemetry (mechanical, grader, shopping_pipeline_v2, coder, implementer): use Level 3 defaults from AGENT_REQUIREMENTS calibrated by analogy. Conservative — over-estimate slightly to avoid 429.

### Thinking tokens caveat

Telemetry source (char-counted raw_response) under-reports by 1.5-3× on cloud thinking models. Mitigation: `out_p90 *= 2.0` for thinking-enabled models when reading from B-table. Switch to `LiteLLM` `usage.reasoning_tokens` field when non-streaming once available; for streaming, fall back to char estimate + scale factor.

## 7. Per-Call Token Logging

New table `model_call_tokens` feeds the B-table. 90-day retention.

```sql
CREATE TABLE model_call_tokens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP NOT NULL DEFAULT (datetime('now')),
    task_id INTEGER,
    agent_type TEXT,
    workflow_step_id TEXT,
    workflow_phase TEXT,
    call_category TEXT,                  -- "main_work" | "overhead"
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    is_streaming INTEGER,                -- 0/1
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    reasoning_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER NOT NULL,
    duration_ms INTEGER,
    iteration_n INTEGER,
    success INTEGER NOT NULL
);

CREATE INDEX idx_mct_task ON model_call_tokens(task_id);
CREATE INDEX idx_mct_step ON model_call_tokens(agent_type, workflow_step_id);
CREATE INDEX idx_mct_recent ON model_call_tokens(timestamp);
```

Single INSERT in `caller.py` after every call (after `_record_metrics`). New helper `record_call_tokens()` in `src/infra/db.py`.

`iteration_n` source: `base.py` ReAct loop already tracks the index — pass through `LLMDispatcher.request()` → `caller.py` via new parameter.

Streaming caveat: when `is_streaming = 1`, `prompt_tokens` and `completion_tokens` are 0 (LiteLLM doesn't aggregate usage in streamed chunks). `total_tokens` is the post-stream estimate. B-table rollup weights streaming rows lower OR excludes them when non-streaming samples ≥ 5 for the same key.

Local llama.cpp streaming with `--metrics` enabled returns accurate counts in the final usage chunk. Cloud streaming is the fuzzy case.

Retention: daily cron at 03:00 deletes rows older than 90 days. ~90k–300k rows steady-state, ~50 bytes/row → ~15 MB max.

## 8. Queue Profile Extension

S4/S5/S6 require unblocked + pending + dep-resolved counts plus token/call projections.

`queue_profile_push.py` extended:

```python
async def build_and_push(db_path):
    rows = await fetch_ready(db_path)
    completed_ids = await get_completed_ids(db_path)  # cached, 30s TTL

    unblocked = []
    for r in rows:
        deps = json.loads(r["depends_on"] or "[]")
        if all(d in completed_ids for d in deps):
            unblocked.append(r)

    profile = QueueProfile(
        total_ready_count=len(unblocked),
        hard_tasks_count=sum(1 for r in unblocked if r["difficulty"] >= 7),
        by_difficulty={d: sum(1 for r in unblocked if r["difficulty"] == d)
                       for d in range(1, 11)},
        by_capability={
            "vision": sum(1 for r in unblocked if needs_vision(r)),
            "thinking": sum(1 for r in unblocked if needs_thinking(r)),
            "function_calling": sum(1 for r in unblocked if needs_tools(r)),
        },
        projected_tokens=sum(estimate_for(r).total_tokens for r in unblocked),
        projected_calls=sum(estimate_for(r).iterations for r in unblocked),
    )
    nerd_herd.push_queue_profile(profile)
```

`completed_ids` cached in process for 30s, invalidated by `on_task_finished` hook.

Dep-resolution done in Python (cleaner than recursive SQL CTE). Fetch_completed_ids restricted to `completed_at > datetime('now', '-7 days')` — older completed tasks can't be deps of pending.

**Latency target: <5 ms per push.** Profile build runs every 2-3 seconds. Cost breakdown:

- `fetch_ready`: indexed scan, <1ms for n<200
- `fetch_completed_ids`: cached, ~free
- Dep walk: O(n × avg_deps) ≈ <1ms
- `estimate_for(r)`: dict lookup chain (~3 lookups per task), 100 tasks ≈ 30µs
- Total: 1-3 ms

`QueueProfile` widened:

```python
@dataclass
class QueueProfile:
    total_ready_count: int = 0
    hard_tasks_count: int = 0
    by_difficulty: dict[int, int] = field(default_factory=dict)
    by_capability: dict[str, int] = field(default_factory=dict)
    projected_tokens: int = 0
    projected_calls: int = 0
```

## 9. RateLimit Matrix

Today's `RateLimits` carries only rpm/tpm/rpd. Adapter drops rpm/tpm. Signal taxonomy needs more cells.

### Rename + expand

`RateLimits` → `RateLimitMatrix` (singular `RateLimit` reads identical at glance to plural container — easy to mis-type).

```python
@dataclass
class RateLimit:
    limit: int | None = None
    remaining: int | None = None
    reset_at: int | None = None
    in_flight: int = 0


@dataclass
class RateLimitMatrix:
    # Request-axis cells
    rpm: RateLimit = field(default_factory=RateLimit)
    rph: RateLimit = field(default_factory=RateLimit)
    rpd: RateLimit = field(default_factory=RateLimit)
    rpw: RateLimit = field(default_factory=RateLimit)
    rpmonth: RateLimit = field(default_factory=RateLimit)

    # Token-axis cells (total)
    tpm: RateLimit = field(default_factory=RateLimit)
    tph: RateLimit = field(default_factory=RateLimit)
    tpd: RateLimit = field(default_factory=RateLimit)
    tpw: RateLimit = field(default_factory=RateLimit)
    tpmonth: RateLimit = field(default_factory=RateLimit)

    # Token-axis cells (split — providers metering input vs output separately)
    itpm: RateLimit = field(default_factory=RateLimit)
    itpd: RateLimit = field(default_factory=RateLimit)
    otpm: RateLimit = field(default_factory=RateLimit)
    otpd: RateLimit = field(default_factory=RateLimit)

    # Cost-axis cells
    cpd: RateLimit = field(default_factory=RateLimit)
    cpmonth: RateLimit = field(default_factory=RateLimit)

    def populated_cells(self) -> Iterator[tuple[str, RateLimit]]: ...
    def token_cells(self) -> Iterator[tuple[str, RateLimit]]: ...
    def request_cells(self) -> Iterator[tuple[str, RateLimit]]: ...
```

Cells with no provider data stay empty; signal code iterates only populated cells.

### Migration

Today only `rpd` is populated. After this change, S1 still computes per-cell pressure but only the rpd cell has data → fold = rpd-only result → same behavior as today. As parsers + static configs land per-provider, more cells populate → richer signal. Zero rollout risk.

### Per-provider parser extensions (separate PRs after main)

- **Anthropic**: `anthropic-ratelimit-{requests,tokens,input-tokens,output-tokens}-{remaining,limit,reset}` (daily)
- **Groq**: `x-ratelimit-{remaining,limit,reset}-{requests,tokens}` (per-minute), `-requests-day` (daily)
- **Gemini**: `quota-{requests,tokens}-{minute,day}`
- **OpenAI**: `x-ratelimit-{remaining,limit,reset}-{requests,tokens}` (per-minute)

Static config seeds (in `kuleden_donen_var/config.py`) for tier-static limits not in headers (e.g. Groq free 14400 RPD per model).

## 10. Combination Logic

How the ten signals + four modifiers reduce to one scalar.

### Step 1 — Compute all signals

Each signal is a pure function. Run all ten. Each returns ∈ [-1, +1] or 0.

### Step 2 — Apply modifiers per signal

```python
for i, S_i in signals:
    weighted_S_i = S_i * M3.weight(i, task.difficulty)
    if weighted_S_i < 0:
        weighted_S_i *= M1.amplifier(relevant_limit)
    if weighted_S_i > 0 and i is the S9 abundance arm:
        weighted_S_i *= M2(model, task, S9_value)
```

M4 is not applied here — it lives in Beckman's threshold function.

### Step 3 — Bucket signals

- **Burden** = {S2, S3}
- **Queue** = {S4, S5, S6}
- **Other** = {S1, S7, S9, S10, S11}

### Step 4 — Within each bucket: worst-signal-wins for negatives

Each signal is a single signed scalar. Within a bucket, take the most-negative value across signals:

```
burden_neg = min(s for s in [S2, S3] if s < 0, default=0)
queue_neg  = min(s for s in [S4, S5, S6] if s < 0, default=0)
other_neg  = min(s for s in [S1, S7, S9, S10, S11] if s < 0, default=0)
```

Reason: any one signal screaming -1 inside burden category should be heard, not averaged with a -0.5 next to it.

### Step 5 — Across buckets: weighted sum

```
negative_total = (
    W_burden * burden_neg +
    W_queue  * queue_neg +
    W_other  * other_neg
)
```

Defaults: `W_burden = 0.5, W_queue = 0.7, W_other = 1.0`. Other carries pool state (S1) and failure (S10) — most authoritative. Queue is forecast — slightly discounted. Burden is per-task — discounted because high call_burden often means it's a hard task that should get the model anyway.

### Step 6 — Positive arm: gated

```
positive_total = 0
if negative_total > -0.2:
    positive_total = max(0, max(s for s in [S1, S9] if s > 0, default=0))
```

Only S1 and S9 carry positive arms (the other signals are conserve/depletion-only by design). Abundance fires only when no significant negative is bothering anyone. `max` not sum because one positive arm is enough — don't double-count "use it or lose it" with "remaining is fat".

### Step 7 — Final scalar

```
return clip(negative_total + positive_total, -1, +1)
```

### Diagnostic struct

```python
@dataclass
class PressureBreakdown:
    scalar: float
    signals: dict[str, float]      # {"S1": -0.3, "S2": 0.0, ...}
    modifiers: dict[str, float]
    bucket_totals: dict[str, float]
    positive_total: float
    negative_total: float
```

`pressure_for_breakdown` returns the full struct; `pressure_for` returns the scalar. Same call internally.

`PressureBreakdown` serialized into `model_pick_log.snapshot_summary` JSON column for offline calibration.

## 11. Beckman vs Hoca Consumption

Single source — both call `pressure_for`, get the same scalar. They use it differently.

### Lean Hoca contract

`fatih_hoca.select(reqs, ...)` returns model + score (existing). Doesn't propagate pressure or breakdown through the `Pick` struct. Beckman calls `pressure_for` separately after the pick. Cheap recompute (sub-ms — no I/O).

### Beckman admission

```python
estimates = estimate_for(task)
pick = await fatih_hoca.select(reqs)
pressure = snap.pressure_for(
    pick.model, task=task,
    est_per_call_tokens=estimates.in_tokens + estimates.out_tokens,
    est_per_task_tokens=estimates.total_tokens,
    est_iterations=estimates.iterations,
)
if pressure >= threshold(urgency):
    admit(task)
```

`threshold(urgency) = 0.5 - urgency` (existing).

### Hoca utilization layer

```python
estimates = estimate_for(task)  # passed once into select
pressure = snap.pressure_for(model, task=task, **estimates)
composite *= 1 + UTILIZATION_K * pressure
```

The `(1 - max(0, fit_excess))` dampener is removed from this multiplier — M2 absorbs it. No double-counting.

### `pressure < threshold` is a feature

Not a bug. Means "do not admit now, wait." Next admission tick: pressure re-computed against fresh snapshot. Local task finishes → S1 idle scarcity rises. Cloud quota resets → S9 fires. Task age grows → urgency goes up → threshold relaxes. Eventually crosses. If nothing changes for many ticks, that's correct — system has no capacity right now.

### Cold local + VRAM invariant

A ready, unloaded local model with free VRAM must produce positive enough pressure to admit a default-urgency task (urgency=0.5 → threshold=0). S9's `cold local + VRAM available` branch returns +0.4. Default-urgency on cold local → admits cleanly.

### Cloud asymmetry — eliminated

Earlier draft treated cloud abundance as gated by M2 (over-qualification dampener). User's equilibrium framing reversed this: cloud admits when its perishability calls for it, even on easy work; local admits when its perishability calls for it. Neither is privileged. M2 is now perishability-conditional (§5), not blanket. Equilibrium falls out naturally — no special-case carveouts.

## 12. Migration / Rollout

No backward compat shims. Single PR series. New tables, type renames, signature changes ship together.

### Order

1. Schema (`model_call_tokens`, `step_token_stats`)
2. Token logging instrumentation in `caller.py`
3. Static `STEP_TOKEN_OVERRIDES` + `AVG_ITERATIONS_BY_AGENT` + recalibrated `AGENT_REQUIREMENTS`
4. `estimate_for` lookup chain
5. `RateLimit` + `RateLimitMatrix` rename + axis cells; adapter copies populated cells (only rpd today — same behavior)
6. `pressure_for` rewrite (10 signals + 4 modifiers + diagnostic struct); all callers updated in same PR
7. `queue_profile_push` extension (dep-resolution + by_difficulty + by_capability + projections)
8. Beckman admission update (new signature, breakdown logging)
9. B-table rollup cron in beckman scheduled_jobs (hourly)
10. Per-provider header parser extensions — each its own PR after main ships

### Estimated PR size

~2500 LoC main PR. Per-provider parser PRs ~150 LoC each.

### Rollback

Pure code revert. New tables stay (no harm). Single-commit revert of merge.

### Risk

- S9 calibration is the equilibrium core; first weights are guesses; simulator must validate before merge.
- M3 difficulty matrix — eyeball values; magnitudes likely need post-hoc tuning from telemetry.
- Cold-start with B-table empty must fall through to A/Level 3 gracefully.

### Safe additions

- Unpopulated `RateLimitMatrix` cells stay 0 in S1.
- New tables don't affect existing reads.
- Diagnostic struct is pure addition.

## 13. Testing

Three layers: unit, simulator, real-mission soak.

### Unit tests

Each signal function pure: `(model, task, snapshot) → float`. One test file per signal. Standard cases covered:

- Empty inputs → 0
- Single populated cell → correct curve value
- Multi-cell fold (worst-axis-wins)
- Boundary cases (0%, 30%, 50%, 70%, 100% bite)
- Cold-start (no telemetry, no history)

Modifier tests cover M1 amplifier curve, M2 perishability conditional thresholds, M3 difficulty matrix, M4 (already covered).

`pressure_for_breakdown` integration tests: handcrafted snapshots, assert final scalar AND per-bucket totals.

### Simulator scenarios

Extend Phase 2d's `run_scenarios.py` with eight new scenarios:

1. **Fat-vs-tiny pool same %util:** 10-RPD vs 1000-RPD, both 50% — tiny should empty faster.
2. **Token-aware filter:** 30k call vs 25k TPM — filtered out.
3. **Cold local + VRAM:** S9 +0.4 → admits → loads → runs.
4. **Free cloud reset imminent:** S9 ≈ +0.95 → wins admissions for easy work.
5. **Paid cloud flush + no hard queue:** M2 fully dampens claude → local wins.
6. **Capability shortage:** S6 fires -1.0 on under-supplied capable models.
7. **Difficulty lookahead:** d=3 candidate with d=9 queue ahead → cloud reserved, local wins.
8. **Equilibrium full mission:** 30 mixed tasks. Assert: cloud RPD never exhausts before reset; local never sits idle while cloud has flush quota; no pool > 80% utilized; total wall-time within 10% of theoretical optimum.

Scenario 8 is the **acceptance gate**.

### Real-mission soak

Post-merge, run one curated i2p mission (~30 tasks, mixed difficulty). Monitor:

- `model_pick_log.snapshot_summary.pressure_breakdown` per pick
- Signal contribution distribution per pool type
- 429 count vs baseline
- Cloud quota utilization at end-of-day vs baseline
- Local GPU idle ratio
- Per-tier work distribution

Telemetry sweep at `docs/research/2026-04-XX-pressure-soak.md`.

### Regression suite

- `run_swap_storm_check.py` must pass unchanged
- `tests/fatih_hoca/test_pick_telemetry.py` — pick_log writes still happen
- `tests/general_beckman/test_next_task_admission.py` — admission still respects threshold

### Coverage targets

- Signal modules: 95%+
- Combination logic: 90%+
- Estimate lookup chain: 100%
- Simulator scenarios: all 8 must pass

## 14. Calibration

Two big hand-tuned surfaces: S9 per-pool branches and M3 difficulty matrix.

### Hand-tuned (must be calibrated)

- S9 maximums per branch (loaded local idle, cold local + VRAM, free cloud, paid cloud right-tool)
- M3 difficulty weight matrix (~10 distinct values)
- Combination weights: `W_burden, W_queue, W_other`
- Gate thresholds: abundance gate (-0.2), M2 perishability triggers (>0.2, >0.5), S2/S4 burden thresholds (30%, 70%)
- Pool-pressure depletion thresholds (per_call: 0.15, time_bucketed: 0.30) — preserved from Phase 2d

### Auto-calibrated (data-driven)

- Estimates (B-table) — hourly rollup from `model_call_tokens` per (agent_type, step_id, phase)
- AGENT_REQUIREMENTS post-MVP — refinement from B-table data
- AVG_ITERATIONS_BY_AGENT — same

### Phases

**Phase 0 — Pre-merge.** Hand-tuned values seeded from current Phase 2d constants where they map; new ones from analytical reasoning + simulator validation. Acceptance: simulator scenario 8 passes.

**Phase 1 — Soak window (1-2 weeks).** Real missions run. Telemetry analysis weekly:

- Pull `model_pick_log.snapshot_summary` rows
- Identify signals saturating at ±1 too often (clamp masks signal) or never breaking 0.1 (dead signal)
- Adjust hand-tuned values, re-run simulator, ship as PR

**Phase 2 — Auto-calibration loop.** Weekly script fits S9 weights to minimize joint cost (idle + 429 + mis-routing). Proposes weight changes as PR with simulator scenarios run. Merge after human review.

### Target function

Per-pool per-day cost components:

- **Idle cost:** unused capacity × tier value
- **429 cost:** per-error fixed penalty (lost task cycle, retry overhead)
- **Mis-routing cost:** task processed by wrong-tier model (d=9 by 9b → likely fails; d=3 by claude_opus → wasted token cost)

Calibration minimizes the sum across all pools, all days. This is the closest measurable expression of "equilibrium."

### Telemetry hooks

`PressureBreakdown` logged on every pick to `model_pick_log.snapshot_summary`. Sufficient to reconstruct any pick's signal-by-signal contribution offline.

New column `model_pick_log.outcome` with `success | 429 | timeout | degenerate | wrong_tier_grade_low`. Lets calibrator correlate signal patterns with bad outcomes.

### Locked vs tunable

**Locked (architectural):** Tier 1 hard filters, combination logic (worst-wins per bucket, weighted sum across), single-scalar admission contract, estimates lookup chain order.

**Tunable (numeric):** All hand-tuned values above.

If calibration discovers a structural problem (e.g. S2 and S3 should never both fire, or S6 needs a per-capability damper), that's a design change — re-enter brainstorm.

## 15. Open Items

1. **Static `STEP_TOKEN_OVERRIDES` content** — full list to be curated from telemetry sweep at PR time, not in this design doc.
2. **Per-provider parser PR contents** — each provider's exact header set documented at PR time.
3. **Auto-calibration script** — Phase 2 work, separate spec.
4. **Coverage gap for non-ReAct agents** — char-count log lines only emit from `BaseAgent.run`. Mechanical/grader/shopping/coder/implementer have 0% coverage. With per-call token logging instrumented in `caller.py` (which all paths route through), this gap closes automatically — no separate work.
5. **Sibling demand projection (graders + summarizers in queue projection)** — deferred. Measure with post-hoc check `actual_calls / projected_calls`. If ratio drifts > 1.2 consistently, fold in.

## 16. References

- Phase 2d finalized: `docs/superpowers/specs/2026-04-19-fatih-hoca-phase2d-finalized-design.md`
- Pool pressure shared (predecessor): `docs/superpowers/specs/2026-04-21-pool-pressure-shared-design.md`
- Token distribution telemetry: `docs/research/2026-04-28-token-distribution.md`
- Cloud subsystem hardening: `docs/superpowers/specs/2026-04-27-cloud-subsystem-hardening-design.md`
- General Beckman Phase 2b: `docs/superpowers/specs/2026-04-18-phase2b-general-beckman-design.md`
- Fatih Hoca: `docs/superpowers/specs/2026-04-14-fatih-hoca-design.md`
