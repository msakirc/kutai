# Fatih Hoca Phase 2d — The Utilization Equilibrium

**Landed**: 2026-04-20. Branch: `feat/fatih-hoca-phase2d`.

This is the design-decision record for Phase 2d. It explains what the
utilization equation is, why it was reshaped during tuning, and the
structural bugs surfaced during the work.

## The Core Idea

Model selection is a **utilization balance** problem with two ends of the
same medallion:

- Don't waste Claude on a d=3 task — it's expensive, there are cheaper
  alternatives, and there are likely harder tasks later that will need it.
- Don't waste Groq quota that resets in 20 minutes — it's free, it's
  expiring, and using a local 9B when Groq is about to vanish is stupid.

Phase 2c split these into a gated urgency layer (boost when imminent
reset, penalize when hard tasks queued) and it didn't reach equilibrium
because they were treated as separate mechanisms with a capability gate
bolted on top. Phase 2d makes them **one equation**, one scalar
per (model, task) pair:

```
fit_excess = (cap_score_100 - cap_needed_for_difficulty(d)) / 100
scarcity   = pool_scarcity(model, snapshot, queue_state, task_difficulty)

if scarcity > 0:
    fit_dampener = max(0, 1 - abs(fit_excess))   # symmetric
else:
    fit_dampener = 1 - max(0, fit_excess)        # over-qual only

composite *= 1 + K * scarcity * fit_dampener
```

`K = UTILIZATION_K = 1.0`. Scarcity is bounded `[-1, +1]`.

## Scarcity Per Pool

### Local (loaded + idle)

- **Loaded + busy**: mild negative (`-0.10`). Don't pile on.
- **Loaded + idle seconds saturated**: positive up to `+0.5`. GPU time
  is sunk — use it.
- **Not loaded**: neutral (0). No idle signal.

### Time-bucketed (free-tier cloud: groq, gemini, openrouter)

Continuous across the full reset horizon — wasting quota is loss
regardless of proximity, just a weaker signal farther out:

- **Remaining < 30%**: negative, scaled by depletion. Conserve.
- **Otherwise**: `+1 × remaining_frac × exp(-reset_in / 24h)`

Reset-horizon weights:

| reset_in | weight | interpretation |
|---|---|---|
| 1h | 0.96 | strong "use it now" |
| 12h | 0.61 | moderate burn |
| 24h | 0.37 | daily-reset reality — still meaningful |
| 48h | 0.14 | weak |
| 72h | 0.05 | near-zero |

The 24h value (≈0.37) is the key realism: daily free quotas are always
"being wasted" in some sense unless fully drained before midnight.

### Per-call (paid cloud: claude, openai, etc.)

Three arms, mutually exclusive by budget state:

- **Abundance**: `remaining_frac > 15% AND task_difficulty ≥ 7` → `+1.0`
  (full). Capability wins when we have budget to spend and the task
  actually needs it.
- **Depletion**: `remaining_frac < 15%` → negative, scales to `-1.0` at
  zero remaining. Conserve the last window of budget.
- **Queue pressure**: `task_difficulty < 7 AND hard tasks queued` →
  negative, scaled by how easy this task is and how many hard tasks are
  coming. Reserve for the hard ones.

Queue pressure only activates on easy tasks, so it doesn't fight the
abundance arm.

## The Dampener's Job

The `fit_dampener` asymmetry is deliberate:

- **Over-qualified + positive scarcity**: reduce boost proportional to
  fit_excess. Groq on a d=3 task with imminent reset still gets a boost,
  but scaled down because the task doesn't actually need Groq's
  capability — just its free credit.
- **Under-qualified + positive scarcity**: reduce boost proportional to
  `abs(fit_excess)`. Burning a wrong tool is itself wasteful. An
  under-qualified local sitting idle should NOT get the full "burn me"
  signal.
- **Over-qualified + negative scarcity**: keep full conservation effect.
  If Claude is over-qualified AND we have hard queued, we want the
  maximum penalty on using Claude here — conservation math should bite
  proportional to how much we'd be wasting.
- **Under-qualified + negative scarcity**: `max(0, fit_excess) = 0`, so
  no dampener. Conservation applies at full strength. (If we're
  under-qualified AND we should conserve, both forces push away from
  this pick — compound.)

## Why This Nearly Failed (Bugs Surfaced During Tuning)

### 1. Base ranking's stickiness crushed cloud

The ranking layer applied `composite *= 1.4` (main) or `2.0` (overhead)
to the loaded local model. This was originally a **swap-storm damper** —
if two locals have close caps, prefer the loaded one to avoid a 30s+ GPU
reload.

But 1.4× composite is 40% — strong enough to make a loaded local beat
any cloud option on nearly every task, including hard tasks where the
local is grossly under-qualified. The utilization equation (K × ±1 × 1)
could only shift composite by ±25% at the time, not enough to overcome.

**Fix**: Dial stickiness to 1.10× (main) / 1.50× (overhead). Swap penalty
softened 0.75 → 0.92 and 0.30 → 0.60 correspondingly. Swap-storm behavior
verified with real registry: `run_swap_storm_check.py` shows 0-0.5% swap
rate across 200-task runs.

### 2. Stickiness on under-qualified loaded models was a capability override

Even at 1.10×, a loaded local on d=8 (cap=55, cap_needed=75) beat Claude
by compounding with cost/speed advantages. Stickiness shouldn't apply
when the loaded model is fundamentally wrong for the task.

**Fix**: `qual_factor = max(0, 1 + fit_excess × 5)` when `fit_excess < 0`.
Stickiness fades from full at `fit_excess = 0` to zero at
`fit_excess ≤ -0.2`. Loaded-but-wrong → no sticky bonus → cloud wins.

### 3. Per-call scarcity was conservation-only

Spec §3.1 said per_call is "≤ 0" — conservation or neutral. But then
Claude has no positive signal anywhere, and base ranking's speed/cost
weights favor Groq even at d=10 where only Claude qualifies. The
equation could demote Claude but never promote it.

**Fix**: Added the abundance arm. Mirror of time_bucketed's imminent-reset
boost: when we have budget AND the task needs the expensive model,
per_call gets `+1.0`. The fit_dampener ensures this doesn't fire
inappropriately on easy tasks (Claude's fit_excess=0.63 at d=3 gives
dampener=0.37, reducing the boost 63%).

### 4. Symmetric fit dampener on positive scarcity

Original dampener only reduced positive scarcity for over-qualified
candidates. But an under-qualified local with idle time was getting the
full `+0.5` boost on hard tasks — "burn me because I'm idle" doesn't
make sense when the model is the wrong tool.

**Fix**: `fit_dampener = max(0, 1 - abs(fit_excess))` for positive
scarcity. Conservation (`scarcity < 0`) keeps the old asymmetric
dampener — we still want full conservation effect when the candidate is
over-qualified.

### 5. Simulator wasn't threading QueueProfile

`select_for_simulation` ignored the current task queue when calling
`rank_candidates`, so the queue-pressure arm of per_call scarcity never
activated during sim runs. Scenarios couldn't differentiate themselves
by queue shape.

**Fix**: Runner builds a live `QueueProfile(total_tasks, hard_tasks_count,
max_difficulty)` per tick from the remaining task slice, installs it on
the global `QuotaPlanner` before each `rank_candidates` call.

### 6. Easy-task-waste metric counted free-pool burns as waste

Spec §7 defined waste as any d≤4 pick with `fit_excess > 0.4`. This
flagged "Groq on d=3 with imminent reset" as waste, but that's exactly
the utilization behavior we want. A free pool burning during a cycle
it would otherwise lose isn't waste — it's intended.

**Fix**: `easy_task_waste` counts only `pool == "per_call"` picks.
Over-qualified picks to free pools or local are not waste — only paid
money spent on easy tasks counts.

### 7. Free-quota-utilization metric under-counted when buckets reset

When a time_bucketed pool resets mid-simulation, `final_state.remaining`
reads as fresh again — erasing pre-reset picks from the accounting.

**Fix**: Metric counts picks per pool from `run.picks` (reset-proof) and
divides by `pool.limit`, capped at 1.0 per pool so one heavily-used pool
doesn't skew the average.

## Final Scenario Results

```
scenario                   hard  waste  free_q  picks
baseline                 100.0%   0.0%    9.0%  groq=90 claude=47 local=45
claude_constrained        97.9%   0.0%    9.5%  groq=95 local=46 claude=41
groq_near_reset          100.0%   0.0%    9.5%  groq=95 claude=45 local=42
diverse_pool             100.0%   0.0%   91.1%  local=104 claude=46 gemini=12 groq=11 or=9
exhaustion_sequence      100.0%   0.0%   41.2%  local=102 claude=47 groq=20 gemini=13
back_to_back_i2p         100.0%   0.0%   20.9%  groq=314 local=126 claude=106
staggered_i2p            100.0%   0.0%   14.7%  groq=147 local=70 claude=56
```

All 7 scenarios hit hard_sat ≥ 97.9% (most 100%), easy_waste = 0%, and
the diverse_pool `free_quota_utilization` target of >70% is met at
91.1%. 378 tests pass across fatih_hoca, nerd_herd, and cross-package.

## What's Out Of Scope

Phase 2d does NOT:

- Change the base weighted composite weights for capability/cost/
  availability/performance/speed. Utilization is orthogonal.
- Change how cost_score is computed. Budget-awareness arrived via the
  per_call scarcity abundance arm, not via a budget-aware cost dimension.
- Wire cloud execution into the live pipeline (`telegram_bot.py` →
  ranking still routes). The equation is ready; wiring is a separate
  stream.
- Graduate `cap_needed_for_difficulty` from hand-tuned to
  empirically-derived from `model_stats`. Data is still sparse.

## Key Files

| file | role |
|---|---|
| `packages/fatih_hoca/src/fatih_hoca/capability_curve.py` | `CAP_NEEDED_BY_DIFFICULTY` dict + `cap_needed_for_difficulty(d)` |
| `packages/fatih_hoca/src/fatih_hoca/scarcity.py` | 3-branch signed scarcity (local / time_bucketed / per_call) |
| `packages/fatih_hoca/src/fatih_hoca/pools.py` | `Pool` enum + `UTILIZATION_K` constant |
| `packages/fatih_hoca/src/fatih_hoca/ranking.py` | `_apply_utilization_layer` + dialed stickiness |
| `packages/fatih_hoca/src/fatih_hoca/counterfactual.py` | Replays model_pick_log under new equation |
| `packages/fatih_hoca/tests/sim/` | Stateful simulator harness + 7 scenarios |
| `packages/fatih_hoca/tests/sim/README.md` | Runnable invocations + extension guide |
| `packages/fatih_hoca/tests/test_scenarios.py` | pytest parametrized over all 7 scenarios |

## Running It

```bash
# Equilibrium table for all 7 scenarios
python packages/fatih_hoca/tests/sim/run_scenarios.py

# Real-registry swap-storm verification
python packages/fatih_hoca/tests/sim/run_swap_storm_check.py

# All tests
python -m pytest packages/fatih_hoca/ packages/nerd_herd/ tests/fatih_hoca/
```
