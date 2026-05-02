# Session Handoff — 2026-05-02

## Open Question (MUST DISCUSS BEFORE CODING)

**Reliability as a pool-pressure signal, not a ranking multiplier.**

Current state (commit `d235af1`): per-model `recent_success_rate` is plumbed
from `kuleden_donen_var.KuledenDonenVar` rolling outcome window through
`CloudModelState.recent_success_rate` into `fatih_hoca/ranking.py`'s
`_apply_utilization_layer`. Applied as a score multiplier
(`sm.score *= max(0.1, success_rate)`) AFTER the pressure equation has
already computed its scalar.

User's call (2026-05-02 19:30 UTC): reliability should be a real
**signal**, not a post-hoc multiplier. Should slot into the S1-S11
framework alongside other pressure signals.

**Concrete shape to discuss:**
- New `S12_reliability(success_rate, samples_n)` signal in
  `packages/nerd_herd/src/nerd_herd/signals/`. Returns 0.0 when fewer
  than min-samples (no data → neutral, don't penalise unknowns).
  Returns negative scaling with low success rate.
- Add to `combine.py`'s OTHER_BUCKET so it gets bucket-weighted with
  S1/S7/S9/S10/S11.
- Wire through `M3_difficulty_weights` if reliability matters more for
  hard tasks.
- Remove the ranking-layer multiplier from `ranking.py` (single source
  of truth = the signal).
- `pressure_for()` in `nerd_herd/types.py:193` accepts the success_rate
  it already reads from `prov_state.models[mid].recent_success_rate`.

**Coupled problem to consider in same discussion:**
Dead-mark TTL (`bc005a0`) and reliability rolling window (`d235af1`)
both default to 1h. When `mark_dead` expires at 1h, the failure that
caused it has ALSO aged out of the reliability window. Model
resurrects with empty outcome history → reliability defaults to 1.0
→ selector ranks it top → first call fails → ❌ → cycle.

Options:
1. Extend reliability window TTL to 24h (decouple from mark_dead).
2. Seed reliability with synthetic failure when `mark_dead` fires.
3. Post-revive penalty flag (force 0.1 reliability for first N calls).
4. Treat as pure signal: at revival, signal returns 0 (neutral, no
   data), but first failure flips it sharply negative — same cycle
   issue still possible on first call after revival.

User's preference and the chosen direction need to be set BEFORE any
code change.

---

## Today's Commits (chronological)

| SHA | Subject |
|-----|---------|
| `ff5f283` | selector: floor needs_function_calling from agent profile |
| `627339d` | db: retry-on-locked + singleton.tx visibility |
| `fa67b96` | registry: drop discovery-diff mark_dead, trust runtime 404 |
| `0781c55` | retry,orch: defer 'no_model' transient pool exhaustion (LATER REVERTED via 6256ad0) |
| `4b73278` | selector: accept needs_json_mode kwarg |
| `8a5ef9b` | dlq: recovery script for 2026-05-02 batch reset (executed: 384/384) |
| `dde55b7` | s9: LOCAL_BUSY_PENALTY sentinel survives M3 weighting |
| `b20fa3a` | retry,caller: quota errors classify as rate_limited, not auth |
| `cf9258e` | beckman: pass per-task token estimates to selector at admission |
| `5e666e6` | beckman: silence ❌ ping when task is deferred for retry |
| `157b270` | Revert "fix(beckman): silence ❌ ping when task is deferred for retry" |
| `ba19057` | selector,kdv: plumb circuit_breaker_open into selector eligibility |
| `2cd62bb` | hallederiz: pop GOOGLE_API_KEY at module init to defeat vertex auto-route |
| `c5ed59a` | beckman: persist no_model attempt cap on task row (LATER SUPERSEDED) |
| `72fbb2a` | embeddings: offload sync sentence-transformers calls to threadpool |
| `bc005a0` | registry: TTL on mark_dead so transient 404s self-heal |
| `d235af1` | reliability: per-model rolling success rate as a pressure signal (CURRENTLY MULTIPLIER, NEEDS SIGNAL REFACTOR — see Open Question) |
| `94591e3` | in_flight,pressure: admission est_tokens reserved against effective remaining |
| `08a6335` | selector,retry: plumb daily_exhausted + extend no_model ladder (LADDER LATER REVERTED) |
| `8068796` | kdv: decrement rpd_remaining locally per call |
| `e778649` | kdv: canary gate + calendar-based rpd reset |
| `e9d5f57` | retry: rewrite vertex_ai_beta exception name for gemini calls |
| `b407853` | beckman: no_model defers don't burn worker_attempts (LATER REVERTED) |
| `311014d` | retry,apply: no_model is pure deferral — no counter, no DLQ (LATER REVERTED) |
| `6387815` | dispatch: DispatchDeferred for mid-task pool saturation (LATER REVERTED) |
| `6256ad0` | retry,dispatch: unify availability ladder + drop no_model special |

## Final Architecture After Today

### Backoff / Retry / Attempts (from `6256ad0`)
- **Single shared availability ladder**: `_BACKOFF_SECONDS = [0, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600]` (10 entries, cap 1h).
- **Default attempt cap**: 10 (was 3 in apply.py and 6 in sweep.py — now both 10).
- **No category-specific caps** beyond the existing quality-bonus heuristic.
- `no_model` falls through to availability path. Same ladder, same cap, same DLQ-at-cap behavior.
- `DispatchDeferred` exception removed. Pool-empty mid-task raises
  `ModelCallFailed(error_category="availability")` and routes through
  `on_task_finished` like any other availability error.

### User's design constraints (recorded for future ref)
- "Counter is for valid worker AND grader attempts." — `worker_attempts`
  semantics. (User reverted my multi-counter splits.)
- "DLQ is for truly not recoverable tasks." — quality/schema bugs after
  exhaustion, not environmental backpressure.
- "Not being dispatched is neither" — Beckman skip-on-None at admission
  is silent (no attempt burn, no ❌). User ALSO accepts that mid-task
  exhaustion does fire ❌ + attempts++ (per `6256ad0`).
- "Backoff says 'you can try after this' not 'you must succeed'."
- "Single picker mechanism." — `Beckman.next_task`. Wake signals
  (`Beckman.on_model_swap`, `KDV` `capacity_restored` events) accelerate
  `next_retry_at`.

### Cloud capacity tracking (kuleden_donen_var)
- **Local rpd decrement** (`8068796`): per-call decrement of
  `rpd_remaining` when `rpd_limit` is known. Gemini doesn't return rpd
  headers, so local count is the only preemptive signal.
- **Calendar-based reset** (`e778649`): `rpd_reset_at` aligned to next
  UTC midnight, not rolling 24h.
- **Canary gate** (`e778649`): `_canary_verified[provider]` flag.
  Until verified, only ONE concurrent call admitted per provider.
  Success → verified. Failure → still unverified, next call becomes
  new canary. Cost: 1 wasted call per uncertainty event.
- **Circuit breaker plumbing** (`ba19057`): `CloudProviderState.
  circuit_breaker_open` plumbed from KDV to selector eligibility.
  Drops entire provider's models when breaker open.
- **daily_exhausted plumbing** (`08a6335`): `CloudModelState.daily_
  exhausted` from KDV to selector eligibility. Drops daily-out models
  upfront.
- **Reliability tracking** (`d235af1`): rolling 30-call / 1h success
  rate per model, fed via `CloudModelState.recent_success_rate`.
  Currently multiplied into score in ranking.py (NEEDS REFACTOR — see
  Open Question).

### Other notable today fixes
- `2cd62bb` — `GOOGLE_API_KEY` popped at module init so litellm doesn't
  auto-route gemini/* to vertex_ai_beta backend. Plus `e9d5f57`
  rewrites the exception class name in surfaced messages.
- `94591e3` — `InFlightCall.est_tokens` carries Beckman's projected
  consumption; pressure equation subtracts in-flight reservations from
  effective rpm/tpm/rpd for the next admission's selector pass.
- `72fbb2a` — sentence-transformers encode offloaded to threadpool.
  Was blocking the asyncio loop, causing `Yaşar Usta` heartbeat
  watchdog to kill the orchestrator every ~15min.
- `8a5ef9b` — DLQ recovery script (`scripts/recover_dlq_2026-05-02.py`).
  Executed once: 384/384 tasks reset. Backups left at
  `.dead_models.json.bak.20260502*`.

## Current Live Issue (Reproducer)

After `6256ad0` restart at 22:30 UTC, Telegram still floods with
"❌ All models failed for X: No model candidates available" messages.

Sample log from 19:31-19:32 UTC shows Beckman admitting tasks every
~3 seconds onto openrouter free-tier ids
(`openrouter/tencent/hy3-preview:free`, `owl-alpha`, `gpt-oss-20b:free`,
etc.) with `(selector cleared pool-pressure gate)`. ~30-40s later,
those tasks fire `ModelCallFailed task #X: No model candidates
available (category=availability)` — meaning dispatcher's retry
recursion exhausted the candidate pool of openrouter ids, all 404'd.

**Root cause (per the Open Question above):** dead-mark TTL (1h) and
reliability rolling window (1h) expire together. After `bc005a0`
shipped, the previously-dead-marked openrouter ids became eligible
again with NO failure history → reliability defaults to 1.0 → selector
ranks them top → admit → fail → ❌. Subsequent failures DO build
history but in the meantime each "first" call after revival is wasted.

The Open Question discussion needs to settle:
1. Reliability as a real pressure signal (not multiplier).
2. How to handle "freshly-revived" model history — TTL extension,
   dead-mark seeding, or explicit penalty period.

## Open Items Beyond This Session

- **Notification noise**: each `ModelCallFailed` log fires a Telegram ❌
  message. The current `6256ad0` design fires ❌ + increments attempts
  per retry, which means a single saturated task fires up to 10 ❌
  messages over its retry budget. User has reverted the silencer
  (`5e666e6` → `157b270`) once before with "I sent wrong messages".
  Status: silencer is NOT in tree. Re-applying would drop notification
  volume by ~90% during saturation but user hasn't re-approved.

- **Sweep section 8 default**: now 10. Tasks that came in with
  `max_worker_attempts=6` (older default) won't auto-bump — they'll DLQ
  at 6/6. New tasks default to 10. No migration script written.

- **constrained_emit `needs_json_mode` TypeError**: fixed in `4b73278`,
  but worth re-verifying live.

- **Sidecar / db lock contention**: `627339d` added retry-on-locked
  for `update_task` and singleton `tx` visibility in slow-region
  warnings. Real holder still unidentified — could be `nerd_herd_sidecar`
  separate process holding writer for >120s during heartbeat watchdog
  triggers. Worth investigating if "Kutay dondu" returns post `72fbb2a`.

## File Paths Referenced

- `packages/general_beckman/src/general_beckman/retry.py` — `decide_retry`,
  `_BACKOFF_SECONDS`, DLQ logic
- `packages/general_beckman/src/general_beckman/apply.py` — `_retry_or_dlq`
- `packages/general_beckman/src/general_beckman/sweep.py` — section 8
  over-cap force-DLQ
- `packages/general_beckman/src/general_beckman/__init__.py` — `next_task`,
  `on_task_finished`, `_send_step_progress` (Telegram ❌ source at line
  ~550)
- `packages/fatih_hoca/src/fatih_hoca/selector.py` — eligibility
  filters, `_check_eligibility`
- `packages/fatih_hoca/src/fatih_hoca/ranking.py` —
  `_apply_utilization_layer` (reliability multiplier currently here)
- `packages/fatih_hoca/src/fatih_hoca/registry.py` — `mark_dead`,
  `is_dead`, TTL machinery
- `packages/nerd_herd/src/nerd_herd/types.py` — `pressure_for`,
  `CloudProviderState`, `CloudModelState`, `InFlightCall`
- `packages/nerd_herd/src/nerd_herd/combine.py` — `combine_signals`,
  bucket weights, abundance gate
- `packages/nerd_herd/src/nerd_herd/signals/` — S1-S11 implementations
  (S12 reliability would go here)
- `packages/kuleden_donen_var/src/kuleden_donen_var/kdv.py` — KDV
  main class, canary gate, outcome window, `recent_success_rate`
- `packages/kuleden_donen_var/src/kuleden_donen_var/rate_limiter.py` —
  rpd local decrement, calendar reset
- `packages/kuleden_donen_var/src/kuleden_donen_var/nerd_herd_adapter.py` —
  `build_cloud_provider_state`, plumbing KDV state to nerd_herd snapshot
- `packages/hallederiz_kadir/src/hallederiz_kadir/caller.py` —
  `_clean_provider_name_in_error`, runtime 404 mark_dead routing,
  custom_llm_provider override
- `packages/hallederiz_kadir/src/hallederiz_kadir/retry.py` —
  `classify_error`, `execute_with_retry`
- `src/core/llm_dispatcher.py` — `ModelCallFailed` raise sites, retry
  recursion
- `src/core/router.py` — `ModelCallFailed` definition
- `src/core/orchestrator.py` — `_dispatch`, `ModelCallFailed` catch,
  result→`on_task_finished` routing
- `src/core/in_flight.py` — `_InFlightEntry`, `reserve_task`,
  `begin_call`, `_push`
- `src/memory/embeddings.py` — `get_embedding`, `to_thread` offload
