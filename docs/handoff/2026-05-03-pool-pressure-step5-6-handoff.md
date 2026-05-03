# Handoff — Pool-Pressure Pipeline Steps 5+6 (next session)

User: sakircimen@gmail.com
Caveman mode: drop articles/filler/pleasantries. Fragments OK.
Code/commits/security: write normal.

Anchor: 2026-05-03

---

## Where we landed (steps 1-4 already shipped)

| SHA       | Step                                                                   |
|-----------|------------------------------------------------------------------------|
| `b2576ce` | 1. KDV outcome window 1h → 24h. Decouples from mark_dead TTL.          |
| `0b2473f` | 2. admission_violations forensics + mid-task urgency bump (+0.1).      |
| `7a555f9` | 3. S1 stock / S9 timing separation + noisy-OR positive arm.            |
| `d2f22d7` | 4. S10 per-model rate signal + delete reliability multiplier.          |

714 tests pass across nerd_herd + fatih_hoca + kuleden_donen_var.

The pipeline is now: KDV outcome window (24h) → CloudModelState
(`recent_success_rate` + `recent_samples_n`) → S10_failure signal in
OTHER_BUCKET → composite scoring with M3 weights + noisy-OR positive
arm. Single source of truth for reliability.

---

## Step 5 — SQLite registry (drop .dead_models.json)

### Goal
Replace flat-JSON dead-model tracking with a real persistent registry.
User feedback (2026-05-03): "Not intelligent enough. We can build a
persistent provider and model registry that doesn't rely on a json or
managed by hacks."

### Current state to replace
- `.dead_models.json` — flat dict `{id: expiration_ts}`. Loaded by
  `fatih_hoca/registry.py:1237`. Backups at `.dead_models.json.bak.*`.
- `mark_dead(identifier)` in `registry.py:1327` — sets `_dead_models[id] =
  now + 3600`, persists to JSON.
- `is_dead(identifier)` in `registry.py:1366-1369` — TTL check, auto-
  cleanup on read.
- Auth-failure cascade in `caller.py:813` — loops every non-local model
  on provider, calls `mark_dead(litellm_name)`. Single bad key creates
  30+ dead-mark entries.
- Discovery revive in `fatih_hoca/__init__.py:279-286` — when
  `discovery.refresh_all()` finds previously-dead id in /v1/models, calls
  `registry.revive(litellm_name)`.

### Target schema (new tables in `src/infra/db.py`)

```sql
CREATE TABLE providers (
    name TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'active',  -- active|dead|degraded
    cause TEXT,                              -- auth|network|manual|...
    marked_at TIMESTAMP,
    revived_at TIMESTAMP,
    last_probe_at TIMESTAMP,
    last_probe_result TEXT,
    key_hash TEXT,                           -- detect rotation
    notes TEXT
);

CREATE TABLE models (
    litellm_name TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',   -- active|dead
    cause TEXT,                              -- 404_permanent|404_transient|auth|...
    marked_at TIMESTAMP,
    revived_at TIMESTAMP,
    source TEXT,                             -- yaml|discovery|runtime
    first_seen_at TIMESTAMP,
    last_success_at TIMESTAMP,
    last_failure_at TIMESTAMP,
    total_calls INTEGER DEFAULT 0,
    total_failures INTEGER DEFAULT 0,
    FOREIGN KEY (provider) REFERENCES providers(name)
);

CREATE TABLE registry_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    scope TEXT NOT NULL,                     -- provider|model
    target TEXT NOT NULL,                    -- name or litellm_name
    event TEXT NOT NULL,                     -- mark_dead|revive|probe|status_change
    cause TEXT,
    actor TEXT,                              -- auto|user|probe|discovery
    payload_json TEXT
);

CREATE INDEX idx_registry_events_target_ts
    ON registry_events(target, timestamp DESC);
CREATE INDEX idx_models_status ON models(status, provider);
```

### Behavior changes

1. **Auth-failure → provider-level dead** (one row, not 30)
   - `caller.py:788-821` auth path: replace per-model loop with
     `mark_provider_dead(provider, cause="auth")`.
   - Selector eligibility: add `is_provider_dead(provider)` check.

2. **mark_dead with cause + per-cause TTL/policy table**

   ```python
   _CAUSE_POLICY = {
       "auth":           {"ttl": None, "manual_revive": True},
       "404_permanent":  {"ttl": 86400, "manual_revive": False},
       "404_transient":  {"ttl": 300,   "manual_revive": False},
       "server_error":   {"ttl": 600,   "manual_revive": False},
       "manual":         {"ttl": None,  "manual_revive": True},
   }
   ```

   - Permanent (auth, manual): no TTL, requires `/revive` Telegram
     command or `.env` mtime watcher trigger.
   - 404_transient (openrouter "no endpoints found" pattern, see
     `caller.py:744-750`): 5min TTL — fast recovery for routing
     failures.
   - 404_permanent: 24h TTL.
   - `caller.py` should pass cause when calling mark_dead.

3. **Probe-on-process-start**
   - At orchestrator boot, fire 1 cheap auth-test request per cloud
     provider. Auth-fail → mark provider dead with cause="auth".
     Auth-OK → trust persisted state.
   - Trivial cost (~2s parallel), prevents stale "key was bad" lockout
     surviving across restart-after-key-fix.

4. **Key rotation detection**
   - Hash provider's API key (sha256, first 8 hex). Store in
     `providers.key_hash`. On startup, compare current vs persisted —
     mismatch → auto-revive provider, force re-probe.

5. **Migration script** (one-shot, idempotent)
   - Read `.dead_models.json`, write rows to `models` table with
     status='dead', cause inferred from existence (since old format
     has no cause: default to "404_permanent" with full TTL).
   - Read YAML catalog, write `models` rows with status='active',
     source='yaml'.
   - Archive `.dead_models.json` to `.dead_models.json.archived`.
   - Log migration summary to `registry_events`.

### Files to touch (estimate)

- `src/infra/db.py` — add 3 tables + indexes + migration helpers
- `packages/fatih_hoca/src/fatih_hoca/registry.py` — gut JSON impl,
  delegate mark_dead/is_dead/revive to new SQL helpers. Keep public
  API stable.
- `packages/fatih_hoca/src/fatih_hoca/__init__.py:279` — discovery
  revive uses new API.
- `packages/hallederiz_kadir/src/hallederiz_kadir/caller.py:744-821` —
  pass cause to mark_dead, use mark_provider_dead for auth.
- `packages/fatih_hoca/src/fatih_hoca/selector.py` — add
  `is_provider_dead` check in eligibility (around line 377 where
  `is_dead` already lives).
- New: `src/infra/registry_store.py` — SQL backend for mark_dead,
  is_dead, mark_provider_dead, is_provider_dead, revive, list_dead,
  cause queries, audit log writes.
- New: `scripts/migrate_dead_models_to_sqlite.py` — one-shot.
- `src/app/telegram_bot.py` — add `/revive <provider>` command.
- Tests: `packages/fatih_hoca/tests/test_registry.py` — extend with
  per-cause TTL tests, provider-level tests.

### Pitfalls / things to watch

- **Backward compatibility during migration**: registry.py's mark_dead
  is called from a half-dozen places. The function signature change
  (adding `cause` param) needs default value to keep callers working.
- **Async vs sync**: registry.py uses sync calls. New SQL backend
  needs sync wrapper around aiosqlite OR migrate registry to async.
  Sync wrapper is safer — registry is called from selector hot path.
- **db lock contention**: handoff `2026-05-02-session-handoff.md`
  notes `627339d` added retry-on-locked for update_task. Same risk
  here — provider/model status updates fire on every failed call.
  Use `get_db()` singleton path (per `pick_log.py` rationale,
  src/infra/pick_log.py:14-23).
- **Probe failure modes**: probe should NOT mark dead on network
  timeout (transient). Only auth failures (401/403) → mark.
- **Key rotation false-positives**: if user has multiple keys
  (KEY1 / KEY2 fallback), hash mismatch isn't always rotation.
  Acceptable false-positive: extra probe call.
- **`/revive` command authorization**: only owner chat_id should be
  allowed (existing pattern in telegram_bot.py).

---

## Step 6 — Provider prior + openrouter sub-provider

### Goal
S10 currently returns 0 (neutral) when samples_n < 5. That's better
than the old 1.0-success-rate-default (which made revived models rank
top), but new/revived models still have no signal at all until 5+
calls accumulate. User design 2026-05-03: provider-level prior fills
the gap.

### Approach

Aggregate `_outcomes` deques across siblings on the same provider.
When a model has fewer than MIN_SAMPLES of its own data, S10 uses the
provider's aggregate rate as a prior.

```python
# new helper in nerd_herd_adapter.py or kdv.py
def provider_prior_rate(kdv, provider: str, window_secs: float = 300.0):
    """Aggregate success rate across all models on provider, last 5min."""
    cutoff = time.time() - window_secs
    total, ok = 0, 0
    for mid in kdv._providers.get(provider, set()):
        dq = kdv._outcomes.get(mid)
        if not dq: continue
        for ts, success in dq:
            if ts < cutoff: continue
            total += 1
            if success: ok += 1
    if total < MIN_PROVIDER_SAMPLES:  # e.g. 3
        return None  # no prior
    return ok / total
```

Plumb prior into `CloudModelState.provider_prior_rate: float | None`.
Update `s10_failure` signature:

```python
def s10_failure(*, success_rate=1.0, samples_n=0,
                provider_prior_rate=None, consecutive_failures=0):
    if samples_n >= MIN_SAMPLES:
        # use own data (current behavior)
        ...
    elif provider_prior_rate is not None:
        # apply curve to prior
        rate = provider_prior_rate
        # same linear interp
        ...
    else:
        rate_signal = 0.0
    ...
```

### Openrouter sub-provider exception

Openrouter is structurally different (routing aggregator, not a
provider). Per-id reliability varies wildly. Group by sub-provider
(litellm_name path segment):

```python
# nerd_herd_adapter.py or new helper
def _prior_key(model) -> str:
    """Identifier for provider_prior aggregation. Openrouter splits
    by sub-provider (vendor) since failure modes are vendor-specific
    (anthropic vs tencent vs meta-llama backends behave very
    differently on openrouter)."""
    if getattr(model, "provider", "") == "openrouter":
        # openrouter/<vendor>/<model>:free → key="openrouter::vendor"
        parts = getattr(model, "litellm_name", "").split("/")
        if len(parts) >= 3:
            return f"openrouter::{parts[1]}"
    return getattr(model, "provider", "")
```

For openrouter, `provider_prior_rate` aggregates only models matching
the same `_prior_key()`. Other providers aggregate at provider level
as-is.

### Files to touch

- `packages/kuleden_donen_var/src/kuleden_donen_var/kdv.py` — add
  `provider_prior_rate(provider_or_key, window_secs)` method.
- `packages/kuleden_donen_var/src/kuleden_donen_var/nerd_herd_adapter.py` —
  compute prior per model, plumb into CloudModelState.
- `packages/nerd_herd/src/nerd_herd/types.py` — add
  `provider_prior_rate: float | None` to CloudModelState.
- `packages/nerd_herd/src/nerd_herd/signals/s10_failure.py` — accept
  prior param, fallback to it when own samples insufficient.
- `packages/nerd_herd/src/nerd_herd/types.py:pressure_for` — pass
  prior through.
- `packages/nerd_herd/tests/signals/test_s10.py` — add prior tests.

### Pitfalls

- **Don't double-count**: if own samples_n >= MIN_SAMPLES, ignore
  prior entirely. Prior is a fallback, not a blend.
- **Openrouter key collision**: `openrouter::tencent` for
  `openrouter/tencent/hy3-preview:free` AND `openrouter/tencent/foo`.
  Same prior key — by design.
- **Cold provider**: if NO models on provider have any samples (full
  cold start), prior is None too → S10 returns 0. Acceptable —
  there's no data anywhere.
- **MIN_PROVIDER_SAMPLES tuning**: too low → noisy. Too high → never
  fires. Start at 3 (lower than per-model MIN_SAMPLES=5 because
  provider aggregates many models so noise floor is naturally
  higher per call but lower per provider-aggregate).

---

## Sequencing within session

Recommended order:

1. **Step 5 first**, in two sub-commits:
   1a. Schema + registry_store + tests (no behavior change yet).
   1b. Wire in caller.py + registry.py + selector.py + migration
       script + telegram /revive command. Single coherent commit.
2. **Step 6** as final commit on top:
   2a. Provider prior helpers + plumbing + openrouter sub-key + tests.

Each sub-commit shippable independently.

---

## Open architectural questions raised but not resolved

(From the original 2026-05-02 session, addressed but worth re-checking
after step 5+6 land:)

- ❌ notification consolidation — silencer reverted by user, no dedup
  yet. Burns user attention during saturation. Worth a `/silent` toggle?
- Sweep section 8 default cap mismatch — old tasks `max_worker_attempts=6`
  won't auto-bump to 10, will DLQ early. No migration script written.
- in-flight reservation (`94591e3`) only Beckman path. Overhead +
  direct-dispatcher calls bypass `est_tokens`. Pool overshoots for
  non-Beckman traffic.
- Estimate quality (`estimated_output_tokens` 4-8× off on heavy tail
  per April scan) — bleeds into in_flight reservations being wrong.
- Discovery cadence undefined — when does `discovery.refresh_all()`
  re-run? Cold-start only? Worth periodic schedule.

---

## Architecture facts (don't relearn)

- **DB_PATH**: `C:\Users\sakir\ai\kutai\kutai.db` per `.env`. Touching
  init_db while orchestrator is running may hang on busy lock — use
  `/restart` or temp-DB for smoke tests.
- **Test command**: `timeout 120 .venv/Scripts/python.exe -W ignore -m
  pytest packages/<pkg>/tests/ -q` — always with timeout.
- **Outcome window** (`_OUTCOME_MAX_AGE_SECONDS`): now 86400 (24h).
  Don't revert without re-introducing the revival cycle.
- **mark_dead TTL** (`_DEAD_TTL_SECONDS`): currently 3600 (1h). Will
  become per-cause table in Step 5.
- **S10 signal**: takes `success_rate`, `samples_n`,
  `consecutive_failures`. After Step 6, add `provider_prior_rate`.
- **Reliability multiplier**: DELETED in `d2f22d7`. Don't reintroduce
  — S10 is single source of truth.
- **POSITIVE_ARM**: noisy-OR over (S1, S9). `1 - (1-S1+)(1-S9+)`.
  Inputs clamped to [0, 1] before composition.
- **S1 abundance** for time_bucketed: now `proportional` (frac × 1.0).
  Time component lives in S9 only.
- **S9 free cloud**: pure proximity `1 - min(1, reset_in / 3600)`.
  Existence check (any cell remaining > 0).

---

## DON'T

- DON'T re-introduce the reliability multiplier in ranking.py. S10
  owns reliability now.
- DON'T re-couple outcome window TTL with mark_dead TTL. Decoupling
  was the whole point of step 1.
- DON'T silently mass-mark cloud models on auth failure. Step 5's
  whole point: provider-level dead, single row.
- DON'T `pytest` without `timeout` prefix.
- DON'T `taskkill llama-server`. Use `/restart` via Telegram.
- DON'T `call_model()` directly — `LLMDispatcher.request()` only.
- DON'T tune simulator thresholds reactively. groq_near_reset 0.85
  floor in Step 3 was a documented intentional trade-off; future
  scenario regressions need investigation, not threshold relaxation.

---

## Where to start (fresh session)

1. Read this handoff.
2. Read `docs/handoff/2026-05-02-session-handoff.md` and
   `docs/handoff/2026-05-03-session-handoff.md` (if exists) for
   broader context on what failure modes the pipeline serves.
3. Re-verify steps 1-4 still pass:
   `timeout 120 .venv/Scripts/python.exe -W ignore -m pytest
   packages/nerd_herd/tests/ packages/fatih_hoca/tests/
   packages/kuleden_donen_var/tests/ -q`
   Expect 714+ passing.
4. Begin Step 5 schema design. Confirm column choices with user
   before coding the SQL.
5. Land Step 5 in 2 sub-commits, then Step 6 as final commit.

User prefers: ship as you go, no excessive confirmation per step,
forensic instrumentation before behavior change.
