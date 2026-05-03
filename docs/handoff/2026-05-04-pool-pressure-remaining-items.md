# Handoff — Pool-Pressure Pipeline: Remaining Items + Live Forensics

User: sakircimen@gmail.com
Caveman mode: drop articles/filler/pleasantries.
Code/commits/security: write normal.

Anchor: 2026-05-04

---

## Where we landed (this session)

| SHA       | What                                                                |
|-----------|---------------------------------------------------------------------|
| `b2576ce` | KDV outcome window 1h → 24h. Decouples from mark_dead TTL.          |
| `0b2473f` | admission_violations forensics + mid-task urgency bump (+0.1).      |
| `7a555f9` | S1 stock / S9 timing separation + noisy-OR positive arm.            |
| `d2f22d7` | S10 per-model rate signal + delete reliability multiplier.          |
| (parallel)| Step 5 SQLite registry — done parallel, not in this session's work. |
| (parallel)| Step 6 provider prior + openrouter sub-vendor.                      |
| `8002ff0` | **Critical fix**: client._parse_snapshot preserves per-model state. |
| `5f7f905` | in_flight est_tokens plumbing for non-Beckman dispatcher paths.     |

755 tests pass.

---

## Live Forensics (production, last 2h pre-fix)

`admission_violations` table (added in `0b2473f`) revealed:

| Count | Site | Reason | Status |
|------:|------|--------|--------|
| 294 | daily_exhausted_at_call | daily_exhausted | **FIXED by `8002ff0`** |
| 85  | kdv_pre_call_refusal | tpm | Open — addressed by `5f7f905`, verify live |
| 82  | kdv_pre_call_refusal | canary_in_flight | Open — investigate volume |
| 69  | kdv_pre_call_refusal | rate_limit | Open — pressure model under-predicts |
| 43  | dispatcher_pool_empty | no_candidates | Should improve after Steps 4+6 |

Top offenders by model:
- gemini (346 across 4 ids) — 138 flash + 97 flash-lite + 59 flash-latest + 52 flash-lite-latest
- groq (119 across 3 ids)
- cerebras (26)
- openrouter (8)

**The 294 daily_exhausted_at_call cluster was the root cause of the
DLQ pile-up.** `_parse_snapshot()` in NerdHerdClient stripped the
`models` dict + `circuit_breaker_open` from sidecar JSON, so selector
eligibility checks (selector.py:489 / :503-525) silently no-op'd.
After `8002ff0`, selector should drop daily_exhausted models BEFORE
ranking → no more admission to exhausted pools → no more
daily_exhausted_at_call violations.

---

## Remaining Items (priority-sorted, with forensics evidence)

### P1 — Verify live behavior of `8002ff0`

After `/restart`, query:

```sql
SELECT site, reason, COUNT(*) AS n
FROM admission_violations
WHERE timestamp > datetime('now', '-1 hour')
GROUP BY site, reason ORDER BY n DESC;
```

Expected: daily_exhausted_at_call near zero. If still firing > 10/h,
something else is bypassing the eligibility check (investigate
selector.py:486-525 path, especially the contextvar route between
admission and the selector's snapshot read).

### P1 — TPM gate violations (85 captured)

`5f7f905` plumbed est_tokens through dispatcher → in_flight overlay.
Should reduce concurrent-admission overshoots on standalone paths
(workflow hooks, classifiers, telegram-driven, shopping). Verify:

```sql
SELECT model, provider, COUNT(*) AS n
FROM admission_violations
WHERE site='kdv_pre_call_refusal' AND reason='tpm'
  AND timestamp > datetime('now', '-1 hour')
GROUP BY model, provider ORDER BY n DESC;
```

Expected: drop materially (50%+) on standalone-heavy providers (groq,
gemini). If still high → estimate quality is the issue (see P3).

### P1 — Notification consolidation (53+ ❌ messages from one DLQ pile)

Each `ModelCallFailed` log fires Telegram ❌. With 10-attempt retry
budget per task and 53 tasks failing, ~500+ ❌ over a few minutes.
Burns user attention budget.

The silencer was reverted in handoff `2026-05-02` (commits `5e666e6`
→ `157b270`). User said "I sent wrong messages" but didn't re-approve.

**Suggested approach** (lower-risk than full silencer):
- Dedup adjacent identical ❌ messages within a 60s window per task
- Roll up "task X retried for availability (5 attempts)" instead of
  5 separate messages
- Add `/silent` toggle for power users during known-saturation periods
- Don't suppress final DLQ message — that's the actual signal

Files: `packages/general_beckman/src/general_beckman/__init__.py:564-604`
(`_send_step_progress`), `apply.py:343-357` (DLQ notify).

### P2 — Canary gate firing 82×

Cold-start design — first call after revival/breaker-reset is held
solo until verified. Volume suggests:
- Frequent restarts (`Yaşar Usta` auto-restart after heartbeat
  watchdog kill)
- Canary-failure resets (provider keeps failing → `_canary_verified`
  re-flips to False on every failure)

Investigate first. Don't tune the gate until cause clear. Probable
fix:
- Reduce restart frequency (root-cause heartbeat kills — see Yaşar
  Usta investigation in handoff `2026-05-02`)
- Or relax canary to allow 2-3 concurrent on previously-successful
  providers, only strict on truly-cold ones

### P2 — Estimate quality (4-8× off on heavy tail)

April scan (`docs/research/2026-04-28-token-distribution.md`):
analyst ×8.34 over-estimate, writer ×3.86, architect ×5.52.

Bleeds into:
- KDV TPM reservation (over-reserves → false rate_limit refusals)
- in_flight overlay est_tokens (over-back-pressures concurrent
  admissions)
- pressure_for S2/S3 burden signals (false negative pressure)

Files: `packages/fatih_hoca/src/fatih_hoca/estimates.py` —
`AGENT_REQUIREMENTS.estimated_output_tokens` per agent_type.

**Suggested fix**: drive estimates from `model_call_tokens` table.
There's a Beckman rollup cron (Task 26) that already aggregates this
per agent_type → median + p90 → profile. Verify it's running and
feeding back into estimates.

### P2 — Estimate the impact of estimate quality fix vs other items

Quick audit (~30min): query `model_call_tokens` per agent_type, get
actual median + p90, compare to AGENT_REQUIREMENTS. Show drift table.
That tells whether this is high-impact or noise.

### P3 — Provider naming normalization

`gemini` vs `google` vs `vertex_ai_beta` rewriting handled per-layer
(`e9d5f57`, `2cd62bb`). Brittle. Single canonicalization layer
missing.

Worth doing if: any future bug surfaces from an ID mismatch (selector
keys ≠ KDV keys). For now, low-priority — `8002ff0` ruled out the
specific litellm_name mismatch I worried about.

### P3 — Discovery cadence undefined

`fatih_hoca/__init__.py:279` discovery.refresh_all() — when does it
re-run? Cold-start only? Periodic? On-demand?

If cold-start only: dead models (per Step 5's SQLite registry) stay
dead until process restart, even when upstream provider restored.
Discovery → revive path won't fire.

Suggested: periodic discovery refresh (every 30min?) wired into
nerd_herd's existing scheduler. Reuse same retry-after / backoff as
KDV.

### P3 — DLQ cause attribution

DLQ tasks logged with opaque error text. No category for
"env failure" vs "schema bug" vs "quality issue" vs "true model
death". Forensics weak — every triage requires manual log diving.

Files: `packages/general_beckman/src/general_beckman/apply.py:313-357`
(`_dlq_write`), `src/infra/dead_letter.py`.

Suggested: extend `quarantine_task` to take `cause_category` enum.
Use `error_category` from the failed call as input.

### P4 — Misc deferred

- **Burn log persistence (S7)**: does burn log survive restart? Cold
  start zeroes S7 for 5min. Low priority.
- **grading_perf_score source**: who writes `model_stats` table?
  Cadence? Stale data poisons Layer 2 perf score. Investigate when
  Layer 2 scoring shows weird picks.
- **Mechanical tasks (salako) admission**: do they consume worker
  slots? Block real LLM work? Verify.

### Skip — `_failure_penalty` in ranking.py

Initially flagged as "duplicates new S10." On closer read it doesn't —
it's per-task fail tracking from the current `failures` list (Failure
objects from THIS task's retry chain), not from KDV outcome history.
S10 reads rolling window across tasks. Different scope. Keep both.

---

## Architecture facts (don't relearn)

- **DB_PATH**: `C:\Users\sakir\ai\kutai\kutai.db` per `.env`. Touching
  init_db while orchestrator running may hang on busy lock.
- **Test command**: `timeout 120 .venv/Scripts/python.exe -W ignore
  -m pytest packages/<pkg>/tests/ -q` — always with timeout.
- **Outcome window** (`_OUTCOME_MAX_AGE_SECONDS`): 86400 (24h).
- **mark_dead TTL**: was 3600s flat. Step 5 SQLite registry should
  introduce per-cause TTLs (verify after Step 5 review).
- **S10 signal**: takes `success_rate`, `samples_n`,
  `provider_prior_rate`, `consecutive_failures`. Reliability
  multiplier in ranking.py is DELETED — single source of truth.
- **POSITIVE_ARM**: noisy-OR over (S1, S9). `1 - (1-S1+)(1-S9+)`.
  Inputs clamped to [0, 1] before composition.
- **S1 abundance** for time_bucketed: `proportional` (frac × 1.0).
  Time component lives in S9 only.
- **S9 free cloud**: pure proximity `1 - min(1, reset_in / 3600)`.
- **in_flight begin_call**: now takes `est_tokens: int = 0`. Beckman
  task slot uses `max(prior_reserve, passed)`. Standalone uses
  passed value directly.
- **Per-model state in CloudProviderState**: includes
  `recent_success_rate`, `recent_samples_n`, `provider_prior_rate`,
  `daily_exhausted`, `rpm_cooldown`. ALL plumbed through HTTP
  snapshot now (was broken pre-`8002ff0`).
- **admission_violations table**: 3 sites write to it
  (caller.py KDV refusal, caller.py daily_exhausted_at_call,
  dispatcher.py pool_empty). Telemetry only — never gates admission.

---

## DON'T

- DON'T revert any of the commits listed above. They form a
  coherent layered fix.
- DON'T re-introduce the reliability multiplier in ranking.py.
- DON'T re-couple outcome window TTL with mark_dead TTL.
- DON'T silently mass-mark cloud models on auth failure (per Step 5).
- DON'T `pytest` without `timeout` prefix.
- DON'T `taskkill llama-server`. Use `/restart` via Telegram.
- DON'T `call_model()` directly — `LLMDispatcher.request()` only.
- DON'T tune simulator thresholds reactively. groq_near_reset 0.85
  floor in Step 3 was a documented intentional trade-off.
- DON'T relax the per-model_state plumbing in `_parse_snapshot()` —
  the silent-strip bug pattern is exactly what caused the 294-
  violation cascade.

---

## Forensics queries you'll want

```sql
-- Recent violations by site + reason
SELECT site, reason, COUNT(*) FROM admission_violations
WHERE timestamp > datetime('now', '-2 hour')
GROUP BY site, reason ORDER BY 3 DESC;

-- Per-model violation hotspots
SELECT model, provider, COUNT(*) FROM admission_violations
WHERE timestamp > datetime('now', '-2 hour') AND model != ''
GROUP BY model, provider ORDER BY 3 DESC LIMIT 20;

-- Reliability snapshot per model (post-S10 wiring)
SELECT picked_model, provider, AVG(picked_score), COUNT(*)
FROM model_pick_log
WHERE timestamp > datetime('now', '-1 hour')
GROUP BY picked_model, provider ORDER BY 4 DESC;

-- Verify Step 4+6 effect — pool_empty trend
SELECT strftime('%Y-%m-%d %H:00', timestamp) AS hour, COUNT(*)
FROM admission_violations
WHERE site='dispatcher_pool_empty'
GROUP BY hour ORDER BY hour DESC LIMIT 24;
```

---

## Where to start (fresh session)

1. Read this handoff + `2026-05-04-unify-non-beckman-paths.md`
   (companion handoff for the architectural question).
2. Read `2026-05-03-pool-pressure-step5-6-handoff.md` for the
   Step 5 SQLite registry context (done parallel by another
   agent).
3. Verify `/restart` happened post `8002ff0` and `5f7f905`. If not,
   ask user to restart.
4. Run the forensics queries above. Compare violation counts to the
   pre-fix baseline at top of this doc.
5. If daily_exhausted_at_call near zero → `8002ff0` confirmed.
   Move to next P1 item.
6. If still firing → investigate selector contextvar/snapshot
   timing. The fix targets the parse strip, but if there's ANOTHER
   bypass path it'll show up here.

User prefers: ship as you go, no excessive confirmation per step,
forensic instrumentation before behavior change. No reactive knob
tightening — collect evidence first.
