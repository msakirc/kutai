# Step 1 — Write-consolidation work-list (2026-06-11)

**Status: EXECUTED 2026-06-11 — merged to main 83ba7d8e (NOT pushed, restart-gated).** All chunks below shipped except comms (deferred, see Decisions). Beckman = single write-owner of missions/tasks/growth_events; founder_actions owns its table; 4 AST-hardened guard tests in packages/general_beckman/tests/test_{task,mission}_write_api.py enforce it (zero exemptions beyond src/infra/db.py + general_beckman). Post-merge: beckman 329/329, app+founder+infra+workflows 517/518 (1 pre-existing z6 reversibility), mr_roboto+coulson 996/996.

Follow-on debt recorded: beckman __init__.py now ~1.9k LOC (thin delegates) — consider write_api.py submodule split; block/unblock PRAGMA probe uncached per call; update_mission now commits immediately (no caller relied on batching, verified); reset_workflow_step returns None vs reset_* int convention.

Original scoping below (kept for reference). Continues `2026-06-11-db-ownership-split-eval-handoff.md` (NEXT STEP). Three exhaustive scout passes produced the full writer matrix below (prod only, tests excluded).

**Goal per table:** one owner module exposes the only write API; every other writer routes through it. This kills the `package→src/infra/db.py` reach and is the precondition for any file split.

---

## 1. `tasks` — owner: **general_beckman** (task master, already owns lifecycle)

Worst offender. Writers found in **7 modules** (handoff said 5 — undercount; add src/workflows + mr_roboto).

### Current writers via db.py helpers
| Helper (db.py) | Callers outside beckman |
|---|---|
| `add_task` (4351) | src/app/api.py:161 · telegram_bot.py ×8 (1724,1734,2586,7534,8726,8764,10425) · src/core/orchestrator.py:421 · workflows/engine/runner.py:229,473 · mr_roboto monitoring_check.py:68, mention_monitor_sweep.py:86 |
| `update_task` (4785) | telegram_bot.py ×11 · orchestrator.py:160 · state_machine.py:165 · infra/dead_letter.py:271,333 · workflows/engine/hooks.py:1670,2261,2297 · runner.py:201 · task_refresh.py:74 · mr_roboto clarify.py:202,325,406, request_interview_data.py:133, __init__.py:759 |
| `update_task_by_context_field` (4805) | workflows/engine/hooks.py:2022 |
| `add_subtasks_atomically` (4959) | workflows/engine/hooks.py:2133 |
| `propagate_skips` (5184) | workflows/engine/hooks.py:2030 |
| `save_task_checkpoint` (5299) | infra/dead_letter.py:329 |
| `cancel_task` (5689) | telegram_bot.py:3322 |
| `reprioritize_task` (5741) | telegram_bot.py:3366 |

### Raw-SQL escapes (worst — bypass even db.py)
- telegram_bot.py:3203,3213,3224 — reset failed/processing→pending, clear deps
- telegram_bot.py:6687 — cancel pending tasks for mission
- telegram_bot.py:10010,10049 — bulk DELETE all tasks
- telegram_bot.py:10937,10945,10953 — reset status for workflow regen
- src/core/startup_recovery.py:28,53 — startup reset processing→pending, clear next_retry_at
- src/infra/dead_letter.py:348 — DLQ recovery reset
- src/founder_actions/__init__.py:499 — unblock: blocked_on_founder_action→pending

### Route-to API (Beckman gains)
`beckman.enqueue()` already covers add_task for LLM work. Needed additions:
- `beckman.add_task()` passthrough for non-LLM/UI adds (or widen enqueue)
- `beckman.update_task_fields()` — narrow, audited field-update (status, context, retry)
- `beckman.cancel(task_id|mission_id)`, `beckman.reprioritize()`
- `beckman.reset_to_pending(filter)` — covers telegram resets, startup_recovery, DLQ recovery, founder unblock
- `beckman.purge_all()` — bulk delete (admin)
- Workflow-engine bulk ops (`add_subtasks_atomically`, `propagate_skips`) — either move into beckman or bless workflows/engine as a beckman-internal collaborator (decision needed; engine is conceptually Beckman's expander arm)

**Effort: L** (~30 call sites in telegram_bot alone; apply.py's ~90 update_task calls are already beckman-internal = no-op). Mechanical but wide; biggest single chunk is telegram_bot.

---

## 2. `missions` — owner: **general_beckman**

Writers in **6 modules** (handoff said 3 — big undercount; mr_roboto has 10 raw-SQL files, plus src/telemetry + src/workflows).

### Via db.py helpers
- `add_mission` (4165) ← telegram_bot.py ×7 · workflows/engine/runner.py:388
- `update_mission` (4278) ← telegram_bot.py ×6 · infra/dead_letter.py:149 · infra/projects.py:33
- `increment_mission_rework_loops` (4287) ← src/telemetry/rework.py:117

### Raw-SQL escapes
- **src/app/api.py:129 — direct INSERT bypassing add_mission → misses product_id/lifecycle init. LATENT BUG, fix first.**
- telegram_topics.py:33,42 — thread id/archived
- telegram_bot.py:5291,7039,9140 — attention budget / cost ceiling / branched_from
- telegram_bot.py:10011,10050 — bulk DELETE all missions
- attention_budget.py:383 — attention budget minutes
- founder_actions/__init__.py:455,490 — block/unblock lifecycle_state (dynamic column)
- workflows/review_density.py:158 — review_density_json
- **mr_roboto ×10 files, ALL raw SQL, zero helpers:** emit_preview_url.py:401,407 · init_mission_github_repo.py:342 · inject_lessons.py:121 · executors/inject_north_star.py:169 · executors/arm_analytics_digest.py:100 · publish_preview_pages.py:245 · executors/validate_target_segment.py:139 · request_interview_data.py:63 · z0_preflight.py:116

### Route-to API
- Fix api.py:129 → `add_mission` (standalone bug fix, ship independently)
- `beckman.update_mission_fields(mission_id, **fields)` — single audited setter covers ~all escapes (preview_url, github_repo_url, context injections, cursors, thread ids, budgets)
- `beckman.block_mission()/unblock_mission()` — wraps founder_actions' lifecycle toggles (pairs with tasks reset)
- mr_roboto pattern: mechanical executors should return results; mission-field writes routed through one beckman call from the executor (mr_roboto already depends on db.py — swap to beckman API, no dep inversion)

**Effort: M–L.** ~30 sites, but `update_mission_fields` is one uniform shape — mostly find/replace per file.

---

## 3. `growth_events` — owner: **general_beckman** (spine-bound, stays in core.db)

Insert path already single-funnel: `insert_growth_event` (db.py:7313). Callers: telegram_bot.py:12079,12544 · webhook_listener.py:535 · infra/dlq_feedback.py:170 · mr_roboto ×11 executor sites (analytics_digest ×4, assign_variant ×3, classify_signals, record_hypothesis ×3, record_verdict ×2, retire_variant, roadmap_sync, score_backlog, score_sunset).

Raw escapes — `properties_json` "superseded"-flag UPDATEs ×5: telegram_bot.py:12071,12536 · roadmap_sync.py:228 · score_backlog.py:341 · score_sunset.py:364.

### Route-to API
- Move `insert_growth_event` behind `beckman.record_growth_event()` (thin)
- Add `beckman.supersede_growth_event(id)` for the 5 raw UPDATEs

**Effort: S.** ~18 mechanical call-site swaps + 2 new thin methods.

---

## 4. `founder_actions` — owner: **src/founder_actions** (already de-facto owner — handoff overstated violation)

Matrix correction: this table is ALREADY consolidated. All inserts (37 call sites across src/app, src/ops, beckman posthooks/z6_admission, mr_roboto executors) route through `founder_actions.create()` (src/founder_actions/__init__.py:207). Status changes route through `update_status`/`resolve`.

**Only violation:** `src/app/attention_budget.py:368` — raw UPDATE `defer_until`.

### Route-to API
- Add `founder_actions.defer(action_id, until)` → swap one call site.
- (Optional, packageability) founder_actions lives in src/, not packages/ — relocation is Step-2 territory, not write-consolidation.

**Effort: XS.**

---

## 5. COMMS family — owner: **decision needed**, recommend **mr_roboto for product-comms; CRM/email stay src/app**

27 tables in family, NO db.py helpers — all writers are raw SQL. The matrix shows it's not one blob; three coherent sub-families:

### 5a. Product-comms (automation) — already mr_roboto-dominant
incidents, status_updates, crisis_events, marketing_freeze, changelog_entries, press_kits, external_reviews, outreach_{sends,warmup,pauses,prospects}, mentions, mention_monitors, external_comms_log.

src/app intrusions to remove (the actual "no single owner" conflicts):
- telegram_bot.py:5759 — UPDATE incidents (manual resolve) → route via mr_roboto incident op
- telegram_bot.py:3924,3944 — mention_monitors add/disable → mr_roboto API
- telegram_bot.py:11554 — outreach_prospects approve → mr_roboto API
- (telegram_bot.py:5709/5710/5750 already dispatch via mr_roboto crisis ops — pattern to copy)

### 5b. CRM + meetings + interviews — src/app-exclusive already
relationships, interactions, consent_records (src/app/crm.py) · meetings (meetings.py + jobs/meeting_brief_dispatch.py:119) · interview_notes (interview.py). Single-owner ✅ — but owner is app, not package → blocks product.db materialization until relocated (Step 2).

### 5c. Email lifecycle — src/app-exclusive already
email_templates/sends/preferences (lifecycle_email.py + jobs/lifecycle_email_send.py) · email_events (src/integrations/email/service.py). Single-owner ✅.

### Cross-family overlap (the one genuine tangle)
press_kit_quotes: written by src/app interview.py:468 + jobs/quote_harvest.py:139, while press_kits written by mr_roboto press_kit_publish.py:69,75. Quotes feed kits across the boundary. Owner decision: either quotes go to mr_roboto API, or quotes stay app-side and press_kit_assemble reads them (current reality — reads are fine, only writes need one owner). Recommend: leave quotes app-owned, kits mr_roboto-owned; revisit at product.db time.

Dead tables found: `launches`, `email_sequences` — created, zero writers. Candidates for deletion sweep (verify reads first per [[feedback_zero_traffic_not_dead]]).

**Effort: M.** 4 telegram_bot intrusions (S) + choosing/codifying the sub-family owners (decision) + optional dead-table sweep.

---

## Execution order (each ships independently, worktree per chunk)

1. **api.py:129 mission-INSERT bug fix** — XS, standalone, do first
2. **founder_actions.defer()** — XS
3. **growth_events consolidation** — S
4. ~~comms: evict 4 telegram_bot write-intrusions~~ — **DEFERRED** (subsystem unvalidated, see Decisions)
5. **missions: `update_mission_fields` + migrate ~30 sites incl. engine** — M
6. **tasks: beckman write API + migrate telegram_bot/core/workflows-engine** — L (split into: UI ops → recovery/reset ops → engine interface — engine ruled EXTERNAL, full routing required)

## Decisions (user, 2026-06-11)
- **workflows/engine = EXTERNAL client of beckman.** Two separate packages, clear interface, totally independent long-term. Engine's direct task writes must route through Beckman API. This grows the tasks chunk: engine's write surface (`add_task` runner.py:229,473 · `update_task` runner.py:201, hooks.py:1670,2261,2297, task_refresh.py:74 · `update_task_by_context_field` hooks.py:2022 · `add_subtasks_atomically` hooks.py:2133 · `propagate_skips` hooks.py:2030 · `add_mission` runner.py:388 · review_density.py:158 mission write) becomes explicit Beckman API methods — this enumerates the engine↔beckman interface, which is the long-term goal anyway.
- **Comms: DEFERRED.** User: emails/interviews/CRM/comms all fresh-built (Z6 + CPS sprints, 2026-05), zero missions have exercised them yet. Don't consolidate ownership of unvalidated subsystem — first real usage may reshape tables/flows. Revisit after first live mission runs through comms. The 4 telegram_bot write-intrusions stay in backlog (cheap, anytime). `launches`/`email_sequences` zero-writer tables = fresh-unfinished, NOT dead — no deletion sweep ([[feedback_zero_traffic_not_dead]]).

## Corrections to prior handoff
- tasks writers = 7 modules not 5; missions = 6 not 3 (mr_roboto missions reach is ×10 files, all raw SQL).
- founder_actions was NOT a real violation — already single-owner, 1 stray UPDATE.
- comms "no single owner" is real but narrow: 4 telegram_bot write sites + press_kit_quotes split. CRM/email sub-families already clean.
