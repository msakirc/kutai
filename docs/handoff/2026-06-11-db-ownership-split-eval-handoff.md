# Handoff: DB ownership / per-package split — evaluation (2026-06-11)

**Status:** evaluation only, NOTHING implemented. No code touched. This is a research/decision handoff.

## The ask (user's framing)

DB usage is sprawl: `packages/*` reach into `src/infra/db.py` directly to read/write. User wants either (a) one subpackage that owns all DB ops, or (b) each package owns its own tables/DB so packages can **materialize independently** (ship as standalone repos/services). User aiming **moonshot** — full per-package independence. Golden rules apply: no dumbness affirmation, no band-aids, tell them if wrong.

## What the research established (4 scout passes over the tree)

### Scale
- `src/infra/db.py` = **8,483 LOC**, **~97 tables** (not 66 — CLAUDE.md is stale), **161 functions**, **94 importing modules**. One shared aiosqlite connection, **WAL**, mission-sharded tx locks (`_get_tx_lock` / `_get_combined_lock` at ~db.py:78–150; the combined lock is defined but **never called** — latent gap).
- Good news: no package opens its own aiosqlite connection. All go through db.py helpers. ~20–30 `get_db()`+raw-SQL escapes (the layering leaks).
- Precedent: `src/shopping/memory/shopping_memory.db` is ALREADY a separate file — works because shopping has **zero FK into missions/tasks**.

### The schema shape: hub-and-spoke, not a flat mess
Every cross-domain FK points into **one place** — Beckman's `missions`/`tasks` (the "spine"). No satellite→satellite coupling. Tiers (verified across all 97):
- **SPINE (3):** missions, tasks, continuations.
- **SPINE-BOUND (17):** hard FK into spine OR in the mission rollback cascade. Cannot separate without losing integrity/atomicity. Incl: artifact_provenance, action_confirmations, mission_budget_alerts, mission_events, mission_pacing_snapshots, mission_tradeoff_prompts, conversations(task_id), hypotheses, experiment_variants, growth_events, mission_lifecycle_log, regen_log, mission_artifacts_index, founder_actions, integration_mappings, action_cooldowns, perf_baselines.
- **SOFT-LEAF (~55):** `mission_id`/`product_id` soft-scoping only, no integrity need. Movable, loses app-FK checks. **~40 of these are `product_id`-scoped CPS/comms/CRM tables** (email_*, press_*, incidents, crisis_events, outreach_*, relationships, meetings, mentions, launches, interactions) — they do NOT reference the spine → coherent separable subsystem.
- **LEAF (22):** zero spine coupling, free to move. Incl: models, providers, model_stats, step_token_stats, kdv_state, api_*, web_source_quality, free_api_registry, smart_search_log, skills, todo_items, scheduled_tasks, credentials, user_preferences, prior_art_cache, mission_lessons, confidence_reliability_scores.

### Cross-domain entanglement (the only things that block a table split)
- **Atomic cross-domain transactions — only 3, and 2 aren't even atomic:**
  1. `add_mission` (db.py:4165) → also seeds cost_budgets (`ensure_mission_cost_row` ~4212). Separate commits → just sequencing.
  2. `record_call_cost` (db.py:7577) → writes model_call_tokens + reads tasks + writes cost_budgets. Separate commits.
  3. `restore_mission_db_rows` (db.py:8084) → **the one genuinely atomic cross-domain tx** (green-tag rollback, under one lock). Spans spine + artifacts. **BUG/feature: it OMITS cost_budgets** (`MISSION_SCOPED_TABLES` ~db.py:7962) → cost is ALREADY non-atomic with rollback.
- **Cross-domain JOINs (read-only, ~4, not blockers):** `estimate_task_cost` (db.py:7810, tokens×tasks), `get_artifact_provenance` (db.py:7061, 3 domains), `estimate_conversation_cost` (db.py:6212, conversations×tasks).

### Hard SQLite facts that kill naive multi-file split (under WAL)
1. **FK constraints cannot cross database files** — split spine-bound tables out → all FKs become unenforceable, app must enforce (orphan sweeps).
2. **Cross-file atomicity dies under WAL** — "if any DB in a multi-file tx is WAL, commit is atomic per-file but NOT across the set." KutAI is WAL. So multi-file forces a choice: keep WAL (lose cross-domain atomicity → `restore_mission_db_rows` non-atomic, half-restored mission on crash) OR drop WAL (readers block writers, worse concurrency). Single file gives both.
3. **JOINs need ATTACH** — re-couples the "independent" files at query time anyway.
- A "helper library for heavy lifting" is GOOD for boilerplate/ports (kills package→src reach) but **cannot unlock multi-file** — these are engine-level, not library-level. Independence of *files* ≠ independence of *data*; coupling lives in the relational model (FKs into spine), not the storage.

## The decisive finding (the access/ownership matrix)

**The binding constraint is NOT the file boundary — it's write-ownership scatter.** Can't give a package its own DB while N modules write its tables.

**Ownership violations (must fix BEFORE any split):**
- `tasks` — 5 writers: general_beckman, src/app, src/core, src/founder_actions, src/infra
- `missions` — 3 writers: beckman, mr_roboto (emit_preview_url.py), src/infra
- `growth_events` — beckman + mr_roboto + src/app
- comms (incidents/changelog_entries/press_kits) — **mr_roboto AND src/app, no single owner**
- `founder_actions` — app + src/founder_actions + infra

**Spine is a universal read-hub:** `missions` read by 10 consumers, `tasks` by 8. No split removes this — everyone needs read-access to the spine. That's the one unavoidable API regardless of decision.

**Map corrections from the matrix:**
- ARTIFACTS (artifact_provenance, mission_green_tags, workspace_snapshots, file_locks) owned by **src/infra, NOT a package** → can't materialize; stays app infra.
- COMMS/CPS has **no single owner** (mr_roboto + src/app split it) → owner must be chosen.
- LEDGER (cost_budgets, model_call_tokens) is **lightly read** — only 3 cross-boundary edges: general_beckman briefing (btable_rollup.py:47, briefing_compose.py:62), src/app daily_briefing.py:119. Cost already rollback-excluded → cheapest spine-referencing split.

Key cross-boundary read refs: missions reads — mr_roboto/__init__.py:243, src/app/api.py:150, src/app/telegram_bot.py:2516, src/workflows/engine/hooks.py:1234, src/context/assembler.py:439. tasks reads — mr_roboto/audit_log.py:341, src/core/startup_recovery.py:22. model_pick_log — fatih_hoca/counterfactual.py:35, mr_roboto/executors/analytics_digest.py:210, src/app/telegram_bot.py:2807. skills — src/memory/skills.py:601, src/memory/self_improvement.py:170, yalayut/migration.py:56.

## Conclusion reached (the moonshot answer)

Per-package independence is **reachable for ~4 clusters cleanly; the spine never** (it's the shared core — that's correct, not a failure; mission rollback needs cross-table atomicity that dies the moment those tables cross a file boundary under WAL).

**Gated by ORDERING, not by the boundary question:**
- **Step 1 (the real work, valuable either way): consolidate writes to one owner per table.** Funnel `tasks`/`missions` writes through Beckman's API; pick a single comms owner; stop src/app raw-SQL on the spine. This IS the already-stated-but-unfinished `beckman.enqueue()` architecture + the ports/adapters cleanup. **It kills the `package→src` reach the user started with — regardless of whether a single file is ever split.**
- **Step 2 (mechanical, lazy, reversible): split the file per-package**, only when a package actually ships alone. Falls out of Step 1.

**Don't decide the file boundary yet** — it's a consequence of Step 1. (Earlier the assistant offered a 3-way file-boundary fork: tight core / max independence / max safety. User correctly deferred it pending the access matrix; the matrix then showed the fork is premature.)

**Split-readiness after Step 1 (cleanest first):** yalayut ✅ now · shopping ✅ already · fatih→registry.db ✅ near (single owner) · ledger.db ✅ cheap (zero atomicity loss, cost rollback-excluded) · beckman/spine ❌ 5 writers, universal hub · comms/product ❌ no single owner · artifacts + app tables ❌ not packages.

Recommended physical end-state IF user proceeds (per-coupling = per-package, they coincide): `core.db` (beckman spine + rollback cascade + growth + artifacts, stays atomic/WAL) · `registry.db` (fatih) · `ledger.db` (cost/tokens) · `product.db` (comms, after owner chosen) · `yalayut.db` · `shopping.db` (exists). ~5–6 files = deploy units, NOT 97-table god-file, NOT per-table sprawl.

## NEXT STEP for new session

Scope **Step 1 — write-consolidation work-list per violated table**: for each of {tasks, missions, growth_events, comms tables, founder_actions}, list every current writer (file:line) → which owner API it should route through → effort. Start with the spine (tasks/missions). This is the moonshot-enabling work and is useful even if no file is ever split.

Then (optional, later) re-pose the file-boundary fork once writes are single-owner.

## Constraints / gotchas to respect
- WAL mode is load-bearing (concurrent readers + non-blocking single writer). Do not casually drop it.
- `restore_mission_db_rows` atomicity is a SAFETY feature (green-tag rollback). Anything that crosses a file boundary with a rollback-cascade table breaks it.
- `_get_combined_lock` (db.py ~145) exists but is never called — `record_call_cost` writes 2 domains without atomic protection. Latent.
- Don't trust `src/core/router.py` scoring copy (stale/dead). Live selection is fatih_hoca.
- Concurrent agent sessions on `main` have crossed prior work — use a git worktree for any implementation.
