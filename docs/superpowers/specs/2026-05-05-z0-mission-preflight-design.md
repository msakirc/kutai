# Z0 — Mission Preflight (design)

**Date:** 2026-05-05
**Source frame:** [docs/i2p-evolution/z0-mission-preflight.md](../../i2p-evolution/z0-mission-preflight.md)
**Audit:** Z0 deep-state audit (this session) — gaps A–L mapped to existing primitives.

## Scope

After scope-reduction passes with the founder, Z0 collapses from "founder + system contract with 12 gaps" to **5 concrete items** that are not already covered by the existing routing layer, i2p Phase 0, or Z6 (real-world bridge). Everything else is dropped, deferred, or absorbed elsewhere.

### In-scope (this spec)

1. **Per-mission $ ceiling** — `cost_ceiling_usd` declared at mission start; selection-time filter in Fatih Hoca + admission backstop in Beckman.
2. **Per-mission Telegram forum topic** — `bot.create_forum_topic()` at mission start, pinned status message, `message_thread_id` on every mission post.
3. **Auto-pause triggers** — $ ceiling reached, repeated DLQ cascade (≥3), collision-guard fired, idle on `none`-locked steps.
4. **Lifecycle state polish** — `lifecycle_state ∈ {active, paused, killed, completed}`, audit log, refined `/pause`/`/resume`, new `/kill_mission` with snapshot.
5. **Reversibility tags + collision guards** — `packages/safety_guard/`: workflow steps carry `reversibility: full|partial|none` + `locked: bool`; executor may escalate (never downgrade); collision guards (no force-push, no parallel-agent overwrite, no shared-history rewrite, no destructive shell outside workspace, no destructive shared-DB DDL).

### Out of scope (rationale)

| Item | Reason |
|---|---|
| Founder profile (name, voice, prior products) | YAGNI for sakir-only-now installs; voice matters once missions generate public-facing content (none yet); skill system already captures patterns |
| Hours estimate | No enforcement (pausing on time is illogical — pausing doesn't save time); notify-only, but the notify alone isn't worth a Q in the wizard |
| Ambition tier matrix | Worth nothing without phase-level tier-eligibility annotations on workflows; that's a Z2 effort. Reversibility-tag system already gates dangerous actions per-step. **Defer until first mission hits the wall.** |
| North-star metric | Z9 (growth) need; nothing reads it today |
| Compliance fingerprint | Folds into i2p Phase 0 (idea-level question) |
| Target platform (web/mobile) | Folds into i2p Phase 0 |
| Founder readiness checklist (legal entity / bank / vendor accounts) | **Lazy capture in Z6** — first action that needs it prompts; reversibility-tag executor is the trigger |
| Vault initial provisioning | Primitive already exists (`src/security/credential_store.py`); Z6 wraps it with founder-context + lazy-capture flow when wiring vendor adapters. Z0 only fixes key derivation: machine-local `.env` secret + Telegram identity, no passphrase ceremony, disk-loss = data-loss (honest) |
| Founder-side mission preflight wizard with 6+ Qs | Reduced to 1 optional Q ($ ceiling) — i2p Phase 0 already covers idea/scope/compliance Qs |

## Single-founder-per-install assumption

KutAI is a personal agent. Each install has exactly one founder, identified by `TELEGRAM_USER_ID` allowlist (existing). No multi-tenancy in schema. No profile portability. No multi-binding. No passphrase recovery ceremony — disk lost = data lost (same as any local app). Vault key = machine-local `.env` secret + Telegram identity hash.

## Architecture overview

No new umbrella package. Z0 = thin slices across existing modules + one small new module for safety guards.

```
Mission start (telegram_bot.py: cmd_mission / shop / classifier)
    │
    ├── DB: insert missions row WITH cost_ceiling_usd, lifecycle_state='active'
    ├── Telegram: create forum topic (message_thread_id), pin status message
    │
    ▼
workflow_engine.start(mission_id)   ← unchanged
    expands phases → enqueues tasks
    │
    ▼
Per-task dispatch (general_beckman + dispatcher)
    │
    ├── BEFORE selection: pass remaining_budget_usd to fatih_hoca.select()
    │     → Fatih Hoca filters cost > remaining models
    │     → empty pool → SelectionFailure(reason='budget') → mission paused
    ├── BEFORE dispatch: check missions.lifecycle_state (paused/killed → skip)
    ├── BACKSTOP: check spent + estimate > ceiling (defense-in-depth)
    │
    ▼
Action execution (workflow step → executor)
    │
    ├── safety_guard.executor_hook: resolve reversibility tag
    ├── safety_guard.collision: guards (force-push, parallel-overwrite, etc.)
    ├── If `none`+locked AND founder idle → wait_human (no idle bypass)
    │
    ▼
On completion: increment missions.spent_usd
    50 / 75 / 90% threshold notify in mission thread (once each)
    100% breach → lifecycle: active → paused (defense backstop, normally caught at selection)
```

### Modules touched

| Module | Change |
|---|---|
| `src/infra/db.py` | schema migration: 4 new `missions` columns + `mission_lifecycle_log` table |
| `src/app/telegram_bot.py` | forum-topic provisioning at mission start; pinned status updater; `/kill_mission`; lifecycle polish on `/pause`/`/resume`; inline button callbacks for Approve/Reject/Pause/Kill |
| `packages/general_beckman/` | admission gate reads `lifecycle_state`; passes `remaining_budget_usd` to selection; ceiling backstop; auto-pause-trigger emitter |
| `packages/fatih_hoca/` | `select()` accepts `remaining_budget_usd: float \| None`; pre-score filter excludes `cost > remaining`; `SelectionFailure(reason='budget')` failure mode |
| `src/workflows/i2p/i2p_v3.json` (and shopping JSON) | add `reversibility` field per step (default `full`, locked on dangerous steps) |
| `src/workflows/engine/` | pre-action hook → `safety_guard.executor_hook` |
| **NEW** `packages/safety_guard/` | ~150–250 LOC: tag resolver, collision guards, executor hook |

## Components (per-item detail)

### 1. Per-mission $ ceiling

**Schema**
```sql
ALTER TABLE missions ADD COLUMN cost_ceiling_usd REAL;        -- NULL = unlimited
ALTER TABLE missions ADD COLUMN spent_usd REAL DEFAULT 0;
```

**Capture.** Single optional Q during `cmd_mission`: "Cost ceiling for this mission ($)? Reply with a number, or 'none' for unlimited." Default behavior if user skips: `NULL` (unlimited).

**Selection-time filter (primary).** `general_beckman.next_task()` calls `fatih_hoca.select(task, remaining_budget_usd=ceiling - spent)`. Fatih Hoca filters candidates with `estimated_cost_usd > remaining_budget` BEFORE scoring. Ceiling=0 + remaining=0 → only $0 models (local) eligible. Cloud never picked.

**Selection failure path.** When task profile mandates cloud (>local context, vision, etc.) and no model fits remaining budget: `SelectionFailure(reason='budget')`. Beckman transitions mission to `paused` with reason `no_model_fits_budget`, posts to thread with Resume / Increase ceiling / Kill buttons.

**Backstop (post-dispatch).** Beckman admission keeps `spent + estimate > ceiling → pause` as defense-in-depth (catches estimate drift, race conditions where in-flight tasks aggregate past ceiling). In-flight tracker prevents race overshoot: admission considers `spent + sum(in_flight_estimates) + new_estimate`.

**Tracking.** `general_beckman.on_task_finished()` increments `missions.spent_usd += task.cost_usd` (cost already tracked per-task today).

**Threshold notifies.** 50% / 75% / 90% fire once each via `mr_roboto.notify_user` to mission thread. 100% triggers pause + ask (Resume / Increase / Kill).

### 2. Per-mission Telegram forum topic

**Schema**
```sql
ALTER TABLE missions ADD COLUMN message_thread_id INTEGER;
```

**Provisioning.** New `TelegramInterface.provision_mission_thread(mission_id, title)` calls `bot.create_forum_topic(chat_id, name=f"#{mission_id} {title}")`, returns `message_thread_id`. Stores in `missions.message_thread_id`. Posts initial pinned status message.

**Pinned status.** Format:
```
Mission #42 — "Build recipe app"
Status: active
Spent: $0.42 / $5.00 (8.4%)
Tasks: 3 done, 2 in flight, 17 queued
[Pause] [Kill]   ← inline buttons
```

Updated by:
- Lifecycle event (state change) → immediate edit
- Threshold notify (50/75/90%) → immediate edit
- Periodic background refresh every N=10 task completions

**All mission-context posts** use `message_thread_id` so they land in the topic, not main chat.

**Confirmation buttons.** Every confirmation post (clarify, none-reversibility step) gets inline `Approve` / `Reject` / `Pause` / `Kill` buttons. Reactions used for ack-only (`👍` = "seen, keep going" — no state change).

**Fallback (no forum perm / forum disabled).** Log warning, fall back to main chat with `[mission #42]` tag prefix on every msg. Mission proceeds. Retry topic creation once on first post; persistent fail → tag-prefix mode for that mission.

### 3. Auto-pause triggers

| Trigger | Condition | Source |
|---|---|---|
| Budget no-fit | Fatih Hoca returns `SelectionFailure(reason='budget')` | beckman.next_task() |
| Ceiling backstop | `spent + estimate > ceiling` | beckman admission |
| DLQ cascade | ≥3 consecutive task failures (terminal, not retry-eligible) | beckman.on_task_finished() |
| Collision guard fired | Action blocked by safety_guard | safety_guard.executor_hook |
| Idle on `none`-locked | `waiting_human > 24h` on a `reversibility=none` step | watchdog (existing scheduled job) |

**Triggers fire as `LifecycleEvent` posted to a single internal queue.** telegram_bot subscribes, handles state transition + thread notify uniformly.

**Multiple triggers same tick:** single pause event, reasons concatenated in `mission_lifecycle_log.reason`.

**Idempotent:** pause-while-paused = no-op.

### 4. Lifecycle states

**Schema**
```sql
ALTER TABLE missions ADD COLUMN lifecycle_state TEXT DEFAULT 'active';

CREATE TABLE mission_lifecycle_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id INTEGER NOT NULL,
    from_state TEXT,
    to_state TEXT NOT NULL,
    reason TEXT,
    triggered_by TEXT,           -- 'founder' | 'auto:ceiling' | 'auto:dlq' | 'auto:collision' | 'auto:idle'
    ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (mission_id) REFERENCES missions(id)
);
```

**States:** `active` (initial) → `paused` ↔ `active` → `completed` | `killed`.
- `killed` is terminal; resume rejected.
- `completed` is terminal; resume rejected.
- `paused` only transitions to `active` (resume) or `killed`.

**Beckman admission gate:** if `lifecycle_state != 'active'`, `next_task()` returns `None` (no dispatch). In-flight tasks finish their current step naturally; no torn state.

**Concurrent transitions:** every state UPDATE uses `WHERE lifecycle_state = ?` so only one wins. Loser silently no-ops.

**Commands**
- `/pause_mission <id>` — refined existing `/pause`. Sets state, finishes in-flight tasks, holds queue.
- `/resume_mission <id>` — refined existing `/resume`. State `paused → active`. Beckman picks up.
- `/kill_mission <id>` — NEW. Graceful termination + state snapshot:
  - Snapshot artifact: `mission_kill_<id>.json` with current artifacts + open tasks + memory pointer.
  - State `* → killed` (terminal).
  - Mid-task: cancellation signal sent, current LLM call allowed to finish; state flips at task boundary.
  - Snapshot write fails → log error, still mark killed (data loss preferable to retry-loop).

**Resume after kill rejected.** `/resume_mission` on a killed mission returns "mission killed; cannot resume."

### 5. Reversibility tags + collision guards (`packages/safety_guard/`)

**Module structure**
```
packages/safety_guard/
├── src/safety_guard/
│   ├── __init__.py        # public API: executor_hook
│   ├── tags.py            # Reversibility enum, tag resolver
│   ├── collision.py       # guards (force-push, file-lock, shell, etc.)
│   └── executor_hook.py   # called by workflow_engine pre-action
└── tests/
```

**Tag resolution (`tags.py`)**
```python
class Reversibility(Enum):
    FULL = ("full", 0)        # value, strictness rank — higher = stricter
    PARTIAL = ("partial", 1)
    NONE = ("none", 2)

    @property
    def rank(self) -> int:
        return self.value[1]

    @classmethod
    def from_str(cls, s: str) -> "Reversibility":
        return next(r for r in cls if r.value[0] == s)

# Static (workflow JSON):
#   step.reversibility = "none"
#   step.locked = true
#
# Dynamic (executor):
#   may pass `runtime_override: Reversibility` to escalate stricter
#   if locked=True, override ignored (workflow author wins)
#   downgrade (lower rank than static) always rejected

def resolve(step: dict, runtime_override: Reversibility | None) -> Reversibility:
    static = Reversibility.from_str(step.get("reversibility", "full"))
    locked = step.get("locked", False)
    if locked or runtime_override is None:
        return static
    if runtime_override.rank < static.rank:  # downgrade attempt
        log.warn("downgrade rejected"); return static
    return runtime_override
```

**Collision guards (`collision.py`)** — fire regardless of tag (defense-in-depth; small LLMs can forget safety):

| Guard | Detects | Action |
|---|---|---|
| `assert_no_force_push(cmd)` | `git push -f`, `--force`, `--force-with-lease` | Block (always); per-mission allowlist for `--force-with-lease` on personal feature branches via `/safety allow <pattern>` |
| `assert_no_shared_history_rewrite(repo, branch)` | `git rebase`/`reset --hard` on shared branches (main, master, develop, release/*) | Block |
| `assert_no_parallel_agent_overwrite(file_path)` | File in another active agent's lock set | Block (file locks via existing workspace machinery if present, else simple in-memory mutex registry) |
| `assert_shell_in_workspace(cmd)` | Shell cmd targets paths outside `WORKSPACE_ROOT` | Block; per-mission allow via `/safety allow` |
| `assert_no_destructive_shared_db(query)` | `DROP TABLE` / `TRUNCATE` on non-mission-scoped tables | Block |

**Hardcoded blocklist (always wins, allowlist cannot override):**
- `git push --force` to `main`/`master`
- `stripe.charges.create` (production keys)
- `vercel deploy --prod` (without explicit founder approval)
- `aws s3 rm` outside workspace
- Any `rm -rf /` or absolute-path destructive shell

**Executor hook (`executor_hook.py`)**
```python
async def pre_action(step: dict, action: dict) -> Decision:
    # Decision: Allow | WaitForFounder(reason) | Block(reason)
    tag = resolve(step, action.get("runtime_reversibility"))

    # Always run collision guards
    for guard in ALL_GUARDS:
        if not guard(action):
            return Block(reason=f"collision: {guard.__name__}")

    # Hard blocklist
    if matches_blocklist(action):
        return Block(reason="blocklist")

    # Reversibility-driven flow
    if tag == Reversibility.NONE:
        if not founder_recently_active() or step.get("locked"):
            return WaitForFounder(reason="non-reversible step")

    return Allow
```

**Workflow JSON tagging.** Workflow-author marks per step:
```json
{ "id": "deploy_prod", "agent": "mechanical", "reversibility": "none", "locked": true, ... }
{ "id": "post_to_twitter", "reversibility": "none", "locked": true, ... }
{ "id": "stripe_test_charge", "reversibility": "none", "locked": false, ... }    // executor may relax to partial in test mode (won't because downgrade rejected; but the field is there for future)
{ "id": "edit_local_file", "reversibility": "full", ... }                         // default; usually omitted
```

Default if absent: `full`. Workflow lint warns at load time on missing tags for steps with executor patterns matching dangerous ops.

## Data flow (worked example)

```
Founder taps "/mission" + types "Build recipe app". Adds "$2 ceiling".
  ↓
telegram_bot.cmd_mission():
  classify → workflow=i2p_v3
  insert missions(title='Build recipe app', cost_ceiling_usd=2.00, lifecycle_state='active')
  thread_id = bot.create_forum_topic("#42 Build recipe app")
  update missions.message_thread_id = thread_id
  pin status: "queued, $0 / $2"
  ↓
workflow_engine.start(42)
  expands i2p_v3 Phase 0 (idea capture) → enqueues 6 tasks
  ↓
beckman.next_task() picks task #1 (mechanical, free)
  state == 'active' ✓
  remaining = 2.00 - 0.00 = 2.00
  fatih_hoca.select(task, remaining_budget_usd=2.00) → gpt-oss-7b ($0)
  dispatch
  ↓
Task #1 executes (mechanical: read existing repo). on_task_finished: cost=0
  spent_usd: 0 → 0
  ↓
... a few tasks later, task #5 = LLM ideation, task profile mandates cloud
  remaining = 2.00 - 0.20 = 1.80
  fatih_hoca.select(..., remaining_budget_usd=1.80) → claude-sonnet-4-6 estimate $0.50, fits → picked
  dispatch
  ↓
Task #5 done, cost actual = $0.55 (drift)
  spent_usd: 0.20 → 0.75
  threshold check: 0.75 / 2.00 = 37.5% — no notify yet
  ↓
... eventually spent reaches $1.00 (50%)
  on_task_finished triggers threshold notify → thread:
    "📊 Mission #42 spent $1.00 / $2.00 (50%)"
  ↓
Task #18 needs cloud, fatih_hoca picks claude estimate $0.30, remaining = 0.50
  dispatch ok
  cost actual = $0.40 — overshoots; spent = 1.00 + 0.40 = 1.40
  ↓
Task #19 mandates cloud (vision call), remaining = 0.60
  fatih_hoca.select(remaining=0.60) → only model that fits is claude-haiku $0.10
  picked, dispatched
  ↓
Task #20 mandates cloud + needs >32k context (only claude-sonnet $0.45)
  fatih_hoca filter: $0.45 > $0.50 remaining? No, fits. Dispatch.
  cost actual $0.55 — spent = 1.40 + 0.55 = 1.95 (97.5%)
  threshold notify 90% fired earlier at 1.80
  ↓
Task #21 mandates cloud, remaining = 0.05
  fatih_hoca.select(remaining=0.05) → empty pool (no cloud fits, local lacks capability)
  SelectionFailure(reason='budget')
  beckman: lifecycle 'active' → 'paused', reason='no_model_fits_budget'
  log to mission_lifecycle_log
  thread notify with [Resume w/ +$2] [Resume unlimited] [Kill] buttons
  pinned status updated to "paused — budget exhausted ($1.95 / $2.00)"
  ↓
Founder taps [Resume w/ +$2]:
  cost_ceiling_usd: 2.00 → 4.00
  lifecycle: paused → active
  beckman picks up, mission continues
```

## Error handling + edge cases

**Cost ceiling**
- Ceiling = NULL → no enforcement, no threshold notifies.
- Ceiling = 0.00 → enforced strict (≠ NULL). Local-only models eligible. First cloud-mandated task pauses with `no_model_fits_budget`.
- Estimate missing → `estimated_cost_usd = 0` (best-effort), reconcile after task. Don't block.
- Reconciliation overshoot → log, mark mission paused if past ceiling, don't claw back.
- Race (two parallel dispatches both fit individually, combined exceeds): in-flight tracker = `spent + sum(in_flight_estimates) + new_estimate` in admission.

**Forum topic**
- Bot lacks `manage_topics` → fallback to main chat with `[mission #N]` prefix.
- Topic creation fails mid-mission start → `message_thread_id=NULL`, retry once on first post; persistent fail → tag-prefix mode.
- Forum disabled (regular group) → tag-prefix mode.
- Telegram down at start → defer thread creation to first post (lazy).

**Lifecycle**
- Concurrent state transitions → row-level guard `WHERE lifecycle_state = ?`. Loser logs nothing.
- `/kill_mission` mid-task → cancellation signal, current LLM call finishes, state flips at task boundary.
- Snapshot write fails on kill → log error, still mark killed.
- Resume after kill or completed → rejected with explanatory message.

**Reversibility / collision**
- Tag missing on step → default `full`. Workflow lint warns at load.
- Locked tag + executor downgrade attempt → log violation, ignored.
- Collision guard false positive (e.g. `--force-with-lease` on personal branch) → per-mission allowlist via `/safety allow <pattern>`. Stored in `mission.context`. Hardcoded blocklist always wins.
- Idle on `none`+locked → `waiting_human` indefinitely. Watchdog reminds via thread post every 24h. Mission stays `active` (other tasks run).

**Migration / back-compat**
- All new columns nullable (or default). Pre-existing missions: `lifecycle_state='active'`, `spent_usd=0`, no ceiling, no thread.
- Beckman pickup of pre-Z0 mission → works unchanged (NULL ceiling = no enforcement, NULL thread = main chat).

## Testing

### Unit (TDD order)

1. **DB migration** (`tests/infra/test_db_migration_z0.py`) — adds 4 columns + lifecycle log table; idempotent re-run; pre-existing rows get sane defaults.
2. **safety_guard tags** (`packages/safety_guard/tests/test_tags.py`) — locked-prevents-downgrade; executor escalation accepted; missing tag = `full`.
3. **safety_guard collision** (`packages/safety_guard/tests/test_collision.py`) — force-push variants detected; shared-branch rebase blocked; shell-allowlist enforced; destructive shared DB blocked; per-pattern allowlist override; hardcoded blocklist wins over allowlist.
4. **safety_guard executor_hook** (`packages/safety_guard/tests/test_executor_hook.py`) — Allow/WaitForFounder/Block returns; idle behavior on `none`-locked.
5. **Beckman admission lifecycle** (`packages/general_beckman/tests/test_admission_lifecycle.py`) — paused/killed states skip dispatch; active dispatches normally.
6. **Beckman admission ceiling** (`packages/general_beckman/tests/test_admission_ceiling.py`) — NULL = no enforcement; 0.00 strict (only $0 models); in-flight tracker prevents race; backstop catches estimate drift.
7. **Fatih Hoca budget filter** (`packages/fatih_hoca/tests/test_budget_filter.py`) — `select()` honors `remaining_budget_usd`; SelectionFailure(reason='budget') on empty pool.
8. **Threshold notifies** (`packages/general_beckman/tests/test_threshold_notifies.py`) — 50/75/90% fire once each.
9. **telegram_bot lifecycle** (`tests/app/test_telegram_bot_lifecycle.py`) — provision_mission_thread happy + missing perm fallback; /pause_mission, /kill_mission, /resume_mission state transitions; resume-after-kill rejected; inline-button callbacks.

### Integration (`tests/integration/`, no LLM, `pytest -m "not llm"`)

- `test_z0_mission_lifecycle.py` — spawn mission, hit ceiling mid-pipeline → auto-pause → resume → complete.
- `test_z0_collision_block.py` — workflow step attempts `git push --force` → blocked, logged, mission continues.
- `test_z0_kill_snapshot.py` — `/kill_mission` mid-flight produces snapshot artifact + state=killed.

### Manual smoke (single-Telegram session, expected from founder)

- Real i2p mission with `$0.50` ceiling → confirm pause + thread message + Resume button works.
- Real i2p mission with `$0.00` ceiling → confirm only local models picked; cloud-mandated step pauses with `no_model_fits_budget`. **Note:** local-LLM tasks have `cost_usd = 0`, so `0.00` ceiling + local-only mission may run forever without tripping. To force pause flow with local-only, use `/pause_mission` directly OR set ceiling to `0.01` and ensure mission picks a cloud model at least once.
- Force-push attempt in test workflow step → confirm `safety_guard` blocks, founder notified.
- `/kill_mission` mid-flight → forum topic shows final pinned snapshot link + state=killed.

## Open questions

- **Per-mission allowlist UX.** `/safety allow <pattern>` is sketched but not detailed. Proposed: command takes pattern, stores in `mission.context.safety_allowlist`, loaded on safety_guard hook. Need to confirm scope (per-mission only, never global; expires when mission completes).
- **Threshold notify de-dup across paused/resumed cycles.** If mission pauses at 90% then resumes, do we re-notify at 90%? Proposed: notifies are per-mission-lifetime (already-fired set persists across pause/resume), not per-segment. Needs confirmation.
- **In-flight cost estimate source.** `task.estimated_cost_usd` field: who populates it? Proposed: Fatih Hoca returns `(picked_model, estimated_cost_usd)` from `select()`; admission caches it on the task row before dispatch. This means Fatih Hoca's selection function gains a return-shape change — non-breaking if we make `estimated_cost_usd` optional with default 0.
- **DLQ cascade definition.** "≥3 consecutive task failures" — does that mean 3 ANY failures or 3 in the same phase? Proposed: 3 consecutive across the mission, reset on any successful completion. Simple, errs on safety.

## Acceptance criteria

- [ ] DB migration applies cleanly on existing kutay.db; pre-existing missions readable and dispatchable.
- [ ] New `cost_ceiling_usd`, `spent_usd`, `message_thread_id`, `lifecycle_state` columns populated on new missions.
- [ ] `mission_lifecycle_log` records every state transition with reason + trigger source.
- [ ] Fatih Hoca `select()` accepts `remaining_budget_usd`; ceiling=0 missions never pick cloud models.
- [ ] When no model fits remaining budget, mission auto-pauses with `no_model_fits_budget`.
- [ ] 50/75/90% thresholds notify exactly once each per mission lifetime.
- [ ] Forum topic provisioned at mission start; pinned status reflects spent/ceiling/state; falls back to tag-prefix gracefully when forum unavailable.
- [ ] `/pause_mission`, `/resume_mission`, `/kill_mission` work with proper state transitions; resume-after-kill rejected.
- [ ] `safety_guard.executor_hook` fires before every action; collision guards block force-push / parallel-overwrite / shared-history-rewrite / out-of-workspace shell / destructive shared DB.
- [ ] Workflow steps tagged with `reversibility`/`locked` honor static-overrides-runtime-downgrade rule.
- [ ] All unit tests + integration tests pass with `timeout 120 pytest tests/ -m "not llm"`.
- [ ] Manual smoke verified by founder for: $0.50 ceiling pause/resume, force-push block, /kill_mission snapshot.

## Cross-references

- Source frame: [docs/i2p-evolution/z0-mission-preflight.md](../../i2p-evolution/z0-mission-preflight.md)
- README: [docs/i2p-evolution/00-README.md](../../i2p-evolution/00-README.md)
- Existing primitives leveraged:
  - `packages/general_beckman/` — admission, ceiling, lifecycle gating
  - `packages/fatih_hoca/` — selection-time budget filter
  - `packages/mr_roboto/` clarify + notify_user — confirmation round-trip + thread posts
  - `src/security/credential_store.py` — vault primitive (Z6 wraps)
  - `src/workflows/engine/` — pre-action hook insertion point
  - `src/app/telegram_bot.py` cmd_mission, /pause, /resume — refined into Z0 lifecycle commands
- Deferred:
  - Founder readiness + vault provisioning UX → Z6 (06-real-world-bridge)
  - Phase tier-eligibility / lazy phase activation → Z2 (02-build-foundation)
  - Compliance fingerprint / target_platform Qs → folded into i2p Phase 0
  - North-star metric → Z9 (09-growth)

## Updates

- 2026-05-05 — initial spec; scope reduced to 5 items after deep audit + founder discussion; tier matrix dropped (no workflow JSON consumers today); founder profile/vault/readiness deferred or pushed to Z6; reversibility tags + collision guards introduced as `packages/safety_guard/`.
