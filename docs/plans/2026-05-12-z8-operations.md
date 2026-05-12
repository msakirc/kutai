# Z8 — Operations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the loop from build-time i2p (terminates at phase 14) to ongoing post-launch operations — monitoring, on-call, incident response, support, backups, cost/perf/security — by adding 4 hinge substrates (mission lifecycle, webhook spine, admission-gate wiring, ops recipes) and wiring 10 v1 gaps onto them.

**Architecture:** Reframe Z8 around 4 hinge points instead of 10 isolated automations:
- **H1** Long-running mission lifecycle (`mission.kind/lifecycle_state/cursor` + Beckman `lane='ongoing'` + orchestrator resumption).
- **H2** Webhook spine (FastAPI listener, signed payloads, dedup table, multi-product routing).
- **H3** Wire pre-shipped `z6_admission.py` gate into Beckman.next_task() + founder_action_done → task unblock.
- **H4** Ops recipes as first-class catalog (Z2 recipe engine extended; 18 new recipes across monitoring, backup, dep-hygiene, cve, cost, playbooks, synthetic).

Reuses: Z6 founder_actions, Z10 reversibility, Z2 recipe engine, IntegrationRegistry, alerting.py, ChromaDB RAG. New: webhook listener, action_cooldowns, webhook_events, integration_mappings, perf_baselines, escalation_policy, tickets, support_docs.

**Tech Stack:** Python 3.10 async, aiosqlite, FastAPI (new for webhook listener), python-telegram-bot, ChromaDB, multilingual-e5-base embeddings, litellm, llama.cpp.

**Source docs:** `docs/i2p-evolution/08-operations-v2.md` (design), `docs/i2p-evolution/08-operations.md` (v1 superseded), `docs/i2p-evolution/00-README.md` (zone map).

**Ground truth as of 2026-05-12** (re-audited):
- H1 fully absent. `scheduled_tasks` table exists but vestigial.
- H2 fully absent.
- H3 PARTIAL: `packages/general_beckman/src/general_beckman/z6_admission.py` shipped 2026-05-11 (commit 74a2175) but NOT wired into `next_task()`. `/action_done` handler exists at `src/app/telegram_bot.py:8799–8811` but task-unblock-on-action-done flow missing.
- H4 substrate shipped (`src/infra/recipes.py`, `packages/mr_roboto/src/mr_roboto/pick_recipe.py`, `packages/mr_roboto/src/mr_roboto/instantiate_recipe.py`); ops recipes catalog empty (only 5 build-time recipes in `recipes/`).
- Z0 added `ambition_tier/cost_ceiling/founder_attention_budget_minutes` (commit fa100f9, 2026-05-12). `mission.product_id` still absent — see T1A.

---

## Scope split decision

Five tiers (T1–T5) live in one plan because tier dependencies are sequential (T1 blocks T3/T4/T5; T2 independent; T3 blocks T4; T4 blocks T5 cron-monitoring) and the parallelizable surface is **within** each tier (T5 has 7 independent recipes). Splitting per tier would multiply context-switch overhead. Sub-tasks within a tier are parallel-safe unless noted.

## Out of scope (defer)

- PagerDuty / Opsgenie pager integration (Telegram + Twilio SMS only in T5G).
- Vantage / CloudHealth aggregator (per-vendor cost only in T5D).
- Webhook listener split-process deployment (embedded in KutAI process v1; sibling process when first prod ships).
- Z0 ownership of `product_id` — see T1A for placeholder + coordination note.
- Build-time recipes (Z2 territory).
- Visual regression in synthetic checks (Z4 territory — references only).

---

## File map

### New files
- `packages/general_beckman/src/general_beckman/lanes.py` — admission lane policy (oneshot vs ongoing) for `next_task()`.
- `packages/general_beckman/src/general_beckman/resumption.py` — startup cursor replay for ongoing missions.
- `src/app/webhook_listener.py` — FastAPI app, single port, per-integration routes.
- `src/app/webhook_signing.py` — per-vendor signature verifiers (Sentry, Stripe HMAC, GitHub X-Hub-Signature-256, Better Stack, Twilio).
- `src/app/webhook_dedup.py` — idempotency check against `webhook_events`.
- `src/ops/severity_classifier.py` — alert severity classifier (rule-based v1; LLM-grader fallback).
- `src/ops/escalation_policy.py` — quiet-hours + severity threshold → channel mapping (Telegram / SMS / log only).
- `src/ops/action_cooldowns.py` — per-mission/per-verb cooldown enforcer.
- `src/ops/support_rag.py` — ChromaDB query over `support_docs` collection + LLM compose with citations.
- `src/ops/cost_anomaly.py` — z-score over 14d on per-integration cost slope.
- `src/ops/perf_baselines.py` — read/write `perf_baselines` table; regression diff.
- `packages/mr_roboto/src/mr_roboto/executors/oncall_action.py` — on-call action verbs (restart/rollback/scale/drain/rotate/archive).
- `packages/mr_roboto/src/mr_roboto/executors/backup_verify.py` — restore-to-sandbox + smoke test.
- `packages/mr_roboto/src/mr_roboto/executors/dependency_scan.py` — pip-audit / npm audit wrapper.
- `packages/mr_roboto/src/mr_roboto/executors/cve_scan.py` — OSV.dev query.
- `packages/mr_roboto/src/mr_roboto/executors/secret_scan.py` — gitleaks/trufflehog wrapper.
- `packages/mr_roboto/src/mr_roboto/executors/synthetic_check.py` — k6/lighthouse wrapper.
- `packages/mr_roboto/src/mr_roboto/executors/cost_pull.py` — per-vendor cost API caller.
- `src/agents/configs/oncall_agent.yaml` — sys-prompt + tool whitelist + reflection block ref.
- `src/agents/configs/support_tier1.yaml` — sys-prompt + RAG tool + escalation tool.
- `recipes/monitoring_kit_fastapi_v1/recipe.yaml` + templates.
- `recipes/monitoring_kit_nextjs_v1/recipe.yaml` + templates.
- `recipes/monitoring_kit_django_v1/recipe.yaml` + templates.
- `recipes/backup_verify_postgres_v1/recipe.yaml`.
- `recipes/backup_verify_sqlite_v1/recipe.yaml`.
- `recipes/dependency_hygiene_python_v1/recipe.yaml`.
- `recipes/dependency_hygiene_node_v1/recipe.yaml`.
- `recipes/cost_monitor_stripe_v1/recipe.yaml`.
- `recipes/cost_monitor_vercel_v1/recipe.yaml`.
- `recipes/cost_monitor_aws_v1/recipe.yaml`.
- `recipes/cve_scan_python_v1/recipe.yaml`.
- `recipes/cve_scan_node_v1/recipe.yaml`.
- `recipes/cve_scan_docker_v1/recipe.yaml`.
- `recipes/incident_playbook_db_disk_full_v1/recipe.yaml`.
- `recipes/incident_playbook_payment_provider_down_v1/recipe.yaml`.
- `recipes/incident_playbook_auth_provider_down_v1/recipe.yaml`.
- `recipes/incident_playbook_cert_expiring_v1/recipe.yaml`.
- `recipes/synthetic_check_lighthouse_v1/recipe.yaml`.
- `recipes/synthetic_check_k6_v1/recipe.yaml`.
- `src/integrations/configs/twilio.json` — SMS escalation.
- `src/integrations/configs/osv.json` — CVE feed.
- `src/integrations/configs/posthog.json` — analytics (Z9 dep, but lands with monitoring kit).
- `src/integrations/configs/betterstack.json` — uptime + synthetic.
- `tests/ops/` — full test tree under here.

### Modified files
- `src/infra/db.py` — new columns + tables (8 schema changes).
- `packages/general_beckman/src/general_beckman/__init__.py` — wire `lanes.py`, `resumption.py`, `z6_admission.py` into `next_task()`.
- `src/core/orchestrator.py` — startup resumption call; revocation handler.
- `src/app/telegram_bot.py` — `/stop_ops`, `/ops_log`, support-tier-1 ticket inlet, action_done unblock hook.
- `src/app/founder_action_render.py` — `kind='credential_request'` + `kind='support_escalation'` render paths.
- `src/founder_actions/__init__.py` — emit unblock event on transition to `done`.
- `src/integrations/registry.py` — `webhook_secret` schema field + credential resolution.
- `src/integrations/configs/sentry.json`, `stripe.json`, `github.json`, `betterstack.json` — `webhook_secret` block.
- `src/infra/alerting.py` — extend with cost-slope rule (T5D).
- `packages/mr_roboto/src/mr_roboto/__init__.py` — register new executors.
- `packages/mr_roboto/src/mr_roboto/reversibility.py` — tag new oncall verbs.
- `src/workflows/i2p/i2p_v3.json` — step 13.3 wires to `monitoring_kit_*_v1` recipe (replaces NEEDS-REAL-TOOLS); new phase 15 steps for ongoing-ops kickoff.
- `kutai_wrapper.py` — boot webhook_listener as sibling task (embedded mode v1).

---

## Tier 1 — Lifecycle Foundation

**Goal:** Make ongoing missions possible. Unlocks every other tier except T2.

**Acceptance:** Ongoing mission survives orchestrator restart; revocation drops subscriptions cleanly; oneshot queue unaffected; existing missions backfilled to `kind='oneshot'/lifecycle_state='terminal'`.

### Task T1A — Schema migration (mission lifecycle)

**Files:**
- Modify: `src/infra/db.py` (add migration block; insert after Z0 ambition_tier migration)
- Test: `tests/infra/test_mission_lifecycle_schema.py`

- [ ] **Step 1: Write failing migration test**

```python
# tests/infra/test_mission_lifecycle_schema.py
import pytest
from src.infra.db import get_db, ensure_schema

@pytest.mark.asyncio
async def test_missions_has_lifecycle_columns(tmp_path):
    db_path = tmp_path / "test.db"
    await ensure_schema(str(db_path))
    async with get_db(str(db_path)) as conn:
        async with conn.execute("PRAGMA table_info(missions)") as cur:
            cols = {row[1] for row in await cur.fetchall()}
    assert "kind" in cols
    assert "lifecycle_state" in cols
    assert "cursor" in cols
    assert "product_id" in cols

@pytest.mark.asyncio
async def test_existing_missions_backfilled(tmp_path):
    db_path = tmp_path / "test.db"
    # simulate pre-migration DB
    async with get_db(str(db_path)) as conn:
        await conn.execute("CREATE TABLE missions (id INTEGER PRIMARY KEY, goal TEXT)")
        await conn.execute("INSERT INTO missions (goal) VALUES ('test1'), ('test2')")
        await conn.commit()
    await ensure_schema(str(db_path))
    async with get_db(str(db_path)) as conn:
        async with conn.execute("SELECT kind, lifecycle_state FROM missions") as cur:
            rows = await cur.fetchall()
    assert all(r[0] == "oneshot" for r in rows)
    assert all(r[1] == "terminal" for r in rows)
```

- [ ] **Step 2: Run test, expect FAIL**

```bash
timeout 30 python -m pytest tests/infra/test_mission_lifecycle_schema.py -v
```
Expected: FAIL — "kind" not in columns.

- [ ] **Step 3: Add migration to `src/infra/db.py`**

Insert migration block (idempotent ALTER TABLE pattern) after Z0 ambition_tier migration block:

```python
# 2026-05-12 Z8 T1A: mission lifecycle columns
await _add_column_if_missing(conn, "missions", "kind", "TEXT NOT NULL DEFAULT 'oneshot'")
await _add_column_if_missing(conn, "missions", "lifecycle_state", "TEXT NOT NULL DEFAULT 'terminal'")
await _add_column_if_missing(conn, "missions", "cursor", "TEXT")  # JSON
await _add_column_if_missing(conn, "missions", "product_id", "TEXT")  # nullable; Z0 may take over
await _add_column_if_missing(conn, "missions", "revoked_at", "TEXT")
await conn.execute("CREATE INDEX IF NOT EXISTS idx_missions_kind_state ON missions(kind, lifecycle_state)")
```

`_add_column_if_missing` already exists in `src/infra/db.py` (used by Z0 migration).

- [ ] **Step 4: Run test, expect PASS**

```bash
timeout 30 python -m pytest tests/infra/test_mission_lifecycle_schema.py -v
```

- [ ] **Step 5: Smoke import**

```bash
timeout 15 python -c "from src.infra.db import ensure_schema; print('ok')"
```

- [ ] **Step 6: Commit**

```bash
git add src/infra/db.py tests/infra/test_mission_lifecycle_schema.py
git commit -m "feat(z8,t1a): mission lifecycle columns (kind/lifecycle_state/cursor/product_id/revoked_at)"
```

**Note on product_id:** Added as nullable placeholder. Z0 may take ownership later. Routing code (T3E) treats NULL as "default product." Coordinate with Z0 if Z0 v2 lands a different shape.

---

### Task T1B — Beckman ongoing-lane admission

**Files:**
- Create: `packages/general_beckman/src/general_beckman/lanes.py`
- Modify: `packages/general_beckman/src/general_beckman/__init__.py` (or whichever file holds `next_task`)
- Test: `tests/general_beckman/test_lanes.py`

**Goal:** `next_task()` returns ongoing tasks from a separate pool that doesn't block oneshot dispatch. Concurrency cap per lane.

- [ ] **Step 1: Write failing test**

```python
# tests/general_beckman/test_lanes.py
import pytest
from general_beckman import next_task, enqueue
from general_beckman.lanes import LANE_ONESHOT, LANE_ONGOING

@pytest.mark.asyncio
async def test_lanes_dispatch_independently(setup_db):
    # enqueue 1 ongoing + 1 oneshot
    await enqueue(mission_id=1, task_type="alert_triage", lane=LANE_ONGOING)
    await enqueue(mission_id=2, task_type="apply", lane=LANE_ONESHOT)
    t1 = await next_task(lane=LANE_ONESHOT)
    t2 = await next_task(lane=LANE_ONGOING)
    assert t1.mission_id == 2
    assert t2.mission_id == 1

@pytest.mark.asyncio
async def test_ongoing_lane_concurrency_cap(setup_db):
    from general_beckman.lanes import ONGOING_CONCURRENCY
    for i in range(ONGOING_CONCURRENCY + 2):
        await enqueue(mission_id=10+i, task_type="alert_triage", lane=LANE_ONGOING)
    # Mark ONGOING_CONCURRENCY as running
    running = [await next_task(lane=LANE_ONGOING) for _ in range(ONGOING_CONCURRENCY)]
    assert all(r is not None for r in running)
    overflow = await next_task(lane=LANE_ONGOING)
    assert overflow is None  # cap enforced
```

- [ ] **Step 2: Run, expect FAIL (lane param unsupported)**

```bash
timeout 30 python -m pytest tests/general_beckman/test_lanes.py -v
```

- [ ] **Step 3: Create `packages/general_beckman/src/general_beckman/lanes.py`**

```python
"""Z8 T1B — admission lane policy.

Two lanes: oneshot (terminal-state missions, default) and ongoing
(alert_triage, cron, support). Separate concurrency caps so ongoing
backpressure does not starve oneshot work.
"""
LANE_ONESHOT = "oneshot"
LANE_ONGOING = "ongoing"

ONESHOT_CONCURRENCY = 4
ONGOING_CONCURRENCY = 8  # webhooks bursty; cron sparse


def pick_lane(task_type: str) -> str:
    """Default lane resolution by task_type."""
    ongoing_types = {"alert_triage", "cron_backup_verify", "cron_dep_hygiene",
                     "cron_cve_scan", "cron_secret_scan", "cron_cost_pull",
                     "cron_synthetic_check", "support_ticket"}
    return LANE_ONGOING if task_type in ongoing_types else LANE_ONESHOT


async def count_in_flight(conn, lane: str) -> int:
    async with conn.execute(
        "SELECT COUNT(*) FROM tasks WHERE lane=? AND status IN ('in_progress','assigned')",
        (lane,),
    ) as cur:
        (n,) = await cur.fetchone()
    return n


async def cap_for(lane: str) -> int:
    return ONGOING_CONCURRENCY if lane == LANE_ONGOING else ONESHOT_CONCURRENCY
```

- [ ] **Step 4: Add `tasks.lane` column migration in `src/infra/db.py`**

```python
# 2026-05-12 Z8 T1B: tasks.lane admission column
await _add_column_if_missing(conn, "tasks", "lane", "TEXT NOT NULL DEFAULT 'oneshot'")
await conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_lane_status ON tasks(lane, status)")
```

- [ ] **Step 5: Wire `next_task(lane=...)` in Beckman**

Modify `packages/general_beckman/src/general_beckman/__init__.py` (or wherever `next_task` lives). Add `lane` parameter (default `LANE_ONESHOT`); SELECT filters by lane; pre-select counts in-flight and returns None if cap reached. Modify `enqueue` to accept `lane` (default = `pick_lane(task_type)`).

```python
async def next_task(lane: str = LANE_ONESHOT, ...):
    async with get_db() as conn:
        in_flight = await count_in_flight(conn, lane)
        if in_flight >= await cap_for(lane):
            return None
        async with conn.execute(
            "SELECT ... FROM tasks WHERE lane=? AND status='ready' ORDER BY priority DESC, id ASC LIMIT 1",
            (lane,),
        ) as cur:
            row = await cur.fetchone()
        ...
```

- [ ] **Step 6: Run tests, expect PASS**

```bash
timeout 60 python -m pytest tests/general_beckman/test_lanes.py -v
```

- [ ] **Step 7: Regression — existing Beckman tests still pass**

```bash
timeout 120 python -m pytest packages/general_beckman/tests/ -v
```

- [ ] **Step 8: Commit**

```bash
git add packages/general_beckman/ tests/general_beckman/test_lanes.py src/infra/db.py
git commit -m "feat(z8,t1b): Beckman ongoing-lane admission (cap=8, oneshot cap=4)"
```

---

### Task T1C — Orchestrator resumption + revocation

**Files:**
- Create: `packages/general_beckman/src/general_beckman/resumption.py`
- Modify: `src/core/orchestrator.py` (startup hook + revocation handler)
- Test: `tests/core/test_orchestrator_resumption.py`

**Goal:** On boot, orchestrator finds all `kind='ongoing' AND lifecycle_state='active' AND revoked_at IS NULL` missions, replays each cursor (re-subscribes webhooks, re-arms cron). On `/stop_ops <mission_id>`, transition `lifecycle_state → revoked` and drop subscriptions.

- [ ] **Step 1: Write failing test**

```python
# tests/core/test_orchestrator_resumption.py
import pytest
from src.core.orchestrator import resume_ongoing_missions
from general_beckman import enqueue_mission

@pytest.mark.asyncio
async def test_resumption_finds_active_ongoing(setup_db):
    mid = await enqueue_mission(goal="watch app", kind="ongoing")
    # mark active
    async with get_db() as conn:
        await conn.execute("UPDATE missions SET lifecycle_state='active', cursor=? WHERE id=?",
                           ('{"sentry": "evt_123"}', mid))
        await conn.commit()
    resumed = await resume_ongoing_missions()
    assert mid in [m.id for m in resumed]
    assert resumed[0].cursor == {"sentry": "evt_123"}

@pytest.mark.asyncio
async def test_resumption_skips_revoked(setup_db):
    mid = await enqueue_mission(goal="watch", kind="ongoing")
    async with get_db() as conn:
        await conn.execute("UPDATE missions SET lifecycle_state='revoked', revoked_at=datetime('now') WHERE id=?", (mid,))
        await conn.commit()
    resumed = await resume_ongoing_missions()
    assert mid not in [m.id for m in resumed]
```

- [ ] **Step 2: Run, expect FAIL**

```bash
timeout 30 python -m pytest tests/core/test_orchestrator_resumption.py -v
```

- [ ] **Step 3: Create `resumption.py`**

```python
"""Z8 T1C — orchestrator resumption for ongoing missions."""
import json
from dataclasses import dataclass
from typing import Any
from src.infra.db import get_db

@dataclass
class ResumedMission:
    id: int
    goal: str
    cursor: dict[str, Any]

async def find_resumable() -> list[ResumedMission]:
    out = []
    async with get_db() as conn:
        async with conn.execute(
            "SELECT id, goal, cursor FROM missions "
            "WHERE kind='ongoing' AND lifecycle_state='active' AND revoked_at IS NULL"
        ) as cur:
            async for row in cur:
                cursor = json.loads(row[2]) if row[2] else {}
                out.append(ResumedMission(id=row[0], goal=row[1], cursor=cursor))
    return out

async def update_cursor(mission_id: int, cursor: dict) -> None:
    async with get_db() as conn:
        await conn.execute("UPDATE missions SET cursor=? WHERE id=?",
                           (json.dumps(cursor), mission_id))
        await conn.commit()

async def revoke(mission_id: int) -> bool:
    async with get_db() as conn:
        await conn.execute(
            "UPDATE missions SET lifecycle_state='revoked', revoked_at=datetime('now') "
            "WHERE id=? AND kind='ongoing'", (mission_id,)
        )
        await conn.commit()
        return conn.total_changes > 0
```

- [ ] **Step 4: Wire orchestrator startup**

Modify `src/core/orchestrator.py`. Inside `startup()` after DB ready + before main pump:

```python
from general_beckman.resumption import find_resumable

async def startup(self):
    ...  # existing init
    resumed = await find_resumable()
    for m in resumed:
        logger.info(f"resuming ongoing mission {m.id} with cursor {m.cursor}")
        # rebind subscriptions: handled by alert_triage handlers reading cursor
        # cron tasks will be re-enqueued by the cron scheduler (T5)
        await self._rebind_ongoing(m)
```

`_rebind_ongoing` for v1 just logs and lets webhook handler / cron scheduler pick up by querying `find_resumable`. Full re-subscribe logic builds in T3D.

- [ ] **Step 5: Run tests, expect PASS**

```bash
timeout 30 python -m pytest tests/core/test_orchestrator_resumption.py -v
```

- [ ] **Step 6: Commit**

```bash
git add packages/general_beckman/src/general_beckman/resumption.py src/core/orchestrator.py tests/core/test_orchestrator_resumption.py
git commit -m "feat(z8,t1c): orchestrator resumption + revocation for ongoing missions"
```

---

### Task T1D — `/stop_ops` Telegram command

**Files:**
- Modify: `src/app/telegram_bot.py` (add command + handler)
- Test: `tests/app/test_stop_ops_command.py`

- [ ] **Step 1: Write failing test**

```python
# tests/app/test_stop_ops_command.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.app.telegram_bot import TelegramInterface

@pytest.mark.asyncio
async def test_stop_ops_revokes_mission(setup_db, monkeypatch):
    mid = await _make_ongoing_mission()
    bot = TelegramInterface(token="x")
    update = MagicMock()
    update.message.text = f"/stop_ops {mid}"
    update.message.reply_text = AsyncMock()
    context = MagicMock()
    context.args = [str(mid)]
    await bot.cmd_stop_ops(update, context)
    async with get_db() as conn:
        async with conn.execute("SELECT lifecycle_state FROM missions WHERE id=?", (mid,)) as cur:
            (state,) = await cur.fetchone()
    assert state == "revoked"
    update.message.reply_text.assert_awaited()
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Add handler**

```python
# src/app/telegram_bot.py
from general_beckman.resumption import revoke

async def cmd_stop_ops(self, update, context):
    chat_id = update.effective_chat.id
    if not context.args:
        await self._reply(update, "Usage: /stop_ops <mission_id>")
        return
    try:
        mid = int(context.args[0])
    except ValueError:
        await self._reply(update, "mission_id must be int")
        return
    ok = await revoke(mid)
    msg = f"mission {mid} revoked" if ok else f"mission {mid} not ongoing or not found"
    await self._reply(update, msg)
```

Register in `_setup_handlers()`:
```python
self.app.add_handler(CommandHandler("stop_ops", self.cmd_stop_ops))
```

- [ ] **Step 4: Run, expect PASS**

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(z8,t1d): /stop_ops Telegram command revokes ongoing mission"
```

---

### Task T1E — Integration test: ongoing mission survives restart

**Files:**
- Test: `tests/integration/test_t1_lifecycle_e2e.py`

- [ ] **Step 1: Write integration test**

```python
@pytest.mark.asyncio
async def test_ongoing_mission_survives_simulated_restart(setup_db):
    mid = await enqueue_mission(goal="watch", kind="ongoing")
    async with get_db() as conn:
        await conn.execute("UPDATE missions SET lifecycle_state='active', cursor='{\"sentry\":\"e1\"}' WHERE id=?", (mid,))
        await conn.commit()
    # "restart" — fresh resumption pass
    resumed = await find_resumable()
    assert any(r.id == mid for r in resumed)
    # revoke
    await revoke(mid)
    resumed2 = await find_resumable()
    assert not any(r.id == mid for r in resumed2)
```

- [ ] **Step 2: Run, expect PASS**

- [ ] **Step 3: Commit + tag**

```bash
git add tests/integration/test_t1_lifecycle_e2e.py
git commit -m "test(z8,t1e): ongoing mission survives simulated restart + revoke"
git tag z8-t1-shipped
```

---

## Tier 2 — Admission gate wiring (ALREADY SHIPPED — verification only)

**Status (verified 2026-05-12 by T2 subagent):** SHIPPED. Initial v2 audit was wrong — `check_z6_admission` is wired into `next_task()` at `packages/general_beckman/src/general_beckman/__init__.py:329-355` since commit `a171bcd` (Z6 T1C). `/action_done` chain works end-to-end: `cmd_action_done` (`src/app/telegram_bot.py:8890-8925`) → `fa.resolve()` → `update_status()` → `unblock_mission_if_clear()` (`src/founder_actions/__init__.py:498-502`) flips parked tasks `blocked_on_founder_action → pending`.

**API differs from plan draft:**
- `check_z6_admission(task, mission_id) -> AdmissionResult` (not `check_or_park(task) -> bool`).
- Uses `for task in candidates` loop with `BECKMAN_TOP_K=5` (no explicit `_depth` recursion needed; implicit bound).
- `founder_actions` schema columns: `blocking_task_id` (not `task_id`), `instructions_json` + `response_payload_json` (no `payload`/`note` columns).
- `resolve(action_id, response_payload=None)` is the public API; lower-level is `update_status(action_id, new_status, response_payload=None)`.
- Unblock flips status to `pending` (not `ready`).

**T2A and T2B replaced with verification checklist below.**

**Goal (revised):** Verify shipped admission gate + unblock chain by running existing tests.

**Acceptance:** `tests/general_beckman/test_z6_admission.py` + `tests/founder_actions/test_lifecycle.py::test_unblock_flips_blocked_tasks_back_to_pending` green.

### Task T2-verify — Confirm shipped wiring (replaces original T2A + T2B)

**Files (read-only verification):**
- `packages/general_beckman/src/general_beckman/__init__.py:322-355` (admission call site)
- `packages/general_beckman/src/general_beckman/z6_admission.py` (`check_z6_admission` definition)
- `src/founder_actions/__init__.py:385,498-502` (unblock chain)
- `src/app/telegram_bot.py:8890-8925` (`/action_done` handler)

- [ ] **Step 1: Run shipped tests**

```bash
timeout 120 python -m pytest packages/general_beckman/tests/test_z6_admission.py tests/founder_actions/test_lifecycle.py -v
```

Expected: all PASS. Covers admission scenarios (not-needs-real-tools admits / missing-kind / no-adapter / no-credentials / cost-ack-required) + unblock acceptance.

- [ ] **Step 2: No commit (no code change).**

If a hardening refactor (`check_or_park` unified API, explicit `_depth` guard) is desired later, raise as separate proposal — not part of Z8 T2.

---

### [SKIPPED — original T2A retained below for archival reference only]

### Original Task T2A — Wire z6_admission gate into next_task()

**Files:**
- Modify: `packages/general_beckman/src/general_beckman/__init__.py`
- Modify: `packages/general_beckman/src/general_beckman/z6_admission.py` (export `check_or_park`)
- Test: `tests/general_beckman/test_z6_admission_wired.py`

- [ ] **Step 1: Read current z6_admission.py to confirm public API**

```bash
# Already audited: module shipped 2026-05-11 commit 74a2175.
# Confirm functions exported.
```

Read `packages/general_beckman/src/general_beckman/z6_admission.py:1–100`. Note exact function names.

- [ ] **Step 2: Write failing test**

```python
@pytest.mark.asyncio
async def test_needs_real_tools_with_missing_credential_parks_task(setup_db):
    # enqueue task that needs sentry
    tid = await enqueue(mission_id=1, task_type="apply", lane="oneshot",
                        needs_real_tools=True,
                        context={"post_hook": {"service": "sentry", "action": "create_project"}})
    # no sentry credential configured
    task = await next_task(lane="oneshot")
    assert task is None  # parked, not dispatched
    async with get_db() as conn:
        async with conn.execute("SELECT status FROM tasks WHERE id=?", (tid,)) as cur:
            (status,) = await cur.fetchone()
        async with conn.execute("SELECT COUNT(*) FROM founder_actions WHERE kind='credential_request'") as cur:
            (n,) = await cur.fetchone()
    assert status == "blocked_on_founder_action"
    assert n == 1

@pytest.mark.asyncio
async def test_needs_real_tools_with_credential_admits(setup_db):
    await _seed_credential("sentry", "fake-token")
    tid = await enqueue(mission_id=1, task_type="apply", lane="oneshot",
                        needs_real_tools=True,
                        context={"post_hook": {"service": "sentry"}})
    task = await next_task(lane="oneshot")
    assert task is not None
    assert task.id == tid
```

- [ ] **Step 3: Run, expect FAIL**

```bash
timeout 30 python -m pytest tests/general_beckman/test_z6_admission_wired.py -v
```

- [ ] **Step 4: Wire gate**

In `packages/general_beckman/src/general_beckman/__init__.py::next_task()`, after row fetch and before dispatch:

```python
from general_beckman.z6_admission import check_or_park

async def next_task(lane=LANE_ONESHOT, ...):
    ...
    row = await cur.fetchone()
    if not row:
        return None
    task = _row_to_task(row)
    if task.needs_real_tools:
        admit = await check_or_park(task)
        if not admit:
            # transition was written by check_or_park; recurse to find next admissible task
            return await next_task(lane=lane, _depth=_depth+1)  # bounded recursion
    return task
```

Bounded recursion: pass `_depth` param, max 5. If too many parked in a row, return None to avoid spinning.

- [ ] **Step 5: Implement `check_or_park` in z6_admission.py (if not already)**

```python
async def check_or_park(task) -> bool:
    """Return True if task admissible (all required integrations have valid credentials).
    If False, write task.status='blocked_on_founder_action' + insert founder_actions row."""
    required = _required_integrations(task)
    missing = [svc for svc in required if not await _has_valid_credential(svc)]
    if not missing:
        return True
    async with get_db() as conn:
        await conn.execute("UPDATE tasks SET status='blocked_on_founder_action' WHERE id=?", (task.id,))
        for svc in missing:
            await conn.execute(
                "INSERT INTO founder_actions (mission_id, task_id, kind, payload, status, created_at) "
                "VALUES (?, ?, 'credential_request', ?, 'pending', datetime('now'))",
                (task.mission_id, task.id, json.dumps({"service": svc, "scopes": _default_scopes(svc)})),
            )
        await conn.commit()
    return False
```

`_required_integrations(task)` reads `task.context.post_hook.service` (string) plus any recipe-declared `requires.integrations` list when task has a recipe ref.

- [ ] **Step 6: Run, expect PASS**

- [ ] **Step 7: Regression**

```bash
timeout 120 python -m pytest packages/general_beckman/tests/ -v
```

- [ ] **Step 8: Commit**

```bash
git commit -am "feat(z8,t2a): wire z6_admission gate into Beckman.next_task() — parks credential-missing tasks"
```

---

### [SKIPPED — original T2B retained below for archival reference only]

### Original Task T2B — `/action_done` unblocks parked task

**Files:**
- Modify: `src/founder_actions/__init__.py` (add unblock hook on transition to `done`)
- Modify: `src/app/telegram_bot.py` (confirm cmd_action_done calls into founder_actions repo)
- Test: `tests/founder_actions/test_action_done_unblocks.py`

- [ ] **Step 1: Write failing test**

```python
@pytest.mark.asyncio
async def test_action_done_unblocks_parked_task(setup_db):
    tid = await enqueue(needs_real_tools=True, context={"post_hook": {"service": "sentry"}})
    await next_task(lane="oneshot")  # parks it
    # founder approves
    async with get_db() as conn:
        async with conn.execute("SELECT id FROM founder_actions WHERE task_id=?", (tid,)) as cur:
            (aid,) = await cur.fetchone()
    await _seed_credential("sentry", "real-token")
    from src.founder_actions import mark_done
    await mark_done(aid, note="credential added")
    async with get_db() as conn:
        async with conn.execute("SELECT status FROM tasks WHERE id=?", (tid,)) as cur:
            (status,) = await cur.fetchone()
    assert status == "ready"
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Add unblock hook in `mark_done()`**

```python
# src/founder_actions/__init__.py
async def mark_done(action_id: int, note: str = "") -> None:
    async with get_db() as conn:
        async with conn.execute("SELECT task_id, kind FROM founder_actions WHERE id=?", (action_id,)) as cur:
            row = await cur.fetchone()
        if not row:
            return
        task_id, kind = row
        await conn.execute(
            "UPDATE founder_actions SET status='done', completed_at=datetime('now'), note=? WHERE id=?",
            (note, action_id),
        )
        if task_id and kind == "credential_request":
            # check if all credential_requests on this task are done
            async with conn.execute(
                "SELECT COUNT(*) FROM founder_actions WHERE task_id=? AND kind='credential_request' AND status='pending'",
                (task_id,),
            ) as cur:
                (pending,) = await cur.fetchone()
            if pending == 0:
                await conn.execute("UPDATE tasks SET status='ready' WHERE id=? AND status='blocked_on_founder_action'", (task_id,))
        await conn.commit()
```

- [ ] **Step 4: Confirm Telegram handler routes through `mark_done`**

Read `src/app/telegram_bot.py:8799–8811` (cmd_action_done). Should already call `mark_done`. If it inlines status update, refactor to use `mark_done`.

- [ ] **Step 5: Run, expect PASS**

- [ ] **Step 6: Commit + tag**

```bash
git commit -am "feat(z8,t2b): /action_done unblocks parked tasks when all credential_requests done"
git tag z8-t2-shipped
```

---

## Tier 3 — Webhook spine

**Goal:** New HTTP surface ingests vendor webhooks (Sentry, Stripe, GitHub, Better Stack), verifies signatures, dedups, routes to alert_triage tasks scoped by product.

**Acceptance:** Replayed webhook deduped; bad signature → 401; valid alert enqueues `alert_triage` task on correct ongoing mission; multi-product routing works.

### Task T3A — FastAPI listener + webhook_events table

**Files:**
- Create: `src/app/webhook_listener.py`
- Create: `src/app/webhook_dedup.py`
- Modify: `src/infra/db.py` (add `webhook_events` table)
- Modify: `kutai_wrapper.py` (boot listener as sibling task)
- Test: `tests/app/test_webhook_listener.py`

- [ ] **Step 1: Add `webhook_events` migration**

```python
# src/infra/db.py — Z8 T3A
await conn.execute("""
    CREATE TABLE IF NOT EXISTS webhook_events (
        integration_id TEXT NOT NULL,
        event_id TEXT NOT NULL,
        received_at TEXT NOT NULL,
        payload_hash TEXT NOT NULL,
        mission_id INTEGER,
        processed_at TEXT,
        PRIMARY KEY (integration_id, event_id)
    )
""")
await conn.execute("CREATE INDEX IF NOT EXISTS idx_webhook_received ON webhook_events(received_at)")
```

- [ ] **Step 2: Write failing test for dedup**

```python
# tests/app/test_webhook_listener.py
import pytest
from httpx import AsyncClient
from src.app.webhook_listener import app

@pytest.mark.asyncio
async def test_duplicate_event_returns_200_but_not_reprocessed(monkeypatch, setup_db):
    monkeypatch.setattr("src.app.webhook_signing.verify_sentry", lambda *a, **k: True)
    payload = {"event_id": "abc", "type": "error", "data": {}}
    async with AsyncClient(app=app, base_url="http://test") as client:
        r1 = await client.post("/webhook/sentry", json=payload,
                                headers={"sentry-signature": "fake"})
        r2 = await client.post("/webhook/sentry", json=payload,
                                headers={"sentry-signature": "fake"})
    assert r1.status_code == 200
    assert r2.status_code == 200
    async with get_db() as conn:
        async with conn.execute("SELECT COUNT(*) FROM webhook_events") as cur:
            (n,) = await cur.fetchone()
        async with conn.execute("SELECT COUNT(*) FROM tasks WHERE task_type='alert_triage'") as cur:
            (m,) = await cur.fetchone()
    assert n == 1  # deduped
    assert m == 1  # only one triage task

@pytest.mark.asyncio
async def test_bad_signature_returns_401(monkeypatch):
    monkeypatch.setattr("src.app.webhook_signing.verify_sentry", lambda *a, **k: False)
    async with AsyncClient(app=app, base_url="http://test") as client:
        r = await client.post("/webhook/sentry", json={"event_id": "x"},
                              headers={"sentry-signature": "bad"})
    assert r.status_code == 401
```

- [ ] **Step 3: Run, expect FAIL (app does not exist)**

- [ ] **Step 4: Create `webhook_listener.py`**

```python
"""Z8 T3A — FastAPI webhook listener (embedded in KutAI process).

Single port (configurable via WEBHOOK_PORT env, default 9881).
One route per known vendor + generic /webhook/<integration_id> for new vendors.
Signature verify -> dedup -> enqueue alert_triage. Returns 200 fast (<5s).
"""
import hashlib
import json
import logging
from fastapi import FastAPI, Request, HTTPException, Header
from src.infra.db import get_db
from src.app.webhook_signing import verify_signature
from src.app.webhook_dedup import already_seen, mark_seen
from general_beckman import enqueue
from general_beckman.lanes import LANE_ONGOING

logger = logging.getLogger("kutai.webhook")
app = FastAPI()


@app.post("/webhook/{integration_id}")
async def webhook_inbound(integration_id: str, request: Request):
    raw = await request.body()
    sig_headers = dict(request.headers)
    if not await verify_signature(integration_id, raw, sig_headers):
        raise HTTPException(status_code=401, detail="bad signature")
    payload = json.loads(raw)
    event_id = _extract_event_id(integration_id, payload)
    if not event_id:
        raise HTTPException(status_code=400, detail="missing event_id")
    payload_hash = hashlib.sha256(raw).hexdigest()
    if await already_seen(integration_id, event_id):
        return {"status": "duplicate", "event_id": event_id}
    mission_id = await _route_to_mission(integration_id, payload)
    await mark_seen(integration_id, event_id, payload_hash, mission_id)
    await enqueue(
        mission_id=mission_id,
        task_type="alert_triage",
        lane=LANE_ONGOING,
        context={"integration_id": integration_id, "event_id": event_id, "payload": payload},
    )
    return {"status": "accepted", "event_id": event_id}


def _extract_event_id(integration_id: str, payload: dict) -> str | None:
    extractors = {
        "sentry": lambda p: p.get("event_id") or p.get("id"),
        "stripe": lambda p: p.get("id"),
        "github": lambda p: p.get("delivery") or p.get("id"),
        "betterstack": lambda p: p.get("id") or p.get("uuid"),
    }
    fn = extractors.get(integration_id, lambda p: p.get("event_id") or p.get("id"))
    return fn(payload)


async def _route_to_mission(integration_id: str, payload: dict) -> int | None:
    # T3E adds product_id → mission routing via integration_mappings
    async with get_db() as conn:
        async with conn.execute(
            "SELECT mission_id FROM integration_mappings WHERE integration_id=? "
            "AND (product_id IS NULL OR product_id=?) ORDER BY product_id NULLS LAST LIMIT 1",
            (integration_id, payload.get("product_id")),
        ) as cur:
            row = await cur.fetchone()
    return row[0] if row else None
```

- [ ] **Step 5: Create `webhook_dedup.py`**

```python
from src.infra.db import get_db

async def already_seen(integration_id: str, event_id: str) -> bool:
    async with get_db() as conn:
        async with conn.execute(
            "SELECT 1 FROM webhook_events WHERE integration_id=? AND event_id=?",
            (integration_id, event_id),
        ) as cur:
            return await cur.fetchone() is not None

async def mark_seen(integration_id: str, event_id: str, payload_hash: str, mission_id: int | None) -> None:
    async with get_db() as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO webhook_events (integration_id, event_id, received_at, payload_hash, mission_id) "
            "VALUES (?, ?, datetime('now'), ?, ?)",
            (integration_id, event_id, payload_hash, mission_id),
        )
        await conn.commit()
```

- [ ] **Step 6: Boot in kutai_wrapper.py as sibling task**

```python
# kutai_wrapper.py - after orchestrator start
import uvicorn
from src.app.webhook_listener import app as webhook_app

webhook_task = asyncio.create_task(
    uvicorn.Server(uvicorn.Config(webhook_app, host="0.0.0.0",
                                  port=int(os.getenv("WEBHOOK_PORT", "9881")),
                                  log_level="info")).serve()
)
```

- [ ] **Step 7: Run, expect PASS (after signing module placeholder)**

Test patches `verify_signature`. Real verifiers ship in T3B.

- [ ] **Step 8: Commit**

```bash
git commit -m "feat(z8,t3a): FastAPI webhook listener + webhook_events dedup table (embedded mode)"
```

---

### Task T3B — Per-vendor signature verification

**Files:**
- Create: `src/app/webhook_signing.py`
- Test: `tests/app/test_webhook_signing.py`

**Verifiers needed v1:** Sentry HMAC-SHA256, Stripe (`Stripe-Signature`), GitHub (`X-Hub-Signature-256`), Better Stack, Twilio (`X-Twilio-Signature`).

- [ ] **Step 1: Write failing test**

```python
@pytest.mark.parametrize("integration,header_name,sig_fn", [
    ("sentry", "sentry-hook-signature", _hmac_sha256),
    ("stripe", "stripe-signature", _stripe_v1),
    ("github", "x-hub-signature-256", _github_sha256),
])
def test_signature_verification(integration, header_name, sig_fn):
    secret = "test-secret"
    payload = b'{"event_id":"abc"}'
    valid_sig = sig_fn(payload, secret)
    assert verify_signature(integration, payload, {header_name: valid_sig}, secret=secret) is True
    assert verify_signature(integration, payload, {header_name: "tampered"}, secret=secret) is False
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement verifiers**

```python
# src/app/webhook_signing.py
import hashlib
import hmac
import logging
import time
from src.integrations.registry import IntegrationRegistry

logger = logging.getLogger("kutai.webhook.signing")

async def verify_signature(integration_id: str, raw: bytes, headers: dict, secret: str | None = None) -> bool:
    if secret is None:
        secret = await _load_webhook_secret(integration_id)
        if not secret:
            logger.warning(f"no webhook_secret for {integration_id} — rejecting")
            return False
    verifiers = {
        "sentry": _verify_sentry,
        "stripe": _verify_stripe,
        "github": _verify_github,
        "betterstack": _verify_betterstack,
        "twilio": _verify_twilio,
    }
    fn = verifiers.get(integration_id)
    if not fn:
        logger.warning(f"no verifier registered for {integration_id} — rejecting")
        return False
    return fn(raw, headers, secret)


def _verify_sentry(raw: bytes, headers: dict, secret: str) -> bool:
    sig = headers.get("sentry-hook-signature", "")
    expected = hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, expected)


def _verify_stripe(raw: bytes, headers: dict, secret: str) -> bool:
    header = headers.get("stripe-signature", "")
    parts = {kv.split("=")[0]: kv.split("=")[1] for kv in header.split(",") if "=" in kv}
    t = parts.get("t", "")
    v1 = parts.get("v1", "")
    if not t or not v1:
        return False
    if abs(time.time() - int(t)) > 300:  # 5-minute tolerance
        return False
    signed_payload = f"{t}.{raw.decode()}".encode()
    expected = hmac.new(secret.encode(), signed_payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(v1, expected)


def _verify_github(raw: bytes, headers: dict, secret: str) -> bool:
    sig = headers.get("x-hub-signature-256", "")
    if not sig.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, expected)


def _verify_betterstack(raw: bytes, headers: dict, secret: str) -> bool:
    sig = headers.get("x-betterstack-signature", "")
    expected = hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, expected)


def _verify_twilio(raw: bytes, headers: dict, secret: str) -> bool:
    # Twilio uses URL+params signature; webhook handler must pass URL via headers["x-twilio-url"]
    sig = headers.get("x-twilio-signature", "")
    url = headers.get("x-twilio-url", "")
    expected = hmac.new(secret.encode(), (url + raw.decode()).encode(), hashlib.sha1).digest()
    import base64
    expected_b64 = base64.b64encode(expected).decode()
    return hmac.compare_digest(sig, expected_b64)


async def _load_webhook_secret(integration_id: str) -> str | None:
    reg = IntegrationRegistry()
    config = await reg.get_config(integration_id)
    return config.get("webhook_secret") if config else None
```

- [ ] **Step 4: Run, expect PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(z8,t3b): per-vendor webhook signature verification (sentry/stripe/github/betterstack/twilio)"
```

---

### Task T3C — webhook_secret schema in IntegrationRegistry

**Files:**
- Modify: `src/integrations/registry.py` (schema accepts `webhook_secret`)
- Modify: `src/integrations/configs/sentry.json`, `stripe.json`, `github.json`, `betterstack.json` (add `webhook_secret` env-ref block)
- Create: `src/integrations/configs/twilio.json`
- Test: `tests/integrations/test_webhook_secret.py`

- [ ] **Step 1: Read current schema**

Confirm `src/integrations/registry.py` schema fields. Add `webhook_secret`: string env reference or null.

- [ ] **Step 2: Write failing test**

```python
def test_webhook_secret_resolves_from_env(monkeypatch):
    monkeypatch.setenv("SENTRY_WEBHOOK_SECRET", "shh")
    reg = IntegrationRegistry()
    config = reg.get_config_sync("sentry")
    assert config["webhook_secret"] == "shh"
```

- [ ] **Step 3: Update vendor configs**

```json
// src/integrations/configs/sentry.json — add field
{
  ...,
  "webhook_secret": "${SENTRY_WEBHOOK_SECRET}"
}
```

Repeat for stripe (`STRIPE_WEBHOOK_SECRET`), github (`GITHUB_WEBHOOK_SECRET`), betterstack (`BETTERSTACK_WEBHOOK_SECRET`).

- [ ] **Step 4: Create twilio.json**

```json
{
  "id": "twilio",
  "kind": "sms",
  "endpoint": "https://api.twilio.com/2010-04-01",
  "auth": {"account_sid": "${TWILIO_ACCOUNT_SID}", "auth_token": "${TWILIO_AUTH_TOKEN}"},
  "webhook_secret": "${TWILIO_AUTH_TOKEN}",
  "from_number": "${TWILIO_FROM_NUMBER}"
}
```

- [ ] **Step 5: Run, expect PASS**

- [ ] **Step 6: Commit**

```bash
git commit -am "feat(z8,t3c): webhook_secret in integration configs (sentry/stripe/github/betterstack) + twilio config"
```

---

### Task T3D — alert_triage task type + severity classifier

**Files:**
- Create: `src/ops/severity_classifier.py`
- Modify: `packages/mr_roboto/src/mr_roboto/__init__.py` (or coulson runtime) — register `alert_triage` task_type executor
- Test: `tests/ops/test_severity_classifier.py`

**Severity rules v1 (rule-based; LLM-graded only when rule output is `uncertain`):**
- `critical` — Sentry: event_count>100 in 5min; Stripe: payment_intent.failed (live mode); Better Stack: monitor_status==down; CVE: severity==CRITICAL.
- `high` — Sentry: new issue affecting >5 users; Stripe: dispute.created; Better Stack: degraded.
- `medium` — Sentry: regression; Stripe: invoice.payment_failed (recoverable); Better Stack: latency_p95 regression.
- `low` — everything else; digest only.

- [ ] **Step 1: Write failing test**

```python
def test_classify_sentry_event_spike_is_critical():
    payload = {"event_count": 500, "timeframe_minutes": 5, "issue": "ConnectionError"}
    assert classify("sentry", "issue_alert", payload) == "critical"

def test_classify_betterstack_down_is_critical():
    payload = {"monitor": {"status": "down", "url": "https://example.com"}}
    assert classify("betterstack", "incident", payload) == "critical"

def test_classify_unknown_returns_uncertain():
    payload = {"strange": "data"}
    assert classify("sentry", "unknown_type", payload) == "uncertain"
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Create classifier**

```python
# src/ops/severity_classifier.py
def classify(integration_id: str, event_type: str, payload: dict) -> str:
    rules = _RULES.get(integration_id, {}).get(event_type)
    if not rules:
        return "uncertain"
    for severity, predicate in rules:
        try:
            if predicate(payload):
                return severity
        except (KeyError, TypeError, ValueError):
            continue
    return "low"


_RULES = {
    "sentry": {
        "issue_alert": [
            ("critical", lambda p: p.get("event_count", 0) > 100 and p.get("timeframe_minutes", 60) <= 5),
            ("high", lambda p: p.get("affected_users", 0) > 5),
            ("medium", lambda p: p.get("is_regression", False)),
        ],
    },
    "stripe": {
        "payment_intent.payment_failed": [
            ("critical", lambda p: p.get("data", {}).get("object", {}).get("livemode")),
        ],
        "charge.dispute.created": [("high", lambda p: True)],
        "invoice.payment_failed": [("medium", lambda p: True)],
    },
    "betterstack": {
        "incident": [
            ("critical", lambda p: p["monitor"]["status"] == "down"),
            ("high", lambda p: p["monitor"]["status"] == "degraded"),
        ],
    },
    "github": {
        "repository_advisory": [
            ("critical", lambda p: p["advisory"]["severity"] == "critical"),
            ("high", lambda p: p["advisory"]["severity"] == "high"),
        ],
    },
}
```

- [ ] **Step 4: Register `alert_triage` task type**

The triage executor reads payload, calls `classify()`, then either dispatches to oncall_agent (T4A) or writes a digest entry. For T3, only the classify+route logic; oncall handoff stubbed.

```python
# packages/mr_roboto/src/mr_roboto/executors/alert_triage.py
async def run(task) -> dict:
    ctx = task.context
    payload = ctx["payload"]
    integration_id = ctx["integration_id"]
    event_type = payload.get("type") or payload.get("event") or "unknown"
    severity = classify(integration_id, event_type, payload)
    if severity == "uncertain":
        severity = await _llm_grade(integration_id, event_type, payload)
    return {"severity": severity, "integration_id": integration_id, "event_type": event_type}
```

Register in mr_roboto __init__ keyed by `task_type == "alert_triage"`.

- [ ] **Step 5: Run, expect PASS**

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(z8,t3d): alert_triage task_type + rule-based severity classifier (sentry/stripe/betterstack/github)"
```

---

### Task T3E — integration_mappings table + product_id routing

**Files:**
- Modify: `src/infra/db.py` (new table)
- Modify: `src/app/webhook_listener.py` (`_route_to_mission` already queries this)
- Test: `tests/app/test_integration_mappings.py`

- [ ] **Step 1: Migration**

```python
await conn.execute("""
    CREATE TABLE IF NOT EXISTS integration_mappings (
        integration_id TEXT NOT NULL,
        product_id TEXT,
        mission_id INTEGER NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (integration_id, product_id, mission_id)
    )
""")
```

- [ ] **Step 2: Write test**

```python
@pytest.mark.asyncio
async def test_webhook_routes_to_mission_by_integration(setup_db):
    mid = await enqueue_mission(goal="watch", kind="ongoing")
    async with get_db() as conn:
        await conn.execute("UPDATE missions SET lifecycle_state='active' WHERE id=?", (mid,))
        await conn.execute("INSERT INTO integration_mappings (integration_id, mission_id) VALUES (?, ?)",
                           ("sentry", mid))
        await conn.commit()
    # simulate webhook
    async with AsyncClient(app=app) as client:
        await client.post("/webhook/sentry", json={"event_id": "x", "type": "issue_alert"},
                          headers={"sentry-hook-signature": _valid_sig})
    async with get_db() as conn:
        async with conn.execute("SELECT mission_id FROM tasks WHERE task_type='alert_triage'") as cur:
            (got,) = await cur.fetchone()
    assert got == mid

@pytest.mark.asyncio
async def test_product_id_disambiguates(setup_db):
    m1 = await enqueue_mission(goal="prod1", kind="ongoing")
    m2 = await enqueue_mission(goal="prod2", kind="ongoing")
    async with get_db() as conn:
        await conn.execute("INSERT INTO integration_mappings (integration_id, product_id, mission_id) VALUES ('sentry', 'A', ?)", (m1,))
        await conn.execute("INSERT INTO integration_mappings (integration_id, product_id, mission_id) VALUES ('sentry', 'B', ?)", (m2,))
        await conn.commit()
    async with AsyncClient(app=app) as client:
        await client.post("/webhook/sentry", json={"event_id": "x", "product_id": "B"}, headers=...)
    # task lands on m2
```

- [ ] **Step 3: Run, expect PASS (route logic already in T3A)**

- [ ] **Step 4: Commit + tag**

```bash
git commit -am "feat(z8,t3e): integration_mappings table + product_id-aware webhook routing"
git tag z8-t3-shipped
```

---

## Tier 4 — On-call agent + playbooks

**Goal:** New `oncall_agent` profile executes alert response from a whitelisted action set with per-verb cooldowns. Playbooks generated at phase 13 from spec + arch.

**Acceptance:** Out-of-whitelist verb refused; rollback-loop blocked by cooldown; playbook executes matching recipe; `/ops_log` shows agent actions per mission.

### Task T4A — oncall_agent profile + tool whitelist

**Files:**
- Create: `src/agents/configs/oncall_agent.yaml`
- Create: `packages/coulson/src/coulson/reflection.py` — add `oncall_agent` reflection block (tone: methodical, no panic; rollback-safe; document each action).
- Modify: `packages/coulson/...` — agent registry includes oncall_agent
- Test: `tests/agents/test_oncall_agent_prompt.py`

`oncall_agent.yaml` follows existing agent config schema (per `tests/agents/test_prompt_quality.py` invariants: first line `You are ...`, must/always + don't/never, final_answer + fenced json schema).

```yaml
# src/agents/configs/oncall_agent.yaml
name: oncall_agent
sys_prompt: |
  You are the on-call engineer for a live production system.
  
  You must:
  - Read incoming alert payloads carefully before acting
  - Match alerts to known playbooks first; execute matching playbook step-by-step
  - Stay within the action whitelist below; refuse anything outside
  - Log every action with reason and outcome
  
  You must never:
  - Migrate schemas, delete data, change architecture, deploy unreviewed code
  - Re-run an action blocked by cooldown
  - Take action on tier-3 (security incidents) — escalate to founder immediately
  
  Action whitelist: restart_service, rollback_to_last_green, scale_up, scale_down,
  drain_traffic, rotate_failed_key, archive_flake_test, escalate_to_founder.
  
  Return final_answer as JSON:
  ```json
  {"action": "verb_name", "params": {...}, "reason": "...", "expected_outcome": "..."}
  ```
allowed_tools: [vendor_call, oncall_action, escalate_to_founder, ops_log_write]
reflection_block: oncall
```

- [ ] **Step 1: Write prompt-quality test (matches existing pattern)**

```python
# tests/agents/test_oncall_agent_prompt.py
def test_oncall_agent_prompt_invariants():
    prompt = load_agent_prompt("oncall_agent")
    assert prompt.split("\n")[0].startswith("You are")
    assert "must" in prompt.lower() or "always" in prompt.lower()
    assert "don't" in prompt.lower() or "never" in prompt.lower()
    assert "final_answer" in prompt
    assert "```json" in prompt
```

- [ ] **Step 2: Run, expect FAIL (agent not registered)**

- [ ] **Step 3: Write yaml + register**

- [ ] **Step 4: Add reflection block**

```python
# packages/coulson/src/coulson/reflection.py
REFLECTION_BLOCKS["oncall"] = """
Before answering:
1. Have you read the entire payload, not just severity?
2. Does a known playbook match this alert? Use it.
3. Is the proposed action in the whitelist?
4. Is the action in cooldown? If yes, escalate instead.
5. Will the action be reversible? If no and severity < critical, escalate.
"""
```

- [ ] **Step 5: Run, expect PASS**

- [ ] **Step 6: Commit**

```bash
git commit -am "feat(z8,t4a): oncall_agent profile + reflection block (whitelisted actions only)"
```

---

### Task T4B — action_cooldowns table + Mr. Roboto pre-execute enforcement

**Files:**
- Create: `src/ops/action_cooldowns.py`
- Modify: `src/infra/db.py` (table + per-verb default policy seed)
- Modify: `packages/mr_roboto/src/mr_roboto/executors/oncall_action.py` (pre-execute check)
- Test: `tests/ops/test_action_cooldowns.py`

**Default policy:** rollback ≤2/hr, restart ≤5/hr, scale ≤3/hr, key-rotate ≤1/24h, drain ≤3/hr, archive_flake ≤10/hr.

- [ ] **Step 1: Migration**

```python
await conn.execute("""
    CREATE TABLE IF NOT EXISTS action_cooldowns (
        mission_id INTEGER NOT NULL,
        verb TEXT NOT NULL,
        invoked_at TEXT NOT NULL,
        outcome TEXT
    )
""")
await conn.execute("CREATE INDEX IF NOT EXISTS idx_cooldown_lookup ON action_cooldowns(mission_id, verb, invoked_at)")
```

- [ ] **Step 2: Test**

```python
@pytest.mark.asyncio
async def test_rollback_blocked_after_2_per_hour(setup_db):
    from src.ops.action_cooldowns import check, record
    for _ in range(2):
        await check(mission_id=1, verb="rollback_to_last_green")  # passes
        await record(mission_id=1, verb="rollback_to_last_green", outcome="ok")
    blocked = await check(mission_id=1, verb="rollback_to_last_green")
    assert blocked is False  # 3rd in 1 hour
```

- [ ] **Step 3: Implement**

```python
# src/ops/action_cooldowns.py
DEFAULT_POLICY = {
    "rollback_to_last_green": {"max_per_hour": 2, "max_per_day": 8},
    "restart_service": {"max_per_hour": 5},
    "scale_up": {"max_per_hour": 3},
    "scale_down": {"max_per_hour": 3},
    "drain_traffic": {"max_per_hour": 3},
    "rotate_failed_key": {"max_per_day": 1},
    "archive_flake_test": {"max_per_hour": 10},
}

async def check(mission_id: int, verb: str) -> bool:
    policy = DEFAULT_POLICY.get(verb, {"max_per_hour": 999})
    async with get_db() as conn:
        if "max_per_hour" in policy:
            async with conn.execute(
                "SELECT COUNT(*) FROM action_cooldowns WHERE mission_id=? AND verb=? "
                "AND invoked_at >= datetime('now', '-1 hour')",
                (mission_id, verb),
            ) as cur:
                (n,) = await cur.fetchone()
            if n >= policy["max_per_hour"]:
                return False
        if "max_per_day" in policy:
            async with conn.execute(
                "SELECT COUNT(*) FROM action_cooldowns WHERE mission_id=? AND verb=? "
                "AND invoked_at >= datetime('now', '-1 day')",
                (mission_id, verb),
            ) as cur:
                (n,) = await cur.fetchone()
            if n >= policy["max_per_day"]:
                return False
    return True

async def record(mission_id: int, verb: str, outcome: str) -> None:
    async with get_db() as conn:
        await conn.execute(
            "INSERT INTO action_cooldowns (mission_id, verb, invoked_at, outcome) "
            "VALUES (?, ?, datetime('now'), ?)",
            (mission_id, verb, outcome),
        )
        await conn.commit()
```

- [ ] **Step 4: Wire into `oncall_action` executor pre-execute**

```python
# packages/mr_roboto/src/mr_roboto/executors/oncall_action.py
from src.ops.action_cooldowns import check, record

async def run(task) -> dict:
    verb = task.context["verb"]
    mission_id = task.mission_id
    if not await check(mission_id, verb):
        return {"status": "blocked_by_cooldown", "verb": verb}
    # delegate to specific verb handler (rollback/restart/scale/etc.)
    outcome = await _execute_verb(verb, task.context.get("params", {}))
    await record(mission_id, verb, outcome["status"])
    return outcome
```

- [ ] **Step 5: Run, expect PASS**

- [ ] **Step 6: Commit**

```bash
git commit -am "feat(z8,t4b): action_cooldowns table + Mr. Roboto pre-execute enforcement (rollback 2/hr, key-rotate 1/24h)"
```

---

### Task T4C — Playbook recipes + phase 13 generator

**Files:**
- Create: 6 starter playbook recipes under `recipes/incident_playbook_*_v1/`
- Modify: `src/workflows/i2p/i2p_v3.json` (phase 13 emits incident_playbooks artifact + generator step)
- Test: `tests/recipes/test_incident_playbook_match.py`

**Starter playbooks v1:**
1. `incident_playbook_db_disk_full_v1` — DB disk >85% → archive_old_rows + alert founder
2. `incident_playbook_payment_provider_down_v1` — Stripe down → enable fallback_queue + email_user
3. `incident_playbook_auth_provider_down_v1` — auth provider down → maintenance_mode_on + status_page_update
4. `incident_playbook_cert_expiring_v1` — cert <7d → renew via certbot (acme.sh fallback)
5. `incident_playbook_error_spike_v1` — Sentry events >100/5min → rollback_to_last_green (cooldown-gated)
6. `incident_playbook_uptime_drop_v1` — Better Stack down → restart_service + escalate if no recovery in 5min

Each recipe schema (extends `src/infra/recipes.py` Recipe with `decision_tree` + `action_sequence`):

```yaml
# recipes/incident_playbook_db_disk_full_v1/recipe.yaml
id: incident_playbook_db_disk_full_v1
kind: incident_playbook
requires:
  tech_stack: [postgres, mysql, sqlite]
  runtime_state:
    - condition: db.disk_used_pct > 85
match:
  alerts:
    - integration: betterstack
      event: db_disk_alert
action_sequence:
  - verb: archive_old_rows
    params: {older_than_days: 90}
    reversibility: partial
  - verb: notify_founder
    params: {summary: "DB disk at {disk_used_pct}%, archived rows older than 90d"}
    reversibility: full
on_failure:
  - verb: escalate_to_founder
    severity: critical
```

- [ ] **Step 1: Write match test**

```python
def test_playbook_match_db_disk_full():
    alert = {"integration": "betterstack", "event": "db_disk_alert", "data": {"disk_used_pct": 87}}
    runtime_state = {"db": {"disk_used_pct": 87}}
    pb = match_playbook(alert, runtime_state)
    assert pb.id == "incident_playbook_db_disk_full_v1"
```

- [ ] **Step 2: Implement `match_playbook` in `src/ops/playbooks.py`**

```python
async def match_playbook(alert: dict, runtime_state: dict) -> Recipe | None:
    candidates = list_recipes(kind="incident_playbook")
    for r in candidates:
        if not _alert_matches(r, alert):
            continue
        if not _runtime_state_matches(r, runtime_state):
            continue
        return r
    return None
```

- [ ] **Step 3: Add 6 recipe.yaml files**

(See "Starter playbooks v1" list above.)

- [ ] **Step 4: Modify i2p_v3.json phase 13**

Add step `13.4 generate_incident_playbooks`:
```json
{
  "id": "13.4",
  "agent": "mechanical",
  "executor": "generate_playbooks",
  "produces": ["incident_playbooks"],
  "instruction": "From spec.architecture + spec.tech_stack, instantiate matching incident_playbook_*_v1 recipes; emit per-mission playbook set as artifact."
}
```

- [ ] **Step 5: Run tests**

- [ ] **Step 6: Commit**

```bash
git commit -am "feat(z8,t4c): 6 incident_playbook recipes + phase 13.4 generator step"
```

---

### Task T4D — `/ops_log` + escalation_policy table

**Files:**
- Create: `src/ops/escalation_policy.py`
- Modify: `src/app/telegram_bot.py` (`/ops_log <mission_id>`)
- Modify: `src/infra/db.py` (escalation_policy table)
- Test: `tests/app/test_ops_log_cmd.py`

`escalation_policy` table: `(mission_id, quiet_hours_start, quiet_hours_end, tier1_channel, tier2_channel, tier3_channel, tz)`. Default: tier1=telegram, tier2=telegram + sms (24h), tier3=sms regardless.

- [ ] **Step 1: Migration + default-policy seed**

```python
await conn.execute("""
    CREATE TABLE IF NOT EXISTS escalation_policy (
        mission_id INTEGER PRIMARY KEY,
        quiet_hours_start TEXT,
        quiet_hours_end TEXT,
        tier1_channel TEXT NOT NULL DEFAULT 'telegram',
        tier2_channel TEXT NOT NULL DEFAULT 'telegram',
        tier3_channel TEXT NOT NULL DEFAULT 'sms',
        tz TEXT DEFAULT 'UTC'
    )
""")
```

- [ ] **Step 2: Test `/ops_log`**

```python
@pytest.mark.asyncio
async def test_ops_log_renders_last_actions(setup_db):
    # seed registry_events with mission_id=5
    async with get_db() as conn:
        for verb in ["restart_service", "rollback_to_last_green"]:
            await conn.execute("INSERT INTO registry_events (mission_id, verb, reversibility, outcome, created_at) VALUES (?,?,?,?, datetime('now'))",
                              (5, verb, "full", "ok"))
        await conn.commit()
    bot = TelegramInterface(token="x")
    update = MagicMock(); update.message.reply_text = AsyncMock()
    context = MagicMock(); context.args = ["5"]
    await bot.cmd_ops_log(update, context)
    args = update.message.reply_text.call_args.args[0]
    assert "restart_service" in args
    assert "rollback_to_last_green" in args
```

- [ ] **Step 3: Implement `/ops_log`**

```python
async def cmd_ops_log(self, update, context):
    if not context.args:
        await self._reply(update, "Usage: /ops_log <mission_id>")
        return
    mid = int(context.args[0])
    async with get_db() as conn:
        async with conn.execute(
            "SELECT verb, reversibility, outcome, created_at FROM registry_events "
            "WHERE mission_id=? ORDER BY id DESC LIMIT 20",
            (mid,),
        ) as cur:
            rows = await cur.fetchall()
    lines = [f"`{v}` · {r} · {o} · {t}" for v, r, o, t in rows]
    body = f"Ops log mission {mid} (last 20):\n" + "\n".join(lines) if rows else f"No actions for mission {mid}"
    await self._reply(update, body, parse_mode="Markdown")
```

- [ ] **Step 4: escalation_policy queries in oncall_agent's escalate_to_founder tool**

```python
# packages/mr_roboto/src/mr_roboto/executors/escalate_to_founder.py
async def run(task):
    severity = task.context["severity"]
    mid = task.mission_id
    policy = await load_policy(mid)
    channel = policy[f"tier{_tier_of(severity)}_channel"]
    is_quiet = _in_quiet_hours(policy)
    if is_quiet and severity != "critical":
        channel = "telegram_log_only"
    await dispatch(channel, task.context["summary"], mission_id=mid)
```

- [ ] **Step 5: Run, expect PASS**

- [ ] **Step 6: Commit + tag**

```bash
git commit -am "feat(z8,t4d): /ops_log Telegram cmd + escalation_policy table + quiet-hours-aware channel routing"
git tag z8-t4-shipped
```

---

## Tier 5 — Cron missions + support tier-1 + cost/perf/security

**Goal:** Ship ongoing missions for backup verify, dep hygiene, CVE scan, secret scan, cost monitor, perf regression, support tier-1, Twilio SMS escalation.

**Acceptance:** Each cron mission runs on schedule; broken backup writes restore-fail digest; ticket below confidence threshold escalates; perf regression on staging blocks promote.

**Parallelization:** T5A–T5G can dispatch in parallel after T5 prep (`cron_scheduler.py`). T5H test sweep last.

### Task T5-prep — cron scheduler (replaces vestigial scheduled_tasks)

**Files:**
- Create: `packages/general_beckman/src/general_beckman/cron.py`
- Modify: `src/core/orchestrator.py` (boot cron scheduler)
- Test: `tests/general_beckman/test_cron.py`

**Approach:** Cron config lives on `missions.cursor` for ongoing missions; orchestrator boot reads each ongoing mission's cursor and arms apscheduler jobs that call `enqueue(task_type=..., lane=ongoing)` at the configured interval.

- [ ] **Step 1: Implement scheduler**

```python
# packages/general_beckman/src/general_beckman/cron.py
import asyncio
from datetime import datetime, timedelta
from general_beckman import enqueue
from general_beckman.lanes import LANE_ONGOING

_TASKS = {}  # mission_id → asyncio.Task

async def arm(mission_id: int, task_type: str, interval_seconds: int) -> None:
    if mission_id in _TASKS:
        _TASKS[mission_id].cancel()
    _TASKS[mission_id] = asyncio.create_task(_loop(mission_id, task_type, interval_seconds))

async def disarm(mission_id: int) -> None:
    t = _TASKS.pop(mission_id, None)
    if t:
        t.cancel()

async def _loop(mission_id: int, task_type: str, interval_seconds: int):
    while True:
        try:
            await enqueue(mission_id=mission_id, task_type=task_type, lane=LANE_ONGOING)
        except Exception as e:
            logger.exception(f"cron enqueue failed mid={mission_id}: {e}")
        await asyncio.sleep(interval_seconds)
```

- [ ] **Step 2: Orchestrator boot wires resumption → cron arm**

In `_rebind_ongoing(mission)` from T1C, read `mission.cursor.cron` and call `arm()` for each registered schedule.

- [ ] **Step 3: Test (use fake clock)**

```python
@pytest.mark.asyncio
async def test_cron_arm_enqueues_on_interval(setup_db, monkeypatch):
    fake_now = MockClock()
    monkeypatch.setattr("asyncio.sleep", fake_now.fast_sleep)
    await arm(mission_id=1, task_type="cron_backup_verify", interval_seconds=3600)
    await fake_now.advance(7200)
    async with get_db() as conn:
        async with conn.execute("SELECT COUNT(*) FROM tasks WHERE task_type='cron_backup_verify'") as cur:
            (n,) = await cur.fetchone()
    assert n >= 2
```

- [ ] **Step 4: Commit**

```bash
git commit -am "feat(z8,t5-prep): cron scheduler for ongoing missions (replaces vestigial scheduled_tasks)"
```

---

### Task T5A — backup_verify cron + recipes

**Files:**
- Create: `recipes/backup_verify_postgres_v1/recipe.yaml`
- Create: `recipes/backup_verify_sqlite_v1/recipe.yaml`
- Create: `packages/mr_roboto/src/mr_roboto/executors/backup_verify.py`
- Test: `tests/ops/test_backup_verify.py`

Executor: spin sandbox container, restore from latest backup, run smoke tests, alert on fail.

- [ ] **Step 1: Recipe yamls** (postgres: pg_restore into ephemeral DB + run smoke SQL; sqlite: copy backup file + SELECT smoke).
- [ ] **Step 2: Executor with subprocess + dry-run flag for tests**
- [ ] **Step 3: Test using sqlite recipe (no docker required)**
- [ ] **Step 4: Commit**

```bash
git commit -am "feat(z8,t5a): backup_verify cron + postgres/sqlite recipes (weekly schedule)"
```

---

### Task T5B — dependency_hygiene cron + recipes

**Files:**
- Create: `recipes/dependency_hygiene_python_v1/recipe.yaml`
- Create: `recipes/dependency_hygiene_node_v1/recipe.yaml`
- Create: `packages/mr_roboto/src/mr_roboto/executors/dependency_scan.py`
- Test: `tests/ops/test_dep_hygiene.py`

Executor wraps `pip-audit` (python) or `npm audit` (node). Auto-merges patch versions if CI green; surfaces minor/major via founder_actions.

- [ ] Recipes + executor + test + commit

```bash
git commit -am "feat(z8,t5b): dependency_hygiene cron + python/node recipes (auto-merge patch, founder for minor/major)"
```

---

### Task T5C — CVE + secret scan cron

**Files:**
- Create: `recipes/cve_scan_python_v1/recipe.yaml`, `cve_scan_node_v1`, `cve_scan_docker_v1`
- Create: `packages/mr_roboto/src/mr_roboto/executors/cve_scan.py` (OSV.dev API), `secret_scan.py` (gitleaks subprocess)
- Create: `src/integrations/configs/osv.json`
- Test: `tests/ops/test_cve_scan.py`

- [ ] Implement + test + commit

```bash
git commit -am "feat(z8,t5c): cve_scan + secret_scan cron (OSV.dev + gitleaks) + recipes"
```

---

### Task T5D — cost monitor extension + slope anomaly

**Files:**
- Create: `src/ops/cost_anomaly.py`
- Create: `recipes/cost_monitor_stripe_v1/recipe.yaml`, `cost_monitor_vercel_v1`, `cost_monitor_aws_v1`
- Create: `packages/mr_roboto/src/mr_roboto/executors/cost_pull.py`
- Modify: `src/infra/alerting.py` (add cost-slope rule)
- Test: `tests/ops/test_cost_anomaly.py`

```python
# src/ops/cost_anomaly.py
import statistics
async def is_anomaly(integration_id: str, today_usd: float, history_14d: list[float]) -> bool:
    if len(history_14d) < 7:
        return False
    mean = statistics.mean(history_14d)
    stdev = statistics.stdev(history_14d) or 0.01
    z = (today_usd - mean) / stdev
    return z > 2.5  # 2.5σ
```

- [ ] Implement + test + commit

```bash
git commit -am "feat(z8,t5d): cost slope anomaly detector + per-vendor recipes (stripe/vercel/aws)"
```

---

### Task T5E — support_tier1 ongoing mission (RAG)

**Files:**
- Create: `src/agents/configs/support_tier1.yaml`
- Create: `src/ops/support_rag.py`
- Modify: `src/infra/db.py` (tickets table)
- Modify: `src/app/telegram_bot.py` (ticket inlet)
- Test: `tests/ops/test_support_rag.py`

`tickets` table: `(id, mission_id, user_id, question, answer, confidence, status, escalated_to_founder, created_at)`.

RAG: embed question (multilingual-e5-base, existing) → top-3 docs from ChromaDB `support_docs` collection → LLM compose with citations. If confidence <0.7 OR sentiment="angry" → escalate via founder_actions kind=`support_escalation`.

Weekly cron: scan resolved tickets → cluster themes via embeddings → propose FAQ additions → founder approves → re-index `support_docs`.

- [ ] Implement + test + commit (multi-step)

```bash
git commit -am "feat(z8,t5e): support_tier1 ongoing mission with ChromaDB RAG + confidence-based escalation"
```

---

### Task T5F — perf regression + perf_baselines

**Files:**
- Create: `src/ops/perf_baselines.py`
- Create: `recipes/synthetic_check_lighthouse_v1/recipe.yaml`, `synthetic_check_k6_v1`
- Create: `packages/mr_roboto/src/mr_roboto/executors/synthetic_check.py`
- Modify: `src/infra/db.py` (perf_baselines table)
- Test: `tests/ops/test_perf_regression.py`

`perf_baselines`: `(mission_id, release_tag, metric, p50, p95, p99, recorded_at)`.

Synthetic check runs post-deploy; if regression >10% vs last green baseline, mark mission task as `regression_detected`; offering rollback verb via oncall_agent.

- [ ] Implement + test + commit

```bash
git commit -am "feat(z8,t5f): perf_baselines + synthetic check recipes (lighthouse/k6) + regression diff"
```

---

### Task T5G — Twilio SMS for tier-3

**Files:**
- Create: `packages/mr_roboto/src/mr_roboto/executors/sms_send.py`
- Modify: `src/ops/escalation_policy.py` (sms channel handler)
- Test: `tests/ops/test_sms_escalation.py`

- [ ] Implement + test + commit

```bash
git commit -am "feat(z8,t5g): Twilio SMS escalation for tier-3 incidents"
```

---

### Task T5H — Z8 full integration sweep + tag

- [ ] **Step 1: Run full Z8 test set**

```bash
timeout 300 python -m pytest tests/ops/ tests/general_beckman/test_lanes.py tests/general_beckman/test_cron.py tests/app/test_webhook_listener.py tests/app/test_stop_ops_command.py tests/app/test_ops_log_cmd.py tests/integration/test_t1_lifecycle_e2e.py tests/recipes/test_incident_playbook_match.py -v
```

- [ ] **Step 2: Mission run check** — start KutAI; send a Sentry test webhook; verify `alert_triage` enqueued, classified, escalated correctly; backup_verify cron arms on resumption; `/ops_log` renders.

- [ ] **Step 3: Update i2p_v3.json step 13.3** — replace NEEDS-REAL-TOOLS with `monitoring_kit_*_v1` recipe wiring.

- [ ] **Step 4: Append `## Updates` to `docs/i2p-evolution/08-operations-v2.md`**

```markdown
- 2026-MM-DD — Z8 T1-T5 shipped. Hinge points landed: H1 lifecycle (mission.kind/lifecycle_state/cursor + Beckman ongoing-lane + orchestrator resumption), H2 webhook spine (FastAPI listener + signing + dedup + product_id routing), H3 admission gate wired into Beckman.next_task() + action_done unblock, H4 ops recipes catalog (18 recipes: monitoring/backup/cve/cost/playbook/synthetic). 7 starter playbooks. Support tier-1 with ChromaDB RAG. Twilio SMS escalation. Commits z8-t1-shipped through z8-t5-shipped.
```

- [ ] **Step 5: Final commit + tag**

```bash
git commit -am "docs(z8): mark Z8 complete in 08-operations-v2.md updates log"
git tag z8-complete
```

---

## Risks

| Risk | Mitigation |
|---|---|
| **Webhook listener crashes silently** in embedded mode → missed alerts | Add health endpoint `/webhook/__health`; Yaşar Usta heartbeat polls it; restart wrapper if down >30s. T5H smoke test. |
| **action_cooldowns hides genuine emergency** (e.g., 3rd rollback IS necessary) | `/force_action <mission_id> <verb>` Telegram cmd bypasses cooldown with founder consent. Deferred to T5H follow-up. |
| **Severity classifier under-fires on novel events** | LLM fallback when rule returns `uncertain`; log all `uncertain` events to `model_pick_log` for offline weight tuning. |
| **Z0 takes ownership of product_id later** with different shape | T1A column is nullable, accepts string. Z0 v2 can rename or relocate; one migration ALTER away. Flag in T1A note. |
| **ChromaDB embedding drift** for support_docs | Reuse existing multilingual-e5-base; T5E test seeds + queries against fixed snapshot. |
| **Twilio costs run away** on incident storm | Daily cap per mission (env: `TWILIO_DAILY_CAP_USD`); when exceeded, escalation downgrades to telegram_log_only. |
| **Backup verify needs prod credentials** to read real backups | T5A v1 reads sandbox/staging backups only; prod-backup verify needs founder_action approval per run. |
| **Recipe sprawl** — 18 new yamls hard to maintain | All under `recipes/`, tested by existing `tests/recipes/` infra (Z2 T5A pattern); shared semgrep schema check (Z2 T3C). |

---

## Cross-zone coordination

- **Z0:** `mission.product_id` ownership — T1A places nullable placeholder; if Z0 v2 lands a typed schema, run one ALTER + backfill.
- **Z6 (06-real-world-bridge-v2):** depends on `vendor_call` executor (shipped 2026-05-11). T2/T3 confirm credential schema honored; no new dependencies.
- **Z10 (cross-cutting):** reversibility tagging applies to all new oncall verbs. T4B reversibility tags written into `packages/mr_roboto/src/mr_roboto/reversibility.py`. action_cooldowns is orthogonal — does NOT replace reversibility.
- **Z3 (review-density):** still running (parallel session). T4 on-call agent severity classification benefits from Z3 multi-pass review when alert payloads are ambiguous; not a hard blocker, classifier ships v1 rule-based.
- **Z9 (growth):** Posthog adapter lands in T5 monitoring_kit recipes; cost monitor (T5D) feeds Z9 unit-economics digest. Z9 plan should reference T5D outputs.
- **Z7 (humanish):** support_tier1 (T5E) escalations and investor-update digests draw from Z8 ops data. Z7 doc references T5E `tickets` table + escalation_policy.

---

## Self-review

**Spec coverage:** All 4 hinges (H1–H4) + all 10 v1 gaps (A–J) → mapped to T1–T5 tasks. Open questions from v2 §6 all resolved or tracked in Risks.

**Placeholder scan:** Zero `TBD/TODO/later/fill in`. All code blocks present. T5A–T5G summarized in 1-task bullets (full sub-steps would 3× plan length; tier-prep + recipe schema in T4C show the pattern).

**Type consistency:** `lane`, `mission_id`, `lifecycle_state`, `webhook_secret`, `verb` names match across tasks. `check_or_park()` (T2A) referenced consistently.

**Known abbreviation:** T5A–T5G abbreviated to avoid duplicate scaffolding. Each follows the pattern: yaml recipe + executor + test + commit. If executing subagent-driven, dispatch T5A–T5G as separate agents with the pattern from T5-prep + T4C as reference.

---

## Execution handoff

**Plan complete and saved to `docs/plans/2026-05-12-z8-operations.md`.**

**Two execution options:**

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch with checkpoints.

Per repo memory `feedback_subagents_always`, subagent-driven is the default for this codebase. Tier ordering:
- T1A → T1B → T1C → T1D → T1E (sequential — T1A migration blocks T1B)
- T2A → T2B (sequential)
- T3A → T3B → T3C → T3D → T3E (T3A blocks T3B/D; T3C parallel-safe with T3B/D after T3A)
- T4A + T4B (parallel) → T4C → T4D
- T5-prep → T5A/B/C/D/E/F/G (parallel after prep) → T5H

T1 + T2 can run in parallel (independent). T3/T4/T5 must wait on T1.
