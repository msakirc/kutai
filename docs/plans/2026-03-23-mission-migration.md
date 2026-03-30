# Goals → Missions Migration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace three confusing concepts (Goal, Project, Product) with one unified concept: **Mission**. Missions consist of tasks. Missions may optionally follow a workflow. Project metadata (repo path, language, framework) moves into `mission.context`.

**Architecture:** SQLite table rename `goals` → `missions` with backward-compatible migration. Kill `projects` table — absorb fields into mission context JSON. Kill `/product` as separate concept — becomes `/mission --workflow`. All `goal_id` refs become `mission_id`. Telegram understands `/mission` and `/mish`.

**Tech Stack:** Python 3.10+, aiosqlite, python-telegram-bot

---

### Task 1: DB Schema Migration — Rename `goals` → `missions`

**Files:**
- Modify: `src/infra/db.py` (lines 63-75 schema, lines 365-420 functions, all `goal_id` refs)

**Step 1: Add migration in `init_db()` — rename table + column**

In `init_db()`, BEFORE the `CREATE TABLE IF NOT EXISTS` blocks, add migration:

```python
# Migration: goals → missions
try:
    cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='goals'")
    if await cursor.fetchone():
        await db.execute("ALTER TABLE goals RENAME TO missions")
        await db.commit()
        logger.info("Migrated: goals → missions")
except Exception as e:
    logger.debug(f"goals→missions migration skipped: {e}")
```

**Step 2: Rename the CREATE TABLE block**

Change `CREATE TABLE IF NOT EXISTS goals` → `CREATE TABLE IF NOT EXISTS missions` (same columns).

**Step 3: Rename ALL `goal_id` columns in other tables to `mission_id`**

For each table that has `goal_id`:
- `tasks.goal_id` → `tasks.mission_id`
- `memory.goal_id` → `memory.mission_id`
- `blackboards.goal_id` → `blackboards.mission_id`
- `file_locks.goal_id` → `file_locks.mission_id`
- `approval_requests.goal_id` → `approval_requests.mission_id`
- `workspace_snapshots.goal_id` → `workspace_snapshots.mission_id`
- `workflow_checkpoints.goal_id` → `workflow_checkpoints.mission_id`

SQLite doesn't support `ALTER TABLE RENAME COLUMN` before 3.25.0, so add migration:

```python
# Migration: goal_id → mission_id in all tables
_GOAL_ID_TABLES = ["tasks", "memory", "file_locks", "approval_requests",
                    "workspace_snapshots", "workflow_checkpoints"]
for tbl in _GOAL_ID_TABLES:
    try:
        await db.execute(f"ALTER TABLE {tbl} RENAME COLUMN goal_id TO mission_id")
    except Exception:
        pass  # Column already renamed or doesn't exist
# blackboards primary key rename
try:
    await db.execute("ALTER TABLE blackboards RENAME COLUMN goal_id TO mission_id")
except Exception:
    pass
await db.commit()
```

**Step 4: Update CREATE TABLE statements**

Replace every `goal_id` with `mission_id` in all CREATE TABLE IF NOT EXISTS blocks. Replace `REFERENCES goals(id)` with `REFERENCES missions(id)`.

**Step 5: Rename all DB API functions**

| Old | New |
|-----|-----|
| `add_goal(title, description, ...)` | `add_mission(title, description, ...)` |
| `get_goal(goal_id)` | `get_mission(mission_id)` |
| `get_active_goals()` | `get_active_missions()` |
| `update_goal(goal_id, **kwargs)` | `update_mission(mission_id, **kwargs)` |
| `_GOAL_COLUMNS` | `_MISSION_COLUMNS` |
| `get_tasks_for_goal(goal_id)` | `get_tasks_for_mission(mission_id)` |
| `get_task_tree(goal_id)` | `get_task_tree(mission_id)` |
| `propagate_skips(goal_id)` | `propagate_skips(mission_id)` |
| `release_goal_locks(goal_id)` | `release_mission_locks(mission_id)` |
| `get_goal_locks(goal_id)` | `get_mission_locks(mission_id)` |
| `get_latest_snapshot(goal_id)` | `get_latest_snapshot(mission_id)` |
| `get_goal_total_cost(goal_id)` | `get_mission_total_cost(mission_id)` |

Keep backward-compat aliases:
```python
# Backward compatibility
add_goal = add_mission
get_goal = get_mission
get_active_goals = get_active_missions
update_goal = update_mission
get_tasks_for_goal = get_tasks_for_mission
```

**Step 6: Update all `add_task()` parameter**

Change `goal_id=None` → `mission_id=None` in `add_task()`. Update the SQL INSERT to use `mission_id`. Keep alias `goal_id` accepted:

```python
async def add_task(title, description, mission_id=None, parent_task_id=None,
                   agent_type="executor", tier="auto", priority=5,
                   requires_approval=False, depends_on=None, context=None,
                   goal_id=None):  # backward compat
    mission_id = mission_id or goal_id
```

**Step 7: Commit**

```
git commit -m "refactor(db): rename goals→missions, goal_id→mission_id with migration"
```

---

### Task 2: Kill `projects` Table — Absorb Into Mission Context

**Files:**
- Modify: `src/infra/projects.py` → convert to thin wrapper over mission context
- Modify: `src/infra/db.py` — remove `project_id` column from missions

**Step 1: Add `workflow` and `repo_path` columns to missions**

```python
# In init_db migration block:
for col, default in [("workflow", ""), ("repo_path", ""), ("language", ""), ("framework", "")]:
    try:
        await db.execute(f"ALTER TABLE missions ADD COLUMN {col} TEXT DEFAULT '{default}'")
    except Exception:
        pass

# Migrate project data into missions
try:
    await db.execute("""
        UPDATE missions SET
            repo_path = (SELECT p.repo_path FROM projects p WHERE p.id = missions.project_id),
            language = (SELECT p.language FROM projects p WHERE p.id = missions.project_id),
            framework = (SELECT p.framework FROM projects p WHERE p.id = missions.project_id)
        WHERE project_id IS NOT NULL
    """)
    await db.commit()
except Exception:
    pass
```

**Step 2: Update `add_mission()` to accept project fields**

```python
async def add_mission(title, description, priority=5, context=None,
                      workflow=None, repo_path=None, language=None, framework=None):
    db = await get_db()
    cursor = await db.execute(
        """INSERT INTO missions (title, description, priority, context, workflow, repo_path, language, framework)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (title, description, priority, json.dumps(context or {}),
         workflow or "", repo_path or "", language or "", framework or "")
    )
    await db.commit()
    return cursor.lastrowid
```

**Step 3: Rewrite `src/infra/projects.py` as thin helpers**

Replace the entire file with helpers that operate on mission fields:

```python
# infra/projects.py — Project context helpers (operates on missions table)

async def set_mission_project(mission_id, repo_path, language="", framework=""):
    """Attach project/codebase context to a mission."""
    from .db import update_mission
    await update_mission(mission_id, repo_path=repo_path, language=language, framework=framework)

async def get_missions_by_repo(repo_path):
    """Find missions linked to a specific codebase."""
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM missions WHERE repo_path = ? ORDER BY created_at DESC", (repo_path,)
    )
    return [dict(row) for row in await cursor.fetchall()]
```

**Step 4: Commit**

```
git commit -m "refactor: absorb projects table into mission columns"
```

---

### Task 3: Kill `/product` — Unify Into `/mission`

**Files:**
- Modify: `src/app/telegram_bot.py` — rename commands, add workflow flag
- Modify: `src/workflows/engine/runner.py` — use `mission_id`

**Step 1: Remove `/product` handler, update `/goal` → `/mission`**

Rename in `_setup_handlers()`:
```python
self.app.add_handler(CommandHandler("mission", self.cmd_mission))
self.app.add_handler(CommandHandler("mish", self.cmd_mission))      # abbreviation
self.app.add_handler(CommandHandler("missions", self.cmd_missions))
# Remove: goal, goalforce, goals, project, projects, product
```

**Step 2: Write unified `cmd_mission()` handler**

```python
async def cmd_mission(self, update, context):
    """Create a mission. Use --workflow to start a full workflow."""
    args = context.args
    if not args:
        await update.message.reply_text(
            "Usage: /mission <description>\n"
            "       /mission --workflow <description>\n"
            "       /mish <description> (shorthand)\n\n"
            "Examples:\n"
            "  /mission Fix the login page bug\n"
            "  /mission --workflow Build an inventory management app"
        )
        return

    # Check for --workflow flag
    workflow = None
    text_args = list(args)
    if "--workflow" in text_args:
        text_args.remove("--workflow")
        workflow = "i2p_v2"

    description = " ".join(text_args)
    # ... rest of goal creation logic, using add_mission(workflow=workflow)
```

**Step 3: Rename `cmd_list_goals` → `cmd_missions`**

Show workflow badge for workflow missions:
```
📋 Active Missions:
  🎯 #42 Fix login bug (3 tasks)
  🔄 #43 Build inventory app [workflow: i2p] (47/130 tasks)
```

**Step 4: Update menu system**

Replace category "🎯 Goals & Tasks" → "🎯 Missions" and "🏗️ Projects" gets absorbed:
```python
("🎯 Missions", "missions", [
    ("🎯 New Mission", "mission", True, "Describe your mission:"),
    ("🔄 New Workflow", "mission_wf", True, "Describe the product/workflow idea:"),
    ("📋 List Missions", "missions", False, None),
    ("📬 Queue", "queue", False, None),
    ("🚫 Cancel", "cancel", True, "Which mission or task ID to cancel?"),
    ("🔢 Priority", "priority", True, "Enter: <id> <priority 1-10>"),
    ("⏸️ Pause", "pause", True, "Which mission ID to pause?"),
    ("▶️ Resume", "resume", True, "Which mission ID to resume?"),
    ("🌳 Graph", "graph", True, "Which mission ID to graph?"),
    ("📋 WF Status", "wfstatus", True, "Which mission ID for workflow status?"),
]),
```

Kill "🏗️ Projects" category entirely. Move workspace to Monitoring.

**Step 5: Commit**

```
git commit -m "refactor(telegram): unify /goal /project /product into /mission"
```

---

### Task 4: Update Orchestrator & Core Modules

**Files:**
- Modify: `src/core/orchestrator.py` — `plan_goal()` → `plan_mission()`
- Modify: `src/collaboration/blackboard.py` — `goal_id` → `mission_id`
- Modify: `src/tools/workspace.py` — `goal_workspace` → `mission_workspace`
- Modify: `src/tools/git_ops.py` — `create_goal_branch` → `create_mission_branch`
- Modify: `src/infra/dead_letter.py` — `_check_goal_health` → `_check_mission_health`
- Modify: `src/context/assembler.py` — goal_id → mission_id
- Modify: `src/workflows/engine/runner.py` — goal_id → mission_id
- Modify: `src/workflows/engine/hooks.py` — goal_id → mission_id
- Modify: `src/workflows/engine/expander.py` — goal_id → mission_id
- Modify: `src/workflows/engine/artifacts.py` — goal_id → mission_id
- Modify: `src/workflows/engine/status.py` — if exists
- Modify: `src/memory/self_improvement.py` — if references goals

**Step 1: Orchestrator rename**

Global find-replace in orchestrator.py:
- `plan_goal` → `plan_mission`
- `goal_id` → `mission_id`
- `get_active_goals` → `get_active_missions`
- `get_tasks_for_goal` → `get_tasks_for_mission`
- `update_goal` → `update_mission`
- `get_goal` → `get_mission`
- Workspace: `get_goal_workspace` → `get_mission_workspace`
- Git: `create_goal_branch` → `create_mission_branch`

**Step 2: Blackboard rename**

In `blackboard.py`, rename all parameter names `goal_id` → `mission_id`. The actual behavior is identical — just parameter names and log messages.

**Step 3: Workspace rename**

In `workspace.py`:
- `get_goal_workspace(goal_id)` → `get_mission_workspace(mission_id)` — returns `workspace/mission_{id}/`
- `list_goal_workspaces()` → `list_mission_workspaces()`
- `cleanup_goal_workspace()` → `cleanup_mission_workspace()`

Add compat aliases.

**Step 4: Git ops rename**

In `git_ops.py`:
- `create_goal_branch(goal_id, title)` → `create_mission_branch(mission_id, title)`
- Branch pattern: `ai/goal-{id}` → `ai/mission-{id}`

**Step 5: Workflow engine rename**

In `runner.py`, `hooks.py`, `expander.py`, `artifacts.py`:
- All `goal_id` params → `mission_id`
- Log messages updated

**Step 6: Dead letter + context assembler**

- `_check_goal_health()` → `_check_mission_health()`
- Context references updated

**Step 7: Commit**

```
git commit -m "refactor: rename goal_id→mission_id across orchestrator, blackboard, workspace, workflows"
```

---

### Task 5: Update API Layer

**Files:**
- Modify: `src/app/api.py` — rename endpoints and models

**Step 1: Rename endpoints**

```python
class MissionCreate(BaseModel):
    title: str
    description: str
    priority: int = 5
    workflow: str | None = None
    repo_path: str | None = None

@app.post("/missions")
async def create_mission(m: MissionCreate):
    mission_id = await add_mission(...)
    return {"mission_id": mission_id}

@app.get("/missions/{mission_id}")
async def get_mission_endpoint(mission_id: int):
    ...

@app.get("/missions")
async def list_missions():
    ...
```

Keep `/goals` as redirect alias for backward compat.

**Step 2: Commit**

```
git commit -m "refactor(api): rename /goals→/missions endpoints"
```

---

### Task 6: Update Message Classifier & Tests

**Files:**
- Modify: `src/app/telegram_bot.py` — classifier prompt + keyword rules
- Modify: `src/core/task_classifier.py` — if references goals
- Modify: `tests/` — update any goal references

**Step 1: Update MESSAGE_CLASSIFIER_PROMPT**

Replace `"goal"` category with `"mission"`:
```
- "mission": complex multi-step project request (was "goal")
```

**Step 2: Update keyword classifier**

Replace goal keywords to return `"mission"` instead of `"goal"`.

**Step 3: Update handle_message routing**

Where it checks `msg_type == "goal"`, change to `msg_type == "mission"`.

**Step 4: Commit**

```
git commit -m "refactor: update classifiers and message routing for mission concept"
```

---

### Task 7: Cleanup & Final Verification

**Files:**
- All modified files
- `docs/plans/` — update plan docs
- `plans/plan_v5.md` — update roadmap

**Step 1: Remove backward compat aliases from db.py**

Once all callers are migrated, remove:
```python
# DELETE THESE:
add_goal = add_mission
get_goal = get_mission
...
```

**Step 2: Search for any remaining "goal" references**

```bash
rg "goal" src/ --type py -l
```

Fix any stragglers.

**Step 3: Syntax verification**

```bash
python -c "import ast; ..."  # for all modified files
```

**Step 4: Final commit**

```
git commit -m "refactor: complete goals→missions migration, remove compat aliases"
```

---

## Migration Summary

| Before | After |
|--------|-------|
| `/goal <desc>` | `/mission <desc>` or `/mish <desc>` |
| `/goals` | `/missions` |
| `/goalforce <desc>` | `/mission <desc>` (refinement is default, `--force` flag to skip) |
| `/project add <path>` | `/mission --repo <path> <desc>` (or set repo on existing) |
| `/projects` | Gone — repo info shown on `/missions` list |
| `/product <idea>` | `/mission --workflow <idea>` |
| `goals` table | `missions` table |
| `goal_id` everywhere | `mission_id` everywhere |
| `projects` table | Gone — `missions.repo_path`, `.language`, `.framework` |
| 3 separate concepts | 1 concept: **Mission** |

## File Change Count

| Category | Files | Effort |
|----------|-------|--------|
| DB schema + migration | 2 | High |
| Telegram commands + menu | 1 | High |
| Core orchestrator | 1 | Medium |
| Blackboard/workspace/git | 3 | Medium |
| Workflow engine | 4 | Medium |
| Projects.py rewrite | 1 | Low |
| API layer | 1 | Low |
| Classifier | 2 | Low |
| Misc (dead_letter, assembler) | 2 | Low |
| **Total** | **~17 files** | |
