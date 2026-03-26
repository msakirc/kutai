# Telegram Bot UX — Redesign Plan

## Current State

### A. Registered Command Handlers (`_setup_handlers`, lines 245–306)

| # | Command | Handler Method | Description |
|---|---------|----------------|-------------|
| 1 | `/start` | `cmd_start` | Show main menu + reply keyboard |
| 2 | `/mission` | `cmd_mission` | Create a new mission (also detects workflow) |
| 3 | `/mish` | `cmd_mission` | Abbreviation for `/mission` |
| 4 | `/missions` | `cmd_missions` | List active missions |
| 5 | `/task` | `cmd_add_task` | Add a standalone task |
| 6 | `/queue` | `cmd_view_queue` | View pending task queue |
| 7 | `/status` | `cmd_status` | System status (counts + cost) |
| 8 | `/digest` | `cmd_digest` | Daily digest |
| 9 | `/debug` | `cmd_debug` | Full system debug dump |
| 10 | `/reset` | `cmd_reset` | Reset stuck/failed tasks |
| 11 | `/resetall` | `cmd_reset_all` | Nuclear wipe |
| 12 | `/cancel` | `cmd_cancel` | Cancel a task + children |
| 13 | `/priority` | `cmd_priority` | Change task priority |
| 14 | `/graph` | `cmd_graph` | Task dependency DAG |
| 15 | `/budget` | `cmd_budget` | View/set daily cost budget |
| 16 | `/modelstats` | `cmd_model_stats` | Model performance stats |
| 17 | `/workspace` | `cmd_workspace` | List mission workspaces |
| 18 | `/progress` | `cmd_progress` | Progress timeline |
| 19 | `/audit` | `cmd_audit` | Audit log for a task |
| 20 | `/metrics` | `cmd_metrics` | System metrics summary |
| 21 | `/replay` | `cmd_replay` | Trace replay for a task |
| 22 | `/ingest` | `cmd_ingest` | Ingest URL/file to knowledge base |
| 23 | `/wfstatus` | `cmd_wfstatus` | Workflow status for a mission |
| 24 | `/resume` | `cmd_resume` | Resume paused/failed workflow |
| 25 | `/pause` | `cmd_pause` | Pause a mission's tasks |
| 26 | `/credential` | `cmd_credential` | Manage stored credentials |
| 27 | `/cost` | `cmd_cost` | Per-mission cost breakdown |
| 28 | `/dlq` | `cmd_dlq` | Dead-letter queue management |
| 29 | `/load` | `cmd_load` | GPU load mode control |
| 30 | `/tune` | `cmd_tune` | Force auto-tuning cycle |
| 31 | `/feedback` | `cmd_feedback` | Rate a completed task |
| 32 | `/improve` | `cmd_improve` | Self-improvement analysis |
| 33 | `/remember` | `cmd_remember` | Store a fact to knowledge base |
| 34 | `/recall` | `cmd_recall` | Search knowledge base |
| 35 | `/autonomy` | `cmd_autonomy` | Set risk autonomy threshold |
| 36 | `/todo` | `cmd_todo` | Add a todo item |
| 37 | `/todos` | `cmd_todos` | List todos |
| 38 | `/cleartodos` | `cmd_cleartodos` | Clear completed todos |
| 39 | `/shop` | `cmd_shop` | Shopping assistant |
| 40 | `/research_product` | `cmd_research_product` | Deep product research |
| 41 | `/price` | `cmd_price` | Quick price check |
| 42 | `/watch` | `cmd_watch` | Price watch setup |
| 43 | `/deals` | `cmd_deals` | Show active deals/watches |
| 44 | `/mystuff` | `cmd_mystuff` | User profile — owned items |
| 45 | `/compare` | `cmd_compare` | Product comparison (X vs Y) |
| 46 | `/kutai_restart` | `cmd_kutai_restart` | Restart KutAI |
| 47 | `/restart` | `cmd_kutai_restart` | Alias for kutai_restart |
| 48 | `/kutai_stop` | `cmd_kutai_stop` | Stop KutAI |
| 49 | `/stop` | `cmd_kutai_stop` | Alias for kutai_stop |

**Total: 49 registered handlers (39 unique commands + 4 aliases: mish, restart, stop, kutai_restart/kutai_stop duplicates)**

### B. Telegram Autocomplete Commands (`set_bot_commands`, lines 219–237)

Only **16 commands** registered for Telegram autocomplete:
`start, mission, missions, queue, status, cancel, progress, digest, shop, price, watch, todo, todos, remember, recall, stop, restart`

### C. Reply Keyboard (persistent bottom buttons, lines 149–157)

```
[ /status ] [ /missions ] [ /queue  ]
[ /todos  ] [ /digest   ] [ /start  ]
[ /stop   ] [ /restart  ]
```

### D. Inline Menu Categories (`MENU_CATEGORIES`, lines 77–134)

| Category | Key | Commands (button label -> callback cmd) |
|---|---|---|
| Missions | `missions` | mission, mission_wf, missions, queue, cancel, priority, pause, resume, graph |
| Monitoring | `monitoring` | status, digest, progress, metrics, audit, replay, cost, modelstats, budget, workspace, wfstatus |
| Shopping | `shopping` | shop, price, watch, research_product |
| Personal | `personal` | todo, todos, cleartodos |
| Knowledge | `knowledge` | remember, recall, ingest, feedback |
| System | `system` | autonomy, credential, tune, load, improve, dlq |
| Danger Zone | `danger` | debug, reset, resetall, kutai_restart, kutai_stop |

### E. Callback Handlers (`handle_callback`, lines 2552–2789)

| Callback Pattern | Action |
|---|---|
| `menu_back` | Return to top-level category menu |
| `menu_cat:<key>` | Show commands for a category |
| `menu_cmd:<cmd>` | Execute a no-arg command |
| `menu_ask:<cmd>` | Prompt user for argument, then execute |
| `todo_toggle:<id>` | Toggle todo done/pending |
| `todo_ai:<id>` | Create task to help with todo |
| `todo_close` | Dismiss todo reminder |
| `todo_help:<id>` | Start AI help flow for a todo |
| `todo_help_cancel` | Cancel todo help flow |
| `wfstatus_dismiss` | Dismiss wfstatus mission picker |
| `wfstatus:<id>` | Show workflow status for mission |
| `wfcancel:<id>` | Cancel a mission |
| `resetall_confirm` | Confirm nuclear wipe |
| `resetall_cancel` | Cancel wipe |
| `approve_<id>` | Approve a task |
| `reject_<id>` | Reject a task |

---

## Issues Found

### 1. Commands missing from Telegram autocomplete (set_bot_commands)

The autocomplete list has only 16 of 39 unique commands. Missing from autocomplete but commonly useful:

- `/task` — quick task creation (power user favorite)
- `/debug` — important for troubleshooting
- `/budget` — cost awareness
- `/wfstatus` — workflow monitoring
- `/pause`, `/resume` — mission control
- `/deals` — shopping complement
- `/compare` — shopping complement
- `/feedback` — quality loop
- `/load` — GPU control

**Impact:** Users typing `/` in the chat won't see most commands in autocomplete.

### 2. Commands missing from inline button menus

These registered handlers have NO button in any menu:

| Command | Why it matters |
|---|---|
| `/task` | Core command — add a one-off task without a mission. No button anywhere. |
| `/deals` | Shopping feature with no menu entry. Users must know the command exists. |
| `/mystuff` | Shopping profile — completely hidden. |
| `/compare` | Product comparison — completely hidden. |
| `/mish` | Alias — fine to exclude from menus. |

### 3. Menu button references commands that only work via alias

- `mission_wf` in MENU_CATEGORIES (line 80) maps via `_CMD_METHOD_MAP` to `cmd_mission_workflow` (line 189). This works, but the button creates a synthetic `/mission_wf` command that isn't registered as a CommandHandler. It only works through the menu button flow. If a user tries `/mission_wf` directly, it will be ignored. This is inconsistent.

### 4. Duplicate / overlapping functionality

| Overlap | Details |
|---|---|
| `/mission --workflow` vs `/mission_wf` button | Two paths to the same thing. The button creates an artificial command that's not a real slash command. |
| `/shop` vs natural language shopping | The message classifier (line 1636) routes shopping messages to the same `shopping_advisor` agent. `/shop` is just a direct shortcut. Not a problem per se, but worth noting. |
| `/restart` + `/kutai_restart` | Same handler. `/kutai_restart` is legacy; `/restart` is the user-facing one. The `kutai_restart` button in Danger Zone uses the legacy name. |
| `/stop` + `/kutai_stop` | Same situation as restart. |
| `/status` vs `/debug` | `/status` shows summary counts; `/debug` shows full task dump. Different enough but naming is confusing — "debug" implies developer mode, not "detailed status". |

### 5. Reply keyboard issues

- `/start` is on the reply keyboard (line 153) but it sends **two messages** (reply keyboard text + inline menu). Tapping it repeatedly clutters the chat.
- `/stop` and `/restart` are on the reply keyboard — dangerous to accidentally tap. No confirmation prompt for `/stop`.
- No shopping shortcut on reply keyboard despite being a major feature.
- `/digest` is on the reply keyboard but is less useful day-to-day than, say, `/shop` or `/wfstatus`.

### 6. Poor button organization / UX issues

- **7 categories** is too many top-level choices. The inline keyboard shows 4 rows of 2 buttons = 8 taps visible. Users must scroll or remember what's where.
- **Monitoring** category has **11 commands** — the most bloated category. Several are niche developer tools (audit, replay, metrics, workspace, modelstats) that don't need prominent placement.
- **System** category mixes user-facing controls (autonomy, load) with developer tools (tune, improve, dlq, credential). A normal user should never need `/credential` or `/tune`.
- **Danger Zone** has `debug` which isn't dangerous — it's read-only. Meanwhile `/cancel` in Missions IS destructive but isn't in Danger Zone.
- The `_CMD_METHOD_MAP` (lines 187–196) is incomplete — it only maps 8 commands. All other commands rely on the `cmd_{name}` convention, which works but is fragile for commands with underscores (e.g., `research_product` correctly resolves to `cmd_research_product`).

### 7. Commands that need arguments but have no guided input flow

These commands require arguments and have no menu button with `needs_arg=True`, so they only work if the user knows the syntax:

- `/task <description>` — not in any menu at all
- `/deals` — no args needed, but not in menu
- `/mystuff` — no args needed, but not in menu
- `/compare <X vs Y>` — not in menu
- `/dlq retry <id>` — has a button, but the arg prompt just says "Enter argument for /dlq:" which is unhelpful for the subcommand syntax

### 8. Missing functionality

| Gap | Description |
|---|---|
| **View task result** | No command to see the result of a completed task by ID. Users must scroll chat history. Should have `/result <task_id>`. |
| **Cancel mission** | `/cancel` only works on tasks (line 658: `cancel_task`). To cancel a mission, users must use the `wfcancel:<id>` inline button from `/wfstatus`. There's no `/cancel_mission <id>` command. |
| **List recent completed** | `get_recent_completed_tasks` is imported (line 16) but never used in any command handler. |
| **Help** | No `/help` command. `/start` shows the menu but doesn't explain what each category does. |

### 9. Shopping menu is incomplete

The Shopping category (lines 102–107) has 4 commands: shop, price, watch, research_product. But the bot also has:
- `/deals` — not in the menu
- `/mystuff` — not in the menu
- `/compare` — not in the menu

These three were added as handlers (lines 290–292) but never added to MENU_CATEGORIES.

### 10. `/stop` has no confirmation

`/restart` and `/stop` execute immediately (lines 631–643). `/resetall` has a confirmation dialog. `/stop` should too — accidentally stopping KutAI is worse than accidentally resetting data, since data persists but the process must be manually restarted.

---

## Proposed Command Structure

### Core Commands (always in autocomplete + reply keyboard)

| Command | Description |
|---|---|
| `/start` | Show menu (should NOT re-send reply keyboard if already set) |
| `/status` | System status |
| `/missions` | List active missions |
| `/queue` | Task queue |
| `/todos` | My todo list |
| `/shop` | Shopping assistant |
| `/help` | **NEW** — Command reference |

### Full Command List (organized by domain)

**Missions & Tasks:**
`/mission`, `/task`, `/missions`, `/queue`, `/cancel`, `/priority`, `/pause`, `/resume`, `/graph`, `/wfstatus`

**Monitoring:**
`/status`, `/digest`, `/progress`, `/cost`, `/budget`

**Shopping:**
`/shop`, `/price`, `/watch`, `/deals`, `/compare`, `/research_product`, `/mystuff`

**Personal:**
`/todo`, `/todos`, `/cleartodos`

**Knowledge:**
`/remember`, `/recall`, `/ingest`, `/feedback`

**System:**
`/autonomy`, `/load`, `/debug`, `/metrics`, `/modelstats`, `/audit`, `/replay`, `/workspace`, `/improve`, `/tune`, `/credential`, `/dlq`

**Control:**
`/restart`, `/stop`, `/reset`, `/resetall`

**Deprecated/Remove:**
- `/kutai_restart`, `/kutai_stop` — keep as hidden aliases but remove from menus
- `/mish` — keep as hidden alias
- `mission_wf` menu entry — merge into `/mission` button with a "Workflow?" follow-up

### New Commands to Add

| Command | Purpose |
|---|---|
| `/result <task_id>` | View the result of a completed task |
| `/help` | Show categorized command reference |
| `/cancel <mission_id\|task_id>` | Extend current cancel to also handle missions (detect mission vs task by checking DB) |

---

## Proposed Button/Menu Layout

### Reply Keyboard (persistent, 3 rows)

```
[ /status ] [ /missions ] [ /queue ]
[ /todos  ] [ /shop     ] [ /deals ]
[ /help   ]
```

Changes:
- Remove `/start` (redundant when keyboard is already showing)
- Remove `/digest` (move to inline menu only)
- Remove `/stop` and `/restart` (dangerous; move to inline menu with confirmation)
- Add `/shop` and `/deals` (high-frequency shopping commands)
- Add `/help` for discoverability

### Inline Menu (from /start or /help, reduced to 5 categories)

```
[ 🎯 Missions & Tasks ] [ 📊 Status & Costs ]
[ 🛒 Shopping          ] [ 📝 Personal        ]
[ ⚙️ System & Admin    ]
```

#### Missions & Tasks
```
[ 🎯 New Mission    ] [ 📋 Add Task       ] [ 📬 Queue          ]
[ 📋 List Missions  ] [ 🔄 Workflow Mission]
[ 🚫 Cancel         ] [ 🔢 Priority       ] [ 🌳 Graph          ]
[ ⏸️ Pause           ] [ ▶️ Resume          ] [ 📋 WF Status      ]
[ ⬅️ Back ]
```

Changes: Added "Add Task" button for `/task`.

#### Status & Costs
```
[ 📊 Status  ] [ 📰 Digest   ] [ 📈 Progress ]
[ 💰 Cost    ] [ 💵 Budget   ] [ 🤖 Models   ]
[ ⬅️ Back ]
```

Changes: Removed audit, replay, metrics, workspace from this category (moved to System).

#### Shopping
```
[ 🛒 Find Product    ] [ 💰 Compare Prices ] [ ⚖️ Compare X vs Y ]
[ ⏰ Price Watch      ] [ 🔍 Deep Research  ] [ 🏷️ My Deals       ]
[ 📦 My Stuff        ]
[ ⬅️ Back ]
```

Changes: Added Compare, Deals, My Stuff buttons.

#### Personal
```
[ 📝 Add Todo ] [ 📋 My Todos ] [ 🗑️ Clear Done ]
[ 💾 Remember ] [ 🔎 Recall   ] [ 📥 Ingest      ]
[ ⭐ Feedback ]
[ ⬅️ Back ]
```

Changes: Merged Knowledge into Personal (they're all personal-use commands).

#### System & Admin
```
[ 🎚️ Autonomy ] [ 🖥️ Load    ] [ 🐛 Debug    ]
[ 📉 Metrics  ] [ 🔍 Audit   ] [ 🔁 Replay   ]
[ 🗂️ Workspaces] [ 🧪 Improve ] [ 🎛️ Tune    ]
[ 📭 DLQ      ] [ 🔑 Credential ]
[ ♻️ Reset    ] [ ☢️ Reset All ] [ 🔄 Restart  ] [ ⏹ Stop ]
[ ⬅️ Back ]
```

Changes: Consolidated System + Danger Zone + dev monitoring tools into one category. Developer tools are at the bottom.

---

## Implementation Steps

### Step 1: Add missing commands to MENU_CATEGORIES

**File:** `src/app/telegram_bot.py`

1. **Add `/task` to Missions category** (after line 82, inside the missions list):
   ```python
   ("📋 Add Task", "task", True, "Describe the task:"),
   ```

2. **Add `/deals`, `/compare`, `/mystuff` to Shopping category** (after line 107):
   ```python
   ("🏷️ My Deals", "deals", False, None),
   ("⚖️ Compare", "compare", True, "Compare what? (e.g. 'iPhone 15 vs Samsung S24')"),
   ("📦 My Stuff", "mystuff", False, None),
   ```

3. **Add `task`, `deals`, `compare`, `mystuff` to `_CMD_METHOD_MAP`** (around line 187):
   ```python
   "task": "cmd_add_task",
   "deals": "cmd_deals",
   "mystuff": "cmd_mystuff",
   "compare": "cmd_compare",
   ```
   Note: `deals`, `mystuff`, `compare` already follow the `cmd_{name}` convention so technically they resolve automatically. But explicit mapping is safer.

### Step 2: Restructure MENU_CATEGORIES (lines 77–134)

Reduce from 7 categories to 5 by:
- Merging "Knowledge" into "Personal"
- Merging "Danger Zone" into "System" (at the bottom of the list)
- Moving dev tools (audit, replay, metrics, workspace) from Monitoring to System

Replace the entire `MENU_CATEGORIES` definition (lines 77–134) with the new layout described above.

### Step 3: Update reply keyboard (lines 149–157)

Replace `REPLY_KEYBOARD` with:
```python
REPLY_KEYBOARD = ReplyKeyboardMarkup(
    [
        [KeyboardButton("/status"), KeyboardButton("/missions"), KeyboardButton("/queue")],
        [KeyboardButton("/todos"), KeyboardButton("/shop"), KeyboardButton("/deals")],
        [KeyboardButton("/help")],
    ],
    resize_keyboard=True,
    is_persistent=True,
)
```

### Step 4: Update `set_bot_commands` (lines 219–237)

Expand to include the most useful commands:
```python
commands = [
    BotCommand("start", "Show main menu"),
    BotCommand("help", "Command reference"),
    BotCommand("mission", "Create a new mission"),
    BotCommand("task", "Add a quick task"),
    BotCommand("missions", "List active missions"),
    BotCommand("queue", "View task queue"),
    BotCommand("status", "System status"),
    BotCommand("cancel", "Cancel a task or mission"),
    BotCommand("pause", "Pause a mission"),
    BotCommand("resume", "Resume a mission"),
    BotCommand("progress", "Mission progress"),
    BotCommand("wfstatus", "Workflow status"),
    BotCommand("digest", "Daily digest"),
    BotCommand("shop", "Shopping assistant"),
    BotCommand("price", "Quick price check"),
    BotCommand("watch", "Set up price watch"),
    BotCommand("deals", "Active deals & watches"),
    BotCommand("compare", "Compare products"),
    BotCommand("todo", "Add a personal todo"),
    BotCommand("todos", "List your todos"),
    BotCommand("remember", "Save to memory"),
    BotCommand("recall", "Search memory"),
    BotCommand("budget", "View/set cost budget"),
    BotCommand("load", "GPU load control"),
    BotCommand("debug", "Full system debug"),
    BotCommand("stop", "Stop KutAI"),
    BotCommand("restart", "Restart KutAI"),
]
```

### Step 5: Add `/help` command

Add handler registration in `_setup_handlers` (after line 246):
```python
self.app.add_handler(CommandHandler("help", self.cmd_help))
```

Add method:
```python
async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show categorized command reference."""
    help_text = (
        "📖 *KutAI Command Reference*\n\n"
        "*Missions & Tasks*\n"
        "/mission — Create a mission\n"
        "/task — Quick task\n"
        "/missions — List missions\n"
        "/queue — Task queue\n"
        "/cancel — Cancel task/mission\n"
        "/pause, /resume — Pause/resume\n"
        "/wfstatus — Workflow status\n\n"
        "*Status & Costs*\n"
        "/status — System overview\n"
        "/digest — Daily digest\n"
        "/progress — Timeline\n"
        "/budget — Cost budget\n\n"
        "*Shopping*\n"
        "/shop — Find products\n"
        "/price — Price check\n"
        "/compare — X vs Y\n"
        "/watch — Price alerts\n"
        "/deals — Active watches\n\n"
        "*Personal*\n"
        "/todo — Add reminder\n"
        "/todos — My list\n"
        "/remember — Save fact\n"
        "/recall — Search facts\n\n"
        "*System*\n"
        "/load — GPU control\n"
        "/autonomy — Risk level\n"
        "/debug — Full dump\n"
        "/stop, /restart — Control\n\n"
        "_Tap /start for button menu._"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")
```

### Step 6: Add `/result` command

Add handler in `_setup_handlers`:
```python
self.app.add_handler(CommandHandler("result", self.cmd_result))
```

Add method:
```python
async def cmd_result(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """View the result of a completed task. /result <task_id>"""
    if not context.args:
        await update.message.reply_text("Usage: /result <task_id>")
        return
    try:
        task_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Task ID must be a number.")
        return
    task = await get_task(task_id)
    if not task:
        await update.message.reply_text(f"Task #{task_id} not found.")
        return
    result = task.get("result", "")
    if not result:
        await update.message.reply_text(
            f"Task #{task_id} ({task['status']}): no result yet."
        )
        return
    header = f"📄 *Result — Task #{task_id}*\n_{task['title']}_\n\n"
    if len(result) > 3500:
        # Send as file
        ...
    else:
        await update.message.reply_text(header + result, parse_mode="Markdown")
```

### Step 7: Extend `/cancel` to handle missions (line 647)

Currently `/cancel` only calls `cancel_task`. Modify to:
1. Try `cancel_task(id)` first
2. If not found, try to cancel as a mission via `update_mission(id, status="cancelled")`
3. Also cancel all pending tasks for that mission

### Step 8: Add confirmation to `/stop` (line 638)

Similar to `cmd_reset_all`, add a confirmation inline button before actually stopping:
```python
async def cmd_kutai_stop(self, update, context):
    keyboard = [[
        InlineKeyboardButton("⏹ Yes, stop", callback_data="stop_confirm"),
        InlineKeyboardButton("Cancel", callback_data="stop_cancel"),
    ]]
    await update.message.reply_text(
        "⚠️ Stop KutAI? You'll need to manually restart.",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
```

Add `stop_confirm` and `stop_cancel` handlers in `handle_callback`.

### Step 9: Fix `/start` double-message issue (line 308)

Don't re-send the reply keyboard if it's already set. Change `cmd_start` to send only the inline menu, and set the reply keyboard only on first interaction (or use a flag).

### Step 10: Use `get_recent_completed_tasks` (imported but unused, line 16)

Either:
- Wire it into the `/result` command as a "recent results" fallback when no ID given
- Or remove the unused import to reduce confusion

### Priority Order

1. **Step 1** — Add missing buttons (quick win, high impact)
2. **Step 2** — Restructure categories (UX improvement)
3. **Step 3** — Update reply keyboard (daily usability)
4. **Step 4** — Update autocomplete (discoverability)
5. **Step 5** — Add `/help` (discoverability)
6. **Step 8** — Confirm on `/stop` (safety)
7. **Step 6** — Add `/result` (missing feature)
8. **Step 7** — Extend `/cancel` for missions (missing feature)
9. **Step 9** — Fix `/start` double-send (polish)
10. **Step 10** — Clean up unused import (tech debt)
