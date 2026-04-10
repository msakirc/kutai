# Todo System X-Ray

## Overview

The todo module lets the user collect tasks via Telegram, get AI suggestions for each, and act on them through an interactive list-detail UI. Suggestions are persisted per-todo, generated lazily, and cached until the todo is edited or reopened.

## Schema

```sql
CREATE TABLE todo_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT DEFAULT '',
    priority TEXT DEFAULT 'normal',     -- high | normal | low
    due_date TIMESTAMP,
    status TEXT DEFAULT 'pending',      -- pending | done
    source TEXT DEFAULT 'explicit',     -- explicit (via /todo) | implicit (NL detection)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    suggestion TEXT,                    -- AI-generated actionable suggestion
    suggestion_agent TEXT,              -- best agent type: researcher | shopping_advisor | assistant | coder
    suggestion_at TIMESTAMP            -- when suggestion was attempted (NULL = never tried)
);
```

### Suggestion State Machine

| suggestion | suggestion_at | Meaning |
|------------|---------------|---------|
| NULL | NULL | Never attempted — eligible for LLM call |
| NULL | timestamp | Attempted and failed (or "no suggestion") — don't retry |
| text | timestamp | Cached suggestion — display in UI |

Editing a todo's title resets all three fields to NULL, making it eligible again. Toggling done->pending also resets them.

## Entry Points

### Adding Todos

- **`/todo <title>`** — explicit add, `source='explicit'`
- **NL detection** — "remind me to buy milk" detected by keyword classifier in `telegram_bot.py`, `source='implicit'`
- Both go through `add_todo()` in `db.py`

### Viewing Todos

- **`/todos`** — calls `build_todo_list_message()` from `reminders.py`
- **Scheduled reminder** — orchestrator calls `_start_todo_suggestions()` on cron, which generates suggestions then sends the list via `send_todo_reminder()`

## UI Flow

```
/todos or scheduled reminder
        |
        v
+-------------------+
| Pending Todos     |
| 1. Buy milk - *hint* |
| 2. Call dentist   |
| [1] [2] [Close]   |
+-------------------+
        | tap "1"
        v
+-------------------+
| #1 - Buy milk     |
| Priority: normal   |
| *suggestion*       |
| [Done][Edit][Help][Cancel] |
| [Back]             |
+-------------------+
```

### Callback Handlers (telegram_bot.py)

| Callback Data | Action |
|---------------|--------|
| `todo_detail:{id}` | Show detail view for one todo |
| `todo_toggle:{id}` | Toggle done/pending, return to list |
| `todo_edit:{id}` | Set pending action, prefill keyboard with current title |
| `todo_delete:{id}` | Delete todo, return to list |
| `todo_help:{id}:{agent}` | Set pending action, prefill keyboard with suggestion |
| `todo_list` | Return to list view (back button) |
| `todo_close` | Delete the message |

### Pending Action Flows

**`_todo_edit`**: User types new title -> `_handle_todo_edit()` updates title and resets suggestion fields.

**`_todo_help`**: User taps prefilled suggestion or types custom request -> `cmd__todo_help()` creates an agent task with the chosen agent type and description.

## Suggestion Generation

**Trigger**: Orchestrator's scheduled task (cron-based, typically every 2h).

**Flow** (`orchestrator.py`):
1. `_start_todo_suggestions()` fetches pending todos
2. Filters to those with `suggestion IS NULL AND suggestion_at IS NULL`
3. If any need suggestions, calls `_generate_suggestions()`
4. Single OVERHEAD LLM call with all todos in prompt (max 10)
5. `_parse_todo_suggestions()` parses response (lenient: handles `N.`/`N)`, markdown, missing items)
6. Persists suggestion + agent per todo, or marks `suggestion_at` on failure
7. Sends reminder via `send_todo_reminder()`

**Key design decisions**:
- OVERHEAD category — no model swap triggered, uses whatever's loaded or cloud
- 45s timeout — allows time for proactive model load
- Failures marked with `suggestion_at` — prevents retry loops on unparseable todos
- Max 10 todos per batch — keeps prompt small

## File Map

| File | Responsibility |
|------|----------------|
| `src/infra/db.py` | Schema, CRUD: `add_todo`, `get_todo`, `get_todos`, `update_todo`, `toggle_todo`, `delete_todo` |
| `src/app/reminders.py` | `build_todo_list_message()`, `build_todo_detail_message()`, `send_todo_reminder()`, `_format_age()` |
| `src/core/orchestrator.py` | `_start_todo_suggestions()`, `_generate_suggestions()`, `_parse_todo_suggestions()` |
| `src/app/telegram_bot.py` | `/todo`, `/todos`, `/cleartodos` commands; all `todo_*` callback handlers; `_handle_todo_edit()`, `cmd__todo_help()` |
| `tests/test_todo.py` | DB CRUD, NL classification, title extraction, suggestion parser |
| `tests/test_todo_ui.py` | List message builder, detail message builder |

## Unused / Future Opportunities

- **`source` field**: Stored as `explicit` or `implicit` but never read back. Could differentiate display (e.g., implicit todos shown with less confidence) or filter reminders.
- **`due_date` field**: Column exists but no UI to set it. Could drive time-sensitive reminder prioritization.
- **`description` field**: Included in LLM prompt when present, but no UI to add/edit it. The edit flow only changes title.
- **DLQ integration**: `build_todo_list_message()` appends DLQ (dead letter queue) tasks below todos with retry/skip buttons — this is unrelated to todos but shares the reminder message.
