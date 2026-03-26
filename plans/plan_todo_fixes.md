# Todo Module — Issues & Fixes

## What's Supposed to Work

Per user requirements (2026-03-25):

1. **Collect todo items** — explicit `/todo` command and implicit detection from natural language ("remind me to...", "don't forget...")
2. **Remind every 2 hours** — between 9am–9pm Turkey time (UTC+3), send pending todos via Telegram with inline buttons
3. **AI suggestions** — before each reminder, spawn LLM tasks to generate one actionable suggestion per pending todo. Include suggestions in the reminder message.
4. **Easy mark-done** — inline ✅ buttons on reminders and todo listings; toggle between done/pending
5. **AI help follow-up** — user taps 🤖 button on a suggestion, gets a ForceReply prompt, sends reply, which creates a task for deeper help

## What's Actually Broken

### Issue 1: CRITICAL — Timezone Mismatch Prevents Reminders From Firing

The **root cause** of reminders not working.

- `get_due_scheduled_tasks()` (db.py:1143) checks: `next_run <= datetime('now')` — SQLite's `datetime('now')` returns **UTC**.
- `_compute_next_run()` (orchestrator.py:755, 781) computes `next_run` using `datetime.now()` — Python's **local time** (UTC+3 on the server).
- The cron expression is `0 9,11,13,15,17,19,21 * * *` — intended as Turkey time hours.

**What happens:** At 11:00 Turkey time (08:00 UTC), `_compute_next_run` stores `next_run = "2026-03-26T13:00:00"` (local). But SQLite compares this against UTC `datetime('now')` which is `08:00`. Since `13:00 > 08:00`, the task appears "not yet due" — it won't fire until 13:00 UTC (16:00 Turkey time). This creates a **3-hour delay** on every scheduled run, and the 21:00 slot (stored as local) won't fire until 21:00 UTC = midnight Turkey time, effectively skipping it.

**Worse:** if the server restarts and `next_run` is NULL, `get_due_scheduled_tasks` returns it immediately (NULL check on line 1143), the first reminder fires once, then the timezone drift kicks in for all subsequent runs.

**Files:**
- `src/infra/db.py:1143` — `datetime('now')` is UTC
- `src/core/orchestrator.py:755` — `datetime.now()` is local time
- `src/core/orchestrator.py:781` — same issue for non-todo scheduled tasks

### Issue 2: Suggestion Tasks May Silently Fail, No Fallback Timeout

- `_start_todo_suggestions()` (orchestrator.py:794–839) creates suggestion tasks with `agent_type="assistant"` and `tier="auto"`.
- `_check_todo_suggestions_complete()` (orchestrator.py:841–882) is only called from `_complete_task()` (orchestrator.py:1408) when a suggestion task finishes.
- **Problem:** If any suggestion task gets stuck, errors, or is never picked up (e.g., no model available, GPU busy), the batch never completes and the reminder **never sends**. There is no timeout/watchdog for suggestion batches.
- The fallback at line 836–839 only fires if ALL tasks failed to be created (deduped). If even one task is created but never completes, the reminder is lost.

**Files:**
- `src/core/orchestrator.py:841-882` — no timeout on batch completion
- `src/core/orchestrator.py:831-839` — fallback only covers creation failure

### Issue 3: Suggestion Batch Query Uses Fragile LIKE Match

- `_check_todo_suggestions_complete()` (orchestrator.py:848–851) queries tasks with `context LIKE '%"batch_id"%'`.
- This is a string-contains search on JSON. If `batch_id` is a substring of another value, or if the JSON is serialized differently (e.g., spaces after colons), it could miss or over-match tasks.
- Not a showstopper, but a reliability risk.

**File:** `src/core/orchestrator.py:851`

### Issue 4: `todo_toggle` Callback Replaces Entire Reminder Message

- When user taps ✅ on a todo in the reminder, `todo_toggle` (telegram_bot.py:2619–2635) calls `query.edit_message_text()` replacing the **entire reminder** (which lists ALL todos) with just the single toggled item's status.
- After one toggle, the user loses the full list and all other buttons.
- This makes it impossible to mark multiple todos done from a single reminder.

**File:** `src/app/telegram_bot.py:2625-2634`

### Issue 5: `_last_todo_help` Is an Instance Attribute — Race Condition

- `cmd__todo_help()` (telegram_bot.py:2964–2979) reads `self._last_todo_help` which was set by the callback at line 1529.
- This is a single instance variable. If two help requests overlap (unlikely with one user, but possible), the second overwrites the first.
- Also, if the user takes too long, there's no expiry — it could fire with stale context.

**File:** `src/app/telegram_bot.py:1529, 2967-2971`

### Issue 6: Cron Hours Are Server-Local, Not Turkey Time

- The cron `0 9,11,13,15,17,19,21 * * *` is meant for Turkey time (UTC+3).
- `_compute_next_run` uses `datetime.now()` which is local time. If the server runs in UTC+3, the hours are correct. But there is no explicit timezone enforcement — if the server ever moves to a different timezone or runs in UTC (common for cloud), the hours are wrong.
- The requirement says "except between 10pm–9am Turkey time". The cron covers 9,11,13,15,17,19,21 which is correct for UTC+3, but only by coincidence of the server timezone.

**File:** `src/core/orchestrator.py:884-943`

## Root Causes

1. **Primary:** `datetime('now')` (UTC) vs `datetime.now()` (local) mismatch means `next_run` values stored are in local time but compared against UTC. Reminders fire at wrong times or not at all.
2. **Secondary:** No timeout on suggestion batch completion means a stuck task blocks the reminder indefinitely.
3. **UX:** Toggle callback destroys the reminder message, making multi-toggle impossible.

## Missing Implementations

1. **No batch timeout watchdog** — if suggestion tasks don't complete within N minutes, the reminder should send without suggestions. Currently there is no such mechanism.
2. **No "snooze" or "remind later" button** — user requirement mentions easy interaction but there's no snooze.
3. **No priority editing from Telegram** — `update_todo()` exists in db.py but there's no command or callback to set priority.
4. **No due date support in commands** — the schema has `due_date` but `/todo` only takes a title; no way to set due dates.
5. **No todo description editing** — implicit todos strip prefixes but there's no way to add/edit descriptions.
6. **Reminder doesn't re-render after toggle** — should rebuild the full list instead of replacing with single-item text.

## Fix Plan

### Priority 1: Fix Timezone Mismatch (CRITICAL — This is why reminders don't fire)

**Option A (recommended):** Normalize everything to UTC.
- In `_compute_next_run()`: use `datetime.now(timezone.utc)` instead of `datetime.now()`.
- Adjust cron hours from Turkey time to UTC: `0 6,8,10,12,14,16,18 * * *` (subtract 3h).
- Alternatively, keep Turkey-time cron but convert the computed `next_run` to UTC before storing.

**Option B:** Change SQLite query to use local time.
- Replace `datetime('now')` with `datetime('now', 'localtime')` in `get_due_scheduled_tasks()`.
- Fragile — depends on server timezone being UTC+3 forever.

**Files to change:**
- `src/infra/db.py:1143` — or —
- `src/core/orchestrator.py:755, 781` and cron expression in `db.py:325`

### Priority 2: Add Suggestion Batch Timeout

- In `check_scheduled_tasks()` or the main loop, add a check: if a `todo_suggest_batch` has been pending for > 5 minutes, send the reminder without suggestions.
- Store batch creation time (already embedded in `batch_id` as a timestamp).
- In the main loop (every 60s), scan for stale batches and force-send.

**Files to change:**
- `src/core/orchestrator.py` — add `_check_stale_todo_batches()` method, call from main loop.

### Priority 3: Fix Toggle Callback to Rebuild Full Reminder

- When `todo_toggle` fires, rebuild the entire reminder message (re-fetch all pending todos, re-render with buttons) instead of replacing with single-item text.
- Reuse `send_todo_reminder`'s formatting logic, or factor it into a shared function.

**Files to change:**
- `src/app/telegram_bot.py:2619-2635` — replace `edit_message_text` with full list rebuild.
- `src/app/reminders.py` — extract formatting into a reusable function.

### Priority 4: Explicit Timezone Handling

- Add `TIMEZONE = ZoneInfo("Europe/Istanbul")` to config.
- Use `datetime.now(TIMEZONE)` in all scheduler code.
- Document that cron expressions are in Turkey time.

**Files to change:**
- `src/app/config.py` — add timezone constant
- `src/core/orchestrator.py` — use it in `_compute_next_run` and scheduler

### Priority 5: Minor UX Improvements

- Add "Remind Later" / snooze button to reminders.
- Add `/todo <title> @high` syntax for priority.
- Add `/todo <title> by <date>` syntax for due dates.
- These are enhancements, not blockers.
