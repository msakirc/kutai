# Todo Module Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the todo reminder actually work — fire on schedule, generate AI suggestions via normal tasks, show item age, provide compact mark-done and help buttons.

**Architecture:** The scheduler triggers the reminder flow. For each pending todo, a suggestion task is created through the normal task queue (local_only, prefer_quality). Once all suggestion tasks complete, the reminder message is assembled and sent via Telegram with inline buttons. The help button uses a two-step flow: prefill the suggestion as editable text, then create a task from the user's reply.

**Tech Stack:** Python, python-telegram-bot (InlineKeyboardButton, ForceReply), SQLite (existing DB), existing task queue and model routing.

**Spec:** `docs/superpowers/specs/2026-03-25-todo-module-fix-design.md`

---

### Task 1: Add `_format_age` helper to reminders.py

**Files:**
- Modify: `src/app/reminders.py`
- Test: `tests/test_todo.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_todo.py`:

```python
from datetime import datetime, timedelta

def test_format_age():
    from src.app.reminders import _format_age
    now = datetime(2026, 3, 26, 12, 0, 0)
    assert _format_age(now - timedelta(minutes=30), now) == "30m ago"
    assert _format_age(now - timedelta(hours=3), now) == "3h ago"
    assert _format_age(now - timedelta(days=2), now) == "2d ago"
    assert _format_age(now - timedelta(weeks=3), now) == "3w ago"
    assert _format_age(now - timedelta(minutes=2), now) == "2m ago"
    # Edge: just created
    assert _format_age(now, now) == "now"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_todo.py::test_format_age -v`
Expected: FAIL — `_format_age` not defined

- [ ] **Step 3: Write minimal implementation**

Add to `src/app/reminders.py`:

```python
from datetime import datetime


def _format_age(created_at, now=None):
    """Return compact relative time: '2m ago', '3h ago', '5d ago', '2w ago'."""
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    if now is None:
        now = datetime.now()
    diff = now - created_at
    minutes = int(diff.total_seconds() / 60)
    if minutes < 1:
        return "now"
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    if days < 14:
        return f"{days}d ago"
    weeks = days // 7
    return f"{weeks}w ago"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_todo.py::test_format_age -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/app/reminders.py tests/test_todo.py
git commit -m "feat(todo): add _format_age helper for compact relative timestamps"
```

---

### Task 2: Rewrite `send_todo_reminder` with new message format

**Files:**
- Modify: `src/app/reminders.py`

This task updates the reminder message to use the new format: priority icon, ID, title, age, and compact button rows. Suggestions are not yet wired — this task accepts an optional `suggestions` dict parameter for later use.

- [ ] **Step 1: Rewrite `send_todo_reminder` in `src/app/reminders.py`**

Replace the entire `send_todo_reminder` function:

```python
async def send_todo_reminder(telegram, suggestions=None):
    """Fetch pending todos, format with inline buttons, send to Telegram.

    Args:
        telegram: TelegramBot instance.
        suggestions: Optional dict {todo_id: suggestion_str} from suggestion tasks.
    """
    try:
        todos = await get_todos(status="pending")
        if not todos:
            return

        suggestions = suggestions or {}
        lines = ["📝 *Pending Todos*\n"]
        done_buttons = []
        help_buttons = []

        for todo in todos:
            tid = todo["id"]
            title = todo["title"]
            priority = todo.get("priority", "normal")
            p_icon = _PRIORITY_ICONS.get(priority, "🟡")
            age = _format_age(todo["created_at"])

            lines.append(f"  {p_icon} *#{tid}* — {title} ({age})")

            suggestion = suggestions.get(tid)
            if suggestion:
                lines.append(f"   💡 {suggestion}")

            done_buttons.append(
                InlineKeyboardButton(f"✅ #{tid}", callback_data=f"todo_toggle:{tid}")
            )
            if suggestion:
                help_buttons.append(
                    InlineKeyboardButton(f"🤖 #{tid}", callback_data=f"todo_help:{tid}")
                )

        text = "\n".join(lines)
        rows = []
        # Pack done buttons, max 4 per row
        for i in range(0, len(done_buttons), 4):
            rows.append(done_buttons[i:i + 4])
        if help_buttons:
            for i in range(0, len(help_buttons), 4):
                rows.append(help_buttons[i:i + 4])
        rows.append([InlineKeyboardButton("🔙 Close", callback_data="todo_close")])

        markup = InlineKeyboardMarkup(rows)

        await telegram.app.bot.send_message(
            chat_id=_get_admin_chat_id(),
            text=text,
            parse_mode="Markdown",
            reply_markup=markup,
        )
        logger.info("todo reminder sent", count=len(todos), suggestions=len(suggestions))

    except Exception as e:
        logger.error("failed to send todo reminder", error=str(e))
```

Also remove `_AI_HELPABLE_KEYWORDS` — no longer needed (suggestions come from AI tasks now).

- [ ] **Step 2: Manually verify the format looks right**

Run: `python -m pytest tests/test_todo.py -v`
Expected: All existing tests still pass.

- [ ] **Step 3: Commit**

```bash
git add src/app/reminders.py
git commit -m "feat(todo): rewrite reminder format with age, compact buttons, suggestions param"
```

---

### Task 3: Add `todo_close` and `todo_help` callback handlers

**Files:**
- Modify: `src/app/telegram_bot.py`

- [ ] **Step 1: Add `todo_close` handler**

In `handle_callback` in `src/app/telegram_bot.py`, in the `# ── Todo Callbacks` section, after the `todo_ai:` handler, add:

```python
        if data == "todo_close":
            try:
                await query.delete_message()
            except Exception:
                await query.edit_message_text("(closed)")
            return
```

- [ ] **Step 2: Add `todo_help` handler with two-step confirmation**

In the same section, add:

```python
        if data.startswith("todo_help:"):
            todo_id = int(data.split(":")[1])
            todo = await get_todo(todo_id)
            if not todo:
                await query.answer("Todo not found")
                return
            # Get suggestion from the reminder message text
            suggestion = self._extract_suggestion_from_message(
                query.message.text, todo_id
            )
            prompt_text = suggestion or f"Help me with: {todo['title']}"
            # Store pending help action so reply is routed correctly
            chat_id = query.message.chat_id
            self._pending_action[chat_id] = {
                "command": "_todo_help",
                "todo_id": todo_id,
                "todo_title": todo["title"],
            }
            from telegram import ForceReply
            await query.message.reply_text(
                f"🤖 *Help with: {todo['title']}*\n"
                f"Suggested action: _{suggestion or 'Help me with this'}_\n\n"
                f"Edit below and send, or tap Cancel.",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("❌ Cancel", callback_data="todo_help_cancel")
                ]]),
            )
            # ForceReply with the suggestion as placeholder to guide input
            await query.message.get_bot().send_message(
                chat_id=query.message.chat_id,
                text=prompt_text,
                reply_markup=ForceReply(
                    selective=True,
                    input_field_placeholder="Edit and send...",
                ),
            )
            return

        if data == "todo_help_cancel":
            chat_id = query.message.chat_id
            self._pending_action.pop(chat_id, None)
            try:
                await query.delete_message()
            except Exception:
                await query.edit_message_text("(cancelled)")
            return
```

- [ ] **Step 3: Add helper to extract suggestion from reminder message text**

Add as a method on `TelegramBot`:

```python
    @staticmethod
    def _extract_suggestion_from_message(message_text, todo_id):
        """Extract the 💡 suggestion line for a given todo from the reminder message."""
        if not message_text:
            return None
        marker = f"*#{todo_id}*"
        lines = message_text.split("\n")
        for i, line in enumerate(lines):
            if marker in line and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith("💡"):
                    return next_line[2:].strip()
        return None
```

- [ ] **Step 4: Wire the help reply in message handler**

In the `handle_message` method, the existing `_pending_action` handling at PRIORITY 0 (around line 1482) already pops the pending action and dispatches to a command handler. Add handling for the special `_todo_help` command.

Add a new method:

```python
    async def cmd__todo_help(self, update, context):
        """Handle the user's reply to a todo help suggestion."""
        text = update.message.text or ""
        chat_id = update.effective_chat.id
        # The pending action was already popped by the caller, but we stored
        # todo_id in context.  We need to get it from the action dict.
        # Actually, _pending_action is popped in handle_message — we need to
        # pass it through. Use a different approach: store on self temporarily.
        todo_info = getattr(self, "_last_todo_help", None)
        if not todo_info:
            await update.message.reply_text("❌ Help session expired. Try again from the reminder.")
            return
        self._last_todo_help = None
        todo_id = todo_info["todo_id"]
        todo_title = todo_info["todo_title"]
        task_id = await add_task(
            title=f"Help with: {todo_title[:40]}",
            description=text,
            tier="auto",
            priority=8,
            context={"todo_id": todo_id, "local_only": True, "prefer_quality": True},
        )
        await update.message.reply_text(f"✅ Task #{task_id} created!")
```

Update the PRIORITY 0 section to handle `_todo_help` specially — before dispatching via `_resolve_cmd_handler`, check if the command is `_todo_help`:

```python
        pending_action = self._pending_action.pop(chat_id, None)
        if pending_action:
            cmd = pending_action["command"]
            if cmd == "_todo_help":
                self._last_todo_help = pending_action
                await self.cmd__todo_help(update, context)
                return
            handler = self._resolve_cmd_handler(cmd)
            # ... rest of existing code
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_todo.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/app/telegram_bot.py
git commit -m "feat(todo): add close, help, and help-cancel callback handlers"
```

---

### Task 4: Wire suggestion tasks into the reminder flow

**Files:**
- Modify: `src/core/orchestrator.py`
- Modify: `src/app/reminders.py`

This is the core change: when the scheduler fires the todo reminder, instead of calling `send_todo_reminder` directly, it creates suggestion tasks. When the last suggestion task completes, it sends the reminder with the collected suggestions.

- [ ] **Step 1: Add suggestion task creation in orchestrator**

In `src/core/orchestrator.py`, replace the todo reminder special handling (lines 737-755):

```python
                # Special handling: todo reminders create suggestion tasks first
                if sched_ctx.get("type") == "todo_reminder":
                    try:
                        await self._start_todo_suggestions()
                    except Exception as e:
                        logger.error(f"[Scheduler] Todo suggestion creation failed: {e}")
                        # Fallback: send reminder without suggestions
                        try:
                            from src.app.reminders import send_todo_reminder
                            if self.telegram:
                                await send_todo_reminder(self.telegram)
                        except Exception:
                            pass
                    # Update last_run / next_run and skip task creation
                    now = datetime.now()
                    next_run = self._compute_next_run(
                        sched.get("cron_expression", "0 * * * *"), now
                    )
                    await update_scheduled_task(
                        sched_id,
                        last_run=now.isoformat(),
                        next_run=next_run.isoformat() if next_run else None,
                    )
                    continue
```

- [ ] **Step 2: Add `_start_todo_suggestions` method**

Add to the Orchestrator class:

```python
    async def _start_todo_suggestions(self):
        """Create one suggestion task per pending todo item."""
        from src.infra.db import get_todos, add_task
        todos = await get_todos(status="pending")
        if not todos:
            return

        batch_id = f"todo_suggest_{int(datetime.now().timestamp())}"
        task_ids = []

        for todo in todos:
            task_id = await add_task(
                title=f"Suggest action for: {todo['title'][:50]}",
                description=(
                    f"The user has this todo item: \"{todo['title']}\"\n"
                    f"Description: {todo.get('description') or '(none)'}\n\n"
                    f"Suggest ONE concrete, actionable way you (an AI assistant) could help. "
                    f"Be creative — even mundane items like 'buy milk' could mean "
                    f"price comparison or online ordering. "
                    f"If you genuinely cannot help, just say 'no suggestion'. "
                    f"Reply with ONLY the suggestion, one sentence, no preamble."
                ),
                agent_type="assistant",
                tier="auto",
                priority=3,  # low — background work
                context={
                    "local_only": True,
                    "prefer_quality": True,
                    "silent": True,
                    "todo_suggest_batch": batch_id,
                    "todo_id": todo["id"],
                    "todo_count": len(todos),
                },
            )
            if task_id:
                task_ids.append(task_id)

        if task_ids:
            logger.info(
                f"[Todo] Created {len(task_ids)} suggestion tasks, batch={batch_id}"
            )
        else:
            # All deduped or failed — send reminder without suggestions
            from src.app.reminders import send_todo_reminder
            if self.telegram:
                await send_todo_reminder(self.telegram)
```

- [ ] **Step 3: Add suggestion completion handler**

In the `process_task` method, after the existing task completion notification logic (around line 1259), add a check for suggestion task completion:

```python
        # Check if this is a todo suggestion task — trigger reminder when all done
        task_ctx = task.get("context", {})
        if isinstance(task_ctx, str):
            import json
            task_ctx = json.loads(task_ctx)
        if task_ctx.get("todo_suggest_batch"):
            await self._check_todo_suggestions_complete(task_ctx)
```

Add the method:

```python
    async def _check_todo_suggestions_complete(self, task_ctx):
        """Check if all suggestion tasks in this batch are done. If so, send reminder."""
        batch_id = task_ctx["todo_suggest_batch"]
        todo_count = task_ctx.get("todo_count", 0)

        # Query all tasks in this batch
        db = await get_db()
        cursor = await db.execute(
            """SELECT id, status, result, context FROM tasks
               WHERE context LIKE ?""",
            (f'%"{batch_id}"%',),
        )
        rows = [dict(r) for r in await cursor.fetchall()]

        # Check if all are terminal (completed or failed)
        terminal = [r for r in rows if r["status"] in ("completed", "failed")]
        if len(terminal) < todo_count and len(terminal) < len(rows):
            return  # Still waiting

        # Collect suggestions
        suggestions = {}
        for row in rows:
            ctx = row.get("context", "{}")
            if isinstance(ctx, str):
                ctx = json.loads(ctx)
            todo_id = ctx.get("todo_id")
            result = row.get("result", "")
            if todo_id and result and "no suggestion" not in result.lower():
                # Clean up the suggestion text
                suggestion = result.strip().strip('"').strip("'")
                if len(suggestion) > 5:  # Skip empty/trivial
                    suggestions[todo_id] = suggestion

        # Send reminder with suggestions
        from src.app.reminders import send_todo_reminder
        if self.telegram:
            await send_todo_reminder(self.telegram, suggestions=suggestions)

        logger.info(
            f"[Todo] Reminder sent with {len(suggestions)}/{len(rows)} suggestions, "
            f"batch={batch_id}"
        )
```

- [ ] **Step 4: Suppress notifications for silent tasks**

In the task completion notification logic (around line 1244), add a check at the top:

```python
        # Silent tasks skip Telegram notification
        task_ctx_raw = task.get("context", "{}")
        if isinstance(task_ctx_raw, str):
            task_ctx_parsed = json.loads(task_ctx_raw)
        else:
            task_ctx_parsed = task_ctx_raw
        if task_ctx_parsed.get("silent"):
            logger.info("task completed (silent)", task_id=task_id, model=model, cost=cost)
        elif is_interactive or not is_goal_subtask:
            await self.telegram.send_result(task_id, task["title"],
                                            result_text, model, cost)
        elif iterations > 3:
            ...
```

This replaces the existing notification block — silent tasks log only, no Telegram message.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/core/orchestrator.py src/app/reminders.py
git commit -m "feat(todo): wire suggestion tasks into reminder flow with silent processing"
```

---

### Task 5: Clean up test data from production DB

**Files:**
- None (runtime DB operation)

- [ ] **Step 1: Delete test artifacts**

Run:

```bash
cd C:/Users/sakir/Dropbox/Workspaces/kutay
.venv/Scripts/python.exe -c "
import asyncio, sqlite3
conn = sqlite3.connect('orchestrator.db')
conn.execute('DELETE FROM todo_items')
conn.commit()
count = conn.execute('SELECT changes()').fetchone()[0]
print(f'Deleted {count} test todo items')
conn.close()
"
```

Expected: `Deleted 8 test todo items`

- [ ] **Step 2: Verify clean state**

```bash
.venv/Scripts/python.exe -c "
import sqlite3
conn = sqlite3.connect('orchestrator.db')
print('Todos:', conn.execute('SELECT count(*) FROM todo_items').fetchone()[0])
sched = conn.execute('SELECT id, title, enabled, last_run, next_run FROM scheduled_tasks WHERE id=9999').fetchone()
print(f'Scheduler: id={sched[0]}, title={sched[1]}, enabled={sched[2]}, last_run={sched[3]}, next_run={sched[4]}')
conn.close()
"
```

Expected: `Todos: 0` and scheduler task 9999 present and enabled.

---

### Task 6: End-to-end verification

**Files:** None (manual testing)

- [ ] **Step 1: Add a test todo**

Send `/todo Buy groceries` in Telegram.
Expected: `📝 Added: *Buy groceries*` with a Done button.

- [ ] **Step 2: Trigger a reminder manually**

```bash
cd C:/Users/sakir/Dropbox/Workspaces/kutay
.venv/Scripts/python.exe -c "
import asyncio
from src.infra.db import init_db, get_todos
asyncio.run(init_db())
todos = asyncio.run(get_todos(status='pending'))
print(f'{len(todos)} pending todos')
for t in todos:
    print(f'  #{t[\"id\"]}: {t[\"title\"]}')
"
```

Verify the todo exists, then wait for the next scheduler cycle (60s) or restart KutAI to trigger it.

- [ ] **Step 3: Verify reminder message format**

Expected in Telegram:
- Priority icon, ID, title, age for each item
- Suggestion lines (if AI produced them) with 💡
- Row of ✅ buttons
- Row of 🤖 buttons (if suggestions exist)
- 🔙 Close button at bottom

- [ ] **Step 4: Test Close button**

Tap 🔙 Close.
Expected: Reminder message is deleted.

- [ ] **Step 5: Test Done button**

Wait for next reminder, tap a ✅ button.
Expected: Inline popup "Done!" and todo marked as completed.

- [ ] **Step 6: Test Help button**

Wait for reminder with suggestions, tap 🤖 button.
Expected: Prefilled message with suggestion text, Cancel button. Edit and send → task created.

- [ ] **Step 7: Final commit**

```bash
git add -A
git commit -m "feat(todo): complete todo reminder with AI suggestions, age, compact UX"
```
