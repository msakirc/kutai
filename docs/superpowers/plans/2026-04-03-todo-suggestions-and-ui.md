# Todo Suggestions & UI Redesign

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist AI suggestions on todo items, make the LLM parser robust, and redesign the Telegram todo UI to a list→detail pattern with prefilled keyboard.

**Architecture:** Add `suggestion` and `suggestion_at` columns to `todo_items`. Suggestion states: NULL/NULL = never attempted, NULL/timestamp = failed (don't retry), text/timestamp = cached. The todo list shows numbered buttons; tapping one opens a detail view with Done/Edit/Help/Cancel actions. Help and Edit both prefill the reply keyboard.

**Tech Stack:** Python 3.10, aiosqlite, python-telegram-bot v20+, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/infra/db.py` | Modify | Add columns, update `_TODO_COLUMNS`, add `clear_todo_suggestion()` |
| `src/app/reminders.py` | Rewrite | New list→detail builder, drop old flat format |
| `src/core/orchestrator.py` | Modify | Rewrite `_start_todo_suggestions()` — only generate for NULL suggestions, robust parser |
| `src/app/telegram_bot.py` | Modify | New callback handlers for detail view, edit flow, help flow |
| `tests/test_todo.py` | Modify | Add tests for new columns, suggestion caching, parser |
| `tests/test_todo_ui.py` | Create | Tests for reminder message builder and detail view |

---

### Task 1: Add suggestion columns to DB

**Files:**
- Modify: `src/infra/db.py:307-317` (schema), `src/infra/db.py:1557-1560` (`_TODO_COLUMNS`)
- Test: `tests/test_todo.py`

- [ ] **Step 1: Write failing test for new columns**

In `tests/test_todo.py`, add to `TestTodoDB`:

```python
def test_suggestion_columns_exist(self):
    """Todo items should have suggestion and suggestion_at columns."""
    todo_id = run_async(self.db_mod.add_todo("Test suggestion"))
    todo = run_async(self.db_mod.get_todo(todo_id))
    self.assertIsNone(todo["suggestion"])
    self.assertIsNone(todo["suggestion_at"])

def test_update_todo_suggestion(self):
    """update_todo should accept suggestion and suggestion_at."""
    todo_id = run_async(self.db_mod.add_todo("Suggest me"))
    run_async(self.db_mod.update_todo(
        todo_id,
        suggestion="Compare prices online",
        suggestion_agent="shopping_advisor",
        suggestion_at="2026-04-03 10:00:00",
    ))
    todo = run_async(self.db_mod.get_todo(todo_id))
    self.assertEqual(todo["suggestion"], "Compare prices online")
    self.assertEqual(todo["suggestion_agent"], "shopping_advisor")
    self.assertEqual(todo["suggestion_at"], "2026-04-03 10:00:00")

def test_clear_todo_suggestion(self):
    """Clearing suggestion resets both fields to NULL."""
    todo_id = run_async(self.db_mod.add_todo("Clear me"))
    run_async(self.db_mod.update_todo(
        todo_id,
        suggestion="Old suggestion",
        suggestion_agent="researcher",
        suggestion_at="2026-04-03 10:00:00",
    ))
    run_async(self.db_mod.update_todo(
        todo_id, suggestion=None, suggestion_agent=None, suggestion_at=None,
    ))
    todo = run_async(self.db_mod.get_todo(todo_id))
    self.assertIsNone(todo["suggestion"])
    self.assertIsNone(todo["suggestion_at"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_todo.py::TestTodoDB::test_suggestion_columns_exist tests/test_todo.py::TestTodoDB::test_update_todo_suggestion tests/test_todo.py::TestTodoDB::test_clear_todo_suggestion -v`

Expected: FAIL — columns don't exist yet.

- [ ] **Step 3: Add columns to schema and migration**

In `src/infra/db.py`, update the CREATE TABLE (line ~310):

```sql
CREATE TABLE IF NOT EXISTS todo_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT DEFAULT '',
    priority TEXT DEFAULT 'normal',
    due_date TIMESTAMP,
    status TEXT DEFAULT 'pending',
    source TEXT DEFAULT 'explicit',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    suggestion TEXT,
    suggestion_agent TEXT,
    suggestion_at TIMESTAMP
)
```

Add migration in `init_db()` (after existing todo_items migrations):

```python
# Migration: add suggestion columns to todo_items
for col in ["suggestion", "suggestion_agent", "suggestion_at"]:
    try:
        await db.execute(f"ALTER TABLE todo_items ADD COLUMN {col} TEXT")
    except Exception:
        pass  # already exists
```

Update `_TODO_COLUMNS` (line ~1557):

```python
_TODO_COLUMNS = frozenset({
    "title", "description", "priority", "due_date",
    "status", "source", "completed_at",
    "suggestion", "suggestion_agent", "suggestion_at",
})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_todo.py::TestTodoDB -v`

Expected: ALL PASS

- [ ] **Step 5: Commit**

```
feat(db): add suggestion/suggestion_agent/suggestion_at columns to todo_items
```

---

### Task 2: Rewrite suggestion generator with caching and robust parser

**Files:**
- Modify: `src/core/orchestrator.py:1153-1282` (`_start_todo_suggestions`)
- Test: `tests/test_todo.py`

- [ ] **Step 1: Write test for suggestion parsing**

Add a new test class to `tests/test_todo.py`:

```python
class TestSuggestionParser(unittest.TestCase):
    """Test the LLM response parser for todo suggestions."""

    @staticmethod
    def _parse(raw: str, todo_count: int) -> list[dict]:
        """Import and call the parser."""
        from src.core.orchestrator import _parse_todo_suggestions
        return _parse_todo_suggestions(raw, todo_count)

    def test_standard_format(self):
        raw = "1. [researcher] Search for nearby shops\n2. [shopping_advisor] Compare prices"
        result = self._parse(raw, 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["suggestion"], "Search for nearby shops")
        self.assertEqual(result[0]["agent"], "researcher")
        self.assertEqual(result[1]["agent"], "shopping_advisor")

    def test_parenthesis_format(self):
        raw = "1) [researcher] Search online\n2) [assistant] Draft a message"
        result = self._parse(raw, 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["suggestion"], "Search online")

    def test_no_agent_tag(self):
        raw = "1. Search for nearby shops\n2. Compare prices online"
        result = self._parse(raw, 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["agent"], "researcher")  # default

    def test_no_suggestion(self):
        raw = "1. [researcher] Search shops\n2. No suggestion\n3. [assistant] Help"
        result = self._parse(raw, 3)
        self.assertEqual(len(result), 3)
        self.assertIsNotNone(result[0]["suggestion"])
        self.assertIsNone(result[1]["suggestion"])
        self.assertIsNotNone(result[2]["suggestion"])

    def test_extra_whitespace_and_markdown(self):
        raw = "  1.  **[researcher]** Search for shops  \n  2. [assistant] Help out  "
        result = self._parse(raw, 2)
        self.assertEqual(len(result), 2)
        self.assertIn("Search", result[0]["suggestion"])

    def test_missing_items(self):
        """LLM only returned 2 of 3 items."""
        raw = "1. [researcher] Search\n3. [assistant] Help"
        result = self._parse(raw, 3)
        self.assertEqual(len(result), 3)
        self.assertIsNotNone(result[0]["suggestion"])
        self.assertIsNone(result[1]["suggestion"])  # missing #2
        self.assertIsNotNone(result[2]["suggestion"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_todo.py::TestSuggestionParser -v`

Expected: FAIL — `_parse_todo_suggestions` doesn't exist.

- [ ] **Step 3: Extract parser as module-level function**

In `src/core/orchestrator.py`, add before the `Orchestrator` class (around line 170):

```python
import re as _re

_VALID_SUGGESTION_AGENTS = {"researcher", "shopping_advisor", "assistant", "coder"}

def _parse_todo_suggestions(raw: str, todo_count: int) -> list[dict]:
    """Parse LLM response into per-todo suggestions.

    Returns a list of length todo_count. Each element is:
      {"suggestion": str | None, "agent": str}

    Lenient parser: handles N. or N) prefixes, optional [agent] tags,
    markdown bold around tags, extra whitespace.
    """
    results = [{"suggestion": None, "agent": "researcher"} for _ in range(todo_count)]
    if not raw or not raw.strip():
        return results

    # Build a map: line_number → parsed content
    parsed_lines: dict[int, tuple[str, str]] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Match: optional whitespace, number, . or ), rest
        m = _re.match(r"(\d+)\s*[.)]\s*(.+)", line)
        if not m:
            continue
        idx = int(m.group(1)) - 1  # 0-based
        text = m.group(2).strip()

        # Skip "no suggestion" variants
        if _re.match(r"(?i)no\s+suggestion|n/a|none|-$", text):
            continue
        if len(text) < 6:
            continue

        # Extract [agent_type] — handle optional markdown bold: **[agent]**
        text = _re.sub(r"^\*{1,2}\[", "[", text)
        text = _re.sub(r"\]\*{1,2}", "]", text)
        agent_m = _re.match(r"\[(\w+)\]\s*(.+)", text)
        if agent_m and agent_m.group(1).lower() in _VALID_SUGGESTION_AGENTS:
            agent = agent_m.group(1).lower()
            suggestion = agent_m.group(2).strip()
        else:
            agent = "researcher"
            suggestion = text

        if 0 <= idx < todo_count:
            parsed_lines[idx] = (suggestion, agent)

    for idx, (suggestion, agent) in parsed_lines.items():
        results[idx] = {"suggestion": suggestion, "agent": agent}

    return results
```

- [ ] **Step 4: Run parser tests**

Run: `pytest tests/test_todo.py::TestSuggestionParser -v`

Expected: ALL PASS

- [ ] **Step 5: Rewrite `_start_todo_suggestions` to use caching**

Replace the body of `_start_todo_suggestions` in `src/core/orchestrator.py` (lines ~1153-1282):

```python
async def _start_todo_suggestions(self):
    """Generate AI suggestions for pending todos that don't have one yet.

    Only queries LLM for todos where suggestion IS NULL and suggestion_at IS NULL
    (never attempted). Todos with suggestion_at set but suggestion NULL were
    previously attempted and failed — skip them.
    """
    from src.infra.db import get_todos, update_todo
    from src.app.reminders import send_todo_reminder

    todos = await get_todos(status="pending")
    if not todos:
        return

    # Filter to todos that need suggestions (never attempted)
    need_suggestions = [
        t for t in todos
        if t.get("suggestion") is None and t.get("suggestion_at") is None
    ]

    if need_suggestions:
        await self._generate_suggestions(need_suggestions)

    # Always send the reminder (suggestions are read from DB by reminders.py)
    if self.telegram:
        await send_todo_reminder(self.telegram)

async def _generate_suggestions(self, todos: list[dict]):
    """Call LLM to generate suggestions for given todos, persist results."""
    from src.infra.db import update_todo

    todo_lines = "\n".join(
        f"{i+1}. {t['title']}"
        + (f" (priority: {t.get('priority', 'normal')})" if t.get("priority") != "normal" else "")
        + (f" (notes: {t['description'][:80]})" if t.get("description") else "")
        for i, t in enumerate(todos[:10])
    )
    prompt = (
        f"The user has {len(todos)} pending todo item(s):\n\n"
        f"{todo_lines}\n\n"
        f"For each item, suggest ONE concrete, actionable way an AI assistant could help "
        f"(e.g. search, compare prices, book, remind, draft a message). "
        f"Be creative — even mundane tasks like 'buy milk' could mean price comparison or online ordering. "
        f"If you genuinely cannot help with an item, write 'no suggestion'.\n\n"
        f"Also pick the best agent type for each suggestion:\n"
        f"  researcher — web search, information gathering, fact-checking\n"
        f"  shopping_advisor — product search, price comparison, deal finding\n"
        f"  assistant — drafting messages, reminders, general help\n"
        f"  coder — writing code, scripts, technical tasks\n\n"
        f"Reply ONLY with a numbered list. Format: NUMBER. [agent_type] suggestion text\n"
        f"Example: 1. [researcher] Search for nearby tire shops and compare prices.\n"
        f"No preamble, no extra commentary."
    )

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    try:
        from src.core.llm_dispatcher import get_dispatcher, CallCategory
        from src.core.router import ModelRequirements

        dispatcher = get_dispatcher()
        reqs = ModelRequirements(
            task="adhoc",
            primary_capability="general",
            difficulty=2,
            estimated_input_tokens=400,
            estimated_output_tokens=150,
            prefer_speed=True,
            priority=2,
        )
        response = await asyncio.wait_for(
            dispatcher.request(
                category=CallCategory.OVERHEAD,
                reqs=reqs,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=45,
        )
        raw = (response.get("content") or "").strip()
        logger.info(f"[Todo] Suggestion LLM response ({len(raw)} chars)")

        parsed = _parse_todo_suggestions(raw, len(todos[:10]))

        for i, todo in enumerate(todos[:10]):
            entry = parsed[i]
            if entry["suggestion"]:
                await update_todo(
                    todo["id"],
                    suggestion=entry["suggestion"],
                    suggestion_agent=entry["agent"],
                    suggestion_at=now_str,
                )
            else:
                # Mark as attempted-but-failed so we don't retry
                await update_todo(todo["id"], suggestion_at=now_str)

        generated = sum(1 for p in parsed if p["suggestion"])
        logger.info(f"[Todo] Generated {generated}/{len(todos[:10])} suggestions")

    except asyncio.TimeoutError:
        logger.warning("[Todo] Suggestion LLM call timed out — marking todos as attempted")
        for todo in todos[:10]:
            await update_todo(todo["id"], suggestion_at=now_str)
    except Exception as exc:
        logger.warning(f"[Todo] Suggestion LLM call failed: {exc} — marking todos as attempted")
        for todo in todos[:10]:
            await update_todo(todo["id"], suggestion_at=now_str)
```

- [ ] **Step 6: Remove the dedup sentinel code**

The old sentinel task creation (recording "Todo suggestions batch" in the tasks table) is no longer needed — suggestion_at on each todo is the state. Remove the sentinel block that was after the LLM call (the `add_task`/`update_task` block with `todo_suggest_sentinel`).

- [ ] **Step 7: Run all todo tests**

Run: `pytest tests/test_todo.py -v`

Expected: ALL PASS

- [ ] **Step 8: Commit**

```
feat(todo): persist suggestions on todo_items, robust LLM parser, skip-on-fail caching
```

---

### Task 3: Rewrite reminders.py for list→detail pattern

**Files:**
- Rewrite: `src/app/reminders.py`
- Test: `tests/test_todo_ui.py` (create)

- [ ] **Step 1: Write tests for new message builders**

Create `tests/test_todo_ui.py`:

```python
"""Tests for todo UI message builders."""
import asyncio
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestTodoListMessage(unittest.TestCase):
    """Test the todo list (overview) message builder."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls._tmp.close()
        cls.db_path = cls._tmp.name
        import src.app.config as config
        cls._orig_db_path = config.DB_PATH
        config.DB_PATH = cls.db_path
        import src.infra.db as db_mod
        db_mod.DB_PATH = cls.db_path
        db_mod._db_connection = None
        cls.db_mod = db_mod
        run_async(db_mod.init_db())

    @classmethod
    def tearDownClass(cls):
        run_async(cls.db_mod.close_db())
        import src.app.config as config
        config.DB_PATH = cls._orig_db_path
        try:
            os.unlink(cls.db_path)
        except OSError:
            pass

    def test_empty_list(self):
        from src.app.reminders import build_todo_list_message
        text, markup = run_async(build_todo_list_message())
        self.assertIsNone(text)

    def test_list_shows_numbered_todos(self):
        from src.app.reminders import build_todo_list_message
        run_async(self.db_mod.add_todo("Item A"))
        run_async(self.db_mod.add_todo("Item B"))
        text, markup = run_async(build_todo_list_message())
        self.assertIn("1.", text)
        self.assertIn("2.", text)
        self.assertIn("Item A", text)
        self.assertIn("Item B", text)

    def test_suggestion_hint_shown(self):
        from src.app.reminders import build_todo_list_message
        tid = run_async(self.db_mod.add_todo("Item C"))
        run_async(self.db_mod.update_todo(
            tid, suggestion="Compare prices online for best deals",
            suggestion_agent="shopping_advisor",
            suggestion_at="2026-04-03 10:00:00",
        ))
        text, markup = run_async(build_todo_list_message())
        # Should show truncated suggestion hint
        self.assertIn("Compare prices", text)
        self.assertIn("💡", text)

    def test_numbered_buttons(self):
        """Buttons should be numbered, not action-specific."""
        from src.app.reminders import build_todo_list_message
        text, markup = run_async(build_todo_list_message())
        self.assertIsNotNone(markup)
        # Check inline keyboard has numbered buttons
        buttons = []
        for row in markup.inline_keyboard:
            for btn in row:
                buttons.append(btn.text)
        # Should have at least "1", "2", "3" style buttons
        numbered = [b for b in buttons if b.strip().isdigit()]
        self.assertGreater(len(numbered), 0)


class TestTodoDetailMessage(unittest.TestCase):
    """Test the todo detail view builder."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        cls._tmp.close()
        cls.db_path = cls._tmp.name
        import src.app.config as config
        cls._orig_db_path = config.DB_PATH
        config.DB_PATH = cls.db_path
        import src.infra.db as db_mod
        db_mod.DB_PATH = cls.db_path
        db_mod._db_connection = None
        cls.db_mod = db_mod
        run_async(db_mod.init_db())

    @classmethod
    def tearDownClass(cls):
        run_async(cls.db_mod.close_db())
        import src.app.config as config
        config.DB_PATH = cls._orig_db_path
        try:
            os.unlink(cls.db_path)
        except OSError:
            pass

    def test_detail_with_suggestion(self):
        from src.app.reminders import build_todo_detail_message
        tid = run_async(self.db_mod.add_todo("Buy milk"))
        run_async(self.db_mod.update_todo(
            tid, suggestion="Compare milk prices online",
            suggestion_agent="shopping_advisor",
            suggestion_at="2026-04-03 10:00:00",
        ))
        todo = run_async(self.db_mod.get_todo(tid))
        text, markup = build_todo_detail_message(todo)
        self.assertIn("Buy milk", text)
        self.assertIn("Compare milk prices", text)
        self.assertIn("💡", text)
        # Should have Help button when suggestion exists
        btn_texts = [b.text for row in markup.inline_keyboard for b in row]
        self.assertIn("🤖 Help", btn_texts)
        self.assertIn("✅ Done", btn_texts)
        self.assertIn("✏️ Edit", btn_texts)

    def test_detail_without_suggestion(self):
        from src.app.reminders import build_todo_detail_message
        tid = run_async(self.db_mod.add_todo("Call dentist"))
        todo = run_async(self.db_mod.get_todo(tid))
        text, markup = build_todo_detail_message(todo)
        self.assertIn("Call dentist", text)
        btn_texts = [b.text for row in markup.inline_keyboard for b in row]
        self.assertNotIn("🤖 Help", btn_texts)
        self.assertIn("✅ Done", btn_texts)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_todo_ui.py -v`

Expected: FAIL — `build_todo_detail_message` doesn't exist, list format doesn't match.

- [ ] **Step 3: Rewrite reminders.py**

Replace `build_todo_list_message` and add `build_todo_detail_message` in `src/app/reminders.py`:

```python
async def build_todo_list_message():
    """Build the numbered todo list overview with suggestion hints.

    Returns:
        (text, markup) tuple, or (None, None) if no pending todos.
    """
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    todos = await get_todos(status="pending")
    if not todos:
        return None, None

    lines = ["📝 *Pending Todos*\n"]
    for i, todo in enumerate(todos):
        num = i + 1
        title = todo["title"]
        suggestion = todo.get("suggestion")
        if suggestion:
            hint = suggestion[:40] + ("..." if len(suggestion) > 40 else "")
            lines.append(f"{num}. {title} — 💡 _{hint}_")
        else:
            lines.append(f"{num}. {title}")

    text = "\n".join(lines)

    # Numbered buttons, max 5 per row
    buttons = []
    row = []
    for i, todo in enumerate(todos):
        row.append(InlineKeyboardButton(
            str(i + 1), callback_data=f"todo_detail:{todo['id']}",
        ))
        if len(row) >= 5:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)

    buttons.append([InlineKeyboardButton("🔙 Close", callback_data="todo_close")])
    markup = InlineKeyboardMarkup(buttons)
    return text, markup


def build_todo_detail_message(todo: dict):
    """Build the detail view for a single todo item.

    Returns:
        (text, markup) tuple.
    """
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    tid = todo["id"]
    title = todo["title"]
    priority = todo.get("priority", "normal")
    p_icon = _PRIORITY_ICONS.get(priority, "🟡")
    age = _format_age(todo["created_at"])
    suggestion = todo.get("suggestion")
    agent_type = todo.get("suggestion_agent", "researcher")

    lines = [f"📝 *#{tid}* — {title}"]
    lines.append(f"Priority: {p_icon} {priority} | Added: {age}")

    if todo.get("description"):
        lines.append(f"\n_{todo['description']}_")

    if suggestion:
        lines.append(f"\n💡 {suggestion}")

    text = "\n".join(lines)

    # Action buttons
    action_row = [
        InlineKeyboardButton("✅ Done", callback_data=f"todo_toggle:{tid}"),
        InlineKeyboardButton("✏️ Edit", callback_data=f"todo_edit:{tid}"),
    ]
    if suggestion:
        action_row.append(
            InlineKeyboardButton("🤖 Help", callback_data=f"todo_help:{tid}:{agent_type}")
        )
    action_row.append(
        InlineKeyboardButton("❌ Cancel", callback_data=f"todo_delete:{tid}")
    )

    buttons = [action_row]
    buttons.append([InlineKeyboardButton("🔙 Back", callback_data="todo_list")])

    markup = InlineKeyboardMarkup(buttons)
    return text, markup
```

Update `send_todo_reminder` — remove `suggestions` parameter (suggestions are now in DB):

```python
async def send_todo_reminder(telegram):
    """Fetch pending todos and send the overview list to Telegram."""
    try:
        text, markup = await build_todo_list_message()
        if not text:
            return

        await telegram.app.bot.send_message(
            chat_id=_get_admin_chat_id(),
            text=text,
            parse_mode="Markdown",
            reply_markup=markup,
        )
        logger.info("todo reminder sent")

    except Exception as e:
        logger.error("failed to send todo reminder", error=str(e))
```

- [ ] **Step 4: Run UI tests**

Run: `pytest tests/test_todo_ui.py -v`

Expected: ALL PASS

- [ ] **Step 5: Commit**

```
feat(todo): list→detail UI pattern with suggestion hints and numbered buttons
```

---

### Task 4: Rewire Telegram callback handlers

**Files:**
- Modify: `src/app/telegram_bot.py:5459-5585` (callback handlers)
- Modify: `src/app/telegram_bot.py:4445-4471` (`cmd_todos`)
- Modify: `src/app/telegram_bot.py:6015-6062` (remove `_extract_suggestion_from_message`, `cmd__todo_help`)

- [ ] **Step 1: Add `todo_detail` callback handler**

In `src/app/telegram_bot.py`, in the callback handler section (around line 5459), add before the existing `todo_toggle` handler:

```python
if data.startswith("todo_detail:"):
    try:
        todo_id = int(data.split(":")[1])
    except (ValueError, IndexError):
        await query.answer("Invalid todo ID")
        return
    todo = await get_todo(todo_id)
    if not todo:
        await query.answer("Todo not found")
        return
    from src.app.reminders import build_todo_detail_message
    text, markup = build_todo_detail_message(todo)

    # Prefill keyboard with suggestion if available
    if todo.get("suggestion"):
        chat_id = query.message.chat_id
        self._pending_action[chat_id] = {
            "command": "_todo_help",
            "todo_id": todo_id,
            "todo_title": todo["title"],
            "suggestion": todo["suggestion"],
            "agent_type": todo.get("suggestion_agent", "researcher"),
        }

    try:
        await query.edit_message_text(
            text, parse_mode="Markdown", reply_markup=markup,
        )
    except Exception:
        await query.message.reply_text(
            text, parse_mode="Markdown", reply_markup=markup,
        )
    return
```

- [ ] **Step 2: Add `todo_list` (back) callback handler**

```python
if data == "todo_list":
    from src.app.reminders import build_todo_list_message
    text, markup = await build_todo_list_message()
    if text:
        try:
            await query.edit_message_text(
                text, parse_mode="Markdown", reply_markup=markup,
            )
        except Exception:
            pass
    else:
        await query.edit_message_text("🎉 All todos done!")
    return
```

- [ ] **Step 3: Add `todo_edit` callback handler**

```python
if data.startswith("todo_edit:"):
    try:
        todo_id = int(data.split(":")[1])
    except (ValueError, IndexError):
        await query.answer("Invalid todo ID")
        return
    todo = await get_todo(todo_id)
    if not todo:
        await query.answer("Todo not found")
        return
    chat_id = query.message.chat_id
    self._pending_action[chat_id] = {
        "command": "_todo_edit",
        "todo_id": todo_id,
    }
    # Prefill keyboard with current title for easy editing
    prefill_kb = ReplyKeyboardMarkup(
        [[KeyboardButton(todo["title"])]],
        resize_keyboard=True,
        one_time_keyboard=True,
    )
    await query.message.reply_text(
        f"✏️ Edit todo *#{todo_id}*\nType the new title (keyboard prefilled):",
        parse_mode="Markdown",
        reply_markup=prefill_kb,
    )
    return
```

- [ ] **Step 4: Add `todo_delete` callback handler**

```python
if data.startswith("todo_delete:"):
    try:
        todo_id = int(data.split(":")[1])
    except (ValueError, IndexError):
        await query.answer("Invalid todo ID")
        return
    await delete_todo(todo_id)
    await query.answer("❌ Deleted")
    # Return to list
    from src.app.reminders import build_todo_list_message
    text, markup = await build_todo_list_message()
    if text:
        try:
            await query.edit_message_text(
                text, parse_mode="Markdown", reply_markup=markup,
            )
        except Exception:
            pass
    else:
        await query.edit_message_text("🎉 All todos done!")
    return
```

- [ ] **Step 5: Modify `todo_toggle` to return to list view**

Update the existing `todo_toggle` handler to navigate back to the list after toggling:

```python
if data.startswith("todo_toggle:"):
    try:
        todo_id = int(data.split(":")[1])
    except (ValueError, IndexError):
        await query.answer("Invalid todo ID")
        return
    new_status = await toggle_todo(todo_id)
    icon = "✅" if new_status == "done" else "⬜"
    await query.answer(f"{icon} {'Done!' if new_status == 'done' else 'Reopened'}")
    # Return to list view
    from src.app.reminders import build_todo_list_message
    text, markup = await build_todo_list_message()
    if text:
        try:
            await query.edit_message_text(
                text, parse_mode="Markdown", reply_markup=markup,
            )
        except Exception:
            pass
    else:
        await query.edit_message_text("🎉 All todos done!")
    return
```

- [ ] **Step 6: Modify `todo_help` to read suggestion from DB**

Replace the existing `todo_help` handler — no more message text extraction:

```python
if data.startswith("todo_help:"):
    parts = data.split(":")
    try:
        todo_id = int(parts[1])
    except (ValueError, IndexError):
        await query.answer("Invalid todo ID")
        return
    agent_type = parts[2] if len(parts) > 2 else "researcher"
    todo = await get_todo(todo_id)
    if not todo:
        await query.answer("Todo not found")
        return
    suggestion = todo.get("suggestion") or f"Help me with: {todo['title']}"
    chat_id = query.message.chat_id
    self._pending_action[chat_id] = {
        "command": "_todo_help",
        "todo_id": todo_id,
        "todo_title": todo["title"],
        "suggestion": suggestion,
        "agent_type": agent_type,
    }
    # Prefill keyboard with suggestion
    prefill_kb = ReplyKeyboardMarkup(
        [[KeyboardButton(suggestion[:60] if len(suggestion) > 60 else suggestion)]],
        resize_keyboard=True,
        one_time_keyboard=True,
    )
    await query.message.reply_text(
        f"🤖 *Help with: {todo['title']}*\n\n"
        f"💡 _{suggestion}_\n\n"
        f"Tap the suggestion, type a custom request, or /cancel.",
        parse_mode="Markdown",
        reply_markup=prefill_kb,
    )
    return
```

- [ ] **Step 7: Add `_todo_edit` pending action handler**

In the `_pending_action` resolution section (around line 3662), add handling for the edit action. Register `_handle_todo_edit` method:

```python
async def _handle_todo_edit(self, update, context):
    """Handle the user's reply with a new todo title."""
    chat_id = update.effective_chat.id
    pending = self._pending_action.pop(chat_id, None)
    if not pending:
        return
    todo_id = pending["todo_id"]
    new_title = update.message.text.strip()
    if not new_title:
        await self._reply(update, "Title can't be empty.")
        return
    # Reset suggestion so it gets regenerated next cycle
    await update_todo(
        todo_id, title=new_title,
        suggestion=None, suggestion_agent=None, suggestion_at=None,
    )
    await self._reply(update, f"✏️ Updated *#{todo_id}*: {new_title}", parse_mode="Markdown")
```

- [ ] **Step 8: Update `cmd_todos` to use new list builder**

Replace `cmd_todos` (line ~4445):

```python
async def cmd_todos(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """List all todos. /todos"""
    from src.app.reminders import build_todo_list_message
    text, markup = await build_todo_list_message()
    if not text:
        await self._reply(update, "📋 No todo items yet. Use /todo to add one.")
        return
    await self._reply(update, text, parse_mode="Markdown", reply_markup=markup)
```

- [ ] **Step 9: Remove `_extract_suggestion_from_message`**

Delete the `_extract_suggestion_from_message` static method (line ~6015-6026) — no longer needed since suggestions come from DB.

- [ ] **Step 10: Remove old `todo_help_accept` and `todo_help_cancel` handlers if superseded**

The accept/cancel inline buttons are replaced by the prefilled keyboard flow. Remove the `todo_help_accept:` and `todo_help_cancel` callback handlers (lines ~5547-5585) if they're no longer referenced.

- [ ] **Step 11: Register `_todo_edit` command resolution**

In the `_pending_action` resolver (around line 3662), add `_todo_edit` to the command dispatch map so it calls `_handle_todo_edit`.

- [ ] **Step 12: Verify with manual import check**

Run: `python -c "from src.app.telegram_bot import TelegramInterface; print('OK')"`

Note: This may fail if python-telegram-bot is not installed in the base Python. As a fallback:

Run: `python -c "import ast; ast.parse(open('src/app/telegram_bot.py', encoding='utf-8').read()); print('OK')"`

Expected: OK

- [ ] **Step 13: Commit**

```
feat(todo): rewire Telegram callbacks for list→detail navigation with edit flow
```

---

### Task 5: Wire everything together and cleanup

**Files:**
- Modify: `src/core/orchestrator.py` (remove old dedup guard)
- Modify: `src/app/telegram_bot.py` (ensure toggle clears suggestion on reopen)
- Modify: `src/infra/db.py` (`toggle_todo` clears suggestion on done→pending)

- [ ] **Step 1: Clear suggestion when todo is reopened**

In `src/infra/db.py`, modify `toggle_todo` — when toggling from done→pending, clear suggestion fields so it gets regenerated:

```python
if current == "done":
    new_status = "pending"
    await db.execute(
        """UPDATE todo_items SET status = 'pending', completed_at = NULL,
           suggestion = NULL, suggestion_agent = NULL, suggestion_at = NULL
           WHERE id = ?""",
        (todo_id,),
    )
```

- [ ] **Step 2: Remove old dedup guard from `_start_todo_suggestions`**

In `src/core/orchestrator.py`, remove the block that queries `tasks` table for "Todo suggestions batch" sentinel (the guard that checked for recent suggestion batches). This is replaced by per-todo `suggestion_at` state.

- [ ] **Step 3: Remove old `send_todo_reminder` call with suggestions parameter**

In `check_scheduled_tasks` fallback path (around line 992-998), update the fallback `send_todo_reminder` call to not pass suggestions:

```python
# Fallback: send reminder without suggestions
try:
    from src.app.reminders import send_todo_reminder
    if self.telegram:
        await send_todo_reminder(self.telegram)
except Exception:
    pass
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/test_todo.py tests/test_todo_ui.py -v`

Expected: ALL PASS

- [ ] **Step 5: Commit**

```
feat(todo): clear suggestions on reopen, remove old dedup sentinel
```

---

### Task 6: Update existing tests and final verification

**Files:**
- Modify: `tests/test_todo.py` (update any tests that reference old suggestion format)
- Modify: `tests/test_real_pipeline.py` (ensure TestTodoDB still passes with new columns)

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/test_todo.py tests/test_todo_ui.py tests/test_real_pipeline.py::TestTodoDB -v`

Expected: ALL PASS

- [ ] **Step 2: Verify no leaked test data**

```python
python -c "
import sqlite3, os, sys
sys.stdout.reconfigure(encoding='utf-8')
db_path = 'kutai.db'
if os.path.exists('.env'):
    for line in open('.env'):
        if line.startswith('DB_PATH='):
            db_path = line.strip().split('=',1)[1].strip('\"').strip(\"'\")
conn = sqlite3.connect(db_path)
rows = conn.execute('SELECT id, title, suggestion, suggestion_at FROM todo_items ORDER BY id').fetchall()
for r in rows:
    print(f'ID={r[0]} | {r[1]} | sugg={r[2]} | at={r[3]}')
conn.close()
"
```

- [ ] **Step 3: Commit**

```
test(todo): verify full test suite passes with new suggestion columns and UI
```

---

## Summary of Changes

| What | Before | After |
|------|--------|-------|
| Suggestion storage | In-memory dict, lost on restart | Persisted on `todo_items` row |
| LLM calls | Every 2h for all todos | Only for todos with `suggestion IS NULL AND suggestion_at IS NULL` |
| Failed suggestions | Silent, retried every 2h | Marked with `suggestion_at`, not retried until edited |
| LLM parser | Rigid regex, silent failure | Lenient parser, handles formatting variants |
| Todo list UI | Flat list with ✅ and 🤖 buttons for each | Numbered list → tap number → detail view |
| Help flow | Extract suggestion from message text | Read from DB, prefill keyboard |
| Edit flow | Not supported | Prefills keyboard with title, clears suggestion for regeneration |
| Dedup sentinel | Task in tasks table | Not needed (per-todo state) |
