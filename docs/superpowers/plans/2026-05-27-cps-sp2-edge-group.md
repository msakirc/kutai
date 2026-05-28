# CPS SP2 — Edge group migration: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the 5-of-6 edge-group `await_inline=True` callsites (telegram, interview, meetings, faq_regen, investor_bullets) off `await_inline` and onto the SP1 durable substrate (`on_complete` / `on_error` / `cont_state` continuations). Site #2 (`task_classifier._enqueue_inline_classifier`) is **explicitly deferred** to SP5-or-later per the spec's Site 2 special case — its migration would require breaking `classify_task`'s synchronous contract.

**Architecture:** Each migrated module exposes a `register_continuations()` function called at import; resume + on_error handlers live in the same module as the original synchronous code. The module name is added to `_HANDLER_MODULES` in `packages/general_beckman/src/general_beckman/continuations.py` so restart-reconcile can find the handlers. **Zero changes to the substrate itself** (SP1 is locked in). **`await_inline` body stays intact** (SP5 deletes it).

**Tech stack:** Python 3.10 async, aiosqlite, pytest-asyncio. Substrate package `general_beckman`. DB layer `src/infra/db.py`.

**Spec:** `docs/superpowers/specs/2026-05-27-cps-sp2-edge-group-design.md`.

**Conventions (project rules — non-negotiable):**
- Run tests with a timeout prefix: `timeout 60 pytest tests/... -v` (zombie pytest holds SQLite write locks and crash-loops KutAI).
- SQLite datetime is `strftime('%Y-%m-%d %H:%M:%S')` / `datetime('now')` — never `isoformat()`.
- Prefix git commands with `rtk` (token-optimized passthrough).
- Lazy cross-module imports (inside functions) to avoid circular imports.
- One commit per task; commit message format `feat(<scope>): migrate <site> to CPS (SP2 Task N)`.
- Each task: failing test first → minimal implementation → green → commit.

**Open questions to resolve before starting (founder):**
1. **Site #2 deletion or deferral?** Spec proposes deferring `_classify_with_llm` to SP5; alternative is deleting the LLM classifier entirely and relying on keyword-only classification in `task_classifier`. **Plan default: defer.**
2. **Site #6 6A vs 6B?** Spec proposes 6B (cap at 1 hypothesis, fold finalize into resume). 6A adds a `investor_bullet_pending` table + counter-fires-finalize. **Plan default: 6B.**

---

## File structure

| File | Responsibility | Change |
|------|----------------|--------|
| `src/app/telegram_bot.py` | Add `_send_telegram_via_resume` helper; refactor `_classify_user_message` to enqueue+return (CPS); replace `_handle_casual` body with CPS enqueue; add `_route_classified_message`; add resume handlers + `register_continuations()`. Delete `_enqueue_inline_chat` (the obsolete helper). | Modify |
| `src/app/interview.py` | Replace `summarize_interview`'s `await_inline=True` with `on_complete`; add `interview.summary_persist_resume` + `_err`; add `register_continuations()`. | Modify |
| `src/app/meetings.py` | Replace `_call_llm_meeting_brief`'s `await_inline=True` with `on_complete`; rename to `enqueue_meeting_brief(ctx, meeting_id, product_id)`; add `meetings.brief_persist_resume` + `_err`; add `register_continuations()`. | Modify |
| `packages/mr_roboto/src/mr_roboto/__init__.py` | `meetings/brief` action handler (~3955-) and `interview/summarize` action handler (~4122-): rebuild around enqueue-and-return rather than await-result. | Modify |
| `src/app/jobs/faq_regen.py` | Replace `_llm_cluster_draft`'s `await_inline=True` with `on_complete`; rename to `enqueue_cluster_draft(cluster, lang)`; add `faq_regen.draft_persist_resume` + `_err`; add `register_continuations()`; adjust `run_faq_regen` accounting. | Modify |
| `src/app/jobs/investor_bullets.py` | Replace `_call_llm_anomaly_hypothesis`'s `await_inline=True` with `on_complete` (6B simplification: cap at 1 anomaly); rename to `enqueue_anomaly_hypothesis(...)`; add `investor_bullets.hypothesis_persist_resume` + `_err`; add `register_continuations()`; adjust `run_investor_bullets` to enqueue + finalize-in-resume. | Modify |
| `packages/general_beckman/src/general_beckman/continuations.py` | Extend `_HANDLER_MODULES` with the 5 new modules. | Modify |
| `tests/app/test_cps_sp2_telegram.py` | New site #1 tests (3 sub-cases: classify, casual, cmd_mission). | Create |
| `tests/app/test_cps_sp2_interview.py` | New site #3 tests. | Create |
| `tests/app/test_cps_sp2_meetings.py` | New site #4 tests. | Create |
| `tests/app/test_cps_sp2_faq_regen.py` | New site #5 tests + restart-reconcile end-to-end. | Create |
| `tests/app/test_cps_sp2_investor_bullets.py` | New site #6 tests (6B path). | Create |

---

## Task 1: Telegram keystone — `_send_telegram_via_resume` + `_handle_casual` migration

**Files:**
- Modify: `src/app/telegram_bot.py` — add helper, migrate `_handle_casual`, add `register_continuations()` (partial — Task 1 wires only the casual-reply handler).
- Modify: `packages/general_beckman/src/general_beckman/continuations.py` — add `"src.app.telegram_bot"` to `_HANDLER_MODULES`.
- Test: `tests/app/test_cps_sp2_telegram.py` (new file, will be extended by later sub-tasks).

**Pre-read (absolute paths):**
- `C:\Users\sakir\Dropbox\Workspaces\kutay\src\app\telegram_bot.py` — re-read lines 80–135 (`_enqueue_inline_chat` helper), 7900–7930 (`_handle_casual`), 560–590 (`_reply`), 346–365 (`get_telegram`).
- `C:\Users\sakir\Dropbox\Workspaces\kutay\packages\general_beckman\src\general_beckman\continuations.py` — full file (~150 lines).
- `C:\Users\sakir\Dropbox\Workspaces\kutay\packages\mr_roboto\src\mr_roboto\executors\analytics_digest.py` — lines 395–420 (the `register_continuations()` pattern to copy).

- [ ] **Step 1: Write the failing test**

Create `tests/app/test_cps_sp2_telegram.py`:

```python
"""SP2 Task 1: Telegram casual-reply CPS migration."""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

import src.infra.db as _db_mod


async def _fresh_db(tmp_path, monkeypatch):
    db_file = tmp_path / "sp2.db"
    monkeypatch.setattr(_db_mod, "DB_PATH", str(db_file))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
    monkeypatch.setattr(_db_mod, "_db_connection", None)
    await _db_mod.init_db()


async def _close_db():
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None


@pytest.mark.asyncio
async def test_handle_casual_enqueues_with_on_complete(tmp_path, monkeypatch):
    """_handle_casual must enqueue with on_complete, NOT await_inline."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        captured = {}

        async def fake_enqueue(spec, **kwargs):
            captured["spec"] = spec
            captured["kwargs"] = kwargs
            return 4242  # child task id

        monkeypatch.setattr("general_beckman.enqueue", fake_enqueue)

        from src.app.telegram_bot import TelegramInterface
        bot = object.__new__(TelegramInterface)

        fake_update = MagicMock()
        fake_update.effective_chat.id = 12345
        fake_update.message.chat.id = 12345

        await bot._handle_casual("Hey, how are you?", fake_update)

        assert captured["kwargs"].get("await_inline") in (False, None), (
            f"await_inline must NOT be set; got {captured['kwargs']!r}"
        )
        assert captured["kwargs"]["on_complete"] == "telegram.casual_reply_resume"
        assert captured["kwargs"]["on_error"] == "telegram.casual_reply_err"
        cs = captured["kwargs"]["cont_state"]
        assert cs["chat_id"] == 12345
        assert cs["text"] == "Hey, how are you?"
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_casual_reply_resume_sends_telegram_message(tmp_path, monkeypatch):
    """Resume must extract content from result['result']['content'] and send to chat_id."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from src.app.telegram_bot import TelegramInterface
        # Build a fake bot with a mock app.bot.send_message.
        bot = object.__new__(TelegramInterface)
        bot.app = MagicMock()
        bot.app.bot.send_message = AsyncMock()
        from src.app.telegram_bot import set_telegram
        set_telegram(bot)

        from src.app.telegram_bot import _casual_reply_resume  # registered handler
        await _casual_reply_resume(
            child_task_id=4242,
            result={"status": "completed",
                    "result": {"content": "I am fine, thanks!"}},
            state={"chat_id": 12345, "text": "Hey, how are you?"},
        )
        bot.app.bot.send_message.assert_awaited_once()
        ((), kw) = bot.app.bot.send_message.call_args.args, bot.app.bot.send_message.call_args.kwargs
        assert bot.app.bot.send_message.call_args.kwargs["chat_id"] == 12345
        assert "I am fine" in bot.app.bot.send_message.call_args.kwargs["text"]
    finally:
        from src.app.telegram_bot import set_telegram
        set_telegram(None)  # type: ignore[arg-type]
        await _close_db()


@pytest.mark.asyncio
async def test_casual_reply_err_sends_fallback_text(tmp_path, monkeypatch):
    """on_error sends the documented fallback string."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from src.app.telegram_bot import TelegramInterface
        bot = object.__new__(TelegramInterface)
        bot.app = MagicMock()
        bot.app.bot.send_message = AsyncMock()
        from src.app.telegram_bot import set_telegram
        set_telegram(bot)

        from src.app.telegram_bot import _casual_reply_err
        await _casual_reply_err(
            child_task_id=4242,
            result={"status": "failed", "error": "timeout"},
            state={"chat_id": 12345, "text": "Hi"},
        )
        bot.app.bot.send_message.assert_awaited_once()
        sent_text = bot.app.bot.send_message.call_args.kwargs["text"]
        assert "task or mission" in sent_text.lower() or "send me" in sent_text.lower()
    finally:
        from src.app.telegram_bot import set_telegram
        set_telegram(None)  # type: ignore[arg-type]
        await _close_db()


@pytest.mark.asyncio
async def test_register_continuations_registers_casual_handlers(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from src.app.telegram_bot import register_continuations
        from general_beckman.continuations import _HANDLERS, register_resume
        # Clear and re-register.
        _HANDLERS.pop("telegram.casual_reply_resume", None)
        _HANDLERS.pop("telegram.casual_reply_err", None)
        register_continuations()
        assert "telegram.casual_reply_resume" in _HANDLERS
        assert "telegram.casual_reply_err" in _HANDLERS
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_telegram_module_added_to_handler_modules():
    """Restart-reconcile import list must include telegram_bot."""
    from general_beckman.continuations import _HANDLER_MODULES
    assert "src.app.telegram_bot" in _HANDLER_MODULES, (
        f"_HANDLER_MODULES = {_HANDLER_MODULES!r}"
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `timeout 60 pytest tests/app/test_cps_sp2_telegram.py -v`
Expected: ALL 5 FAIL — `cannot import _casual_reply_resume`, missing `register_continuations`, `_HANDLER_MODULES` does not contain `src.app.telegram_bot`.

- [ ] **Step 3: Add `_send_telegram_via_resume` helper**

In `src/app/telegram_bot.py`, after `set_telegram` (line ~363) and before `enqueue_launch_mission` (line ~366), add:

```python
async def _send_telegram_via_resume(
    chat_id: int, text: str, *, parse_mode: str | None = None
) -> bool:
    """Send a Telegram message from a CPS resume context.

    Resumes have no live `Update` object — they must reach the bot via the
    module-level singleton and the raw bot API. Returns False if the bot is
    uninitialised or the chat is unreachable (silent drop — matches the
    pre-CPS `await_inline` 'reply got lost' behaviour, e.g. user blocked
    the bot between enqueue and terminal).
    """
    try:
        tg = get_telegram()
    except RuntimeError:
        return False
    if tg is None or not getattr(tg, "app", None):
        return False
    try:
        kwargs = {"chat_id": chat_id, "text": text[:4000]}
        if parse_mode:
            kwargs["parse_mode"] = parse_mode
        await tg.app.bot.send_message(**kwargs)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("resume telegram send failed",
                     chat_id=chat_id, error=str(exc))
        return False
```

- [ ] **Step 4: Add `_casual_reply_resume` and `_casual_reply_err`**

In `src/app/telegram_bot.py`, *before* the `TelegramInterface` class (after the helper from Step 3), add:

```python
async def _casual_reply_resume(
    child_task_id: int, result: dict, state: dict
) -> None:
    """CPS resume for `_handle_casual` (SP2 Task 1).

    state = {"chat_id": int, "text": str}
    """
    chat_id = state.get("chat_id")
    if chat_id is None:
        logger.debug("casual_reply_resume: missing chat_id in state",
                     child_task_id=child_task_id)
        return
    agent_output = (result or {}).get("result") or {}
    if isinstance(agent_output, dict):
        content = agent_output.get("content", "")
    else:
        content = str(agent_output)
    if isinstance(content, list):  # multimodal pieces — flatten
        content = "\n".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in content
        )
    reply = (content or "Hey! How can I help?").strip()
    await _send_telegram_via_resume(chat_id, reply[:1000])


async def _casual_reply_err(
    child_task_id: int, result: dict, state: dict
) -> None:
    """CPS on_error for `_handle_casual` (SP2 Task 1)."""
    chat_id = state.get("chat_id")
    if chat_id is None:
        return
    await _send_telegram_via_resume(
        chat_id, "Hey! Send me a task or mission to work on."
    )
```

- [ ] **Step 5: Add `register_continuations()`**

In `src/app/telegram_bot.py`, immediately after `_casual_reply_err`, add:

```python
def register_continuations() -> None:
    """Register all Telegram CPS resume / on_error handlers (SP2).

    Called at import-time AND by `register_startup_handlers()` for
    restart-reconcile. Idempotent.
    """
    try:
        from general_beckman.continuations import register_resume
        register_resume("telegram.casual_reply_resume", _casual_reply_resume)
        register_resume("telegram.casual_reply_err",    _casual_reply_err)
        # Sub-tasks 1.5 / 1.6 add `telegram.message_route_resume` etc. here.
    except Exception as exc:  # noqa: BLE001
        logger.debug("telegram continuation registration deferred",
                     error=str(exc))


# Register at import so restart-reconcile finds these handlers.
register_continuations()
```

- [ ] **Step 6: Migrate `_handle_casual` to CPS**

In `src/app/telegram_bot.py`, replace the body of `_handle_casual` (currently lines 7904-7927) with:

```python
    async def _handle_casual(self, text: str, update: Update):
        """Handle casual messages with a quick LLM response (no task creation).

        SP2: enqueues an LLM child with on_complete + on_error; returns
        immediately. The eventual reply is sent from `_casual_reply_resume`
        via the module-level Telegram singleton (the live `update` object
        is dead by then).
        """
        try:
            from general_beckman import enqueue
            chat_id = update.effective_chat.id
            await enqueue(
                {
                    "title": "telegram-casual-chat",
                    "description": f"Casual chat reply: {text[:80]!r}",
                    "agent_type": "assistant",
                    "kind": "chat",
                    "context": {
                        "llm_call": {
                            "raw_dispatch": True,
                            "task": "assistant",
                            "agent_type": "assistant",
                            "difficulty": 2,
                            "messages": [{"role": "user", "content": text}],
                            "prefer_speed": True,
                            "priority": 1,
                            "estimated_input_tokens": 100,
                            "estimated_output_tokens": 100,
                            "call_category": "overhead",
                        },
                    },
                },
                on_complete="telegram.casual_reply_resume",
                on_error="telegram.casual_reply_err",
                cont_state={"chat_id": chat_id, "text": text},
            )
        except Exception as exc:
            # Local failure (DB write, validation) — fall through to the
            # synchronous fallback so the user always sees *something*.
            logger.debug("casual chat enqueue failed", error=str(exc))
            await self._reply(update, "Hey! Send me a task or mission to work on.")
```

- [ ] **Step 7: Extend `_HANDLER_MODULES`**

In `packages/general_beckman/src/general_beckman/continuations.py`, update `_HANDLER_MODULES` (~line 119) to:

```python
_HANDLER_MODULES = (
    "mr_roboto.executors.analytics_digest",
    "mr_roboto.executors.classify_signals",
    # CPS SP2 — edge-group migrations:
    "src.app.telegram_bot",
)
```

(Subsequent tasks append `src.app.interview`, `src.app.meetings`, `src.app.jobs.faq_regen`, `src.app.jobs.investor_bullets` as they migrate.)

- [ ] **Step 8: Run the Task 1 tests to verify PASS**

Run: `timeout 60 pytest tests/app/test_cps_sp2_telegram.py -v`
Expected: 5 PASS.

- [ ] **Step 9: Regression sweep — telegram tests**

Run: `timeout 60 pytest tests/app/test_telegram_casual_chat_enqueue.py -v`
Expected: This legacy SP1-shim test will FAIL (it asserted `await_inline=True`). **Update the legacy test** to assert the new CPS shape — i.e. swap the assertion from `kwargs.get("await_inline") is True` to `kwargs.get("on_complete") == "telegram.casual_reply_resume"`. Run again. Expected: PASS.

> Per `[[feedback_no_agent_modes]]`, we do not maintain a dual-mode contract; the old `await_inline` assertion is dead and must be replaced, not coexisted-with.

- [ ] **Step 10: Commit**

```bash
rtk git add src/app/telegram_bot.py packages/general_beckman/src/general_beckman/continuations.py tests/app/test_cps_sp2_telegram.py tests/app/test_telegram_casual_chat_enqueue.py
rtk git commit -m "feat(telegram): migrate _handle_casual to CPS (SP2 Task 1)"
```

---

## Task 1.5: Telegram `_classify_user_message` migration + `_route_classified_message` extraction

**Files:**
- Modify: `src/app/telegram_bot.py` — extract `_route_classified_message`, refactor `_classify_user_message` to CPS, add resume + on_error for `telegram.message_route_*`.
- Modify: `tests/app/test_cps_sp2_telegram.py` — append site-#1a tests.

**Pre-read:**
- `C:\Users\sakir\Dropbox\Workspaces\kutay\src\app\telegram_bot.py` — re-read lines 6657 (`handle_message` start) through 7270 (the full routing chain). The block at 7185-7254 (route by classification) is what gets extracted into `_route_classified_message`.
- `C:\Users\sakir\Dropbox\Workspaces\kutay\src\app\telegram_bot.py:7358-7412` — current `_classify_user_message`.
- The spec's §The keystone subsection.

- [ ] **Step 1: Write failing tests (append to `tests/app/test_cps_sp2_telegram.py`)**

```python
@pytest.mark.asyncio
async def test_classify_user_message_enqueues_with_cps(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        captured = {}

        async def fake_enqueue(spec, **kwargs):
            captured["spec"] = spec
            captured["kwargs"] = kwargs
            return 5151

        monkeypatch.setattr("general_beckman.enqueue", fake_enqueue)

        from src.app.telegram_bot import TelegramInterface
        bot = object.__new__(TelegramInterface)
        bot._pending_clarifications = {}

        # The fn used to RETURN dict; now returns None (queued).
        rv = await bot._classify_user_message("How is the coffee mission going?", chat_id=999)
        assert rv is None, f"_classify_user_message must return None (queued), got {rv!r}"
        assert captured["kwargs"]["on_complete"] == "telegram.message_route_resume"
        assert captured["kwargs"]["on_error"] == "telegram.message_route_err"
        cs = captured["kwargs"]["cont_state"]
        assert cs["chat_id"] == 999
        assert cs["text"] == "How is the coffee mission going?"
        assert cs["flow"] == "message_route"
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_message_route_resume_routes_via_extracted_helper(tmp_path, monkeypatch):
    """Resume must call `_route_classified_message` with parsed classification."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from src.app.telegram_bot import _message_route_resume, TelegramInterface, set_telegram
        bot = object.__new__(TelegramInterface)
        bot.app = MagicMock()
        bot.app.bot.send_message = AsyncMock()
        bot.user_last_task_id = {}
        bot._pending_clarifications = {}
        bot._route_classified_message = AsyncMock()
        set_telegram(bot)

        await _message_route_resume(
            child_task_id=5151,
            result={"status": "completed",
                    "result": {"content": '{"type": "casual", "confidence": 0.9}'}},
            state={"chat_id": 999, "text": "hi", "flow": "message_route"},
        )

        bot._route_classified_message.assert_awaited_once()
        ((args), _) = bot._route_classified_message.call_args.args, bot._route_classified_message.call_args.kwargs
        # signature: _route_classified_message(chat_id, text, classification)
        assert args[0] == 999
        assert args[1] == "hi"
        assert args[2]["type"] == "casual"
    finally:
        from src.app.telegram_bot import set_telegram
        set_telegram(None)  # type: ignore[arg-type]
        await _close_db()
```

- [ ] **Step 2: Verify tests fail**

Run: `timeout 60 pytest tests/app/test_cps_sp2_telegram.py::test_classify_user_message_enqueues_with_cps tests/app/test_cps_sp2_telegram.py::test_message_route_resume_routes_via_extracted_helper -v`
Expected: 2 FAIL.

- [ ] **Step 3: Extract `_route_classified_message`**

In `src/app/telegram_bot.py`, add a new method on `TelegramInterface` (place it directly above `_classify_user_message` at line 7358):

```python
    async def _route_classified_message(
        self,
        chat_id: int,
        text: str,
        classification: dict,
    ) -> None:
        """Route a classified message to its handler.

        Extracted from the inline block at handle_message:7185-7254 so it can
        be invoked from a CPS resume context — where there is no live `Update`
        object and replies must use `bot.app.bot.send_message(chat_id, ...)`.
        """
        msg_type = classification.get("type") or "task"
        msg_workflow = classification.get("workflow")
        logger.info("message classified (cps)", msg_type=msg_type,
                    workflow=msg_workflow, text_preview=text[:50])

        if msg_type in ("bug_report", "feature_request", "ui_note", "feedback"):
            await self._handle_user_input_resume(msg_type, text, chat_id)
            return
        if msg_type in ("progress_inquiry", "status_query"):
            await self._handle_status_query_resume(text, chat_id)
            return
        if msg_type == "question":
            from src.infra.db import add_task
            task_id = await add_task(
                title=f"Q: {text[:50]}", description=text,
                tier="auto", priority=TASK_PRIORITY.get("high", 8),
                agent_type="assistant",
            )
            self.user_last_task_id[chat_id] = task_id
            await _send_telegram_via_resume(chat_id, f"❓ Task #{task_id} queued.")
            return
        if msg_type == "shopping":
            from src.infra.db import add_task
            task_id = await add_task(
                title=text[:80], description=text, tier="auto",
                priority=TASK_PRIORITY.get("high", 8),
                agent_type="shopping_advisor",
                context={"chat_id": chat_id},
            )
            if task_id is None:
                await _send_telegram_via_resume(
                    chat_id, "🛒 A shopping search for this is already in progress."
                )
                return
            self.user_last_task_id[chat_id] = task_id
            await _send_telegram_via_resume(
                chat_id,
                f"🛒 Shopping task #{task_id} queued.\n"
                "I'll search prices and compare options for you.",
            )
            return
        if msg_type == "casual":
            # Recursively enqueue casual reply via CPS (no chained await).
            from general_beckman import enqueue
            await enqueue(
                {
                    "title": "telegram-casual-chat",
                    "description": f"Casual chat reply: {text[:80]!r}",
                    "agent_type": "assistant",
                    "kind": "chat",
                    "context": {"llm_call": {
                        "raw_dispatch": True, "task": "assistant",
                        "agent_type": "assistant", "difficulty": 2,
                        "messages": [{"role": "user", "content": text}],
                        "prefer_speed": True, "priority": 1,
                        "estimated_input_tokens": 100,
                        "estimated_output_tokens": 100,
                        "call_category": "overhead",
                    }},
                },
                on_complete="telegram.casual_reply_resume",
                on_error="telegram.casual_reply_err",
                cont_state={"chat_id": chat_id, "text": text},
            )
            return
        # Fall-through: treat as plain task.
        from src.infra.db import add_task
        task_id = await add_task(
            title=text[:80], description=text, tier="auto",
            priority=TASK_PRIORITY.get("normal", 5),
            agent_type="executor",
            context={"chat_id": chat_id},
        )
        if task_id is not None:
            self.user_last_task_id[chat_id] = task_id
            await _send_telegram_via_resume(chat_id, f"📋 Task #{task_id} queued.")
```

> NOTE: this introduces two helpers (`_handle_user_input_resume`, `_handle_status_query_resume`) that mirror the synchronous-with-Update versions. For SP2 simplicity, **these are thin shims**: each one looks up the chat_id, builds whatever DB state it needs, and replies via `_send_telegram_via_resume`. The existing `_handle_user_input` and `_handle_status_query` keep their `Update`-taking signatures intact for the rest of the bot. **If founder review finds this too much duplication, fold the `Update`-aware sites into the resume path in a follow-up.** This task's tests only assert that the routing dispatches to the casual path; the other branches are validated by manual smoke (Task 8).

- [ ] **Step 4: Refactor `_classify_user_message` to CPS**

Replace the body of `_classify_user_message` (currently 7358-7412) with:

```python
    async def _classify_user_message(
        self, text: str, *, chat_id: int | None = None
    ) -> None:
        """Enqueue classification + return immediately (SP2 CPS).

        After classification, the resume routes the message via
        `_route_classified_message`. **No return value** — the answer arrives
        on the user's chat asynchronously. Callers that previously used the
        return value (cmd_mission at line 2204) must call this with chat_id
        and trust the resume to drive routing; see Task 1.6.
        """
        try:
            context_parts = []
            if self._pending_clarifications:
                context_parts.append("System has pending clarification requests")
            messages = [{
                "role": "user",
                "content": self.MESSAGE_CLASSIFIER_PROMPT.format(
                    message=text[:300],
                    context="; ".join(context_parts) if context_parts else "none",
                ),
            }]
            from general_beckman import enqueue
            await enqueue(
                {
                    "title": "telegram-classifier",
                    "description": f"Classify telegram message: {text[:80]!r}",
                    "agent_type": "classifier",
                    "kind": "classifier",
                    "context": {"llm_call": {
                        "raw_dispatch": True, "task": "router",
                        "agent_type": "classifier", "difficulty": 2,
                        "messages": messages, "prefer_speed": True,
                        "needs_json_mode": True, "priority": 2,
                        "estimated_input_tokens": 300,
                        "estimated_output_tokens": 50,
                        "call_category": "overhead",
                    }},
                },
                on_complete="telegram.message_route_resume",
                on_error="telegram.message_route_err",
                cont_state={
                    "chat_id": chat_id, "text": text,
                    "flow": "message_route",
                },
            )
        except Exception as exc:
            logger.debug("classification enqueue failed, keyword fallback",
                         error=str(exc))
            # Local failure — route via keyword on the spot.
            if chat_id is not None:
                cls = self._classify_message_by_keywords(text)
                await self._route_classified_message(chat_id, text, cls)
```

- [ ] **Step 5: Add the resume + on_error handlers (module-level)**

In `src/app/telegram_bot.py`, near `_casual_reply_resume`, add:

```python
async def _message_route_resume(
    child_task_id: int, result: dict, state: dict
) -> None:
    """CPS resume for `_classify_user_message` (SP2 Task 1.5).

    Parses the classification JSON from the LLM response and dispatches via
    `TelegramInterface._route_classified_message`. Falls back to keyword
    classification if JSON parse fails.
    """
    chat_id = state.get("chat_id")
    text = state.get("text") or ""
    if chat_id is None:
        return
    try:
        tg = get_telegram()
    except RuntimeError:
        return
    if tg is None:
        return

    # Parse classification.
    agent_output = (result or {}).get("result") or {}
    content = (
        agent_output.get("content", "")
        if isinstance(agent_output, dict) else str(agent_output)
    )
    classification: dict
    try:
        from src.core.task_classifier import _extract_json
        parsed = _extract_json((content or "").strip())
        msg_type = parsed.get("type") or "task"
        confidence = parsed.get("confidence", 0.5)
        workflow = parsed.get("workflow")
        if confidence < 0.4:
            classification = tg._classify_message_by_keywords(text)
        else:
            classification = {"type": msg_type}
            if workflow:
                classification["workflow"] = workflow
    except Exception:
        classification = tg._classify_message_by_keywords(text)

    await tg._route_classified_message(chat_id, text, classification)


async def _message_route_err(
    child_task_id: int, result: dict, state: dict
) -> None:
    """CPS on_error for `_classify_user_message` (SP2 Task 1.5)."""
    chat_id = state.get("chat_id")
    text = state.get("text") or ""
    if chat_id is None:
        return
    try:
        tg = get_telegram()
    except RuntimeError:
        return
    if tg is None:
        return
    cls = tg._classify_message_by_keywords(text)
    await tg._route_classified_message(chat_id, text, cls)
```

Then update `register_continuations()` to include the new handlers:

```python
def register_continuations() -> None:
    try:
        from general_beckman.continuations import register_resume
        register_resume("telegram.casual_reply_resume",   _casual_reply_resume)
        register_resume("telegram.casual_reply_err",      _casual_reply_err)
        register_resume("telegram.message_route_resume",  _message_route_resume)
        register_resume("telegram.message_route_err",     _message_route_err)
    except Exception as exc:  # noqa: BLE001
        logger.debug("telegram continuation registration deferred",
                     error=str(exc))
```

- [ ] **Step 6: Update `handle_message` caller (line ~7176)**

In `handle_message`, change the classification call site:

```python
        if keyword_result["type"] in _KEYWORD_AUTHORITATIVE_TYPES:
            classification = keyword_result
            # Synchronous keyword path — route inline as before.
            await self._route_classified_message(chat_id, text, classification)
            return
        else:
            # CPS path — fire-and-forget; resume will route.
            await self._classify_user_message(text, chat_id=chat_id)
            return
```

> The block from current line 7178 (`msg_type = classification["type"]`) through line 7254 is removed — it is now entirely inside `_route_classified_message`. The follow-up "PRIORITY 3: Mission vs task" block (line 7256+) is *also* moved into the fall-through branch of `_route_classified_message` — but only the parts that handled `msg_type == "task"` and `msg_type == "followup"`. Verify by re-reading 7256-7270 before editing.

- [ ] **Step 7: Run tests**

Run: `timeout 60 pytest tests/app/test_cps_sp2_telegram.py -v`
Expected: All 7 PASS (5 from Task 1 + 2 from 1.5).

- [ ] **Step 8: Run app-level regression**

Run: `timeout 120 pytest tests/app/test_telegram_classifier_enqueue.py tests/app/test_telegram_casual_chat_enqueue.py -v`
Expected: Legacy tests will need similar updates — assert CPS shape, drop `await_inline=True` assertion. Fix in place.

- [ ] **Step 9: Commit**

```bash
rtk git add src/app/telegram_bot.py tests/app/test_cps_sp2_telegram.py tests/app/test_telegram_classifier_enqueue.py
rtk git commit -m "feat(telegram): migrate _classify_user_message + extract _route_classified_message to CPS (SP2 Task 1.5)"
```

---

## Task 1.6: `cmd_mission` migration

**Files:**
- Modify: `src/app/telegram_bot.py` — `cmd_mission` (line 2161), the only other caller of `_classify_user_message`.
- Modify: `tests/app/test_cps_sp2_telegram.py` — append site-#1c tests.

**Pre-read:**
- `C:\Users\sakir\Dropbox\Workspaces\kutay\src\app\telegram_bot.py:2161-2262` — full `cmd_mission` body.

- [ ] **Step 1: Append failing test**

```python
@pytest.mark.asyncio
async def test_cmd_mission_uses_cps_classification(tmp_path, monkeypatch):
    """cmd_mission --no-workflow path goes through CPS classifier."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        captured = {}

        async def fake_enqueue(spec, **kwargs):
            captured["spec"] = spec
            captured["kwargs"] = kwargs
            return 6262

        monkeypatch.setattr("general_beckman.enqueue", fake_enqueue)

        from src.app.telegram_bot import TelegramInterface
        bot = object.__new__(TelegramInterface)
        bot._pending_clarifications = {}
        bot.user_last_task_id = {}
        bot._reply = AsyncMock()
        bot._classify_user_message = AsyncMock()

        fake_update = MagicMock()
        fake_update.message.chat_id = 7777
        fake_context = MagicMock()
        fake_context.args = ["Build", "a", "login", "page"]

        await bot.cmd_mission(fake_update, fake_context)
        bot._classify_user_message.assert_awaited_once()
        # chat_id keyword must be passed (otherwise resume can't reply)
        kwargs = bot._classify_user_message.call_args.kwargs
        assert kwargs.get("chat_id") == 7777
    finally:
        await _close_db()
```

- [ ] **Step 2: Verify fail**

Run: `timeout 60 pytest tests/app/test_cps_sp2_telegram.py::test_cmd_mission_uses_cps_classification -v`

- [ ] **Step 3: Update `cmd_mission`**

In `src/app/telegram_bot.py`, replace lines 2202-2207 (the `if not workflow: ... await self._classify_user_message(description)`) block with:

```python
        # SP2 CPS: classification is fire-and-forget. If the user did not
        # ask for a workflow, dispatch through the CPS classifier; the
        # resume will create the mission (workflow or plain) based on the
        # classification.
        if not workflow:
            chat_id = update.message.chat_id
            # Cache the description so the resume can build the mission.
            self._pending_mission[chat_id] = description
            await self._classify_user_message(description, chat_id=chat_id)
            await self._reply(
                update,
                f"🎯 Classifying mission… (id pending)",
            )
            return
        # Explicit --workflow → unchanged synchronous path.
```

> The synchronous mission-creation branch (the `if workflow: ...` block at 2210-2229) stays intact for the explicit `--workflow` case. The implicit path is now CPS; the resume's `_route_classified_message` already handles `msg_type == "task"` by creating a plain task, but a mission-flavored description should create a mission. **The resume reads `self._pending_mission[chat_id]` to recover the original description, then creates the mission, then clears the entry.** Add this branch to `_route_classified_message`:

In `_route_classified_message`, BEFORE the fall-through "plain task" block, add:

```python
        # cmd_mission CPS branch — descriptions cached in _pending_mission.
        mission_desc = self._pending_mission.pop(chat_id, None)
        if mission_desc is not None:
            from src.infra.db import add_mission
            wf = classification.get("workflow")
            if wf == "i2p":
                try:
                    from ..workflows.engine.runner import WorkflowRunner
                    runner = WorkflowRunner()
                    mission_id = await runner.start(
                        workflow_name="i2p_v3",
                        initial_input={"raw_idea": mission_desc,
                                       "product_name": mission_desc[:50]},
                        title=mission_desc[:80],
                    )
                    await _send_telegram_via_resume(
                        chat_id,
                        f"🔄 Workflow mission #{mission_id} created!"
                    )
                except Exception as e:
                    logger.error("workflow mission failed", error=str(e))
                    await _send_telegram_via_resume(
                        chat_id, f"❌ {_friendly_error(str(e))}"
                    )
                return
            mission_id = await add_mission(
                title=mission_desc[:80], description=mission_desc, priority=7,
            )
            await _send_telegram_via_resume(
                chat_id, f"🎯 Mission #{mission_id} created."
            )
            return
```

- [ ] **Step 4: Run all Telegram CPS tests**

Run: `timeout 60 pytest tests/app/test_cps_sp2_telegram.py -v`
Expected: 8 PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/app/telegram_bot.py tests/app/test_cps_sp2_telegram.py
rtk git commit -m "feat(telegram): migrate cmd_mission classification path to CPS (SP2 Task 1.6)"
```

---

## Task 2: SKIPPED — Site #2 explicit deferral

Per the SP2 spec §Site 2 special case: `src/core/task_classifier.py::_enqueue_inline_classifier` (line 259) **stays on `await_inline=True`** for SP2. Its sync caller contract (`classify_task` → `TaskClassification` dataclass consumed in-thread by `add_task`) cannot be safely inverted without a separate spec.

- [ ] **Step 1: Confirm deferral**

Run: `rtk grep "await_inline=True" src/core/task_classifier.py`
Expected: one match (line 283). This is the intentional carve-out.

- [ ] **Step 2: Document the carve-out**

Add a `# SP5-DEFERRED:` comment marker above `task_classifier.py:283`:

```python
    # SP5-DEFERRED: this is the one edge-group await_inline site SP2 keeps,
    # because classify_task's caller (add_task) consumes the returned
    # TaskClassification synchronously. CPS-migrating this requires
    # redesigning task admission — see SP2 spec §Site 2 special case.
    tr = await general_beckman.enqueue(spec, parent_id=None, await_inline=True)
```

- [ ] **Step 3: Commit**

```bash
rtk git add src/core/task_classifier.py
rtk git commit -m "docs(task_classifier): mark _enqueue_inline_classifier as SP5-deferred (SP2 Task 2)"
```

---

## Task 3: Migrate `interview.summarize_interview` (site #3)

**Files:**
- Modify: `src/app/interview.py` — split `summarize_interview` into kickoff + resume; add `register_continuations()`.
- Modify: `packages/mr_roboto/src/mr_roboto/__init__.py:4122-4135` — `interview/summarize` action handler now treats the call as fire-and-forget.
- Modify: `packages/general_beckman/src/general_beckman/continuations.py` — append `"src.app.interview"` to `_HANDLER_MODULES`.
- Test: `tests/app/test_cps_sp2_interview.py` (new).

**Pre-read:**
- `C:\Users\sakir\Dropbox\Workspaces\kutay\src\app\interview.py:231-316` — full `summarize_interview`.
- `C:\Users\sakir\Dropbox\Workspaces\kutay\packages\mr_roboto\src\mr_roboto\__init__.py:4122-4135` — the action handler.
- The spec's §Site 3 caller pattern.

> **Re-audit finding to honor:** `interview.py:265` reads `task_result.output`, a field that does NOT exist on `TaskResult`. The current code silently writes empty summaries. The CPS migration must route through `result["result"]["content"]` (the correct dispatcher-response shape).

- [ ] **Step 1: Write failing tests**

Create `tests/app/test_cps_sp2_interview.py`:

```python
"""SP2 Task 3: interview.summarize_interview CPS migration."""
import json
import pytest
from unittest.mock import AsyncMock

import src.infra.db as _db_mod


async def _fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "iv.db"))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
    monkeypatch.setattr(_db_mod, "_db_connection", None)
    await _db_mod.init_db()


async def _close_db():
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None


async def _seed_note(transcript: str = "A real conversation transcript.") -> int:
    db = await _db_mod.get_db()
    await db.execute(
        "INSERT INTO interview_notes (note_id, product_id, transcript_md) "
        "VALUES (?, ?, ?)", (1, "prod1", transcript))
    await db.commit()
    return 1


@pytest.mark.asyncio
async def test_summarize_interview_enqueues_with_on_complete(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        await _seed_note()
        captured = {}

        async def fake_enqueue(spec, **kwargs):
            captured["spec"] = spec
            captured["kwargs"] = kwargs
            return 9001

        monkeypatch.setattr("general_beckman.enqueue", fake_enqueue)

        from src.app.interview import summarize_interview
        res = await summarize_interview(note_id=1, product_id="prod1")
        assert res["ok"] is True
        assert res.get("queued") is True
        assert res["note_id"] == 1
        assert captured["kwargs"].get("await_inline") in (False, None)
        assert captured["kwargs"]["on_complete"] == "interview.summary_persist_resume"
        assert captured["kwargs"]["on_error"] == "interview.summary_persist_err"
        cs = captured["kwargs"]["cont_state"]
        assert cs == {"note_id": 1, "product_id": "prod1"}
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_summary_resume_writes_db_from_result_content(tmp_path, monkeypatch):
    """Resume must extract JSON from result['result']['content'] (NOT .output)."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        await _seed_note()
        from src.app.interview import _summary_persist_resume
        await _summary_persist_resume(
            child_task_id=9001,
            result={"status": "completed", "result": {"content": json.dumps({
                "bullets": ["Point A", "Point B"],
                "quotes": ["Verbatim quote"],
                "insights": "Founder-level take.",
                "action_items": ["Follow up email"],
            })}},
            state={"note_id": 1, "product_id": "prod1"},
        )
        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT summary_md, quotes_json, insights_md, action_items_json "
            "FROM interview_notes WHERE note_id=?", (1,))
        row = await cur.fetchone()
        assert "Point A" in (row[0] or "")
        assert json.loads(row[1] or "[]") == ["Verbatim quote"]
        assert "Founder-level" in (row[2] or "")
        assert json.loads(row[3] or "[]") == ["Follow up email"]
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_summary_resume_tolerates_non_json_content(tmp_path, monkeypatch):
    """Regex-rescue path (current code line 271) must still work."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        await _seed_note()
        from src.app.interview import _summary_persist_resume
        content_with_prose = (
            "Here is the structured response:\n"
            "{\"bullets\": [\"Embedded\"], \"quotes\": [], "
            "\"insights\": \"\", \"action_items\": []}\n"
            "(end of response)"
        )
        await _summary_persist_resume(
            child_task_id=9002,
            result={"status": "completed",
                    "result": {"content": content_with_prose}},
            state={"note_id": 1, "product_id": "prod1"},
        )
        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT summary_md FROM interview_notes WHERE note_id=?", (1,))
        assert "Embedded" in (await cur.fetchone())[0]
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_summary_on_error_leaves_row_intact(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        await _seed_note()
        from src.app.interview import _summary_persist_err
        await _summary_persist_err(
            child_task_id=9003,
            result={"status": "failed", "error": "timeout"},
            state={"note_id": 1, "product_id": "prod1"},
        )
        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT summary_md FROM interview_notes WHERE note_id=?", (1,))
        assert (await cur.fetchone())[0] is None
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_register_continuations_registers_interview_handlers(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from src.app.interview import register_continuations
        from general_beckman.continuations import _HANDLERS
        _HANDLERS.pop("interview.summary_persist_resume", None)
        _HANDLERS.pop("interview.summary_persist_err", None)
        register_continuations()
        assert "interview.summary_persist_resume" in _HANDLERS
        assert "interview.summary_persist_err" in _HANDLERS
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_interview_module_in_handler_modules():
    from general_beckman.continuations import _HANDLER_MODULES
    assert "src.app.interview" in _HANDLER_MODULES
```

- [ ] **Step 2: Verify fail**

Run: `timeout 60 pytest tests/app/test_cps_sp2_interview.py -v`
Expected: 6 FAIL.

- [ ] **Step 3: Migrate `summarize_interview`**

Replace `summarize_interview` body in `src/app/interview.py` (231-316) with:

```python
async def summarize_interview(note_id: int, product_id: str) -> dict:
    """LLM-bound: read transcript, enqueue summary LLM via CPS, return.

    SP2: the actual persistence happens in `_summary_persist_resume` when
    the child completes. Returns ``{"ok": True, "queued": True, ...}``
    immediately so the mr_roboto action handler proceeds without blocking.
    """
    from src.infra.db import get_db

    db = await get_db()
    cur = await db.execute(
        "SELECT transcript_md FROM interview_notes "
        "WHERE note_id=? AND product_id=?",
        (note_id, product_id),
    )
    row = await cur.fetchone()
    if row is None:
        return {"ok": False, "error": f"note not found: note_id={note_id}"}
    transcript = row[0] or ""
    if not transcript.strip():
        return {"ok": False, "error": "transcript_md is empty; run transcribe first"}

    prompt = _SUMMARIZE_PROMPT.format(transcript=transcript)

    from general_beckman import enqueue as beckman_enqueue
    child_id = await beckman_enqueue(
        {
            "title": f"Interview summary note_id={note_id}",
            "agent_type": "summarizer",
            "kind": "overhead",
            "context": json.dumps({"prompt": prompt, "note_id": note_id}),
        },
        on_complete="interview.summary_persist_resume",
        on_error="interview.summary_persist_err",
        cont_state={"note_id": note_id, "product_id": product_id},
        lane="overhead",
    )
    logger.info("interview: summarize enqueued",
                note_id=note_id, child_task_id=child_id)
    return {"ok": True, "queued": True, "note_id": note_id,
            "child_task_id": child_id}
```

Then add the resume + on_error + registration to the same module (top-level, after `summarize_interview`):

```python
async def _summary_persist_resume(
    child_task_id: int, result: dict, state: dict
) -> None:
    """CPS resume for `summarize_interview` (SP2 Task 3).

    Parses structured JSON from the LLM response and writes the four
    interview_notes columns. This is where the latent
    `task_result.output` bug (interview.py:265) is fixed — we now route
    through the documented `result["result"]["content"]` shape.
    """
    note_id = state.get("note_id")
    if note_id is None:
        return

    agent_output = (result or {}).get("result") or {}
    if isinstance(agent_output, dict):
        content = agent_output.get("content", "")
    else:
        content = str(agent_output)
    if isinstance(content, list):
        content = "\n".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in content
        )
    raw = str(content or "").strip()

    structured: dict[str, Any] = {}
    try:
        structured = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        import re
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                structured = json.loads(match.group())
            except json.JSONDecodeError:
                structured = {}

    bullets = structured.get("bullets") or []
    quotes = structured.get("quotes") or []
    insights = structured.get("insights") or ""
    action_items = structured.get("action_items") or []

    summary_md = "## Interview Summary\n\n"
    if bullets:
        summary_md += "\n".join(f"- {b}" for b in bullets)

    from src.infra.db import get_db
    db = await get_db()
    await db.execute(
        "UPDATE interview_notes "
        "SET summary_md=?, quotes_json=?, insights_md=?, action_items_json=? "
        "WHERE note_id=?",
        (summary_md, json.dumps(quotes), insights,
         json.dumps(action_items), note_id),
    )
    await db.commit()
    logger.info("interview: summary persisted via CPS resume",
                note_id=note_id, bullets=len(bullets), quotes=len(quotes))


async def _summary_persist_err(
    child_task_id: int, result: dict, state: dict
) -> None:
    """CPS on_error for `summarize_interview` (SP2 Task 3).

    Matches the pre-CPS behavior: leave `summary_md` NULL on LLM failure
    (the pre-CPS code wrote an empty summary; this is a strict
    improvement — explicit NULL signals 'no summary' rather than 'summary
    was empty').
    """
    logger.warning(
        "interview: summary LLM failed; leaving row untouched",
        note_id=state.get("note_id"),
        error=(result or {}).get("error"),
    )


def register_continuations() -> None:
    """Register interview-pipeline CPS handlers (SP2). Idempotent."""
    try:
        from general_beckman.continuations import register_resume
        register_resume("interview.summary_persist_resume", _summary_persist_resume)
        register_resume("interview.summary_persist_err",    _summary_persist_err)
    except Exception as exc:  # noqa: BLE001
        logger.debug("interview continuation registration deferred",
                     error=str(exc))


register_continuations()
```

- [ ] **Step 4: Update the mr_roboto action handler**

In `packages/mr_roboto/src/mr_roboto/__init__.py:4122-4135`, the `interview/summarize` block now receives a `{"ok": True, "queued": True}` envelope. Existing code already returns `Action(status="completed", result=res)` on that — no change needed beyond confirming the path still returns "completed". Verify by re-reading lines 4122-4135 and adjusting if the `if not res.get("ok"):` branch needs to *not* trip on queued results.

```python
            res = await summarize_interview(note_id=note_id, product_id=product_id)
            if not res.get("ok"):
                return Action(status="failed",
                              error=res.get("error", "summarize failed"),
                              result=res)
            # SP2: res may carry {"queued": True}; persistence happens in
            # the CPS resume. The mechanical action is still 'completed'.
            return Action(status="completed", result=res)
```

- [ ] **Step 5: Extend `_HANDLER_MODULES`**

In `packages/general_beckman/src/general_beckman/continuations.py:119`:

```python
_HANDLER_MODULES = (
    "mr_roboto.executors.analytics_digest",
    "mr_roboto.executors.classify_signals",
    "src.app.telegram_bot",
    "src.app.interview",
)
```

- [ ] **Step 6: Run tests**

Run: `timeout 60 pytest tests/app/test_cps_sp2_interview.py -v`
Expected: 6 PASS.

- [ ] **Step 7: Regression for legacy z7 test**

Run: `timeout 60 pytest tests/z7/test_b7_interview_pipeline.py -v`
Expected: may have one test that asserts `summary_md` is non-empty after `summarize_interview` returns. **Update that test** to drive the resume manually (call `_summary_persist_resume` with a fake result dict) — the synchronous summarize-and-assert pattern is gone. Expected after fix: PASS.

- [ ] **Step 8: Commit**

```bash
rtk git add src/app/interview.py packages/mr_roboto/src/mr_roboto/__init__.py packages/general_beckman/src/general_beckman/continuations.py tests/app/test_cps_sp2_interview.py tests/z7/test_b7_interview_pipeline.py
rtk git commit -m "feat(interview): migrate summarize_interview to CPS (SP2 Task 3)"
```

---

## Task 4: Migrate `meetings._call_llm_meeting_brief` (site #4)

**Files:**
- Modify: `src/app/meetings.py` — rename `_call_llm_meeting_brief` to `enqueue_meeting_brief` (returns child id); add resume + on_error; add `register_continuations()`.
- Modify: `packages/mr_roboto/src/mr_roboto/__init__.py:3955-…` — `meetings/brief` action handler enqueues and returns.
- Modify: `packages/general_beckman/src/general_beckman/continuations.py` — append `"src.app.meetings"`.
- Test: `tests/app/test_cps_sp2_meetings.py` (new).

**Pre-read:**
- `C:\Users\sakir\Dropbox\Workspaces\kutay\src\app\meetings.py:338-471` — full `_call_llm_meeting_brief` + `_parse_brief_llm_response`.
- `C:\Users\sakir\Dropbox\Workspaces\kutay\packages\mr_roboto\src\mr_roboto\__init__.py:3955-4030` — the brief action handler.

- [ ] **Step 1: Failing test**

Create `tests/app/test_cps_sp2_meetings.py`:

```python
"""SP2 Task 4: meetings._call_llm_meeting_brief CPS migration."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

import src.infra.db as _db_mod


async def _fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "m.db"))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
    monkeypatch.setattr(_db_mod, "_db_connection", None)
    await _db_mod.init_db()


async def _close_db():
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None


async def _seed_meeting(meeting_id=1, product_id="prod1") -> None:
    db = await _db_mod.get_db()
    await db.execute(
        "INSERT INTO meetings (meeting_id, product_id, contact_id, "
        "scheduled_for, purpose) VALUES (?, ?, ?, "
        "strftime('%Y-%m-%d %H:%M:%S','now'), 'test')",
        (meeting_id, product_id, 7),
    )
    await db.commit()


@pytest.mark.asyncio
async def test_enqueue_meeting_brief_uses_cps(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        await _seed_meeting()
        captured = {}

        async def fake_enqueue(spec, **kwargs):
            captured["spec"] = spec
            captured["kwargs"] = kwargs
            return 7777

        monkeypatch.setattr("general_beckman.enqueue", fake_enqueue)

        from src.app.meetings import enqueue_meeting_brief
        ctx = {"contact": {"name": "X"}, "meeting": {"meeting_id": 1}}
        cid = await enqueue_meeting_brief(
            ctx, meeting_id=1, product_id="prod1"
        )
        assert cid == 7777
        assert captured["kwargs"].get("await_inline") in (False, None)
        assert captured["kwargs"]["on_complete"] == "meetings.brief_persist_resume"
        assert captured["kwargs"]["on_error"] == "meetings.brief_persist_err"
        cs = captured["kwargs"]["cont_state"]
        assert cs["meeting_id"] == 1
        assert cs["product_id"] == "prod1"
        assert "ctx" in cs
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_brief_resume_writes_brief_md_and_notifies(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        await _seed_meeting()
        # Stub Telegram singleton.
        fake_tg = MagicMock()
        fake_tg.app.bot.send_message = AsyncMock()
        monkeypatch.setattr("src.app.telegram_bot.get_telegram",
                            lambda: fake_tg)
        monkeypatch.setenv("TELEGRAM_ADMIN_CHAT_ID", "777")

        from src.app.meetings import _brief_persist_resume
        ctx = {"contact": {"name": "Alice"}, "meeting": {"meeting_id": 1}}
        await _brief_persist_resume(
            child_task_id=7777,
            result={"status": "completed",
                    "result": {"content": json.dumps({
                        "talking_points": ["TP1", "TP2"],
                        "suggested_asks": ["Ask1"],
                    })}},
            state={"meeting_id": 1, "product_id": "prod1", "ctx": ctx},
        )
        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT brief_md FROM meetings WHERE meeting_id=?", (1,))
        brief_md = (await cur.fetchone())[0]
        assert "TP1" in (brief_md or "")
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_brief_on_error_writes_unavailable_brief(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        await _seed_meeting()
        from src.app.meetings import _brief_persist_err
        ctx = {"contact": {"name": "X"}, "meeting": {"meeting_id": 1}}
        await _brief_persist_err(
            child_task_id=7777,
            result={"status": "failed", "error": "timeout"},
            state={"meeting_id": 1, "product_id": "prod1", "ctx": ctx},
        )
        db = await _db_mod.get_db()
        cur = await db.execute(
            "SELECT brief_md FROM meetings WHERE meeting_id=?", (1,))
        brief_md = (await cur.fetchone())[0]
        assert "unavailable" in (brief_md or "").lower()
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_meetings_module_in_handler_modules():
    from general_beckman.continuations import _HANDLER_MODULES
    assert "src.app.meetings" in _HANDLER_MODULES
```

- [ ] **Step 2: Verify fail**

Run: `timeout 60 pytest tests/app/test_cps_sp2_meetings.py -v`

- [ ] **Step 3: Implement**

Replace `_call_llm_meeting_brief` (meetings.py:338-428) with two pieces:

```python
async def enqueue_meeting_brief(
    ctx: dict, *, meeting_id: int, product_id: str
) -> int:
    """Enqueue the meeting-brief LLM call via CPS.

    Returns the child task id immediately. The brief is composed and
    persisted by `_brief_persist_resume` when the child completes.
    """
    import time
    import uuid

    from general_beckman import enqueue
    from general_beckman.lanes import LANE_ONESHOT

    ctx_summary = _summarise_ctx_for_llm(ctx)
    prompt = (
        "You are a chief-of-staff preparing a founder for an upcoming meeting.\n"
        # ... unchanged from existing code 357-371 ...
    )
    messages = [{"role": "user", "content": prompt}]
    _suffix = f"{time.monotonic_ns() % 1_000_000:06d}-{uuid.uuid4().hex[:6]}"

    return await enqueue(
        {
            "title": f"meeting_brief:llm:{_suffix}",
            "description": "Draft meeting talking points and suggested asks.",
            "agent_type": "reviewer",
            "kind": "overhead",
            "priority": 2,
            "context": {"llm_call": {
                "raw_dispatch": True, "call_category": "overhead",
                "task": "reviewer", "agent_type": "reviewer",
                "difficulty": 3, "messages": messages, "failures": [],
                "estimated_input_tokens": 500,
                "estimated_output_tokens": 300,
            }},
        },
        lane=LANE_ONESHOT,
        on_complete="meetings.brief_persist_resume",
        on_error="meetings.brief_persist_err",
        # Persist the FULL ctx in cont_state — compose_brief_md needs it.
        cont_state={"meeting_id": meeting_id, "product_id": product_id,
                    "ctx": ctx},
    )
```

Add `_brief_persist_resume`, `_brief_persist_err`, and `register_continuations()` (analogous to interview.py — extract content, call `_parse_brief_llm_response`, call `compose_brief_md`, `UPDATE meetings SET brief_md=?, brief_generated_at=strftime('%Y-%m-%d %H:%M:%S','now') WHERE meeting_id=?`, then Telegram notify if `TELEGRAM_ADMIN_CHAT_ID` env var set). The `on_error` does the same but calls `compose_brief_md(ctx, llm_unavailable=True)`.

- [ ] **Step 4: Update mr_roboto handler**

In `packages/mr_roboto/src/mr_roboto/__init__.py:3955-4030`, replace the action body to call `enqueue_meeting_brief` and return `Action(status="completed", result={"queued": True, "meeting_id": meeting_id, "child_task_id": child_id})`. The DB write + Telegram notify currently happen inline in this action — both move to the resume. Delete the inline `UPDATE meetings SET brief_md...` and `tg.app.bot.send_message(...)` blocks from the action handler.

- [ ] **Step 5: Extend `_HANDLER_MODULES`**

```python
_HANDLER_MODULES = (
    ..., "src.app.meetings",
)
```

- [ ] **Step 6: Tests pass**

Run: `timeout 60 pytest tests/app/test_cps_sp2_meetings.py tests/z7/test_b4_meeting_brief.py -v`
Expected: PASS (with the legacy z7 test possibly needing an update similar to Task 3 step 7).

- [ ] **Step 7: Commit**

```bash
rtk git add src/app/meetings.py packages/mr_roboto/src/mr_roboto/__init__.py packages/general_beckman/src/general_beckman/continuations.py tests/app/test_cps_sp2_meetings.py tests/z7/test_b4_meeting_brief.py
rtk git commit -m "feat(meetings): migrate _call_llm_meeting_brief to CPS (SP2 Task 4)"
```

---

## Task 5: Migrate `faq_regen._llm_cluster_draft` (site #5) + restart-reconcile E2E

**Files:**
- Modify: `src/app/jobs/faq_regen.py`.
- Modify: `packages/general_beckman/src/general_beckman/continuations.py` — append `"src.app.jobs.faq_regen"`.
- Test: `tests/app/test_cps_sp2_faq_regen.py` (new) — **includes the restart-reconcile end-to-end test** from spec §Testing.

**Pre-read:**
- `C:\Users\sakir\Dropbox\Workspaces\kutay\src\app\jobs\faq_regen.py:108-201` and 326-369 (`run_faq_regen`).

- [ ] **Step 1: Failing test (including restart-reconcile)**

Create `tests/app/test_cps_sp2_faq_regen.py`. In addition to the happy/error path tests (mirroring Tasks 3-4), add the restart-reconcile test:

```python
@pytest.mark.asyncio
async def test_faq_regen_restart_reconcile_fires_resume(tmp_path, monkeypatch):
    """End-to-end CPS-on-restart test: child terminal while orchestrator down → reconcile fires."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        # Enqueue the draft via the migrated callsite.
        from src.app.jobs.faq_regen import enqueue_cluster_draft
        cluster = [{"question": "Q1", "answer": "A1"}] * 5
        child_id = await enqueue_cluster_draft(cluster, lang="en")
        assert isinstance(child_id, int)

        db = await _db_mod.get_db()
        # Confirm continuation row created.
        cur = await db.execute(
            "SELECT status FROM continuations WHERE child_task_id=?", (child_id,))
        assert (await cur.fetchone())[0] == "pending"

        # Set child terminal.
        await db.execute(
            "UPDATE tasks SET status='completed', result=? WHERE id=?",
            (json.dumps({"content": json.dumps({
                "question": "Reconciled Q?",
                "answer": "Reconciled A.",
            })}), child_id),
        )
        await db.commit()

        # Simulate restart — clear in-memory registry.
        from general_beckman.continuations import (
            _HANDLERS, register_startup_handlers, reconcile_continuations,
        )
        _HANDLERS.clear()

        emissions = []

        async def fake_emit(*, mission_id, entry, cluster_size):
            emissions.append((entry, cluster_size))

        monkeypatch.setattr("src.app.jobs.faq_regen._emit_faq_founder_action",
                            fake_emit)

        # Re-import to re-register handlers (this is what
        # register_startup_handlers does in production).
        register_startup_handlers()
        await reconcile_continuations()
        import asyncio
        await asyncio.sleep(0.1)

        assert emissions, "reconcile must have fired the resume"
        entry, size = emissions[0]
        assert entry["question"] == "Reconciled Q?"
        assert entry["lang"] == "en"
        assert size == 5
    finally:
        await _close_db()
```

(Plus the same enqueue / resume / on_error / module-in-list tests as Tasks 3/4.)

- [ ] **Step 2: Verify fail; implement; re-run.**

- [ ] **Step 3: Commit**

```bash
rtk git add src/app/jobs/faq_regen.py packages/general_beckman/src/general_beckman/continuations.py tests/app/test_cps_sp2_faq_regen.py tests/z7/test_a8_faq_flywheel.py
rtk git commit -m "feat(faq_regen): migrate _llm_cluster_draft to CPS + restart-reconcile E2E (SP2 Task 5)"
```

---

## Task 6: SKIPPED — Site #6 explicit deferral

Per the SP2 spec §Site 6 deferral (founder decision 2026-05-27): `src/app/jobs/investor_bullets.py::_call_llm_anomaly_hypothesis` (line 204) **stays on `await_inline=True`** for SP2. The anomaly-hypothesis path is unreachable in production today (per the 2026-05-17 Z7 unwired-features handoff — `missions.product_id` is NULL, fetchers return `{}`, hypothesis never fires), so the architectural cost of CPS migration is not justified.

- [ ] **Step 1: Confirm deferral**

Run: `rtk grep "await_inline=True" src/app/jobs/investor_bullets.py`
Expected: one match (line 204). This is the intentional carve-out.

- [ ] **Step 2: Document the carve-out**

Add a `# SP5-DEFERRED:` comment marker above `investor_bullets.py:204`:

```python
            # SP5-DEFERRED: investor_bullets' anomaly-hypothesis path is
            # unreachable in production today (missions.product_id is NULL
            # per the 2026-05-17 Z7 handoff, so fetchers return {} and this
            # code path never fires). CPS-migrating it costs either ~80 LOC +
            # a pending-table schema or a kickoff/finalize split — neither
            # justified while the upstream producer is missing. See SP2 spec
            # §Site 6 deferral.
            await_inline=True,
```

- [ ] **Step 3: Commit**

```bash
rtk git add src/app/jobs/investor_bullets.py
rtk git commit -m "docs(investor_bullets): mark _call_llm_anomaly_hypothesis as SP5-deferred (SP2 Task 6)"
```

---

## Task 7: Restart-reconcile cross-check (lightweight)

**Files:** none (verification only).

The full restart-reconcile end-to-end is in Task 5 (faq_regen). This task simply asserts that **every migrated module's `register_continuations()` is invoked by `register_startup_handlers()`**.

- [ ] **Step 1: Verify `_HANDLER_MODULES` is complete**

Run: `rtk grep -n "^_HANDLER_MODULES" packages/general_beckman/src/general_beckman/continuations.py`
Inspect output — must contain exactly:
```
"mr_roboto.executors.analytics_digest",
"mr_roboto.executors.classify_signals",
"src.app.telegram_bot",
"src.app.interview",
"src.app.meetings",
"src.app.jobs.faq_regen",
```
(`src.app.jobs.investor_bullets` is NOT in the list — site #6 deferred to SP5+.)

- [ ] **Step 2: Confirm each module exposes `register_continuations`**

Run for each module: `rtk grep "^def register_continuations" src/app/telegram_bot.py src/app/interview.py src/app/meetings.py src/app/jobs/faq_regen.py`
Expected: one match per file.

- [ ] **Step 3: Integration test — manual orchestrator boot**

Run: `timeout 30 python -c "from general_beckman.continuations import register_startup_handlers, _HANDLERS; register_startup_handlers(); print(sorted(_HANDLERS.keys()))"`
Expected output (order may vary):
```
['faq_regen.draft_persist_err', 'faq_regen.draft_persist_resume', 'growth.classify_signals_complete', 'growth.store_weekly_digest', 'interview.summary_persist_err', 'interview.summary_persist_resume', 'meetings.brief_persist_err', 'meetings.brief_persist_resume', 'telegram.casual_reply_err', 'telegram.casual_reply_resume', 'telegram.message_route_err', 'telegram.message_route_resume']
```

- [ ] **Step 4: Commit (if any wiring fix needed)**

```bash
rtk git add -A
rtk git commit -m "test(beckman): SP2 restart-reconcile module wiring complete (SP2 Task 7)"
```

---

## Task 8: Full-suite regression + acceptance gate

**Files:** none (verification only).

- [ ] **Step 1: Run the migrated-suite + parent suites**

Run: `timeout 180 pytest tests/app/ tests/core/ tests/beckman/ tests/z7/ -v`
Expected: PASS. Any test that breaks because it asserted on the old `await_inline=True` shape must be updated *in this task*, not deferred — the assertion contract for the 5 migrated sites is no longer dual-mode.

- [ ] **Step 2: Acceptance grep — 4 sites no longer use await_inline**

Run: `rtk grep "await_inline=True" src/app/telegram_bot.py src/app/interview.py src/app/meetings.py src/app/jobs/faq_regen.py`
Expected: **zero matches**.

- [ ] **Step 3: Acceptance grep — Site 2 + Site 6 explicit carve-outs preserved**

Run: `rtk grep "await_inline=True" src/core/task_classifier.py src/app/jobs/investor_bullets.py`
Expected: **exactly two matches** — `task_classifier.py:~283` (Site #2 deferred) and `investor_bullets.py:~204` (Site #6 deferred). Both must have a `# SP5-DEFERRED:` comment directly above.

- [ ] **Step 4: Acceptance grep — primitive body intact**

Run: `rtk grep "_inline_waiters" packages/general_beckman/src/general_beckman/__init__.py`
Expected: matches (the body is intact; SP5 deletes it).

Run: `rtk grep "resolve_inline\|INLINE_TIMEOUT" packages/general_beckman/src/general_beckman/__init__.py`
Expected: matches. The await_inline primitive is untouched.

- [ ] **Step 5: Acceptance grep — substrate unchanged**

Run: `rtk git diff main -- packages/general_beckman/src/general_beckman/continuations.py`
Expected: the ONLY change is the `_HANDLER_MODULES` tuple (5 new entries). No logic changes.

Run: `rtk git diff main -- packages/general_beckman/src/general_beckman/__init__.py`
Expected: **zero changes**. SP2 must not touch the substrate.

Run: `rtk git diff main -- src/infra/db.py`
Expected: **zero changes**.

- [ ] **Step 6: Manual smoke (founder-driven, optional)**

If a staging Telegram environment is available:
1. Send a casual message (`"hi how are you"`) to the bot. Within ~30s, receive a reply.
2. Send a task description (`"build a login page"`). Within ~60s, receive a "Task #N queued" confirmation.
3. Restart the orchestrator while a casual message is in-flight. Confirm the reply arrives after restart (proving reconcile + handler registration).

- [ ] **Step 7: Acceptance checklist (cite the test that proves each)**

- All 4 sites no longer pass `await_inline=True` — Step 2 grep.
- Site #2 + Site #6 carve-outs preserved — Step 3 grep.
- `await_inline` body itself untouched — Step 4 + 5 grep.
- SP1 substrate tests still green — `timeout 120 pytest tests/beckman/test_continuations_durable.py tests/beckman/test_continuations.py -v`.
- New SP2 tests green — Steps 1.
- One site's continuation survives restart and fires — Task 5's `test_faq_regen_restart_reconcile_fires_resume`.
- No regressions in `tests/app/`, `tests/core/` — Step 1.
- Interview `task_result.output` latent bug fixed — `test_summary_resume_writes_db_from_result_content` (Task 3 step 1) asserts on parsed content reaching the DB.

- [ ] **Step 8: Commit (if any fixups needed)**

```bash
rtk git add -A
rtk git commit -m "test(beckman): SP2 full-suite regression green (SP2 Task 8)"
```

---

## Self-Review (completed by plan author)

**Spec coverage matrix.** Every section of `2026-05-27-cps-sp2-edge-group-design.md` is mapped to plan tasks:
- §Scope guard (6 sites enumerated) — Task 1, 1.5, 1.6 (#1a/1b/1c), Task 2 (#2 deferral), Tasks 3-6 (#3-#6).
- §The keystone pattern — Task 1 establishes; Tasks 3-6 follow it mechanically.
- §Result-field shape (esp. latent `interview.py:265` bug) — Task 3 step 3 fixes it; Task 3 step 1's tests assert on `result["result"]["content"]` shape.
- §Per-site design table — one task per row (#2 deferral acknowledged).
- §Cross-cutting decisions: `register_continuations()` per module — every task adds it; `_HANDLER_MODULES` extended in every task except Task 2; Telegram-singleton lazy lookup — Task 1 step 3; `_pending_action` idempotence — Task 1.5 step 3; helper de-dup `_send_telegram_via_resume` — Task 1 step 3; no mode flags — explicit in Task 1 step 9 (legacy tests updated, not coexisted); fallback lift proof — each task’s on_error mirrors current fallback.
- §Testing strategy — happy + failure + restart-reconcile (Task 5).
- §Acceptance — Task 8 enumerates each criterion + its proving test/grep.
- §Out-of-scope — Task 2 reiterates Site 2 deferral; no SP3/SP4/SP5 sites touched.

**Placeholder scan.** Every code block in the plan is paste-able. The two exceptions are intentional:
- The full `_parse_brief_llm_response` body in Task 4 step 3 is omitted (it already exists in `meetings.py:431-471` and is unchanged) — the plan references it by line range rather than inlining.
- The full `compose_brief_md` extension for the Telegram-notify section of Task 4 is referenced via the existing code at `mr_roboto/__init__.py:3997-4030` (which moves into the resume verbatim, with `self._reply` → `_send_telegram_via_resume` swap).

**Type consistency.** All resume handlers conform to `async (task_id: int, result: dict, state: dict) -> None`. All `cont_state` payloads are JSON-serializable dicts of plain types (int, str, list, dict). `enqueue` return type (the child task id, `int`) is consistently the return value of every kickoff function. `_send_telegram_via_resume` returns `bool` and is always awaited.

**SP1 discoveries honored.** The spike's finding that `result` is reconstructed from `tasks.result` JSON on reconcile is honored: Task 5's restart-reconcile test stores the LLM response as `json.dumps({"content": json.dumps({...})})` into `tasks.result` to mirror the production shape, then asserts the resume parses through both JSON layers correctly. The 3-arg handler signature `(task_id, result, state)` matches SP1's `dispatch_on_complete` contract. The `needs_clarification` non-terminal status mapping is not exercised by SP2 (none of the edge-group LLM calls produce that status), so no SP2 test asserts on it — but if a future LLM-call lane emits it, the substrate's existing `fire_for_task` correctly leaves the row pending.

**Open questions surfaced for founder.** Two are noted at the top of the plan (Site #2 deletion vs deferral; 6A vs 6B for investor_bullets). The plan defaults to the conservative choice (defer #2, pick 6B) so execution can proceed without blocking.