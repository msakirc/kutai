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
        from general_beckman.continuations import _HANDLERS
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


# ─── Task 1.5: _classify_user_message + _route_classified_message ──────────


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
        rv = await bot._classify_user_message(
            "How is the coffee mission going?", chat_id=999
        )
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
        from src.app.telegram_bot import (
            _message_route_resume,
            TelegramInterface,
            set_telegram,
        )
        bot = object.__new__(TelegramInterface)
        bot.app = MagicMock()
        bot.app.bot.send_message = AsyncMock()
        bot.user_last_task_id = {}
        bot._pending_clarifications = {}
        bot._pending_mission = {}
        bot._route_classified_message = AsyncMock()
        set_telegram(bot)

        await _message_route_resume(
            child_task_id=5151,
            result={"status": "completed",
                    "result": {"content": '{"type": "casual", "confidence": 0.9}'}},
            state={"chat_id": 999, "text": "hi", "flow": "message_route"},
        )

        bot._route_classified_message.assert_awaited_once()
        args = bot._route_classified_message.call_args.args
        # signature: _route_classified_message(chat_id, text, classification)
        assert args[0] == 999
        assert args[1] == "hi"
        assert args[2]["type"] == "casual"
    finally:
        from src.app.telegram_bot import set_telegram
        set_telegram(None)  # type: ignore[arg-type]
        await _close_db()


# ─── Task 1.6: cmd_mission migration ───────────────────────────────────────


@pytest.mark.asyncio
async def test_cmd_mission_uses_cps_classification(tmp_path, monkeypatch):
    """cmd_mission's no-workflow path goes through the CPS classifier."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from src.app.telegram_bot import TelegramInterface
        bot = object.__new__(TelegramInterface)
        bot._pending_clarifications = {}
        bot._pending_mission = {}
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
        # The mission description must be cached in _pending_mission for the resume.
        assert bot._pending_mission.get(7777) == "Build a login page"
    finally:
        await _close_db()


# ─── SP2 fix-ups: bug 1 (followup linkage), bug 2 (Z0 ceiling), bug 3 (status query) ──


@pytest.mark.asyncio
async def test_followup_route_sets_parent_task_id(tmp_path, monkeypatch):
    """`_route_classified_message` for ``followup`` must thread the parent
    task id into the new task — pre-SP2 ``handle_message`` did this and the
    CPS path lost it when collapsing into the fall-through plain-task
    branch."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from src.app.telegram_bot import TelegramInterface, set_telegram

        bot = object.__new__(TelegramInterface)
        bot.app = MagicMock()
        bot.app.bot.send_message = AsyncMock()
        bot.user_last_task_id = {}
        bot._pending_clarifications = {}
        bot._pending_mission = {}
        bot._pending_action = {}
        set_telegram(bot)

        # Seed a "previous" task by the same chat so user_last_task_id maps
        # cleanly. We do NOT need find_followup_context to return is_followup —
        # the fallback in _route_classified_message uses user_last_task_id.
        from src.infra.db import add_task

        parent_id = await add_task(
            title="Previous shopping search",
            description="Looking for a coffee machine",
            tier="auto",
            priority=8,
            agent_type="shopping_advisor",
            context={"chat_id": 8888},
        )
        bot.user_last_task_id[8888] = parent_id

        # Stub find_followup_context so it doesn't actually hit chroma.
        async def fake_followup_context(chat_id: int, text: str):
            return {"is_followup": True, "parent_task_id": parent_id,
                    "context": []}
        monkeypatch.setattr(
            "src.memory.conversations.find_followup_context",
            fake_followup_context,
        )
        # The function is also imported at module-level in telegram_bot —
        # patch the local reference too so the in-module call hits the stub.
        monkeypatch.setattr(
            "src.app.telegram_bot.find_followup_context",
            fake_followup_context,
        )

        await bot._route_classified_message(
            chat_id=8888,
            text="any update on that?",
            classification={"type": "followup"},
        )

        # The new task must be wired to the parent.
        from src.infra.db import get_db
        db = await get_db()
        cur = await db.execute(
            "SELECT id, parent_task_id, title FROM tasks "
            "WHERE id != ? ORDER BY id DESC LIMIT 1",
            (parent_id,),
        )
        row = await cur.fetchone()
        assert row is not None, "no follow-up task was created"
        assert row[1] == parent_id, (
            f"parent_task_id={row[1]} but expected {parent_id} (title={row[2]!r})"
        )
        assert bot.user_last_task_id[8888] == row[0]
    finally:
        from src.app.telegram_bot import set_telegram
        set_telegram(None)  # type: ignore[arg-type]
        await _close_db()


@pytest.mark.asyncio
async def test_clarification_response_route_sets_parent_task_id(tmp_path, monkeypatch):
    """`_route_classified_message` for ``clarification_response`` must thread
    the parent task id via ``_find_followup_parent``."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from src.app.telegram_bot import TelegramInterface, set_telegram

        bot = object.__new__(TelegramInterface)
        bot.app = MagicMock()
        bot.app.bot.send_message = AsyncMock()
        bot.user_last_task_id = {}
        bot._pending_clarifications = {}
        bot._pending_mission = {}
        bot._pending_action = {}
        set_telegram(bot)

        from src.infra.db import add_task

        parent_id = await add_task(
            title="Clarification needed",
            description="Which colour?",
            tier="auto",
            priority=8,
            agent_type="shopping_advisor",
            context={"chat_id": 9999},
        )
        bot.user_last_task_id[9999] = parent_id

        # _find_followup_parent calls find_followup_context first; on miss
        # falls back to user_last_task_id. Force it through the fallback by
        # making find_followup_context return is_followup=False so the
        # fallback is exercised.
        async def fake_followup_context(chat_id: int, text: str):
            return {"is_followup": False, "context": []}
        monkeypatch.setattr(
            "src.memory.conversations.find_followup_context",
            fake_followup_context,
        )
        monkeypatch.setattr(
            "src.app.telegram_bot.find_followup_context",
            fake_followup_context,
        )

        await bot._route_classified_message(
            chat_id=9999,
            text="The blue one please",
            classification={"type": "clarification_response"},
        )

        from src.infra.db import get_db
        db = await get_db()
        cur = await db.execute(
            "SELECT id, parent_task_id, title FROM tasks "
            "WHERE id != ? ORDER BY id DESC LIMIT 1",
            (parent_id,),
        )
        row = await cur.fetchone()
        assert row is not None, "no clarification-response task was created"
        assert row[1] == parent_id, (
            f"parent_task_id={row[1]} but expected {parent_id}"
        )
        assert bot.user_last_task_id[9999] == row[0]
    finally:
        from src.app.telegram_bot import set_telegram
        set_telegram(None)  # type: ignore[arg-type]
        await _close_db()


@pytest.mark.asyncio
async def test_mission_plain_sets_z0_ceiling_pending_action(tmp_path, monkeypatch):
    """The ``_pending_mission`` resume branch must, on a plain (no-workflow)
    mission, set ``_pending_action[chat_id] = {'kind': 'z0_ceiling', ...}``
    and send the cost-ceiling prompt — pre-SP2 ``cmd_mission`` did this and
    the CPS resume previously just sent "🎯 Mission #X created."."""
    await _fresh_db(tmp_path, monkeypatch)
    sent = []

    async def fake_send(chat_id, text, *, parse_mode=None):
        sent.append((chat_id, text, parse_mode))
        return True

    monkeypatch.setattr(
        "src.app.telegram_bot._send_telegram_via_resume", fake_send,
    )
    try:
        from src.app.telegram_bot import TelegramInterface, set_telegram

        bot = object.__new__(TelegramInterface)
        bot.app = MagicMock()
        bot.app.bot.send_message = AsyncMock()
        bot.user_last_task_id = {}
        bot._pending_clarifications = {}
        bot._pending_mission = {}
        bot._pending_action = {}
        bot.orchestrator = None  # explicitly skip plan_mission
        set_telegram(bot)

        # Stage the mission description as cmd_mission would have done.
        bot._pending_mission[5555] = "Build a coffee app"

        # Stub ensure_mission_topic so it doesn't try to talk to a real bot.
        async def fake_ensure(*args, **kwargs):
            return None
        monkeypatch.setattr(
            "src.app.telegram_topics.ensure_mission_topic", fake_ensure,
        )

        # No "workflow" in classification → plain mission path.
        await bot._route_classified_message(
            chat_id=5555,
            text="ignored — _pending_mission overrides",
            classification={"type": "mission"},
        )

        assert 5555 in bot._pending_action, (
            f"_pending_action was not armed; got {bot._pending_action!r}"
        )
        pa = bot._pending_action[5555]
        assert pa["kind"] == "z0_ceiling", (
            f"pending action kind={pa.get('kind')!r}, expected 'z0_ceiling'"
        )
        assert isinstance(pa.get("mission_id"), int) and pa["mission_id"] > 0

        # Cost-ceiling prompt must have been sent via _send_telegram_via_resume.
        ceiling_msgs = [m for m in sent if "cost ceiling" in m[1].lower()]
        assert ceiling_msgs, (
            f"cost-ceiling prompt missing; sent messages = {sent!r}"
        )
        chat_id, text, _ = ceiling_msgs[0]
        assert chat_id == 5555
        # user_last_task_id should have been cleared per pre-SP2 cmd_mission.
        assert bot.user_last_task_id.get(5555) is None
    finally:
        from src.app.telegram_bot import set_telegram
        set_telegram(None)  # type: ignore[arg-type]
        await _close_db()


@pytest.mark.asyncio
async def test_status_query_resume_invokes_status_handler(tmp_path, monkeypatch):
    """LLM-classified ``status_query`` must invoke the rich DB-lookup handler
    instead of the pre-fix static placeholder. Verified by seeding a task
    whose title matches the query subject and asserting the formatted
    "Found N matching item(s)" output reaches ``_send_telegram_via_resume``."""
    await _fresh_db(tmp_path, monkeypatch)
    sent = []

    async def fake_send(chat_id, text, *, parse_mode=None):
        sent.append((chat_id, text, parse_mode))
        return True

    monkeypatch.setattr(
        "src.app.telegram_bot._send_telegram_via_resume", fake_send,
    )
    try:
        from src.app.telegram_bot import TelegramInterface, set_telegram

        bot = object.__new__(TelegramInterface)
        bot.app = MagicMock()
        bot.app.bot.send_message = AsyncMock()
        bot.user_last_task_id = {}
        bot._pending_clarifications = {}
        bot._pending_mission = {}
        bot._pending_action = {}
        set_telegram(bot)

        # Seed a task whose title contains the subject keyword.
        from src.infra.db import add_task

        await add_task(
            title="coffee machine search",
            description="Hunting for an espresso machine",
            tier="auto",
            priority=8,
            agent_type="shopping_advisor",
            context={"chat_id": 4321},
        )

        await bot._route_classified_message(
            chat_id=4321,
            text="how is the coffee machine search going?",
            classification={"type": "status_query"},
        )

        # The fix routes through _build_status_query_response, which formats
        # "📊 Found N matching item(s):". The pre-fix static placeholder was
        # "📊 Use /tasks or /missions to see status." — assert we did NOT
        # send that and DID send the rich match.
        joined = "\n".join(m[1] for m in sent)
        assert "Use /tasks or /missions to see status" not in joined, (
            f"static placeholder leaked through; sent={sent!r}"
        )
        assert any("Found" in m[1] and "matching" in m[1] for m in sent), (
            f"rich status_query lookup did not run; sent={sent!r}"
        )
        # Title preview must reach the user.
        assert any("coffee machine" in m[1] for m in sent)
    finally:
        from src.app.telegram_bot import set_telegram
        set_telegram(None)  # type: ignore[arg-type]
        await _close_db()
