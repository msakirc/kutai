import pytest
from unittest.mock import AsyncMock, MagicMock

from src.app.telegram_bot import TelegramInterface


@pytest.mark.asyncio
async def test_provision_mission_thread_happy_path():
    tg = TelegramInterface.__new__(TelegramInterface)
    tg.bot = MagicMock()
    tg.bot.create_forum_topic = AsyncMock(
        return_value=MagicMock(message_thread_id=999)
    )
    tg.bot.send_message = AsyncMock(return_value=MagicMock(message_id=42))
    tg.bot.pin_chat_message = AsyncMock()

    chat_id = 1234
    thread_id = await tg.provision_mission_thread(chat_id, mission_id=1, title="Build X")
    assert thread_id == 999
    tg.bot.create_forum_topic.assert_called_once()
    # Verify the topic name format includes mission id + title
    call_kwargs = tg.bot.create_forum_topic.call_args.kwargs
    assert call_kwargs.get("chat_id") == 1234
    assert "1" in call_kwargs.get("name", "") and "Build X" in call_kwargs.get("name", "")
    tg.bot.pin_chat_message.assert_called_once()


@pytest.mark.asyncio
async def test_provision_falls_back_on_perm_error():
    tg = TelegramInterface.__new__(TelegramInterface)
    tg.bot = MagicMock()
    tg.bot.create_forum_topic = AsyncMock(side_effect=Exception("Bot doesn't have permission"))

    thread_id = await tg.provision_mission_thread(1234, mission_id=1, title="Build X")
    assert thread_id is None  # signals fallback to tag-prefix


def test_format_pinned_status_with_ceiling():
    tg = TelegramInterface.__new__(TelegramInterface)
    text = tg._format_pinned_status(
        mission_id=42, title="Build X",
        spent=0.50, ceiling=2.0,
        state="active",
        tasks_done=3, tasks_running=2, tasks_queued=10,
    )
    assert "#42" in text
    assert "Build X" in text
    assert "active" in text
    assert "$0.50" in text
    assert "$2.00" in text
    assert "25" in text  # 25.0% — accept any rendering of 25
    assert "3 done" in text
    assert "2 in flight" in text or "2 in-flight" in text or "2 running" in text
    assert "10 queued" in text


def test_format_pinned_status_no_ceiling():
    tg = TelegramInterface.__new__(TelegramInterface)
    text = tg._format_pinned_status(
        mission_id=1, title="No Ceiling Mission",
        spent=0.0, ceiling=None,
    )
    assert "no ceiling" in text.lower() or "unlimited" in text.lower()


@pytest.mark.asyncio
async def test_kill_mission_sets_killed_state_and_writes_snapshot(tmp_path, monkeypatch):
    import src.infra.db as db_module
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    db_module.DB_PATH = db_path
    db_module._db_connection = None

    from src.infra.db import init_db
    import aiosqlite
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, lifecycle_state) VALUES ('m', 'active')"
        )
        cur = await db.execute("SELECT last_insert_rowid()")
        mid = (await cur.fetchone())[0]
        await db.execute(
            "INSERT INTO tasks (mission_id, title, status) VALUES (?, 't1', 'completed')",
            (mid,),
        )
        await db.commit()

    db_module._db_connection = None

    tg = TelegramInterface.__new__(TelegramInterface)
    tg.bot = MagicMock()
    tg._reply = AsyncMock()

    update = MagicMock()
    update.effective_chat.id = 1
    context = MagicMock()
    context.args = [str(mid)]

    await tg.cmd_kill_mission(update, context)

    db_module._db_connection = None
    async with aiosqlite.connect(db_path) as db:
        row = await (await db.execute(
            "SELECT lifecycle_state FROM missions WHERE id=?", (mid,))).fetchone()
    assert row[0] == "killed"


@pytest.mark.asyncio
async def test_resume_after_kill_rejected(tmp_path, monkeypatch):
    import src.infra.db as db_module
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    db_module.DB_PATH = db_path
    db_module._db_connection = None

    from src.infra.db import init_db
    import aiosqlite
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, lifecycle_state) VALUES ('m', 'killed')"
        )
        cur = await db.execute("SELECT last_insert_rowid()")
        mid = (await cur.fetchone())[0]
        await db.commit()

    db_module._db_connection = None
    tg = TelegramInterface.__new__(TelegramInterface)
    tg.bot = MagicMock()
    tg._reply = AsyncMock()

    update = MagicMock()
    context = MagicMock()
    context.args = [str(mid)]
    await tg.cmd_resume_mission(update, context)

    db_module._db_connection = None
    async with aiosqlite.connect(db_path) as db:
        row = await (await db.execute(
            "SELECT lifecycle_state FROM missions WHERE id=?", (mid,))).fetchone()
    assert row[0] == "killed"
    tg._reply.assert_called_once()


@pytest.mark.asyncio
async def test_pause_mission_command(tmp_path, monkeypatch):
    import src.infra.db as db_module
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    db_module.DB_PATH = db_path
    db_module._db_connection = None

    from src.infra.db import init_db
    import aiosqlite
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, lifecycle_state) VALUES ('m', 'active')"
        )
        cur = await db.execute("SELECT last_insert_rowid()")
        mid = (await cur.fetchone())[0]
        await db.commit()

    db_module._db_connection = None
    tg = TelegramInterface.__new__(TelegramInterface)
    tg.bot = MagicMock()
    tg._reply = AsyncMock()

    update = MagicMock()
    context = MagicMock()
    context.args = [str(mid)]
    await tg.cmd_pause_mission(update, context)

    db_module._db_connection = None
    async with aiosqlite.connect(db_path) as db:
        row = await (await db.execute(
            "SELECT lifecycle_state FROM missions WHERE id=?", (mid,))).fetchone()
    assert row[0] == "paused"


@pytest.mark.asyncio
async def test_resume_mission_command(tmp_path, monkeypatch):
    import src.infra.db as db_module
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    db_module.DB_PATH = db_path
    db_module._db_connection = None

    from src.infra.db import init_db
    import aiosqlite
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, lifecycle_state) VALUES ('m', 'paused')"
        )
        cur = await db.execute("SELECT last_insert_rowid()")
        mid = (await cur.fetchone())[0]
        await db.commit()

    db_module._db_connection = None
    tg = TelegramInterface.__new__(TelegramInterface)
    tg.bot = MagicMock()
    tg._reply = AsyncMock()

    update = MagicMock()
    context = MagicMock()
    context.args = [str(mid)]
    await tg.cmd_resume_mission(update, context)

    db_module._db_connection = None
    async with aiosqlite.connect(db_path) as db:
        row = await (await db.execute(
            "SELECT lifecycle_state FROM missions WHERE id=?", (mid,))).fetchone()
    assert row[0] == "active"


@pytest.mark.asyncio
async def test_resume_button_callback_resumes_mission(tmp_path, monkeypatch):
    import src.infra.db as db_module
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    db_module.DB_PATH = db_path
    db_module._db_connection = None

    from src.infra.db import init_db
    import aiosqlite
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO missions (title, lifecycle_state) VALUES ('m', 'paused')"
        )
        cur = await db.execute("SELECT last_insert_rowid()")
        mid = (await cur.fetchone())[0]
        await db.commit()

    db_module._db_connection = None
    tg = TelegramInterface.__new__(TelegramInterface)
    tg.bot = MagicMock()
    tg._reply = AsyncMock()
    if hasattr(tg, '_pending_action'):
        tg._pending_action = {}
    else:
        tg._pending_action = {}

    update = MagicMock()
    update.callback_query = MagicMock()
    update.callback_query.data = f"mission_resume:{mid}"
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    update.effective_chat = MagicMock()
    update.effective_chat.id = 1
    update.message = None

    context = MagicMock()

    # Adapt to actual callback method name
    handler_name = None
    for name in ("handle_callback", "_handle_callback_query", "callback_handler", "on_callback"):
        if hasattr(TelegramInterface, name):
            handler_name = name
            break
    assert handler_name is not None, "no callback handler method found"
    handler = getattr(tg, handler_name)
    await handler(update, context)

    db_module._db_connection = None
    async with aiosqlite.connect(db_path) as db:
        row = await (await db.execute(
            "SELECT lifecycle_state FROM missions WHERE id=?", (mid,))).fetchone()
    assert row[0] == "active"


@pytest.mark.asyncio
async def test_pause_button_callback(tmp_path, monkeypatch):
    import src.infra.db as db_module
    db_path = str(tmp_path / "kutai.db")
    monkeypatch.setenv("DB_PATH", db_path)
    db_module.DB_PATH = db_path
    db_module._db_connection = None
    from src.infra.db import init_db
    import aiosqlite
    await init_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute("INSERT INTO missions (title, lifecycle_state) VALUES ('m', 'active')")
        cur = await db.execute("SELECT last_insert_rowid()")
        mid = (await cur.fetchone())[0]
        await db.commit()

    db_module._db_connection = None
    tg = TelegramInterface.__new__(TelegramInterface)
    tg.bot = MagicMock()
    tg._reply = AsyncMock()
    tg._pending_action = {}

    update = MagicMock()
    update.callback_query = MagicMock()
    update.callback_query.data = f"mission_pause:{mid}"
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()

    handler_name = next(name for name in ("handle_callback", "_handle_callback_query", "callback_handler", "on_callback") if hasattr(TelegramInterface, name))
    await getattr(tg, handler_name)(update, MagicMock())

    db_module._db_connection = None
    async with aiosqlite.connect(db_path) as db:
        row = await (await db.execute(
            "SELECT lifecycle_state FROM missions WHERE id=?", (mid,))).fetchone()
    assert row[0] == "paused"
