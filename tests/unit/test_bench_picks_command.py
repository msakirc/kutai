"""Tests for the /bench_picks Telegram command."""
import sys
import types
import pytest
from unittest.mock import AsyncMock, MagicMock


def _stub_telegram_modules():
    """Install minimal telegram stubs so telegram_bot.py can be imported."""
    if "telegram" in sys.modules:
        return  # already stubbed or real package present

    # telegram top-level
    tg = types.ModuleType("telegram")
    for name in ("Update", "InlineKeyboardButton", "InlineKeyboardMarkup",
                 "BotCommand", "ReplyKeyboardMarkup", "KeyboardButton"):
        setattr(tg, name, MagicMock())
    sys.modules["telegram"] = tg

    # telegram.ext
    tg_ext = types.ModuleType("telegram.ext")
    for name in ("Application", "CommandHandler", "MessageHandler",
                 "CallbackQueryHandler", "filters", "ContextTypes"):
        setattr(tg_ext, name, MagicMock())
    sys.modules["telegram.ext"] = tg_ext


@pytest.fixture(autouse=True)
def _patch_telegram(monkeypatch):
    _stub_telegram_modules()
    # Stub out heavy src imports that telegram_bot pulls in at module level
    for mod in [
        "src.infra.db",
        "src.memory.conversations",
        "src.memory.ingest",
        "src.memory.preferences",
        "src.tools.workspace",
        "src.tools.free_apis",
    ]:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()

    # Stub relative-import helpers that telegram_bot uses
    for mod in [
        "src.infra.times",
        "src.app.config",
    ]:
        if mod not in sys.modules:
            m = MagicMock()
            sys.modules[mod] = m

    yield

    # Remove telegram_bot from sys.modules so env-var patches don't leak
    sys.modules.pop("src.app.telegram_bot", None)


@pytest.mark.asyncio
async def test_bench_picks_empty(tmp_path, monkeypatch):
    import aiosqlite
    db_path = tmp_path / "test.db"
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute(
            """CREATE TABLE model_pick_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                task_name TEXT NOT NULL, picked_model TEXT NOT NULL, picked_score REAL NOT NULL
            )"""
        )
        await db.commit()

    from src.app.telegram_bot import TelegramInterface
    import src.app.telegram_bot as tg_mod
    monkeypatch.setattr(tg_mod, "DB_PATH", str(db_path))

    bot = TelegramInterface.__new__(TelegramInterface)
    bot._reply = AsyncMock()

    update = MagicMock()
    ctx = MagicMock()
    await bot.cmd_bench_picks(update, ctx)

    combined = " ".join(str(a) for a in bot._reply.call_args.args) + str(bot._reply.call_args.kwargs)
    assert "no pick log" in combined.lower() or "no entries" in combined.lower()


@pytest.mark.asyncio
async def test_bench_picks_with_rows(tmp_path, monkeypatch):
    import aiosqlite
    db_path = tmp_path / "test.db"
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute(
            """CREATE TABLE model_pick_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                task_name TEXT NOT NULL, picked_model TEXT NOT NULL, picked_score REAL NOT NULL
            )"""
        )
        for (task, model, score) in [
            ("coder", "qwen3-coder", 9.1),
            ("coder", "qwen3-coder", 9.2),
            ("coder", "apriel", 7.5),
            ("researcher", "gpt-oss", 8.8),
        ]:
            await db.execute(
                "INSERT INTO model_pick_log (task_name, picked_model, picked_score) VALUES (?, ?, ?)",
                (task, model, score),
            )
        await db.commit()

    from src.app.telegram_bot import TelegramInterface
    import src.app.telegram_bot as tg_mod
    monkeypatch.setattr(tg_mod, "DB_PATH", str(db_path))

    bot = TelegramInterface.__new__(TelegramInterface)
    bot._reply = AsyncMock()

    update = MagicMock()
    ctx = MagicMock()
    await bot.cmd_bench_picks(update, ctx)

    combined = " ".join(str(a) for a in bot._reply.call_args.args) + str(bot._reply.call_args.kwargs)
    assert "qwen3-coder" in combined
    assert "coder" in combined
    assert "2" in combined  # count for qwen3-coder
