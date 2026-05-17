import asyncio
from types import SimpleNamespace

import pytest

from src.infra.db import init_db, get_db
from src.app.telegram_bot import TelegramInterface


@pytest.fixture
def loop():
    lp = asyncio.new_event_loop()
    yield lp
    lp.close()


class _StubMessage:
    def __init__(self, text=""):
        self.text = text
        self.replies = []

    async def reply_text(self, text, **kwargs):
        self.replies.append((text, kwargs))
        return SimpleNamespace(message_id=1)


class _StubUpdate:
    def __init__(self, text=""):
        self.message = _StubMessage(text)
        self.effective_chat = SimpleNamespace(id=123)
        self.effective_user = SimpleNamespace(id=123)
        self.callback_query = None


def _telegram():
    # build the interface without a live bot — only handler bodies tested.
    return TelegramInterface.__new__(TelegramInterface)


def test_cmd_yalayut_overview(loop):
    async def _run():
        await init_db()
        tg = _telegram()
        update = _StubUpdate("/yalayut")
        ctx = SimpleNamespace(args=[])
        await tg.cmd_yalayut(update, ctx)
        assert update.message.replies
        text = update.message.replies[0][0]
        assert "yalayut" in text.lower() or "catalog" in text.lower()
    loop.run_until_complete(_run())


def test_cmd_yalayut_sources_pending(loop):
    async def _run():
        await init_db()
        db = await get_db()
        await db.execute(
            "INSERT INTO yalayut_source_candidates "
            "(candidate_source_id, source_type, endpoint, state, proposed_at) "
            "VALUES ('github:a/b', 'github_path', 'x', 'pending', "
            "datetime('now'))")
        await db.commit()
        tg = _telegram()
        update = _StubUpdate("/yalayut sources pending")
        ctx = SimpleNamespace(args=["sources", "pending"])
        await tg.cmd_yalayut(update, ctx)
        assert update.message.replies
        assert "github:a/b" in update.message.replies[-1][0]
    loop.run_until_complete(_run())


def test_yalayut_callback_approve_source(loop):
    async def _run():
        await init_db()
        db = await get_db()
        cur = await db.execute(
            "INSERT INTO yalayut_source_candidates "
            "(candidate_source_id, source_type, endpoint, state, proposed_at) "
            "VALUES ('github:cb/src', 'github_path', 'x', 'pending', "
            "datetime('now'))")
        await db.commit()
        cand_id = cur.lastrowid

        tg = _telegram()
        answered = []
        edited = []
        cq = SimpleNamespace(
            data=f"yal:src_approve_trusted:{cand_id}",
            answer=lambda *a, **k: _async_noop(answered),
            edit_message_text=lambda *a, **k: _async_noop(edited),
            message=_StubMessage(),
        )
        update = SimpleNamespace(callback_query=cq,
                                 effective_chat=SimpleNamespace(id=123))
        await tg.handle_yalayut_callback(update, SimpleNamespace())
        cur = await db.execute(
            "SELECT trusted FROM yalayut_sources "
            "WHERE source_id = 'github:cb/src'")
        row = await cur.fetchone()
        await cur.close()
        assert row is not None and row[0] == 1
    loop.run_until_complete(_run())


def _async_noop(sink):
    async def _c():
        sink.append(True)
    return _c()
