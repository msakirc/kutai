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


@pytest.fixture(autouse=True)
def _admin_chat_is_caller(monkeypatch):
    """The /yalayut command is owner-only (Fix 4). The stub updates here use
    chat id 123, so make 123 the configured admin chat for functional tests.
    The dedicated guard tests override this with their own monkeypatch."""
    import src.app.telegram_bot as _tg_mod
    monkeypatch.setattr(_tg_mod, "TELEGRAM_ADMIN_CHAT_ID", "123")


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


# ── Fix 4 — /yalayut admin guard ────────────────────────────────────────────


def test_cmd_yalayut_refuses_non_admin_chat(loop, monkeypatch):
    """A non-admin chat must be refused — /yalayut enable can flip a
    T3-quarantined artifact to enabled with no re-vetting, so the command
    is owner-only."""
    async def _run():
        import src.app.telegram_bot as _tg_mod
        await init_db()
        # Configure an admin chat that is NOT the caller's chat (123).
        monkeypatch.setattr(_tg_mod, "TELEGRAM_ADMIN_CHAT_ID", "999")

        tg = _telegram()
        update = _StubUpdate("/yalayut enable 1")
        ctx = SimpleNamespace(args=["enable", "1"])
        await tg.cmd_yalayut(update, ctx)

        assert update.message.replies, "must reply"
        reply = update.message.replies[-1][0].lower()
        assert "not authorized" in reply, (
            f"non-admin chat must be refused; got {reply!r}")
    loop.run_until_complete(_run())


def test_cmd_yalayut_allows_admin_chat(loop, monkeypatch):
    """The configured admin chat must pass the guard (overview reply)."""
    async def _run():
        import src.app.telegram_bot as _tg_mod
        await init_db()
        # Admin chat == caller chat (123).
        monkeypatch.setattr(_tg_mod, "TELEGRAM_ADMIN_CHAT_ID", "123")

        tg = _telegram()
        update = _StubUpdate("/yalayut")
        ctx = SimpleNamespace(args=[])
        await tg.cmd_yalayut(update, ctx)

        assert update.message.replies
        reply = update.message.replies[-1][0].lower()
        assert "not authorized" not in reply
        assert "catalog" in reply or "yalayut" in reply
    loop.run_until_complete(_run())


def test_yalayut_callback_refuses_non_admin_chat(loop, monkeypatch):
    """The inline-button handler must also reject a non-admin chat."""
    async def _run():
        import src.app.telegram_bot as _tg_mod
        await init_db()
        monkeypatch.setattr(_tg_mod, "TELEGRAM_ADMIN_CHAT_ID", "999")

        tg = _telegram()
        answered = []
        cq = SimpleNamespace(
            data="yal:vet_approve:1",
            answer=lambda *a, **k: _async_noop(answered),
            edit_message_text=lambda *a, **k: _async_noop([]),
            message=_StubMessage(),
        )
        update = SimpleNamespace(callback_query=cq,
                                 effective_chat=SimpleNamespace(id=123))
        await tg.handle_yalayut_callback(update, SimpleNamespace())
        # answered exactly once (the "not authorized" answer), no action ran.
        assert answered, "callback must be answered (refused)"
    loop.run_until_complete(_run())
