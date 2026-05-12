"""Z6 polish P3 — verify /credential schema and /credential log routing.

T2B added schema, T2C added log. This test pins the dispatcher behaviour
so a future refactor can't silently regress the subcommand wiring.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_iface():
    from src.app.telegram_bot import TelegramInterface
    iface = TelegramInterface.__new__(TelegramInterface)
    iface._reply = AsyncMock()
    return iface


def test_cmd_credential_help_lists_schema_and_log_subcommands():
    """The no-args usage message advertises both subcommands."""
    import inspect
    from src.app.telegram_bot import TelegramInterface

    src = inspect.getsource(TelegramInterface.cmd_credential)
    assert "/credential schema" in src, (
        "P3: usage text must advertise /credential schema"
    )
    assert "/credential log" in src, (
        "P3: usage text must advertise /credential log"
    )


@pytest.mark.asyncio
async def test_credential_schema_dispatches_to_describe_schema(monkeypatch):
    """`/credential schema <svc>` calls security.credential_schemas.describe_schema."""
    from src.app import telegram_bot as tb_mod
    iface = _make_iface()

    fake_describe = MagicMock(return_value="*schema for github*")
    import src.security.credential_schemas as schemas_mod
    monkeypatch.setattr(schemas_mod, "describe_schema", fake_describe)

    update = MagicMock()
    ctx = MagicMock()
    ctx.args = ["schema", "github"]
    await iface.cmd_credential(update, ctx)
    fake_describe.assert_called_once_with("github")
    # _reply should have been awaited once with the schema text.
    iface._reply.assert_awaited()
    last_call = iface._reply.await_args_list[-1]
    assert "schema for github" in last_call.args[1]


@pytest.mark.asyncio
async def test_credential_log_dispatches_to_recent_events(monkeypatch):
    """`/credential log <svc> [N]` calls credential_audit.recent_events."""
    from src.app import telegram_bot as tb_mod
    iface = _make_iface()

    fake_recent = AsyncMock(return_value=[{
        "service_name": "github", "agent": "coder", "action": "get",
        "success": True, "accessed_at": "2026-05-12T10:00:00",
        "mission_id": 7, "error": None,
    }])
    import src.security.credential_audit as audit_mod
    monkeypatch.setattr(audit_mod, "recent_events", fake_recent)

    update = MagicMock()
    ctx = MagicMock()
    ctx.args = ["log", "github", "5"]
    await iface.cmd_credential(update, ctx)
    fake_recent.assert_awaited_once_with("github", limit=5)
    iface._reply.assert_awaited()
    last_call = iface._reply.await_args_list[-1]
    body = last_call.args[1]
    assert "Access log: github" in body
    assert "agent=coder" in body


@pytest.mark.asyncio
async def test_credential_log_default_limit_20(monkeypatch):
    """No N arg → default limit=20."""
    iface = _make_iface()
    fake_recent = AsyncMock(return_value=[])
    import src.security.credential_audit as audit_mod
    monkeypatch.setattr(audit_mod, "recent_events", fake_recent)
    update = MagicMock()
    ctx = MagicMock()
    ctx.args = ["log", "github"]
    await iface.cmd_credential(update, ctx)
    fake_recent.assert_awaited_once_with("github", limit=20)


def test_credential_command_is_registered_in_handler_setup():
    """The /credential handler must be wired in _setup_handlers."""
    import inspect
    from src.app.telegram_bot import TelegramInterface
    src = inspect.getsource(TelegramInterface._setup_handlers)
    assert 'CommandHandler("credential"' in src, (
        "/credential must be registered with python-telegram-bot"
    )
    assert "self.cmd_credential" in src
