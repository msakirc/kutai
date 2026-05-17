"""Z7 B9 — external-comms audit log: REAL run() dispatch-path tests.

The existing test_audit_log.py "real-path" tests are tautological — they
call log_publish_action() directly, never mr_roboto.run(). If the
audit-log wiring inside run() regressed (the `_log_external_publish`
call removed), those tests would stay green.

These tests drive the genuine seam: mr_roboto.run(task) -> _run_dispatch
-> _log_external_publish -> log_publish_action -> log_external_send -> DB.
Only the outermost external boundary (Telegram send) is faked; the audit
row write is real SQLite. Reintroduce the B9 bug (drop the
_log_external_publish call from run()) and these fail.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
async def _db_reset():
    import src.infra.db as _dbmod
    if _dbmod._db_connection is not None:
        try:
            await _dbmod._db_connection.close()
        except Exception:
            pass
    _dbmod._db_connection = None
    yield
    if _dbmod._db_connection is not None:
        try:
            await _dbmod._db_connection.close()
        except Exception:
            pass
    _dbmod._db_connection = None


async def _setup_db(tmp_path, monkeypatch):
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()


class _FakeBot:
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text, reply_markup=None):
        self.sent.append((chat_id, text))


class _FakeTelegram:
    def __init__(self):
        self.app = type("A", (), {"bot": _FakeBot()})()
        self.notifications = []

    async def send_notification(self, text, retries=2, reply_markup=None):
        self.notifications.append(text)


@pytest.mark.asyncio
async def test_run_notify_user_writes_audit_row(tmp_path, monkeypatch):
    """mr_roboto.run() of an external-publish verb lands an external_comms_log row.

    Drives the real run() -> _run_dispatch -> _log_external_publish chain.
    Only the Telegram send (true external boundary) is faked.
    """
    await _setup_db(tmp_path, monkeypatch)
    # Bypass the critic gate — it is an unrelated LLM hop, not the seam.
    monkeypatch.setenv("KUTAI_CRITIC_GATE", "off")
    # Fake only the Telegram boundary inside the notify_user executor.
    import mr_roboto.notify_user as _nu
    monkeypatch.setattr(_nu, "get_telegram", lambda: _FakeTelegram())

    import mr_roboto
    from mr_roboto.audit_log import search_sends

    task = {
        "id": 1,
        "mission_id": 77,
        "payload": {
            "action": "notify_user",
            "message": "Mission 77 deployed to production.",
            "chat_id": 12345,
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed", f"run() failed: {action.error}"

    rows = await search_sends(mission_id=77)
    assert len(rows) == 1, (
        "mr_roboto.run() of notify_user produced NO external_comms_log row — "
        "the B9 audit trail is not wired into the dispatch path"
    )
    assert rows[0]["channel"] == "telegram"


@pytest.mark.asyncio
async def test_run_non_publish_verb_writes_no_audit_row(tmp_path, monkeypatch):
    """A non-external verb dispatched via run() must NOT write an audit row.

    Confirms the audit log is selective (keyed on EXTERNAL_PUBLISH_VERBS),
    not blindly logging every mechanical verb.
    """
    await _setup_db(tmp_path, monkeypatch)
    import mr_roboto
    from mr_roboto.audit_log import search_sends

    # run_cmd is a plain local verb — not an external publish.
    task = {
        "id": 2,
        "mission_id": 88,
        "payload": {"action": "run_cmd", "cmd": "echo hello"},
    }
    await mr_roboto.run(task)
    assert await search_sends(mission_id=88) == [], (
        "a non-external verb wrote an external_comms_log row — audit log "
        "is not gated on EXTERNAL_PUBLISH_VERBS"
    )


@pytest.mark.asyncio
async def test_audit_completeness_check_handler_detects_gap(tmp_path, monkeypatch):
    """The audit_completeness_check posthook handler runs a REAL gap scan.

    Seeds an external-publish action_confirmation with NO matching
    external_comms_log row — the handler must report a 'warning' with the
    gap, proving it is not a no-op stub.
    """
    await _setup_db(tmp_path, monkeypatch)
    from src.infra.db import get_db

    db = await get_db()
    # An irreversible external confirmation, requested >5min ago, unaudited.
    await db.execute(
        "INSERT INTO tasks (id, mission_id, title, description, agent_type, status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (900, 55, "publish status", "publish status", "mechanical", "completed"),
    )
    await db.execute(
        "INSERT INTO action_confirmations "
        "(task_id, verb, reversibility, payload_summary, requested_at, verdict) "
        "VALUES (?, ?, ?, ?, datetime('now', '-10 minutes'), ?)",
        (900, "incident/publish_status", "irreversible", "publish", "approved"),
    )
    await db.commit()

    from general_beckman.posthook_handlers.audit_completeness_check import handle

    res = await handle({"id": 901, "mission_id": 55}, {})
    assert res["status"] == "warning", (
        f"audit_completeness_check did not flag the unaudited send: {res}"
    )
    assert res.get("gaps"), "warning returned but no gaps listed"
    assert any(
        g.get("verb") == "incident/publish_status" for g in res["gaps"]
    )


@pytest.mark.asyncio
async def test_audit_completeness_check_handler_clean_when_logged(
    tmp_path, monkeypatch
):
    """No gap when the external send has a matching external_comms_log row."""
    await _setup_db(tmp_path, monkeypatch)
    from src.infra.db import get_db
    from mr_roboto.audit_log import log_external_send

    db = await get_db()
    await db.execute(
        "INSERT INTO tasks (id, mission_id, title, description, agent_type, status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (910, 66, "publish status", "publish status", "mechanical", "completed"),
    )
    cur = await db.execute(
        "INSERT INTO action_confirmations "
        "(task_id, verb, reversibility, payload_summary, requested_at, verdict) "
        "VALUES (?, ?, ?, ?, datetime('now', '-10 minutes'), ?)",
        (910, "incident/publish_status", "irreversible", "publish", "approved"),
    )
    await db.commit()
    confirmation_id = cur.lastrowid

    # Audit row joined on vendor_call_id == action_confirmations.id closes the gap.
    await log_external_send(
        channel="public",
        content="We resolved the incident.",
        source_mission_id=66,
        vendor_call_id=confirmation_id,
    )

    from general_beckman.posthook_handlers.audit_completeness_check import handle

    res = await handle({"id": 911, "mission_id": 66}, {})
    assert res["status"] == "ok", f"expected clean audit, got: {res}"
