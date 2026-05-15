"""Tests for Z7 T1D (B9) — mr_roboto/audit_log.py.

Covers:
- log_external_send() writes a row with correct sha256 hash and gzip+base64 encoding.
- decode_content() round-trips the encoded body.
- search_sends() filters by recipient, channel, mission_id.
- pending_audit_gaps() returns gaps (mocked action_confirmations).
- audit_completeness_check verb in mr_roboto dispatcher raises alert for gaps.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import pytest


# ── DB fixture ────────────────────────────────────────────────────────────────

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
    from src.infra.db import init_db, get_db
    await init_db()
    return await get_db()


# ── Encoding helpers ──────────────────────────────────────────────────────────

class TestEncoding:
    def test_content_hash_is_sha256(self):
        from mr_roboto.audit_log import _encode_content
        body = "Hello, founder!"
        content_hash, _ = _encode_content(body)
        expected = hashlib.sha256(body.encode("utf-8")).hexdigest()
        assert content_hash == expected

    def test_content_md_is_gzip_base64(self):
        from mr_roboto.audit_log import _encode_content
        body = "Hello, founder!"
        _, content_md = _encode_content(body)
        compressed = base64.b64decode(content_md.encode("ascii"))
        raw = gzip.decompress(compressed)
        assert raw.decode("utf-8") == body

    def test_decode_content_round_trips(self):
        from mr_roboto.audit_log import _encode_content, decode_content
        original = "Telegram message body: discount 50%!"
        _, content_md = _encode_content(original)
        assert decode_content(content_md) == original

    def test_bytes_body_hashed_correctly(self):
        from mr_roboto.audit_log import _encode_content
        body = b"\x00\x01\x02binary"
        content_hash, content_md = _encode_content(body)
        expected = hashlib.sha256(body).hexdigest()
        assert content_hash == expected

    def test_empty_body(self):
        from mr_roboto.audit_log import _encode_content, decode_content
        body = ""
        content_hash, content_md = _encode_content(body)
        assert content_hash == hashlib.sha256(b"").hexdigest()
        assert decode_content(content_md) == ""


# ── log_external_send tests ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_log_external_send_writes_row(tmp_path, monkeypatch):
    db = await _setup_db(tmp_path, monkeypatch)
    from mr_roboto.audit_log import log_external_send
    log_id = await log_external_send(
        channel="telegram",
        content="Hello founder!",
        recipient="@founder",
        source_mission_id=42,
        reversibility="irreversible",
    )
    assert isinstance(log_id, int)
    assert log_id > 0

    cur = await db.execute(
        "SELECT log_id, channel, recipient, content_hash, content_md, "
        "       source_mission_id, reversibility "
        "FROM external_comms_log WHERE log_id = ?",
        (log_id,),
    )
    row = await cur.fetchone()
    # SELECT order: log_id=0, channel=1, recipient=2, content_hash=3, content_md=4,
    #               source_mission_id=5, reversibility=6
    assert row is not None
    assert row[1] == "telegram"
    assert row[2] == "@founder"
    content_hash = row[3]
    content_md = row[4]
    assert content_hash == hashlib.sha256(b"Hello founder!").hexdigest()
    from mr_roboto.audit_log import decode_content
    assert decode_content(content_md) == "Hello founder!"
    assert row[5] == 42  # source_mission_id
    assert row[6] == "irreversible"


@pytest.mark.asyncio
async def test_log_external_send_returns_correct_hash(tmp_path, monkeypatch):
    await _setup_db(tmp_path, monkeypatch)
    from mr_roboto.audit_log import log_external_send
    body = "SMS alert: payment failed"
    log_id = await log_external_send(
        channel="sms",
        content=body,
        recipient="+15551234567",
        reversibility="irreversible",
    )
    from src.infra.db import get_db
    db = await get_db()
    cur = await db.execute(
        "SELECT content_hash FROM external_comms_log WHERE log_id = ?",
        (log_id,),
    )
    row = await cur.fetchone()
    expected = hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert row[0] == expected


@pytest.mark.asyncio
async def test_log_external_send_optional_fields(tmp_path, monkeypatch):
    await _setup_db(tmp_path, monkeypatch)
    from mr_roboto.audit_log import log_external_send
    log_id = await log_external_send(
        channel="email",
        content="Newsletter",
        # No recipient, no mission_id
    )
    from src.infra.db import get_db
    db = await get_db()
    cur = await db.execute(
        "SELECT recipient, source_mission_id FROM external_comms_log WHERE log_id = ?",
        (log_id,),
    )
    row = await cur.fetchone()
    assert row[0] is None
    assert row[1] is None


# ── search_sends tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_sends_by_recipient(tmp_path, monkeypatch):
    await _setup_db(tmp_path, monkeypatch)
    from mr_roboto.audit_log import log_external_send, search_sends
    await log_external_send(channel="telegram", content="msg1", recipient="@alice")
    await log_external_send(channel="telegram", content="msg2", recipient="@bob")
    await log_external_send(channel="sms", content="msg3", recipient="+1234")

    results = await search_sends(recipient="@alice")
    assert len(results) == 1
    assert results[0]["recipient"] == "@alice"


@pytest.mark.asyncio
async def test_search_sends_by_channel(tmp_path, monkeypatch):
    await _setup_db(tmp_path, monkeypatch)
    from mr_roboto.audit_log import log_external_send, search_sends
    await log_external_send(channel="telegram", content="t1")
    await log_external_send(channel="sms", content="s1")
    await log_external_send(channel="telegram", content="t2")

    results = await search_sends(channel="telegram")
    assert all(r["channel"] == "telegram" for r in results)
    assert len(results) == 2


@pytest.mark.asyncio
async def test_search_sends_by_mission_id(tmp_path, monkeypatch):
    await _setup_db(tmp_path, monkeypatch)
    from mr_roboto.audit_log import log_external_send, search_sends
    await log_external_send(channel="telegram", content="m42", source_mission_id=42)
    await log_external_send(channel="telegram", content="m99", source_mission_id=99)

    results = await search_sends(mission_id=42)
    assert len(results) == 1
    assert results[0]["source_mission_id"] == 42


@pytest.mark.asyncio
async def test_search_sends_empty_result(tmp_path, monkeypatch):
    await _setup_db(tmp_path, monkeypatch)
    from mr_roboto.audit_log import search_sends
    results = await search_sends(recipient="@nobody")
    assert results == []


# ── pending_audit_gaps tests ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pending_audit_gaps_empty_when_no_vendor_calls(tmp_path, monkeypatch):
    await _setup_db(tmp_path, monkeypatch)
    from mr_roboto.audit_log import pending_audit_gaps
    gaps = await pending_audit_gaps(window_minutes=5)
    assert gaps == []


@pytest.mark.asyncio
async def test_pending_audit_gaps_finds_gap(tmp_path, monkeypatch):
    """A vendor_call row with reversibility != 'full' and no comms log row = gap."""
    db = await _setup_db(tmp_path, monkeypatch)
    # Insert a fake action_confirmations row older than 5 minutes
    import datetime as _dt
    old_ts = (_dt.datetime.utcnow() - _dt.timedelta(minutes=10)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    # Check if action_confirmations table exists; if not, skip
    try:
        await db.execute(
            "INSERT INTO action_confirmations "
            "(id, action, mission_id, reversibility, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (9001, "vendor_call", 1, "partial", "approved", old_ts),
        )
        await db.commit()
    except Exception:
        pytest.skip("action_confirmations table not available in this schema")

    from mr_roboto.audit_log import pending_audit_gaps
    gaps = await pending_audit_gaps(window_minutes=5)
    gap_ids = [g["vendor_call_id"] for g in gaps]
    assert 9001 in gap_ids


@pytest.mark.asyncio
async def test_pending_audit_gaps_no_gap_when_logged(tmp_path, monkeypatch):
    """A vendor_call with an existing comms log row is not a gap."""
    db = await _setup_db(tmp_path, monkeypatch)
    import datetime as _dt
    old_ts = (_dt.datetime.utcnow() - _dt.timedelta(minutes=10)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    try:
        await db.execute(
            "INSERT INTO action_confirmations "
            "(id, action, mission_id, reversibility, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (9002, "vendor_call", 1, "partial", "approved", old_ts),
        )
        await db.commit()
    except Exception:
        pytest.skip("action_confirmations table not available in this schema")

    # Now log the send with vendor_call_id=9002
    from mr_roboto.audit_log import log_external_send, pending_audit_gaps
    await log_external_send(
        channel="telegram", content="done", vendor_call_id=9002
    )
    gaps = await pending_audit_gaps(window_minutes=5)
    gap_ids = [g["vendor_call_id"] for g in gaps]
    assert 9002 not in gap_ids


# ── audit_completeness_check dispatcher test ──────────────────────────────────

@pytest.mark.asyncio
async def test_audit_completeness_check_no_gaps_completes(tmp_path, monkeypatch):
    """When there are no gaps, the mechanical action completes with gaps_found=0."""
    await _setup_db(tmp_path, monkeypatch)

    # Patch pending_audit_gaps to return empty
    import mr_roboto.audit_log as _al
    monkeypatch.setattr(_al, "pending_audit_gaps", lambda **_: _coro([]))

    async def _coro(val):
        return val

    monkeypatch.setattr(_al, "pending_audit_gaps", lambda **_: _coro([]))

    from mr_roboto import run as mr_run
    task = {
        "agent_type": "mechanical",
        "mission_id": 1,
        "context": {"payload": {"action": "audit_completeness_check"}},
    }
    # mr_roboto.run() expects payload at task["payload"]
    task2 = {
        "mission_id": 1,
        "payload": {"action": "audit_completeness_check", "window_minutes": 5},
    }
    result = await mr_run(task2)
    assert result.status == "completed"
    assert result.result.get("gaps_found") == 0


@pytest.mark.asyncio
async def test_audit_completeness_check_gaps_trigger_alerts(tmp_path, monkeypatch):
    """When gaps exist, escalate_to_founder is called for each gap."""
    await _setup_db(tmp_path, monkeypatch)

    fake_gaps = [
        {"vendor_call_id": 1, "verb": "vendor_call", "mission_id": 10, "created_at": "2026-01-01 10:00:00"},
        {"vendor_call_id": 2, "verb": "notify_user", "mission_id": 11, "created_at": "2026-01-01 11:00:00"},
    ]
    calls: list = []

    async def _fake_gaps(**_):
        return fake_gaps

    async def _fake_escalate(task):
        calls.append(task)
        return {"ok": True}

    import mr_roboto.audit_log as _al
    import mr_roboto.executors.escalate_to_founder as _ef
    monkeypatch.setattr(_al, "pending_audit_gaps", _fake_gaps)
    monkeypatch.setattr(_ef, "run", _fake_escalate)

    from mr_roboto import run as mr_run
    task = {
        "mission_id": 1,
        "payload": {"action": "audit_completeness_check", "window_minutes": 5},
    }
    result = await mr_run(task)
    assert result.status == "completed"
    assert result.result.get("gaps_found") == 2
    assert result.result.get("alerts_sent") == 2
    assert len(calls) == 2


# ── EXTERNAL_PUBLISH_VERBS ────────────────────────────────────────────────────

def test_external_publish_verbs_includes_key_verbs():
    from mr_roboto.audit_log import EXTERNAL_PUBLISH_VERBS
    assert "notify_user" in EXTERNAL_PUBLISH_VERBS
    assert "clarify" in EXTERNAL_PUBLISH_VERBS
    assert "escalate_to_founder" in EXTERNAL_PUBLISH_VERBS
    assert "vendor_call" in EXTERNAL_PUBLISH_VERBS


def test_external_publish_verbs_excludes_readonly():
    from mr_roboto.audit_log import EXTERNAL_PUBLISH_VERBS
    assert "verify_artifacts" not in EXTERNAL_PUBLISH_VERBS
    assert "workspace_snapshot" not in EXTERNAL_PUBLISH_VERBS
    assert "git_commit" not in EXTERNAL_PUBLISH_VERBS
