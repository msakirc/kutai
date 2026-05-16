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
    """An action_confirmations row with reversibility != 'full' and no
    external_comms_log row = gap. Exercises the REAL action_confirmations
    schema (id, task_id, verb, reversibility, requested_at, ...)."""
    db = await _setup_db(tmp_path, monkeypatch)
    # Insert a task so mission_id can be derived via the tasks join.
    await db.execute(
        "INSERT INTO tasks (id, mission_id, title) VALUES (?, ?, ?)",
        (5001, 1, "publish task"),
    )
    # Insert a confirmation row requested more than 5 minutes ago.
    import datetime as _dt
    old_ts = (_dt.datetime.utcnow() - _dt.timedelta(minutes=10)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    await db.execute(
        "INSERT INTO action_confirmations "
        "(id, task_id, verb, reversibility, requested_at, verdict) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (9001, 5001, "notify_user", "partial", old_ts, "approved"),
    )
    await db.commit()

    from mr_roboto.audit_log import pending_audit_gaps
    gaps = await pending_audit_gaps(window_minutes=5)
    gap_ids = [g["vendor_call_id"] for g in gaps]
    assert 9001 in gap_ids
    gap = next(g for g in gaps if g["vendor_call_id"] == 9001)
    assert gap["verb"] == "notify_user"
    assert gap["mission_id"] == 1  # derived via tasks join


@pytest.mark.asyncio
async def test_pending_audit_gaps_no_gap_when_logged(tmp_path, monkeypatch):
    """A confirmation with a matching external_comms_log row is not a gap."""
    db = await _setup_db(tmp_path, monkeypatch)
    await db.execute(
        "INSERT INTO tasks (id, mission_id, title) VALUES (?, ?, ?)",
        (5002, 1, "publish task"),
    )
    import datetime as _dt
    old_ts = (_dt.datetime.utcnow() - _dt.timedelta(minutes=10)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    await db.execute(
        "INSERT INTO action_confirmations "
        "(id, task_id, verb, reversibility, requested_at, verdict) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (9002, 5002, "notify_user", "partial", old_ts, "approved"),
    )
    await db.commit()

    # Now log the send with vendor_call_id=9002 — closes the gap.
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
    # Telegram / founder-facing sends
    assert "notify_user" in EXTERNAL_PUBLISH_VERBS
    assert "clarify" in EXTERNAL_PUBLISH_VERBS
    assert "escalate_to_founder" in EXTERNAL_PUBLISH_VERBS
    # Public publish + real email
    assert "changelog/publish" in EXTERNAL_PUBLISH_VERBS
    assert "incident/publish_status" in EXTERNAL_PUBLISH_VERBS
    assert "publish_synchronized" in EXTERNAL_PUBLISH_VERBS
    assert "outreach/send" in EXTERNAL_PUBLISH_VERBS
    assert "email/send_via_provider" in EXTERNAL_PUBLISH_VERBS


def test_external_publish_verbs_excludes_readonly():
    from mr_roboto.audit_log import EXTERNAL_PUBLISH_VERBS
    assert "verify_artifacts" not in EXTERNAL_PUBLISH_VERBS
    assert "workspace_snapshot" not in EXTERNAL_PUBLISH_VERBS
    assert "git_commit" not in EXTERNAL_PUBLISH_VERBS


def test_every_publish_verb_has_a_channel():
    """Every external-publish verb must resolve a default channel so
    log_publish_action never writes a useless 'unknown' channel."""
    from mr_roboto.audit_log import EXTERNAL_PUBLISH_VERBS, _CHANNEL_BY_VERB
    missing = EXTERNAL_PUBLISH_VERBS - set(_CHANNEL_BY_VERB)
    assert not missing, f"verbs missing a default channel: {missing}"


def test_publish_verbs_are_real_dispatcher_verbs():
    """EXTERNAL_PUBLISH_VERBS must name verbs the mr_roboto dispatcher
    actually knows — a stale name would silently never log."""
    from mr_roboto.audit_log import EXTERNAL_PUBLISH_VERBS
    from mr_roboto.reversibility import VERB_REVERSIBILITY
    unknown = EXTERNAL_PUBLISH_VERBS - set(VERB_REVERSIBILITY)
    assert not unknown, f"publish verbs not in reversibility registry: {unknown}"


def _dispatcher_verb_strings() -> set[str]:
    """Parse mr_roboto/__init__.py for every `if action == "<verb>"` string.

    The reversibility registry can name verbs that no longer have a
    dispatcher branch (e.g. demo/distribute/flip_to_public is only an
    instruction string). This scans the *actual* dispatch chain so we catch
    a publish verb that would never be matched and never logged.
    """
    import ast
    import pathlib
    import mr_roboto

    src = pathlib.Path(mr_roboto.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    verbs: set[str] = set()
    for node in ast.walk(tree):
        # `action == "<verb>"`
        if isinstance(node, ast.Compare) and len(node.ops) == 1 \
                and isinstance(node.ops[0], ast.Eq):
            left, right = node.left, node.comparators[0]
            for a, b in ((left, right), (right, left)):
                if isinstance(a, ast.Name) and a.id == "action" \
                        and isinstance(b, ast.Constant) \
                        and isinstance(b.value, str):
                    verbs.add(b.value)
        # `action.startswith("<prefix>")` — record the prefix
        if isinstance(node, ast.Call) \
                and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "startswith" \
                and isinstance(node.func.value, ast.Name) \
                and node.func.value.id == "action" \
                and node.args and isinstance(node.args[0], ast.Constant):
            verbs.add(node.args[0].value)
    return verbs


def test_publish_verbs_match_dispatcher_if_chain():
    """Every EXTERNAL_PUBLISH_VERBS entry must be reachable by the real
    dispatcher — not merely present in the reversibility registry. A
    registry-only name (e.g. the `demo/distribute/flip_to_public` instruction
    string) would silently never produce an audit row.

    A verb is reachable if it is an exact `if action == ...` match, is
    covered by an `action.startswith(prefix)` branch, OR is an
    `oncall_action` sub-verb (dispatched through the on-call gateway, which
    IS a real `if action ==` branch)."""
    from mr_roboto.audit_log import EXTERNAL_PUBLISH_VERBS
    from mr_roboto.executors.oncall_action import WHITELISTED_VERBS

    dispatcher_verbs = _dispatcher_verb_strings()
    missing = set()
    for verb in EXTERNAL_PUBLISH_VERBS:
        if verb in dispatcher_verbs:
            continue
        if any(verb.startswith(p) and p != verb for p in dispatcher_verbs):
            continue
        if verb in WHITELISTED_VERBS:
            continue
        missing.add(verb)
    assert not missing, (
        f"publish verbs with no dispatcher branch (stale names): {missing}"
    )


def test_new_publish_verbs_present():
    """The Z7 fix Task 4 Gap 2 verbs are registered."""
    from mr_roboto.audit_log import EXTERNAL_PUBLISH_VERBS, _CHANNEL_BY_VERB
    for verb in ("init_mission_github_repo", "emit_preview_url",
                 "demo/distribute", "eas_submit", "fastlane"):
        assert verb in EXTERNAL_PUBLISH_VERBS, f"{verb} not registered"
        assert verb in _CHANNEL_BY_VERB, f"{verb} missing a default channel"


# ── log_publish_action — REAL-PATH wiring tests ───────────────────────────────
# These exercise the actual run() → dispatch → log_publish_action path. Only
# the outermost network/Telegram boundary is faked; the audit-log write is
# real (real SQLite, real INSERT). If B9 wiring regresses (decorator removed,
# wrong status key), these tests fail.

@pytest.mark.asyncio
async def test_log_publish_action_logs_completed_action(tmp_path, monkeypatch):
    """An Action(status='completed') for an external-publish verb writes a row."""
    await _setup_db(tmp_path, monkeypatch)
    from mr_roboto.audit_log import log_publish_action, search_sends
    from mr_roboto.actions import Action

    action = Action(status="completed", result={"sent": True})
    task = {
        "mission_id": 7,
        "payload": {
            "action": "notify_user",
            "message": "Mission 7 deployed.",
            "recipient": "@founder",
        },
    }
    log_id = await log_publish_action("notify_user", action, task)
    assert isinstance(log_id, int) and log_id > 0

    rows = await search_sends(mission_id=7)
    assert len(rows) == 1
    assert rows[0]["channel"] == "telegram"
    assert rows[0]["recipient"] == "@founder"


@pytest.mark.asyncio
async def test_log_publish_action_skips_failed_action(tmp_path, monkeypatch):
    """A failed Action writes NO audit row — the send did not happen."""
    await _setup_db(tmp_path, monkeypatch)
    from mr_roboto.audit_log import log_publish_action, search_sends
    from mr_roboto.actions import Action

    action = Action(status="failed", error="critic vetoed")
    task = {"mission_id": 8, "payload": {"action": "notify_user", "message": "x"}}
    log_id = await log_publish_action("notify_user", action, task)
    assert log_id is None
    assert await search_sends(mission_id=8) == []


@pytest.mark.asyncio
async def test_log_publish_action_skips_suppressed_inner_result(tmp_path, monkeypatch):
    """email/send_via_provider returns a 'completed' Action whose inner result
    says status='suppressed' — nothing was delivered, so no audit row."""
    await _setup_db(tmp_path, monkeypatch)
    from mr_roboto.audit_log import log_publish_action, search_sends
    from mr_roboto.actions import Action

    action = Action(status="completed", result={"status": "suppressed"})
    task = {
        "mission_id": 9,
        "payload": {"action": "email/send_via_provider", "to": "a@b.com",
                    "body_md": "hi"},
    }
    log_id = await log_publish_action("email/send_via_provider", action, task)
    assert log_id is None
    assert await search_sends(mission_id=9) == []


@pytest.mark.asyncio
async def test_log_publish_action_skips_warmup_quota_exceeded_outreach(
    tmp_path, monkeypatch
):
    """outreach/send returns a 'completed' Action whose inner result says
    status='warmup_quota_exceeded' — the email was held, NOT sent — so no
    audit row. (Before the Gap 1 fix this wrote a false audit row.)"""
    await _setup_db(tmp_path, monkeypatch)
    from mr_roboto.audit_log import log_publish_action, search_sends
    from mr_roboto.actions import Action

    action = Action(
        status="completed", result={"status": "warmup_quota_exceeded"}
    )
    task = {
        "mission_id": 21,
        "payload": {"action": "outreach/send", "target_email": "p@b.com",
                    "body_md": "Hi there"},
    }
    log_id = await log_publish_action("outreach/send", action, task)
    assert log_id is None
    assert await search_sends(mission_id=21) == []


@pytest.mark.asyncio
async def test_log_publish_action_skips_disabled_and_gdpr_outreach(
    tmp_path, monkeypatch
):
    """outreach/send 'disabled' (feature flag off) and 'gdpr_blocked' (no
    opt-in) inner statuses also write NO audit row — nothing was delivered."""
    await _setup_db(tmp_path, monkeypatch)
    from mr_roboto.audit_log import log_publish_action, search_sends
    from mr_roboto.actions import Action

    for inner_status in ("disabled", "gdpr_blocked"):
        action = Action(status="completed", result={"status": inner_status})
        task = {
            "mission_id": 22,
            "payload": {"action": "outreach/send", "target_email": "p@b.com",
                        "body_md": "Hi"},
        }
        log_id = await log_publish_action("outreach/send", action, task)
        assert log_id is None, f"{inner_status} must not log"
    assert await search_sends(mission_id=22) == []


@pytest.mark.asyncio
async def test_log_publish_action_skips_skipped_flag_verb(tmp_path, monkeypatch):
    """eas_submit / fastlane return a 'completed' Action wrapping
    {'skipped': True} when the CLI is absent — nothing was published."""
    await _setup_db(tmp_path, monkeypatch)
    from mr_roboto.audit_log import log_publish_action, search_sends
    from mr_roboto.actions import Action

    action = Action(status="completed", result={"skipped": True})
    task = {
        "mission_id": 23,
        "payload": {"action": "eas_submit", "platform": "ios"},
    }
    log_id = await log_publish_action("eas_submit", action, task)
    assert log_id is None
    assert await search_sends(mission_id=23) == []


@pytest.mark.asyncio
async def test_log_publish_action_logs_sent_email(tmp_path, monkeypatch):
    """email/send_via_provider with inner status='sent' writes a row."""
    await _setup_db(tmp_path, monkeypatch)
    from mr_roboto.audit_log import log_publish_action, search_sends
    from mr_roboto.actions import Action

    action = Action(status="completed", result={"status": "sent"})
    task = {
        "mission_id": 10,
        "payload": {"action": "email/send_via_provider",
                    "to": "customer@example.com", "body_md": "Release notes"},
    }
    log_id = await log_publish_action("email/send_via_provider", action, task)
    assert isinstance(log_id, int) and log_id > 0
    rows = await search_sends(mission_id=10)
    assert len(rows) == 1
    assert rows[0]["channel"] == "email"
    assert rows[0]["recipient"] == "customer@example.com"


@pytest.mark.asyncio
async def test_log_publish_action_ignores_non_publish_verb(tmp_path, monkeypatch):
    """A read-only verb never produces an audit row even if it 'completed'."""
    await _setup_db(tmp_path, monkeypatch)
    from mr_roboto.audit_log import log_publish_action, search_sends
    from mr_roboto.actions import Action

    action = Action(status="completed", result={})
    task = {"mission_id": 11, "payload": {"action": "verify_artifacts"}}
    log_id = await log_publish_action("verify_artifacts", action, task)
    assert log_id is None
    assert await search_sends(mission_id=11) == []


@pytest.mark.asyncio
async def test_run_dispatch_wires_audit_log_for_notify_user(tmp_path, monkeypatch):
    """END-TO-END: mr_roboto.run() on a real external-publish verb lands an
    external_comms_log row. Only the Telegram boundary (notify_user executor)
    is faked. If the run() → log_publish_action wiring is removed, this fails."""
    await _setup_db(tmp_path, monkeypatch)

    # Fake ONLY the Telegram-send boundary; everything else is real.
    import mr_roboto.notify_user as _nu

    async def _fake_notify(task):
        return {"sent": True, "chat_id": task["payload"].get("chat_id")}

    monkeypatch.setattr(_nu, "notify_user", _fake_notify)
    # Disable the critic gate so the test stays at the audit-log boundary.
    monkeypatch.setenv("KUTAI_CRITIC_GATE", "off")

    from mr_roboto import run as mr_run
    from mr_roboto.audit_log import search_sends

    msg = "Build green - deploying."
    task = {
        "mission_id": 55,
        "payload": {
            "action": "notify_user",
            "message": msg,
            "chat_id": 123,
            "critic_gate": False,
        },
    }
    result = await mr_run(task)
    assert result.status == "completed"

    rows = await search_sends(mission_id=55)
    assert len(rows) == 1, "run() must auto-log external-publish verbs (B9)"
    assert rows[0]["channel"] == "telegram"
    # The logged content hash must match the message body.
    assert rows[0]["content_hash"] == hashlib.sha256(
        msg.encode("utf-8")
    ).hexdigest()
