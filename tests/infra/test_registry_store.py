"""Tests for src.infra.registry_store — SQLite provider/model registry."""
from __future__ import annotations

import time

import pytest

from src.infra import registry_store as rs


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path):
    """Each test gets a fresh tmp DB. Schema is auto-created on first
    access via _ensure_schema."""
    db_path = tmp_path / "registry_test.db"
    rs.set_db_path(str(db_path))
    yield
    rs.close()


# ── Model API ────────────────────────────────────────────────────────────


def test_unknown_model_is_not_dead():
    assert rs.is_dead("nobody/at-home") is False


def test_register_then_mark_dead():
    rs.register_model("gemini/foo", "gemini")
    assert rs.is_dead("gemini/foo") is False
    rs.mark_dead("gemini/foo", cause="404_permanent")
    assert rs.is_dead("gemini/foo") is True


def test_mark_dead_without_register():
    """mark_dead must work even if model never registered — caller can hit
    a 404 on a never-pre-registered id."""
    rs.mark_dead("gemini/some-future-model", cause="404_permanent")
    assert rs.is_dead("gemini/some-future-model") is True
    # Provider derived from path prefix.
    cause = rs.get_model_cause("gemini/some-future-model")
    assert cause == "404_permanent"


def test_revive_clears_dead():
    rs.mark_dead("gemini/x", cause="404_permanent")
    assert rs.is_dead("gemini/x") is True
    rs.revive("gemini/x", actor="discovery")
    assert rs.is_dead("gemini/x") is False


def test_revive_unknown_is_noop():
    rs.revive("nobody/new")  # must not raise


def test_revive_already_active_is_noop():
    rs.register_model("gemini/foo", "gemini")
    rs.revive("gemini/foo")
    assert rs.is_dead("gemini/foo") is False


def _freeze_time(monkeypatch, epoch: float) -> None:
    """Helper: patch rs._now_iso + rs.time.time so all time reads in
    registry_store snap to `epoch`. Avoids patching base time.gmtime
    which recurses through helper closures."""
    iso = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(epoch))
    monkeypatch.setattr(rs, "_now_iso", lambda: iso)
    monkeypatch.setattr(rs.time, "time", lambda: epoch)


def test_auto_revive_on_ttl_expiry(monkeypatch):
    """status='dead' rows whose expires_at has passed should auto-revive
    on next is_dead() call."""
    rs.mark_dead("openrouter/foo", cause="404_transient")  # 5min TTL
    assert rs.is_dead("openrouter/foo") is True
    _freeze_time(monkeypatch, time.time() + 600)  # +10min, past 5min TTL
    assert rs.is_dead("openrouter/foo") is False
    # Revive event should appear in audit log.
    events = rs.recent_events("openrouter/foo")
    revive_events = [e for e in events if e["event"] == "revive"]
    assert len(revive_events) >= 1
    assert revive_events[0]["cause"] == "auto_expiry"


def test_auth_cause_has_no_ttl():
    """Auth-cause rows have expires_at=NULL — operator must /revive."""
    rs.mark_dead("openrouter/foo", cause="auth")
    assert rs.is_dead("openrouter/foo") is True
    cause = rs.get_model_cause("openrouter/foo")
    assert cause == "auth"


def test_unknown_cause_treated_as_404_permanent(caplog):
    rs.mark_dead("gemini/x", cause="bogus_cause")
    assert rs.get_model_cause("gemini/x") == "404_permanent"


def test_remarking_refreshes_expiry(monkeypatch):
    """Repeated mark_dead with same cause refreshes marked_at + expires_at —
    sustained failures keep the entry out longer."""
    t0 = time.time()
    rs.mark_dead("openrouter/foo", cause="404_transient")
    # Advance time within original 5min TTL, re-mark.
    _freeze_time(monkeypatch, t0 + 200)
    rs.mark_dead("openrouter/foo", cause="404_transient")
    # Advance past original 5min window from FIRST mark, but within new
    # 5min window measured from the re-mark.
    _freeze_time(monkeypatch, t0 + 400)
    assert rs.is_dead("openrouter/foo") is True


def test_list_dead_returns_only_dead():
    rs.mark_dead("a/x", cause="404_permanent")
    rs.mark_dead("b/y", cause="auth")
    rs.register_model("c/z", "c")  # alive
    dead = rs.list_dead()
    names = {d["litellm_name"] for d in dead}
    assert names == {"a/x", "b/y"}


def test_empty_or_falsy_identifier_is_safe():
    rs.mark_dead("")  # no-op, no row created
    rs.revive("")
    assert rs.is_dead("") is False


# ── Provider API ─────────────────────────────────────────────────────────


def test_unknown_provider_not_dead():
    assert rs.is_provider_dead("openrouter") is False


def test_mark_provider_dead_then_revive():
    rs.mark_provider_dead("openrouter", cause="auth", actor="caller")
    assert rs.is_provider_dead("openrouter") is True
    rs.revive_provider("openrouter", actor="user")
    assert rs.is_provider_dead("openrouter") is False


def test_provider_revive_unknown_is_noop():
    rs.revive_provider("nobody")  # must not raise


def test_register_provider_with_key_hash():
    rs.register_provider("openrouter", key_hash="abc12345")
    assert rs.get_provider_key_hash("openrouter") == "abc12345"
    # Re-register with new key updates the hash (rotation case).
    rs.register_provider("openrouter", key_hash="def67890")
    assert rs.get_provider_key_hash("openrouter") == "def67890"


def test_hash_key_first8_hex():
    h = rs.hash_key("sk-test-1234")
    assert len(h) == 8
    assert all(c in "0123456789abcdef" for c in h)
    # Empty key → empty hash, no crash.
    assert rs.hash_key("") == ""


# ── Audit log ────────────────────────────────────────────────────────────


def test_mark_dead_emits_event():
    rs.mark_dead("gemini/x", cause="404_permanent", actor="caller")
    events = rs.recent_events("gemini/x")
    assert len(events) == 1
    e = events[0]
    assert e["event"] == "mark_dead"
    assert e["cause"] == "404_permanent"
    assert e["actor"] == "caller"
    assert e["scope"] == "model"


def test_revive_emits_event():
    rs.mark_dead("gemini/x", cause="404_permanent")
    rs.revive("gemini/x", actor="discovery")
    events = rs.recent_events("gemini/x")
    revive_events = [e for e in events if e["event"] == "revive"]
    assert len(revive_events) == 1
    assert revive_events[0]["actor"] == "discovery"


def test_recent_events_no_target_returns_all():
    rs.mark_dead("a/x")
    rs.mark_dead("b/y")
    rs.mark_provider_dead("c", cause="auth")
    events = rs.recent_events()
    assert len(events) >= 3


def test_event_payload_json_round_trip():
    rs.mark_dead(
        "gemini/x", cause="404_permanent",
        payload={"http_status": 404, "body_excerpt": "model not found"},
    )
    events = rs.recent_events("gemini/x")
    import json
    payload = json.loads(events[0]["payload_json"])
    assert payload["http_status"] == 404


# ── Schema idempotency ──────────────────────────────────────────────────


def test_repeated_get_conn_idempotent_schema():
    """Multiple connection openings (e.g. set_db_path + new test) must not
    fail on duplicate CREATE."""
    rs.mark_dead("a/x")
    rs.close()
    # Re-open same path
    # _isolated_db fixture already pointed us at this DB; reopen is fine
    assert rs.is_dead("a/x") is True
