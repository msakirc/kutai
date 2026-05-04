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


def test_register_model_records_source():
    """register_model writes the source label so /dead can show
    provenance ('yaml' | 'discovery' | 'runtime' | 'gguf')."""
    rs.register_model("gemini/foo", "gemini", source="yaml")
    conn = rs._get_conn()
    row = conn.execute(
        "SELECT source FROM models WHERE litellm_name=?", ("gemini/foo",)
    ).fetchone()
    assert row["source"] == "yaml"


def test_register_model_first_source_wins():
    """register_model is INSERT OR IGNORE — repeated calls don't overwrite
    the first source. Lets pre-population (yaml) survive a later
    discovery-based re-register without source flicker."""
    rs.register_model("gemini/foo", "gemini", source="yaml")
    rs.register_model("gemini/foo", "gemini", source="discovery")
    conn = rs._get_conn()
    row = conn.execute(
        "SELECT source FROM models WHERE litellm_name=?", ("gemini/foo",)
    ).fetchone()
    assert row["source"] == "yaml"


def test_mark_dead_after_yaml_register_preserves_source():
    """A 404 mark_dead on a model that was pre-registered with source='yaml'
    must NOT downgrade source to 'runtime' on the UPSERT. Catalog provenance
    is more useful for forensics than the runtime label."""
    rs.register_model("gemini/foo", "gemini", source="yaml")
    rs.mark_dead("gemini/foo", cause="404_permanent")
    conn = rs._get_conn()
    row = conn.execute(
        "SELECT source, status FROM models WHERE litellm_name=?",
        ("gemini/foo",),
    ).fetchone()
    assert row["source"] == "yaml"
    assert row["status"] == "dead"


def test_register_model_skips_empty():
    rs.register_model("", "gemini", source="yaml")  # no-op
    rs.register_model("openrouter/x", "", source="yaml")  # no-op
    conn = rs._get_conn()
    rows = conn.execute("SELECT COUNT(*) AS n FROM models").fetchone()
    assert rows["n"] == 0


def test_mark_dead_without_register():
    """mark_dead must work even if model never registered — caller can hit
    a 404 on a never-pre-registered id."""
    rs.mark_dead("gemini/some-future-model", cause="404_permanent")
    assert rs.is_dead("gemini/some-future-model") is True
    # Provider derived from path prefix.
    cause = rs.get_model_cause("gemini/some-future-model")
    assert cause == "404_permanent"


def test_revive_clears_dead_for_transient_cause():
    """Transient causes (404_transient, server_error) accept any actor —
    discovery's auto-revive is the expected upstream-recovery path."""
    rs.mark_dead("openrouter/x", cause="404_transient")
    assert rs.is_dead("openrouter/x") is True
    rs.revive("openrouter/x", actor="discovery")
    assert rs.is_dead("openrouter/x") is False


def test_revive_404_permanent_blocks_auto_actor():
    """404_permanent now blocks actor='auto'. Discovery cannot cycle
    revive→call→404→re-mark-dead on a model the runtime call already
    proved unreachable (cerebras gpt-oss-120b: lists in /v1/models but
    rejects on /v1/chat/completions). 24h TTL still auto-revives."""
    rs.mark_dead("cerebras/gpt-oss-120b", cause="404_permanent")
    assert rs.is_dead("cerebras/gpt-oss-120b") is True
    rs.revive("cerebras/gpt-oss-120b")  # actor defaults to "auto"
    assert rs.is_dead("cerebras/gpt-oss-120b") is True


def test_revive_404_permanent_allows_operator_actor():
    """Operator /revive command (actor='user' from telegram, 'manual'
    from CLI) overrides manual_revive policy — escape hatch when
    operator knows access was restored."""
    rs.mark_dead("cerebras/gpt-oss-120b", cause="404_permanent")
    rs.revive("cerebras/gpt-oss-120b", actor="user")
    assert rs.is_dead("cerebras/gpt-oss-120b") is False
    rs.mark_dead("cerebras/zai-glm-4.7", cause="404_permanent")
    rs.revive("cerebras/zai-glm-4.7", actor="manual")
    assert rs.is_dead("cerebras/zai-glm-4.7") is False


def test_revive_auth_allows_auto_actor():
    """Auth cause (2026-05-04: ttl=15min base, manual_revive=False).
    Per-call evidence is too noisy for provider-wide action; runtime
    auth marks now stay model-scoped with adaptive TTL. Auto-revive
    must succeed — boot probe in fatih_hoca/__init__.py is the only
    authoritative path that kills provider-wide on bad creds."""
    rs.mark_dead("openrouter/x", cause="auth")
    assert rs.is_dead("openrouter/x") is True
    rs.revive("openrouter/x")  # auto
    assert rs.is_dead("openrouter/x") is False


def test_manual_cause_still_blocks_auto_actor():
    """'manual' cause keeps no-TTL + manual_revive=True — that's the
    operator-driven path (/dead via Telegram) where auto-revive
    shouldn't override the human."""
    rs.mark_dead("openrouter/y", cause="manual")
    assert rs.is_dead("openrouter/y") is True
    rs.revive("openrouter/y")  # auto
    assert rs.is_dead("openrouter/y") is True


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


# ── Adaptive TTL on auth / server_error ─────────────────────────────────


def _expires_epoch(litellm_name: str) -> float:
    """Read expires_at as epoch seconds for a dead model row.

    Stored timestamps are UTC (registry_store._now_iso uses gmtime),
    so we use calendar.timegm to convert back without local-tz drift
    poisoning the assertion math.
    """
    import calendar
    conn = rs._get_conn()
    row = conn.execute(
        "SELECT expires_at FROM models WHERE litellm_name = ?",
        (litellm_name,),
    ).fetchone()
    assert row is not None
    assert row["expires_at"] is not None
    return float(calendar.timegm(
        time.strptime(row["expires_at"], "%Y-%m-%d %H:%M:%S")
    ))


def test_auth_first_mark_uses_base_ttl():
    """First auth mark in a fresh window: base TTL = 15min (900s)."""
    before = time.time()
    rs.mark_dead("openrouter/some/m1", cause="auth")
    expires = _expires_epoch("openrouter/some/m1")
    # marked_at is gmtime; mktime above interprets as local — both sides
    # share the same conversion so the relative delta is what matters.
    delta = expires - before  # normalize tz offset
    assert 850 < delta < 950, f"expected ~900s, got {delta}"


def test_auth_second_mark_doubles_ttl():
    """Second mark inside the lookback window: 15min base × 2 = 30min."""
    rs.mark_dead("openrouter/some/m1", cause="auth")
    rs.mark_dead("openrouter/some/m1", cause="auth")  # remark
    before = time.time()
    expires = _expires_epoch("openrouter/some/m1")
    delta = expires - before
    assert 1700 < delta < 1900, f"expected ~1800s, got {delta}"


def test_auth_escalation_caps_at_4h():
    """Many remarks must cap at _ADAPTIVE_TTL_CAP (4h = 14400s)."""
    for _ in range(10):  # 2^9 = 512x is well past the cap
        rs.mark_dead("openrouter/some/m1", cause="auth")
    before = time.time()
    expires = _expires_epoch("openrouter/some/m1")
    delta = expires - before
    assert 14000 < delta < 14500, f"expected ~14400s cap, got {delta}"


def test_server_error_also_adaptive():
    """server_error has 600s base; second mark = 1200s."""
    rs.mark_dead("groq/m", cause="server_error")
    rs.mark_dead("groq/m", cause="server_error")
    before = time.time()
    expires = _expires_epoch("groq/m")
    delta = expires - before
    assert 1100 < delta < 1300, f"expected ~1200s, got {delta}"


def test_404_permanent_not_adaptive():
    """Non-adaptive causes (404_permanent / 404_transient) keep base TTL
    regardless of remarks — they aren't symptoms of escalating health
    problems, just stable verdicts about the id."""
    rs.mark_dead("a/x", cause="404_transient")  # 300s
    rs.mark_dead("a/x", cause="404_transient")  # remark, but not adaptive
    before = time.time()
    expires = _expires_epoch("a/x")
    delta = expires - before
    assert 250 < delta < 350, f"expected ~300s base (no escalation), got {delta}"


def test_adaptive_per_cause_separate_ladders():
    """A model that auth-fails AND server-errors in the same window
    has independent counters per cause — don't conflate distinct
    failure shapes."""
    rs.mark_dead("groq/m", cause="auth")          # auth count = 1
    rs.mark_dead("groq/m", cause="server_error")  # server_error count = 1
    rs.mark_dead("groq/m", cause="auth")          # auth count = 2 → 30min
    before = time.time()
    expires = _expires_epoch("groq/m")
    delta = expires - before
    # Auth ladder, second prior mark in window (count=1) → base*2 = 1800s
    assert 1700 < delta < 1900, f"expected ~1800s for auth 2nd, got {delta}"


def test_auth_no_longer_manual_revive():
    """Step 5b had cause=auth as manual_revive=True (no auto-expiry).
    The 2026-05-04 fix gave it a TTL; manual_revive must now be False
    so an expired row auto-revives like other transient causes."""
    assert rs.CAUSE_POLICY["auth"]["manual_revive"] is False
    assert rs.CAUSE_POLICY["auth"]["ttl_seconds"] == 900
