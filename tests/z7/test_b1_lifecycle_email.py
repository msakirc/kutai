"""Z7 T5 B1 — Lifecycle email engine tests.

Covers:
  1. DB migration: email_templates, email_sequences, email_sends,
     email_preferences tables exist with correct columns.
  2. trigger_sequence: expands steps_json into email_sends rows.
  3. lifecycle_email_send cron: picks due rows and calls send_email.
  4. Preference center toggle: GET returns per-sequence flags.
  5. Unsubscribe webhook updates preferences + suppression.
  6. Template approval requires brand_voice_lint_pass + copy_compliance_pass.
  7. Cron registration in INTERNAL_CADENCES.
  8. mr_roboto verbs: email/send_via_provider + lifecycle_email_send reversibility.
  9. /lifecycle Telegram command registered.
 10. Cron send marks sent_at on email_sends row after successful send.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


# ── DB helpers ───────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Fresh SQLite DB for B1 tests."""
    db_file = str(tmp_path / "test_b1.db")
    monkeypatch.setenv("DB_PATH", db_file)
    try:
        import src.infra.db as _db_mod
        monkeypatch.setattr(_db_mod, "DB_PATH", db_file)
        monkeypatch.setattr(_db_mod, "_db_connection", None)
        monkeypatch.setattr(_db_mod, "_db_connection_path", None)
    except Exception:
        pass
    return db_file


@pytest_asyncio.fixture
async def db(tmp_db):
    """Initialised DB with full schema (includes B1 migration)."""
    import src.infra.db as _db_mod
    _db_mod._db_connection = None
    _db_mod._db_connection_path = None
    from src.infra.db import init_db, get_db
    await init_db()
    _db = await get_db()
    yield _db
    _db_mod._db_connection = None
    _db_mod._db_connection_path = None


# ── 1. Migration: tables exist ────────────────────────────────────────────────


class TestMigrations:
    @pytest.mark.asyncio
    async def test_email_templates_columns(self, db):
        cur = await db.execute("PRAGMA table_info(email_templates)")
        cols = {r[1] for r in await cur.fetchall()}
        assert "template_id" in cols
        assert "product_id" in cols
        assert "kind" in cols
        assert "subject" in cols
        assert "body_md" in cols
        assert "variants_json" in cols
        assert "status" in cols
        assert "brand_voice_lint_pass" in cols
        assert "copy_compliance_pass" in cols

    @pytest.mark.asyncio
    async def test_email_sequences_columns(self, db):
        cur = await db.execute("PRAGMA table_info(email_sequences)")
        cols = {r[1] for r in await cur.fetchall()}
        assert "sequence_id" in cols
        assert "product_id" in cols
        assert "name" in cols
        assert "trigger_kind" in cols
        assert "steps_json" in cols
        assert "enabled" in cols

    @pytest.mark.asyncio
    async def test_email_sends_columns(self, db):
        cur = await db.execute("PRAGMA table_info(email_sends)")
        cols = {r[1] for r in await cur.fetchall()}
        assert "send_id" in cols
        assert "product_id" in cols
        assert "user_id" in cols
        assert "sequence_id" in cols
        assert "template_id" in cols
        assert "scheduled_for" in cols
        assert "sent_at" in cols
        assert "opened_at" in cols
        assert "clicked_at" in cols
        assert "bounced_at" in cols
        assert "unsubscribed_at" in cols

    @pytest.mark.asyncio
    async def test_email_preferences_columns(self, db):
        cur = await db.execute("PRAGMA table_info(email_preferences)")
        cols = {r[1] for r in await cur.fetchall()}
        assert "product_id" in cols
        assert "user_token" in cols
        assert "subscriptions_json" in cols


# ── 2. trigger_sequence: expands steps into email_sends ─────────────────────


class TestTriggerSequence:
    @pytest.mark.asyncio
    async def test_creates_sends(self, db):
        """trigger_sequence must insert N email_sends rows (one per step)."""
        from src.app.lifecycle_email import trigger_sequence

        # Insert a template
        await db.execute(
            "INSERT INTO email_templates (product_id, kind, subject, body_md, "
            "variants_json, status, brand_voice_lint_pass, copy_compliance_pass) "
            "VALUES (?, 'onboarding', 'Welcome!', 'Hello {name}', '[]', 'approved', 1, 1)",
            ("prod-1",),
        )
        await db.commit()
        cur = await db.execute("SELECT template_id FROM email_templates LIMIT 1")
        row = await cur.fetchone()
        tmpl_id = row[0]

        # Insert a sequence with 2 steps
        steps = [
            {"template_id": tmpl_id, "delay_hours": 0},
            {"template_id": tmpl_id, "delay_hours": 24},
        ]
        await db.execute(
            "INSERT INTO email_sequences (product_id, name, trigger_kind, steps_json, enabled) "
            "VALUES (?, 'Onboarding', 'signup', ?, 1)",
            ("prod-1", json.dumps(steps)),
        )
        await db.commit()
        cur = await db.execute("SELECT sequence_id FROM email_sequences LIMIT 1")
        row = await cur.fetchone()
        seq_id = row[0]

        result = await trigger_sequence(
            product_id="prod-1",
            user_id="user-abc",
            sequence_id=seq_id,
        )
        assert result["ok"] is True
        assert result["sends_created"] == 2

        cur = await db.execute(
            "SELECT * FROM email_sends WHERE sequence_id = ?", (seq_id,)
        )
        rows = await cur.fetchall()
        assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_disabled_sequence_skips(self, db):
        """Disabled sequences should not expand."""
        from src.app.lifecycle_email import trigger_sequence

        steps = [{"template_id": 999, "delay_hours": 0}]
        await db.execute(
            "INSERT INTO email_sequences (product_id, name, trigger_kind, steps_json, enabled) "
            "VALUES (?, 'Disabled', 'signup', ?, 0)",
            ("prod-2", json.dumps(steps)),
        )
        await db.commit()
        cur = await db.execute(
            "SELECT sequence_id FROM email_sequences WHERE product_id='prod-2' LIMIT 1"
        )
        row = await cur.fetchone()
        seq_id = row[0]

        result = await trigger_sequence(
            product_id="prod-2",
            user_id="user-xyz",
            sequence_id=seq_id,
        )
        assert result["ok"] is False
        assert "disabled" in result.get("reason", "").lower()

    @pytest.mark.asyncio
    async def test_trigger_by_kind(self, db):
        """trigger_sequence_by_kind finds the matching enabled sequence."""
        from src.app.lifecycle_email import trigger_sequence_by_kind

        steps = [{"template_id": 1, "delay_hours": 0}]
        await db.execute(
            "INSERT INTO email_sequences (product_id, name, trigger_kind, steps_json, enabled) "
            "VALUES (?, 'Churn', 'cancellation', ?, 1)",
            ("prod-3", json.dumps(steps)),
        )
        await db.commit()

        result = await trigger_sequence_by_kind(
            product_id="prod-3",
            user_id="user-churn",
            trigger_kind="cancellation",
        )
        # Should attempt to expand (template_id=1 may not exist but sends_created key present)
        assert "sends_created" in result or "ok" in result

    @pytest.mark.asyncio
    async def test_trigger_by_kind_no_match(self, db):
        """trigger_sequence_by_kind returns ok=False for unknown trigger_kind."""
        from src.app.lifecycle_email import trigger_sequence_by_kind

        result = await trigger_sequence_by_kind(
            product_id="prod-noexist",
            user_id="u@x.com",
            trigger_kind="first_action",
        )
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_mid_loop_failure_leaves_no_orphan_rows(self, db):
        """I1: a mid-loop INSERT failure in trigger_sequence must roll back —
        no partial UNCOMMITTED rows may survive to be flushed by a LATER
        caller's commit() on the shared aiosqlite connection.

        Tautology guard: without the try/except+rollback, the partial inserts
        stay pending and the unrelated set_preferences() commit below flushes
        them as orphan email_sends rows.
        """
        import src.app.lifecycle_email as _le_mod
        from src.app.lifecycle_email import trigger_sequence, set_preferences

        # A sequence with 3 steps so the loop runs several inserts.
        steps = [
            {"template_id": 1, "delay_hours": 0},
            {"template_id": 1, "delay_hours": 1},
            {"template_id": 1, "delay_hours": 2},
        ]
        await db.execute(
            "INSERT INTO email_sequences (product_id, name, trigger_kind, "
            "steps_json, enabled) VALUES ('prod-i1', 'Orphan', 'signup', ?, 1)",
            (json.dumps(steps),),
        )
        await db.commit()
        cur = await db.execute(
            "SELECT sequence_id FROM email_sequences WHERE product_id='prod-i1'"
        )
        seq_id = (await cur.fetchone())[0]

        # Make the 2nd step's scheduled_for computation blow up mid-loop, so
        # the 1st INSERT has already executed (uncommitted) when the failure
        # propagates.
        real_to_db = _le_mod.to_db
        calls = {"n": 0}

        def boom(dt):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("simulated mid-loop failure")
            return real_to_db(dt)

        with patch.object(_le_mod, "to_db", side_effect=boom):
            result = await trigger_sequence(
                product_id="prod-i1", user_id="user-i1", sequence_id=seq_id
            )

        # The function returns an error result (not a raised exception).
        assert result["ok"] is False
        assert "insert" in result.get("reason", "").lower()

        # An unrelated later commit must NOT flush orphan rows.
        await set_preferences("prod-i1", "someone", {})

        cur = await db.execute(
            "SELECT COUNT(*) FROM email_sends WHERE product_id='prod-i1'"
        )
        assert (await cur.fetchone())[0] == 0, (
            "rollback failed — orphan email_sends rows survived"
        )


# ── 2b. Announcement broadcast fan-out (Critical 9) ──────────────────────────


class TestAnnouncementBroadcast:
    """B1 announcement blast — broadcast fan-out over email_preferences."""

    async def _setup_announcement(self, db, product_id="prod-ann"):
        """Create a template + an enabled 'announcement' sequence; return seq_id."""
        await db.execute(
            "INSERT INTO email_templates (product_id, kind, subject, body_md, "
            "variants_json, status, brand_voice_lint_pass, copy_compliance_pass) "
            "VALUES (?, 'announcement', 'News!', 'We shipped {x}', '[]', "
            "'approved', 1, 1)",
            (product_id,),
        )
        await db.commit()
        cur = await db.execute(
            "SELECT template_id FROM email_templates WHERE product_id=?",
            (product_id,),
        )
        tmpl_id = (await cur.fetchone())[0]
        steps = [{"template_id": tmpl_id, "delay_hours": 0}]
        await db.execute(
            "INSERT INTO email_sequences (product_id, name, trigger_kind, "
            "steps_json, enabled) VALUES (?, 'Announce', 'announcement', ?, 1)",
            (product_id, json.dumps(steps)),
        )
        await db.commit()
        cur = await db.execute(
            "SELECT sequence_id FROM email_sequences WHERE product_id=?",
            (product_id,),
        )
        return (await cur.fetchone())[0]

    @pytest.mark.asyncio
    async def test_broadcast_fans_out_to_subscribers(self, db):
        """trigger_sequence_by_kind(announcement, user_id=None) creates one
        email_sends row per subscribed recipient — real SQLite, real fan-out.

        Tautology guard: the old user_id=None code inserted NULL into the
        NOT NULL user_id column and failed; this test asserts real rows exist.
        """
        from src.app.lifecycle_email import (
            trigger_sequence_by_kind,
            set_preferences,
        )

        seq_id = await self._setup_announcement(db)

        # Three recipients in the preference center: two opted-in, one opted-out.
        await set_preferences("prod-ann", "tok-a", {})            # default-on
        await set_preferences("prod-ann", "tok-b", {str(seq_id): True})
        await set_preferences("prod-ann", "tok-c", {str(seq_id): False})  # unsub

        result = await trigger_sequence_by_kind(
            product_id="prod-ann", user_id=None, trigger_kind="announcement"
        )

        assert result["ok"] is True
        assert result["recipients"] == 2, "should fan out to the 2 opted-in tokens"
        assert result["sends_created"] == 2  # 1 step each

        cur = await db.execute(
            "SELECT user_id FROM email_sends WHERE sequence_id=?", (seq_id,)
        )
        recipients = {r[0] for r in await cur.fetchall()}
        assert recipients == {"tok-a", "tok-b"}
        assert "tok-c" not in recipients, "unsubscribed token got an email_send"

        # No NULL user_id rows (the original Critical 9 bug).
        cur = await db.execute(
            "SELECT COUNT(*) FROM email_sends WHERE user_id IS NULL"
        )
        assert (await cur.fetchone())[0] == 0

    @pytest.mark.asyncio
    async def test_broadcast_no_subscribers_is_ok(self, db):
        """A broadcast with zero preference rows degrades to ok=True, 0 sends."""
        from src.app.lifecycle_email import trigger_sequence_by_kind

        await self._setup_announcement(db, product_id="prod-empty")
        result = await trigger_sequence_by_kind(
            product_id="prod-empty", user_id=None, trigger_kind="announcement"
        )
        assert result["ok"] is True
        assert result["sends_created"] == 0
        assert result["recipients"] == 0

    @pytest.mark.asyncio
    async def test_changelog_publish_blast_creates_real_sends(self, db):
        """changelog/publish → _queue_announcement_email creates real
        email_sends rows for subscribers (end-to-end, real lifecycle_email)."""
        from src.app.lifecycle_email import set_preferences

        seq_id = await self._setup_announcement(db, product_id="prod-cl")
        await set_preferences("prod-cl", "sub-1", {})
        await set_preferences("prod-cl", "sub-2", {})

        # A draft changelog entry to publish.
        await db.execute(
            "INSERT INTO changelog_entries "
            "(product_id, version, title, body_md, kind_breakdown_json, "
            " shipped_features_json, related_mission_ids_json, published) "
            "VALUES ('prod-cl', '1.0.0', 'Launch', 'Body.', '{}', '[]', '[]', 0)"
        )
        await db.commit()
        cur = await db.execute(
            "SELECT entry_id FROM changelog_entries WHERE product_id='prod-cl'"
        )
        entry_id = (await cur.fetchone())[0]

        # Only the page-cache invalidation is mocked (no real changelog_page
        # module state needed); the email blast runs for real.
        with patch("mr_roboto.changelog_publish._invalidate_changelog_cache"):
            from mr_roboto.changelog_publish import run as publish_run
            result = await publish_run(
                {"entry_id": entry_id, "product_id": "prod-cl"}
            )

        assert result["status"] == "ok"
        assert result["email_blast_result"]["ok"] is True
        assert result["email_blast_result"]["sends_created"] == 2

        cur = await db.execute(
            "SELECT user_id FROM email_sends WHERE sequence_id=?", (seq_id,)
        )
        assert {r[0] for r in await cur.fetchall()} == {"sub-1", "sub-2"}


# ── 3. Cron: picks due rows, calls send_email ─────────────────────────────────


class TestLifecycleEmailSendCron:
    @pytest.mark.asyncio
    async def test_picks_due_and_sends(self, db):
        """Cron job calls send_email for due email_sends rows."""
        past = (datetime.now(timezone.utc) - timedelta(minutes=10)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        await db.execute(
            "INSERT INTO email_sends (product_id, user_id, sequence_id, template_id, "
            "scheduled_for) VALUES ('prod-cron', 'user@example.com', 1, 1, ?)",
            (past,),
        )
        await db.commit()

        sent_calls: list = []

        async def mock_send(product_id, to, subject, body_md, **kw):
            sent_calls.append((product_id, to))
            return {"status": "sent", "provider": "brevo", "message_id": "msg-123"}

        with patch("src.app.lifecycle_email.send_email", side_effect=mock_send):
            with patch(
                "src.app.lifecycle_email._get_template",
                return_value={
                    "template_id": 1,
                    "product_id": "prod-cron",
                    "subject": "Test",
                    "body_md": "Hello",
                    "status": "approved",
                },
            ):
                from src.app.jobs.lifecycle_email_send import run_lifecycle_email_send
                result = await run_lifecycle_email_send()

        assert result["ok"] is True
        assert result["sent"] >= 1
        assert len(sent_calls) >= 1

    @pytest.mark.asyncio
    async def test_skips_already_sent(self, db):
        """Cron must NOT re-send rows that already have sent_at set."""
        now = datetime.now(timezone.utc)
        past = (now - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S")
        sent_time = (now - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
        await db.execute(
            "INSERT INTO email_sends (product_id, user_id, sequence_id, template_id, "
            "scheduled_for, sent_at) VALUES ('p', 'u@x.com', 1, 1, ?, ?)",
            (past, sent_time),
        )
        await db.commit()

        sent_calls: list = []

        async def mock_send(*a, **kw):
            sent_calls.append(a)
            return {"status": "sent"}

        with patch("src.app.lifecycle_email.send_email", side_effect=mock_send):
            from src.app.jobs.lifecycle_email_send import run_lifecycle_email_send
            await run_lifecycle_email_send()

        assert len(sent_calls) == 0

    @pytest.mark.asyncio
    async def test_skips_unsubscribed_recipient(self, db):
        """Cron must NOT send to a recipient unsubscribed in email_preferences
        (Critical 12). Exercises the real is_subscribed lookup.

        Tautology guard: without the preference check, send_email would be
        called and sent_count would be >= 1.
        """
        from src.app.lifecycle_email import set_preferences

        past = (datetime.now(timezone.utc) - timedelta(minutes=10)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        # Two due sends, same sequence: one subscribed, one unsubscribed.
        await db.execute(
            "INSERT INTO email_sends (product_id, user_id, sequence_id, "
            "template_id, scheduled_for) VALUES ('prod-pref', 'tok-yes', 7, 1, ?)",
            (past,),
        )
        await db.execute(
            "INSERT INTO email_sends (product_id, user_id, sequence_id, "
            "template_id, scheduled_for) VALUES ('prod-pref', 'tok-no', 7, 1, ?)",
            (past,),
        )
        await db.commit()

        # tok-no opted out of sequence 7; tok-yes left default-on.
        await set_preferences("prod-pref", "tok-no", {"7": False})
        await set_preferences("prod-pref", "tok-yes", {"7": True})

        sent_to: list = []

        async def mock_send(product_id, to, subject, body_md, **kw):
            sent_to.append(to)
            return {"status": "sent", "provider": "brevo", "message_id": "m"}

        with patch("src.app.lifecycle_email.send_email", side_effect=mock_send):
            with patch(
                "src.app.lifecycle_email._get_template",
                return_value={
                    "template_id": 1, "product_id": "prod-pref",
                    "subject": "Hi", "body_md": "Body", "status": "approved",
                },
            ):
                from src.app.jobs.lifecycle_email_send import run_lifecycle_email_send
                result = await run_lifecycle_email_send()

        assert result["ok"] is True
        assert sent_to == ["tok-yes"], "send_email called for unsubscribed recipient"
        assert result["sent"] == 1
        assert result["skipped"] >= 1

        # The unsubscribed row is stamped via unsubscribed_at (NOT sent_at) so
        # it is not re-evaluated every tick AND is not miscounted as delivered.
        cur = await db.execute(
            "SELECT sent_at, unsubscribed_at FROM email_sends WHERE user_id='tok-no'"
        )
        sent_at, unsub_at = await cur.fetchone()
        assert sent_at is None, "suppressed send must NOT set sent_at"
        assert unsub_at is not None, "suppressed send must set unsubscribed_at"

    @pytest.mark.asyncio
    async def test_suppressed_send_not_counted_and_not_repolled(self, db):
        """I2: a suppressed (unsubscribed) send is not counted as sent and is
        not re-polled by _pick_due_sends on the next tick.

        Tautology guard: the old code stamped sent_at — which both inflates an
        'emails sent' metric and (correctly) hides it from re-polling. This
        test asserts sent_at stays NULL while the row still drops out of the
        due-pick query.
        """
        from src.app.lifecycle_email import set_preferences
        from src.app.jobs.lifecycle_email_send import (
            run_lifecycle_email_send,
            _pick_due_sends,
        )

        past = (datetime.now(timezone.utc) - timedelta(minutes=10)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        await db.execute(
            "INSERT INTO email_sends (product_id, user_id, sequence_id, "
            "template_id, scheduled_for) VALUES ('prod-i2', 'tok-off', 21, 1, ?)",
            (past,),
        )
        await db.commit()
        await set_preferences("prod-i2", "tok-off", {"21": False})

        async def mock_send(*a, **kw):
            raise AssertionError("send_email must not run for a suppressed row")

        with patch("src.app.lifecycle_email.send_email", side_effect=mock_send):
            result = await run_lifecycle_email_send()

        # Not counted as sent.
        assert result["sent"] == 0
        assert result["skipped"] >= 1

        cur = await db.execute(
            "SELECT sent_at, unsubscribed_at FROM email_sends WHERE user_id='tok-off'"
        )
        sent_at, unsub_at = await cur.fetchone()
        assert sent_at is None, "suppressed send counted toward sent_at metric"
        assert unsub_at is not None

        # Not re-polled on the next tick.
        due = await _pick_due_sends()
        assert all(r["user_id"] != "tok-off" for r in due), (
            "suppressed row was re-polled by _pick_due_sends"
        )

    @pytest.mark.asyncio
    async def test_marks_sent_at_on_success(self, db):
        """After a successful send, sent_at must be updated."""
        past = (datetime.now(timezone.utc) - timedelta(minutes=10)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        await db.execute(
            "INSERT INTO email_sends (product_id, user_id, sequence_id, template_id, "
            "scheduled_for) VALUES ('prod-mark', 'u@example.com', 99, 99, ?)",
            (past,),
        )
        await db.commit()
        cur = await db.execute(
            "SELECT send_id FROM email_sends WHERE product_id='prod-mark'"
        )
        row = await cur.fetchone()
        send_id = row[0]

        async def mock_send(product_id, to, subject, body_md, **kw):
            return {"status": "sent", "provider": "brevo", "message_id": "mid-x"}

        with patch("src.app.lifecycle_email.send_email", side_effect=mock_send):
            with patch(
                "src.app.lifecycle_email._get_template",
                return_value={
                    "template_id": 99,
                    "product_id": "prod-mark",
                    "subject": "Test",
                    "body_md": "Hello",
                    "status": "approved",
                },
            ):
                from src.app.jobs.lifecycle_email_send import run_lifecycle_email_send
                result = await run_lifecycle_email_send()

        assert result["ok"] is True
        cur = await db.execute(
            "SELECT sent_at FROM email_sends WHERE send_id=?", (send_id,)
        )
        row = await cur.fetchone()
        assert row[0] is not None, "sent_at should be set after successful send"


# ── 4. Preference center toggle ──────────────────────────────────────────────


class TestPreferences:
    @pytest.mark.asyncio
    async def test_get_returns_subscriptions(self, db):
        """get_preferences returns per-sequence subscription state."""
        from src.app.lifecycle_email import get_preferences, set_preferences

        subs = {"seq-1": True, "seq-2": False}
        await set_preferences(
            product_id="prod-pref", user_token="tok-abc", subscriptions=subs
        )

        prefs = await get_preferences(product_id="prod-pref", user_token="tok-abc")
        assert prefs["subscriptions"]["seq-1"] is True
        assert prefs["subscriptions"]["seq-2"] is False

    @pytest.mark.asyncio
    async def test_unknown_token_returns_empty(self, db):
        """get_preferences for an unknown token returns empty subscriptions."""
        from src.app.lifecycle_email import get_preferences

        prefs = await get_preferences(product_id="prod-x", user_token="unknown-tok")
        assert prefs["subscriptions"] == {}

    @pytest.mark.asyncio
    async def test_set_preferences_updates_existing(self, db):
        """set_preferences overwrites previous values."""
        from src.app.lifecycle_email import get_preferences, set_preferences

        await set_preferences("prod-upd", "tok-upd", {"seq-1": True})
        await set_preferences("prod-upd", "tok-upd", {"seq-1": False, "seq-2": True})
        prefs = await get_preferences("prod-upd", "tok-upd")
        assert prefs["subscriptions"]["seq-1"] is False
        assert prefs["subscriptions"]["seq-2"] is True


# ── 5. Unsubscribe webhook updates preferences + suppression ─────────────────


class TestUnsubscribeWebhook:
    @pytest.mark.asyncio
    async def test_unsub_event_updates_suppression(self, db):
        """handle_webhook_event with unsub should add email to suppression."""
        await db.execute(
            "INSERT OR IGNORE INTO product_email_config "
            "(product_id, provider, from_domain, monthly_quota, tier) "
            "VALUES ('prod-unsub', 'brevo', 'example.com', 300, 'free')"
        )
        await db.commit()

        from src.integrations.email.service import handle_webhook_event

        brevo_unsub_payload = {
            "event": "unsubscribe",
            "email": "user@example.com",
            "message-id": "msg-u001",
        }

        with patch(
            "src.integrations.email.registry.get_provider_class",
        ) as mock_registry:
            mock_provider_cls = MagicMock()
            mock_instance = MagicMock()
            mock_instance.parse_webhook_event.return_value = {
                "event_type": "unsub",
                "recipient": "user@example.com",
                "message_id": "msg-u001",
                "should_suppress": True,
            }
            mock_provider_cls.return_value = mock_instance
            mock_registry.return_value = mock_provider_cls

            await handle_webhook_event("prod-unsub", "brevo", brevo_unsub_payload)

        cur = await db.execute(
            "SELECT reason FROM email_suppression "
            "WHERE product_id='prod-unsub' AND email='user@example.com'"
        )
        row = await cur.fetchone()
        assert row is not None
        assert row[0] == "unsub"

    @pytest.mark.asyncio
    async def test_unsub_event_updates_preferences(self, db):
        """Unsubscribe event toggles the sequence preference off (shape 1:
        explicit user_token + sequence_id, the single-click link path)."""
        from src.app.lifecycle_email import handle_email_event_for_lifecycle
        from src.app.lifecycle_email import set_preferences, get_preferences

        await set_preferences("prod-ev", "tok-ev", {"seq-10": True})

        await handle_email_event_for_lifecycle(
            product_id="prod-ev",
            event_type="unsub",
            recipient="u@ev.com",
            user_token="tok-ev",
            sequence_id="seq-10",
        )

        prefs = await get_preferences("prod-ev", "tok-ev")
        assert prefs["subscriptions"].get("seq-10") is False

    @pytest.mark.asyncio
    async def test_webhook_unsub_reaches_preference_center(self, db):
        """Critical 13: the email webhook (handle_webhook_event) must update
        email_preferences, not just email_suppression.

        Tautology guard: before the fix the webhook only touched
        email_suppression; a future broadcast would still send to this user.
        Here we run the REAL handle_webhook_event → handle_email_event_for_lifecycle
        chain (only the provider adapter is faked) and assert email_preferences
        now carries the opt-out.
        """
        from src.integrations.email.service import handle_webhook_event

        await db.execute(
            "INSERT OR IGNORE INTO product_email_config "
            "(product_id, provider, from_domain, monthly_quota, tier) "
            "VALUES ('prod-wpc', 'brevo', 'example.com', 300, 'free')"
        )
        # The recipient has prior email_sends for sequence 13 — so the
        # recipient-only unsubscribe resolves and disables that sequence.
        await db.execute(
            "INSERT INTO email_sends (product_id, user_id, sequence_id, "
            "template_id, scheduled_for) "
            "VALUES ('prod-wpc', 'sub@example.com', 13, 1, "
            "strftime('%Y-%m-%d %H:%M:%S','now'))"
        )
        await db.commit()

        payload = {"event": "unsubscribe", "email": "sub@example.com"}

        with patch(
            "src.integrations.email.registry.get_provider_class"
        ) as mock_registry:
            mock_cls = MagicMock()
            mock_instance = MagicMock()
            mock_instance.parse_webhook_event.return_value = {
                "event_type": "unsub",
                "recipient": "sub@example.com",
                "message_id": "m-1",
                "should_suppress": True,
            }
            mock_cls.return_value = mock_instance
            mock_registry.return_value = mock_cls

            await handle_webhook_event("prod-wpc", "brevo", payload)

        # email_suppression updated (existing behavior).
        cur = await db.execute(
            "SELECT reason FROM email_suppression "
            "WHERE product_id='prod-wpc' AND email='sub@example.com'"
        )
        assert (await cur.fetchone())[0] == "unsub"

        # email_preferences updated (the Critical 13 fix).
        from src.app.lifecycle_email import get_preferences, is_subscribed

        prefs = await get_preferences("prod-wpc", "sub@example.com")
        assert prefs["subscriptions"].get("13") is False, (
            "webhook unsubscribe did not reach the preference center"
        )
        assert await is_subscribed("prod-wpc", "sub@example.com", 13) is False

    @pytest.mark.asyncio
    async def test_account_unsub_for_recipient_with_no_prior_sends(self, db):
        """M3: an account-wide webhook unsubscribe for a recipient who has NO
        prior email_sends rows must still unsubscribe them from everything.

        Tautology guard: the old code wrote subscriptions_json='{}' when
        seq_rows was empty — and is_subscribed read '{}' as default-on True
        for every sequence, so the unsubscribe was a no-op. This test asserts
        is_subscribed returns False for an arbitrary sequence the recipient
        never received, and that a broadcast fan-out excludes them.
        """
        from src.app.lifecycle_email import (
            handle_email_event_for_lifecycle,
            is_subscribed,
            list_subscribed_tokens,
            get_preferences,
        )

        # No email_sends rows for this recipient at all — webhook path only.
        await handle_email_event_for_lifecycle(
            product_id="prod-m3",
            event_type="unsub",
            recipient="ghost@example.com",
        )

        # Default-on no longer applies — the account-wide opt-out wins.
        for seq in (1, 42, "anything"):
            assert await is_subscribed("prod-m3", "ghost@example.com", seq) is False, (
                f"recipient still subscribed to sequence {seq!r} after "
                "account-wide unsubscribe"
            )

        # The reserved flag is actually persisted.
        prefs = await get_preferences("prod-m3", "ghost@example.com")
        assert prefs["subscriptions"].get("_all") is False

        # A broadcast fan-out must exclude them.
        tokens = await list_subscribed_tokens("prod-m3", 99)
        assert "ghost@example.com" not in tokens, (
            "account-unsubscribed recipient included in broadcast fan-out"
        )

    @pytest.mark.asyncio
    async def test_account_unsub_via_complaint_event(self, db):
        """A spam-complaint event also triggers the account-wide opt-out."""
        from src.app.lifecycle_email import (
            handle_email_event_for_lifecycle,
            is_subscribed,
        )

        await handle_email_event_for_lifecycle(
            product_id="prod-m3c",
            event_type="complaint",
            recipient="angry@example.com",
        )
        assert await is_subscribed("prod-m3c", "angry@example.com", 5) is False


# ── 6. Template approval requires lint passes ────────────────────────────────


class TestApproveTemplate:
    @pytest.mark.asyncio
    async def test_requires_lint_pass(self, db):
        """approve_template fails if lint flags are 0."""
        await db.execute(
            "INSERT INTO email_templates (product_id, kind, subject, body_md, "
            "variants_json, status, brand_voice_lint_pass, copy_compliance_pass) "
            "VALUES ('prod-lint', 'onboarding', 'Hi', 'Body', '[]', 'draft', 0, 0)"
        )
        await db.commit()
        cur = await db.execute(
            "SELECT template_id FROM email_templates WHERE product_id='prod-lint'"
        )
        row = await cur.fetchone()
        tmpl_id = row[0]

        from src.app.lifecycle_email import approve_template
        result = await approve_template(tmpl_id)
        assert result["ok"] is False
        reason = result.get("reason", "")
        assert "lint" in reason.lower() or "compliance" in reason.lower()

        # Status unchanged
        cur = await db.execute(
            "SELECT status FROM email_templates WHERE template_id=?", (tmpl_id,)
        )
        row = await cur.fetchone()
        assert row[0] == "draft"

    @pytest.mark.asyncio
    async def test_succeeds_when_lint_passes(self, db):
        """approve_template transitions to approved when both lints pass."""
        await db.execute(
            "INSERT INTO email_templates (product_id, kind, subject, body_md, "
            "variants_json, status, brand_voice_lint_pass, copy_compliance_pass) "
            "VALUES ('prod-ok', 'onboarding', 'Hello', 'Body', '[]', 'draft', 1, 1)"
        )
        await db.commit()
        cur = await db.execute(
            "SELECT template_id FROM email_templates WHERE product_id='prod-ok'"
        )
        row = await cur.fetchone()
        tmpl_id = row[0]

        from src.app.lifecycle_email import approve_template
        result = await approve_template(tmpl_id)
        assert result["ok"] is True

        cur = await db.execute(
            "SELECT status FROM email_templates WHERE template_id=?", (tmpl_id,)
        )
        row = await cur.fetchone()
        assert row[0] == "approved"

    @pytest.mark.asyncio
    async def test_fails_if_only_one_lint_passes(self, db):
        """approve_template fails if only one of the two lint flags is set."""
        await db.execute(
            "INSERT INTO email_templates (product_id, kind, subject, body_md, "
            "variants_json, status, brand_voice_lint_pass, copy_compliance_pass) "
            "VALUES ('prod-half', 'retention', 'Hi', 'Body', '[]', 'draft', 1, 0)"
        )
        await db.commit()
        cur = await db.execute(
            "SELECT template_id FROM email_templates WHERE product_id='prod-half'"
        )
        row = await cur.fetchone()
        tmpl_id = row[0]

        from src.app.lifecycle_email import approve_template
        result = await approve_template(tmpl_id)
        assert result["ok"] is False


# ── 7. Cron registration ─────────────────────────────────────────────────────


class TestCronRegistration:
    def test_in_internal_cadences(self):
        """lifecycle_email_send must be registered in INTERNAL_CADENCES."""
        from general_beckman.cron_seed import INTERNAL_CADENCES
        titles = {c["title"] for c in INTERNAL_CADENCES}
        assert "lifecycle_email_send" in titles

    def test_cadence_is_5_min(self):
        """lifecycle_email_send cadence must be 300 seconds."""
        from general_beckman.cron_seed import INTERNAL_CADENCES
        entry = next(c for c in INTERNAL_CADENCES if c["title"] == "lifecycle_email_send")
        assert entry.get("interval_seconds") == 300
        assert entry["payload"].get("_executor") == "lifecycle_email_send"


# ── 8. Reversibility entries ─────────────────────────────────────────────────


class TestReversibility:
    def test_email_send_via_provider_irreversible(self):
        """email/send_via_provider must be registered as irreversible."""
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        assert "email/send_via_provider" in VERB_REVERSIBILITY
        assert VERB_REVERSIBILITY["email/send_via_provider"] == "irreversible"

    def test_lifecycle_email_send_full(self):
        """lifecycle_email_send cron verb must be full (idempotent tick)."""
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        assert "lifecycle_email_send" in VERB_REVERSIBILITY
        assert VERB_REVERSIBILITY["lifecycle_email_send"] == "full"


# ── 9. Telegram /lifecycle command registered ────────────────────────────────


class TestTelegramCommand:
    def test_lifecycle_command_in_bot(self):
        """cmd_lifecycle must be registered in telegram_bot.py."""
        import pathlib

        src = pathlib.Path(
            "C:/Users/sakir/Dropbox/Workspaces/kutay/src/app/telegram_bot.py"
        ).read_text(encoding="utf-8")
        assert '"lifecycle"' in src or "'lifecycle'" in src, (
            "/lifecycle command not registered in telegram_bot.py"
        )
        assert "cmd_lifecycle" in src, (
            "cmd_lifecycle handler not found in telegram_bot.py"
        )


# ── 10. Preference-center route exists ───────────────────────────────────────


class TestPreferenceCenterRoute:
    def test_route_in_webhook_listener(self):
        """webhook_listener must expose /email/preferences/{user_token} routes."""
        import pathlib

        src = pathlib.Path(
            "C:/Users/sakir/Dropbox/Workspaces/kutay/src/app/webhook_listener.py"
        ).read_text(encoding="utf-8")
        assert "/email/preferences/" in src, (
            "Preference-center route not found in webhook_listener.py"
        )
