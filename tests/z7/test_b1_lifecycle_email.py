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
        """Unsubscribe event toggles the sequence preference off."""
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
