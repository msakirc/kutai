"""Z7 T2A — Email-send shared service tests.

Covers:
  1. provider registry resolves brevo / resend / postmark / ses.
  2. send routes to the right adapter (brevo, resend).
  3. quota guard blocks sends over the monthly cap.
  4. suppression list filters a known-bad address.
  5. webhook event parsing (open/click/bounce/unsub) per provider.
  6. paid-tier stubs raise NotImplementedError.
  7. test-mode redirect (EMAIL_TEST_MODE) sends to founder inbox.
  8. EmailProvider ABC contract enforced.
  9. service.send_email returns the expected dict shape.
 10. suppression webhook inserts a db row.
"""
from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── DB helpers ──────────────────────────────────────────────────────────────


async def _setup_db(tmp_path, monkeypatch):
    db_path = tmp_path / "z7_email.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    monkeypatch.setenv("KUTAY_DEV_ALLOW_INSECURE_VAULT", "1")
    from src.infra import db as db_mod

    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(db_mod, "_db_connection", None, raising=False)
    monkeypatch.setattr(db_mod, "_db_connection_path", None, raising=False)
    await db_mod.init_db()
    return db_mod


async def _seed_product_config(
    db_mod,
    product_id: str = "prod-123",
    provider: str = "brevo",
    monthly_quota: int = 100,
    tier: str = "free",
    api_key_ref: str = "brevo_api_key",
    from_domain: str = "example.com",
):
    db = await db_mod.get_db()
    await db.execute(
        """INSERT OR REPLACE INTO product_email_config
           (product_id, provider, from_domain, api_key_ref, monthly_quota, tier)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (product_id, provider, from_domain, api_key_ref, monthly_quota, tier),
    )
    await db.commit()


# ── 1. Provider registry ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_registry_resolves_brevo():
    from src.integrations.email.registry import get_provider_class

    cls = get_provider_class("brevo")
    assert cls is not None
    assert cls.__name__ == "BrevoProvider"


@pytest.mark.asyncio
async def test_registry_resolves_resend():
    from src.integrations.email.registry import get_provider_class

    cls = get_provider_class("resend")
    assert cls is not None
    assert cls.__name__ == "ResendProvider"


@pytest.mark.asyncio
async def test_registry_resolves_postmark():
    from src.integrations.email.registry import get_provider_class

    cls = get_provider_class("postmark")
    assert cls is not None


@pytest.mark.asyncio
async def test_registry_resolves_ses():
    from src.integrations.email.registry import get_provider_class

    cls = get_provider_class("ses")
    assert cls is not None


@pytest.mark.asyncio
async def test_registry_unknown_raises():
    from src.integrations.email.registry import get_provider_class

    with pytest.raises(KeyError):
        get_provider_class("nonexistent_provider")


# ── 2. send routes to correct adapter ────────────────────────────────────


@pytest.mark.asyncio
async def test_send_routes_to_brevo(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    await _seed_product_config(db_mod, provider="brevo", monthly_quota=300)

    # Patch credential store to return API key
    monkeypatch.setenv("KUTAY_DEV_ALLOW_INSECURE_VAULT", "1")
    from src.security import credential_store as cs

    async def _fake_get_cred(name):
        return {"api_key": "brevo-test-key"}

    monkeypatch.setattr(cs, "get_credential", _fake_get_cred)

    # Patch BrevoProvider.send
    sent_calls = []

    from src.integrations.email.providers import brevo as brevo_mod

    async def _fake_send(self, to, subject, body_md, headers=None, idempotency_key=None):
        sent_calls.append({"to": to, "subject": subject, "provider": "brevo"})
        return {"status": "sent", "provider": "brevo", "message_id": "msg-001"}

    monkeypatch.setattr(brevo_mod.BrevoProvider, "send", _fake_send)

    from src.integrations.email.service import send_email

    result = await send_email(
        product_id="prod-123",
        to="user@test.com",
        subject="Hello",
        body_md="**Hello** world",
    )

    assert result["status"] == "sent"
    assert result["provider"] == "brevo"
    assert len(sent_calls) == 1
    assert sent_calls[0]["to"] == "user@test.com"


@pytest.mark.asyncio
async def test_send_routes_to_resend(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    await _seed_product_config(db_mod, provider="resend", monthly_quota=3000)

    monkeypatch.setenv("KUTAY_DEV_ALLOW_INSECURE_VAULT", "1")
    from src.security import credential_store as cs

    async def _fake_get_cred(name):
        return {"api_key": "resend-test-key"}

    monkeypatch.setattr(cs, "get_credential", _fake_get_cred)

    sent_calls = []

    from src.integrations.email.providers import resend as resend_mod

    async def _fake_send(self, to, subject, body_md, headers=None, idempotency_key=None):
        sent_calls.append({"to": to, "provider": "resend"})
        return {"status": "sent", "provider": "resend", "message_id": "re-001"}

    monkeypatch.setattr(resend_mod.ResendProvider, "send", _fake_send)

    from src.integrations.email import service as svc_mod

    # reload to pick fresh monkeypatches
    import importlib

    importlib.reload(svc_mod)

    result = await svc_mod.send_email(
        product_id="prod-123",
        to="user@test.com",
        subject="Test Subject",
        body_md="plain body",
    )

    assert result["status"] == "sent"
    assert result["provider"] == "resend"


# ── 3. Quota guard ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_quota_guard_blocks_over_cap(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    # Very low quota: 2 emails/month
    await _seed_product_config(db_mod, provider="brevo", monthly_quota=2)

    monkeypatch.setenv("KUTAY_DEV_ALLOW_INSECURE_VAULT", "1")
    from src.security import credential_store as cs

    async def _fake_get_cred(name):
        return {"api_key": "brevo-key"}

    monkeypatch.setattr(cs, "get_credential", _fake_get_cred)

    # Insert 2 sent email records for this month to exhaust quota
    db = await db_mod.get_db()
    await db.execute(
        """INSERT INTO email_events (product_id, event_type, recipient, provider, occurred_at)
           VALUES ('prod-123', 'sent', 'a@x.com', 'brevo', datetime('now'))""",
    )
    await db.execute(
        """INSERT INTO email_events (product_id, event_type, recipient, provider, occurred_at)
           VALUES ('prod-123', 'sent', 'b@x.com', 'brevo', datetime('now'))""",
    )
    await db.commit()

    from src.integrations.email.service import send_email

    result = await send_email(
        product_id="prod-123",
        to="user@test.com",
        subject="Blocked",
        body_md="should not send",
    )

    assert result["status"] == "quota_blocked"
    assert "quota" in result


@pytest.mark.asyncio
async def test_quota_not_blocked_under_cap(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    await _seed_product_config(db_mod, provider="brevo", monthly_quota=300)

    monkeypatch.setenv("KUTAY_DEV_ALLOW_INSECURE_VAULT", "1")
    from src.security import credential_store as cs

    async def _fake_get_cred(name):
        return {"api_key": "brevo-key"}

    monkeypatch.setattr(cs, "get_credential", _fake_get_cred)

    from src.integrations.email.providers import brevo as brevo_mod

    async def _fake_send(self, to, subject, body_md, headers=None, idempotency_key=None):
        return {"status": "sent", "provider": "brevo", "message_id": "msg-ok"}

    monkeypatch.setattr(brevo_mod.BrevoProvider, "send", _fake_send)

    from src.integrations.email.service import send_email

    result = await send_email(
        product_id="prod-123",
        to="user@test.com",
        subject="Under cap",
        body_md="should send",
    )

    assert result["status"] == "sent"


# ── 4. Suppression filter ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_suppression_filters_known_bad_address(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    await _seed_product_config(db_mod, provider="brevo", monthly_quota=300)

    # Insert suppression row
    db = await db_mod.get_db()
    await db.execute(
        """INSERT INTO email_suppression (product_id, email, reason)
           VALUES ('prod-123', 'bounced@example.com', 'bounce')""",
    )
    await db.commit()

    monkeypatch.setenv("KUTAY_DEV_ALLOW_INSECURE_VAULT", "1")
    from src.security import credential_store as cs

    async def _fake_get_cred(name):
        return {"api_key": "brevo-key"}

    monkeypatch.setattr(cs, "get_credential", _fake_get_cred)

    from src.integrations.email.service import send_email

    result = await send_email(
        product_id="prod-123",
        to="bounced@example.com",
        subject="Test",
        body_md="blocked by suppression",
    )

    assert result["status"] == "suppressed"
    assert result.get("reason") == "bounce"


@pytest.mark.asyncio
async def test_suppression_allows_clean_address(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    await _seed_product_config(db_mod, provider="brevo", monthly_quota=300)

    monkeypatch.setenv("KUTAY_DEV_ALLOW_INSECURE_VAULT", "1")
    from src.security import credential_store as cs

    async def _fake_get_cred(name):
        return {"api_key": "brevo-key"}

    monkeypatch.setattr(cs, "get_credential", _fake_get_cred)

    from src.integrations.email.providers import brevo as brevo_mod

    async def _fake_send(self, to, subject, body_md, headers=None, idempotency_key=None):
        return {"status": "sent", "provider": "brevo", "message_id": "msg-ok"}

    monkeypatch.setattr(brevo_mod.BrevoProvider, "send", _fake_send)

    from src.integrations.email.service import send_email

    result = await send_email(
        product_id="prod-123",
        to="clean@example.com",
        subject="Test",
        body_md="should pass",
    )

    assert result["status"] == "sent"


# ── 5. Webhook event parsing ──────────────────────────────────────────────


def test_brevo_parse_open_event():
    from src.integrations.email.providers.brevo import BrevoProvider

    payload = {
        "event": "opened",
        "email": "user@example.com",
        "messageId": "<abc123@brevo.com>",
        "id": 12345,
    }
    provider = BrevoProvider(api_key="k", from_domain="d.com")
    event = provider.parse_webhook_event(payload)
    assert event["event_type"] == "open"
    assert event["recipient"] == "user@example.com"


def test_brevo_parse_bounce_event():
    from src.integrations.email.providers.brevo import BrevoProvider

    payload = {
        "event": "hard_bounce",
        "email": "bad@example.com",
        "messageId": "<xyz@brevo.com>",
        "id": 99,
    }
    provider = BrevoProvider(api_key="k", from_domain="d.com")
    event = provider.parse_webhook_event(payload)
    assert event["event_type"] == "bounce"
    assert event["should_suppress"] is True


def test_brevo_parse_unsub_event():
    from src.integrations.email.providers.brevo import BrevoProvider

    payload = {"event": "unsubscribed", "email": "user@example.com", "id": 1}
    provider = BrevoProvider(api_key="k", from_domain="d.com")
    event = provider.parse_webhook_event(payload)
    assert event["event_type"] == "unsub"
    assert event["should_suppress"] is True


def test_resend_parse_click_event():
    from src.integrations.email.providers.resend import ResendProvider

    payload = {
        "type": "email.clicked",
        "data": {
            "email_id": "re-abc",
            "to": ["clicker@example.com"],
            "click": {"link": "https://example.com"},
        },
    }
    provider = ResendProvider(api_key="k", from_domain="d.com")
    event = provider.parse_webhook_event(payload)
    assert event["event_type"] == "click"
    assert event["recipient"] == "clicker@example.com"


def test_resend_parse_bounce_event():
    from src.integrations.email.providers.resend import ResendProvider

    payload = {
        "type": "email.bounced",
        "data": {
            "email_id": "re-xyz",
            "to": ["bad@example.com"],
        },
    }
    provider = ResendProvider(api_key="k", from_domain="d.com")
    event = provider.parse_webhook_event(payload)
    assert event["event_type"] == "bounce"
    assert event["should_suppress"] is True


# ── 6. Paid-tier stubs raise NotImplementedError ─────────────────────────


@pytest.mark.asyncio
async def test_postmark_send_raises_not_implemented():
    from src.integrations.email.providers.postmark import PostmarkProvider

    provider = PostmarkProvider(api_key="k", from_domain="d.com")
    with pytest.raises(NotImplementedError):
        await provider.send(
            to="x@y.com", subject="test", body_md="body"
        )


@pytest.mark.asyncio
async def test_ses_send_raises_not_implemented():
    from src.integrations.email.providers.ses import SESProvider

    provider = SESProvider(api_key="k", from_domain="d.com")
    with pytest.raises(NotImplementedError):
        await provider.send(
            to="x@y.com", subject="test", body_md="body"
        )


# ── 7. Test mode redirect ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_email_test_mode_redirects_to_founder(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    await _seed_product_config(db_mod, provider="brevo", monthly_quota=300)

    monkeypatch.setenv("EMAIL_TEST_MODE", "1")
    monkeypatch.setenv("FOUNDER_EMAIL", "founder@example.com")
    monkeypatch.setenv("KUTAY_DEV_ALLOW_INSECURE_VAULT", "1")

    from src.security import credential_store as cs

    async def _fake_get_cred(name):
        return {"api_key": "brevo-key"}

    monkeypatch.setattr(cs, "get_credential", _fake_get_cred)

    sent_to = []

    from src.integrations.email.providers import brevo as brevo_mod

    async def _fake_send(self, to, subject, body_md, headers=None, idempotency_key=None):
        sent_to.append(to)
        return {"status": "sent", "provider": "brevo", "message_id": "test-001"}

    monkeypatch.setattr(brevo_mod.BrevoProvider, "send", _fake_send)

    from src.integrations.email import service as svc_mod
    import importlib
    importlib.reload(svc_mod)

    result = await svc_mod.send_email(
        product_id="prod-123",
        to="real-user@example.com",
        subject="Product Update",
        body_md="hello",
    )

    assert result["status"] == "sent"
    # The actual send went to the founder, not the real user
    assert len(sent_to) == 1
    assert sent_to[0] == "founder@example.com"


# ── 8. EmailProvider ABC contract ─────────────────────────────────────────


def test_email_provider_abc_cannot_be_instantiated():
    from src.integrations.email.base import EmailProvider
    import inspect

    # ABC with abstract methods cannot be instantiated
    assert inspect.isabstract(EmailProvider)


# ── 9. send_email return shape ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_email_returns_complete_shape(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    await _seed_product_config(db_mod, provider="brevo", monthly_quota=300)

    monkeypatch.setenv("KUTAY_DEV_ALLOW_INSECURE_VAULT", "1")
    from src.security import credential_store as cs

    async def _fake_get_cred(name):
        return {"api_key": "brevo-key"}

    monkeypatch.setattr(cs, "get_credential", _fake_get_cred)

    from src.integrations.email.providers import brevo as brevo_mod

    async def _fake_send(self, to, subject, body_md, headers=None, idempotency_key=None):
        return {"status": "sent", "provider": "brevo", "message_id": "shp-001"}

    monkeypatch.setattr(brevo_mod.BrevoProvider, "send", _fake_send)

    from src.integrations.email.service import send_email

    result = await send_email(
        product_id="prod-123",
        to="user@test.com",
        subject="Shape test",
        body_md="body text",
        idempotency_key="idem-abc",
    )

    # Required keys in every non-error result
    assert "status" in result
    assert result["status"] in ("sent", "quota_blocked", "suppressed", "error")


# ── 10. Suppression webhook inserts a DB row ──────────────────────────────


@pytest.mark.asyncio
async def test_webhook_bounce_adds_suppression_row(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    await _seed_product_config(db_mod, provider="brevo", monthly_quota=300)

    from src.integrations.email.service import handle_webhook_event

    await handle_webhook_event(
        product_id="prod-123",
        provider="brevo",
        raw_payload={
            "event": "hard_bounce",
            "email": "bounced@example.com",
            "messageId": "<b1@brevo.com>",
            "id": 777,
        },
    )

    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT email, reason FROM email_suppression "
        "WHERE product_id = 'prod-123' AND email = 'bounced@example.com'"
    )
    row = await cur.fetchone()
    assert row is not None
    assert row[0] == "bounced@example.com"
    assert row[1] == "bounce"


@pytest.mark.asyncio
async def test_webhook_unsub_adds_suppression_row(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    await _seed_product_config(db_mod, provider="brevo", monthly_quota=300)

    from src.integrations.email.service import handle_webhook_event

    await handle_webhook_event(
        product_id="prod-123",
        provider="brevo",
        raw_payload={
            "event": "unsubscribed",
            "email": "unsub@example.com",
            "id": 888,
        },
    )

    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT email, reason FROM email_suppression "
        "WHERE product_id = 'prod-123' AND email = 'unsub@example.com'"
    )
    row = await cur.fetchone()
    assert row is not None
    assert row[1] == "unsub"


@pytest.mark.asyncio
async def test_webhook_open_does_not_suppress(tmp_path, monkeypatch):
    db_mod = await _setup_db(tmp_path, monkeypatch)
    await _seed_product_config(db_mod, provider="brevo", monthly_quota=300)

    from src.integrations.email.service import handle_webhook_event

    await handle_webhook_event(
        product_id="prod-123",
        provider="brevo",
        raw_payload={
            "event": "opened",
            "email": "open@example.com",
            "id": 999,
        },
    )

    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT email FROM email_suppression "
        "WHERE product_id = 'prod-123' AND email = 'open@example.com'"
    )
    row = await cur.fetchone()
    # open events do NOT suppress
    assert row is None


# ── Config not found ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_email_no_config_returns_error(tmp_path, monkeypatch):
    await _setup_db(tmp_path, monkeypatch)
    # No product_email_config row inserted

    from src.integrations.email.service import send_email

    result = await send_email(
        product_id="nonexistent-product",
        to="user@test.com",
        subject="Test",
        body_md="body",
    )

    assert result["status"] == "error"
    assert "config" in result.get("reason", "").lower() or "not found" in result.get("reason", "").lower()


# ── Z7 fix-pass: CRLF header-injection sanitization ───────────────────────────


@pytest.mark.asyncio
async def test_send_email_strips_crlf_from_subject_and_headers(tmp_path, monkeypatch):
    """send_email must strip \\r and \\n from subject + header names/values
    so a caller-supplied value cannot inject extra SMTP headers."""
    db_mod = await _setup_db(tmp_path, monkeypatch)
    await _seed_product_config(db_mod, provider="brevo", monthly_quota=300)

    monkeypatch.setenv("KUTAY_DEV_ALLOW_INSECURE_VAULT", "1")
    from src.security import credential_store as cs

    async def _fake_get_cred(name):
        return {"api_key": "brevo-test-key"}

    monkeypatch.setattr(cs, "get_credential", _fake_get_cred)

    captured = {}

    from src.integrations.email.providers import brevo as brevo_mod

    async def _fake_send(self, to, subject, body_md, headers=None, idempotency_key=None):
        captured["subject"] = subject
        captured["headers"] = headers
        return {"status": "sent", "provider": "brevo", "message_id": "msg-crlf"}

    monkeypatch.setattr(brevo_mod.BrevoProvider, "send", _fake_send)

    from src.integrations.email.service import send_email

    result = await send_email(
        product_id="prod-123",
        to="user@test.com",
        subject="Hello\r\nBcc: attacker@evil.com",
        body_md="body",
        headers={
            "X-Custom\r\nInjected": "value\nX-Evil: yes",
        },
    )

    assert result["status"] == "sent"
    # Subject must have no CR/LF left.
    assert "\r" not in captured["subject"]
    assert "\n" not in captured["subject"]
    assert captured["subject"] == "HelloBcc: attacker@evil.com"
    # Header name + value must both be sanitized.
    assert captured["headers"] is not None
    for name, value in captured["headers"].items():
        assert "\r" not in name and "\n" not in name
        assert "\r" not in value and "\n" not in value


def test_strip_crlf_helper():
    """The _strip_crlf helper removes both CR and LF."""
    from src.integrations.email.service import _strip_crlf

    assert _strip_crlf("a\r\nb") == "ab"
    assert _strip_crlf("plain") == "plain"
    assert _strip_crlf("\n\r\n") == ""


# ── Z7 fix-pass: Brevo htmlContent HTML escaping ──────────────────────────────


@pytest.mark.asyncio
async def test_brevo_escapes_html_in_body(monkeypatch):
    """BrevoProvider.send must HTML-escape body_md before interpolating it
    into the <pre>...</pre> htmlContent so markup cannot be injected."""
    from src.integrations.email.providers import brevo as brevo_mod

    captured = {}

    class _FakeResp:
        status = 201

        async def json(self, content_type=None):
            return {"messageId": "msg-html"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def post(self, url, headers=None, data=None, timeout=None):
            captured["data"] = data
            return _FakeResp()

    monkeypatch.setattr(brevo_mod.aiohttp, "ClientSession", _FakeSession)

    provider = brevo_mod.BrevoProvider(api_key="k", from_domain="example.com")
    result = await provider.send(
        to="user@test.com",
        subject="Subj",
        body_md="<script>alert(1)</script> & friends",
    )

    assert result["status"] == "sent"
    payload = json.loads(captured["data"])
    # The raw markup must not survive into htmlContent.
    assert "<script>" not in payload["htmlContent"]
    assert "&lt;script&gt;" in payload["htmlContent"]
    assert "&amp;" in payload["htmlContent"]
    # textContent stays raw (not HTML).
    assert payload["textContent"] == "<script>alert(1)</script> & friends"
