"""Z7 T6 A7 — Cold outreach + deliverability spine + reply-handling tests.

Covers:
  1. Feature flag off → outreach/send is a no-op ("outreach disabled").
  2. Suppression filter blocks a known-bad email (email_suppression table).
  3. Warmup quota blocks over-ramp send (outreach_warmup table).
  4. GDPR jurisdiction blocked without explicit opt-in.
  5. handle_reply classifies positive_interest reply.
  6. handle_reply classifies unsubscribe → adds to email_suppression.
  7. deliverability_check pauses campaign + surfaces founder_action on high bounce rate.
  8. Outreach/draft verb dispatches to beckman.enqueue (LLM-bound).
  9. Outreach/send injects List-Unsubscribe header, postal address, unsubscribe link.
 10. SPF/DKIM/DMARC founder_action surface (DNS check + founder_action emit).
 11. outreach_warmup and outreach_sends tables created by migration.
 12. Reversibility tags registered for new verbs.
"""
from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── DB helpers ──────────────────────────────────────────────────────────────


async def _setup_db(tmp_path, monkeypatch):
    db_path = tmp_path / "z7_outreach.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    monkeypatch.setenv("KUTAY_DEV_ALLOW_INSECURE_VAULT", "1")
    from src.infra import db as db_mod

    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    monkeypatch.setattr(db_mod, "_db_connection", None, raising=False)
    monkeypatch.setattr(db_mod, "_db_connection_path", None, raising=False)
    await db_mod.init_db()
    return db_mod


async def _seed_product_config(db_mod, product_id: str = "prod-outreach-1"):
    db = await db_mod.get_db()
    await db.execute(
        """INSERT OR REPLACE INTO product_email_config
           (product_id, provider, from_domain, api_key_ref, monthly_quota, tier)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (product_id, "brevo", "outreach.example.com", "brevo_key", 10000, "free"),
    )
    await db.commit()


async def _seed_warmup(db_mod, product_id: str, domain: str, day: int, sent_count: int, target_count: int):
    db = await db_mod.get_db()
    await db.execute(
        """INSERT OR REPLACE INTO outreach_warmup
           (product_id, domain, day, sent_count, target_count)
           VALUES (?, ?, ?, ?, ?)""",
        (product_id, domain, day, sent_count, target_count),
    )
    await db.commit()


async def _seed_suppression(db_mod, product_id: str, email: str, reason: str = "bounce"):
    db = await db_mod.get_db()
    await db.execute(
        """INSERT OR IGNORE INTO email_suppression
           (product_id, email, reason)
           VALUES (?, ?, ?)""",
        (product_id, email, reason),
    )
    await db.commit()


async def _seed_outreach_send(db_mod, product_id: str, target_email: str, list_id: str = "list-1"):
    db = await db_mod.get_db()
    cur = await db.execute(
        """INSERT INTO outreach_sends
           (product_id, list_id, target_email, template_id, sent_at)
           VALUES (?, ?, ?, ?, datetime('now'))""",
        (product_id, list_id, target_email, "tmpl-1"),
    )
    await db.commit()
    return cur.lastrowid


# ── 1. Feature flag OFF → outreach/send is a no-op ──────────────────────────


@pytest.mark.asyncio
async def test_send_noop_when_outreach_disabled(tmp_path, monkeypatch):
    """outreach/send returns 'outreach disabled' when OUTREACH_ENABLED != 1."""
    db_mod = await _setup_db(tmp_path, monkeypatch)
    await _seed_product_config(db_mod)

    monkeypatch.setenv("OUTREACH_ENABLED", "0")

    from mr_roboto.outreach_send import run_outreach_send

    result = await run_outreach_send(
        product_id="prod-outreach-1",
        list_id="list-1",
        target_email="prospect@example.com",
        template_id="tmpl-1",
        subject="Hello",
        body_md="Hi there",
        postal_address="123 Main St, City, Country",
        unsubscribe_base_url="https://example.com/unsub",
    )

    assert result["status"] == "disabled"
    assert "outreach" in result.get("reason", "").lower()


# ── 2. Suppression filter blocks known-bad email ─────────────────────────────


@pytest.mark.asyncio
async def test_send_blocked_by_suppression(tmp_path, monkeypatch):
    """outreach/send returns 'suppressed' when email is on email_suppression."""
    db_mod = await _setup_db(tmp_path, monkeypatch)
    await _seed_product_config(db_mod)
    await _seed_suppression(db_mod, "prod-outreach-1", "bounced@badmail.com", "bounce")

    monkeypatch.setenv("OUTREACH_ENABLED", "1")

    from mr_roboto.outreach_send import run_outreach_send

    result = await run_outreach_send(
        product_id="prod-outreach-1",
        list_id="list-1",
        target_email="bounced@badmail.com",
        template_id="tmpl-1",
        subject="Hello",
        body_md="Hi there",
        postal_address="123 Main St",
        unsubscribe_base_url="https://example.com/unsub",
    )

    assert result["status"] == "suppressed"


# ── 3. Warmup quota blocks over-ramp send ────────────────────────────────────


@pytest.mark.asyncio
async def test_send_blocked_by_warmup_quota(tmp_path, monkeypatch):
    """outreach/send returns 'warmup_quota_exceeded' when sent_count >= target."""
    db_mod = await _setup_db(tmp_path, monkeypatch)
    await _seed_product_config(db_mod)
    # day=1, sent_count=50, target_count=50 → quota already hit
    await _seed_warmup(db_mod, "prod-outreach-1", "example.com", 1, 50, 50)

    monkeypatch.setenv("OUTREACH_ENABLED", "1")

    from mr_roboto.outreach_send import run_outreach_send

    result = await run_outreach_send(
        product_id="prod-outreach-1",
        list_id="list-1",
        target_email="new_prospect@example.com",
        template_id="tmpl-1",
        subject="Hello",
        body_md="Hi there",
        postal_address="123 Main St",
        unsubscribe_base_url="https://example.com/unsub",
    )

    assert result["status"] == "warmup_quota_exceeded"


# ── 4. GDPR jurisdiction blocked without opt-in ───────────────────────────────


@pytest.mark.asyncio
async def test_send_blocked_for_gdpr_without_optin(tmp_path, monkeypatch):
    """GDPR-jurisdiction targets blocked unless explicit opt-in consent is present."""
    db_mod = await _setup_db(tmp_path, monkeypatch)
    await _seed_product_config(db_mod)

    monkeypatch.setenv("OUTREACH_ENABLED", "1")

    from mr_roboto.outreach_send import run_outreach_send

    result = await run_outreach_send(
        product_id="prod-outreach-1",
        list_id="list-1",
        target_email="eu_person@example.de",
        template_id="tmpl-1",
        subject="Hello",
        body_md="Hi there",
        postal_address="123 Main St",
        unsubscribe_base_url="https://example.com/unsub",
        jurisdiction="GDPR",
        has_explicit_opt_in=False,
    )

    assert result["status"] == "gdpr_blocked"


# ── 5. handle_reply classifies positive_interest ─────────────────────────────


@pytest.mark.asyncio
async def test_handle_reply_positive_interest(tmp_path, monkeypatch):
    """handle_reply classifies a positive reply and drafts a follow-up."""
    db_mod = await _setup_db(tmp_path, monkeypatch)
    await _seed_product_config(db_mod)
    send_id = await _seed_outreach_send(db_mod, "prod-outreach-1", "interested@example.com")

    monkeypatch.setenv("OUTREACH_ENABLED", "1")

    # Mock the LLM classify call and the beckman enqueue
    mock_classify = AsyncMock(return_value="positive_interest")
    mock_log_interaction = AsyncMock(return_value=42)

    with (
        patch("mr_roboto.outreach_handle_reply._classify_reply", mock_classify),
        patch("mr_roboto.outreach_handle_reply._log_to_crm", mock_log_interaction),
    ):
        from mr_roboto.outreach_handle_reply import run_outreach_handle_reply

        result = await run_outreach_handle_reply(
            product_id="prod-outreach-1",
            send_id=send_id,
            reply_body="I'm very interested! Can we schedule a call?",
            reply_from="interested@example.com",
        )

    assert result["status"] == "ok"
    assert result["classification"] == "positive_interest"


# ── 6. handle_reply unsubscribe → adds to email_suppression ─────────────────


@pytest.mark.asyncio
async def test_handle_reply_unsubscribe_adds_suppression(tmp_path, monkeypatch):
    """handle_reply on unsubscribe-request auto-suppresses the email."""
    db_mod = await _setup_db(tmp_path, monkeypatch)
    await _seed_product_config(db_mod)
    send_id = await _seed_outreach_send(db_mod, "prod-outreach-1", "unsub@example.com")

    monkeypatch.setenv("OUTREACH_ENABLED", "1")

    mock_classify = AsyncMock(return_value="unsubscribe_request")
    mock_log = AsyncMock(return_value=1)

    with (
        patch("mr_roboto.outreach_handle_reply._classify_reply", mock_classify),
        patch("mr_roboto.outreach_handle_reply._log_to_crm", mock_log),
    ):
        from mr_roboto.outreach_handle_reply import run_outreach_handle_reply

        result = await run_outreach_handle_reply(
            product_id="prod-outreach-1",
            send_id=send_id,
            reply_body="Please remove me from your list",
            reply_from="unsub@example.com",
        )

    assert result["status"] == "ok"
    assert result["classification"] == "unsubscribe_request"
    assert result.get("suppressed") is True

    # Verify DB row was inserted
    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT reason FROM email_suppression WHERE product_id=? AND email=?",
        ("prod-outreach-1", "unsub@example.com"),
    )
    row = await cur.fetchone()
    assert row is not None
    assert row[0] == "unsub"


# ── 7. deliverability_check pauses on high bounce rate ───────────────────────


@pytest.mark.asyncio
async def test_deliverability_check_pauses_on_high_bounce_rate(tmp_path, monkeypatch):
    """outreach_deliverability_check returns paused when bounce rate >5%."""
    db_mod = await _setup_db(tmp_path, monkeypatch)
    await _seed_product_config(db_mod)

    # Seed outreach_sends: 100 sends, 10 bounced (10% bounce rate)
    db = await db_mod.get_db()
    product_id = "prod-outreach-1"
    list_id = "list-bounce-test"
    for i in range(100):
        await db.execute(
            """INSERT INTO outreach_sends
               (product_id, list_id, target_email, template_id, sent_at)
               VALUES (?, ?, ?, ?, datetime('now'))""",
            (product_id, list_id, f"user{i}@example.com", "tmpl-1"),
        )
    # Mark 10 as bounced
    for i in range(10):
        await db.execute(
            """UPDATE outreach_sends
               SET bounced_at = datetime('now')
               WHERE target_email = ?""",
            (f"user{i}@example.com",),
        )
    await db.commit()

    mock_emit_fa = AsyncMock(return_value=MagicMock(id=999))

    with patch("mr_roboto.outreach_deliverability_check._emit_founder_action", mock_emit_fa):
        from mr_roboto.outreach_deliverability_check import run_deliverability_check

        result = await run_deliverability_check(
            product_id=product_id,
            list_id=list_id,
        )

    assert result["status"] == "paused"
    assert result["bounce_rate"] > 0.05
    mock_emit_fa.assert_awaited_once()


# ── 8. outreach/draft dispatches via beckman enqueue ─────────────────────────


@pytest.mark.asyncio
async def test_outreach_draft_uses_beckman_enqueue(tmp_path, monkeypatch):
    """outreach/draft dispatches an LLM task via general_beckman.enqueue."""
    db_mod = await _setup_db(tmp_path, monkeypatch)

    mock_enqueue = AsyncMock(return_value={"task_id": 123})

    with patch("mr_roboto.outreach_draft.enqueue", mock_enqueue):
        from mr_roboto.outreach_draft import run_outreach_draft

        result = await run_outreach_draft(
            product_id="prod-outreach-1",
            mission_id=1,
            prospect_data={
                "name": "Alice",
                "company": "Acme Corp",
                "role": "VP Engineering",
            },
            template_id="tmpl-1",
            list_id="list-1",
        )

    assert result["status"] == "enqueued"
    mock_enqueue.assert_awaited_once()


# ── 9. outreach/send injects CAN-SPAM headers ────────────────────────────────


@pytest.mark.asyncio
async def test_send_injects_can_spam_headers(tmp_path, monkeypatch):
    """outreach/send always injects List-Unsubscribe header + postal address."""
    db_mod = await _setup_db(tmp_path, monkeypatch)
    await _seed_product_config(db_mod)

    monkeypatch.setenv("OUTREACH_ENABLED", "1")
    captured_headers = {}
    captured_body = {}

    async def mock_send_email(
        product_id, to, subject, body_md, *, headers=None, idempotency_key=None
    ):
        captured_headers.update(headers or {})
        captured_body["body_md"] = body_md
        return {"status": "sent", "message_id": "msg-123", "provider": "brevo"}

    with patch("src.integrations.email.service.send_email", mock_send_email):
        from mr_roboto.outreach_send import run_outreach_send

        result = await run_outreach_send(
            product_id="prod-outreach-1",
            list_id="list-1",
            target_email="prospect@fresh.com",
            template_id="tmpl-1",
            subject="Hello",
            body_md="Hi {{name}}",
            postal_address="123 Main St, San Francisco, CA 94105, USA",
            unsubscribe_base_url="https://example.com/unsub",
        )

    # List-Unsubscribe header must be present
    assert "List-Unsubscribe" in captured_headers
    assert "unsub" in captured_headers["List-Unsubscribe"].lower() or "example.com" in captured_headers["List-Unsubscribe"]
    # Body must contain postal address
    assert "123 Main St" in captured_body.get("body_md", "")
    # Body must contain unsubscribe link
    assert "unsub" in captured_body.get("body_md", "").lower() or "example.com/unsub" in captured_body.get("body_md", "")


# ── 10. SPF/DKIM/DMARC founder_action surface ─────────────────────────────────


@pytest.mark.asyncio
async def test_spf_dkim_dmarc_check_emits_founder_action_when_missing(tmp_path, monkeypatch):
    """verify_outreach_domain emits founder_action when SPF/DKIM/DMARC is missing."""
    db_mod = await _setup_db(tmp_path, monkeypatch)

    mock_dns_check = AsyncMock(return_value={
        "spf": False,
        "dkim": False,
        "dmarc": False,
    })
    mock_emit_fa = AsyncMock(return_value=MagicMock(id=777))

    with (
        patch("mr_roboto.outreach_domain_verify._check_dns_records", mock_dns_check),
        patch("mr_roboto.outreach_domain_verify._emit_founder_action", mock_emit_fa),
    ):
        from mr_roboto.outreach_domain_verify import run_domain_verify

        result = await run_domain_verify(
            product_id="prod-outreach-1",
            mission_id=1,
            domain="outreach.example.com",
        )

    assert result["status"] in ("incomplete", "founder_action_emitted")
    mock_emit_fa.assert_awaited_once()


# ── 11. DB migrations create outreach_warmup and outreach_sends ───────────────


@pytest.mark.asyncio
async def test_outreach_tables_exist_after_migration(tmp_path, monkeypatch):
    """DB init creates outreach_warmup and outreach_sends tables."""
    db_mod = await _setup_db(tmp_path, monkeypatch)
    db = await db_mod.get_db()

    # Check outreach_warmup
    cur = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='outreach_warmup'"
    )
    row = await cur.fetchone()
    assert row is not None, "outreach_warmup table not created"

    # Check outreach_sends
    cur = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='outreach_sends'"
    )
    row = await cur.fetchone()
    assert row is not None, "outreach_sends table not created"

    # Check columns on outreach_warmup
    cur = await db.execute("PRAGMA table_info(outreach_warmup)")
    cols = {row[1] for row in await cur.fetchall()}
    assert "product_id" in cols
    assert "domain" in cols
    assert "day" in cols
    assert "sent_count" in cols
    assert "target_count" in cols

    # Check columns on outreach_sends
    cur = await db.execute("PRAGMA table_info(outreach_sends)")
    cols = {row[1] for row in await cur.fetchall()}
    assert "send_id" in cols
    assert "product_id" in cols
    assert "list_id" in cols
    assert "target_email" in cols
    assert "template_id" in cols
    assert "sent_at" in cols
    assert "opened_at" in cols
    assert "replied_at" in cols
    assert "bounced_at" in cols


# ── 12. Reversibility tags registered ────────────────────────────────────────


def test_outreach_reversibility_tags_registered():
    """All A7 verbs are registered in VERB_REVERSIBILITY."""
    from mr_roboto.reversibility import VERB_REVERSIBILITY

    # outreach/draft: LLM-bound enqueue → idempotent, full
    assert "outreach/draft" in VERB_REVERSIBILITY
    # outreach/send: sends real email → irreversible
    assert "outreach/send" in VERB_REVERSIBILITY
    assert VERB_REVERSIBILITY["outreach/send"] == "irreversible"
    # outreach/handle_reply: DB writes + optional suppression → full
    assert "outreach/handle_reply" in VERB_REVERSIBILITY
    # outreach_deliverability_check: read-only scan + advisory founder_action
    assert "outreach_deliverability_check" in VERB_REVERSIBILITY


# ── 13. outreach_deliverability_check registered in POST_HOOK_REGISTRY ────────


def test_deliverability_check_in_posthook_registry():
    """outreach_deliverability_check kind is in the post-hook registry."""
    from general_beckman.posthooks import POST_HOOK_REGISTRY

    assert "outreach_deliverability_check" in POST_HOOK_REGISTRY
    spec = POST_HOOK_REGISTRY["outreach_deliverability_check"]
    assert spec.verb == "outreach_deliverability_check"
    assert spec.default_severity == "warning"
