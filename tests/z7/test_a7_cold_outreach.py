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


# ── 3b. No warmup row → day-1 quota applied (Critical 5 fix) ────────────────


@pytest.mark.asyncio
async def test_send_blocked_when_no_warmup_row_and_day1_quota_exhausted(tmp_path, monkeypatch):
    """A fresh domain with NO warmup row must not bypass the quota gate.

    Critical 5 fix path: before the fix, outreach/send only enforced the quota
    'if warmup_row is not None' — a brand-new domain (zero rows) sent with ZERO
    throttle.  After the fix, the missing row is treated as day-1 (target=50),
    a seed row is inserted, and quota is enforced uniformly.

    This test exercises the no-row path directly:
      1. Start with a genuinely fresh domain (no outreach_warmup row).
      2. First send succeeds — the fix seeds a day-1 row (sent_count=0→1).
      3. Bump sent_count to 50 in DB (simulates exhausting the day-1 budget).
      4. Second send must be blocked with warmup_quota_exceeded.

    Failure under the reverted bug ('if warmup_row is not None'):
      Step 1-2: warmup_row is None → quota gate is SKIPPED entirely; no seeding.
      Step 3: nothing to bump (row was never inserted) → DB update is a no-op.
      Step 4: warmup_row is still None → gate still skipped → send goes through.
      Assert fails: 'sent' != 'warmup_quota_exceeded'.
    """
    db_mod = await _setup_db(tmp_path, monkeypatch)
    await _seed_product_config(db_mod)

    monkeypatch.setenv("OUTREACH_ENABLED", "1")

    from mr_roboto.outreach_send import run_outreach_send

    db = await db_mod.get_db()

    # Precondition: genuinely no warmup row for this domain.
    cur = await db.execute(
        "SELECT COUNT(*) FROM outreach_warmup WHERE product_id=? AND domain=?",
        ("prod-outreach-1", "freshco.io"),
    )
    row = await cur.fetchone()
    assert row[0] == 0, "precondition: no warmup row for freshco.io"

    # Step 1: First send — fix seeds the row; mock email so no real call.
    with patch(
        "src.integrations.email.service.send_email",
        new=AsyncMock(return_value={"status": "sent", "message_id": "msg-seed-1"}),
    ):
        first_result = await run_outreach_send(
            product_id="prod-outreach-1",
            list_id="list-1",
            target_email="cto@freshco.io",
            template_id="tmpl-1",
            subject="Hello",
            body_md="Hi there",
            postal_address="123 Main St",
            unsubscribe_base_url="https://example.com/unsub",
        )

    assert first_result["status"] == "sent", (
        "First send to a fresh domain must succeed (quota not yet exhausted)"
    )

    # Step 2: Verify the fix seeded a day-1 row.
    cur = await db.execute(
        "SELECT sent_count, target_count, day FROM outreach_warmup "
        "WHERE product_id=? AND domain=?",
        ("prod-outreach-1", "freshco.io"),
    )
    warmup = await cur.fetchone()
    assert warmup is not None, "fix must seed a day-1 warmup row on first send to a fresh domain"
    assert warmup[2] == 1, "seeded row must be day 1"
    assert warmup[1] == 50, "day-1 target must be 50"

    # Step 3: Exhaust the day-1 budget by setting sent_count to target_count.
    await db.execute(
        "UPDATE outreach_warmup SET sent_count=target_count "
        "WHERE product_id=? AND domain=? AND day=1",
        ("prod-outreach-1", "freshco.io"),
    )
    await db.commit()

    # Step 4: Second send must be blocked.
    second_result = await run_outreach_send(
        product_id="prod-outreach-1",
        list_id="list-1",
        target_email="cto@freshco.io",
        template_id="tmpl-1",
        subject="Hello again",
        body_md="Hi again",
        postal_address="123 Main St",
        unsubscribe_base_url="https://example.com/unsub",
    )

    assert second_result["status"] == "warmup_quota_exceeded", (
        "After exhausting day-1 budget, send must be blocked. "
        "If 'sent' is returned, the quota gate was bypassed (Critical 5 bug not fixed)."
    )


@pytest.mark.asyncio
async def test_send_seeds_warmup_row_for_fresh_domain(tmp_path, monkeypatch):
    """outreach/send inserts a day-1 warmup row for a domain with no prior record.

    After the fix, a fresh domain gets a seeded row (day=1, target=50) on first
    evaluation so subsequent calls can enforce the quota.
    """
    db_mod = await _setup_db(tmp_path, monkeypatch)
    await _seed_product_config(db_mod)

    monkeypatch.setenv("OUTREACH_ENABLED", "1")

    # Mock the actual email send so we don't need Brevo credentials.
    from unittest.mock import patch, AsyncMock
    with patch(
        "src.integrations.email.service.send_email",
        new=AsyncMock(return_value={"status": "sent", "message_id": "msg-seed-test"}),
    ):
        from mr_roboto.outreach_send import run_outreach_send

        await run_outreach_send(
            product_id="prod-outreach-1",
            list_id="list-1",
            target_email="first@newdomain.io",
            template_id="tmpl-1",
            subject="Hello",
            body_md="Hi there",
            postal_address="123 Main St",
            unsubscribe_base_url="https://example.com/unsub",
        )

    # Verify that a day-1 warmup row was created for the domain.
    db = await db_mod.get_db()
    cur = await db.execute(
        "SELECT day, target_count FROM outreach_warmup "
        "WHERE product_id=? AND domain=? ORDER BY day",
        ("prod-outreach-1", "newdomain.io"),
    )
    rows = await cur.fetchall()
    assert len(rows) >= 1, "day-1 warmup row must be seeded on first send to a fresh domain"
    day1 = rows[0]
    assert day1[0] == 1
    assert day1[1] == 50, "day-1 target must be 50 (most restrictive ramp bucket)"


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


# ── 14. clear_pause un-pauses a campaign (the un-pause path fix) ──────────────


@pytest.mark.asyncio
async def test_clear_pause_unblocks_gate_2b(tmp_path, monkeypatch):
    """Full round-trip: pause → Gate 2b blocks → clear_pause → Gate 2b allows.

    This test directly exercises the defect: before the fix there was no
    clear_pause function, so once outreach_pauses had an un-cleared row the
    campaign was permanently locked out.
    """
    db_mod = await _setup_db(tmp_path, monkeypatch)
    await _seed_product_config(db_mod)

    product_id = "prod-outreach-1"
    list_id = "list-pause-test"

    # ── Step 1: Write an active pause row (simulates deliverability_check) ──
    db = await db_mod.get_db()
    await db.execute(
        "INSERT INTO outreach_pauses (product_id, list_id, reason) "
        "VALUES (?, ?, ?)",
        (product_id, list_id, "bounce rate 12.0% exceeds 5% threshold"),
    )
    await db.commit()

    # ── Step 2: Gate 2b must block while the pause row is un-cleared ────────
    monkeypatch.setenv("OUTREACH_ENABLED", "1")

    from mr_roboto.outreach_send import run_outreach_send

    blocked = await run_outreach_send(
        product_id=product_id,
        list_id=list_id,
        target_email="prospect@example.com",
        template_id="tmpl-1",
        subject="Hello",
        body_md="Hi there",
        postal_address="123 Main St",
        unsubscribe_base_url="https://example.com/unsub",
    )
    assert blocked["status"] == "campaign_paused", (
        f"Gate 2b must block while cleared_at IS NULL; got {blocked['status']!r}"
    )

    # ── Step 3: clear_pause stamps cleared_at ───────────────────────────────
    from mr_roboto.outreach_deliverability_check import clear_pause

    clear_result = await clear_pause(product_id=product_id, list_id=list_id)
    assert clear_result["status"] == "cleared", (
        f"clear_pause must return 'cleared' when an active row exists; got {clear_result!r}"
    )

    # Verify cleared_at is now set in the DB
    cur = await db.execute(
        "SELECT cleared_at FROM outreach_pauses "
        "WHERE product_id=? AND list_id=?",
        (product_id, list_id),
    )
    row = await cur.fetchone()
    assert row is not None
    assert row[0] is not None, "cleared_at must be stamped after clear_pause"

    # ── Step 4: Gate 2b must now allow sending (mock the actual email call) ─
    with patch(
        "src.integrations.email.service.send_email",
        new=AsyncMock(return_value={"status": "sent", "message_id": "msg-resume-1"}),
    ):
        allowed = await run_outreach_send(
            product_id=product_id,
            list_id=list_id,
            target_email="prospect@example.com",
            template_id="tmpl-1",
            subject="Hello",
            body_md="Hi there",
            postal_address="123 Main St",
            unsubscribe_base_url="https://example.com/unsub",
        )

    assert allowed["status"] == "sent", (
        f"After clear_pause, Gate 2b must let sends through; got {allowed['status']!r}"
    )


@pytest.mark.asyncio
async def test_clear_pause_returns_not_paused_when_no_active_row(tmp_path, monkeypatch):
    """clear_pause returns 'not_paused' when there is no un-cleared row."""
    db_mod = await _setup_db(tmp_path, monkeypatch)

    from mr_roboto.outreach_deliverability_check import clear_pause

    result = await clear_pause(product_id="prod-no-pause", list_id="list-x")
    assert result["status"] == "not_paused"


@pytest.mark.asyncio
async def test_clear_pause_idempotent_already_cleared(tmp_path, monkeypatch):
    """clear_pause on an already-cleared row returns 'not_paused' (idempotent)."""
    db_mod = await _setup_db(tmp_path, monkeypatch)

    from dabidabi.times import db_now

    db = await db_mod.get_db()
    await db.execute(
        "INSERT INTO outreach_pauses (product_id, list_id, reason, cleared_at) "
        "VALUES (?, ?, ?, ?)",
        ("prod-already-cleared", "list-y", "old bounce issue", db_now()),
    )
    await db.commit()

    from mr_roboto.outreach_deliverability_check import clear_pause

    result = await clear_pause(product_id="prod-already-cleared", list_id="list-y")
    assert result["status"] == "not_paused"
