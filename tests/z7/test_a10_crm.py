"""Z7 T4 A10 — CRM-as-log + A10.r1 consent ledger tests.

Covers:
  1. DB migrations: relationships, interactions, consent_records tables exist.
  2. add_contact: create and retrieve a contact.
  3. list_contacts: filter by category, returns last_interaction.
  4. log_interaction: write with relative follow-up parsing (2w, 3d, 1m).
  5. follow_ups query: returns only interactions where follow_up_at <= now+7d.
  6. consent grant / revoke / expiry via has_consent.
  7. follow_up_reminder digest text generation.
  8. Reversibility entries registered for all CRM mr_roboto verbs.
  9. Telegram command handlers registered for all 5 CRM commands.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio


# ── DB helpers ───────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Fresh SQLite DB for A10 tests."""
    db_file = str(tmp_path / "test_a10.db")
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
    """Initialised DB with full schema (includes A10 migrations)."""
    import src.infra.db as _db_mod
    _db_mod._db_connection = None
    _db_mod._db_connection_path = None
    from src.infra.db import init_db, get_db
    await init_db()
    _db = await get_db()
    yield _db
    _db_mod._db_connection = None
    _db_mod._db_connection_path = None


# ===========================================================================
# 1. DB migrations
# ===========================================================================


class TestMigrations:
    @pytest.mark.asyncio
    async def test_relationships_table_exists(self, db):
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='relationships'"
        ) as cur:
            row = await cur.fetchone()
        assert row is not None, "relationships table should exist"

    @pytest.mark.asyncio
    async def test_interactions_table_exists(self, db):
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='interactions'"
        ) as cur:
            row = await cur.fetchone()
        assert row is not None, "interactions table should exist"

    @pytest.mark.asyncio
    async def test_consent_records_table_exists(self, db):
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='consent_records'"
        ) as cur:
            row = await cur.fetchone()
        assert row is not None, "consent_records table should exist"

    @pytest.mark.asyncio
    async def test_relationships_has_product_id(self, db):
        async with db.execute("PRAGMA table_info(relationships)") as cur:
            cols = {row[1] for row in await cur.fetchall()}
        assert "product_id" in cols
        assert "handle" in cols
        assert "category" in cols
        assert "email" in cols

    @pytest.mark.asyncio
    async def test_interactions_has_product_id(self, db):
        async with db.execute("PRAGMA table_info(interactions)") as cur:
            cols = {row[1] for row in await cur.fetchall()}
        assert "product_id" in cols
        assert "contact_id" in cols
        assert "kind" in cols
        assert "follow_up_at" in cols

    @pytest.mark.asyncio
    async def test_consent_records_has_product_id(self, db):
        async with db.execute("PRAGMA table_info(consent_records)") as cur:
            cols = {row[1] for row in await cur.fetchall()}
        assert "product_id" in cols
        assert "contact_id" in cols
        assert "purpose" in cols
        assert "revoked_at" in cols


# ===========================================================================
# 2. add_contact
# ===========================================================================


class TestAddContact:
    @pytest.mark.asyncio
    async def test_add_contact_roundtrip(self, db, monkeypatch):
        from src.app.crm import add_contact, get_contact_by_handle
        contact_id = await add_contact(
            product_id="prod1",
            handle="@alice",
            display_name="Alice Smith",
            category="customer",
        )
        assert isinstance(contact_id, int) and contact_id > 0
        contact = await get_contact_by_handle(product_id="prod1", handle="@alice")
        assert contact is not None
        assert contact["display_name"] == "Alice Smith"
        assert contact["category"] == "customer"

    @pytest.mark.asyncio
    async def test_add_contact_with_email(self, db, monkeypatch):
        from src.app.crm import add_contact, get_contact_by_handle
        await add_contact(
            product_id="prod1",
            handle="@bob",
            display_name="Bob Jones",
            category="investor",
            email="bob@example.com",
        )
        contact = await get_contact_by_handle(product_id="prod1", handle="@bob")
        assert contact["email"] == "bob@example.com"

    @pytest.mark.asyncio
    async def test_add_contact_duplicate_handle_updates(self, db, monkeypatch):
        """Adding the same handle twice should upsert (not raise)."""
        from src.app.crm import add_contact, get_contact_by_handle
        await add_contact(
            product_id="prod1",
            handle="@carol",
            display_name="Carol",
            category="partner",
        )
        # Second call with updated display_name
        await add_contact(
            product_id="prod1",
            handle="@carol",
            display_name="Carol Updated",
            category="partner",
        )
        contact = await get_contact_by_handle(product_id="prod1", handle="@carol")
        assert contact["display_name"] == "Carol Updated"


# ===========================================================================
# 3. list_contacts
# ===========================================================================


class TestListContacts:
    @pytest.mark.asyncio
    async def test_list_all_contacts(self, db, monkeypatch):
        from src.app.crm import add_contact, list_contacts
        await add_contact("prod1", "@x1", "X1", "customer")
        await add_contact("prod1", "@x2", "X2", "investor")
        contacts = await list_contacts(product_id="prod1")
        handles = {c["handle"] for c in contacts}
        assert "@x1" in handles
        assert "@x2" in handles

    @pytest.mark.asyncio
    async def test_list_contacts_filtered_by_category(self, db, monkeypatch):
        from src.app.crm import add_contact, list_contacts
        await add_contact("prod2", "@cust1", "C1", "customer")
        await add_contact("prod2", "@inv1", "I1", "investor")
        customers = await list_contacts(product_id="prod2", category="customer")
        assert all(c["category"] == "customer" for c in customers)
        handles = {c["handle"] for c in customers}
        assert "@cust1" in handles
        assert "@inv1" not in handles

    @pytest.mark.asyncio
    async def test_list_contacts_includes_last_interaction(self, db, monkeypatch):
        from src.app.crm import add_contact, list_contacts, log_interaction
        cid = await add_contact("prod3", "@user1", "User1", "customer")
        await log_interaction(
            product_id="prod3",
            contact_id=cid,
            kind="call",
            summary="First call",
        )
        contacts = await list_contacts(product_id="prod3")
        user = next(c for c in contacts if c["handle"] == "@user1")
        # last_interaction should be populated
        assert user.get("last_interaction") is not None


# ===========================================================================
# 4. log_interaction + relative follow-up parsing
# ===========================================================================


class TestLogInteraction:
    @pytest.mark.asyncio
    async def test_log_basic_interaction(self, db, monkeypatch):
        from src.app.crm import add_contact, log_interaction, list_interactions
        cid = await add_contact("prod4", "@dave", "Dave", "partner")
        iid = await log_interaction(
            product_id="prod4",
            contact_id=cid,
            kind="meeting",
            summary="Discussed partnership",
        )
        assert isinstance(iid, int) and iid > 0
        interactions = await list_interactions(product_id="prod4", contact_id=cid)
        assert len(interactions) >= 1
        assert interactions[0]["summary"] == "Discussed partnership"

    @pytest.mark.asyncio
    async def test_log_interaction_with_relative_followup_2w(self, db, monkeypatch):
        """follow_up='2w' should set follow_up_at ~ 14 days from now."""
        from src.app.crm import add_contact, log_interaction, list_interactions
        cid = await add_contact("prod5", "@eve", "Eve", "customer")
        await log_interaction(
            product_id="prod5",
            contact_id=cid,
            kind="email",
            summary="Follow up in 2 weeks",
            follow_up="2w",
        )
        interactions = await list_interactions(product_id="prod5", contact_id=cid)
        fu = interactions[0]["follow_up_at"]
        assert fu is not None
        # Should be roughly 14 days from now (within 1 day tolerance)
        fu_dt = datetime.fromisoformat(fu)
        if fu_dt.tzinfo is None:
            fu_dt = fu_dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = fu_dt - now
        assert 13 <= delta.days <= 15, f"Expected ~14 days, got {delta.days}"

    @pytest.mark.asyncio
    async def test_log_interaction_with_relative_followup_3d(self, db, monkeypatch):
        """follow_up='3d' should set follow_up_at ~ 3 days from now."""
        from src.app.crm import add_contact, log_interaction, list_interactions
        cid = await add_contact("prod6", "@frank", "Frank", "journalist")
        await log_interaction(
            product_id="prod6",
            contact_id=cid,
            kind="message",
            summary="Short follow-up",
            follow_up="3d",
        )
        interactions = await list_interactions(product_id="prod6", contact_id=cid)
        fu = interactions[0]["follow_up_at"]
        fu_dt = datetime.fromisoformat(fu)
        if fu_dt.tzinfo is None:
            fu_dt = fu_dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = fu_dt - now
        assert 2 <= delta.days <= 4, f"Expected ~3 days, got {delta.days}"

    @pytest.mark.asyncio
    async def test_log_interaction_with_relative_followup_1m(self, db, monkeypatch):
        """follow_up='1m' should set follow_up_at ~ 30 days from now."""
        from src.app.crm import add_contact, log_interaction, list_interactions
        cid = await add_contact("prod7", "@gina", "Gina", "advisor")
        await log_interaction(
            product_id="prod7",
            contact_id=cid,
            kind="call",
            summary="Monthly advisor check-in",
            follow_up="1m",
        )
        interactions = await list_interactions(product_id="prod7", contact_id=cid)
        fu = interactions[0]["follow_up_at"]
        fu_dt = datetime.fromisoformat(fu)
        if fu_dt.tzinfo is None:
            fu_dt = fu_dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = fu_dt - now
        assert 28 <= delta.days <= 32, f"Expected ~30 days, got {delta.days}"

    @pytest.mark.asyncio
    async def test_log_interaction_no_followup(self, db, monkeypatch):
        from src.app.crm import add_contact, log_interaction, list_interactions
        cid = await add_contact("prod8", "@hank", "Hank", "vendor")
        await log_interaction(
            product_id="prod8",
            contact_id=cid,
            kind="other",
            summary="No follow-up needed",
        )
        interactions = await list_interactions(product_id="prod8", contact_id=cid)
        assert interactions[0]["follow_up_at"] is None


# ===========================================================================
# 5. follow_ups query window (today + 7 days)
# ===========================================================================


class TestFollowUps:
    @pytest.mark.asyncio
    async def test_follow_ups_within_7_days(self, db, monkeypatch):
        """get_pending_follow_ups returns interactions due within 7 days."""
        from src.app.crm import add_contact, log_interaction, get_pending_follow_ups
        cid = await add_contact("prod9", "@ian", "Ian", "customer")
        # Due in 3 days — should appear
        await log_interaction(
            product_id="prod9",
            contact_id=cid,
            kind="email",
            summary="Follow up soon",
            follow_up="3d",
        )
        results = await get_pending_follow_ups(product_id="prod9", within_days=7)
        assert len(results) >= 1
        summaries = [r["summary"] for r in results]
        assert "Follow up soon" in summaries

    @pytest.mark.asyncio
    async def test_follow_ups_excludes_far_future(self, db, monkeypatch):
        """Interactions due in 30 days should NOT appear in 7-day window."""
        from src.app.crm import add_contact, log_interaction, get_pending_follow_ups
        cid = await add_contact("prod10", "@julia", "Julia", "partner")
        await log_interaction(
            product_id="prod10",
            contact_id=cid,
            kind="meeting",
            summary="Way in the future",
            follow_up="1m",
        )
        results = await get_pending_follow_ups(product_id="prod10", within_days=7)
        summaries = [r["summary"] for r in results]
        assert "Way in the future" not in summaries

    @pytest.mark.asyncio
    async def test_follow_ups_excludes_done(self, db, monkeypatch):
        """Done interactions (done=1) should not appear in pending list."""
        from src.app.crm import add_contact, log_interaction, get_pending_follow_ups, mark_follow_up_done
        cid = await add_contact("prod11", "@kevin", "Kevin", "customer")
        iid = await log_interaction(
            product_id="prod11",
            contact_id=cid,
            kind="call",
            summary="Already handled",
            follow_up="2d",
        )
        await mark_follow_up_done(iid)
        results = await get_pending_follow_ups(product_id="prod11", within_days=7)
        summaries = [r["summary"] for r in results]
        assert "Already handled" not in summaries


# ===========================================================================
# 6. Consent grant / revoke / expiry via has_consent
# ===========================================================================


class TestConsent:
    @pytest.mark.asyncio
    async def test_has_consent_false_when_no_record(self, db, monkeypatch):
        from src.app.crm import has_consent
        result = await has_consent(
            product_id="prod12",
            contact_id=999,
            purpose="quote_use",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_has_consent_true_after_grant(self, db, monkeypatch):
        from src.app.crm import add_contact, grant_consent, has_consent
        cid = await add_contact("prod13", "@lena", "Lena", "customer")
        await grant_consent(
            product_id="prod13",
            contact_id=cid,
            purpose="data_processing",
            source_evidence_url="https://example.com/tos",
        )
        result = await has_consent(
            product_id="prod13",
            contact_id=cid,
            purpose="data_processing",
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_has_consent_false_after_revoke(self, db, monkeypatch):
        from src.app.crm import add_contact, grant_consent, revoke_consent, has_consent
        cid = await add_contact("prod14", "@mike", "Mike", "customer")
        await grant_consent(
            product_id="prod14",
            contact_id=cid,
            purpose="marketing_email",
            source_evidence_url="https://example.com/terms",
        )
        await revoke_consent(
            product_id="prod14",
            contact_id=cid,
            purpose="marketing_email",
        )
        result = await has_consent(
            product_id="prod14",
            contact_id=cid,
            purpose="marketing_email",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_has_consent_false_after_expiry(self, db, monkeypatch):
        """Expired consent (expires_at in the past) returns False."""
        from src.app.crm import add_contact, grant_consent, has_consent
        cid = await add_contact("prod15", "@nancy", "Nancy", "customer")
        # Grant with expires_at in the past
        past = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        await grant_consent(
            product_id="prod15",
            contact_id=cid,
            purpose="interview_recording",
            source_evidence_url="https://example.com/consent",
            expires_at=past,
        )
        result = await has_consent(
            product_id="prod15",
            contact_id=cid,
            purpose="interview_recording",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_has_consent_true_with_future_expiry(self, db, monkeypatch):
        """Consent with future expires_at is still valid."""
        from src.app.crm import add_contact, grant_consent, has_consent
        cid = await add_contact("prod16", "@oliver", "Oliver", "advisor")
        future = (datetime.now(timezone.utc) + timedelta(days=365)).strftime("%Y-%m-%d %H:%M:%S")
        await grant_consent(
            product_id="prod16",
            contact_id=cid,
            purpose="case_study",
            source_evidence_url="https://example.com/ok",
            expires_at=future,
        )
        result = await has_consent(
            product_id="prod16",
            contact_id=cid,
            purpose="case_study",
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_check_consent_returns_details(self, db, monkeypatch):
        """check_consent returns the consent record dict."""
        from src.app.crm import add_contact, grant_consent, check_consent
        cid = await add_contact("prod17", "@patricia", "Patricia", "investor")
        await grant_consent(
            product_id="prod17",
            contact_id=cid,
            purpose="data_processing",
            source_evidence_url="https://example.com/gdpr",
        )
        record = await check_consent(
            product_id="prod17",
            contact_id=cid,
            purpose="data_processing",
        )
        assert record is not None
        assert record["purpose"] == "data_processing"
        assert record["revoked_at"] is None


# ===========================================================================
# 7. follow_up_reminder digest
# ===========================================================================


class TestFollowUpReminderDigest:
    @pytest.mark.asyncio
    async def test_reminder_digest_empty(self, db, monkeypatch):
        """Digest with no pending follow-ups returns an 'all clear' message."""
        from src.app.jobs.follow_up_reminder import build_digest
        text = await build_digest(product_id="prod_empty")
        assert isinstance(text, str)
        assert len(text) > 0

    @pytest.mark.asyncio
    async def test_reminder_digest_with_pending(self, db, monkeypatch):
        """Digest with pending follow-ups mentions the contact handle."""
        from src.app.crm import add_contact, log_interaction
        from src.app.jobs.follow_up_reminder import build_digest
        cid = await add_contact("prod18", "@quinn", "Quinn", "customer")
        await log_interaction(
            product_id="prod18",
            contact_id=cid,
            kind="call",
            summary="Check about contract renewal",
            follow_up="2d",
        )
        text = await build_digest(product_id="prod18")
        assert "@quinn" in text or "Quinn" in text or "Check about contract renewal" in text

    @pytest.mark.asyncio
    async def test_run_follow_up_reminder_returns_ok(self, db, monkeypatch):
        """run_follow_up_reminder() returns {"ok": True}."""
        from src.app.jobs.follow_up_reminder import run_follow_up_reminder
        result = await run_follow_up_reminder()
        assert result.get("ok") is True


# ===========================================================================
# 8. Reversibility entries for CRM mr_roboto verbs
# ===========================================================================


class TestReversibility:
    def test_crm_verbs_in_reversibility_registry(self):
        """All CRM verbs should have reversibility entries registered."""
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        expected_verbs = [
            "crm/add_contact",
            "crm/log_interaction",
            "crm/grant_consent",
            "crm/revoke_consent",
        ]
        for verb in expected_verbs:
            assert verb in VERB_REVERSIBILITY, f"Missing reversibility for {verb}"


# ===========================================================================
# 9. Telegram command handlers registered
# ===========================================================================


class TestTelegramHandlers:
    def test_crm_commands_registered(self):
        """All 5 CRM commands must be registered in _setup_handlers."""
        import ast
        import pathlib
        src = pathlib.Path(
            "C:/Users/sakir/Dropbox/Workspaces/kutay/src/app/telegram_bot.py"
        ).read_text(encoding="utf-8")
        expected_commands = ["contact", "log", "contacts", "follow_ups", "consent"]
        for cmd in expected_commands:
            assert f'CommandHandler("{cmd}"' in src, f"Missing handler for /{cmd}"
