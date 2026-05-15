"""Z7 T3E — B6: Crisis comms tiered playbook tests.

Covers:
  1. DB migration: crisis_events table + marketing_freeze table exist after init_db.
  2. Crisis event lifecycle: open → active → resolved.
  3. crisis/freeze_marketing writes per-product freeze flag; resume clears it.
  4. crisis/draft_holding produces holding-statement variants (LLM mocked).
  5. crisis/disclosure_timer: Tier-3 timer emits escalating founder_action every 6h.
  6. Tier classification founder_action emitted on open.
  7. Tier 3+ counsel-engaged? founder_action with two-button ack.
  8. /crisis commands parse correctly: open, resume, status.
  9. Reversibility entries registered for all B6 crisis verbs.
  10. Trigger from B3 critical incident opens a crisis_events row.
  11. Playbook files exist for all 4 tiers.
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile

import pytest
import pytest_asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Fresh SQLite DB for B6 tests."""
    db_file = str(tmp_path / "test_b6.db")
    monkeypatch.setenv("DB_PATH", db_file)
    try:
        import src.app.config as _cfg
        monkeypatch.setattr(_cfg, "DB_PATH", db_file)
    except Exception:
        pass
    try:
        import src.infra.db as _db_mod
        monkeypatch.setattr(_db_mod, "DB_PATH", db_file)
        monkeypatch.setattr(_db_mod, "_db_connection", None)
        monkeypatch.setattr(_db_mod, "_db_connection_path", None)
    except Exception:
        pass
    return db_file


@pytest_asyncio.fixture
async def initialized_db(tmp_db):
    """Initialize DB with full schema (includes Z7 B6 migrations)."""
    import src.infra.db as _db_mod
    _db_mod._db_connection = None
    _db_mod._db_connection_path = None

    from src.infra.db import init_db, get_db
    await init_db()
    db = await get_db()
    yield db

    _db_mod._db_connection = None
    _db_mod._db_connection_path = None


# ===========================================================================
# 1. DB migrations
# ===========================================================================

class TestMigrations:
    @pytest.mark.asyncio
    async def test_crisis_events_table_exists(self, initialized_db):
        """crisis_events table created by 2026-05-15-z7-crisis-events migration."""
        db = initialized_db
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='crisis_events'"
        ) as cur:
            row = await cur.fetchone()
        assert row is not None, "crisis_events table should exist"

    @pytest.mark.asyncio
    async def test_marketing_freeze_table_exists(self, initialized_db):
        """marketing_freeze table created by 2026-05-15-z7-marketing-freeze migration."""
        db = initialized_db
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='marketing_freeze'"
        ) as cur:
            row = await cur.fetchone()
        assert row is not None, "marketing_freeze table should exist"

    @pytest.mark.asyncio
    async def test_crisis_events_columns(self, initialized_db):
        """crisis_events has required columns including product_id."""
        db = initialized_db
        async with db.execute("PRAGMA table_info(crisis_events)") as cur:
            cols = {row[1] for row in await cur.fetchall()}
        required = {
            "event_id", "product_id", "opened_at", "tier", "source",
            "summary", "status", "resolved_at", "postmortem_url"
        }
        missing = required - cols
        assert not missing, f"crisis_events missing columns: {missing}"

    @pytest.mark.asyncio
    async def test_marketing_freeze_columns(self, initialized_db):
        """marketing_freeze has required columns for per-product freeze."""
        db = initialized_db
        async with db.execute("PRAGMA table_info(marketing_freeze)") as cur:
            cols = {row[1] for row in await cur.fetchall()}
        required = {"freeze_id", "product_id", "event_id", "frozen_at", "resumed_at"}
        missing = required - cols
        assert not missing, f"marketing_freeze missing columns: {missing}"


# ===========================================================================
# 2. Crisis event lifecycle
# ===========================================================================

class TestCrisisEventLifecycle:
    @pytest.mark.asyncio
    async def test_open_crisis_event(self, initialized_db):
        """Opening a crisis event inserts a row with status='active'."""
        from mr_roboto.crisis_open import open_crisis_event

        db = initialized_db
        event = await open_crisis_event(
            product_id="prod-abc",
            tier=2,
            source="manual",
            summary="Extended downtime on payment service",
        )
        assert event["event_id"] > 0
        assert event["status"] == "active"
        assert event["tier"] == 2
        assert event["product_id"] == "prod-abc"
        assert event["source"] == "manual"

    @pytest.mark.asyncio
    async def test_resolve_crisis_event(self, initialized_db):
        """Resolving a crisis event sets status='resolved' and resolved_at."""
        from mr_roboto.crisis_open import open_crisis_event, resolve_crisis_event

        event = await open_crisis_event(
            product_id="prod-xyz",
            tier=1,
            source="manual",
            summary="Brand misstep on Twitter",
        )
        event_id = event["event_id"]
        result = await resolve_crisis_event(event_id=event_id, product_id="prod-xyz")
        assert result["status"] == "resolved"
        assert result["resolved_at"] is not None

    @pytest.mark.asyncio
    async def test_status_active_only_active(self, initialized_db):
        """Querying active events returns only open (not resolved) rows."""
        from mr_roboto.crisis_open import open_crisis_event, resolve_crisis_event

        e1 = await open_crisis_event(product_id="p1", tier=1, source="manual", summary="s1")
        e2 = await open_crisis_event(product_id="p1", tier=2, source="manual", summary="s2")
        await resolve_crisis_event(event_id=e1["event_id"], product_id="p1")

        db = initialized_db
        async with db.execute(
            "SELECT event_id FROM crisis_events WHERE product_id=? AND status='active'",
            ("p1",),
        ) as cur:
            active = await cur.fetchall()
        active_ids = {row[0] for row in active}
        assert e2["event_id"] in active_ids
        assert e1["event_id"] not in active_ids


# ===========================================================================
# 3. crisis/freeze_marketing and resume
# ===========================================================================

class TestFreezeMarketing:
    @pytest.mark.asyncio
    async def test_freeze_writes_row(self, initialized_db, monkeypatch):
        """freeze_marketing inserts a marketing_freeze row for the product."""
        from mr_roboto.crisis_freeze_marketing import run as freeze_run

        result = await freeze_run({
            "product_id": "prod-freeze-test",
            "event_id": 42,
        })
        assert result["status"] == "ok"
        assert result["frozen"] is True

        db = initialized_db
        async with db.execute(
            "SELECT freeze_id, resumed_at FROM marketing_freeze "
            "WHERE product_id=? AND event_id=?",
            ("prod-freeze-test", 42),
        ) as cur:
            row = await cur.fetchone()
        assert row is not None, "freeze row should be inserted"
        assert row[1] is None, "resumed_at should be null initially"

    @pytest.mark.asyncio
    async def test_is_frozen_returns_true(self, initialized_db):
        """is_marketing_frozen() returns True after freeze."""
        from mr_roboto.crisis_freeze_marketing import run as freeze_run, is_marketing_frozen

        await freeze_run({"product_id": "prod-check", "event_id": 10})
        assert await is_marketing_frozen("prod-check") is True

    @pytest.mark.asyncio
    async def test_resume_clears_freeze(self, initialized_db):
        """resume sets resumed_at; is_marketing_frozen() then returns False."""
        from mr_roboto.crisis_freeze_marketing import (
            run as freeze_run,
            resume_marketing_freeze,
            is_marketing_frozen,
        )

        await freeze_run({"product_id": "prod-resume", "event_id": 11})
        assert await is_marketing_frozen("prod-resume") is True

        await resume_marketing_freeze("prod-resume")
        assert await is_marketing_frozen("prod-resume") is False

    @pytest.mark.asyncio
    async def test_freeze_idempotent(self, initialized_db):
        """Calling freeze twice for same product+event is idempotent."""
        from mr_roboto.crisis_freeze_marketing import run as freeze_run

        r1 = await freeze_run({"product_id": "prod-idem", "event_id": 99})
        r2 = await freeze_run({"product_id": "prod-idem", "event_id": 99})
        assert r1["status"] == "ok"
        assert r2["status"] == "ok"

        db = initialized_db
        async with db.execute(
            "SELECT COUNT(*) FROM marketing_freeze WHERE product_id=? AND event_id=?",
            ("prod-idem", 99),
        ) as cur:
            cnt = (await cur.fetchone())[0]
        # Idempotent: only one active freeze row
        assert cnt == 1

    @pytest.mark.asyncio
    async def test_not_frozen_by_default(self, initialized_db):
        """is_marketing_frozen() returns False for product never frozen."""
        from mr_roboto.crisis_freeze_marketing import is_marketing_frozen

        assert await is_marketing_frozen("prod-never-frozen") is False


# ===========================================================================
# 4. crisis/draft_holding produces variants (LLM mocked)
# ===========================================================================

class TestDraftHolding:
    @pytest.mark.asyncio
    async def test_draft_holding_returns_variants(self, initialized_db, monkeypatch):
        """draft_holding returns at least 2 holding-statement variants."""
        # Mock LLM call
        async def _mock_llm(tier, summary, playbook_text):
            return [
                f"Variant A for tier {tier}: We are aware of the issue.",
                f"Variant B for tier {tier}: Our team is investigating.",
            ]

        monkeypatch.setattr(
            "mr_roboto.crisis_draft_holding._call_llm_draft",
            _mock_llm,
        )
        from mr_roboto.crisis_draft_holding import run as draft_run

        result = await draft_run({
            "product_id": "prod-draft",
            "event_id": 1,
            "tier": 2,
            "summary": "Extended outage on authentication service",
        })
        assert result["status"] == "ok"
        assert len(result["variants"]) >= 2
        assert all(isinstance(v, str) and len(v) > 10 for v in result["variants"])

    @pytest.mark.asyncio
    async def test_draft_holding_reads_playbook(self, monkeypatch):
        """draft_holding reads the tier-specific playbook file."""
        captured = {}

        async def _mock_llm(tier, summary, playbook_text):
            captured["playbook_text"] = playbook_text
            captured["tier"] = tier
            return ["mock variant"]

        monkeypatch.setattr(
            "mr_roboto.crisis_draft_holding._call_llm_draft",
            _mock_llm,
        )
        from mr_roboto.crisis_draft_holding import run as draft_run

        result = await draft_run({
            "product_id": "prod-pb",
            "event_id": 2,
            "tier": 1,
            "summary": "Brand misstep",
        })
        assert result["status"] == "ok"
        # Playbook text should be non-empty (file was read)
        assert len(captured.get("playbook_text", "")) > 0
        assert captured["tier"] == 1

    @pytest.mark.asyncio
    async def test_draft_holding_missing_event_id(self, monkeypatch):
        """draft_holding returns error when event_id missing."""
        from mr_roboto.crisis_draft_holding import run as draft_run

        result = await draft_run({
            "product_id": "prod-err",
            "tier": 1,
            "summary": "test",
        })
        assert result["status"] == "error"
        assert "event_id" in result["error"]

    @pytest.mark.asyncio
    async def test_draft_holding_llm_fallback(self, monkeypatch):
        """draft_holding returns a fallback variant if LLM raises."""
        async def _fail_llm(tier, summary, playbook_text):
            raise RuntimeError("LLM unavailable")

        monkeypatch.setattr(
            "mr_roboto.crisis_draft_holding._call_llm_draft",
            _fail_llm,
        )
        from mr_roboto.crisis_draft_holding import run as draft_run

        result = await draft_run({
            "product_id": "prod-fallback",
            "event_id": 5,
            "tier": 3,
            "summary": "Security breach detected",
        })
        assert result["status"] == "ok"
        assert len(result["variants"]) >= 1


# ===========================================================================
# 5. crisis/disclosure_timer — Tier 3 escalating reminders
# ===========================================================================

class TestDisclosureTimer:
    @pytest.mark.asyncio
    async def test_disclosure_timer_emits_founder_action(self, initialized_db, monkeypatch):
        """disclosure_timer emits a founder_action with escalating urgency."""
        created_actions = []

        async def _mock_fa_create(**kwargs):
            created_actions.append(kwargs)
            from types import SimpleNamespace
            return SimpleNamespace(id=99)

        monkeypatch.setattr(
            "mr_roboto.crisis_disclosure_timer._create_founder_action",
            _mock_fa_create,
        )
        from mr_roboto.crisis_disclosure_timer import run as timer_run

        result = await timer_run({
            "product_id": "prod-t3",
            "event_id": 7,
            "mission_id": 1,
            "hours_elapsed": 12,
            "jurisdiction": "GDPR",
        })
        assert result["status"] == "ok"
        assert len(created_actions) == 1
        fa = created_actions[0]
        assert fa["kind"] == "generic"
        assert "72" in fa["title"] or "disclosure" in fa["title"].lower()

    @pytest.mark.asyncio
    async def test_disclosure_timer_urgency_increases(self, initialized_db, monkeypatch):
        """Urgency escalates as hours_elapsed increases toward 72h."""
        urgencies = []

        async def _mock_fa_create(**kwargs):
            urgencies.append(kwargs.get("urgent", False))
            from types import SimpleNamespace
            return SimpleNamespace(id=99)

        monkeypatch.setattr(
            "mr_roboto.crisis_disclosure_timer._create_founder_action",
            _mock_fa_create,
        )
        from mr_roboto.crisis_disclosure_timer import run as timer_run

        # 12h elapsed — not urgent yet
        await timer_run({
            "product_id": "prod-urg",
            "event_id": 8,
            "mission_id": 1,
            "hours_elapsed": 12,
            "jurisdiction": "GDPR",
        })
        # 60h elapsed — should be urgent (>80% of 72h used)
        await timer_run({
            "product_id": "prod-urg",
            "event_id": 8,
            "mission_id": 1,
            "hours_elapsed": 60,
            "jurisdiction": "GDPR",
        })
        assert len(urgencies) == 2
        assert urgencies[1] is True  # 60h should be urgent

    @pytest.mark.asyncio
    async def test_disclosure_timer_only_tier3(self, initialized_db, monkeypatch):
        """disclosure_timer is a no-op when tier < 3."""
        called = []

        async def _mock_fa_create(**kwargs):
            called.append(kwargs)
            from types import SimpleNamespace
            return SimpleNamespace(id=99)

        monkeypatch.setattr(
            "mr_roboto.crisis_disclosure_timer._create_founder_action",
            _mock_fa_create,
        )
        from mr_roboto.crisis_disclosure_timer import run as timer_run

        result = await timer_run({
            "product_id": "prod-t1",
            "event_id": 9,
            "mission_id": 1,
            "hours_elapsed": 24,
            "tier": 2,  # Not Tier 3
            "jurisdiction": "GDPR",
        })
        # No founder_action for Tier 2
        assert result["status"] == "skipped" or len(called) == 0


# ===========================================================================
# 6. Tier classification founder_action
# ===========================================================================

class TestTierClassification:
    @pytest.mark.asyncio
    async def test_open_emits_tier_classification_founder_action(
        self, initialized_db, monkeypatch
    ):
        """Opening a crisis event emits a tier-classification founder_action."""
        created = []

        async def _mock_fa_create(**kwargs):
            created.append(kwargs)
            from types import SimpleNamespace
            return SimpleNamespace(id=55)

        monkeypatch.setattr(
            "mr_roboto.crisis_open._create_founder_action",
            _mock_fa_create,
        )
        from mr_roboto.crisis_open import open_crisis_event

        await open_crisis_event(
            product_id="prod-tier-class",
            tier=2,
            source="incident",
            summary="Extended outage",
        )
        assert len(created) >= 1
        # Should have tier-classification card
        titles = [c.get("title", "").lower() for c in created]
        assert any("tier" in t or "crisis" in t for t in titles)

    @pytest.mark.asyncio
    async def test_tier3_emits_counsel_founder_action(
        self, initialized_db, monkeypatch
    ):
        """Tier 3+ open emits a counsel-engaged? founder_action."""
        created = []

        async def _mock_fa_create(**kwargs):
            created.append(kwargs)
            from types import SimpleNamespace
            return SimpleNamespace(id=66)

        monkeypatch.setattr(
            "mr_roboto.crisis_open._create_founder_action",
            _mock_fa_create,
        )
        from mr_roboto.crisis_open import open_crisis_event

        await open_crisis_event(
            product_id="prod-counsel",
            tier=3,
            source="manual",
            summary="Credentials leaked — GDPR breach suspected",
        )
        # Should have at least 2 cards: tier-classify + counsel-engaged
        assert len(created) >= 2
        all_text = " ".join(c.get("title", "") + c.get("why", "") for c in created).lower()
        assert "counsel" in all_text or "legal" in all_text

    @pytest.mark.asyncio
    async def test_tier4_emits_counsel_founder_action(
        self, initialized_db, monkeypatch
    ):
        """Tier 4 also emits a counsel-engaged? founder_action."""
        created = []

        async def _mock_fa_create(**kwargs):
            created.append(kwargs)
            from types import SimpleNamespace
            return SimpleNamespace(id=77)

        monkeypatch.setattr(
            "mr_roboto.crisis_open._create_founder_action",
            _mock_fa_create,
        )
        from mr_roboto.crisis_open import open_crisis_event

        await open_crisis_event(
            product_id="prod-t4",
            tier=4,
            source="manual",
            summary="Regulatory action filed",
        )
        all_text = " ".join(c.get("title", "") + c.get("why", "") for c in created).lower()
        assert "counsel" in all_text or "legal" in all_text


# ===========================================================================
# 7. /crisis command parsing
# ===========================================================================

class TestCrisisCommandParsing:
    def test_parse_open_with_tier(self):
        """parse_crisis_cmd parses '/crisis open 2' correctly."""
        from mr_roboto.crisis_open import parse_crisis_cmd

        cmd = parse_crisis_cmd("/crisis open 2 credentials leaked")
        assert cmd["subcommand"] == "open"
        assert cmd["tier"] == 2
        assert "credentials" in cmd.get("summary", "")

    def test_parse_open_without_tier_defaults_to_1(self):
        """parse_crisis_cmd defaults to tier=1 when tier not given."""
        from mr_roboto.crisis_open import parse_crisis_cmd

        cmd = parse_crisis_cmd("/crisis open brand misstep happened")
        assert cmd["subcommand"] == "open"
        assert cmd["tier"] == 1

    def test_parse_resume(self):
        """parse_crisis_cmd parses '/crisis resume prod-xyz'."""
        from mr_roboto.crisis_open import parse_crisis_cmd

        cmd = parse_crisis_cmd("/crisis resume prod-xyz")
        assert cmd["subcommand"] == "resume"
        assert cmd["product_id"] == "prod-xyz"

    def test_parse_status(self):
        """parse_crisis_cmd parses '/crisis status'."""
        from mr_roboto.crisis_open import parse_crisis_cmd

        cmd = parse_crisis_cmd("/crisis status")
        assert cmd["subcommand"] == "status"

    def test_parse_unknown_subcommand(self):
        """parse_crisis_cmd returns error for unknown subcommand."""
        from mr_roboto.crisis_open import parse_crisis_cmd

        cmd = parse_crisis_cmd("/crisis foobar")
        assert "error" in cmd or cmd["subcommand"] == "unknown"


# ===========================================================================
# 8. Reversibility entries for B6 verbs
# ===========================================================================

class TestReversibility:
    def test_crisis_freeze_marketing_reversible(self):
        """crisis/freeze_marketing is registered as 'full' (DB write, reversible)."""
        from mr_roboto.reversibility import VERB_REVERSIBILITY

        assert "crisis/freeze_marketing" in VERB_REVERSIBILITY
        assert VERB_REVERSIBILITY["crisis/freeze_marketing"] == "full"

    def test_crisis_draft_holding_reversible(self):
        """crisis/draft_holding is registered as 'full' (draft only, no publish)."""
        from mr_roboto.reversibility import VERB_REVERSIBILITY

        assert "crisis/draft_holding" in VERB_REVERSIBILITY
        assert VERB_REVERSIBILITY["crisis/draft_holding"] == "full"

    def test_crisis_disclosure_timer_reversible(self):
        """crisis/disclosure_timer is registered as 'irreversible' (surfaces to founder)."""
        from mr_roboto.reversibility import VERB_REVERSIBILITY

        assert "crisis/disclosure_timer" in VERB_REVERSIBILITY
        # Timer emits a founder_action which is visible — irreversible
        assert VERB_REVERSIBILITY["crisis/disclosure_timer"] == "irreversible"


# ===========================================================================
# 9. Trigger from B3 critical incident
# ===========================================================================

class TestIncidentTrigger:
    @pytest.mark.asyncio
    async def test_trigger_from_critical_incident(self, initialized_db, monkeypatch):
        """trigger_crisis_from_incident opens crisis_events row for critical incidents."""
        from mr_roboto.crisis_open import trigger_crisis_from_incident

        # Mock founder_action creation
        async def _noop_fa(**kwargs):
            from types import SimpleNamespace
            return SimpleNamespace(id=1)

        monkeypatch.setattr("mr_roboto.crisis_open._create_founder_action", _noop_fa)

        result = await trigger_crisis_from_incident(
            product_id="prod-incident",
            incident_id=55,
            severity="critical",
            summary="DB disk full causing payment failures",
        )
        assert result["opened"] is True
        assert result["event_id"] > 0
        assert result["tier"] == 2  # outage/data issue = Tier 2

    @pytest.mark.asyncio
    async def test_non_critical_incident_no_crisis(self, initialized_db, monkeypatch):
        """trigger_crisis_from_incident does NOT open crisis for minor incidents."""
        from mr_roboto.crisis_open import trigger_crisis_from_incident

        result = await trigger_crisis_from_incident(
            product_id="prod-minor",
            incident_id=56,
            severity="minor",
            summary="Slow page load",
        )
        assert result["opened"] is False


# ===========================================================================
# 10. Playbook files exist
# ===========================================================================

class TestPlaybookFiles:
    def test_tier1_playbook_exists(self):
        """playbooks/crisis_comms_tier1.md exists."""
        import os

        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "playbooks", "crisis_comms_tier1.md"
        )
        assert os.path.isfile(os.path.normpath(path)), "Tier 1 playbook missing"

    def test_tier2_playbook_exists(self):
        """playbooks/crisis_comms_tier2.md exists."""
        import os

        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "playbooks", "crisis_comms_tier2.md"
        )
        assert os.path.isfile(os.path.normpath(path)), "Tier 2 playbook missing"

    def test_tier3_playbook_exists(self):
        """playbooks/crisis_comms_tier3.md exists."""
        import os

        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "playbooks", "crisis_comms_tier3.md"
        )
        assert os.path.isfile(os.path.normpath(path)), "Tier 3 playbook missing"

    def test_tier4_playbook_exists(self):
        """playbooks/crisis_comms_tier4.md exists."""
        import os

        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "playbooks", "crisis_comms_tier4.md"
        )
        assert os.path.isfile(os.path.normpath(path)), "Tier 4 playbook missing"

    def test_playbook_files_have_content(self):
        """All four playbook files have non-trivial content (>100 bytes each)."""
        import os

        base = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..", "playbooks")
        )
        for tier in range(1, 5):
            path = os.path.join(base, f"crisis_comms_tier{tier}.md")
            size = os.path.getsize(path)
            assert size > 100, f"Tier {tier} playbook too small ({size} bytes)"
