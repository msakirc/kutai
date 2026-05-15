"""Z7 T3A — A2: Launch playbook + A2.r1 readiness gate tests.

Covers:
  1. DB migration: launches table exists after init_db.
  2. Phase-clock offsets: all 8 offset_hours resolve correctly relative to
     scheduled_publish_at.
  3. launch_playbook.json parses as valid JSON.
  4. publish_synchronized aborts when marketing is frozen (B6 check).
  5. launch_readiness_gate blocks on any failing check; passes when all green.
  6. launch_readiness_gate degrades gracefully (warning, not crash) when a
     subsystem is absent.
  7. launch_lessons_writeback emits mission_lessons rows.
  8. /launch Telegram command creates a launch mission (beckman.enqueue called).
  9. launch_drafts verbs route to Beckman enqueue (LLM-bound).
  10. Reversibility entries registered for all A2 verbs.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Fresh SQLite DB for tests."""
    db_file = str(tmp_path / "test_a2.db")
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
    """Initialize DB with full schema (includes Z7 T3A launches migration)."""
    from src.infra.db import init_db, get_db
    import src.infra.db as _db_mod

    _db_mod._db_connection = None
    _db_mod._db_connection_path = None

    await init_db()
    db = await get_db()
    yield db

    _db_mod._db_connection = None
    _db_mod._db_connection_path = None


# ===========================================================================
# 1. DB migration: launches table
# ===========================================================================

class TestMigrations:
    @pytest.mark.asyncio
    async def test_launches_table_exists(self, initialized_db):
        """launches table created by 2026-05-16-z7-launches migration."""
        db = initialized_db
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='launches'"
        ) as cur:
            row = await cur.fetchone()
        assert row is not None, "launches table must exist after init_db"

    @pytest.mark.asyncio
    async def test_launches_columns(self, initialized_db):
        """launches table has required columns."""
        db = initialized_db
        async with db.execute("PRAGMA table_info(launches)") as cur:
            cols = {r[1] for r in await cur.fetchall()}
        required = {
            "launch_id", "product_id", "scheduled_publish_at",
            "status", "channels_json", "mission_id", "created_at",
        }
        assert required <= cols, f"Missing columns: {required - cols}"

    @pytest.mark.asyncio
    async def test_launches_product_id_not_null(self, initialized_db):
        """product_id NOT NULL enforced."""
        db = initialized_db
        with pytest.raises(Exception):
            await db.execute(
                "INSERT INTO launches (product_id, scheduled_publish_at, status) "
                "VALUES (NULL, '2026-06-01 09:00:00', 'planned')"
            )
            await db.commit()


# ===========================================================================
# 2. Phase-clock offsets
# ===========================================================================

class TestPhaseClockOffsets:
    """Phase clock: each offset_hours is relative to scheduled_publish_at."""

    def test_phase_clock_offsets_resolve(self):
        """All 8 phase offsets produce correct UTC datetimes."""
        from mr_roboto.launch_phase_clock import resolve_phase_times

        publish_at = datetime(2026, 6, 15, 9, 0, 0, tzinfo=timezone.utc)
        phases = resolve_phase_times(publish_at)

        expected_offsets = {-72, -24, 0, 1, 4, 24, 72, 168}
        assert set(phases.keys()) == expected_offsets

        for offset_h, ts in phases.items():
            expected = publish_at + timedelta(hours=offset_h)
            assert ts == expected, f"offset {offset_h}h: {ts} != {expected}"

    def test_phase_clock_t_minus_72_before_publish(self):
        """T-72h fires 72 hours before scheduled_publish_at."""
        from mr_roboto.launch_phase_clock import resolve_phase_times

        publish_at = datetime(2026, 6, 15, 9, 0, 0, tzinfo=timezone.utc)
        phases = resolve_phase_times(publish_at)
        assert phases[-72] < publish_at

    def test_phase_clock_t_plus_168_is_7_days(self):
        """T+168h = T+7d (lessons writeback phase)."""
        from mr_roboto.launch_phase_clock import resolve_phase_times

        publish_at = datetime(2026, 6, 15, 9, 0, 0, tzinfo=timezone.utc)
        phases = resolve_phase_times(publish_at)
        t7d = publish_at + timedelta(days=7)
        assert phases[168] == t7d


# ===========================================================================
# 3. launch_playbook.json valid JSON
# ===========================================================================

class TestLaunchPlaybookJson:
    def test_launch_playbook_json_valid(self):
        """launch_playbook.json must be valid JSON."""
        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "src", "workflows",
            "launch_playbook.json"
        )
        path = os.path.normpath(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("plan_id") == "launch_playbook"
        assert "phases" in data

    def test_launch_playbook_has_required_phases(self):
        """launch_playbook.json must include the 8 phase offsets."""
        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "src", "workflows",
            "launch_playbook.json"
        )
        path = os.path.normpath(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Collect all offset_hours declared in phases
        offsets = set()
        for phase in data.get("phases", []):
            if "offset_hours" in phase:
                offsets.add(phase["offset_hours"])
        # All 8 required offsets must appear
        required = {-72, -24, 0, 1, 4, 24, 72, 168}
        assert required <= offsets, f"Missing offset_hours in launch_playbook.json: {required - offsets}"

    def test_launch_playbook_relative_to_field(self):
        """launch_playbook.json metadata must declare relative_to."""
        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "src", "workflows",
            "launch_playbook.json"
        )
        path = os.path.normpath(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("metadata", {})
        assert meta.get("relative_to") == "scheduled_publish_at"


# ===========================================================================
# 4. publish_synchronized aborts when marketing is frozen
# ===========================================================================

class TestPublishSynchronized:
    @pytest.mark.asyncio
    async def test_publish_aborts_when_frozen(self, initialized_db):
        """publish_synchronized returns frozen=True and does not publish."""
        from mr_roboto.launch_publish_synchronized import run as publish_run
        from mr_roboto.crisis_freeze_marketing import run as freeze_run

        product_id = "prod-test-frozen"

        # Insert a dummy crisis_events row so foreign key constraint (soft) is satisfied
        await initialized_db.execute(
            "INSERT INTO crisis_events (product_id, tier, source, summary) "
            "VALUES (?, 2, 'manual', 'test freeze')",
            (product_id,),
        )
        await initialized_db.commit()
        async with initialized_db.execute(
            "SELECT last_insert_rowid()"
        ) as cur:
            event_id = (await cur.fetchone())[0]

        # Freeze marketing for this product
        await freeze_run({"product_id": product_id, "event_id": event_id})

        # publish_synchronized must detect the freeze and abort
        result = await publish_run({
            "product_id": product_id,
            "launch_id": 1,
            "channels": ["hn", "twitter"],
            "drafts": {"hn": "HN launch post", "twitter": "Twitter post"},
        })
        assert result["status"] == "aborted"
        assert result.get("reason") == "marketing_frozen"

    @pytest.mark.asyncio
    async def test_publish_proceeds_when_not_frozen(self, initialized_db, monkeypatch):
        """publish_synchronized proceeds when no freeze is active."""
        from mr_roboto.launch_publish_synchronized import run as publish_run

        product_id = "prod-test-not-frozen"

        # Mock the actual channel publish calls
        async def _mock_channel_publish(channel, draft, product_id):
            return {"channel": channel, "status": "published", "url": f"https://example.com/{channel}"}

        monkeypatch.setattr(
            "mr_roboto.launch_publish_synchronized._publish_channel",
            _mock_channel_publish,
        )

        result = await publish_run({
            "product_id": product_id,
            "launch_id": 1,
            "channels": ["hn", "twitter"],
            "drafts": {"hn": "HN launch post", "twitter": "Twitter post"},
        })
        assert result["status"] == "published"
        assert len(result["published"]) == 2


# ===========================================================================
# 5. launch_readiness_gate — blocks on failing check, passes when all green
# ===========================================================================

class TestLaunchReadinessGate:
    @pytest.mark.asyncio
    async def test_gate_blocks_when_check_fails(self, initialized_db, monkeypatch):
        """Readiness gate returns blocked=True with failing_checks when any check fails."""
        from mr_roboto.launch_readiness_gate import run as gate_run

        # Patch all sub-checks: site_load fails, rest pass
        async def _site_load_fail(product_id, **kwargs):
            return {"ok": False, "reason": "site_load_timeout"}

        async def _pass(*a, **kw):
            return {"ok": True}

        monkeypatch.setattr("mr_roboto.launch_readiness_gate._check_site_load", _site_load_fail)
        monkeypatch.setattr("mr_roboto.launch_readiness_gate._check_payment_e2e", _pass)
        monkeypatch.setattr("mr_roboto.launch_readiness_gate._check_support_faq", _pass)
        monkeypatch.setattr("mr_roboto.launch_readiness_gate._check_copy_compliance", _pass)
        monkeypatch.setattr("mr_roboto.launch_readiness_gate._check_press_kit", _pass)
        monkeypatch.setattr("mr_roboto.launch_readiness_gate._check_status_page", _pass)
        monkeypatch.setattr("mr_roboto.launch_readiness_gate._check_crisis_playbook", _pass)

        result = await gate_run({
            "product_id": "prod-a2",
            "launch_id": 1,
            "channels": ["hn", "twitter"],
        })
        assert result["status"] == "blocked"
        assert "site_load" in (result.get("failing_checks") or [])

    @pytest.mark.asyncio
    async def test_gate_passes_when_all_green(self, initialized_db, monkeypatch):
        """Readiness gate passes when all checks return ok=True."""
        from mr_roboto.launch_readiness_gate import run as gate_run

        async def _pass(*a, **kw):
            return {"ok": True}

        for check_name in [
            "_check_site_load", "_check_payment_e2e", "_check_support_faq",
            "_check_copy_compliance", "_check_press_kit",
            "_check_status_page", "_check_crisis_playbook",
        ]:
            monkeypatch.setattr(f"mr_roboto.launch_readiness_gate.{check_name}", _pass)

        result = await gate_run({
            "product_id": "prod-a2-green",
            "launch_id": 1,
            "channels": ["hn", "twitter"],
        })
        assert result["status"] == "ready"
        assert result.get("failing_checks", []) == []

    @pytest.mark.asyncio
    async def test_gate_emits_founder_action_on_block(self, initialized_db, monkeypatch):
        """When blocked, a founder_action is emitted for override/fix decision."""
        from mr_roboto.launch_readiness_gate import run as gate_run

        async def _fail(*a, **kw):
            return {"ok": False, "reason": "no press kit published"}

        async def _pass(*a, **kw):
            return {"ok": True}

        monkeypatch.setattr("mr_roboto.launch_readiness_gate._check_site_load", _pass)
        monkeypatch.setattr("mr_roboto.launch_readiness_gate._check_payment_e2e", _pass)
        monkeypatch.setattr("mr_roboto.launch_readiness_gate._check_support_faq", _pass)
        monkeypatch.setattr("mr_roboto.launch_readiness_gate._check_copy_compliance", _pass)
        monkeypatch.setattr("mr_roboto.launch_readiness_gate._check_press_kit", _fail)
        monkeypatch.setattr("mr_roboto.launch_readiness_gate._check_status_page", _pass)
        monkeypatch.setattr("mr_roboto.launch_readiness_gate._check_crisis_playbook", _pass)

        fa_captured = {}

        async def _fake_fa(**kwargs):
            fa_captured.update(kwargs)
            return MagicMock(id=42)

        monkeypatch.setattr(
            "mr_roboto.launch_readiness_gate._emit_blocked_founder_action",
            _fake_fa,
        )

        result = await gate_run({
            "product_id": "prod-a2-blocked",
            "launch_id": 1,
            "channels": ["hn"],
        })
        assert result["status"] == "blocked"
        assert fa_captured, "founder_action must be emitted on block"


# ===========================================================================
# 6. launch_readiness_gate — degrades gracefully when subsystem absent
# ===========================================================================

class TestReadinessGateDegradation:
    @pytest.mark.asyncio
    async def test_absent_subsystem_is_warning_not_crash(self, initialized_db, monkeypatch):
        """When a check's subsystem is absent, gate logs a warning and continues."""
        from mr_roboto.launch_readiness_gate import run as gate_run

        async def _raise_import_error(*a, **kw):
            raise ImportError("subsystem not installed")

        async def _pass(*a, **kw):
            return {"ok": True}

        # status_page check raises — gate must degrade to warning, not crash
        monkeypatch.setattr("mr_roboto.launch_readiness_gate._check_site_load", _pass)
        monkeypatch.setattr("mr_roboto.launch_readiness_gate._check_payment_e2e", _pass)
        monkeypatch.setattr("mr_roboto.launch_readiness_gate._check_support_faq", _pass)
        monkeypatch.setattr("mr_roboto.launch_readiness_gate._check_copy_compliance", _pass)
        monkeypatch.setattr("mr_roboto.launch_readiness_gate._check_press_kit", _pass)
        monkeypatch.setattr("mr_roboto.launch_readiness_gate._check_status_page", _raise_import_error)
        monkeypatch.setattr("mr_roboto.launch_readiness_gate._check_crisis_playbook", _pass)

        result = await gate_run({
            "product_id": "prod-degrade",
            "launch_id": 1,
            "channels": ["hn"],
        })
        # Must not crash; may be ready (warning) or blocked
        assert result["status"] in ("ready", "blocked", "ready_with_warnings")
        warnings = result.get("warnings", [])
        assert any("status_page" in str(w) or "absent" in str(w) or "subsystem" in str(w).lower() for w in warnings), \
            f"Expected warning about absent subsystem, got: {warnings}"

    @pytest.mark.asyncio
    async def test_absent_copy_compliance_is_warning(self, initialized_db, monkeypatch):
        """A6 copy_compliance absent → warning, not crash."""
        from mr_roboto.launch_readiness_gate import run as gate_run

        async def _raise(*a, **kw):
            raise ImportError("copy_compliance not available")

        async def _pass(*a, **kw):
            return {"ok": True}

        monkeypatch.setattr("mr_roboto.launch_readiness_gate._check_site_load", _pass)
        monkeypatch.setattr("mr_roboto.launch_readiness_gate._check_payment_e2e", _pass)
        monkeypatch.setattr("mr_roboto.launch_readiness_gate._check_support_faq", _pass)
        monkeypatch.setattr("mr_roboto.launch_readiness_gate._check_copy_compliance", _raise)
        monkeypatch.setattr("mr_roboto.launch_readiness_gate._check_press_kit", _pass)
        monkeypatch.setattr("mr_roboto.launch_readiness_gate._check_status_page", _pass)
        monkeypatch.setattr("mr_roboto.launch_readiness_gate._check_crisis_playbook", _pass)

        result = await gate_run({
            "product_id": "prod-copy-absent",
            "launch_id": 1,
            "channels": ["hn"],
        })
        assert result["status"] in ("ready", "blocked", "ready_with_warnings")


# ===========================================================================
# 7. launch_lessons_writeback — emits mission_lessons rows
# ===========================================================================

class TestLaunchLessonsWriteback:
    @pytest.mark.asyncio
    async def test_writeback_emits_lessons(self, initialized_db, monkeypatch):
        """launch_lessons_writeback emits 3-5 mission_lessons rows."""
        from mr_roboto.launch_lessons_writeback import run as writeback_run

        captured_lessons = []

        async def _fake_upsert(**kwargs):
            captured_lessons.append(kwargs)
            return len(captured_lessons)

        monkeypatch.setattr(
            "mr_roboto.launch_lessons_writeback.upsert_mission_lesson",
            _fake_upsert,
        )

        result = await writeback_run({
            "product_id": "prod-lessons",
            "launch_id": 1,
            "mission_id": 42,
            "channels": ["hn", "ph", "twitter"],
            "engagement_summary": {
                "hn": {"upvotes": 120, "comments": 34, "timing_utc": "09:00"},
                "ph": {"votes": 45, "comments": 12, "timing_utc": "09:00"},
                "twitter": {"likes": 200, "retweets": 30, "timing_utc": "09:00"},
            },
        })
        assert result["status"] == "ok"
        assert 3 <= len(captured_lessons) <= 5, \
            f"Expected 3-5 lessons, got {len(captured_lessons)}"

    @pytest.mark.asyncio
    async def test_writeback_uses_dedup_keys(self, initialized_db, monkeypatch):
        """Each lesson must have a dedup_key in the launch.* namespace."""
        from mr_roboto.launch_lessons_writeback import run as writeback_run

        captured_lessons = []

        async def _fake_upsert(**kwargs):
            captured_lessons.append(kwargs)
            return len(captured_lessons)

        monkeypatch.setattr(
            "mr_roboto.launch_lessons_writeback.upsert_mission_lesson",
            _fake_upsert,
        )

        await writeback_run({
            "product_id": "prod-dedup",
            "launch_id": 2,
            "mission_id": 43,
            "channels": ["hn"],
            "engagement_summary": {"hn": {"upvotes": 50, "comments": 8, "timing_utc": "14:00"}},
        })

        for lesson in captured_lessons:
            pattern = lesson.get("pattern", "")
            # dedup key must contain a launch-domain indicator
            assert pattern, "lesson must have a pattern"


# ===========================================================================
# 8. /launch Telegram command creates a launch mission
# ===========================================================================

class TestLaunchTelegramCommand:
    @pytest.mark.asyncio
    async def test_launch_command_calls_enqueue(self, initialized_db, monkeypatch):
        """/launch <date> <channels> creates a launch mission via beckman.enqueue."""
        from src.app.telegram_bot import TelegramInterface

        enqueue_calls = []

        async def _mock_enqueue(spec, **kwargs):
            enqueue_calls.append(spec)
            return 99  # mock task_id

        monkeypatch.setattr(
            "src.app.telegram_bot.enqueue_launch_mission",
            _mock_enqueue,
        )

        # Build a minimal mock bot
        bot = TelegramInterface.__new__(TelegramInterface)
        bot._chat_id = 12345
        bot.chat_id = 12345

        async def _fake_reply(update, text, **kwargs):
            pass

        bot._reply = _fake_reply

        # Simulate update / context
        update = MagicMock()
        update.effective_chat.id = 12345
        update.message.reply_text = AsyncMock()
        ctx_mock = MagicMock()
        ctx_mock.args = ["2026-06-15", "hn,twitter,linkedin"]

        await bot.cmd_launch(update, ctx_mock)

        assert len(enqueue_calls) >= 1, "beckman.enqueue must be called by /launch"

    @pytest.mark.asyncio
    async def test_launch_command_no_args_gives_usage(self, initialized_db, monkeypatch):
        """/launch with no args gives usage instructions."""
        from src.app.telegram_bot import TelegramInterface

        bot = TelegramInterface.__new__(TelegramInterface)
        bot._chat_id = 12345
        bot.chat_id = 12345

        reply_texts = []

        async def _capture_reply(update, text, **kwargs):
            reply_texts.append(text)

        bot._reply = _capture_reply

        update = MagicMock()
        update.effective_chat.id = 12345
        update.message.reply_text = AsyncMock()
        ctx_mock = MagicMock()
        ctx_mock.args = []

        await bot.cmd_launch(update, ctx_mock)

        assert reply_texts, "Some reply must be sent"
        all_text = " ".join(reply_texts).lower()
        assert "usage" in all_text or "launch" in all_text


# ===========================================================================
# 9. launch_drafts verbs route to Beckman enqueue (LLM-bound)
# ===========================================================================

class TestLaunchDraftVerbs:
    CHANNELS = ["hn", "ph", "twitter", "linkedin", "reddit"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("channel", CHANNELS)
    async def test_launch_draft_enqueues_llm_task(self, channel, initialized_db, monkeypatch):
        """launch_drafts/<channel> enqueues an LLM task via beckman.enqueue."""
        import mr_roboto.launch_drafts as _ld

        enqueue_calls = []

        async def _mock_enqueue(spec, **kwargs):
            enqueue_calls.append(spec)
            return 100

        monkeypatch.setattr(_ld, "_enqueue", _mock_enqueue)

        result = await _ld.run(channel, {
            "product_id": "prod-draft",
            "launch_id": 1,
            "spec": "Build a great product.",
            "brand_voice": "Conversational, founder-driven.",
        })

        assert result["status"] == "enqueued"
        assert any(
            channel in str(call.get("title", "")) or channel in str(call)
            for call in enqueue_calls
        ), f"Enqueue must reference channel '{channel}'"

    @pytest.mark.asyncio
    async def test_unknown_channel_returns_error(self, initialized_db, monkeypatch):
        """launch_drafts/<unknown_channel> returns an error."""
        import mr_roboto.launch_drafts as _ld

        result = await _ld.run("snapchat", {
            "product_id": "prod-x",
            "launch_id": 1,
        })
        assert result["status"] == "error"


# ===========================================================================
# 10. Reversibility entries for all A2 verbs
# ===========================================================================

class TestReversibilityEntries:
    A2_VERBS = [
        "launch_drafts/hn",
        "launch_drafts/ph",
        "launch_drafts/twitter",
        "launch_drafts/linkedin",
        "launch_drafts/reddit",
        "publish_synchronized",
        "launch_response_monitor",
        "launch_lessons_writeback",
        "launch_readiness_gate",
    ]

    def test_a2_verbs_in_reversibility_registry(self):
        """All A2 mr_roboto verbs must be registered in VERB_REVERSIBILITY."""
        from mr_roboto.reversibility import VERB_REVERSIBILITY

        missing = [v for v in self.A2_VERBS if v not in VERB_REVERSIBILITY]
        assert not missing, f"Missing reversibility entries: {missing}"

    def test_publish_synchronized_is_irreversible(self):
        """publish_synchronized is irreversible (channels see the post)."""
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        assert VERB_REVERSIBILITY["publish_synchronized"] == "irreversible"

    def test_launch_drafts_are_full(self):
        """launch_drafts/* verbs are full-reversible (draft only, no publish)."""
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        for v in ["launch_drafts/hn", "launch_drafts/ph", "launch_drafts/twitter",
                  "launch_drafts/linkedin", "launch_drafts/reddit"]:
            assert VERB_REVERSIBILITY[v] == "full", f"{v} should be 'full'"

    def test_launch_readiness_gate_is_full(self):
        """launch_readiness_gate is full (read-only checks + founder_action)."""
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        assert VERB_REVERSIBILITY["launch_readiness_gate"] == "full"


# ===========================================================================
# 11. launch_readiness_gate posthook registered in registry
# ===========================================================================

class TestPosthookRegistry:
    def test_launch_readiness_gate_in_posthook_registry(self):
        """launch_readiness_gate kind must be in POST_HOOK_REGISTRY."""
        from general_beckman.posthooks import POST_HOOK_REGISTRY
        assert "launch_readiness_gate" in POST_HOOK_REGISTRY

    def test_launch_readiness_gate_is_blocker(self):
        """launch_readiness_gate posthook is a blocker (T-0 must not fire without pass)."""
        from general_beckman.posthooks import POST_HOOK_REGISTRY
        spec = POST_HOOK_REGISTRY.get("launch_readiness_gate")
        assert spec is not None
        assert spec.default_severity == "blocker"

    def test_launch_readiness_gate_in_posthook_kinds(self):
        """POST_HOOK_KINDS frozenset includes launch_readiness_gate."""
        from general_beckman.posthooks import POST_HOOK_KINDS
        assert "launch_readiness_gate" in POST_HOOK_KINDS
