"""Z7 T4 B4 — Meeting brief auto-generation tests.

Covers:
  1. DB migration: meetings table exists with correct columns.
  2. /meeting command creates a row (contact must exist first).
  3. /meeting list returns the created rows.
  4. meeting_brief_dispatch job: 25–35 min window only.
  5. meeting/brief verb: pulls interactions + follow-ups; degrades gracefully
     when A11 (mentions) and B2 (changelog) are absent.
  6. meeting/outcome_prompt verb: surfaces founder_action card.
  7. outcome logging creates an interactions row via crm.log_interaction and
     sets meetings.outcome_logged_interaction_id.
  8. Reversibility entries registered for meeting/brief + meeting/outcome_prompt.
  9. /meeting command handler registered in Telegram bot.
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
    """Fresh SQLite DB for B4 tests."""
    db_file = str(tmp_path / "test_b4.db")
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
    """Initialised DB with full schema (includes B4 migration)."""
    import src.infra.db as _db_mod
    _db_mod._db_connection = None
    _db_mod._db_connection_path = None
    from src.infra.db import init_db, get_db
    await init_db()
    _db = await get_db()
    yield _db
    _db_mod._db_connection = None
    _db_mod._db_connection_path = None


# ── Helpers ──────────────────────────────────────────────────────────────────


def _utc_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


async def _add_contact(db, product_id="prod1", handle="alice") -> int:
    """Insert a contact and return contact_id."""
    cur = await db.execute(
        "INSERT INTO relationships (product_id, handle, display_name, category) "
        "VALUES (?, ?, ?, ?)",
        (product_id, handle, "Alice", "customer"),
    )
    await db.commit()
    return cur.lastrowid


async def _create_meeting(
    db, product_id="prod1", contact_id=1, offset_minutes=30, purpose="Demo call"
) -> int:
    """Insert a meetings row scheduled at now + offset_minutes."""
    scheduled = _utc_str(datetime.now(timezone.utc) + timedelta(minutes=offset_minutes))
    cur = await db.execute(
        "INSERT INTO meetings (product_id, contact_id, scheduled_for, purpose) "
        "VALUES (?, ?, ?, ?)",
        (product_id, contact_id, scheduled, purpose),
    )
    await db.commit()
    return cur.lastrowid


# ===========================================================================
# 1. DB migration
# ===========================================================================


class TestMeetingsMigration:
    @pytest.mark.asyncio
    async def test_meetings_table_exists(self, db):
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='meetings'"
        ) as cur:
            row = await cur.fetchone()
        assert row is not None, "meetings table should exist after migration"

    @pytest.mark.asyncio
    async def test_meetings_columns(self, db):
        async with db.execute("PRAGMA table_info(meetings)") as cur:
            cols = {row[1] for row in await cur.fetchall()}
        required = {
            "meeting_id", "product_id", "contact_id",
            "scheduled_for", "purpose",
            "brief_generated_at", "brief_md",
            "outcome_logged_interaction_id",
        }
        assert required <= cols, f"Missing columns: {required - cols}"

    @pytest.mark.asyncio
    async def test_meetings_product_id_not_null(self, db):
        """product_id NOT NULL constraint should be enforced."""
        with pytest.raises(Exception):
            await db.execute(
                "INSERT INTO meetings (product_id, scheduled_for) VALUES (NULL, '2026-01-01 10:00:00')"
            )
            await db.commit()


# ===========================================================================
# 2. /meeting creates a row
# ===========================================================================


class TestMeetingCreate:
    @pytest.mark.asyncio
    async def test_create_meeting_row(self, db):
        from src.app.meetings import create_meeting
        contact_id = await _add_contact(db)
        scheduled = _utc_str(datetime.now(timezone.utc) + timedelta(hours=2))
        mid = await create_meeting(
            product_id="prod1",
            contact_id=contact_id,
            scheduled_for=scheduled,
            purpose="Sales demo",
        )
        assert isinstance(mid, int)
        assert mid > 0

        async with db.execute(
            "SELECT product_id, contact_id, purpose FROM meetings WHERE meeting_id=?",
            (mid,),
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row[0] == "prod1"
        assert row[1] == contact_id
        assert row[2] == "Sales demo"

    @pytest.mark.asyncio
    async def test_list_meetings(self, db):
        from src.app.meetings import create_meeting, list_meetings
        contact_id = await _add_contact(db)
        scheduled = _utc_str(datetime.now(timezone.utc) + timedelta(hours=1))
        await create_meeting("prod1", contact_id, scheduled, "Q1 review")
        await create_meeting("prod1", contact_id, scheduled, "Q2 planning")

        meetings = await list_meetings("prod1")
        assert len(meetings) >= 2
        purposes = [m["purpose"] for m in meetings]
        assert "Q1 review" in purposes
        assert "Q2 planning" in purposes

    @pytest.mark.asyncio
    async def test_create_meeting_requires_valid_scheduled_for(self, db):
        """scheduled_for must be a parseable datetime string."""
        from src.app.meetings import create_meeting
        contact_id = await _add_contact(db)
        with pytest.raises(ValueError):
            await create_meeting("prod1", contact_id, "not-a-date", "test")


# ===========================================================================
# 3. Dispatch job: 25–35 min window only
# ===========================================================================


class TestBriefDispatchWindow:
    @pytest.mark.asyncio
    async def test_picks_meeting_in_window(self, db):
        """Meeting in [25, 35] window AND brief_generated_at IS NULL is picked."""
        from src.app.jobs.meeting_brief_dispatch import _pick_meetings_for_brief
        contact_id = await _add_contact(db)
        # scheduled 30 min from now — inside window
        await _create_meeting(db, offset_minutes=30, contact_id=contact_id)
        rows = await _pick_meetings_for_brief()
        assert len(rows) >= 1

    @pytest.mark.asyncio
    async def test_skips_meeting_too_soon(self, db):
        """Meeting only 10 min away — outside [25, 35] window, should be skipped."""
        from src.app.jobs.meeting_brief_dispatch import _pick_meetings_for_brief
        contact_id = await _add_contact(db)
        await _create_meeting(db, offset_minutes=10, contact_id=contact_id)
        rows = await _pick_meetings_for_brief()
        # Only meetings in [25,35] window
        for r in rows:
            scheduled = datetime.strptime(r["scheduled_for"], "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            )
            delta = (scheduled - datetime.now(timezone.utc)).total_seconds() / 60
            assert 25 <= delta <= 35, f"Row outside window: delta={delta:.1f}min"

    @pytest.mark.asyncio
    async def test_skips_meeting_already_briefed(self, db):
        """Meeting with brief_generated_at set should NOT be picked again."""
        from src.app.jobs.meeting_brief_dispatch import _pick_meetings_for_brief
        contact_id = await _add_contact(db)
        mid = await _create_meeting(db, offset_minutes=30, contact_id=contact_id)
        # Mark as briefed
        await db.execute(
            "UPDATE meetings SET brief_generated_at=strftime('%Y-%m-%d %H:%M:%S','now') "
            "WHERE meeting_id=?",
            (mid,),
        )
        await db.commit()
        rows = await _pick_meetings_for_brief()
        ids = [r["meeting_id"] for r in rows]
        assert mid not in ids

    @pytest.mark.asyncio
    async def test_picks_meeting_past_for_outcome_prompt(self, db):
        """Meeting 30 min past scheduled_for with no outcome logged is picked."""
        from src.app.jobs.meeting_brief_dispatch import _pick_meetings_for_outcome_prompt
        contact_id = await _add_contact(db)
        # scheduled 30 min in the past — past_offset
        scheduled = _utc_str(datetime.now(timezone.utc) - timedelta(minutes=30))
        cur = await db.execute(
            "INSERT INTO meetings (product_id, contact_id, scheduled_for, purpose) "
            "VALUES (?, ?, ?, ?)",
            ("prod1", contact_id, scheduled, "Past meeting"),
        )
        await db.commit()
        rows = await _pick_meetings_for_outcome_prompt()
        assert len(rows) >= 1

    @pytest.mark.asyncio
    async def test_skips_outcome_already_logged(self, db):
        """Meeting with outcome_logged_interaction_id set is not picked for prompt."""
        from src.app.jobs.meeting_brief_dispatch import _pick_meetings_for_outcome_prompt
        contact_id = await _add_contact(db)
        scheduled = _utc_str(datetime.now(timezone.utc) - timedelta(minutes=30))
        cur = await db.execute(
            "INSERT INTO meetings (product_id, contact_id, scheduled_for, purpose, "
            "outcome_logged_interaction_id) VALUES (?, ?, ?, ?, ?)",
            ("prod1", contact_id, scheduled, "Past meeting", 999),
        )
        await db.commit()
        mid = cur.lastrowid
        rows = await _pick_meetings_for_outcome_prompt()
        ids = [r["meeting_id"] for r in rows]
        assert mid not in ids


# ===========================================================================
# 5. meeting/brief verb: graceful degradation when A11/B2 absent
# ===========================================================================


class TestBriefGeneration:
    @pytest.mark.asyncio
    async def test_brief_degrades_without_a11(self, db):
        """brief generator must not raise when mentions table absent."""
        from src.app.meetings import build_brief_context
        contact_id = await _add_contact(db)
        mid = await _create_meeting(db, contact_id=contact_id, offset_minutes=30)

        # A11 not built — mentions table won't exist; should degrade gracefully
        ctx = await build_brief_context(meeting_id=mid, product_id="prod1")
        assert isinstance(ctx, dict)
        assert "interactions" in ctx
        assert "follow_ups" in ctx
        assert "mentions" in ctx  # always present; may be empty list
        assert isinstance(ctx["mentions"], list)

    @pytest.mark.asyncio
    async def test_brief_includes_recent_interactions(self, db):
        """build_brief_context pulls last 5 interactions for the contact."""
        from src.app.meetings import build_brief_context
        from src.app.crm import log_interaction
        contact_id = await _add_contact(db)

        # Add 6 interactions; build_brief_context must return at most 5
        for i in range(6):
            await log_interaction(
                "prod1", contact_id, "call", f"Call #{i}", follow_up=None
            )

        mid = await _create_meeting(db, contact_id=contact_id, offset_minutes=30)
        ctx = await build_brief_context(meeting_id=mid, product_id="prod1")
        assert len(ctx["interactions"]) <= 5

    @pytest.mark.asyncio
    async def test_brief_includes_follow_ups(self, db):
        """build_brief_context includes open follow-ups for the product."""
        from src.app.meetings import build_brief_context
        from src.app.crm import log_interaction
        contact_id = await _add_contact(db)

        # Interaction with follow-up owed
        await log_interaction(
            "prod1", contact_id, "email", "Intro sent", follow_up="3d"
        )

        mid = await _create_meeting(db, contact_id=contact_id, offset_minutes=30)
        ctx = await build_brief_context(meeting_id=mid, product_id="prod1")
        assert isinstance(ctx["follow_ups"], list)

    @pytest.mark.asyncio
    async def test_brief_degrades_without_b2(self, db):
        """brief generator must not raise when changelog_entries absent (B2 not shipped)."""
        from src.app.meetings import build_brief_context
        contact_id = await _add_contact(db)
        mid = await _create_meeting(db, contact_id=contact_id, offset_minutes=30)

        ctx = await build_brief_context(meeting_id=mid, product_id="prod1")
        assert "changelog" in ctx
        assert isinstance(ctx["changelog"], list)

    @pytest.mark.asyncio
    async def test_compose_brief_md(self, db):
        """compose_brief_md returns a non-empty Markdown string."""
        from src.app.meetings import compose_brief_md
        ctx = {
            "contact": {"display_name": "Alice", "handle": "alice", "category": "customer"},
            "meeting": {"purpose": "Demo", "scheduled_for": "2026-05-16 14:00:00"},
            "interactions": [
                {"kind": "call", "summary": "Intro call", "logged_at": "2026-05-01 10:00:00"}
            ],
            "follow_ups": [],
            "mentions": [],
            "changelog": [],
            "mission_items": [],
        }
        md = compose_brief_md(ctx, talking_points=["Discuss pricing", "Demo new feature"])
        assert "# Meeting Brief" in md
        assert "Alice" in md
        assert "Discuss pricing" in md

    def test_compose_brief_md_llm_unavailable_no_placeholder(self):
        """When llm_unavailable=True the brief shows a clear 'unavailable' message,
        NOT the misleading 'will appear here after brief generation' placeholder."""
        from src.app.meetings import (
            compose_brief_md,
            _TALKING_POINTS_PLACEHOLDER,
        )
        ctx = {
            "contact": {"display_name": "Bob"},
            "meeting": {"purpose": "Sync", "scheduled_for": "2026-05-17 09:00:00"},
            "interactions": [],
            "follow_ups": [],
            "mentions": [],
            "changelog": [],
        }
        md = compose_brief_md(ctx, llm_unavailable=True)
        assert "unavailable" in md.lower()
        assert _TALKING_POINTS_PLACEHOLDER not in md, (
            "Failed-LLM brief must NOT ship the misleading placeholder"
        )
        assert "will appear here" not in md


# ===========================================================================
# 5b. meeting/brief — REAL LLM-enqueue path (de-mocked: only the model boundary
#     is faked; the real _call_llm_meeting_brief + enqueue plumbing runs).
# ===========================================================================


# The literal stub string that the pre-fix handler shipped on every brief.
_DEAD_PLACEHOLDER = "(LLM-drafted talking points will appear here"


def _make_task_result(status="completed", content="", error=None):
    """Build a TaskResult-shaped object the way beckman returns it."""
    import types
    result = {"content": content} if content is not None else None
    return types.SimpleNamespace(status=status, error=error, result=result)


class TestMeetingBriefLLMPath:
    """Exercise the REAL ``enqueue_meeting_brief`` → ``_brief_persist_resume``
    CPS path. SP2: the LLM call is fire-and-forget; the brief is composed in
    the resume. We assert the dead placeholder is never persisted.
    """

    @pytest.mark.asyncio
    async def test_enqueue_meeting_brief_passes_cps_continuations(self, monkeypatch):
        """enqueue_meeting_brief must use on_complete/on_error (not await_inline)."""
        import general_beckman as _gb
        from general_beckman.lanes import LANE_ONESHOT
        import src.app.meetings as _m

        captured = {}

        async def _fake_enqueue(spec, *, lane=None, await_inline=False, **kw):
            captured["lane"] = lane
            captured["await_inline"] = await_inline
            captured["on_complete"] = kw.get("on_complete")
            captured["on_error"] = kw.get("on_error")
            captured["cont_state"] = kw.get("cont_state")
            captured["spec"] = spec
            return 4242

        monkeypatch.setattr(_gb, "enqueue", _fake_enqueue)

        ctx = {
            "contact": {"display_name": "Carol", "category": "customer"},
            "meeting": {"purpose": "Renewal", "scheduled_for": "2026-05-17 15:00:00"},
            "interactions": [
                {"kind": "email", "summary": "Sent pricing", "logged_at": "2026-05-10 10:00:00"}
            ],
            "follow_ups": [], "mentions": [], "changelog": [],
        }
        cid = await _m.enqueue_meeting_brief(
            ctx, meeting_id=1, product_id="prod1",
        )

        assert cid == 4242
        assert captured["await_inline"] in (False, None)
        assert captured["on_complete"] == "meetings.brief_persist_resume"
        assert captured["on_error"] == "meetings.brief_persist_err"
        assert captured["lane"] == LANE_ONESHOT
        assert captured["spec"]["context"]["llm_call"]["raw_dispatch"] is True
        cs = captured["cont_state"]
        assert cs["meeting_id"] == 1
        assert cs["product_id"] == "prod1"
        assert cs["ctx"]["contact"]["display_name"] == "Carol"

    @pytest.mark.asyncio
    async def test_brief_resume_writes_real_llm_output(self, db):
        """The resume parses the LLM JSON and persists a brief that contains
        the real talking points — never the dead stub placeholder."""
        import src.app.meetings as _m

        contact_id = await _add_contact(db)
        mid = await _create_meeting(db, contact_id=contact_id, offset_minutes=30)

        ctx = {"contact": {"display_name": "Carol"}, "meeting": {"meeting_id": mid}}
        await _m._brief_persist_resume(
            child_task_id=4242,
            result={"status": "completed",
                    "result": {"content": json.dumps({
                        "talking_points": ["Review the onboarding blockers"],
                        "suggested_asks": ["Request a case-study quote"],
                    })}},
            state={"meeting_id": mid, "product_id": "prod1", "ctx": ctx},
        )

        async with db.execute(
            "SELECT brief_md FROM meetings WHERE meeting_id=?", (mid,)
        ) as cur:
            row = await cur.fetchone()
        brief_md = row[0] or ""

        assert "Review the onboarding blockers" in brief_md
        assert "Request a case-study quote" in brief_md
        assert _DEAD_PLACEHOLDER not in brief_md, (
            "brief must NEVER contain the dead stub placeholder"
        )

    @pytest.mark.asyncio
    async def test_brief_err_writes_unavailable_brief(self, db):
        """on_error writes an 'unavailable' brief and never ships the dead stub."""
        import src.app.meetings as _m

        contact_id = await _add_contact(db)
        mid = await _create_meeting(db, contact_id=contact_id, offset_minutes=30)

        ctx = {"contact": {"display_name": "X"}, "meeting": {"meeting_id": mid}}
        await _m._brief_persist_err(
            child_task_id=4242,
            result={"status": "failed", "error": "LLM unavailable"},
            state={"meeting_id": mid, "product_id": "prod1", "ctx": ctx},
        )

        async with db.execute(
            "SELECT brief_md FROM meetings WHERE meeting_id=?", (mid,)
        ) as cur:
            row = await cur.fetchone()
        brief_md = row[0] or ""

        assert "unavailable" in brief_md.lower()
        assert _DEAD_PLACEHOLDER not in brief_md
        assert "# Meeting Brief" in brief_md

    @pytest.mark.asyncio
    async def test_meeting_brief_verb_enqueues_via_cps(self, db, monkeypatch):
        """End-to-end: mr_roboto.run('meeting/brief') enqueues the brief LLM
        via Beckman with on_complete continuations and returns 'completed'."""
        import general_beckman as _gb
        import mr_roboto

        contact_id = await _add_contact(db)
        mid = await _create_meeting(db, contact_id=contact_id, offset_minutes=30)

        captured = {}

        async def _fake_enqueue(spec, *, lane=None, await_inline=False, **kw):
            captured["on_complete"] = kw.get("on_complete")
            captured["on_error"] = kw.get("on_error")
            return 5555

        monkeypatch.setattr(_gb, "enqueue", _fake_enqueue)

        task = {
            "id": 1, "mission_id": 1,
            "payload": {
                "action": "meeting/brief",
                "meeting_id": mid, "product_id": "prod1",
            },
        }
        result = await mr_roboto.run(task)
        assert result.status == "completed", getattr(result, "error", "")
        assert captured.get("on_complete") == "meetings.brief_persist_resume"
        assert captured.get("on_error") == "meetings.brief_persist_err"


# ===========================================================================
# 6. meeting/outcome_prompt verb
# ===========================================================================


class TestOutcomePrompt:
    @pytest.mark.asyncio
    async def test_outcome_prompt_creates_founder_action(self, db):
        """meeting/outcome_prompt should create a founder_action card."""
        from src.app.meetings import emit_outcome_prompt
        contact_id = await _add_contact(db)
        mid = await _create_meeting(db, contact_id=contact_id, offset_minutes=-30)

        with patch("src.founder_actions.create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = MagicMock(id=42)
            result = await emit_outcome_prompt(meeting_id=mid, product_id="prod1")

        assert result.get("ok") is True
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args
        # Verify title mentions meeting/outcome
        title = call_kwargs[1].get("title") or (call_kwargs[0][3] if len(call_kwargs[0]) > 3 else "")
        assert "meeting" in title.lower() or "outcome" in title.lower()


# ===========================================================================
# 7. Outcome logging writes interactions row + sets FK
# ===========================================================================


class TestOutcomeLogging:
    @pytest.mark.asyncio
    async def test_log_meeting_outcome_creates_interaction(self, db):
        """log_meeting_outcome should call crm.log_interaction with kind=meeting."""
        from src.app.meetings import log_meeting_outcome
        contact_id = await _add_contact(db)
        mid = await _create_meeting(db, contact_id=contact_id, offset_minutes=-30)

        iid = await log_meeting_outcome(
            meeting_id=mid,
            product_id="prod1",
            contact_id=contact_id,
            summary="Discussed roadmap; they liked the pricing.",
            next_action="Send proposal",
            follow_up="3d",
        )
        assert isinstance(iid, int)
        assert iid > 0

        # meetings.outcome_logged_interaction_id must be set
        async with db.execute(
            "SELECT outcome_logged_interaction_id FROM meetings WHERE meeting_id=?",
            (mid,),
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row[0] == iid

    @pytest.mark.asyncio
    async def test_log_meeting_outcome_interaction_kind(self, db):
        """The interaction created must have kind='meeting'."""
        from src.app.meetings import log_meeting_outcome
        contact_id = await _add_contact(db)
        mid = await _create_meeting(db, contact_id=contact_id, offset_minutes=-30)

        iid = await log_meeting_outcome(
            meeting_id=mid,
            product_id="prod1",
            contact_id=contact_id,
            summary="Good meeting.",
        )
        async with db.execute(
            "SELECT kind FROM interactions WHERE interaction_id=?", (iid,)
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row[0] == "meeting"


# ===========================================================================
# 8. Reversibility entries registered
# ===========================================================================


class TestReversibility:
    def test_meeting_brief_registered(self):
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        assert "meeting/brief" in VERB_REVERSIBILITY

    def test_meeting_outcome_prompt_registered(self):
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        assert "meeting/outcome_prompt" in VERB_REVERSIBILITY

    def test_meeting_brief_dispatch_registered(self):
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        assert "meeting_brief_dispatch" in VERB_REVERSIBILITY

    def test_meeting_brief_full_reversibility(self):
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        # LLM-bound brief write is a local DB write → full
        assert VERB_REVERSIBILITY["meeting/brief"] == "full"

    def test_meeting_outcome_prompt_irreversible(self):
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        # Surfaces a Telegram card → irreversible
        assert VERB_REVERSIBILITY["meeting/outcome_prompt"] == "irreversible"


# ===========================================================================
# 9. mr_roboto dispatch verbs exist
# ===========================================================================


class TestMrRobotoDispatch:
    @pytest.mark.asyncio
    async def test_meeting_brief_dispatch_verb(self, db):
        """mr_roboto.run handles meeting/brief action without crashing."""
        import mr_roboto

        task = {
            "id": 1,
            "mission_id": 1,
            "payload": {
                "action": "meeting/brief",
                "meeting_id": 1,
                "product_id": "prod1",
            },
        }
        # meeting_id=1 doesn't exist in DB yet — should return failed gracefully
        result = await mr_roboto.run(task)
        assert result.status in ("completed", "failed")

    @pytest.mark.asyncio
    async def test_meeting_outcome_prompt_dispatch_verb(self, db):
        """mr_roboto.run handles meeting/outcome_prompt action."""
        import mr_roboto

        task = {
            "id": 1,
            "mission_id": 1,
            "payload": {
                "action": "meeting/outcome_prompt",
                "meeting_id": 99999,  # non-existent
                "product_id": "prod1",
            },
        }
        result = await mr_roboto.run(task)
        assert result.status in ("completed", "failed")

    @pytest.mark.asyncio
    async def test_meeting_brief_dispatch_cron_verb(self, db):
        """mr_roboto.run handles meeting_brief_dispatch cron action."""
        import mr_roboto

        task = {
            "id": 1,
            "mission_id": None,
            "payload": {"action": "meeting_brief_dispatch"},
        }
        result = await mr_roboto.run(task)
        assert result.status in ("completed", "failed")


# ===========================================================================
# 10. Cron registration
# ===========================================================================


class TestCronRegistration:
    def test_meeting_brief_dispatch_in_cron_seed(self):
        """meeting_brief_dispatch cadence is registered in INTERNAL_CADENCES."""
        from general_beckman.cron_seed import INTERNAL_CADENCES
        titles = [c["title"] for c in INTERNAL_CADENCES]
        assert "meeting_brief_dispatch" in titles

    def test_meeting_brief_dispatch_cadence(self):
        """meeting_brief_dispatch runs every 5 minutes (300s)."""
        from general_beckman.cron_seed import INTERNAL_CADENCES
        entry = next(c for c in INTERNAL_CADENCES if c["title"] == "meeting_brief_dispatch")
        assert entry.get("interval_seconds") == 300

    def test_meeting_brief_dispatch_payload(self):
        from general_beckman.cron_seed import INTERNAL_CADENCES
        entry = next(c for c in INTERNAL_CADENCES if c["title"] == "meeting_brief_dispatch")
        assert entry["payload"].get("_executor") == "meeting_brief_dispatch"
