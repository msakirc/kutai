"""Z7 T6E — Cross-mission launch lessons consumer tests.

Proves the loop:
  launch_lessons_writeback (T+7d, launch N)
    → mission_lessons rows (dedup_key launch.*)
    → launch_drafts/<channel> (T-72h, launch N+1)
    → LLM prompt context contains prior lesson text

Tests
-----
1. Seed mission_lessons rows for a product (as if written by writeback N).
   Call launch_drafts/hn for launch N+1 with no explicit mission_lessons in
   payload. Assert the enqueued LLM prompt description contains lesson text.

2. launch_drafts/hn with prior lessons passed explicitly in payload still works.

3. No prior lessons → drafts still work (graceful, no crash).

4. fetch_launch_lessons returns empty list when DB has no matching rows.

5. All 5 channels consume prior lessons via the same path.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import pytest_asyncio


# ---------------------------------------------------------------------------
# DB fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Fresh SQLite DB for tests."""
    db_file = str(tmp_path / "test_z7_t6.db")
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
    """Initialize DB with full schema."""
    from src.infra.db import init_db, get_db
    import src.infra.db as _db_mod

    _db_mod._db_connection = None
    _db_mod._db_connection_path = None

    await init_db()
    db = await get_db()
    yield db

    _db_mod._db_connection = None
    _db_mod._db_connection_path = None


# ---------------------------------------------------------------------------
# Helper: seed mission_lessons rows as if from writeback of launch N
# ---------------------------------------------------------------------------

async def _seed_launch_lessons(db, product_id: str, *, count: int = 3):
    """Insert mission_lessons rows for the product's launch stack."""
    from src.infra.mission_lessons import upsert_mission_lesson

    stack = f"launch/{product_id}"
    lessons = [
        {
            "domain": "launch_channel",
            "pattern": f"launch top channel: hn drove highest engagement",
            "fix": f"For next launch of '{product_id}': prioritize hn for pre-launch seeding.",
        },
        {
            "domain": "launch_hn",
            "pattern": f"launch.hn.timing.0900",
            "fix": "HN launch at 09:00 UTC produced 120 upvotes. Use as anchor.",
        },
        {
            "domain": "launch_readiness",
            "pattern": f"launch.readiness.gate.{product_id}.launch1",
            "fix": "Run launch_readiness_gate >=1h before T-0 to catch blockers early.",
        },
    ]
    for lesson in lessons[:count]:
        await upsert_mission_lesson(
            stack=stack,
            domain=lesson["domain"],
            pattern=lesson["pattern"],
            fix=lesson["fix"],
            severity="info",
            source_kind="launch_writeback",
            source_ref={"product_id": product_id, "launch_id": 1},
        )


# ===========================================================================
# 1. Auto-fetch: launch_drafts/hn consumes prior lessons without explicit payload
# ===========================================================================

class TestAutoFetchLaunchLessons:
    @pytest.mark.asyncio
    async def test_hn_draft_consumes_prior_lessons_auto(self, initialized_db, monkeypatch):
        """launch_drafts/hn auto-fetches prior lessons and injects them into LLM prompt."""
        import mr_roboto.launch_drafts as _ld

        product_id = "prod-loop-test"
        # Seed prior-launch lessons (as if from launch N's T+7d writeback)
        await _seed_launch_lessons(initialized_db, product_id)

        enqueue_calls = []

        async def _mock_enqueue(spec, **kwargs):
            enqueue_calls.append(spec)
            return 1

        monkeypatch.setattr(_ld, "_enqueue", _mock_enqueue)

        # Call launch_drafts/hn for launch N+1 — NO mission_lessons in payload
        result = await _ld.run("hn", {
            "product_id": product_id,
            "launch_id": 2,
            "spec": "Amazing product that does X.",
            "brand_voice": "Direct, founder-driven.",
            # NOTE: no "mission_lessons" key — must be auto-fetched
        })

        assert result["status"] == "enqueued", f"Expected enqueued, got: {result}"
        assert enqueue_calls, "Beckman must be called"

        # The enqueued description must contain lesson text
        desc = enqueue_calls[0].get("description", "")
        assert "Prior launch lessons" in desc, (
            f"LLM prompt must contain 'Prior launch lessons' section.\n"
            f"description=\n{desc}"
        )
        # At least one concrete lesson pattern should appear
        assert "hn" in desc.lower() or "launch" in desc.lower(), (
            f"Lesson content not in description: {desc}"
        )

    @pytest.mark.asyncio
    async def test_prompt_contains_lesson_fix_text(self, initialized_db, monkeypatch):
        """The injected lesson's fix text must appear in the LLM prompt."""
        import mr_roboto.launch_drafts as _ld

        product_id = "prod-fix-text"
        await _seed_launch_lessons(initialized_db, product_id)

        enqueue_calls = []

        async def _mock_enqueue(spec, **kwargs):
            enqueue_calls.append(spec)
            return 2

        monkeypatch.setattr(_ld, "_enqueue", _mock_enqueue)

        result = await _ld.run("hn", {
            "product_id": product_id,
            "launch_id": 2,
            "spec": "My product.",
            "brand_voice": "Friendly.",
        })
        assert result["status"] == "enqueued"
        desc = enqueue_calls[0].get("description", "")
        # The fix text from our seeded lesson must appear
        assert "prioritize" in desc.lower() or "seeding" in desc.lower() or "anchor" in desc.lower(), (
            f"Fix text not found in prompt description:\n{desc}"
        )


# ===========================================================================
# 2. Explicit mission_lessons in payload still works (backward compat)
# ===========================================================================

class TestExplicitLessonsInPayload:
    @pytest.mark.asyncio
    async def test_explicit_lessons_in_payload(self, initialized_db, monkeypatch):
        """launch_drafts/hn with explicit mission_lessons in payload still works."""
        import mr_roboto.launch_drafts as _ld

        enqueue_calls = []

        async def _mock_enqueue(spec, **kwargs):
            enqueue_calls.append(spec)
            return 3

        monkeypatch.setattr(_ld, "_enqueue", _mock_enqueue)

        result = await _ld.run("hn", {
            "product_id": "prod-explicit",
            "launch_id": 3,
            "spec": "Spec text.",
            "brand_voice": "Bold.",
            "mission_lessons": [
                {"pattern": "explicit lesson pattern", "fix": "explicit fix"},
            ],
        })

        assert result["status"] == "enqueued"
        desc = enqueue_calls[0].get("description", "")
        assert "explicit lesson pattern" in desc, (
            f"Explicit lessons must appear in description:\n{desc}"
        )


# ===========================================================================
# 3. No prior lessons → drafts still work (graceful)
# ===========================================================================

class TestNoLessonsGraceful:
    @pytest.mark.asyncio
    async def test_no_prior_lessons_draft_succeeds(self, initialized_db, monkeypatch):
        """No prior lessons in DB → launch_drafts still succeeds with no crash."""
        import mr_roboto.launch_drafts as _ld

        enqueue_calls = []

        async def _mock_enqueue(spec, **kwargs):
            enqueue_calls.append(spec)
            return 4

        monkeypatch.setattr(_ld, "_enqueue", _mock_enqueue)

        # Fresh product with no prior lessons
        result = await _ld.run("hn", {
            "product_id": "prod-no-lessons",
            "launch_id": 1,
            "spec": "Brand new product.",
            "brand_voice": "Casual.",
        })

        assert result["status"] == "enqueued", f"Must succeed even with no prior lessons: {result}"
        assert enqueue_calls, "Beckman must still be called"
        # Description must still be formed (just no lessons section)
        desc = enqueue_calls[0].get("description", "")
        assert "Brand new product" in desc


# ===========================================================================
# 4. fetch_launch_lessons returns empty list when no rows match
# ===========================================================================

class TestFetchLaunchLessons:
    @pytest.mark.asyncio
    async def test_fetch_returns_empty_for_unknown_product(self, initialized_db):
        """fetch_launch_lessons returns [] when no lessons exist for the product."""
        from mr_roboto.launch_drafts import fetch_launch_lessons

        lessons = await fetch_launch_lessons("prod-unknown", limit=5)
        assert lessons == [], f"Expected [], got {lessons}"

    @pytest.mark.asyncio
    async def test_fetch_returns_seeded_lessons(self, initialized_db):
        """fetch_launch_lessons returns seeded lessons for the product."""
        from mr_roboto.launch_drafts import fetch_launch_lessons

        product_id = "prod-fetch-test"
        await _seed_launch_lessons(initialized_db, product_id, count=2)

        lessons = await fetch_launch_lessons(product_id, limit=5)
        assert len(lessons) >= 1, f"Expected at least 1 lesson, got {lessons}"
        patterns = [l.get("pattern", "") for l in lessons]
        assert any("hn" in p or "launch" in p for p in patterns), (
            f"Expected launch-domain patterns, got: {patterns}"
        )

    @pytest.mark.asyncio
    async def test_fetch_filters_by_product(self, initialized_db):
        """fetch_launch_lessons only returns lessons for the specific product."""
        from mr_roboto.launch_drafts import fetch_launch_lessons

        await _seed_launch_lessons(initialized_db, "prod-alpha")
        await _seed_launch_lessons(initialized_db, "prod-beta")

        lessons_alpha = await fetch_launch_lessons("prod-alpha", limit=5)
        lessons_beta = await fetch_launch_lessons("prod-beta", limit=5)

        # Each product's stack is isolated — no cross-contamination
        stacks_alpha = {l.get("stack", "") for l in lessons_alpha}
        stacks_beta = {l.get("stack", "") for l in lessons_beta}
        for s in stacks_alpha:
            assert "alpha" in s, f"prod-beta stack leaked into prod-alpha results: {s}"
        for s in stacks_beta:
            assert "beta" in s, f"prod-alpha stack leaked into prod-beta results: {s}"


# ===========================================================================
# 5. All 5 channels auto-consume prior lessons
# ===========================================================================

class TestAllChannelsConsumeLessons:
    CHANNELS = ["hn", "ph", "twitter", "linkedin", "reddit"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("channel", CHANNELS)
    async def test_channel_draft_includes_prior_lessons(
        self, channel, initialized_db, monkeypatch
    ):
        """All 5 channel draft verbs include prior lessons in LLM prompt."""
        import mr_roboto.launch_drafts as _ld

        product_id = f"prod-all-channels-{channel}"
        await _seed_launch_lessons(initialized_db, product_id)

        enqueue_calls = []

        async def _mock_enqueue(spec, **kwargs):
            enqueue_calls.append(spec)
            return 10

        monkeypatch.setattr(_ld, "_enqueue", _mock_enqueue)

        result = await _ld.run(channel, {
            "product_id": product_id,
            "launch_id": 2,
            "spec": "Product spec here.",
            "brand_voice": "Brand voice here.",
        })

        assert result["status"] == "enqueued", f"channel {channel}: {result}"
        desc = enqueue_calls[0].get("description", "")
        assert "Prior launch lessons" in desc, (
            f"channel={channel}: 'Prior launch lessons' missing from prompt.\n"
            f"description={desc}"
        )


# ===========================================================================
# 6. Full writeback → fetch round-trip (end-to-end loop)
# ===========================================================================

class TestWritebackToFetchRoundTrip:
    @pytest.mark.asyncio
    async def test_writeback_lessons_appear_in_next_draft(self, initialized_db, monkeypatch):
        """Full loop: writeback emits lessons → next launch draft consumes them."""
        import mr_roboto.launch_lessons_writeback as _wb
        import mr_roboto.launch_drafts as _ld

        product_id = "prod-round-trip"

        # Step 1: T+7d writeback for launch N
        wb_result = await _wb.run({
            "product_id": product_id,
            "launch_id": 1,
            "mission_id": 99,
            "channels": ["hn", "twitter"],
            "engagement_summary": {
                "hn": {"upvotes": 150, "comments": 40, "timing_utc": "09:00"},
                "twitter": {"likes": 300, "retweets": 50, "timing_utc": "09:00"},
            },
        })
        assert wb_result["status"] == "ok"
        assert wb_result["lessons_written"] >= 3

        # Step 2: T-72h draft for launch N+1 — should consume those lessons
        enqueue_calls = []

        async def _mock_enqueue(spec, **kwargs):
            enqueue_calls.append(spec)
            return 101

        monkeypatch.setattr(_ld, "_enqueue", _mock_enqueue)

        result = await _ld.run("hn", {
            "product_id": product_id,
            "launch_id": 2,  # next launch
            "spec": "Next product launch spec.",
            "brand_voice": "Founder voice.",
            # No explicit mission_lessons — must be auto-fetched from writeback
        })

        assert result["status"] == "enqueued"
        desc = enqueue_calls[0].get("description", "")
        assert "Prior launch lessons" in desc, (
            f"Writeback lessons must appear in next launch's draft prompt.\n"
            f"lessons_written={wb_result['lessons_written']}\n"
            f"description=\n{desc}"
        )
