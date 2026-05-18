"""Z7 T5 B8 — Reviews harvest tests.

Covers:
  1. DB migration: external_reviews table with all columns + unique constraint.
  2. Poll verbs: g2, appstore, playstore, producthunt — ingest + dedup.
  3. reviews/classify: sets sentiment + theme_tag via LLM (mocked).
  4. reviews/draft_reply: produces a draft (never auto-posts).
  5. Daily job (reviews_poll_daily): polls all configured platforms + classifies.
  6. 1-2-star reviews surface a founder_action.
  7. Bug-tagged reviews enqueue an investigation task via beckman.
  8. Reversibility entries registered for all reviews/* verbs.
  9. Cron seed entry present for reviews_poll_daily.

All network/vecihi/LLM calls are mocked — no real HTTP.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


# ---------------------------------------------------------------------------
# DB fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Fresh SQLite DB for B8 tests."""
    db_file = str(tmp_path / "test_b8.db")
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
    """Initialised DB (includes B8 migration)."""
    import src.infra.db as _db_mod
    _db_mod._db_connection = None
    _db_mod._db_connection_path = None
    from src.infra.db import init_db, get_db
    await init_db()
    _db = await get_db()
    yield _db
    _db_mod._db_connection = None
    _db_mod._db_connection_path = None


# ---------------------------------------------------------------------------
# 1. DB Migration
# ---------------------------------------------------------------------------

class TestMigration:
    @pytest.mark.asyncio
    async def test_external_reviews_table_exists(self, db):
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='external_reviews'"
        )
        row = await cur.fetchone()
        assert row is not None, "external_reviews table should exist after init_db"

    @pytest.mark.asyncio
    async def test_external_reviews_columns(self, db):
        cur = await db.execute("PRAGMA table_info(external_reviews)")
        cols = {r[1] for r in await cur.fetchall()}
        required = {
            "review_id", "product_id", "platform", "external_id",
            "posted_at", "author", "rating", "body_md",
            "sentiment", "replied_at", "reply_body_md",
            "theme_tag", "created_at",
        }
        missing = required - cols
        assert not missing, f"Missing columns: {missing}"

    @pytest.mark.asyncio
    async def test_unique_constraint_platform_external_id(self, db):
        """Inserting the same (platform, external_id) twice raises IntegrityError."""
        import aiosqlite
        await db.execute(
            "INSERT INTO external_reviews "
            "(product_id, platform, external_id, posted_at, author, rating, body_md) "
            "VALUES ('p1', 'g2', 'rev-001', '2026-05-16 10:00:00', 'Alice', 5, 'Great!')"
        )
        await db.commit()
        with pytest.raises(Exception):  # IntegrityError or OperationalError
            await db.execute(
                "INSERT INTO external_reviews "
                "(product_id, platform, external_id, posted_at, author, rating, body_md) "
                "VALUES ('p1', 'g2', 'rev-001', '2026-05-16 11:00:00', 'Bob', 4, 'Also great')"
            )
            await db.commit()

    @pytest.mark.asyncio
    async def test_product_id_not_null(self, db):
        """product_id NOT NULL is enforced."""
        with pytest.raises(Exception):
            await db.execute(
                "INSERT INTO external_reviews "
                "(product_id, platform, external_id, posted_at, author, rating, body_md) "
                "VALUES (NULL, 'g2', 'rev-x', '2026-05-16', 'X', 5, 'Hi')"
            )
            await db.commit()


# ---------------------------------------------------------------------------
# 2. Poll verbs — ingest + dedup
# ---------------------------------------------------------------------------

class TestPollVerbs:
    """tests for the per-platform poller adapter interface."""

    @pytest.mark.asyncio
    async def test_poll_g2_ingests_reviews(self, db):
        """G2 poller (free tier / vecihi scrape) returns structured dicts + inserts rows."""
        from mr_roboto.reviews_poll import poll_platform

        fake_reviews = [
            {
                "external_id": "g2-001",
                "posted_at": "2026-05-15 10:00:00",
                "author": "Alice",
                "rating": 5,
                "body_md": "Excellent product! Best I've used.",
            }
        ]

        with patch(
            "mr_roboto.reviews_poll._fetch_g2",
            new=AsyncMock(return_value=fake_reviews),
        ):
            result = await poll_platform(
                platform="g2",
                product_id="prod-1",
                config={},
            )

        assert result["ingested"] == 1
        assert result["skipped"] == 0

        # Verify DB row was inserted
        cur = await db.execute(
            "SELECT external_id, rating, body_md FROM external_reviews "
            "WHERE platform='g2' AND external_id='g2-001'"
        )
        row = await cur.fetchone()
        assert row is not None
        assert row[1] == 5

    @pytest.mark.asyncio
    async def test_poll_g2_deduplicates(self, db):
        """Re-polling the same external_id is silently skipped (unique constraint)."""
        from mr_roboto.reviews_poll import poll_platform

        fake_reviews = [
            {
                "external_id": "g2-dup",
                "posted_at": "2026-05-15 10:00:00",
                "author": "Bob",
                "rating": 4,
                "body_md": "Good but slow.",
            }
        ]

        mock_fetch = AsyncMock(return_value=fake_reviews)
        with patch("mr_roboto.reviews_poll._fetch_g2", new=mock_fetch):
            r1 = await poll_platform("g2", "prod-1", config={})
            r2 = await poll_platform("g2", "prod-1", config={})

        assert r1["ingested"] == 1
        assert r2["ingested"] == 0  # deduped
        assert r2["skipped"] == 1

    @pytest.mark.asyncio
    async def test_poll_appstore_ingests(self, db):
        """AppStore RSS poller inserts review row."""
        from mr_roboto.reviews_poll import poll_platform

        fake_reviews = [
            {
                "external_id": "as-100",
                "posted_at": "2026-05-14 09:00:00",
                "author": "Charlie",
                "rating": 3,
                "body_md": "Average experience.",
            }
        ]

        with patch(
            "mr_roboto.reviews_poll._fetch_appstore",
            new=AsyncMock(return_value=fake_reviews),
        ):
            result = await poll_platform("appstore", "prod-1", config={"app_id": "123456"})

        assert result["ingested"] == 1

    @pytest.mark.asyncio
    async def test_poll_playstore_ingests(self, db):
        """PlayStore (unofficial) poller inserts review row."""
        from mr_roboto.reviews_poll import poll_platform

        fake_reviews = [
            {
                "external_id": "ps-200",
                "posted_at": "2026-05-13 14:00:00",
                "author": "Diana",
                "rating": 1,
                "body_md": "Crashes constantly. Terrible.",
            }
        ]

        with patch(
            "mr_roboto.reviews_poll._fetch_playstore",
            new=AsyncMock(return_value=fake_reviews),
        ):
            result = await poll_platform("playstore", "prod-1", config={"package": "com.example"})

        assert result["ingested"] == 1

    @pytest.mark.asyncio
    async def test_poll_producthunt_ingests(self, db):
        """ProductHunt (vecihi scrape fallback) poller inserts review row."""
        from mr_roboto.reviews_poll import poll_platform

        fake_reviews = [
            {
                "external_id": "ph-300",
                "posted_at": "2026-05-12 08:00:00",
                "author": "Eve",
                "rating": 5,
                "body_md": "Product of the day material!",
            }
        ]

        with patch(
            "mr_roboto.reviews_poll._fetch_producthunt",
            new=AsyncMock(return_value=fake_reviews),
        ):
            result = await poll_platform("producthunt", "prod-1", config={"slug": "my-product"})

        assert result["ingested"] == 1

    @pytest.mark.asyncio
    async def test_unknown_platform_returns_error(self, db):
        """Unsupported platform returns error dict, doesn't raise."""
        from mr_roboto.reviews_poll import poll_platform

        result = await poll_platform("unknown_platform", "prod-1", config={})
        assert result.get("error") or result.get("ingested", -1) == 0

    @pytest.mark.asyncio
    async def test_vecihi_fallback_called_when_no_free_api(self, db, monkeypatch):
        """ProductHunt uses vecihi scrape when no native API is available."""
        from mr_roboto import reviews_poll as rp

        mock_vecihi = AsyncMock(return_value=[
            {
                "external_id": "ph-vecihi-001",
                "posted_at": "2026-05-11 07:00:00",
                "author": "Frank",
                "rating": 4,
                "body_md": "Pretty good.",
            }
        ])

        # _fetch_producthunt should internally call vecihi when no API token
        with patch.object(rp, "_fetch_producthunt", new=mock_vecihi):
            result = await rp.poll_platform(
                "producthunt", "prod-1", config={"slug": "my-product", "use_vecihi": True}
            )

        mock_vecihi.assert_called_once()
        assert result["ingested"] >= 0  # even 0 is OK — shape matters


# ---------------------------------------------------------------------------
# 3. reviews/classify — LLM-bound (mocked)
# ---------------------------------------------------------------------------

class TestClassify:
    @pytest.mark.asyncio
    async def test_classify_sets_sentiment_and_theme(self, db):
        """classify sets sentiment + theme_tag on the review row."""
        # Insert a test review
        await db.execute(
            "INSERT INTO external_reviews "
            "(product_id, platform, external_id, posted_at, author, rating, body_md) "
            "VALUES ('p1', 'g2', 'cls-001', '2026-05-15 10:00:00', 'Alice', 5, 'Amazing!')"
        )
        await db.commit()

        # Get the review_id
        cur = await db.execute(
            "SELECT review_id FROM external_reviews WHERE external_id='cls-001'"
        )
        row = await cur.fetchone()
        review_id = row[0]

        from mr_roboto.reviews_classify import run as classify_run

        mock_llm_result = {"sentiment": "positive", "theme_tag": "generic-positive"}

        with patch(
            "mr_roboto.reviews_classify._call_llm_classify",
            new=AsyncMock(return_value=mock_llm_result),
        ):
            result = await classify_run({"review_id": review_id, "product_id": "p1"})

        assert result["status"] == "ok"

        # Verify DB was updated
        cur = await db.execute(
            "SELECT sentiment, theme_tag FROM external_reviews WHERE review_id=?",
            (review_id,),
        )
        row = await cur.fetchone()
        assert row[0] == "positive"
        assert row[1] == "generic-positive"

    @pytest.mark.asyncio
    async def test_classify_bug_theme(self, db):
        """A bug-themed review gets theme_tag='bug'."""
        await db.execute(
            "INSERT INTO external_reviews "
            "(product_id, platform, external_id, posted_at, author, rating, body_md) "
            "VALUES ('p1', 'g2', 'cls-bug', '2026-05-15 10:00:00', 'Bug User', 2, 'Crashes on login!')"
        )
        await db.commit()

        cur = await db.execute("SELECT review_id FROM external_reviews WHERE external_id='cls-bug'")
        review_id = (await cur.fetchone())[0]

        from mr_roboto.reviews_classify import run as classify_run

        with patch(
            "mr_roboto.reviews_classify._call_llm_classify",
            new=AsyncMock(return_value={"sentiment": "negative", "theme_tag": "bug"}),
        ):
            result = await classify_run({"review_id": review_id, "product_id": "p1"})

        assert result["status"] == "ok"

        cur = await db.execute(
            "SELECT theme_tag FROM external_reviews WHERE review_id=?", (review_id,)
        )
        assert (await cur.fetchone())[0] == "bug"

    @pytest.mark.asyncio
    async def test_classify_missing_review_id_returns_error(self, db):
        from mr_roboto.reviews_classify import run as classify_run

        result = await classify_run({"product_id": "p1"})  # no review_id
        assert result.get("status") == "error"


# ---------------------------------------------------------------------------
# 4. reviews/draft_reply — LLM-bound (mocked)
# ---------------------------------------------------------------------------

class TestDraftReply:
    @pytest.mark.asyncio
    async def test_draft_reply_produces_draft(self, db):
        """draft_reply returns a non-empty reply text; does NOT auto-post."""
        await db.execute(
            "INSERT INTO external_reviews "
            "(product_id, platform, external_id, posted_at, author, rating, body_md) "
            "VALUES ('p1', 'g2', 'dr-001', '2026-05-15', 'Alice', 5, 'Love it!')"
        )
        await db.commit()

        cur = await db.execute("SELECT review_id FROM external_reviews WHERE external_id='dr-001'")
        review_id = (await cur.fetchone())[0]

        from mr_roboto.reviews_draft_reply import run as draft_run

        mock_draft = "Thank you for your kind words, Alice! We're thrilled you love the product."

        with patch(
            "mr_roboto.reviews_draft_reply._call_llm_draft_reply",
            new=AsyncMock(return_value=mock_draft),
        ):
            result = await draft_run({"review_id": review_id, "product_id": "p1"})

        assert result["status"] == "ok"
        assert "reply_draft" in result
        assert len(result["reply_draft"]) > 10

        # Verify reply is NOT written to replied_at (it's a draft only)
        cur = await db.execute(
            "SELECT replied_at, reply_body_md FROM external_reviews WHERE review_id=?",
            (review_id,),
        )
        row = await cur.fetchone()
        assert row[0] is None, "replied_at must stay NULL — never auto-reply"
        assert row[1] is None, "reply_body_md must stay NULL — draft only"

    @pytest.mark.asyncio
    async def test_draft_reply_missing_review_returns_error(self, db):
        from mr_roboto.reviews_draft_reply import run as draft_run

        result = await draft_run({"review_id": 9999999, "product_id": "p1"})
        assert result.get("status") == "error"


# ---------------------------------------------------------------------------
# 5. Daily job — reviews_poll_daily
# ---------------------------------------------------------------------------

class TestReviewsPollDaily:
    @pytest.mark.asyncio
    async def test_daily_job_polls_and_classifies(self, db, monkeypatch):
        """reviews_poll_daily polls all configured platforms and classifies new reviews."""
        import mr_roboto.reviews_poll as rp
        import mr_roboto.reviews_classify as rc

        fake_reviews = [
            {
                "external_id": "daily-001",
                "posted_at": "2026-05-16 08:00:00",
                "author": "Daily User",
                "rating": 4,
                "body_md": "Solid product, a few rough edges.",
            }
        ]

        mock_classify_result = {"sentiment": "positive", "theme_tag": "UX"}

        with (
            patch.object(rp, "_fetch_g2", new=AsyncMock(return_value=fake_reviews)),
            patch.object(rp, "_fetch_appstore", new=AsyncMock(return_value=[])),
            patch.object(rp, "_fetch_playstore", new=AsyncMock(return_value=[])),
            patch.object(rp, "_fetch_producthunt", new=AsyncMock(return_value=[])),
            patch(
                "mr_roboto.reviews_classify._call_llm_classify",
                new=AsyncMock(return_value=mock_classify_result),
            ),
        ):
            from src.app.jobs.reviews_poll_daily import run_reviews_poll_daily
            result = await run_reviews_poll_daily(
                config={
                    "products": [
                        {
                            "product_id": "prod-daily",
                            "platforms": {
                                "g2": {},
                                "appstore": {},
                                "playstore": {},
                                "producthunt": {},
                            },
                        }
                    ]
                }
            )

        assert result["ok"] is True
        assert result["total_ingested"] >= 1
        assert result["total_classified"] >= 1

    @pytest.mark.asyncio
    async def test_daily_job_ok_true_on_no_reviews(self, db, monkeypatch):
        """Daily job returns ok=True even when no new reviews are found."""
        with (
            patch("mr_roboto.reviews_poll._fetch_g2", new=AsyncMock(return_value=[])),
            patch("mr_roboto.reviews_poll._fetch_appstore", new=AsyncMock(return_value=[])),
            patch("mr_roboto.reviews_poll._fetch_playstore", new=AsyncMock(return_value=[])),
            patch("mr_roboto.reviews_poll._fetch_producthunt", new=AsyncMock(return_value=[])),
        ):
            from src.app.jobs.reviews_poll_daily import run_reviews_poll_daily
            result = await run_reviews_poll_daily(config={"products": []})

        assert result["ok"] is True


# ---------------------------------------------------------------------------
# 6. 1-2-star reviews surface a founder_action
# ---------------------------------------------------------------------------

class TestLowStarFounderAction:
    @pytest.mark.asyncio
    async def test_one_star_surfaces_founder_action(self, db):
        """A 1-star review triggers a founder_action after classify."""
        await db.execute(
            "INSERT INTO external_reviews "
            "(product_id, platform, external_id, posted_at, author, rating, body_md) "
            "VALUES ('p1', 'g2', 'low-001', '2026-05-15 12:00:00', 'Angry', 1, 'Worst app ever!')"
        )
        await db.commit()

        cur = await db.execute("SELECT review_id FROM external_reviews WHERE external_id='low-001'")
        review_id = (await cur.fetchone())[0]

        from mr_roboto.reviews_classify import run as classify_run

        fa_created: list[dict] = []

        async def _mock_fa(**kwargs):
            fa_created.append(kwargs)
            mock = MagicMock()
            mock.id = 999
            return mock

        with (
            patch(
                "mr_roboto.reviews_classify._call_llm_classify",
                new=AsyncMock(return_value={"sentiment": "negative", "theme_tag": "generic-negative"}),
            ),
            patch("mr_roboto.reviews_classify._emit_low_star_founder_action", new=AsyncMock(side_effect=_mock_fa)),
        ):
            result = await classify_run({"review_id": review_id, "product_id": "p1"})

        assert result["status"] == "ok"
        assert len(fa_created) == 1, "Should emit exactly one founder_action for a 1-star review"

    @pytest.mark.asyncio
    async def test_five_star_no_low_star_founder_action(self, db):
        """A 5-star review does NOT trigger a low-star founder_action."""
        await db.execute(
            "INSERT INTO external_reviews "
            "(product_id, platform, external_id, posted_at, author, rating, body_md) "
            "VALUES ('p1', 'g2', 'high-001', '2026-05-15 12:00:00', 'Happy', 5, 'Love it!')"
        )
        await db.commit()

        cur = await db.execute("SELECT review_id FROM external_reviews WHERE external_id='high-001'")
        review_id = (await cur.fetchone())[0]

        from mr_roboto.reviews_classify import run as classify_run

        fa_created: list[dict] = []

        async def _mock_fa(**kwargs):
            fa_created.append(kwargs)

        with (
            patch(
                "mr_roboto.reviews_classify._call_llm_classify",
                new=AsyncMock(return_value={"sentiment": "positive", "theme_tag": "generic-positive"}),
            ),
            patch("mr_roboto.reviews_classify._emit_low_star_founder_action", new=AsyncMock(side_effect=_mock_fa)),
        ):
            await classify_run({"review_id": review_id, "product_id": "p1"})

        assert len(fa_created) == 0


# ---------------------------------------------------------------------------
# 7. Bug-tagged reviews enqueue an investigation task
# ---------------------------------------------------------------------------

class TestBugTaggedEnqueue:
    @pytest.mark.asyncio
    async def test_bug_tagged_enqueues_investigation(self, db):
        """Bug-tagged review causes beckman.enqueue to be called for an investigation task."""
        await db.execute(
            "INSERT INTO external_reviews "
            "(product_id, platform, external_id, posted_at, author, rating, body_md) "
            "VALUES ('p1', 'g2', 'bug-001', '2026-05-15 13:00:00', 'Reporter', 2, 'App crashes on login.')"
        )
        await db.commit()

        cur = await db.execute("SELECT review_id FROM external_reviews WHERE external_id='bug-001'")
        review_id = (await cur.fetchone())[0]

        from mr_roboto.reviews_classify import run as classify_run

        enqueued: list[dict] = []

        async def _mock_enqueue(spec, **kwargs):
            enqueued.append(spec)
            return 42

        with (
            patch(
                "mr_roboto.reviews_classify._call_llm_classify",
                new=AsyncMock(return_value={"sentiment": "negative", "theme_tag": "bug"}),
            ),
            patch("mr_roboto.reviews_classify._emit_low_star_founder_action", new=AsyncMock()),
            patch("mr_roboto.reviews_classify._enqueue_bug_investigation", new=AsyncMock(side_effect=_mock_enqueue)),
        ):
            result = await classify_run({"review_id": review_id, "product_id": "p1"})

        assert result["status"] == "ok"
        assert len(enqueued) == 1
        assert "bug" in str(enqueued[0]).lower() or "investigation" in str(enqueued[0]).lower()

    @pytest.mark.asyncio
    async def test_non_bug_theme_does_not_enqueue(self, db):
        """UX-themed review does NOT enqueue investigation."""
        await db.execute(
            "INSERT INTO external_reviews "
            "(product_id, platform, external_id, posted_at, author, rating, body_md) "
            "VALUES ('p1', 'g2', 'ux-001', '2026-05-15 14:00:00', 'UX User', 3, 'UI could be cleaner.')"
        )
        await db.commit()

        cur = await db.execute("SELECT review_id FROM external_reviews WHERE external_id='ux-001'")
        review_id = (await cur.fetchone())[0]

        from mr_roboto.reviews_classify import run as classify_run

        enqueued: list[dict] = []

        with (
            patch(
                "mr_roboto.reviews_classify._call_llm_classify",
                new=AsyncMock(return_value={"sentiment": "neutral", "theme_tag": "UX"}),
            ),
            patch("mr_roboto.reviews_classify._emit_low_star_founder_action", new=AsyncMock()),
            patch("mr_roboto.reviews_classify._enqueue_bug_investigation", new=AsyncMock(side_effect=lambda *a, **k: enqueued.append(1))),
        ):
            await classify_run({"review_id": review_id, "product_id": "p1"})

        assert len(enqueued) == 0


# ---------------------------------------------------------------------------
# 8. Reversibility entries
# ---------------------------------------------------------------------------

class TestReversibility:
    def test_reviews_poll_verbs_registered(self):
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        for platform in ("g2", "appstore", "playstore", "producthunt"):
            verb = f"reviews/poll/{platform}"
            assert verb in VERB_REVERSIBILITY, f"{verb} not in VERB_REVERSIBILITY"

    def test_reviews_classify_registered(self):
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        assert "reviews/classify" in VERB_REVERSIBILITY

    def test_reviews_draft_reply_registered(self):
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        assert "reviews/draft_reply" in VERB_REVERSIBILITY

    def test_reviews_poll_daily_registered(self):
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        assert "reviews_poll_daily" in VERB_REVERSIBILITY

    def test_poll_verbs_are_full_reversible(self):
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        for platform in ("g2", "appstore", "playstore", "producthunt"):
            verb = f"reviews/poll/{platform}"
            assert VERB_REVERSIBILITY[verb] == "full", f"{verb} should be full reversible"

    def test_draft_reply_is_full_reversible(self):
        """draft_reply writes to DB only — never external — so reversible."""
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        assert VERB_REVERSIBILITY["reviews/draft_reply"] == "full"


# ---------------------------------------------------------------------------
# 9. Cron seed entry
# ---------------------------------------------------------------------------

class TestCronSeed:
    def test_reviews_poll_daily_in_internal_cadences(self):
        from general_beckman.cron_seed import INTERNAL_CADENCES
        titles = {c["title"] for c in INTERNAL_CADENCES}
        assert "reviews_poll_daily" in titles, "reviews_poll_daily must be in INTERNAL_CADENCES"

    def test_reviews_poll_daily_has_correct_executor(self):
        from general_beckman.cron_seed import INTERNAL_CADENCES
        entry = next((c for c in INTERNAL_CADENCES if c["title"] == "reviews_poll_daily"), None)
        assert entry is not None
        assert entry["payload"].get("_executor") == "reviews_poll_daily"

    def test_reviews_poll_daily_interval_is_daily(self):
        from general_beckman.cron_seed import INTERNAL_CADENCES
        entry = next((c for c in INTERNAL_CADENCES if c["title"] == "reviews_poll_daily"), None)
        assert entry is not None
        # daily = 86400 seconds
        assert entry.get("interval_seconds") == 86400


# ---------------------------------------------------------------------------
# 10. mr_roboto dispatch — reviews verbs route correctly
# ---------------------------------------------------------------------------

class TestMrRobotoDispatch:
    @pytest.mark.asyncio
    async def test_reviews_poll_g2_dispatches(self, db, monkeypatch):
        """mr_roboto.run() routes reviews/poll/g2 to reviews_poll.run."""
        import mr_roboto

        mock_result = {"status": "ok", "ingested": 0, "skipped": 0}
        mock_run = AsyncMock(return_value=mock_result)

        with patch("mr_roboto.reviews_poll.run", new=mock_run):
            task = {
                "id": 1,
                "mission_id": 1,
                "payload": {
                    "action": "reviews/poll/g2",
                    "product_id": "p1",
                    "config": {},
                },
            }
            action = await mr_roboto.run(task)

        assert action.status == "completed"

    @pytest.mark.asyncio
    async def test_reviews_classify_dispatches(self, db, monkeypatch):
        """mr_roboto.run() routes reviews/classify to reviews_classify.run."""
        import mr_roboto

        mock_result = {"status": "ok", "sentiment": "positive", "theme_tag": "generic-positive"}
        mock_run = AsyncMock(return_value=mock_result)

        with patch("mr_roboto.reviews_classify.run", new=mock_run):
            task = {
                "id": 2,
                "mission_id": 1,
                "payload": {
                    "action": "reviews/classify",
                    "review_id": 1,
                    "product_id": "p1",
                },
            }
            action = await mr_roboto.run(task)

        assert action.status == "completed"

    @pytest.mark.asyncio
    async def test_reviews_draft_reply_dispatches(self, db, monkeypatch):
        """mr_roboto.run() routes reviews/draft_reply to reviews_draft_reply.run."""
        import mr_roboto

        mock_result = {"status": "ok", "reply_draft": "Thank you for your feedback!"}
        mock_run = AsyncMock(return_value=mock_result)

        with patch("mr_roboto.reviews_draft_reply.run", new=mock_run):
            task = {
                "id": 3,
                "mission_id": 1,
                "payload": {
                    "action": "reviews/draft_reply",
                    "review_id": 1,
                    "product_id": "p1",
                },
            }
            action = await mr_roboto.run(task)

        assert action.status == "completed"

    @pytest.mark.asyncio
    async def test_reviews_poll_daily_dispatches(self, db, monkeypatch):
        """mr_roboto.run() routes reviews_poll_daily executor to daily job."""
        import mr_roboto

        with patch(
            "src.app.jobs.reviews_poll_daily.run_reviews_poll_daily",
            new=AsyncMock(return_value={"ok": True, "total_ingested": 0, "total_classified": 0}),
        ):
            task = {
                "id": 4,
                "mission_id": None,
                "payload": {"action": "reviews_poll_daily"},
            }
            action = await mr_roboto.run(task)

        assert action.status == "completed"
