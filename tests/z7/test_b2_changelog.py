"""Z7 T5 B2 — Changelog as public artifact tests.

Covers:
  1. DB migration: changelog_entries table exists with correct columns.
  2. changelog/draft verb: maps conventional-commit prefixes to KAC buckets,
     runs A5 brand_voice_lint + A6 copy_compliance (degrade gracefully).
  3. changelog/publish verb: marks entry published, regenerates RSS,
     invalidates in-app banner cache, queues B1 announcement email blast
     (degrades gracefully when no announcement sequence).
  4. GET /changelog — HTML page renders published entries.
  5. GET /changelog.rss — RSS feed of published entries.
  6. GET /changelog/latest.json — in-app banner data endpoint.
  7. changelog_freshness posthook: flags a goal:public_release mission
     that shipped without a changelog entry.
  8. Posthook registered in POST_HOOK_REGISTRY.
  9. mr_roboto reversibility entries for changelog/* verbs.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


# ── DB helpers ───────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Fresh SQLite DB for B2 tests."""
    db_file = str(tmp_path / "test_b2.db")
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
    """Initialised DB with full schema (includes B2 migration)."""
    import src.infra.db as _db_mod
    _db_mod._db_connection = None
    _db_mod._db_connection_path = None
    from src.infra.db import init_db, get_db
    await init_db()
    _db = await get_db()
    yield _db
    _db_mod._db_connection = None
    _db_mod._db_connection_path = None


# ── 1. Migration: table exists ───────────────────────────────────────────────


class TestMigration:
    @pytest.mark.asyncio
    async def test_changelog_entries_columns(self, db):
        cur = await db.execute("PRAGMA table_info(changelog_entries)")
        cols = {r[1] for r in await cur.fetchall()}
        assert "entry_id" in cols
        assert "product_id" in cols
        assert "version" in cols
        assert "released_at" in cols
        assert "title" in cols
        assert "body_md" in cols
        assert "kind_breakdown_json" in cols
        assert "shipped_features_json" in cols
        assert "related_mission_ids_json" in cols
        assert "external_url" in cols
        assert "published" in cols

    @pytest.mark.asyncio
    async def test_changelog_entries_can_insert_and_query(self, db):
        await db.execute(
            "INSERT INTO changelog_entries "
            "(product_id, version, title, body_md, kind_breakdown_json, "
            " shipped_features_json, related_mission_ids_json, published) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
            ("prod1", "1.0.0", "Initial Release", "First release body.",
             '{"added":["Feature A"],"fixed":[]}',
             '["Feature A"]',
             '[1,2]'),
        )
        await db.commit()
        cur = await db.execute(
            "SELECT entry_id, version FROM changelog_entries WHERE product_id=?",
            ("prod1",),
        )
        row = await cur.fetchone()
        assert row is not None
        assert row[1] == "1.0.0"


# ── 2. Commit prefix → KAC bucket mapping ────────────────────────────────────


class TestCommitMapping:
    """Tests for the commit-prefix-to-KAC-bucket mapper in changelog/draft."""

    def _get_mapper(self):
        from mr_roboto.changelog_draft import map_commits_to_kac_buckets
        return map_commits_to_kac_buckets

    def test_feat_maps_to_added(self):
        mapper = self._get_mapper()
        result = mapper(["feat(ui): add dashboard", "feat: new API endpoint"])
        assert "added" in result
        assert any("dashboard" in s for s in result["added"])

    def test_fix_maps_to_fixed(self):
        mapper = self._get_mapper()
        result = mapper(["fix(auth): resolve token expiry"])
        assert "fixed" in result
        assert any("token" in s for s in result["fixed"])

    def test_refactor_maps_to_changed(self):
        mapper = self._get_mapper()
        result = mapper(["refactor: simplify routing logic"])
        assert "changed" in result

    def test_deprecate_maps_to_deprecated(self):
        mapper = self._get_mapper()
        result = mapper(["deprecate: old v1 endpoint"])
        assert "deprecated" in result

    def test_remove_maps_to_removed(self):
        mapper = self._get_mapper()
        result = mapper(["remove: legacy CSV export"])
        assert "removed" in result

    def test_chore_and_docs_ignored(self):
        mapper = self._get_mapper()
        result = mapper(["chore: update deps", "docs: fix typo"])
        total = sum(len(v) for v in result.values())
        assert total == 0

    def test_mixed_commits(self):
        mapper = self._get_mapper()
        commits = [
            "feat: add OAuth support",
            "fix: crash on empty input",
            "refactor: restructure DB layer",
            "chore: bump pytest",
        ]
        result = mapper(commits)
        assert len(result["added"]) == 1
        assert len(result["fixed"]) == 1
        assert len(result["changed"]) == 1

    def test_empty_commits(self):
        mapper = self._get_mapper()
        result = mapper([])
        assert result == {"added": [], "changed": [], "fixed": [], "deprecated": [], "removed": []}


# ── 3. changelog/draft verb (mr_roboto) ──────────────────────────────────────


class TestChangelogDraftVerb:
    """Tests for mr_roboto changelog/draft (non-LLM path, git-log mocked)."""

    @pytest.mark.asyncio
    async def test_draft_creates_db_row(self, db):
        """draft verb writes a draft (published=0) changelog_entries row."""
        from mr_roboto.changelog_draft import run as draft_run

        commits = [
            "feat: add export button",
            "fix: null pointer in login",
        ]

        with (
            patch("mr_roboto.changelog_draft._git_log_since_last_entry",
                  AsyncMock(return_value=commits)),
            patch("mr_roboto.changelog_draft._run_brand_voice_lint",
                  AsyncMock(return_value={"status": "ok", "violations": []})),
            patch("mr_roboto.changelog_draft._run_copy_compliance",
                  AsyncMock(return_value={"status": "ok", "violations": []})),
            patch("mr_roboto.changelog_draft._emit_founder_action",
                  AsyncMock(return_value=MagicMock(id=42))),
        ):
            result = await draft_run({
                "product_id": "prod1",
                "mission_id": 1,
                "version": "1.1.0",
            })

        assert result["status"] == "ok"
        assert result.get("entry_id") is not None

        cur = await db.execute(
            "SELECT published, version FROM changelog_entries WHERE entry_id=?",
            (result["entry_id"],),
        )
        row = await cur.fetchone()
        assert row is not None
        assert row[0] == 0  # draft (not published)
        assert row[1] == "1.1.0"

    @pytest.mark.asyncio
    async def test_draft_degrades_when_brand_voice_lint_unavailable(self, db):
        """draft succeeds even when brand_voice_lint raises ImportError."""
        from mr_roboto.changelog_draft import run as draft_run

        with (
            patch("mr_roboto.changelog_draft._git_log_since_last_entry",
                  AsyncMock(return_value=["feat: something"])),
            patch("mr_roboto.changelog_draft._run_brand_voice_lint",
                  AsyncMock(side_effect=ImportError("A5 not available"))),
            patch("mr_roboto.changelog_draft._run_copy_compliance",
                  AsyncMock(return_value={"status": "ok", "violations": []})),
            patch("mr_roboto.changelog_draft._emit_founder_action",
                  AsyncMock(return_value=None)),
        ):
            result = await draft_run({
                "product_id": "prod2",
                "mission_id": 2,
                "version": "2.0.0",
            })

        assert result["status"] == "ok"
        assert result.get("lint_degraded") is True

    @pytest.mark.asyncio
    async def test_draft_degrades_when_copy_compliance_unavailable(self, db):
        """draft succeeds even when copy_compliance raises ImportError."""
        from mr_roboto.changelog_draft import run as draft_run

        with (
            patch("mr_roboto.changelog_draft._git_log_since_last_entry",
                  AsyncMock(return_value=["fix: something"])),
            patch("mr_roboto.changelog_draft._run_brand_voice_lint",
                  AsyncMock(return_value={"status": "ok", "violations": []})),
            patch("mr_roboto.changelog_draft._run_copy_compliance",
                  AsyncMock(side_effect=ImportError("A6 not available"))),
            patch("mr_roboto.changelog_draft._emit_founder_action",
                  AsyncMock(return_value=None)),
        ):
            result = await draft_run({
                "product_id": "prod3",
                "mission_id": 3,
                "version": "3.0.0",
            })

        assert result["status"] == "ok"
        assert result.get("compliance_degraded") is True


# ── 4. changelog/publish verb ─────────────────────────────────────────────────


class TestChangelogPublishVerb:
    """Tests for mr_roboto changelog/publish."""

    @pytest_asyncio.fixture
    async def draft_entry(self, db):
        """Insert a draft entry and return its entry_id."""
        await db.execute(
            "INSERT INTO changelog_entries "
            "(product_id, version, title, body_md, kind_breakdown_json, "
            " shipped_features_json, related_mission_ids_json, published) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
            ("prod_pub", "1.2.0", "Feature Release", "Added stuff.",
             '{"added":["stuff"],"fixed":[]}', '["stuff"]', '[]'),
        )
        await db.commit()
        cur = await db.execute(
            "SELECT entry_id FROM changelog_entries WHERE product_id=? AND version=?",
            ("prod_pub", "1.2.0"),
        )
        row = await cur.fetchone()
        return row[0]

    @pytest.mark.asyncio
    async def test_publish_marks_entry_published(self, db, draft_entry):
        from mr_roboto.changelog_publish import run as publish_run

        with (
            patch("mr_roboto.changelog_publish._invalidate_changelog_cache"),
            patch("mr_roboto.changelog_publish._queue_announcement_email",
                  AsyncMock(return_value={"ok": True, "sends_created": 2})),
        ):
            result = await publish_run({
                "entry_id": draft_entry,
                "product_id": "prod_pub",
            })

        assert result["status"] == "ok"
        cur = await db.execute(
            "SELECT published FROM changelog_entries WHERE entry_id=?",
            (draft_entry,),
        )
        row = await cur.fetchone()
        assert row[0] == 1  # now published

    @pytest.mark.asyncio
    async def test_publish_invalidates_cache(self, db, draft_entry):
        from mr_roboto.changelog_publish import run as publish_run

        invalidated = []

        with (
            patch("mr_roboto.changelog_publish._invalidate_changelog_cache",
                  side_effect=lambda: invalidated.append(True)),
            patch("mr_roboto.changelog_publish._queue_announcement_email",
                  AsyncMock(return_value={"ok": False, "reason": "no sequence"})),
        ):
            await publish_run({"entry_id": draft_entry, "product_id": "prod_pub"})

        assert len(invalidated) == 1

    @pytest.mark.asyncio
    async def test_publish_queues_email_blast(self, db, draft_entry):
        """publish calls trigger_sequence_by_kind with kind='announcement'."""
        from mr_roboto.changelog_publish import run as publish_run

        queued = []

        async def _mock_queue(product_id, **kwargs):
            queued.append(product_id)
            return {"ok": True, "sends_created": 3}

        with (
            patch("mr_roboto.changelog_publish._invalidate_changelog_cache"),
            patch("mr_roboto.changelog_publish._queue_announcement_email",
                  _mock_queue),
        ):
            result = await publish_run({"entry_id": draft_entry, "product_id": "prod_pub"})

        assert result["status"] == "ok"
        assert "prod_pub" in queued

    @pytest.mark.asyncio
    async def test_publish_degrades_when_no_announcement_sequence(self, db, draft_entry):
        """publish succeeds even when no announcement sequence exists."""
        from mr_roboto.changelog_publish import run as publish_run

        with (
            patch("mr_roboto.changelog_publish._invalidate_changelog_cache"),
            patch("mr_roboto.changelog_publish._queue_announcement_email",
                  AsyncMock(return_value={"ok": False, "reason": "no enabled announcement sequence"})),
        ):
            result = await publish_run({"entry_id": draft_entry, "product_id": "prod_pub"})

        # Must still succeed (email blast is a best-effort channel)
        assert result["status"] == "ok"
        assert result.get("email_blast_skipped") is True


# ── 5. Routes: /changelog, /changelog.rss, /changelog/latest.json ───────────


class TestChangelogRoutes:
    """Tests for the changelog page module."""

    @pytest_asyncio.fixture(autouse=True)
    async def clear_changelog_cache(self):
        """Ensure changelog page cache is clean for every route test."""
        from src.app.changelog_page import invalidate_cache
        invalidate_cache()
        yield
        invalidate_cache()

    @pytest.mark.asyncio
    async def test_changelog_html_renders_published_entries(self, db):
        """HTML route includes published entry versions."""
        await db.execute(
            "INSERT INTO changelog_entries "
            "(product_id, version, title, body_md, kind_breakdown_json, "
            " shipped_features_json, related_mission_ids_json, published) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
            ("prod1", "2.0.0", "Big Release", "Added lots.",
             '{"added":["A","B"],"fixed":["C"]}', '["A","B"]', '[]'),
        )
        await db.commit()

        from src.app.changelog_page import changelog_html_handler
        html = await changelog_html_handler()
        assert "2.0.0" in html
        assert "Big Release" in html

    @pytest.mark.asyncio
    async def test_changelog_html_excludes_draft_entries(self, db):
        """Draft entries (published=0) must NOT appear on the public page."""
        await db.execute(
            "INSERT INTO changelog_entries "
            "(product_id, version, title, body_md, kind_breakdown_json, "
            " shipped_features_json, related_mission_ids_json, published) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
            ("prod1", "99.0.0-draft", "Secret Draft", "Not yet.",
             '{}', '[]', '[]'),
        )
        await db.commit()

        from src.app.changelog_page import changelog_html_handler
        html = await changelog_html_handler()
        assert "99.0.0-draft" not in html
        assert "Secret Draft" not in html

    @pytest.mark.asyncio
    async def test_changelog_rss_contains_published_entry(self, db):
        await db.execute(
            "INSERT INTO changelog_entries "
            "(product_id, version, title, body_md, kind_breakdown_json, "
            " shipped_features_json, related_mission_ids_json, published) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
            ("prod1", "3.0.0", "RSS Entry", "RSS body.",
             '{"added":["X"],"fixed":[]}', '["X"]', '[]'),
        )
        await db.commit()

        from src.app.changelog_page import changelog_rss_handler
        rss = await changelog_rss_handler()
        assert "3.0.0" in rss
        assert "RSS Entry" in rss
        assert "<rss" in rss

    @pytest.mark.asyncio
    async def test_changelog_rss_excludes_draft(self, db):
        await db.execute(
            "INSERT INTO changelog_entries "
            "(product_id, version, title, body_md, kind_breakdown_json, "
            " shipped_features_json, related_mission_ids_json, published) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
            ("prod1", "88.0-draft", "Draft Only", "Private.",
             '{}', '[]', '[]'),
        )
        await db.commit()

        from src.app.changelog_page import changelog_rss_handler
        rss = await changelog_rss_handler()
        assert "88.0-draft" not in rss

    @pytest.mark.asyncio
    async def test_changelog_latest_json_returns_most_recent_published(self, db):
        await db.execute(
            "INSERT INTO changelog_entries "
            "(product_id, version, title, body_md, kind_breakdown_json, "
            " shipped_features_json, related_mission_ids_json, published) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
            ("prod1", "4.0.0", "Latest Release", "Latest body.",
             '{"added":["New"]}', '["New"]', '[]'),
        )
        await db.commit()

        from src.app.changelog_page import changelog_latest_json_handler
        payload = await changelog_latest_json_handler()
        assert isinstance(payload, dict)
        assert payload.get("version") == "4.0.0"
        assert payload.get("title") == "Latest Release"

    @pytest.mark.asyncio
    async def test_changelog_latest_json_returns_none_when_empty(self, db):
        from src.app.changelog_page import changelog_latest_json_handler
        payload = await changelog_latest_json_handler()
        assert payload is None or payload == {}


# ── 6. changelog_freshness posthook ──────────────────────────────────────────


class TestChangelogFreshnessPosthook:
    """Tests for the changelog_freshness posthook handler."""

    @pytest.mark.asyncio
    async def test_freshness_flags_missing_entry(self, db):
        """Fires when a public_release mission context has no changelog entry."""
        # Insert a mission with goal:public_release in context, completed
        await db.execute(
            "INSERT INTO missions "
            "(id, title, status, created_at) "
            "VALUES (?, ?, 'completed', datetime('now','-2 days'))",
            (9001, "Test Product Launch"),
        )
        await db.commit()

        # No changelog_entries row → should flag
        from general_beckman.posthook_handlers.changelog_freshness import handle

        # The posthook relies on context.goal or context.product_id with
        # is_public_release flag; we set is_public_release=True in context
        task = {
            "id": 1,
            "mission_id": 9001,
            "context": {
                "product_id": "prod_missing_log",
                "is_public_release": True,
            },
        }

        fa_emitted = []

        async def _mock_fa(**kwargs):
            fa_emitted.append(kwargs)
            return MagicMock(id=55)

        with patch(
            "general_beckman.posthook_handlers.changelog_freshness._emit_founder_action",
            _mock_fa,
        ):
            result = await handle(task, {})

        assert result["status"] == "flagged"
        assert len(fa_emitted) == 1

    @pytest.mark.asyncio
    async def test_freshness_ok_when_entry_exists(self, db):
        """Returns ok when a published changelog entry exists for the mission."""
        await db.execute(
            "INSERT INTO missions "
            "(id, title, status, created_at) "
            "VALUES (?, ?, 'completed', datetime('now','-1 days'))",
            (9002, "Another Launch"),
        )
        await db.execute(
            "INSERT INTO changelog_entries "
            "(product_id, version, title, body_md, kind_breakdown_json, "
            " shipped_features_json, related_mission_ids_json, published) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
            ("prod_ok", "5.0.0", "Release", "Body.",
             '{}', '[]', '[9002]'),
        )
        await db.commit()

        from general_beckman.posthook_handlers.changelog_freshness import handle

        task = {
            "id": 2,
            "mission_id": 9002,
            "context": {
                "product_id": "prod_ok",
                "is_public_release": True,
            },
        }

        result = await handle(task, {})
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_freshness_skip_when_no_product_id(self, db):
        from general_beckman.posthook_handlers.changelog_freshness import handle

        task = {
            "id": 3,
            "mission_id": 9003,
            "context": {},
        }
        result = await handle(task, {})
        assert result["status"] == "skip"


# ── 7. Posthook registered in POST_HOOK_REGISTRY ─────────────────────────────


class TestPosthookRegistration:
    def test_changelog_freshness_in_registry(self):
        from general_beckman.posthooks import POST_HOOK_REGISTRY
        assert "changelog_freshness" in POST_HOOK_REGISTRY

    def test_changelog_freshness_verb(self):
        from general_beckman.posthooks import POST_HOOK_REGISTRY
        spec = POST_HOOK_REGISTRY["changelog_freshness"]
        assert spec.verb == "changelog_freshness"


# ── 8. mr_roboto reversibility ───────────────────────────────────────────────


class TestReversibility:
    def test_changelog_draft_is_full(self):
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        assert VERB_REVERSIBILITY.get("changelog/draft") == "full"

    def test_changelog_publish_is_irreversible(self):
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        assert VERB_REVERSIBILITY.get("changelog/publish") == "irreversible"

    def test_changelog_freshness_is_full(self):
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        assert VERB_REVERSIBILITY.get("changelog_freshness") == "full"


# ── 9. Webhook listener routes exist ─────────────────────────────────────────


class TestWebhookRoutes:
    def test_changelog_route_registered(self):
        from src.app.webhook_listener import app
        paths = [r.path for r in app.routes]
        assert "/changelog" in paths

    def test_changelog_rss_route_registered(self):
        from src.app.webhook_listener import app
        paths = [r.path for r in app.routes]
        assert "/changelog.rss" in paths

    def test_changelog_latest_json_route_registered(self):
        from src.app.webhook_listener import app
        paths = [r.path for r in app.routes]
        assert "/changelog/latest.json" in paths
