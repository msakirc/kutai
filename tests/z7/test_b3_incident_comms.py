"""Z7 T3D — B3: Status page + customer-facing incident comms tests.

Covers:
  1. DB migration: incidents + status_updates tables exist after init_db.
  2. Incident lifecycle: open → status_updates → resolve.
  3. PII / hostname redaction in draft_update (redact_internal + redact_secrets + redact_user_pii).
  4. /status route renders active incidents (HTML + RSS).
  5. founder_action emitted pre-publish (incident_update_review posthook handler).
  6. incident/publish_status writes status_updates row + updates incidents.
  7. incident/draft_postmortem writes artifact + emits founder_action.
  8. incident_comms.json parses as valid JSON.
  9. Reversibility entries registered for all three B3 verbs.
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
    """Provide a fresh in-memory-style SQLite DB for tests via tmp file."""
    db_file = str(tmp_path / "test_b3.db")
    monkeypatch.setenv("DB_PATH", db_file)
    # Patch config module so db.py picks up the new path.
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
    """Initialize DB with full schema (includes Z7 T3D migrations)."""
    from src.infra.db import init_db, get_db, _db_connection
    import src.infra.db as _db_mod

    # Clear cached connection so init_db opens against tmp_db.
    _db_mod._db_connection = None
    _db_mod._db_connection_path = None

    await init_db()
    db = await get_db()
    yield db

    # Cleanup
    _db_mod._db_connection = None
    _db_mod._db_connection_path = None


# ===========================================================================
# 1. DB migrations: incidents + status_updates tables exist
# ===========================================================================

class TestMigrations:
    @pytest.mark.asyncio
    async def test_incidents_table_exists(self, initialized_db):
        """incidents table created by 2026-05-15-z7-incidents migration."""
        db = initialized_db
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='incidents'"
        ) as cur:
            row = await cur.fetchone()
        assert row is not None, "incidents table should exist after init_db"

    @pytest.mark.asyncio
    async def test_status_updates_table_exists(self, initialized_db):
        """status_updates table created by 2026-05-15-z7-status-updates migration."""
        db = initialized_db
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='status_updates'"
        ) as cur:
            row = await cur.fetchone()
        assert row is not None, "status_updates table should exist after init_db"

    @pytest.mark.asyncio
    async def test_incidents_columns(self, initialized_db):
        """incidents table has all required columns."""
        db = initialized_db
        async with db.execute("PRAGMA table_info(incidents)") as cur:
            rows = await cur.fetchall()
        cols = [r[1] for r in rows]
        required = [
            "incident_id", "product_id", "opened_at", "resolved_at",
            "severity", "affected_components_json", "customer_impact_summary",
            "current_status_md", "postmortem_url",
        ]
        for col in required:
            assert col in cols, f"incidents.{col} should exist"

    @pytest.mark.asyncio
    async def test_status_updates_columns(self, initialized_db):
        """status_updates table has all required columns."""
        db = initialized_db
        async with db.execute("PRAGMA table_info(status_updates)") as cur:
            rows = await cur.fetchall()
        cols = [r[1] for r in rows]
        required = [
            "update_id", "product_id", "incident_id", "posted_at",
            "body_md", "status_kind",
        ]
        for col in required:
            assert col in cols, f"status_updates.{col} should exist"

    @pytest.mark.asyncio
    async def test_migration_versioned_idempotent(self, initialized_db):
        """Running init_db again is idempotent — no duplicate migration errors."""
        from src.infra.db import init_db
        # Should not raise.
        await init_db()

        db = initialized_db
        async with db.execute(
            "SELECT COUNT(*) FROM schema_migrations WHERE version LIKE '2026-05-15-z7-incidents%'"
        ) as cur:
            row = await cur.fetchone()
        assert row[0] == 1, "migration should be recorded exactly once"


# ===========================================================================
# 2. Incident lifecycle (open → status_updates → resolve)
# ===========================================================================

class TestIncidentLifecycle:
    @pytest.mark.asyncio
    async def test_open_and_resolve(self, initialized_db):
        """Can insert an incident, add status_updates, then resolve it."""
        from src.infra.db import get_db
        from src.infra.times import db_now

        db = await get_db()
        now = db_now()

        # Open incident.
        cur = await db.execute(
            "INSERT INTO incidents (product_id, severity, affected_components_json, "
            "customer_impact_summary) VALUES (?, ?, ?, ?)",
            ("prod-alpha", "major", '["api", "webhooks"]', "Some API requests are failing."),
        )
        await db.commit()
        incident_id = cur.lastrowid

        # Add status update.
        cur2 = await db.execute(
            "INSERT INTO status_updates (product_id, incident_id, body_md, status_kind) "
            "VALUES (?, ?, ?, ?)",
            ("prod-alpha", incident_id, "We are investigating an issue with the API.", "investigating"),
        )
        await db.commit()
        update_id = cur2.lastrowid

        # Resolve.
        await db.execute(
            "UPDATE incidents SET resolved_at = ?, current_status_md = ? WHERE incident_id = ?",
            (now, "The issue is resolved.", incident_id),
        )
        await db.commit()

        # Verify.
        async with db.execute(
            "SELECT resolved_at, current_status_md FROM incidents WHERE incident_id = ?",
            (incident_id,),
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row[0] is not None, "resolved_at should be set"
        assert "resolved" in row[1].lower()

        async with db.execute(
            "SELECT status_kind FROM status_updates WHERE update_id = ?",
            (update_id,),
        ) as cur:
            su_row = await cur.fetchone()
        assert su_row[0] == "investigating"

    @pytest.mark.asyncio
    async def test_product_id_not_null(self, initialized_db):
        """product_id NOT NULL constraint is enforced on incidents."""
        from src.infra.db import get_db
        import aiosqlite

        db = await get_db()
        with pytest.raises(Exception):
            await db.execute(
                "INSERT INTO incidents (product_id, severity, affected_components_json) "
                "VALUES (NULL, 'minor', '[]')"
            )
            await db.commit()


# ===========================================================================
# 3. PII / hostname redaction in draft_update
# ===========================================================================

class TestRedaction:
    def test_redact_internal_strips_private_ip(self):
        """Private IPv4 addresses are replaced with [internal-ip]."""
        from mr_roboto.incident_draft_update import redact_internal
        text = "Server 192.168.1.100:8080 is down. Failover to 10.0.0.5."
        result = redact_internal(text)
        assert "192.168" not in result
        assert "10.0.0.5" not in result
        assert "[internal-ip]" in result

    def test_redact_internal_strips_stack_traces(self):
        """Python-style stack traces are replaced wholesale."""
        from mr_roboto.incident_draft_update import redact_internal
        text = (
            "Error in production:\n"
            "Traceback (most recent call last):\n"
            '  File "src/api/views.py", line 42, in handle_request\n'
            "    raise ValueError('bad token')\n"
            "ValueError: bad token\n"
            "Next update at 14:00 UTC."
        )
        result = redact_internal(text)
        assert "Traceback" not in result
        assert "views.py" not in result
        assert "[internal error detail redacted]" in result
        # Non-sensitive content is preserved.
        assert "Next update" in result

    def test_redact_internal_strips_internal_hostname(self):
        """*.internal / *.local hostnames are replaced."""
        from mr_roboto.incident_draft_update import redact_internal
        text = "Connecting to db.prod.internal:5432 failed. Using fallback."
        result = redact_internal(text)
        assert ".internal" not in result
        assert "[internal-host]" in result

    def test_redact_alert_cleans_nested_dict(self):
        """_redact_alert recurses into nested dicts and list values."""
        from mr_roboto.incident_draft_update import _redact_alert
        alert = {
            "message": "Connection to db.staging.internal refused",
            "details": {
                "traceback": "Traceback (most recent call last):\n  File 'x.py'",
                "ip": "10.1.2.3",
            },
            "user_email": "alice@example.com",
        }
        safe = _redact_alert(alert)
        # Internal hostname removed.
        assert ".internal" not in safe["message"]
        # Stack trace block removed.
        assert "Traceback" not in safe["details"]["traceback"]
        # Private IP removed.
        assert "10.1.2.3" not in safe["details"]["ip"]
        # User email (PII) removed.
        assert "alice@example.com" not in safe["user_email"]

    @pytest.mark.asyncio
    async def test_draft_update_verb_returns_ok_with_mock_llm(
        self, initialized_db, monkeypatch
    ):
        """incident/draft_update returns ok status and redacted draft."""
        from src.infra.db import get_db
        from src.infra.times import db_now

        db = await get_db()
        cur = await db.execute(
            "INSERT INTO incidents (product_id, severity, affected_components_json, "
            "customer_impact_summary) VALUES (?, ?, ?, ?)",
            ("prod-beta", "critical", '["payments"]', "Payment processing is impaired."),
        )
        await db.commit()
        incident_id = cur.lastrowid

        # Mock LLM call to avoid beckman.enqueue dependency.
        async def _mock_llm_draft(**kwargs):
            return "We are investigating an issue affecting payment processing."

        import mr_roboto.incident_draft_update as _idu
        monkeypatch.setattr(_idu, "_call_llm_draft", _mock_llm_draft)

        from mr_roboto.incident_draft_update import run as draft_update_run
        result = await draft_update_run({
            "incident_id": incident_id,
            "product_id": "prod-beta",
            "alert_details": {
                "host": "db.internal",
                "traceback": "Traceback (most recent call last):\n  File 'x.py', line 5",
                "ip": "192.168.1.50",
            },
            "status_kind": "investigating",
        })

        assert result["status"] == "ok"
        assert "draft" in result
        assert result["redaction_applied"] is True
        # Draft should not contain internal infra details.
        draft = result["draft"]
        assert ".internal" not in draft
        assert "192.168" not in draft

    @pytest.mark.asyncio
    async def test_draft_update_missing_incident_id(self, initialized_db):
        """incident/draft_update returns error if incident_id missing."""
        from mr_roboto.incident_draft_update import run as draft_update_run
        result = await draft_update_run({"product_id": "prod-x", "alert_details": {}})
        assert result["status"] == "error"
        assert "incident_id" in result["error"]


# ===========================================================================
# 4. /status route renders active incidents (HTML + RSS)
# ===========================================================================

class TestStatusPageRendering:
    @pytest.mark.asyncio
    async def test_status_html_renders_all_ok_when_no_incidents(
        self, initialized_db, monkeypatch
    ):
        """Status page shows 'All systems operational' when no open incidents."""
        import src.app.status_page as sp
        monkeypatch.setattr(sp, "_cache", {})

        async def _no_incidents(product_id=None):
            return []

        async def _no_stats(days=90):
            return {}

        async def _no_updates(limit=20):
            return []

        monkeypatch.setattr(sp, "_fetch_active_incidents", _no_incidents)
        monkeypatch.setattr(sp, "_fetch_uptime_stats", _no_stats)
        monkeypatch.setattr(sp, "_fetch_recent_updates", _no_updates)

        html_content = await sp.status_html_handler()
        assert "All systems operational" in html_content
        assert "<!DOCTYPE html>" in html_content
        assert "/status.rss" in html_content

    @pytest.mark.asyncio
    async def test_status_html_renders_active_incident(
        self, initialized_db, monkeypatch
    ):
        """Status page shows active incident details when one is open."""
        import src.app.status_page as sp
        monkeypatch.setattr(sp, "_cache", {})

        async def _one_incident(product_id=None):
            return [{
                "incident_id": 42,
                "product_id": "prod-x",
                "opened_at": "2026-05-15 10:00:00",
                "severity": "major",
                "affected_components": ["api", "webhooks"],
                "customer_impact_summary": "API calls are delayed.",
                "current_status_md": "We are investigating.",
            }]

        async def _no_stats(days=90):
            return {"api": 99.1}

        async def _no_updates(limit=20):
            return []

        monkeypatch.setattr(sp, "_fetch_active_incidents", _one_incident)
        monkeypatch.setattr(sp, "_fetch_uptime_stats", _no_stats)
        monkeypatch.setattr(sp, "_fetch_recent_updates", _no_updates)

        html_content = await sp.status_html_handler()
        assert "Incident #42" in html_content
        assert "MAJOR" in html_content
        assert "api" in html_content.lower()
        assert "API calls are delayed" in html_content

    @pytest.mark.asyncio
    async def test_rss_feed_contains_updates(self, initialized_db, monkeypatch):
        """RSS feed contains status_updates as items."""
        import src.app.status_page as sp
        monkeypatch.setattr(sp, "_cache", {})

        async def _no_incidents(product_id=None):
            return []

        async def _no_stats(days=90):
            return {}

        async def _two_updates(limit=20):
            return [
                {
                    "update_id": 1,
                    "product_id": "prod-x",
                    "incident_id": 42,
                    "posted_at": "2026-05-15 10:30:00",
                    "body_md": "We are monitoring the fix.",
                    "status_kind": "monitoring",
                },
                {
                    "update_id": 2,
                    "product_id": "prod-x",
                    "incident_id": 42,
                    "posted_at": "2026-05-15 11:00:00",
                    "body_md": "The issue is resolved.",
                    "status_kind": "resolved",
                },
            ]

        monkeypatch.setattr(sp, "_fetch_active_incidents", _no_incidents)
        monkeypatch.setattr(sp, "_fetch_uptime_stats", _no_stats)
        monkeypatch.setattr(sp, "_fetch_recent_updates", _two_updates)

        rss_content = await sp.status_rss_handler()
        assert '<?xml version="1.0"' in rss_content
        assert "<rss" in rss_content
        assert "Incident #42" in rss_content
        assert "monitoring" in rss_content.lower()
        assert "resolved" in rss_content.lower()

    @pytest.mark.asyncio
    async def test_rss_feed_empty_when_no_updates(self, initialized_db, monkeypatch):
        """RSS feed renders valid XML even when there are no updates."""
        import src.app.status_page as sp
        monkeypatch.setattr(sp, "_cache", {})

        async def _empty(**kwargs):
            return []

        async def _no_stats(days=90):
            return {}

        monkeypatch.setattr(sp, "_fetch_active_incidents", _empty)
        monkeypatch.setattr(sp, "_fetch_uptime_stats", _no_stats)
        monkeypatch.setattr(sp, "_fetch_recent_updates", _empty)

        rss_content = await sp.status_rss_handler()
        assert '<?xml version="1.0"' in rss_content
        assert "<channel>" in rss_content

    def test_cache_invalidation(self, monkeypatch):
        """invalidate_cache() clears the module-level cache."""
        import src.app.status_page as sp
        sp._cache["html"] = "<html>old</html>"
        sp._cache["rss"] = "<rss/>"
        sp._cache["fetched_at"] = 999999999.0

        sp.invalidate_cache()
        assert sp._cache == {}


# ===========================================================================
# 5. founder_action emitted pre-publish (incident_update_review handler)
# ===========================================================================

class TestIncidentUpdateReviewHandler:
    @pytest.mark.asyncio
    async def test_emits_founder_action_with_draft(self, initialized_db, monkeypatch):
        """incident_update_review handler emits a founder_action for the draft."""
        from general_beckman.posthook_handlers.incident_update_review import handle

        emitted = []

        async def _mock_emit(**kwargs):
            emitted.append(kwargs)
            return type("FA", (), {"id": 99})()

        import general_beckman.posthook_handlers.incident_update_review as _h
        monkeypatch.setattr(_h, "_emit_founder_action", _mock_emit)

        task = {
            "id": 1,
            "mission_id": 10,
            "context": json.dumps({}),
        }
        result_ctx = {
            "draft": "We are investigating payment delays.",
            "incident_id": 7,
            "product_id": "prod-gamma",
            "status_kind": "investigating",
        }

        result = await handle(task, result_ctx)
        assert result["status"] == "ok"
        assert result["founder_action_id"] == 99
        assert len(emitted) == 1
        assert emitted[0]["incident_id"] == 7
        assert emitted[0]["product_id"] == "prod-gamma"

    @pytest.mark.asyncio
    async def test_skips_when_no_draft(self, initialized_db, monkeypatch):
        """incident_update_review handler skips when draft is empty."""
        from general_beckman.posthook_handlers.incident_update_review import handle

        task = {"id": 1, "mission_id": 10, "context": "{}"}
        result = await handle(task, {})
        assert result["status"] == "skip"

    @pytest.mark.asyncio
    async def test_skips_when_no_incident_id(self, initialized_db, monkeypatch):
        """incident_update_review handler skips when incident_id missing."""
        from general_beckman.posthook_handlers.incident_update_review import handle

        task = {"id": 1, "mission_id": 10, "context": "{}"}
        result = await handle(task, {"draft": "Some text", "product_id": "p1"})
        assert result["status"] == "skip"

    @pytest.mark.asyncio
    async def test_posthook_payload_carries_draft_from_source_ctx(
        self, initialized_db, monkeypatch
    ):
        """_apply_request_posthook uses a.source_ctx (with draft) not DB ctx.

        This is the end-to-end regression test for Part C of the z7 fix:
        `_posthook_agent_and_payload` must receive the enriched source_ctx that
        rewrite.py builds by merging result scalars, not the raw DB-read ctx
        which has no 'draft' key.

        Without the fix (passing `ctx` instead of `a.source_ctx`), the payload
        produced by _posthook_agent_and_payload has `draft=""`, so the
        incident_update_review handler skips instead of emitting founder_action.
        """
        import json as _json
        from src.infra.db import add_task, get_task, update_task
        from general_beckman.result_router import RequestPostHook
        from general_beckman.apply import _posthook_agent_and_payload

        # Create a source task whose stored context does NOT contain 'draft'
        # (this mirrors the real situation: incident/draft_update stores draft
        # only in its result dict, never writes it back to the task context).
        source_task_id = await add_task(
            title="incident draft_update",
            description="",
            agent_type="mechanical",
            context={
                "incident_id": 42,
                "product_id": "prod-test",
                "status_kind": "investigating",
                # NOTE: no "draft" key here — this is the key invariant.
            },
        )
        source = await get_task(source_task_id)
        assert source is not None

        # Simulate what rewrite.py does: build a source_ctx that merges the
        # result scalar "draft" into the task context.
        source_ctx_enriched = {
            "incident_id": 42,
            "product_id": "prod-test",
            "status_kind": "investigating",
            "draft": "We are investigating an issue with payment processing.",
        }

        hook = RequestPostHook(
            source_task_id=source_task_id,
            kind="incident_update_review",
            source_ctx=source_ctx_enriched,
        )

        # Verify: the payload produced by _posthook_agent_and_payload carries
        # the draft from source_ctx_enriched, not the empty DB context.
        _, payload = _posthook_agent_and_payload(hook, source, source_ctx_enriched)
        inner = payload.get("payload", {})
        assert inner.get("draft") == "We are investigating an issue with payment processing.", (
            "payload.payload.draft must come from a.source_ctx (enriched with result scalars), "
            "not from the DB-read ctx which has no 'draft' key"
        )
        assert inner.get("incident_id") == 42
        assert inner.get("product_id") == "prod-test"

        # Counterproof: if we (incorrectly) pass the DB-read ctx instead, draft
        # would be empty — confirming the bug that was present before this fix.
        import json as _json
        from general_beckman.apply import _parse_ctx
        db_ctx = _parse_ctx(source)
        _, bad_payload = _posthook_agent_and_payload(hook, source, db_ctx)
        bad_inner = bad_payload.get("payload", {})
        assert bad_inner.get("draft", "") == "", (
            "Counterproof: DB-read ctx has no draft — confirms old path was broken"
        )


# ===========================================================================
# 6. incident/publish_status writes status_updates row + updates incidents
# ===========================================================================

class TestPublishStatus:
    @pytest.mark.asyncio
    async def test_publish_creates_status_update_row(self, initialized_db):
        """incident/publish_status inserts a status_updates row."""
        from src.infra.db import get_db
        from mr_roboto.incident_publish_status import run as publish_run

        db = await get_db()
        cur = await db.execute(
            "INSERT INTO incidents (product_id, severity, affected_components_json) "
            "VALUES (?, ?, ?)",
            ("prod-delta", "minor", '[]'),
        )
        await db.commit()
        incident_id = cur.lastrowid

        result = await publish_run({
            "incident_id": incident_id,
            "product_id": "prod-delta",
            "body_md": "We have identified the issue.",
            "status_kind": "identified",
        })

        assert result["status"] == "ok"
        assert "update_id" in result

        # Verify DB row.
        async with db.execute(
            "SELECT body_md, status_kind FROM status_updates WHERE update_id = ?",
            (result["update_id"],),
        ) as cur:
            row = await cur.fetchone()
        assert row is not None
        assert row[0] == "We have identified the issue."
        assert row[1] == "identified"

    @pytest.mark.asyncio
    async def test_publish_resolved_sets_resolved_at(self, initialized_db):
        """Publishing with status_kind='resolved' sets incidents.resolved_at."""
        from src.infra.db import get_db
        from mr_roboto.incident_publish_status import run as publish_run

        db = await get_db()
        cur = await db.execute(
            "INSERT INTO incidents (product_id, severity, affected_components_json) "
            "VALUES (?, ?, ?)",
            ("prod-epsilon", "major", '["auth"]'),
        )
        await db.commit()
        incident_id = cur.lastrowid

        result = await publish_run({
            "incident_id": incident_id,
            "product_id": "prod-epsilon",
            "body_md": "Auth service is fully restored.",
            "status_kind": "resolved",
        })
        assert result["status"] == "ok"

        async with db.execute(
            "SELECT resolved_at FROM incidents WHERE incident_id = ?",
            (incident_id,),
        ) as cur:
            row = await cur.fetchone()
        assert row[0] is not None, "resolved_at should be set on resolution"

    @pytest.mark.asyncio
    async def test_publish_calls_invalidate_cache(self, initialized_db, monkeypatch):
        """incident/publish_status calls status_page.invalidate_cache() after a successful publish.

        This is the real-path test for Critical 4: the verb's docstring claimed it
        invalidated the cache but the original code never called invalidate_cache().
        After the fix the real call must fire (not just be mocked at the seam).
        """
        from src.infra.db import get_db
        from mr_roboto.incident_publish_status import run as publish_run
        import src.app.status_page as sp

        # Pre-populate the cache so we can observe it being cleared.
        sp._cache["html"] = "<html>stale</html>"
        sp._cache["rss"] = "<rss/>"
        sp._cache["fetched_at"] = 999_999_999.0

        db = await get_db()
        cur = await db.execute(
            "INSERT INTO incidents (product_id, severity, affected_components_json) "
            "VALUES (?, ?, ?)",
            ("prod-cache-test", "minor", '[]'),
        )
        await db.commit()
        incident_id = cur.lastrowid

        result = await publish_run({
            "incident_id": incident_id,
            "product_id": "prod-cache-test",
            "body_md": "We have identified the issue.",
            "status_kind": "identified",
        })

        assert result["status"] == "ok"
        # The real invalidate_cache() must have been called — cache must be empty now.
        assert sp._cache == {}, (
            "publish_status must call invalidate_cache(); stale cache was still present"
        )

    @pytest.mark.asyncio
    async def test_publish_invalid_status_kind_returns_error(self, initialized_db):
        """incident/publish_status rejects unknown status_kind."""
        from mr_roboto.incident_publish_status import run as publish_run
        result = await publish_run({
            "incident_id": 999,
            "product_id": "prod-x",
            "body_md": "Some text",
            "status_kind": "unknown_kind",
        })
        assert result["status"] == "error"
        assert "status_kind" in result["error"]

    @pytest.mark.asyncio
    async def test_publish_unknown_incident_returns_error(self, initialized_db):
        """incident/publish_status returns error if incident not found."""
        from mr_roboto.incident_publish_status import run as publish_run
        result = await publish_run({
            "incident_id": 99999,
            "product_id": "prod-x",
            "body_md": "Some text",
            "status_kind": "monitoring",
        })
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_publish_empty_body_returns_error(self, initialized_db):
        """incident/publish_status rejects empty body_md."""
        from src.infra.db import get_db
        from mr_roboto.incident_publish_status import run as publish_run

        db = await get_db()
        cur = await db.execute(
            "INSERT INTO incidents (product_id, severity, affected_components_json) "
            "VALUES (?, ?, ?)",
            ("prod-zeta", "minor", '[]'),
        )
        await db.commit()
        incident_id = cur.lastrowid

        result = await publish_run({
            "incident_id": incident_id,
            "product_id": "prod-zeta",
            "body_md": "   ",
            "status_kind": "monitoring",
        })
        assert result["status"] == "error"
        assert "body_md" in result["error"]


# ===========================================================================
# 7. incident/draft_postmortem writes artifact + emits founder_action
# ===========================================================================

class TestDraftPostmortem:
    @pytest.mark.asyncio
    async def test_draft_postmortem_writes_markdown_file(
        self, initialized_db, tmp_path, monkeypatch
    ):
        """incident/draft_postmortem writes a .md file to workspace_path."""
        from src.infra.db import get_db
        from mr_roboto.incident_draft_postmortem import run as postmortem_run

        db = await get_db()
        from src.infra.times import db_now
        now = db_now()

        # Open and resolve incident.
        cur = await db.execute(
            "INSERT INTO incidents (product_id, severity, affected_components_json, "
            "customer_impact_summary, resolved_at, opened_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("prod-eta", "critical", '["checkout"]',
             "Checkout was unavailable for 2 hours.", now, now),
        )
        await db.commit()
        incident_id = cur.lastrowid

        # Add a status update.
        await db.execute(
            "INSERT INTO status_updates (product_id, incident_id, body_md, status_kind) "
            "VALUES (?, ?, ?, ?)",
            ("prod-eta", incident_id, "Checkout service is now restored.", "resolved"),
        )
        await db.commit()

        # Mock founder_action emission.
        emitted_fa = []
        async def _mock_emit_fa(**kwargs):
            emitted_fa.append(kwargs)
            return type("FA", (), {"id": 55})()

        import mr_roboto.incident_draft_postmortem as _pdm
        monkeypatch.setattr(_pdm, "_emit_founder_action", _mock_emit_fa)

        workspace = str(tmp_path / "workspace")
        result = await postmortem_run({
            "incident_id": incident_id,
            "product_id": "prod-eta",
            "workspace_path": workspace,
            "mission_id": 0,
        })

        assert result["status"] == "ok"
        artifact_path = result["artifact_path"]
        assert os.path.isfile(artifact_path), f"artifact should exist at {artifact_path}"

        content = open(artifact_path, encoding="utf-8").read()
        assert "Postmortem" in content
        assert "Checkout" in content
        assert "checkout" in content.lower()
        assert result["founder_action_id"] == 55
        assert len(emitted_fa) == 1

    @pytest.mark.asyncio
    async def test_draft_postmortem_missing_incident_returns_error(self, initialized_db):
        """incident/draft_postmortem returns error if incident not found."""
        from mr_roboto.incident_draft_postmortem import run as postmortem_run
        result = await postmortem_run({
            "incident_id": 88888,
            "product_id": "prod-x",
        })
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_draft_postmortem_missing_incident_id_returns_error(self, initialized_db):
        """incident/draft_postmortem returns error if incident_id missing."""
        from mr_roboto.incident_draft_postmortem import run as postmortem_run
        result = await postmortem_run({"product_id": "prod-x"})
        assert result["status"] == "error"
        assert "incident_id" in result["error"]


# ===========================================================================
# 8. incident_comms.json parses as valid JSON
# ===========================================================================

class TestMissionTemplate:
    def test_incident_comms_json_is_valid(self):
        """incident_comms.json parses as valid JSON with required keys."""
        import pathlib
        here = pathlib.Path(__file__).parent
        template_path = here.parent.parent / "src" / "workflows" / "incident_comms.json"
        assert template_path.exists(), f"Mission template not found at {template_path}"

        with open(template_path, encoding="utf-8") as fh:
            data = json.load(fh)

        assert data.get("plan_id") == "incident_comms"
        assert "phases" in data
        assert len(data["phases"]) >= 2

    def test_incident_comms_has_draft_and_publish_steps(self):
        """incident_comms.json contains draft_update and publish steps."""
        import pathlib
        here = pathlib.Path(__file__).parent
        template_path = here.parent.parent / "src" / "workflows" / "incident_comms.json"
        with open(template_path, encoding="utf-8") as fh:
            data = json.load(fh)

        all_step_ids = []
        for phase in data.get("phases", []):
            for step in phase.get("steps", []):
                all_step_ids.append(step.get("step_id", ""))

        assert any("draft" in s for s in all_step_ids), "Should have a draft step"
        assert any("publish" in s for s in all_step_ids), "Should have a publish step"
        assert any("postmortem" in s for s in all_step_ids), "Should have a postmortem step"


# ===========================================================================
# 9. Reversibility entries for B3 verbs
# ===========================================================================

class TestReversibilityRegistry:
    def test_incident_verbs_registered(self):
        """All B3 verbs have reversibility entries."""
        from mr_roboto.reversibility import VERB_REVERSIBILITY

        assert "incident/draft_update" in VERB_REVERSIBILITY
        assert "incident/publish_status" in VERB_REVERSIBILITY
        assert "incident/draft_postmortem" in VERB_REVERSIBILITY
        assert "incident_update_review" in VERB_REVERSIBILITY

    def test_publish_status_is_irreversible(self):
        """incident/publish_status is tagged irreversible (customers see it)."""
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        assert VERB_REVERSIBILITY["incident/publish_status"] == "irreversible"

    def test_draft_verbs_are_full(self):
        """Draft verbs are fully reversible (no external side-effect)."""
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        assert VERB_REVERSIBILITY["incident/draft_update"] == "full"
        assert VERB_REVERSIBILITY["incident/draft_postmortem"] == "full"

    def test_incident_update_review_is_irreversible(self):
        """incident_update_review is irreversible (Telegram message to founder)."""
        from mr_roboto.reversibility import VERB_REVERSIBILITY
        assert VERB_REVERSIBILITY["incident_update_review"] == "irreversible"
