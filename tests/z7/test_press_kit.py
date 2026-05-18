"""Z7 T3C — A4 Press kit tests (versioned binary store + audience-segmented variants).

Covers:
  1. press_kit/assemble: builds manifest with 4 audience variants
     (investor / journalist / partner / candidate), calls LLM for one-pager
     draft, emits founder_action pre-publish.
  2. press_kit/publish: uploads (mocked S3 or local fallback), returns
     permanent URLs per audience, retains old versions.
  3. press_kit_freshness posthook: flags a kit >90 days old when spec has
     changed; emits founder_action; skips when kit is fresh.
  4. press_kit_quotes: rows can be inserted and queried per product.
  5. DB migration: press_kits and press_kit_quotes tables exist after init_db.
"""
from __future__ import annotations

import json
import os
import zipfile
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AUDIENCE_VARIANTS = ("investor", "journalist", "partner", "candidate")


# ===========================================================================
# 1. press_kit/assemble verb
# ===========================================================================

class TestPressKitAssemble:
    """Tests for press_kit/assemble mr_roboto verb."""

    @pytest.mark.asyncio
    async def test_assemble_returns_manifest_with_four_audiences(
        self, tmp_path, monkeypatch
    ):
        """Assemble builds a manifest with 4 audience variants."""
        from mr_roboto.press_kit_assemble import run as assemble

        # Mock LLM call for one-pager draft
        async def _mock_llm(spec_text, audience):
            return f"One-pager for {audience}: {spec_text[:30]}"

        monkeypatch.setattr(
            "mr_roboto.press_kit_assemble._draft_one_pager_llm", _mock_llm
        )

        # Mock founder_action emission (pre-publish gate)
        emitted_actions = []

        async def _mock_emit_fa(**kwargs):
            emitted_actions.append(kwargs)
            return type("FA", (), {"id": 42})()

        monkeypatch.setattr(
            "mr_roboto.press_kit_assemble._emit_founder_action", _mock_emit_fa
        )

        # Mock version lookup — no DB in unit tests
        async def _mock_get_latest_version(product_id):
            return 0

        monkeypatch.setattr(
            "mr_roboto.press_kit_assemble._get_latest_version",
            _mock_get_latest_version,
        )

        result = await assemble(
            mission_id=1,
            product_id="prod_abc",
            spec_text="A SaaS tool for busy founders.",
            workspace_path=str(tmp_path),
            logo_path="",
            screenshot_paths=[],
            founder_bio="Alice, 15 years in SaaS.",
            fact_sheet_md="- Founded 2025\n- 10k users",
            quotes=[],
            past_mentions=[],
        )

        assert result["ok"], result.get("error")
        manifest = result["manifest"]

        # All 4 audience variants present
        for audience in AUDIENCE_VARIANTS:
            assert audience in manifest["variants"], \
                f"Missing audience variant: {audience}"
            v = manifest["variants"][audience]
            assert v["zip_path"], f"No zip_path for {audience}"
            assert os.path.exists(v["zip_path"]), \
                f"Zip not created for {audience}: {v['zip_path']}"

        # Version is an integer >= 1
        assert isinstance(manifest["version"], int)
        assert manifest["version"] >= 1

        # product_id on manifest
        assert manifest["product_id"] == "prod_abc"

        # founder_action emitted
        assert len(emitted_actions) == 1
        assert emitted_actions[0]["product_id"] == "prod_abc"

    @pytest.mark.asyncio
    async def test_assemble_zips_contain_audience_specific_content(
        self, tmp_path, monkeypatch
    ):
        """Each audience zip has a one-pager.md with audience-specific content."""
        from mr_roboto.press_kit_assemble import run as assemble

        async def _mock_llm(spec_text, audience):
            return f"ONE_PAGER_{audience.upper()}"

        monkeypatch.setattr(
            "mr_roboto.press_kit_assemble._draft_one_pager_llm", _mock_llm
        )

        async def _mock_emit_fa(**kwargs):
            return type("FA", (), {"id": 1})()

        monkeypatch.setattr(
            "mr_roboto.press_kit_assemble._emit_founder_action", _mock_emit_fa
        )

        async def _mock_get_latest_version2(product_id):
            return 0

        monkeypatch.setattr(
            "mr_roboto.press_kit_assemble._get_latest_version",
            _mock_get_latest_version2,
        )

        result = await assemble(
            mission_id=1,
            product_id="prod_test",
            spec_text="spec",
            workspace_path=str(tmp_path),
            logo_path="",
            screenshot_paths=[],
            founder_bio="",
            fact_sheet_md="",
            quotes=[],
            past_mentions=[],
        )

        assert result["ok"]
        for audience in AUDIENCE_VARIANTS:
            zip_path = result["manifest"]["variants"][audience]["zip_path"]
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
                # one-pager.md must be present in every variant
                assert "one_pager.md" in names, \
                    f"one_pager.md missing in {audience} zip; got: {names}"
                content = zf.read("one_pager.md").decode()
                assert audience.upper() in content, \
                    f"Audience not in one_pager.md for {audience}: {content[:200]}"

    @pytest.mark.asyncio
    async def test_assemble_increments_version(self, tmp_path, monkeypatch):
        """Second assemble call for same product increments version."""
        from mr_roboto.press_kit_assemble import run as assemble

        async def _mock_llm(spec_text, audience):
            return "draft"

        monkeypatch.setattr(
            "mr_roboto.press_kit_assemble._draft_one_pager_llm", _mock_llm
        )

        async def _mock_emit_fa(**kwargs):
            return type("FA", (), {"id": 1})()

        monkeypatch.setattr(
            "mr_roboto.press_kit_assemble._emit_founder_action", _mock_emit_fa
        )

        # Mock version lookup — first call returns None (no prior kit)
        # second returns version 1
        _calls = []

        async def _mock_get_latest_version(product_id):
            v = len(_calls)
            _calls.append(v)
            return v

        monkeypatch.setattr(
            "mr_roboto.press_kit_assemble._get_latest_version",
            _mock_get_latest_version,
        )

        r1 = await assemble(
            mission_id=1,
            product_id="prod_incr",
            spec_text="v1",
            workspace_path=str(tmp_path),
            logo_path="",
            screenshot_paths=[],
            founder_bio="",
            fact_sheet_md="",
            quotes=[],
            past_mentions=[],
        )
        r2 = await assemble(
            mission_id=1,
            product_id="prod_incr",
            spec_text="v2",
            workspace_path=str(tmp_path),
            logo_path="",
            screenshot_paths=[],
            founder_bio="",
            fact_sheet_md="",
            quotes=[],
            past_mentions=[],
        )

        assert r1["ok"] and r2["ok"]
        assert r2["manifest"]["version"] > r1["manifest"]["version"]


# ===========================================================================
# 2. press_kit/publish verb
# ===========================================================================

class TestPressKitPublish:
    """Tests for press_kit/publish mr_roboto verb."""

    @pytest.mark.asyncio
    async def test_publish_local_fallback_returns_permanent_urls(
        self, tmp_path, monkeypatch
    ):
        """publish returns permanent URLs for each audience (local fallback)."""
        from mr_roboto.press_kit_publish import run as publish

        # Build a minimal manifest (4 audience zips already on disk)
        variants = {}
        for audience in AUDIENCE_VARIANTS:
            zip_path = str(tmp_path / f"press_kit_v1_{audience}.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("one_pager.md", f"# {audience}")
            variants[audience] = {"zip_path": zip_path}

        manifest = {
            "product_id": "prod_pub",
            "version": 1,
            "variants": variants,
            "created_at": "2026-05-15 12:00:00",
        }

        # Mock DB persist
        async def _mock_persist(product_id, version, mission_id, manifest_json,
                                 urls_json):
            pass

        monkeypatch.setattr(
            "mr_roboto.press_kit_publish._persist_kit", _mock_persist
        )

        # Use tmp_path as local store root
        monkeypatch.setenv("PRESS_KIT_BUCKET", "")
        monkeypatch.setattr(
            "mr_roboto.press_kit_publish.LOCAL_STORE_ROOT", str(tmp_path / "store")
        )

        result = await publish(
            mission_id=1,
            product_id="prod_pub",
            manifest=manifest,
        )

        assert result["ok"], result.get("error")
        assert "urls" in result
        for audience in AUDIENCE_VARIANTS:
            assert audience in result["urls"], f"No URL for {audience}"
            url = result["urls"][audience]
            # URL format: /press-kit/v{N}/{audience}/
            assert f"/v{manifest['version']}/{audience}/" in url, \
                f"Unexpected URL shape for {audience}: {url}"

    @pytest.mark.asyncio
    async def test_publish_retains_older_version(self, tmp_path, monkeypatch):
        """publish does not delete older version zips."""
        from mr_roboto.press_kit_publish import run as publish

        store_root = str(tmp_path / "store")
        monkeypatch.setenv("PRESS_KIT_BUCKET", "")
        monkeypatch.setattr("mr_roboto.press_kit_publish.LOCAL_STORE_ROOT", store_root)

        async def _mock_persist(*args, **kwargs):
            pass

        monkeypatch.setattr("mr_roboto.press_kit_publish._persist_kit", _mock_persist)

        # Publish v1 first
        variants_v1 = {}
        for audience in AUDIENCE_VARIANTS:
            zip_path = str(tmp_path / f"v1_{audience}.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("one_pager.md", "v1")
            variants_v1[audience] = {"zip_path": zip_path}

        await publish(
            mission_id=1,
            product_id="prod_retain",
            manifest={"product_id": "prod_retain", "version": 1, "variants": variants_v1, "created_at": "2026-05-15 10:00:00"},
        )

        # Publish v2
        variants_v2 = {}
        for audience in AUDIENCE_VARIANTS:
            zip_path = str(tmp_path / f"v2_{audience}.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("one_pager.md", "v2")
            variants_v2[audience] = {"zip_path": zip_path}

        await publish(
            mission_id=1,
            product_id="prod_retain",
            manifest={"product_id": "prod_retain", "version": 2, "variants": variants_v2, "created_at": "2026-05-15 12:00:00"},
        )

        # v1 zips should still be on disk under store_root
        v1_dir = os.path.join(store_root, "prod_retain", "v1")
        assert os.path.isdir(v1_dir), f"v1 dir gone after v2 publish: {v1_dir}"

    @pytest.mark.asyncio
    async def test_publish_superseded_stub(self, tmp_path, monkeypatch):
        """Older published version gets a 'see latest' stub marker."""
        from mr_roboto.press_kit_publish import run as publish

        store_root = str(tmp_path / "store")
        monkeypatch.setenv("PRESS_KIT_BUCKET", "")
        monkeypatch.setattr("mr_roboto.press_kit_publish.LOCAL_STORE_ROOT", store_root)

        async def _mock_persist(*args, **kwargs):
            pass

        monkeypatch.setattr("mr_roboto.press_kit_publish._persist_kit", _mock_persist)

        # Publish v1
        variants_v1 = {}
        for audience in AUDIENCE_VARIANTS:
            zip_path = str(tmp_path / f"sv1_{audience}.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("one_pager.md", "v1")
            variants_v1[audience] = {"zip_path": zip_path}

        await publish(
            mission_id=1,
            product_id="prod_stub",
            manifest={"product_id": "prod_stub", "version": 1, "variants": variants_v1, "created_at": "2026-05-15 10:00:00"},
        )

        # Publish v2
        variants_v2 = {}
        for audience in AUDIENCE_VARIANTS:
            zip_path = str(tmp_path / f"sv2_{audience}.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("one_pager.md", "v2")
            variants_v2[audience] = {"zip_path": zip_path}

        r2 = await publish(
            mission_id=1,
            product_id="prod_stub",
            manifest={"product_id": "prod_stub", "version": 2, "variants": variants_v2, "created_at": "2026-05-15 12:00:00"},
        )

        assert r2["ok"]
        # v1 stub marker should exist
        stub_path = os.path.join(store_root, "prod_stub", "v1", "_superseded.json")
        assert os.path.exists(stub_path), f"superseded stub not found at {stub_path}"
        stub = json.loads(open(stub_path).read())
        assert "latest_version" in stub
        assert stub["latest_version"] == 2


# ===========================================================================
# 3. press_kit_freshness posthook
# ===========================================================================

class TestPressKitFreshnessPosthook:
    """Tests for press_kit_freshness posthook handler."""

    @pytest.mark.asyncio
    async def test_freshness_flags_stale_kit(self, monkeypatch):
        """Kit >90 days old with spec change emits founder_action."""
        from general_beckman.posthook_handlers.press_kit_freshness import handle

        # Latest kit is 100 days old
        async def _mock_get_latest_kit(product_id):
            from datetime import datetime, timedelta
            old_ts = (datetime.utcnow() - timedelta(days=100)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            return {
                "kit_id": 1,
                "product_id": product_id,
                "version": 1,
                "created_at": old_ts,
                "manifest_json": json.dumps({"spec_hash": "old_hash"}),
            }

        monkeypatch.setattr(
            "general_beckman.posthook_handlers.press_kit_freshness._get_latest_kit",
            _mock_get_latest_kit,
        )

        # Spec has changed (different hash)
        async def _mock_get_spec_hash(product_id):
            return "new_hash"

        monkeypatch.setattr(
            "general_beckman.posthook_handlers.press_kit_freshness._get_spec_hash",
            _mock_get_spec_hash,
        )

        emitted = []

        async def _mock_emit_fa(**kwargs):
            emitted.append(kwargs)
            return type("FA", (), {"id": 99})()

        monkeypatch.setattr(
            "general_beckman.posthook_handlers.press_kit_freshness._emit_founder_action",
            _mock_emit_fa,
        )

        task = {"id": 1, "mission_id": 10, "context": json.dumps({"product_id": "prod_stale"})}
        result = await handle(task, {})

        assert result["status"] == "flagged"
        assert len(emitted) == 1
        assert emitted[0]["product_id"] == "prod_stale"

    @pytest.mark.asyncio
    async def test_freshness_skips_fresh_kit(self, monkeypatch):
        """Kit <90 days old skips founder_action emission."""
        from general_beckman.posthook_handlers.press_kit_freshness import handle

        async def _mock_get_latest_kit(product_id):
            from datetime import datetime, timedelta
            fresh_ts = (datetime.utcnow() - timedelta(days=30)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            return {
                "kit_id": 1,
                "product_id": product_id,
                "version": 1,
                "created_at": fresh_ts,
                "manifest_json": json.dumps({"spec_hash": "same_hash"}),
            }

        monkeypatch.setattr(
            "general_beckman.posthook_handlers.press_kit_freshness._get_latest_kit",
            _mock_get_latest_kit,
        )

        async def _mock_get_spec_hash(product_id):
            return "same_hash"

        monkeypatch.setattr(
            "general_beckman.posthook_handlers.press_kit_freshness._get_spec_hash",
            _mock_get_spec_hash,
        )

        task = {"id": 1, "mission_id": 10, "context": json.dumps({"product_id": "prod_fresh"})}
        result = await handle(task, {})

        assert result["status"] == "ok"
        assert result.get("reason") == "fresh"

    @pytest.mark.asyncio
    async def test_freshness_skips_when_no_kit(self, monkeypatch):
        """No existing kit → skip (no founder_action)."""
        from general_beckman.posthook_handlers.press_kit_freshness import handle

        async def _mock_get_latest_kit(product_id):
            return None

        monkeypatch.setattr(
            "general_beckman.posthook_handlers.press_kit_freshness._get_latest_kit",
            _mock_get_latest_kit,
        )

        task = {"id": 1, "mission_id": 10, "context": json.dumps({"product_id": "prod_none"})}
        result = await handle(task, {})

        assert result["status"] == "skip"

    @pytest.mark.asyncio
    async def test_freshness_skips_when_no_product_id(self):
        """No product_id in context → skip gracefully."""
        from general_beckman.posthook_handlers.press_kit_freshness import handle

        task = {"id": 1, "mission_id": 10, "context": "{}"}
        result = await handle(task, {})

        assert result["status"] == "skip"


# ===========================================================================
# 4. press_kit_quotes table
# ===========================================================================

class TestPressKitQuotes:
    """Tests for press_kit_quotes DB operations."""

    @pytest.mark.asyncio
    async def test_insert_and_fetch_quote(self, tmp_path, monkeypatch):
        """press_kit_quotes rows can be inserted and retrieved per product."""
        import aiosqlite

        db_path = str(tmp_path / "test.db")

        async with aiosqlite.connect(db_path) as db:
            # Create the tables (subset needed for this test)
            await db.execute(
                "CREATE TABLE press_kit_quotes ("
                " quote_id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " product_id TEXT NOT NULL,"
                " kit_id INTEGER,"
                " source_kind TEXT NOT NULL,"
                " speaker TEXT,"
                " body TEXT NOT NULL,"
                " approved INTEGER NOT NULL DEFAULT 0,"
                " created_at TEXT DEFAULT (datetime('now'))"
                ")"
            )
            await db.commit()

            await db.execute(
                "INSERT INTO press_kit_quotes "
                "(product_id, kit_id, source_kind, speaker, body, approved) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("prod_q", 1, "interview", "Jane Doe, CEO", "Best tool I've used.", 1),
            )
            await db.commit()

            cur = await db.execute(
                "SELECT body FROM press_kit_quotes WHERE product_id = ?",
                ("prod_q",),
            )
            row = await cur.fetchone()
            assert row is not None
            assert row[0] == "Best tool I've used."


# ===========================================================================
# 5. DB migration: tables exist after init_db
# ===========================================================================

class TestPressKitMigration:
    """Verify press_kits + press_kit_quotes tables are created by init_db."""

    @pytest.mark.asyncio
    async def test_press_kits_table_exists(self, tmp_path, monkeypatch):
        """After init_db, press_kits table has expected columns."""
        import aiosqlite
        import src.infra.db as _db_mod

        db_path = str(tmp_path / "migrate_test.db")

        # Patch DB_PATH in the db module's namespace and reset the connection
        # cache so init_db opens a fresh connection to our temp DB.
        monkeypatch.setattr(_db_mod, "DB_PATH", db_path, raising=False)
        _db_mod._db_connection = None
        _db_mod._db_connection_path = None

        await _db_mod.init_db()

        async with aiosqlite.connect(db_path) as db:
            cur = await db.execute("PRAGMA table_info(press_kits)")
            cols = {row[1] for row in await cur.fetchall()}

        assert "kit_id" in cols
        assert "product_id" in cols
        assert "version" in cols
        assert "mission_id" in cols
        assert "manifest_json" in cols
        assert "published_url" in cols
        assert "created_at" in cols

    @pytest.mark.asyncio
    async def test_press_kit_quotes_table_exists(self, tmp_path, monkeypatch):
        """After init_db, press_kit_quotes table has expected columns."""
        import aiosqlite
        import src.infra.db as _db_mod

        db_path = str(tmp_path / "migrate_test2.db")

        monkeypatch.setattr(_db_mod, "DB_PATH", db_path, raising=False)
        _db_mod._db_connection = None
        _db_mod._db_connection_path = None

        await _db_mod.init_db()

        async with aiosqlite.connect(db_path) as db:
            cur = await db.execute("PRAGMA table_info(press_kit_quotes)")
            cols = {row[1] for row in await cur.fetchall()}

        assert "quote_id" in cols
        assert "product_id" in cols
        assert "kit_id" in cols
        assert "body" in cols
        assert "approved" in cols


# ===========================================================================
# 6. mr_roboto dispatch routing
# ===========================================================================

class TestMrRobotoDispatch:
    """Verify press_kit/assemble and press_kit/publish route correctly."""

    @pytest.mark.asyncio
    async def test_assemble_routes_via_mr_roboto(self, tmp_path, monkeypatch):
        """mr_roboto.run dispatches press_kit/assemble to the action module."""
        import mr_roboto

        async def _mock_assemble(**kwargs):
            return {"ok": True, "manifest": {"version": 1, "product_id": "x", "variants": {a: {"zip_path": ""} for a in AUDIENCE_VARIANTS}}}

        monkeypatch.setattr("mr_roboto.press_kit_assemble.run", _mock_assemble)

        task = {
            "id": 1,
            "mission_id": 1,
            "context": json.dumps({"executor": "mechanical", "payload": {"action": "press_kit/assemble", "product_id": "x", "spec_text": "s", "workspace_path": str(tmp_path)}}),
            "payload": {"action": "press_kit/assemble", "product_id": "x", "spec_text": "s", "workspace_path": str(tmp_path)},
        }

        result = await mr_roboto.run(task)
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_publish_routes_via_mr_roboto(self, tmp_path, monkeypatch):
        """mr_roboto.run dispatches press_kit/publish to the action module."""
        import mr_roboto

        manifest = {
            "product_id": "x",
            "version": 1,
            "variants": {a: {"zip_path": ""} for a in AUDIENCE_VARIANTS},
            "created_at": "2026-05-15 12:00:00",
        }

        async def _mock_publish(**kwargs):
            return {
                "ok": True,
                "urls": {a: f"/press-kit/v1/{a}/" for a in AUDIENCE_VARIANTS},
            }

        monkeypatch.setattr("mr_roboto.press_kit_publish.run", _mock_publish)

        task = {
            "id": 2,
            "mission_id": 1,
            "context": json.dumps({"executor": "mechanical", "payload": {"action": "press_kit/publish", "product_id": "x", "manifest": manifest}}),
            "payload": {"action": "press_kit/publish", "product_id": "x", "manifest": manifest},
        }

        result = await mr_roboto.run(task)
        assert result.status == "completed"
        assert "urls" in result.result
