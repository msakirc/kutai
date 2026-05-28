"""SP2 Task 5: faq_regen._llm_cluster_draft CPS migration + restart-reconcile E2E."""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock

import src.infra.db as _db_mod


async def _fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "faq.db"))
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
    monkeypatch.setattr(_db_mod, "_db_connection", None)
    await _db_mod.init_db()


async def _close_db():
    if _db_mod._db_connection is not None:
        await _db_mod._db_connection.close()
        _db_mod._db_connection = None


@pytest.mark.asyncio
async def test_enqueue_cluster_draft_uses_cps(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        captured = {}

        async def fake_enqueue(spec, **kwargs):
            captured["spec"] = spec
            captured["kwargs"] = kwargs
            return 8001

        monkeypatch.setattr("general_beckman.enqueue", fake_enqueue)

        from src.app.jobs.faq_regen import enqueue_cluster_draft
        cluster = [{"question": "Q1", "answer": "A1"}] * 5
        cid = await enqueue_cluster_draft(cluster, lang="en")
        assert cid == 8001
        assert captured["kwargs"].get("await_inline") in (False, None)
        assert captured["kwargs"]["on_complete"] == "faq_regen.draft_persist_resume"
        assert captured["kwargs"]["on_error"] == "faq_regen.draft_persist_err"
        cs = captured["kwargs"]["cont_state"]
        assert cs["lang"] == "en"
        assert cs["cluster_size"] == 5
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_draft_resume_emits_founder_action(tmp_path, monkeypatch):
    """Resume must parse JSON content and call _emit_faq_founder_action."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        emissions = []

        async def fake_emit(*, mission_id, entry, cluster_size):
            emissions.append({
                "mission_id": mission_id,
                "entry": entry,
                "cluster_size": cluster_size,
            })

        monkeypatch.setattr(
            "src.app.jobs.faq_regen._emit_faq_founder_action", fake_emit,
        )

        from src.app.jobs.faq_regen import _draft_persist_resume
        await _draft_persist_resume(
            child_task_id=8001,
            result={"status": "completed", "result": {"content": json.dumps({
                "question": "How do I reset my password?",
                "answer": "Click 'Forgot password' on the login page.",
            })}},
            state={"lang": "en", "cluster_size": 5},
        )
        assert emissions, "resume must have emitted a founder action"
        entry = emissions[0]["entry"]
        assert entry["question"] == "How do I reset my password?"
        assert entry["lang"] == "en"
        assert emissions[0]["cluster_size"] == 5
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_draft_resume_no_emit_when_content_unparseable(tmp_path, monkeypatch):
    """When content has no JSON the resume must NOT emit (silently no-op)."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        emissions = []

        async def fake_emit(*, mission_id, entry, cluster_size):
            emissions.append(entry)

        monkeypatch.setattr(
            "src.app.jobs.faq_regen._emit_faq_founder_action", fake_emit,
        )

        from src.app.jobs.faq_regen import _draft_persist_resume
        await _draft_persist_resume(
            child_task_id=8002,
            result={"status": "completed",
                    "result": {"content": "I cannot draft an FAQ — no useful content."}},
            state={"lang": "en", "cluster_size": 5},
        )
        assert emissions == []
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_draft_on_error_no_emit(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        emissions = []

        async def fake_emit(*, mission_id, entry, cluster_size):
            emissions.append(entry)

        monkeypatch.setattr(
            "src.app.jobs.faq_regen._emit_faq_founder_action", fake_emit,
        )

        from src.app.jobs.faq_regen import _draft_persist_err
        await _draft_persist_err(
            child_task_id=8003,
            result={"status": "failed", "error": "timeout"},
            state={"lang": "en", "cluster_size": 5},
        )
        assert emissions == []
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_register_continuations_registers_faq_handlers(tmp_path, monkeypatch):
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from src.app.jobs.faq_regen import register_continuations
        from general_beckman.continuations import _HANDLERS
        _HANDLERS.pop("faq_regen.draft_persist_resume", None)
        _HANDLERS.pop("faq_regen.draft_persist_err", None)
        register_continuations()
        assert "faq_regen.draft_persist_resume" in _HANDLERS
        assert "faq_regen.draft_persist_err" in _HANDLERS
    finally:
        await _close_db()


@pytest.mark.asyncio
async def test_faq_regen_module_in_handler_modules():
    from general_beckman.continuations import _HANDLER_MODULES
    assert "src.app.jobs.faq_regen" in _HANDLER_MODULES


# ─── Restart-reconcile end-to-end (mandatory SP2 verdict round-trip) ────────


@pytest.mark.asyncio
async def test_faq_regen_restart_reconcile_fires_resume(tmp_path, monkeypatch):
    """E2E CPS-on-restart: child terminal while orchestrator down → reconcile
    fires the resume."""
    await _fresh_db(tmp_path, monkeypatch)
    try:
        from src.app.jobs.faq_regen import enqueue_cluster_draft
        cluster = [{"question": "Q1", "answer": "A1"}] * 5
        child_id = await enqueue_cluster_draft(cluster, lang="en")
        assert isinstance(child_id, int)

        db = await _db_mod.get_db()
        # Confirm continuation row created.
        cur = await db.execute(
            "SELECT status FROM continuations WHERE child_task_id=?", (child_id,))
        row = await cur.fetchone()
        assert row is not None
        assert row[0] == "pending"

        # Set child terminal — outside the orchestrator's claw.
        await db.execute(
            "UPDATE tasks SET status='completed', result=? WHERE id=?",
            (json.dumps({"content": json.dumps({
                "question": "Reconciled Q?",
                "answer": "Reconciled A.",
            })}), child_id),
        )
        await db.commit()

        # Simulate restart — clear in-memory registry then re-register +
        # reconcile (production sequence in src.app.run / orchestrator).
        from general_beckman.continuations import (
            _HANDLERS, register_startup_handlers, reconcile_continuations,
        )
        _HANDLERS.clear()

        emissions = []

        async def fake_emit(*, mission_id, entry, cluster_size):
            emissions.append((entry, cluster_size))

        monkeypatch.setattr(
            "src.app.jobs.faq_regen._emit_faq_founder_action", fake_emit,
        )

        register_startup_handlers()
        await reconcile_continuations()
        # Handler dispatch is detached (asyncio.create_task); yield a tick
        # so it runs before we assert.
        await asyncio.sleep(0.3)

        assert emissions, "reconcile must have fired the resume"
        entry, size = emissions[0]
        assert entry["question"] == "Reconciled Q?"
        assert entry["lang"] == "en"
        assert size == 5
    finally:
        await _close_db()
