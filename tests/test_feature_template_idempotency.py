"""Test feature-template duplicate-expansion idempotency (handoff P).

Mission 46 phase 8 had two waves of feat tasks (4015-4023 + 4040-onward)
because workflow_advance fired twice for the parent step (or a reset
re-triggered expansion). Each per-feature expansion produces tasks
titled ``[8.{fid}.<step>] ...``; a single SQL pre-check finds already-
expanded fids and skips them on subsequent calls.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.workflows.engine.hooks import _trigger_template_expansion


def _features_payload() -> str:
    import json
    return json.dumps([
        {"id": "feat_a", "name": "Feature A"},
        {"id": "feat_b", "name": "Feature B"},
    ])


def _fake_db_with_existing_titles(titles: list[str]) -> MagicMock:
    """Mock get_db returning a cursor that yields the given titles."""
    cursor = MagicMock()
    cursor.fetchall = AsyncMock(return_value=[(t,) for t in titles])
    cursor.close = AsyncMock()
    db = MagicMock()
    db.execute = AsyncMock(return_value=cursor)
    return db


@pytest.mark.asyncio
async def test_already_expanded_features_skipped(monkeypatch):
    """When tasks for [8.feat_a.*] already exist, expansion should
    skip feat_a but still process feat_b."""
    db = _fake_db_with_existing_titles([
        "[8.feat_a.write_code] do thing",
        "[8.feat_a.test_code] verify",
    ])

    async def _get_db():
        return db

    monkeypatch.setattr("src.infra.db.get_db", _get_db)
    monkeypatch.setattr("src.infra.db.get_mission", AsyncMock(return_value=None))

    inserted: list[dict] = []

    async def _insert(**kw):
        inserted.append(kw)
        return len(inserted) + 1000

    monkeypatch.setattr("src.infra.db.add_task", _insert)
    monkeypatch.setattr("src.infra.db.update_task", AsyncMock())

    fake_template = [{"id": "write_code", "name": "write {feature_name}"}]
    fake_wf = MagicMock()
    fake_wf.get_template = MagicMock(return_value=fake_template)

    monkeypatch.setattr(
        "src.workflows.engine.loader.load_workflow",
        lambda _name: fake_wf,
    )
    monkeypatch.setattr(
        "src.workflows.engine.expander.expand_template",
        lambda *a, **kw: [{"id": "x", "name": "y", "instruction": "i", "agent": "coder"}],
    )
    monkeypatch.setattr(
        "src.workflows.engine.expander.expand_steps_to_tasks",
        lambda *a, **kw: [{
            "title": "[8.feat_b.write_code] do",
            "description": "x",
            "agent_type": "coder",
            "mission_id": 1,
            "context": {},
            "depends_on": [],
        }],
    )

    await _trigger_template_expansion(mission_id=1, backlog_text=_features_payload())

    # feat_a should be skipped; feat_b should be inserted.
    titles = [r["title"] for r in inserted]
    assert any("feat_b" in t for t in titles), (
        "feat_b should still be expanded"
    )
    assert not any("feat_a" in t for t in titles), (
        "feat_a should be skipped (already expanded)"
    )


@pytest.mark.asyncio
async def test_first_run_expands_all(monkeypatch):
    """No existing rows -> both features expand normally."""
    db = _fake_db_with_existing_titles([])

    async def _get_db():
        return db

    monkeypatch.setattr("src.infra.db.get_db", _get_db)
    monkeypatch.setattr("src.infra.db.get_mission", AsyncMock(return_value=None))

    inserted: list[dict] = []

    async def _insert(**kw):
        inserted.append(kw)
        return len(inserted) + 1000

    monkeypatch.setattr("src.infra.db.add_task", _insert)
    monkeypatch.setattr("src.infra.db.update_task", AsyncMock())

    fake_template = [{"id": "write_code", "name": "write"}]
    fake_wf = MagicMock()
    fake_wf.get_template = MagicMock(return_value=fake_template)

    monkeypatch.setattr(
        "src.workflows.engine.loader.load_workflow",
        lambda _name: fake_wf,
    )
    monkeypatch.setattr(
        "src.workflows.engine.expander.expand_template",
        lambda *a, **kw: [{"id": "x", "name": "y", "instruction": "i", "agent": "coder"}],
    )

    counter = {"n": 0}

    def _expand_steps(*a, **kw):
        counter["n"] += 1
        return [{
            "title": f"[8.feat_{counter['n']}.write_code] do",
            "description": "x",
            "agent_type": "coder",
            "mission_id": 1,
            "context": {},
            "depends_on": [],
        }]

    monkeypatch.setattr(
        "src.workflows.engine.expander.expand_steps_to_tasks",
        _expand_steps,
    )

    await _trigger_template_expansion(
        mission_id=2, backlog_text=_features_payload(),
    )
    # Two per-feature expansions + the cross-feature integration test.
    feat_inserts = [r for r in inserted if r["agent_type"] == "coder"]
    assert len(feat_inserts) == 2, (
        f"expected 2 feature inserts, got {len(feat_inserts)}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
