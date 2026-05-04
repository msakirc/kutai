"""Coverage for _trigger_template_expansion bound + dedup + halt behaviour.

Mission 57 wedge: a 1-entry backlog produced 56 expanded features in
phase 8. Schema validator now blocks bad backlogs upstream, but the
expander is defence-in-depth: cap, dedup, halt-on-empty.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.workflows.engine.hooks import _trigger_template_expansion


def _fake_db(titles: list[str]):
    cursor = MagicMock()
    cursor.fetchall = AsyncMock(return_value=[(t,) for t in titles])
    cursor.close = AsyncMock()
    db = MagicMock()
    db.execute = AsyncMock(return_value=cursor)
    return db


def _patch_workflow_machinery(monkeypatch, inserted: list[dict], expanded_fids: list[str]):
    monkeypatch.setattr("src.infra.db.get_db", AsyncMock(return_value=_fake_db([])))
    monkeypatch.setattr("src.infra.db.get_mission", AsyncMock(return_value=None))

    async def _insert(**kw):
        inserted.append(kw)
        return len(inserted) + 1000

    monkeypatch.setattr("src.infra.db.add_task", _insert)
    monkeypatch.setattr("src.infra.db.update_task", AsyncMock())

    fake_wf = MagicMock()
    fake_wf.get_template = MagicMock(return_value=[{"id": "x"}])
    monkeypatch.setattr(
        "src.workflows.engine.loader.load_workflow",
        lambda _name: fake_wf,
    )

    def _capture_expand(template, params=None, prefix=""):
        # Record the fid each time expand_template is called — one call per
        # feature that survived dedup + cap. Integration-step insertion
        # below does not call expand_template, so this counter is clean.
        if params and "feature_id" in params:
            expanded_fids.append(params["feature_id"])
        return [{"id": "x"}]

    monkeypatch.setattr(
        "src.workflows.engine.expander.expand_template",
        _capture_expand,
    )
    # One task per feature; the title encodes the fid so we can assert
    # which features actually got inserted.
    def _expand_to_tasks(expanded, mission_id, initial_context, **kw):
        # Caller provides the prefix indirectly via expanded steps; in our
        # fake we just record an empty marker. The test inspects insert call
        # ``title`` from prefix the real code passes into expand_template; in
        # this fake we just track count.
        return [{
            "title": "[stub]",
            "description": "x",
            "agent_type": "coder",
            "mission_id": mission_id,
            "context": {},
            "depends_on": [],
        }]

    monkeypatch.setattr(
        "src.workflows.engine.expander.expand_steps_to_tasks",
        _expand_to_tasks,
    )


# ── Halt-on-empty / non-list ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_array_halts_no_expansion(monkeypatch):
    inserted: list[dict] = []
    expanded_fids: list[str] = []
    _patch_workflow_machinery(monkeypatch, inserted, expanded_fids)
    await _trigger_template_expansion(mission_id=99, backlog_text="[]")
    assert inserted == []
    assert expanded_fids == []


@pytest.mark.asyncio
async def test_non_list_halts(monkeypatch):
    inserted: list[dict] = []
    expanded_fids: list[str] = []
    _patch_workflow_machinery(monkeypatch, inserted, expanded_fids)
    await _trigger_template_expansion(
        mission_id=99,
        backlog_text=json.dumps({"not": "a list"}),
    )
    assert inserted == []
    assert expanded_fids == []


@pytest.mark.asyncio
async def test_garbage_json_halts(monkeypatch):
    inserted: list[dict] = []
    expanded_fids: list[str] = []
    _patch_workflow_machinery(monkeypatch, inserted, expanded_fids)
    await _trigger_template_expansion(mission_id=99, backlog_text="{not json")
    assert inserted == []
    assert expanded_fids == []


# ── Dedup ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_duplicate_fids_collapsed_to_first(monkeypatch):
    inserted: list[dict] = []
    expanded_fids: list[str] = []
    _patch_workflow_machinery(monkeypatch, inserted, expanded_fids)
    backlog = json.dumps([
        {"id": "F-001", "name": "A"},
        {"id": "F-001", "name": "A-dup"},
        {"id": "F-002", "name": "B"},
    ])
    await _trigger_template_expansion(mission_id=99, backlog_text=backlog)
    # 2 features survive dedup (F-001 first wins, F-002 second).
    assert expanded_fids == ["F-001", "F-002"]


@pytest.mark.asyncio
async def test_missing_id_skipped(monkeypatch):
    inserted: list[dict] = []
    expanded_fids: list[str] = []
    _patch_workflow_machinery(monkeypatch, inserted, expanded_fids)
    backlog = json.dumps([
        {"name": "no id here"},
        {"id": "", "name": "blank id"},
        {"id": "   ", "name": "whitespace id"},
        {"id": "F-001", "name": "good"},
    ])
    await _trigger_template_expansion(mission_id=99, backlog_text=backlog)
    assert expanded_fids == ["F-001"]


@pytest.mark.asyncio
async def test_non_dict_entries_silently_dropped(monkeypatch):
    inserted: list[dict] = []
    expanded_fids: list[str] = []
    _patch_workflow_machinery(monkeypatch, inserted, expanded_fids)
    backlog = json.dumps([
        "not a dict",
        42,
        None,
        {"id": "F-001", "name": "good"},
    ])
    await _trigger_template_expansion(mission_id=99, backlog_text=backlog)
    assert expanded_fids == ["F-001"]


# ── Bound ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cap_applied_via_config(monkeypatch):
    inserted: list[dict] = []
    expanded_fids: list[str] = []
    _patch_workflow_machinery(monkeypatch, inserted, expanded_fids)
    monkeypatch.setattr(
        "src.app.config.MAX_FEATURES_PER_MISSION", 3
    )
    backlog = json.dumps([
        {"id": f"F-{i:03d}", "name": f"feat {i}"} for i in range(10)
    ])
    await _trigger_template_expansion(mission_id=99, backlog_text=backlog)
    assert len(expanded_fids) == 3
    assert expanded_fids == ["F-000", "F-001", "F-002"]


@pytest.mark.asyncio
async def test_under_cap_all_processed(monkeypatch):
    inserted: list[dict] = []
    expanded_fids: list[str] = []
    _patch_workflow_machinery(monkeypatch, inserted, expanded_fids)
    monkeypatch.setattr("src.app.config.MAX_FEATURES_PER_MISSION", 50)
    backlog = json.dumps([
        {"id": f"F-{i:03d}", "name": f"feat {i}"} for i in range(7)
    ])
    await _trigger_template_expansion(mission_id=99, backlog_text=backlog)
    assert len(expanded_fids) == 7


@pytest.mark.asyncio
async def test_mission57_replay_bounded_to_cap(monkeypatch):
    """Mission 57 emitted F-00 + F-001..F-055 (56 entries). With the
    default cap of 30, expansion is capped — and there's no path here to
    invent features beyond what the backlog contains."""
    inserted: list[dict] = []
    expanded_fids: list[str] = []
    _patch_workflow_machinery(monkeypatch, inserted, expanded_fids)
    monkeypatch.setattr("src.app.config.MAX_FEATURES_PER_MISSION", 30)
    backlog = [{"id": "F-00", "name": "infra"}]
    backlog.extend(
        {"id": f"F-{i:03d}", "name": f"feat {i}"} for i in range(1, 56)
    )
    await _trigger_template_expansion(
        mission_id=57, backlog_text=json.dumps(backlog)
    )
    assert len(expanded_fids) == 30
    # First 30 in order — no invented entries past the cap.
    expected = ["F-00"] + [f"F-{i:03d}" for i in range(1, 30)]
    assert expanded_fids == expected
