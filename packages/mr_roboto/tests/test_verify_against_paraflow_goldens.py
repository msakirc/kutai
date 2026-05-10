"""Tests for mr_roboto.verify_against_paraflow_goldens (C21)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from c21_paraflow_diff import load_golden
from mr_roboto.verify_against_paraflow_goldens import (
    verify_against_paraflow_goldens,
)
import mr_roboto


def _write(p: Path, body: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


def _make_full_mission(root: Path) -> None:
    g = load_golden("truthrate")
    for slot, gp in g.files.items():
        _write(root / f"{slot}.md", gp.read_text("utf-8"))
    for stem in g.screen_plan_keys:
        src = g.root / "screen_plans" / f"{stem}_screen_plan.md"
        if src.is_file():
            _write(
                root / "screen_plans" / f"{stem}_screen_plan.md",
                src.read_text("utf-8"),
            )
    for stem in g.screen_keys:
        src = g.root / "screens" / f"{stem}.html"
        if src.is_file():
            _write(root / "screens" / f"{stem}.html", src.read_text("utf-8"))
    tokens = {
        "_schema_version": "1",
        "colors": {"a": 1},
        "typography": {"a": 1},
        "border_radius": {"a": 1},
        "spacing": {"a": 1},
        "layout": {"a": 1},
    }
    _write(root / ".style" / "design_tokens.json", json.dumps(tokens))


@pytest.mark.asyncio
async def test_verify_par_when_workspace_full(tmp_path, monkeypatch):
    mission = tmp_path / "mission_42"
    _make_full_mission(mission)

    # Skip DB persist (test env).
    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(
        "mr_roboto.verify_against_paraflow_goldens._persist", _noop
    )

    res = await verify_against_paraflow_goldens(
        mission_id=42, workspace_path=str(mission)
    )
    assert res["ok"] is True
    assert res["verdict"] == "paraflow_par"
    assert res["mission_id"] == 42
    assert res["archetype"] == "truthrate"


@pytest.mark.asyncio
async def test_verify_gap_when_workspace_empty(tmp_path, monkeypatch):
    mission = tmp_path / "mission_empty"
    mission.mkdir()

    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(
        "mr_roboto.verify_against_paraflow_goldens._persist", _noop
    )

    res = await verify_against_paraflow_goldens(
        mission_id=99, workspace_path=str(mission)
    )
    assert res["verdict"] == "paraflow_gap"
    assert res["ok"] is False
    assert "charter" in res["gaps"]


@pytest.mark.asyncio
async def test_verify_unknown_archetype_returns_error(tmp_path, monkeypatch):
    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(
        "mr_roboto.verify_against_paraflow_goldens._persist", _noop
    )
    res = await verify_against_paraflow_goldens(
        mission_id=1,
        archetype="not_a_real_archetype",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is False
    assert "error" in res
    assert res["verdict"] == "paraflow_gap"


@pytest.mark.asyncio
async def test_mr_roboto_run_dispatches_action(tmp_path, monkeypatch):
    mission = tmp_path / "mission_7"
    _make_full_mission(mission)

    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(
        "mr_roboto.verify_against_paraflow_goldens._persist", _noop
    )

    task = {
        "id": 1,
        "mission_id": 7,
        "payload": {
            "action": "verify_against_paraflow_goldens",
            "archetype": "truthrate",
            "workspace_path": str(mission),
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.result["verdict"] == "paraflow_par"
