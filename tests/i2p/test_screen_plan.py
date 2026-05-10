"""Z1 Tier 3 (C3+A10+C14) — verify_screen_plan_shape contract tests."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from mr_roboto.verify_screen_plan_shape import verify_screen_plan_shape
from mr_roboto import run as mr_roboto_run


_HERE = Path(__file__).resolve().parent
_FIX_DIR = _HERE / "reviewer_regression" / "fixtures" / "v1" / "5_10"


def _load(stem: str) -> dict:
    return json.loads((_FIX_DIR / f"{stem}.json").read_text(encoding="utf-8"))


def test_good_screen_plan_passes():
    fx = _load("good_screen_plan")
    res = verify_screen_plan_shape(plan_text=fx["screen_plan_md"])
    assert res["ok"] is True, res
    assert res["frontmatter_present"] is True
    assert set(res["frontmatter_keys"]).issuperset(
        {"_schema_version", "mission_id", "screen_id", "route", "surface", "inherits_shell"}
    )
    assert set(res["states_subsections"]) >= {"Default", "Empty", "Loading", "Error"}
    assert res["missing_state_subsections"] == []


def test_bad_screen_plan_rejected_for_missing_states_and_route():
    fx = _load("bad_screen_plan")
    res = verify_screen_plan_shape(plan_text=fx["screen_plan_md"])
    assert res["ok"] is False
    # Loading + Error are missing.
    assert "Loading" in res["missing_state_subsections"]
    assert "Error" in res["missing_state_subsections"]
    # route is missing from frontmatter.
    assert "route" in res["missing_frontmatter_keys"]


def test_missing_frontmatter_rejected():
    res = verify_screen_plan_shape(
        plan_text="# Home\n\nNo frontmatter at all.\n\n## Search\n- bar\n"
    )
    assert res["ok"] is False
    assert res["frontmatter_present"] is False


def test_missing_description_rejected():
    md = (
        "---\n_schema_version: \"1\"\nmission_id: 1\nscreen_id: home\n"
        "route: \"/\"\nsurface: mobile\ninherits_shell: []\n---\n\n"
        "# Home\n\n"
        "## Search\n- bar\n\n"
        "## States\n\n### Default\nx\n\n### Empty\nx\n\n"
        "### Loading\nx\n\n### Error\nx\n"
    )
    res = verify_screen_plan_shape(plan_text=md)
    assert res["ok"] is False
    assert any("description" in p for p in res["problems"])


def test_no_section_beyond_states_rejected():
    md = (
        "---\n_schema_version: \"1\"\nmission_id: 1\nscreen_id: home\n"
        "route: \"/\"\nsurface: mobile\ninherits_shell: []\n---\n\n"
        "# Home\n\nA description paragraph.\n\n"
        "## States\n\n### Default\nx\n\n### Empty\nx\n\n"
        "### Loading\nx\n\n### Error\nx\n"
    )
    res = verify_screen_plan_shape(plan_text=md)
    assert res["ok"] is False
    assert any("content sections" in p for p in res["problems"])


def test_placeholder_text_rejected():
    fx = _load("good_screen_plan")
    bad = fx["screen_plan_md"].replace("Allow users", "TODO Allow users")
    res = verify_screen_plan_shape(plan_text=bad)
    assert res["ok"] is False
    assert res["placeholders"]


def test_dispatcher_completes_on_good():
    fx = _load("good_screen_plan")
    task = {
        "id": 0, "mission_id": 0,
        "payload": {
            "action": "verify_screen_plan_shape",
            "plan_text": fx["screen_plan_md"],
        },
    }
    result = asyncio.run(mr_roboto_run(task))
    assert result.status == "completed", result.error
    assert result.result["ok"] is True


def test_dispatcher_fails_on_bad():
    fx = _load("bad_screen_plan")
    task = {
        "id": 0, "mission_id": 0,
        "payload": {
            "action": "verify_screen_plan_shape",
            "plan_text": fx["screen_plan_md"],
        },
    }
    result = asyncio.run(mr_roboto_run(task))
    assert result.status == "failed"


def test_path_mode(tmp_path: Path):
    fx_good = _load("good_screen_plan")
    fx_bad = _load("bad_screen_plan")
    p_good = tmp_path / "home.md"
    p_bad = tmp_path / "search.md"
    p_good.write_text(fx_good["screen_plan_md"], encoding="utf-8")
    p_bad.write_text(fx_bad["screen_plan_md"], encoding="utf-8")
    res = verify_screen_plan_shape(plan_paths=[str(p_good), str(p_bad)])
    assert res["ok"] is False
    assert len(res["per_file"]) == 2
    assert res["per_file"][0]["ok"] is True
    assert res["per_file"][1]["ok"] is False
