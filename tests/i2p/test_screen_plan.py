"""Z1 Tier 3 (C3+A10+C14) — verify_screen_plan_shape contract tests."""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from mr_roboto.verify_screen_plan_shape import (
    verify_screen_plan_shape,
    normalize_screen_plan,
)
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


# ── directory produces: 5.20a/5.20b declare `.screens/` (the individual
# filenames are unknown at workflow-author time), so the checks payload points
# at the DIRECTORY. The verifier must self-expand it to its `*.md` files —
# mirroring verify_screen_consistency — else it `open()`s a directory and the
# gate reports the uninformative `problems=[]` that DLQ'd m90 task 567454.


def test_dir_path_mode_expands_to_md_files(tmp_path: Path):
    fx_good = _load("good_screen_plan")
    screens = tmp_path / ".screens"
    screens.mkdir()
    (screens / "home.md").write_text(fx_good["screen_plan_md"], encoding="utf-8")
    (screens / "settings.md").write_text(fx_good["screen_plan_md"], encoding="utf-8")
    res = verify_screen_plan_shape(plan_paths=[str(screens) + "/"])
    assert res["ok"] is True, res
    assert len(res["per_file"]) == 2
    assert all(pf["ok"] for pf in res["per_file"])


def test_dir_path_mode_empty_dir_fails(tmp_path: Path):
    """An empty `.screens/` is not a vacuous pass — nothing was authored."""
    screens = tmp_path / ".screens"
    screens.mkdir()
    res = verify_screen_plan_shape(plan_paths=[str(screens) + "/"])
    assert res["ok"] is False


# ── deterministic frontmatter repair (mirrors normalize_user_flow) ──────────
# The analyst authors the screen's SEMANTIC content (sections, states); the
# MECHANICAL frontmatter keys `mission_id` (a mission constant, derivable from
# the path) and `inherits_shell` (the chunk contract "same across the chunk",
# derivable from the modal of sibling plans) are shell the engine guarantees
# regardless of model compliance. m90 5.20b (task 567455): the chunk-b analyst
# dropped both keys on 5 screens → degenerate-repeat DLQ. normalize_screen_plan
# injects only the MISSING mechanical keys; it never fabricates body content.


def _plan_missing(keys_to_drop):
    base = (
        "---\n_schema_version: \"1\"\nmission_id: 90\nscreen_id: errands\n"
        "route: /errands\nsurface: web\ninherits_shell: [\"Header\"]\n---\n\n"
        "# Errands\n\nA description.\n\n## Content\n- x\n\n"
        "## States\n\n### Default\nx\n\n### Empty\nx\n\n### Loading\nx\n\n### Error\nx\n"
    )
    for k in keys_to_drop:
        base = re.sub(rf"^{k}:.*\n", "", base, flags=re.M)
    return base


def test_normalize_injects_missing_mission_id():
    md = _plan_missing(["mission_id"])
    out, changed = normalize_screen_plan(md, mission_id="90")
    assert changed is True
    fm, _ = __import__("mr_roboto.verify_screen_plan_shape",
                       fromlist=["_parse_frontmatter"])._parse_frontmatter(out)
    assert fm.get("mission_id") == "90"


def test_normalize_does_not_inject_inherits_shell():
    """inherits_shell is a SEMANTIC choice grounded in shared_shell.md — never a
    guess. A plan that drops it is a real failure the shape gate rejects (→
    re-pend), NOT something the repair fabricates a value for."""
    md = _plan_missing(["inherits_shell"])
    out, changed = normalize_screen_plan(md, mission_id="90")
    assert changed is False
    assert "inherits_shell:" not in out.split("---", 2)[1]


def test_normalize_noop_when_mission_id_present():
    md = _plan_missing([])
    out, changed = normalize_screen_plan(md, mission_id="90")
    assert changed is False
    assert out == md


def test_normalize_no_frontmatter_is_noop():
    md = "# Errands\n\nNo frontmatter here.\n"
    out, changed = normalize_screen_plan(md, mission_id="90")
    assert changed is False and out == md


def test_dir_repair_stamps_mission_id_only(tmp_path: Path):
    """The m90 567455 scenario, honest version: the mechanical stamp fills the
    KNOWN mission_id (from the path) and persists it; inherits_shell is NOT
    fabricated, so a plan that dropped it still fails the shape gate (the model
    must supply it, grounded in shared_shell.md)."""
    screens = tmp_path / "mission_90" / ".screens"
    (screens / "errands").mkdir(parents=True)
    bad = _plan_missing(["mission_id"])                 # only mission_id dropped
    bad_path = screens / "errands" / "screen_plan.md"
    bad_path.write_text(bad, encoding="utf-8")
    res = verify_screen_plan_shape(plan_paths=[str(screens) + "/"])
    assert res["ok"] is True, res
    assert "mission_id: 90" in bad_path.read_text(encoding="utf-8")


def test_dir_repair_does_not_fabricate_inherits_shell(tmp_path: Path):
    screens = tmp_path / "mission_90" / ".screens"
    (screens / "errands").mkdir(parents=True)
    bad = _plan_missing(["mission_id", "inherits_shell"])
    bad_path = screens / "errands" / "screen_plan.md"
    bad_path.write_text(bad, encoding="utf-8")
    res = verify_screen_plan_shape(plan_paths=[str(screens) + "/"])
    assert res["ok"] is False   # inherits_shell still missing → honest fail
    repaired = bad_path.read_text(encoding="utf-8")
    assert "mission_id: 90" in repaired                 # known constant stamped
    assert "inherits_shell:" not in repaired.split("---", 2)[1]  # not fabricated


def test_dir_path_mode_finds_nested_per_screen_files(tmp_path: Path):
    """PRODUCTION layout: the step writes `.screens/<slug>/screen_plan.md`
    (one subdir per screen), so directory expansion must recurse."""
    fx_good = _load("good_screen_plan")
    screens = tmp_path / ".screens"
    (screens / "home").mkdir(parents=True)
    (screens / "settings").mkdir(parents=True)
    (screens / "home" / "screen_plan.md").write_text(
        fx_good["screen_plan_md"], encoding="utf-8")
    (screens / "settings" / "screen_plan.md").write_text(
        fx_good["screen_plan_md"], encoding="utf-8")
    res = verify_screen_plan_shape(plan_paths=[str(screens) + "/"])
    assert res["ok"] is True, res
    assert len(res["per_file"]) == 2


def test_dir_path_mode_flags_a_bad_file_in_the_directory(tmp_path: Path):
    fx_good = _load("good_screen_plan")
    fx_bad = _load("bad_screen_plan")
    screens = tmp_path / ".screens"
    screens.mkdir()
    (screens / "home.md").write_text(fx_good["screen_plan_md"], encoding="utf-8")
    (screens / "search.md").write_text(fx_bad["screen_plan_md"], encoding="utf-8")
    res = verify_screen_plan_shape(plan_paths=[str(screens) + "/"])
    assert res["ok"] is False
    assert len(res["per_file"]) == 2
