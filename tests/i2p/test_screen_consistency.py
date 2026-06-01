"""Z1 Tier 3 (C18+B8) — verify_screen_consistency contract tests."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from mr_roboto.verify_screen_consistency import verify_screen_consistency
from mr_roboto import run as mr_roboto_run


_HERE = Path(__file__).resolve().parent
_FIX_DIR = _HERE / "reviewer_regression" / "fixtures" / "v1" / "5_10"


def _load(stem: str) -> dict:
    return json.loads((_FIX_DIR / f"{stem}.json").read_text(encoding="utf-8"))


def test_good_consistency_passes():
    fx = _load("good_screen_consistency")
    res = verify_screen_consistency(
        screen_plans=fx["screen_plans"],
        shared_shell_components=fx["shared_shell_components"],
    )
    assert res["ok"] is True, res
    assert res["mismatches"] == []
    assert sorted(res["canonical_inherits_shell"]) == ["header", "tab_bar"]


def test_bad_consistency_rejects_unannotated_drift():
    fx = _load("bad_screen_consistency")
    res = verify_screen_consistency(
        screen_plans=fx["screen_plans"],
        shared_shell_components=fx["shared_shell_components"],
    )
    assert res["ok"] is False
    assert res["mismatches"]
    assert any(m["screen_id"] == "profile" for m in res["mismatches"])


def test_override_comment_tolerated():
    plans = {
        "home": (
            "---\n_schema_version: \"1\"\nscreen_id: home\n"
            "route: \"/\"\nsurface: mobile\ninherits_shell: [\"header\", \"tab_bar\"]\n---\n"
            "# Home\n"
        ),
        "wizard": (
            "---\n_schema_version: \"1\"\nscreen_id: wizard\n"
            "route: \"/onboard\"\nsurface: mobile\ninherits_shell: []\n---\n"
            "# Onboarding wizard\n\n"
            "<!-- inherits_shell_override: full-screen onboarding has no chrome -->\n"
        ),
    }
    res = verify_screen_consistency(screen_plans=plans)
    assert res["ok"] is True
    assert "wizard" in res["override_comments"]


def test_unknown_shell_component_rejected():
    plans = {
        "home": (
            "---\nscreen_id: home\nroute: \"/\"\nsurface: mobile\n"
            "inherits_shell: [\"header\", \"footer_unknown\"]\n---\n# Home\n"
        ),
    }
    res = verify_screen_consistency(
        screen_plans=plans,
        shared_shell_components=["header", "tab_bar"],
    )
    assert res["ok"] is False
    assert res["out_of_set"]


def test_dispatcher_completes_on_good():
    fx = _load("good_screen_consistency")
    task = {
        "id": 0, "mission_id": 0,
        "payload": {
            "action": "verify_screen_consistency",
            "screen_plans": fx["screen_plans"],
            "shared_shell_components": fx["shared_shell_components"],
        },
    }
    result = asyncio.run(mr_roboto_run(task))
    assert result.status == "completed", result.error
    assert result.result["ok"] is True


def test_dispatcher_fails_on_bad():
    fx = _load("bad_screen_consistency")
    task = {
        "id": 0, "mission_id": 0,
        "payload": {
            "action": "verify_screen_consistency",
            "screen_plans": fx["screen_plans"],
            "shared_shell_components": fx["shared_shell_components"],
        },
    }
    result = asyncio.run(mr_roboto_run(task))
    assert result.status == "failed"


def test_path_mode(tmp_path: Path):
    fx = _load("good_screen_consistency")
    paths = []
    for sid, md in fx["screen_plans"].items():
        p = tmp_path / f"{sid}.md"
        p.write_text(md, encoding="utf-8")
        paths.append(str(p))
    res = verify_screen_consistency(
        screen_plan_paths=paths,
        shared_shell_components=fx["shared_shell_components"],
    )
    assert res["ok"] is True


def test_no_input_rejected():
    res = verify_screen_consistency()
    assert res["ok"] is False
    assert "no input" in str(res["problems"]).lower() or res.get("error")


def test_directory_path_globs_md_files(tmp_path: Path):
    """A directory entry in screen_plan_paths must glob its *.md files.

    The per-screen plans are written under a runtime dir (mission_<id>/.screens/)
    whose individual filenames are not known at workflow-author time, so the
    `checks` entry points at the DIRECTORY. The verb must expand it to the
    contained .md plans rather than failing to open the dir."""
    fx = _load("good_screen_consistency")
    d = tmp_path / ".screens"
    d.mkdir()
    for sid, md in fx["screen_plans"].items():
        (d / f"{sid}.md").write_text(md, encoding="utf-8")
    res = verify_screen_consistency(
        screen_plan_paths=[str(d)],
        shared_shell_components=fx["shared_shell_components"],
    )
    assert res["ok"] is True, res
    assert res["mismatches"] == []


def test_directory_with_no_md_files_rejected(tmp_path: Path):
    d = tmp_path / ".screens"
    d.mkdir()
    res = verify_screen_consistency(screen_plan_paths=[str(d)])
    assert res["ok"] is False
