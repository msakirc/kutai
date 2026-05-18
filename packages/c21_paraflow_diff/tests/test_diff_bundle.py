"""Unit tests for c21_paraflow_diff.diff_bundle."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from c21_paraflow_diff import (
    GoldenNotFoundError,
    diff_bundle,
    load_golden,
    DEFAULT_GOLDENS_ROOT,
)


def _write(p: Path, body: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


def _make_full_mission(root: Path) -> None:
    """Mirror the truthrate golden's shape under ``root``."""
    g = load_golden("truthrate")
    # Copy each golden artifact verbatim so coherence is 1.0.
    for slot, gp in g.files.items():
        _write(root / f"{slot}.md", gp.read_text(encoding="utf-8"))
    # Screen plans + HTMLs by name.
    sp_dir = root / "screen_plans"
    for stem in g.screen_plan_keys:
        src = (g.root / "screen_plans" / f"{stem}_screen_plan.md")
        if src.is_file():
            _write(sp_dir / f"{stem}_screen_plan.md", src.read_text("utf-8"))
    sc_dir = root / "screens"
    for stem in g.screen_keys:
        src = g.root / "screens" / f"{stem}.html"
        if src.is_file():
            _write(sc_dir / f"{stem}.html", src.read_text("utf-8"))
    # Design tokens covering every paraflow axis.
    tokens = {
        "_schema_version": "1",
        "colors": {"primary": "#000"},
        "typography": {"family": "Inter"},
        "border_radius": {"md": "8px"},
        "spacing": {"md": "16px"},
        "layout": {"max_width": "390px"},
    }
    _write(root / ".style" / "design_tokens.json", json.dumps(tokens))


# ---------------------------------------------------------------------------
# load_golden
# ---------------------------------------------------------------------------


def test_load_golden_truthrate_has_all_artifacts():
    g = load_golden("truthrate")
    assert g.archetype == "truthrate"
    for slot in (
        "charter",
        "personas",
        "prd",
        "user_flow",
        "style_guide_light",
        "style_guide_dark",
    ):
        assert slot in g.files, f"{slot} missing in golden"
    assert len(g.screen_plan_keys) >= 5
    assert len(g.screen_keys) >= 5
    assert "product positioning" in g.charter_sections
    assert any(s.startswith("1") or "background" in s for s in g.prd_sections)


def test_load_golden_unknown_archetype_raises():
    with pytest.raises(GoldenNotFoundError):
        load_golden("does_not_exist")


def test_default_goldens_root_resolves_under_repo():
    assert DEFAULT_GOLDENS_ROOT.is_dir()
    assert (DEFAULT_GOLDENS_ROOT / "truthrate").is_dir()


# ---------------------------------------------------------------------------
# diff_bundle — happy path
# ---------------------------------------------------------------------------


def test_diff_bundle_par_when_mission_mirrors_golden(tmp_path):
    mission = tmp_path / "mission_99"
    _make_full_mission(mission)

    res = diff_bundle(str(mission), "truthrate")
    assert res["archetype"] == "truthrate"
    assert res["verdict"] == "paraflow_par", (
        f"expected par, got {res['verdict']} score={res['score']} "
        f"gaps={res['gaps']}"
    )
    assert res["score"] >= 0.85
    assert res["coverage"]["charter"] is True
    assert res["coverage"]["screen_plans"]["present"] is True
    assert res["coverage"]["screens"]["present"] is True
    assert res["design_fitness"]["score"] == 1.0


def test_diff_bundle_returns_full_shape(tmp_path):
    mission = tmp_path / "mission_1"
    _make_full_mission(mission)
    res = diff_bundle(str(mission), "truthrate")
    for k in (
        "archetype",
        "mission_workspace_path",
        "goldens_root",
        "coverage",
        "coherence",
        "design_fitness",
        "gaps",
        "score",
        "verdict",
    ):
        assert k in res, f"missing key {k}"


# ---------------------------------------------------------------------------
# diff_bundle — gap detection
# ---------------------------------------------------------------------------


def test_diff_bundle_gap_when_workspace_empty(tmp_path):
    mission = tmp_path / "mission_empty"
    mission.mkdir()
    res = diff_bundle(str(mission), "truthrate")
    assert res["verdict"] == "paraflow_gap"
    assert "charter" in res["gaps"]
    assert "personas" in res["gaps"]
    assert "prd" in res["gaps"]
    assert "user_flow" in res["gaps"]
    assert "screen_plans" in res["gaps"]
    assert "screens" in res["gaps"]
    assert res["score"] < 0.5


def test_diff_bundle_gap_when_workspace_does_not_exist(tmp_path):
    mission = tmp_path / "never_created"
    res = diff_bundle(str(mission), "truthrate")
    assert res["verdict"] == "paraflow_gap"


# ---------------------------------------------------------------------------
# diff_bundle — partial verdict
# ---------------------------------------------------------------------------


def test_diff_bundle_partial_when_half_present(tmp_path):
    mission = tmp_path / "mission_half"
    g = load_golden("truthrate")
    # Copy 5 of 6 .md artifacts + screen_plans dir; skip screens dir,
    # personas, design tokens. Should land in partial band.
    for slot in (
        "charter",
        "prd",
        "user_flow",
        "style_guide_light",
        "style_guide_dark",
    ):
        _write(mission / f"{slot}.md", g.files[slot].read_text("utf-8"))
    sp_dir = mission / "screen_plans"
    for stem in g.screen_plan_keys:
        src = g.root / "screen_plans" / f"{stem}_screen_plan.md"
        if src.is_file():
            _write(sp_dir / f"{stem}_screen_plan.md", src.read_text("utf-8"))

    res = diff_bundle(str(mission), "truthrate")
    assert res["verdict"] == "paraflow_partial", (
        f"expected partial, got {res['verdict']} score={res['score']} "
        f"gaps={res['gaps']}"
    )
    assert "personas" in res["gaps"]
    assert "screens" in res["gaps"]


# ---------------------------------------------------------------------------
# diff_bundle — coherence + design fitness mechanics
# ---------------------------------------------------------------------------


def test_diff_bundle_coherence_flags_missing_sections(tmp_path):
    mission = tmp_path / "mission_thin"
    # Charter present but with only 1 of 5 paraflow sections.
    _write(
        mission / "charter.md",
        "# Charter\n\n## 1) Product Positioning\n\nstub.\n",
    )
    res = diff_bundle(str(mission), "truthrate")
    coh = res["coherence"]["charter"]
    assert "product positioning" in coh["matched"]
    assert "brand keywords" in coh["missing"]
    assert coh["score"] < 1.0
    assert any(g.startswith("coherence:charter") for g in res["gaps"])


def test_diff_bundle_design_fitness_flags_missing_axes(tmp_path):
    mission = tmp_path / "mission_partial_tokens"
    _make_full_mission(mission)
    # Drop spacing axis from tokens.
    tokens = json.loads(
        (mission / ".style" / "design_tokens.json").read_text("utf-8")
    )
    tokens.pop("spacing", None)
    (mission / ".style" / "design_tokens.json").write_text(
        json.dumps(tokens), encoding="utf-8"
    )
    res = diff_bundle(str(mission), "truthrate")
    assert "spacing" in res["design_fitness"]["missing_axes"]
    assert any(
        g.startswith("design_tokens_axes:") for g in res["gaps"]
    )


def test_diff_bundle_design_fitness_no_tokens_file(tmp_path):
    mission = tmp_path / "mission_no_tokens"
    _make_full_mission(mission)
    (mission / ".style" / "design_tokens.json").unlink()
    res = diff_bundle(str(mission), "truthrate")
    assert res["design_fitness"]["tokens_present"] is False
    assert res["design_fitness"]["score"] == 0.0


# ---------------------------------------------------------------------------
# diff_bundle — unknown archetype
# ---------------------------------------------------------------------------


def test_diff_bundle_unknown_archetype_raises(tmp_path):
    with pytest.raises(GoldenNotFoundError):
        diff_bundle(str(tmp_path), "no_such_archetype")
