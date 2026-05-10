"""Z1 Tier 5B (B5) — spec_consistency_check contract tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mr_roboto.spec_consistency_check import spec_consistency_check


def _seed_minimal_spec(workspace: Path, mission_id: int = 42) -> Path:
    """Create a minimal phase-≤6 locked spec on disk. Returns mission_dir."""
    md = workspace / f"mission_{mission_id}"
    (md / ".charter").mkdir(parents=True, exist_ok=True)
    (md / ".charter" / "product_charter.md").write_text(
        "## Product Positioning\nA tool.\n\n## Brand Keywords\n- **calm** — quiet UI\n",
        encoding="utf-8",
    )
    (md / "non_goals.md").write_text(
        "# Non-goals\n\n- We will not build a social network feature here.\n"
        "- We will not support cryptocurrency payment rails.\n",
        encoding="utf-8",
    )
    (md / ".style").mkdir(parents=True, exist_ok=True)
    (md / ".style" / "design_tokens.json").write_text(
        json.dumps({
            "_schema_version": "1",
            "color": {
                "primary": "#1A2B3C",
                "accent": "#FF8800",
            },
            "font": {
                "family": "Inter",
            },
        }),
        encoding="utf-8",
    )
    (md / "surfaces.md").write_text(
        "# Surfaces\n\n- mobile (primary)\n- web\n",
        encoding="utf-8",
    )
    (md / ".adrs").mkdir(parents=True, exist_ok=True)
    (md / ".adrs" / "ADR-0001.json").write_text(
        json.dumps({
            "id": "ADR-0001",
            "title": "Pick a backend stack",
            "options": [
                {"id": "A", "name": "FastAPI", "status": "chosen"},
                {"id": "B", "name": "Rails", "status": "rejected"},
                {"id": "C", "name": "Django", "status": "rejected"},
            ],
        }),
        encoding="utf-8",
    )
    return md


def test_no_drift_case(tmp_path):
    md = _seed_minimal_spec(tmp_path)
    # Phase-7 file using only declared tokens / surface
    (md / ".phase_7").mkdir(parents=True, exist_ok=True)
    (md / ".phase_7" / "tokens.css").write_text(
        ":root { --primary: #1A2B3C; --accent: #FF8800; }\n",
        encoding="utf-8",
    )
    res = spec_consistency_check(
        mission_id=42,
        current_phase="phase_7",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is True, res
    assert res["drift_items"] == []
    # Report path is still emitted.
    assert Path(res["report_path"]).exists()


def test_drift_rejected_tech_from_adr(tmp_path):
    md = _seed_minimal_spec(tmp_path)
    # Phase-8 backlog mentions Rails (a rejected ADR option)
    (md / ".phase_8").mkdir(parents=True, exist_ok=True)
    (md / ".phase_8" / "task_001.md").write_text(
        "Implement the auth module using Rails ActiveRecord patterns.\n",
        encoding="utf-8",
    )
    res = spec_consistency_check(
        mission_id=42,
        current_phase="phase_8",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is False
    assert any("R1 stack_drift" in d["conflict"] for d in res["drift_items"])
    assert any("rails" in d["conflict"].lower() for d in res["drift_items"])


def test_drift_token_color_not_in_design_tokens(tmp_path):
    md = _seed_minimal_spec(tmp_path)
    (md / ".phase_7").mkdir(parents=True, exist_ok=True)
    (md / ".phase_7" / "tokens.css").write_text(
        ":root { --primary: #DEADBE; --accent: #FF8800; }\n",
        encoding="utf-8",
    )
    res = spec_consistency_check(
        mission_id=42,
        current_phase="phase_7",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is False
    assert any("R2 token_drift" in d["conflict"] for d in res["drift_items"])


def test_drift_surface_not_declared(tmp_path):
    md = _seed_minimal_spec(tmp_path)
    (md / ".phase_8").mkdir(parents=True, exist_ok=True)
    (md / ".phase_8" / "desktop_router.md").write_text(
        "Add a desktop wrapper for the home screen.\n",
        encoding="utf-8",
    )
    res = spec_consistency_check(
        mission_id=42,
        current_phase="phase_8",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is False
    assert any("R3 surface_drift" in d["conflict"] for d in res["drift_items"])


def test_missing_spec_artifact_fails_soft(tmp_path):
    """When charter is missing, the check warns but does NOT fail."""
    md = tmp_path / "mission_99"
    md.mkdir()
    # Create a phase-7 file that would otherwise drift, but no spec is present.
    (md / ".phase_7").mkdir(parents=True, exist_ok=True)
    (md / ".phase_7" / "anything.md").write_text(
        "Use Rails for everything.\n", encoding="utf-8"
    )
    res = spec_consistency_check(
        mission_id=99,
        current_phase="phase_7",
        workspace_path=str(tmp_path),
    )
    # No charter, no ADRs => no rejected_tech => no R1 hit.
    # No design_tokens => no R2. No surfaces.md => no R3.
    # No non_goals => no R4. So fail-soft = ok=True.
    assert res["ok"] is True, res
    assert "spec_artifact_missing:charter" in res["warnings"]


def test_missing_mission_dir_returns_ok(tmp_path):
    res = spec_consistency_check(
        mission_id=12345,
        current_phase="phase_7",
        workspace_path=str(tmp_path),
    )
    assert res["ok"] is True
    assert any("mission_dir_missing" in w for w in res["warnings"])


@pytest.mark.asyncio
async def test_dispatch_via_mr_roboto_run_no_drift(tmp_path):
    import mr_roboto
    md = _seed_minimal_spec(tmp_path)
    (md / ".phase_7").mkdir(parents=True, exist_ok=True)
    (md / ".phase_7" / "tokens.css").write_text(
        ":root { --primary: #1A2B3C; }\n", encoding="utf-8"
    )
    task = {
        "id": 1,
        "mission_id": 42,
        "phase": "phase_7",
        "payload": {
            "action": "spec_consistency_check",
            "current_phase": "phase_7",
            "workspace_path": str(tmp_path),
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed", action


@pytest.mark.asyncio
async def test_dispatch_via_mr_roboto_run_drift_returns_needs_review(tmp_path):
    import mr_roboto
    md = _seed_minimal_spec(tmp_path)
    (md / ".phase_8").mkdir(parents=True, exist_ok=True)
    (md / ".phase_8" / "task.md").write_text(
        "Use Rails for the new module.\n", encoding="utf-8"
    )
    task = {
        "id": 2,
        "mission_id": 42,
        "phase": "phase_8",
        "payload": {
            "action": "spec_consistency_check",
            "current_phase": "phase_8",
            "workspace_path": str(tmp_path),
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "needs_review", action
    assert action.result.get("drift_items")


def test_report_envelope_shape(tmp_path):
    md = _seed_minimal_spec(tmp_path)
    (md / ".phase_8").mkdir(parents=True, exist_ok=True)
    (md / ".phase_8" / "task.md").write_text(
        "Use Rails for the new module.\n", encoding="utf-8"
    )
    res = spec_consistency_check(
        mission_id=42,
        current_phase="phase_8",
        workspace_path=str(tmp_path),
    )
    env = res["envelope"]
    assert env["_schema_version"] == "1"
    assert isinstance(env["drift_items"], list)
    for it in env["drift_items"]:
        assert set(it.keys()) >= {
            "phase", "artifact", "conflict", "suggested_resolution"
        }
