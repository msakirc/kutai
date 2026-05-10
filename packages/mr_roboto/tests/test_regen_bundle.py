"""Tests for `regen_bundle` mechanical action (Z1 Tier 4A — C19).

Bundle-level regen on a directional change. Founder says "darker / less warm
/ more clinical"; action identifies the affected slice (style guide + every
HTML + screen plans referencing color tokens) and re-emits each in
dependency order with a shared axis prompt fragment.
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest

import mr_roboto


def _seed_paraflow_bundle(mission_root):
    """Create the canonical paraflow-shape artifact set for tests."""
    mission_root.mkdir(parents=True, exist_ok=True)
    style_dir = mission_root / ".style"
    style_dir.mkdir()
    (style_dir / "design_tokens.json").write_text('{"colors": {"primary": "#0a5"}}', encoding="utf-8")
    (style_dir / "style_guide.md").write_text("# Style Guide\nForest green primary.\n", encoding="utf-8")

    plans_dir = mission_root / "screen_plans"
    plans_dir.mkdir()
    (plans_dir / "01_home.md").write_text("# Home\ncolor: primary\n", encoding="utf-8")
    (plans_dir / "02_detail.md").write_text("# Detail\ncolor: primary\n", encoding="utf-8")

    proto_dir = mission_root / ".prototype"
    proto_dir.mkdir()
    (proto_dir / "01_home.html").write_text("<html><body class='bg-emerald-700'>home</body></html>", encoding="utf-8")
    (proto_dir / "02_detail.html").write_text("<html><body class='bg-emerald-700'>detail</body></html>", encoding="utf-8")


@pytest.mark.asyncio
async def test_regen_bundle_tone_axis_touches_style_html_and_plans(tmp_path):
    mission_root = tmp_path / "mission_42"
    _seed_paraflow_bundle(mission_root)

    task = {
        "id": 200,
        "mission_id": 42,
        "payload": {
            "action": "regen_bundle",
            "axis": "tone",
            "direction": "darker",
            "workspace_path": str(tmp_path),
        },
    }

    # Mock the per-artifact emitter — return a marker we can check.
    async def fake_emit(*, artifact_path, change_description, axis_fragment, **_):
        return {"text": f"REGEN({axis_fragment})::{os.path.basename(artifact_path)}"}

    with patch(
        "mr_roboto.regen._invoke_emitter",
        new=AsyncMock(side_effect=fake_emit),
    ):
        action = await mr_roboto.run(task)

    assert action.status == "completed", action.error
    res = action.result
    affected = res["affected"]
    # Expect: style_guide + design_tokens + 2 HTMLs + 2 screen plans = 6.
    assert len(affected) == 6
    rel_paths = {a["artifact_path"] for a in affected}
    assert any("style_guide.md" in p for p in rel_paths)
    assert any("design_tokens.json" in p for p in rel_paths)
    assert any("01_home.html" in p for p in rel_paths)
    assert any("02_detail.html" in p for p in rel_paths)
    assert any("01_home.md" in p for p in rel_paths)
    assert any("02_detail.md" in p for p in rel_paths)


@pytest.mark.asyncio
async def test_regen_bundle_dependency_order_style_first(tmp_path):
    """Style tokens regen before HTML (HTML consumes tokens)."""
    mission_root = tmp_path / "mission_9"
    _seed_paraflow_bundle(mission_root)

    order = []

    async def fake_emit(*, artifact_path, **_):
        order.append(artifact_path)
        return {"text": "x"}

    task = {
        "id": 1,
        "mission_id": 9,
        "payload": {
            "action": "regen_bundle",
            "axis": "tone",
            "direction": "darker",
            "workspace_path": str(tmp_path),
        },
    }

    with patch("mr_roboto.regen._invoke_emitter",
               new=AsyncMock(side_effect=fake_emit)):
        action = await mr_roboto.run(task)

    assert action.status == "completed"
    # Style/tokens (anything in .style/) must come before any HTML.
    style_idx = [i for i, p in enumerate(order) if "/.style/" in p.replace("\\", "/")]
    html_idx = [i for i, p in enumerate(order) if p.endswith(".html")]
    assert style_idx and html_idx
    assert max(style_idx) < min(html_idx)


@pytest.mark.asyncio
async def test_regen_bundle_unknown_axis_fails(tmp_path):
    mission_root = tmp_path / "mission_1"
    _seed_paraflow_bundle(mission_root)
    task = {
        "id": 1,
        "mission_id": 1,
        "payload": {
            "action": "regen_bundle",
            "axis": "tempo",  # not in axis registry
            "direction": "faster",
            "workspace_path": str(tmp_path),
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "failed"
    assert "axis" in (action.error or "").lower()


@pytest.mark.asyncio
async def test_regen_bundle_records_log_with_scope_bundle(tmp_path):
    mission_root = tmp_path / "mission_4"
    _seed_paraflow_bundle(mission_root)

    captured = []

    async def fake_log(**kwargs):
        captured.append(kwargs)
        return len(captured)

    async def fake_emit(*, artifact_path, **_):
        return {"text": "x"}

    task = {
        "id": 10,
        "mission_id": 4,
        "payload": {
            "action": "regen_bundle",
            "axis": "tone",
            "direction": "less warm",
            "workspace_path": str(tmp_path),
        },
    }

    with patch("mr_roboto.regen._invoke_emitter",
               new=AsyncMock(side_effect=fake_emit)), \
         patch("mr_roboto.regen._record_regen_log", side_effect=fake_log):
        action = await mr_roboto.run(task)

    assert action.status == "completed"
    # Every affected artifact gets a log row, all tagged scope=bundle.
    assert len(captured) >= 1
    assert all(c["scope"] == "bundle" for c in captured)
    assert all(c["change_description"] == "less warm" for c in captured)


@pytest.mark.asyncio
async def test_regen_bundle_no_artifacts_returns_empty(tmp_path):
    """Mission with no matching artifacts returns completed with affected=[]."""
    (tmp_path / "mission_99").mkdir()
    task = {
        "id": 1,
        "mission_id": 99,
        "payload": {
            "action": "regen_bundle",
            "axis": "tone",
            "direction": "darker",
            "workspace_path": str(tmp_path),
        },
    }
    action = await mr_roboto.run(task)
    assert action.status == "completed"
    assert action.result["affected"] == []
