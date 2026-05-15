"""Tests for the visual_review mechanical verb (Z4 T2B).

All vision / LLM calls are mocked — no real network requests.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_png(path: str) -> None:
    """Write a minimal 1-pixel PNG file so os.path.exists() passes."""
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01"
        b"\x00\x00\x00\x01"
        b"\x08\x02"
        b"\x00\x00\x00"
        b"\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    with open(path, "wb") as f:
        f.write(png_bytes)


def _make_vision_response(findings: list[dict]) -> str:
    return json.dumps({"findings": findings})


# ---------------------------------------------------------------------------
# Context manager that patches analyze_image and _load_artifact at their
# respective source locations (both are lazily imported inside visual_review).
# ---------------------------------------------------------------------------

def _patch_vision_and_artifacts(vision_return=None, artifact_return=None):
    """Return a contextmanager that patches both vision and artifact loading."""
    from contextlib import ExitStack, contextmanager

    @contextmanager
    def _cm():
        mock_vision = AsyncMock(return_value=vision_return if vision_return is not None else _make_vision_response([]))
        mock_artifact = AsyncMock(return_value=artifact_return)
        with patch("src.tools.vision.analyze_image", mock_vision), \
             patch("mr_roboto.executors.social_preview_check._load_artifact", mock_artifact):
            yield mock_vision, mock_artifact

    return _cm()


# ---------------------------------------------------------------------------
# Tests: soft-skip cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_captured_paths_soft_skip():
    """T3C: empty captured_paths triggers auto-capture; with no preview URL
    capture soft-skips → visual_review also soft-skips with 'no preview'."""
    from mr_roboto.visual_review import visual_review

    result = await visual_review(
        mission_id=1,
        step_id="step_1",
        captured_paths=[],
    )

    assert result["skipped"] is True
    assert result["verdict"] == "pass"
    assert result["findings"] == []
    # T3C: reason is now "no preview" (capture_screenshots found no URL)
    assert "no preview" in result["reason"] or "captured" in result["reason"]


@pytest.mark.asyncio
async def test_vision_capability_unavailable_soft_skip():
    """When vision call returns a capability-unavailable error string → soft-skip."""
    from mr_roboto.visual_review import visual_review

    with tempfile.TemporaryDirectory() as tmpdir:
        captured = os.path.join(tmpdir, "home_light_desktop.png")
        _fake_png(captured)

        capability_error = "Error: no eligible model with vision capability"

        with _patch_vision_and_artifacts(vision_return=capability_error):
            result = await visual_review(
                mission_id=1,
                step_id="step_1",
                captured_paths=[captured],
                workspace_path=tmpdir,
            )

    assert result["skipped"] is True
    assert result["verdict"] == "pass"
    assert result["reason"] == "vision_capability_unavailable"


@pytest.mark.asyncio
async def test_vision_capability_unavailable_raised_exception():
    """When analyze_image raises with a capability error → soft-skip."""
    from mr_roboto.visual_review import visual_review

    with tempfile.TemporaryDirectory() as tmpdir:
        captured = os.path.join(tmpdir, "home_light_desktop.png")
        _fake_png(captured)

        async def raise_vision(*args, **kwargs):
            raise RuntimeError("no eligible model with vision capability")

        with patch("src.tools.vision.analyze_image", side_effect=raise_vision), \
             patch("mr_roboto.executors.social_preview_check._load_artifact", new_callable=AsyncMock) as mock_artifact:
            mock_artifact.return_value = None
            result = await visual_review(
                mission_id=1,
                step_id="step_1",
                captured_paths=[captured],
                workspace_path=tmpdir,
            )

    assert result["skipped"] is True
    assert result["verdict"] == "pass"
    assert result["reason"] == "vision_capability_unavailable"


# ---------------------------------------------------------------------------
# Tests: DIFF mode (baseline present)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_diff_mode_calls_analyze_with_two_images():
    """When a baseline exists, analyze_image is called with [captured, baseline]."""
    from mr_roboto.visual_review import visual_review

    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_dir = os.path.join(tmpdir, "baseline")
        os.makedirs(baseline_dir)

        captured = os.path.join(tmpdir, "home_light_desktop.png")
        baseline = os.path.join(baseline_dir, "home_light_desktop.png")
        _fake_png(captured)
        _fake_png(baseline)

        with _patch_vision_and_artifacts() as (mock_vision, _):
            result = await visual_review(
                mission_id=1,
                step_id="step_1",
                captured_paths=[captured],
                baseline_dir=baseline_dir,
                workspace_path=tmpdir,
            )

    # analyze_image must have been called with a LIST of two paths
    call_args = mock_vision.call_args
    filepaths_arg = call_args[0][0] if call_args[0] else call_args[1].get("filepaths")
    assert isinstance(filepaths_arg, list), "expected list of paths"
    assert len(filepaths_arg) == 2
    assert filepaths_arg[0] == captured
    assert filepaths_arg[1] == baseline

    assert result["skipped"] is False
    assert result["verdict"] == "pass"


# ---------------------------------------------------------------------------
# Tests: AUDIT mode (no baseline)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_mode_calls_analyze_with_one_image():
    """When no baseline exists, analyze_image is called with [captured] only."""
    from mr_roboto.visual_review import visual_review

    with tempfile.TemporaryDirectory() as tmpdir:
        # No baseline directory / file
        captured = os.path.join(tmpdir, "home_light_desktop.png")
        _fake_png(captured)

        with _patch_vision_and_artifacts() as (mock_vision, _):
            result = await visual_review(
                mission_id=1,
                step_id="step_1",
                captured_paths=[captured],
                workspace_path=tmpdir,
            )

    call_args = mock_vision.call_args
    filepaths_arg = call_args[0][0] if call_args[0] else call_args[1].get("filepaths")
    assert isinstance(filepaths_arg, list)
    assert len(filepaths_arg) == 1
    assert filepaths_arg[0] == captured

    assert result["skipped"] is False
    assert result["verdict"] == "pass"


# ---------------------------------------------------------------------------
# Tests: severity rule mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_color_delta_e_above_threshold_becomes_blocker():
    from mr_roboto.visual_review import visual_review

    findings = [{
        "kind": "color",
        "component": "Button",
        "description": "Color mismatch with delta_e=6.2 difference",
        "expected": "#1a73e8",
        "observed": "#2980b9",
        "severity_hint": "warning",
    }]

    with tempfile.TemporaryDirectory() as tmpdir:
        captured = os.path.join(tmpdir, "home_light_desktop.png")
        _fake_png(captured)

        with _patch_vision_and_artifacts(vision_return=_make_vision_response(findings)):
            result = await visual_review(
                mission_id=1,
                step_id="s1",
                captured_paths=[captured],
                workspace_path=tmpdir,
            )

    assert result["verdict"] == "fail"
    assert len(result["findings"]) == 1
    assert result["findings"][0]["severity"] == "blocker"


@pytest.mark.asyncio
async def test_shadow_finding_becomes_info():
    from mr_roboto.visual_review import visual_review

    findings = [{
        "kind": "shadow",
        "component": "Card",
        "description": "Shadow elevation slightly different",
        "expected": "4dp shadow",
        "observed": "6dp shadow",
        "severity_hint": "warning",
    }]

    with tempfile.TemporaryDirectory() as tmpdir:
        captured = os.path.join(tmpdir, "home_light_desktop.png")
        _fake_png(captured)

        with _patch_vision_and_artifacts(vision_return=_make_vision_response(findings)):
            result = await visual_review(
                mission_id=1,
                step_id="s1",
                captured_paths=[captured],
                workspace_path=tmpdir,
            )

    assert result["verdict"] == "pass"  # no blocker
    assert result["findings"][0]["severity"] == "info"


@pytest.mark.asyncio
async def test_verdict_fails_on_blocker():
    from mr_roboto.visual_review import visual_review

    findings = [{
        "kind": "missing_component",
        "component": "NavBar",
        "description": "NavBar component is missing from the page",
        "expected": "NavBar present at top",
        "observed": "No NavBar found",
        "severity_hint": "blocker",
    }]

    with tempfile.TemporaryDirectory() as tmpdir:
        captured = os.path.join(tmpdir, "home_light_desktop.png")
        _fake_png(captured)

        with _patch_vision_and_artifacts(vision_return=_make_vision_response(findings)):
            result = await visual_review(
                mission_id=1,
                step_id="s1",
                captured_paths=[captured],
                workspace_path=tmpdir,
            )

    assert result["verdict"] == "fail"
    assert result["findings"][0]["severity"] == "blocker"
    assert result["skipped"] is False


# ---------------------------------------------------------------------------
# Tests: filename parsing → breakpoint/route/mode
# ---------------------------------------------------------------------------


def test_filename_parsing_standard_format():
    from mr_roboto.visual_review import _parse_filename

    meta = _parse_filename("/some/path/home_light_desktop.png")
    assert meta["route"] == "home"
    assert meta["mode"] == "light"
    assert meta["breakpoint"] == "desktop"


def test_filename_parsing_unknown_format():
    from mr_roboto.visual_review import _parse_filename

    meta = _parse_filename("/some/path/unknownfile.png")
    # Should not raise; returns empty strings
    assert isinstance(meta["route"], str)
    assert isinstance(meta["mode"], str)
    assert isinstance(meta["breakpoint"], str)


@pytest.mark.asyncio
async def test_finding_carries_filename_metadata():
    from mr_roboto.visual_review import visual_review

    findings = [{
        "kind": "color",
        "component": "Hero",
        "description": "Color mismatch detected",
        "expected": "blue",
        "observed": "red",
        "severity_hint": "warning",
    }]

    with tempfile.TemporaryDirectory() as tmpdir:
        captured = os.path.join(tmpdir, "checkout_dark_mobile.png")
        _fake_png(captured)

        with _patch_vision_and_artifacts(vision_return=_make_vision_response(findings)):
            result = await visual_review(
                mission_id=1,
                step_id="s1",
                captured_paths=[captured],
                workspace_path=tmpdir,
            )

    f = result["findings"][0]
    assert f["route"] == "checkout"
    assert f["mode"] == "dark"
    assert f["breakpoint"] == "mobile"
    assert f["source"] == "visual_review"


# ---------------------------------------------------------------------------
# Tests: .kutay/visual.yaml threshold loading
# ---------------------------------------------------------------------------


def test_load_thresholds_defaults():
    """When no .kutay/visual.yaml is present, defaults are returned."""
    from mr_roboto.visual_review import _load_thresholds

    with patch("os.path.isfile", return_value=False):
        thresholds = _load_thresholds()

    assert thresholds["color_delta_e"] == 4
    assert thresholds["layout_shift_px"] == 2


def test_load_thresholds_from_yaml(tmp_path):
    """When .kutay/visual.yaml exists with custom values, they are used."""
    import sys
    # The module is in sys.modules under its dotted name even though
    # mr_roboto.__init__ shadows the attribute with the function.
    import importlib
    vr_mod = importlib.import_module("mr_roboto.visual_review")
    _load_thresholds = vr_mod._load_thresholds

    visual_yaml = tmp_path / "visual.yaml"
    visual_yaml.write_text("color_delta_e: 2\nlayout_shift_px: 5\n", encoding="utf-8")

    import builtins
    _orig_open = builtins.open
    _orig_isfile = os.path.isfile

    def patched_isfile(p):
        if isinstance(p, str) and p.endswith("visual.yaml"):
            return True
        return _orig_isfile(p)

    def patched_open(p, *args, **kwargs):
        if isinstance(p, str) and p.endswith("visual.yaml"):
            return _orig_open(str(visual_yaml), *args, **kwargs)
        return _orig_open(p, *args, **kwargs)

    with patch("os.path.isfile", side_effect=patched_isfile), \
         patch("builtins.open", side_effect=patched_open):
        thresholds = _load_thresholds()

    assert thresholds["color_delta_e"] == 2.0
    assert thresholds["layout_shift_px"] == 5.0


@pytest.mark.asyncio
async def test_threshold_override_changes_severity():
    """With a stricter color_delta_e of 10, a ΔE=6.2 finding becomes warning."""
    import importlib
    vr_mod = importlib.import_module("mr_roboto.visual_review")
    visual_review = vr_mod.visual_review

    findings = [{
        "kind": "color",
        "component": "Button",
        "description": "Color mismatch with delta_e=6.2 difference",
        "expected": "#1a73e8",
        "observed": "#2980b9",
        "severity_hint": "warning",
    }]

    with tempfile.TemporaryDirectory() as tmpdir:
        captured = os.path.join(tmpdir, "home_light_desktop.png")
        _fake_png(captured)

        # Override threshold to 10 (higher than 6.2 → not a blocker)
        # Patch _load_thresholds at the module level using sys.modules
        import sys
        orig_fn = vr_mod._load_thresholds
        vr_mod._load_thresholds = lambda: {"color_delta_e": 10, "layout_shift_px": 2}
        try:
            with _patch_vision_and_artifacts(vision_return=_make_vision_response(findings)):
                result = await visual_review(
                    mission_id=1,
                    step_id="s1",
                    captured_paths=[captured],
                    workspace_path=tmpdir,
                )
        finally:
            vr_mod._load_thresholds = orig_fn

    # ΔE=6.2 < threshold=10 → not blocker; fall through to "warning"
    assert result["findings"][0]["severity"] == "warning"
    assert result["verdict"] == "pass"


# ---------------------------------------------------------------------------
# Z4 T3C — self-capture (empty/None captured_paths calls capture_screenshots)
# ---------------------------------------------------------------------------


def _patch_capture_fn(monkeypatch, mock_fn):
    """Patch capture_screenshots in sys.modules['mr_roboto.capture_screenshots'].

    visual_review does ``from mr_roboto.capture_screenshots import
    capture_screenshots as _capture`` lazily at call time. Since mr_roboto.__init__
    imports capture_screenshots (the function) and shadows the name, we must patch
    via sys.modules to reach the actual submodule object.
    """
    import sys
    import importlib
    # Ensure the submodule is loaded
    importlib.import_module("mr_roboto.capture_screenshots")
    cs_mod = sys.modules["mr_roboto.capture_screenshots"]
    monkeypatch.setattr(cs_mod, "capture_screenshots", mock_fn)


@pytest.mark.asyncio
async def test_t3c_none_captured_paths_calls_capture(monkeypatch):
    """T3C: None captured_paths → capture_screenshots is called (mocked)."""
    from mr_roboto.visual_review import visual_review

    captured_png_path = "/fake/ws/step1/home_light_375.png"

    mock_capture = AsyncMock(return_value={
        "ok": True,
        "skipped": False,
        "captured_paths": [captured_png_path],
        "route_count": 1,
        "frame_count": 1,
    })

    _patch_capture_fn(monkeypatch, mock_capture)

    with _patch_vision_and_artifacts(vision_return='{"findings":[]}'):
        result = await visual_review(
            mission_id=5,
            step_id="step_capture_test",
            captured_paths=None,
            routes=["/home"],
            produces=["pages/index.tsx"],
            workspace_path="/fake/ws",
        )

    mock_capture.assert_awaited_once()
    call_kwargs = mock_capture.call_args.kwargs
    assert call_kwargs["mission_id"] == 5
    assert call_kwargs["step_id"] == "step_capture_test"
    assert call_kwargs["routes"] == ["/home"]
    assert call_kwargs["produces"] == ["pages/index.tsx"]
    assert call_kwargs["workspace_path"] == "/fake/ws"

    assert result["skipped"] is False
    assert result["verdict"] == "pass"


@pytest.mark.asyncio
async def test_t3c_empty_list_captured_paths_calls_capture(monkeypatch):
    """T3C: empty list captured_paths → capture_screenshots is called."""
    from mr_roboto.visual_review import visual_review

    mock_capture = AsyncMock(return_value={
        "ok": True,
        "skipped": False,
        "captured_paths": [],  # capture returns empty → soft-skip
        "route_count": 0,
        "frame_count": 0,
    })

    _patch_capture_fn(monkeypatch, mock_capture)

    result = await visual_review(
        mission_id=1,
        step_id="step_no_frames",
        captured_paths=[],
    )

    mock_capture.assert_awaited_once()
    assert result["skipped"] is True
    assert result["verdict"] == "pass"


@pytest.mark.asyncio
async def test_t3c_capture_soft_skip_propagates(monkeypatch):
    """T3C: when capture soft-skips (no preview URL), visual_review soft-skips too."""
    from mr_roboto.visual_review import visual_review

    mock_capture = AsyncMock(return_value={
        "ok": True,
        "skipped": True,
        "reason": "no real preview_url available",
        "captured_paths": [],
    })

    _patch_capture_fn(monkeypatch, mock_capture)

    result = await visual_review(
        mission_id=2,
        step_id="step_no_preview",
        captured_paths=None,
    )

    assert result["skipped"] is True
    assert result["verdict"] == "pass"
    assert "no preview" in result["reason"]


@pytest.mark.asyncio
async def test_t3c_provided_paths_skips_capture(monkeypatch):
    """T3C: when captured_paths is a non-empty list, capture_screenshots is NOT called."""
    from mr_roboto.visual_review import visual_review

    mock_capture = AsyncMock()
    _patch_capture_fn(monkeypatch, mock_capture)

    with tempfile.TemporaryDirectory() as tmpdir:
        captured = os.path.join(tmpdir, "home_light_375.png")
        _fake_png(captured)

        with _patch_vision_and_artifacts(vision_return='{"findings":[]}'):
            result = await visual_review(
                mission_id=3,
                step_id="step_direct",
                captured_paths=[captured],
                workspace_path=tmpdir,
            )

    mock_capture.assert_not_awaited()
    assert result["skipped"] is False
