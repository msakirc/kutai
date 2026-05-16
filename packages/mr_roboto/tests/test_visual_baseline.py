"""Tests for Z4 T5A/T5B/T5C — cross-mission baseline store, component crops,
and device capture-mode hook.

All vision / Playwright calls are mocked — no real network or browser.
"""
from __future__ import annotations

import importlib as _importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Pre-import the capture_screenshots module so we can patch its attributes
_cs_mod = _importlib.import_module("mr_roboto.capture_screenshots")
# run_cmd module object — patched directly because the package __init__
# re-exports the run_cmd *function* under the same name (shadows the module
# for dotted-string patch targets).
_run_cmd_mod = _importlib.import_module("mr_roboto.run_cmd")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_png(path: str) -> None:
    """Write a minimal valid 1-pixel PNG so os.path.isfile() passes."""
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00"
        b"\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(png_bytes)


# ---------------------------------------------------------------------------
# T5A — token_hash
# ---------------------------------------------------------------------------

def test_token_hash_none_returns_sentinel():
    from mr_roboto.visual_baseline import token_hash
    assert token_hash(None) == "notokens"


def test_token_hash_empty_dict_returns_sentinel():
    from mr_roboto.visual_baseline import token_hash
    assert token_hash({}) == "notokens"


def test_token_hash_stable():
    from mr_roboto.visual_baseline import token_hash
    tokens = {"color": "#fff", "font-size": "16px"}
    h1 = token_hash(tokens)
    h2 = token_hash(tokens)
    assert h1 == h2
    assert len(h1) == 12
    assert all(c in "0123456789abcdef" for c in h1)


def test_token_hash_order_independent():
    from mr_roboto.visual_baseline import token_hash
    a = {"x": 1, "y": 2}
    b = {"y": 2, "x": 1}
    assert token_hash(a) == token_hash(b)


def test_token_hash_different_tokens_differ():
    from mr_roboto.visual_baseline import token_hash
    assert token_hash({"color": "red"}) != token_hash({"color": "blue"})


# ---------------------------------------------------------------------------
# T5A — cross_mission_baseline_dir
# ---------------------------------------------------------------------------

def test_cross_mission_baseline_dir():
    from mr_roboto.visual_baseline import cross_mission_baseline_dir
    result = cross_mission_baseline_dir("/repo", "abc123def456")
    assert result == os.path.join("/repo", ".visual_baseline", "abc123def456")


# ---------------------------------------------------------------------------
# T5A — resolve_baseline (precedence)
# ---------------------------------------------------------------------------

def test_resolve_baseline_per_mission_wins(tmp_path):
    from mr_roboto.visual_baseline import resolve_baseline

    # Create files in both dirs
    mission_dir = tmp_path / "mission_baseline"
    mission_dir.mkdir()
    cross_dir = tmp_path / "cross_baseline"
    cross_dir.mkdir()

    basename = "root_light_375.png"
    _fake_png(str(mission_dir / basename))
    _fake_png(str(cross_dir / basename))

    result = resolve_baseline(
        basename,
        mission_baseline_dir=str(mission_dir),
        cross_dir=str(cross_dir),
    )
    assert result == str(mission_dir / basename)


def test_resolve_baseline_cross_mission_fallback(tmp_path):
    from mr_roboto.visual_baseline import resolve_baseline

    mission_dir = tmp_path / "empty_mission"
    mission_dir.mkdir()
    cross_dir = tmp_path / "cross"
    cross_dir.mkdir()

    basename = "root_dark_768.png"
    _fake_png(str(cross_dir / basename))

    result = resolve_baseline(
        basename,
        mission_baseline_dir=str(mission_dir),
        cross_dir=str(cross_dir),
    )
    assert result == str(cross_dir / basename)


def test_resolve_baseline_returns_none_when_absent(tmp_path):
    from mr_roboto.visual_baseline import resolve_baseline

    mission_dir = tmp_path / "empty_mission"
    mission_dir.mkdir()
    cross_dir = tmp_path / "empty_cross"
    cross_dir.mkdir()

    result = resolve_baseline(
        "root_light_375.png",
        mission_baseline_dir=str(mission_dir),
        cross_dir=str(cross_dir),
    )
    assert result is None


def test_resolve_baseline_both_none_returns_none():
    from mr_roboto.visual_baseline import resolve_baseline
    result = resolve_baseline("root_light_375.png", mission_baseline_dir=None, cross_dir=None)
    assert result is None


# ---------------------------------------------------------------------------
# T5A — promote_to_cross_mission (idempotent)
# ---------------------------------------------------------------------------

def test_promote_to_cross_mission_creates_copy(tmp_path):
    from mr_roboto.visual_baseline import promote_to_cross_mission

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    frame = str(src_dir / "root_light_375.png")
    _fake_png(frame)

    cross_dir = str(tmp_path / "cross")

    dest = promote_to_cross_mission(frame, cross_dir)
    assert os.path.isfile(dest)
    assert os.path.basename(dest) == "root_light_375.png"


def test_promote_to_cross_mission_idempotent(tmp_path):
    from mr_roboto.visual_baseline import promote_to_cross_mission

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    frame = str(src_dir / "root_light_375.png")
    _fake_png(frame)
    cross_dir = str(tmp_path / "cross")

    dest1 = promote_to_cross_mission(frame, cross_dir)
    dest2 = promote_to_cross_mission(frame, cross_dir)
    assert dest1 == dest2
    assert os.path.isfile(dest2)


# ---------------------------------------------------------------------------
# T5A — tokens_changed
# ---------------------------------------------------------------------------

def test_tokens_changed_true_when_absent(tmp_path):
    from mr_roboto.visual_baseline import tokens_changed
    result = tokens_changed(str(tmp_path), "abc123def456")
    assert result is True


def test_tokens_changed_false_when_same(tmp_path):
    from mr_roboto.visual_baseline import _write_token_hash, tokens_changed
    thash = "abc123def456"
    _write_token_hash(str(tmp_path), thash)
    assert tokens_changed(str(tmp_path), thash) is False


def test_tokens_changed_true_when_different(tmp_path):
    from mr_roboto.visual_baseline import _write_token_hash, tokens_changed
    _write_token_hash(str(tmp_path), "oldhash")
    assert tokens_changed(str(tmp_path), "newhash") is True


# ---------------------------------------------------------------------------
# T5A — visual_review uses cross-mission baseline when no per-mission exists
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_visual_review_uses_cross_mission_baseline(tmp_path):
    """visual_review must enter DIFF mode when only a cross-mission baseline exists."""
    from mr_roboto.visual_review import visual_review

    workspace = tmp_path / "ws"
    workspace.mkdir()
    mission_ws = workspace / "mission_99"
    mission_ws.mkdir()

    # Captured frame
    captured_dir = mission_ws / ".visual" / "captured" / "step_x"
    captured_dir.mkdir(parents=True)
    frame_name = "root_light_375.png"
    captured_path = str(captured_dir / frame_name)
    _fake_png(captured_path)

    # Empty per-mission baseline (no file there)
    mission_baseline = mission_ws / ".visual" / "baseline"
    mission_baseline.mkdir(parents=True)

    # Cross-mission baseline with matching file
    # Compute the token hash for our mock design tokens
    from mr_roboto.visual_baseline import token_hash, cross_mission_baseline_dir
    tokens = {"color": "#fff"}
    thash = token_hash(tokens)
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = here
    for _ in range(3):  # tests/ → mr_roboto/ → packages/ → repo root
        repo_root = os.path.dirname(repo_root)
    cross_dir = tmp_path / ".visual_baseline" / thash
    cross_dir.mkdir(parents=True)
    _fake_png(str(cross_dir / frame_name))

    diff_was_called = []

    async def mock_analyze_image(paths, question=""):
        diff_was_called.append(len(paths))
        return json.dumps({"findings": []})

    # Patch design tokens, vision call, and repo-root computation so
    # cross_mission_baseline_dir resolves to tmp_path (not the real repo).
    with patch("src.tools.vision.analyze_image", side_effect=mock_analyze_image), \
         patch("mr_roboto.executors.social_preview_check._load_artifact",
               AsyncMock(return_value=tokens)), \
         patch("mr_roboto.visual_baseline.cross_mission_baseline_dir",
               lambda _root, _hash: str(tmp_path / ".visual_baseline" / _hash)):
        result = await visual_review(
            mission_id=99,
            step_id="step_x",
            captured_paths=[captured_path],
            baseline_dir=str(mission_baseline),
            workspace_path=str(workspace),
        )

    assert result["skipped"] is False
    # DIFF mode sends 2 images (captured + baseline); AUDIT sends 1.
    assert diff_was_called, "vision was never called"
    assert diff_was_called[0] == 2, "expected DIFF mode (2 images) but got AUDIT (1 image)"


# ---------------------------------------------------------------------------
# T5B — capture_screenshots: components arg produces extra cropped frames
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_capture_screenshots_components_produces_crops(tmp_path):
    """When components are provided, extra locator crops are captured."""
    from mr_roboto.capture_screenshots import capture_screenshots

    preview_txt = tmp_path / ".preview" / "last_preview_url.txt"
    preview_txt.parent.mkdir(parents=True)
    preview_txt.write_text("http://localhost:3000")

    captured_full: list[str] = []
    captured_crop: list[str] = []

    # Mock the locator + its screenshot method
    mock_locator = MagicMock()
    mock_locator.screenshot = AsyncMock(side_effect=lambda path: captured_crop.append(path))

    mock_page = AsyncMock()
    mock_page.add_style_tag = AsyncMock()
    mock_page.goto = AsyncMock()
    mock_page.emulate_media = AsyncMock()
    mock_page.close = AsyncMock()
    mock_page.locator = MagicMock(return_value=mock_locator)
    mock_page.screenshot = AsyncMock(
        side_effect=lambda path, full_page=False: captured_full.append(path)
    )

    mock_ctx = AsyncMock()
    mock_ctx.add_init_script = AsyncMock()
    mock_ctx.new_page = AsyncMock(return_value=mock_page)
    mock_ctx.close = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_browser = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_ctx)
    mock_browser.close = AsyncMock()

    mock_pw = AsyncMock()
    mock_pw.chromium = AsyncMock()
    mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_pw.__aenter__ = AsyncMock(return_value=mock_pw)
    mock_pw.__aexit__ = AsyncMock(return_value=False)

    mock_playwright_factory = MagicMock(return_value=mock_pw)

    with patch.object(_cs_mod, "_get_async_playwright", return_value=mock_playwright_factory), \
         patch.object(_cs_mod.importlib, "import_module", return_value=MagicMock()):
        result = await capture_screenshots(
            mission_id=7,
            step_id="step_comp",
            routes=["/"],
            workspace_path=str(tmp_path),
            components=[{"name": "header", "selector": "header"}],
        )

    assert result["ok"] is True
    assert result["skipped"] is False
    # 8 full-page frames per route (2 modes × 4 breakpoints)
    assert len(captured_full) == 8
    # 8 component crops (one per frame)
    assert len(captured_crop) == 8
    # All crops appear in captured_paths
    total = result["frame_count"]
    assert total == 16, f"expected 16 frames (8 full + 8 crops), got {total}"


@pytest.mark.asyncio
async def test_capture_screenshots_no_components_unchanged(tmp_path):
    """Without components, behaviour is exactly as before — 8 frames only."""
    from mr_roboto.capture_screenshots import capture_screenshots

    preview_txt = tmp_path / ".preview" / "last_preview_url.txt"
    preview_txt.parent.mkdir(parents=True)
    preview_txt.write_text("http://localhost:3000")

    captured: list[str] = []

    mock_page = AsyncMock()
    mock_page.add_style_tag = AsyncMock()
    mock_page.goto = AsyncMock()
    mock_page.emulate_media = AsyncMock()
    mock_page.close = AsyncMock()
    mock_page.screenshot = AsyncMock(side_effect=lambda path, full_page=False: captured.append(path))
    mock_page.locator = MagicMock()  # should never be called

    mock_ctx = AsyncMock()
    mock_ctx.add_init_script = AsyncMock()
    mock_ctx.new_page = AsyncMock(return_value=mock_page)
    mock_ctx.close = AsyncMock()

    mock_browser = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_ctx)
    mock_browser.close = AsyncMock()

    mock_pw = AsyncMock()
    mock_pw.chromium = AsyncMock()
    mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_pw.__aenter__ = AsyncMock(return_value=mock_pw)
    mock_pw.__aexit__ = AsyncMock(return_value=False)

    mock_playwright_factory = MagicMock(return_value=mock_pw)

    with patch.object(_cs_mod, "_get_async_playwright", return_value=mock_playwright_factory), \
         patch.object(_cs_mod.importlib, "import_module", return_value=MagicMock()):
        result = await capture_screenshots(
            mission_id=8,
            step_id="step_nocomp",
            routes=["/"],
            workspace_path=str(tmp_path),
        )

    assert result["ok"] is True
    assert result["frame_count"] == 8
    mock_page.locator.assert_not_called()


# ---------------------------------------------------------------------------
# T5C / Z5 T4a — capture_mode="device"
#
# Z5 T4a turned the former device-mode soft-skip stub into a real
# implementation (Playwright device descriptors + adb + xcrun simctl arms).
# With no preview URL, no adb and a non-macOS host every arm soft-skips, so
# the device run still returns a clean soft-skip envelope — but the reason is
# now the aggregate of per-arm reasons, not the old "Z5 not implemented"
# placeholder. Full device-mode coverage lives in
# ``test_z5_t4a_device_capture.py``.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_capture_mode_device_soft_skip_when_no_backend(tmp_path):
    """capture_mode='device' with no preview / adb / macOS → soft-skip envelope."""
    from mr_roboto.capture_screenshots import capture_screenshots

    with patch.object(_cs_mod.importlib, "import_module", return_value=MagicMock()), \
         patch.object(_cs_mod.sys, "platform", "win32"), \
         patch.object(_cs_mod.platform, "system", return_value="Windows"), \
         patch.object(_run_cmd_mod, "run_cmd",
               AsyncMock(return_value={"ok": False, "error": "executable not found: adb",
                                       "stdout_tail": "", "stderr_tail": ""})):
        result = await capture_screenshots(
            mission_id=1,
            step_id="step_device",
            routes=["/"],
            workspace_path=str(tmp_path),
            capture_mode="device",
        )

    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["capture_mode"] == "device"
    assert result["captured_paths"] == []


@pytest.mark.asyncio
async def test_capture_mode_viewport_is_default(tmp_path):
    """capture_mode omitted / 'viewport' does NOT dispatch to device mode."""
    from mr_roboto.capture_screenshots import capture_screenshots

    # No preview URL → soft-skip for the viewport reason (no preview URL).
    result = await capture_screenshots(
        mission_id=1,
        step_id="step_viewport_default",
        routes=["/"],
        workspace_path=str(tmp_path),
        # capture_mode not passed → defaults to "viewport"
    )

    # Should reach the viewport URL check, not the device dispatch.
    assert result["skipped"] is True
    assert result["capture_mode"] == "viewport"
    assert "preview_url" in result.get("reason", "")
