"""Z5 T4a — mobile device visual review.

Covers ``capture_mode="device"`` in ``mr_roboto.capture_screenshots`` and the
device-namespaced-frame extension in ``mr_roboto.visual_review`` /
``mr_roboto.visual_baseline``.

All Playwright + ``run_cmd`` calls are mocked so the suite runs on Windows CI
with no browser, no ``adb``, no ``xcrun``.

What is asserted
----------------
- device mode produces device-namespaced frames (``{route}_{device}_{mode}``)
- the ``xcrun simctl`` arm soft-skips on a non-macOS host
- the ``adb`` arm soft-skips when ``adb`` is absent and when no device attached
- ``capture_mode="viewport"`` frame-name output is byte-for-byte unchanged
  (snapshot of the frame-name list)
- ``visual_review`` parses device-namespaced filenames and routes them to
  device-mode baselines (basename resolution); AUDIT mode works for device
  frames with no baseline
"""
from __future__ import annotations

import importlib as _importlib
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Pre-import the modules so their attributes can be patched.
# ``mr_roboto.run_cmd`` is fetched as a module object explicitly: the package
# ``__init__.py`` re-exports the ``run_cmd`` *function* under the same name,
# which shadows the submodule for dotted-string patch targets.
_cs_mod = _importlib.import_module("mr_roboto.capture_screenshots")
_run_cmd_mod = _importlib.import_module("mr_roboto.run_cmd")

from mr_roboto.capture_screenshots import (
    _DEVICE_DESCRIPTORS,
    _device_slug,
    capture_screenshots,
)
from mr_roboto.visual_review import _parse_filename


# ===========================================================================
# Helpers
# ===========================================================================

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


def _build_mock_playwright(captured: list[str], *, devices: dict | None = None):
    """Return a mock ``async_playwright`` factory that records screenshot paths.

    ``devices`` is the ``pw.devices`` registry — defaults to descriptors for
    every entry in ``_DEVICE_DESCRIPTORS`` so the Playwright arm can resolve
    each one.
    """
    if devices is None:
        devices = {
            name: {
                "viewport": {"width": 390, "height": 844},
                "device_scale_factor": 3,
                "user_agent": f"mock-ua-{name}",
                "is_mobile": True,
                "has_touch": True,
            }
            for name in _DEVICE_DESCRIPTORS
        }

    async def _fake_screenshot(path, full_page=False):
        captured.append(str(path))
        os.makedirs(os.path.dirname(str(path)), exist_ok=True)
        open(str(path), "wb").close()

    mock_page = AsyncMock()
    mock_page.add_style_tag = AsyncMock()
    mock_page.goto = AsyncMock()
    mock_page.emulate_media = AsyncMock()
    mock_page.screenshot = _fake_screenshot
    mock_page.close = AsyncMock()

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
    # pw.devices is a plain dict-like registry — MagicMock so __getitem__ works.
    mock_pw.devices = devices
    mock_pw.__aenter__ = AsyncMock(return_value=mock_pw)
    mock_pw.__aexit__ = AsyncMock(return_value=False)

    return lambda: mock_pw


def _adb_absent_result(*args, **kwargs):
    """run_cmd result emulating ``adb`` not found on PATH."""
    return {
        "exit": -1, "stdout_tail": "", "stderr_tail": "",
        "duration_s": 0.0, "timed_out": False, "ok": False,
        "error": "executable not found: adb",
    }


# ===========================================================================
# _device_slug unit
# ===========================================================================

def test_device_slug_iphone():
    assert _device_slug("iPhone 14") == "iphone_14"


def test_device_slug_pixel():
    assert _device_slug("Pixel 7") == "pixel_7"


def test_device_descriptor_set_has_ios_and_android():
    """The representative set must carry at least one iOS and one Android shape."""
    slugs = {_device_slug(d) for d in _DEVICE_DESCRIPTORS}
    assert any("iphone" in s for s in slugs), "no iOS-shaped device"
    assert any("pixel" in s for s in slugs), "no Android-shaped device"


# ===========================================================================
# _parse_filename — device-namespaced frames
# ===========================================================================

def test_parse_filename_playwright_device_frame():
    meta = _parse_filename("home_iphone_14_light.png")
    assert meta["device"] == "iphone_14"
    assert meta["route"] == "home"
    assert meta["mode"] == "light"
    assert meta["breakpoint"] == ""


def test_parse_filename_playwright_device_nested_route():
    meta = _parse_filename("foo_bar_pixel_7_light.png")
    assert meta["device"] == "pixel_7"
    assert meta["route"] == "foo_bar"
    assert meta["mode"] == "light"


def test_parse_filename_adb_real_device_frame():
    meta = _parse_filename("device_emulator_5554_android.png")
    assert meta["device"] == "emulator_5554_android"
    assert meta["route"] == ""
    assert meta["breakpoint"] == ""


def test_parse_filename_simctl_real_device_frame():
    meta = _parse_filename("device_ios_simulator.png")
    assert meta["device"] == "ios_simulator"
    assert meta["route"] == ""


def test_parse_filename_viewport_frame_unchanged():
    """Z4 viewport frames still parse to route/mode/breakpoint, device empty."""
    meta = _parse_filename("root_light_375.png")
    assert meta["route"] == "root"
    assert meta["mode"] == "light"
    assert meta["breakpoint"] == "375"
    assert meta["device"] == ""


# ===========================================================================
# capture_mode="device" — Playwright device-descriptor arm
# ===========================================================================

@pytest.mark.asyncio
async def test_device_mode_produces_device_namespaced_frames(tmp_path):
    """device mode (Playwright arm) writes one frame per route × device,
    namespaced ``{route}_{device}_{mode}.png``."""
    ws = tmp_path / "mission_1"
    ws.mkdir()
    preview = ws / ".preview"
    preview.mkdir()
    (preview / "last_preview_url.txt").write_text(
        "https://expo-preview.example.com\n", encoding="utf-8"
    )

    captured: list[str] = []
    factory = _build_mock_playwright(captured)

    # adb arm: patch run_cmd at its import site (lazy import inside the arm).
    with patch.object(_cs_mod, "_get_async_playwright", return_value=factory), \
         patch.object(_cs_mod.importlib, "import_module", return_value=MagicMock()), \
         patch.object(_run_cmd_mod, "run_cmd", AsyncMock(side_effect=_adb_absent_result)):
        result = await capture_screenshots(
            mission_id=1,
            step_id="7.device_vis",
            routes=["/", "/about"],
            workspace_path=str(ws),
            capture_mode="device",
        )

    assert result["ok"] is True
    assert result["skipped"] is False
    assert result["capture_mode"] == "device"

    # 2 routes × 2 devices = 4 device-descriptor frames.
    assert len(captured) == 4
    expected_devices = {_device_slug(d) for d in _DEVICE_DESCRIPTORS}
    for route_slug in ("root", "about"):
        for dslug in expected_devices:
            name = f"{route_slug}_{dslug}_light.png"
            assert any(name in p for p in captured), f"missing device frame {name}"

    # All frames live under the per-step capture dir.
    expected_dir = os.path.join(str(ws), ".visual", "captured", "7.device_vis")
    assert all(p.startswith(expected_dir) for p in result["captured_paths"])

    # Per-arm detail is reported.
    assert "playwright_device" in result["device_detail"]
    assert result["device_detail"]["playwright_device"]["frame_count"] == 4


# ===========================================================================
# xcrun simctl arm — soft-skip on a non-macOS host
# ===========================================================================

@pytest.mark.asyncio
async def test_simctl_arm_soft_skips_on_windows(tmp_path):
    """On a non-macOS host the simctl arm soft-skips with a macOS-runner reason."""
    from mr_roboto.capture_screenshots import _capture_device_simctl_arm

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Force a non-Darwin host regardless of where the suite actually runs.
    with patch.object(_cs_mod.sys, "platform", "win32"), \
         patch.object(_cs_mod.platform, "system", return_value="Windows"):
        arm = await _capture_device_simctl_arm(
            mission_id=1,
            workspace_path=str(tmp_path),
            out_dir=str(out_dir),
        )

    assert arm["arm"] == "simctl"
    assert arm["ok"] is True
    assert arm["skipped"] is True
    assert "macOS" in arm["reason"]
    assert arm["captured_paths"] == []


@pytest.mark.asyncio
async def test_device_mode_simctl_soft_skip_in_full_run(tmp_path):
    """Full device run on Windows: simctl arm soft-skips, does not hard-fail."""
    ws = tmp_path / "mission_2"
    ws.mkdir()
    preview = ws / ".preview"
    preview.mkdir()
    (preview / "last_preview_url.txt").write_text(
        "https://expo-preview.example.com\n", encoding="utf-8"
    )

    captured: list[str] = []
    factory = _build_mock_playwright(captured)

    with patch.object(_cs_mod, "_get_async_playwright", return_value=factory), \
         patch.object(_cs_mod.importlib, "import_module", return_value=MagicMock()), \
         patch.object(_cs_mod.sys, "platform", "win32"), \
         patch.object(_cs_mod.platform, "system", return_value="Windows"), \
         patch.object(_run_cmd_mod, "run_cmd", AsyncMock(side_effect=_adb_absent_result)):
        result = await capture_screenshots(
            mission_id=2,
            step_id="step_x",
            routes=["/"],
            workspace_path=str(ws),
            capture_mode="device",
        )

    assert result["ok"] is True  # simctl soft-skip is not a hard fail
    detail = result["device_detail"]["simctl"]
    assert detail["skipped"] is True
    assert "macOS" in detail["reason"]


# ===========================================================================
# adb arm — soft-skip when adb absent / no device
# ===========================================================================

@pytest.mark.asyncio
async def test_adb_arm_soft_skips_when_adb_absent(tmp_path):
    """adb arm soft-skips cleanly when ``adb`` is not on PATH."""
    from mr_roboto.capture_screenshots import _capture_device_adb_arm

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with patch.object(_run_cmd_mod, "run_cmd", AsyncMock(side_effect=_adb_absent_result)):
        arm = await _capture_device_adb_arm(
            mission_id=1,
            workspace_path=str(tmp_path),
            out_dir=str(out_dir),
        )

    assert arm["arm"] == "adb"
    assert arm["ok"] is True
    assert arm["skipped"] is True
    assert "adb" in arm["reason"].lower()
    assert arm["captured_paths"] == []


@pytest.mark.asyncio
async def test_adb_arm_soft_skips_when_no_device(tmp_path):
    """adb present but no device attached → soft-skip with no-device reason."""
    from mr_roboto.capture_screenshots import _capture_device_adb_arm

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    async def _adb_no_device(mission_id, cmd, **kwargs):
        # `adb devices` succeeds but lists no attached device.
        if cmd[:2] == ["adb", "devices"]:
            return {
                "exit": 0,
                "stdout_tail": "List of devices attached\n\n",
                "stderr_tail": "", "duration_s": 0.1,
                "timed_out": False, "ok": True,
            }
        return {"ok": False, "error": "unexpected", "stdout_tail": "", "stderr_tail": ""}

    with patch.object(_run_cmd_mod, "run_cmd", AsyncMock(side_effect=_adb_no_device)):
        arm = await _capture_device_adb_arm(
            mission_id=1,
            workspace_path=str(tmp_path),
            out_dir=str(out_dir),
        )

    assert arm["arm"] == "adb"
    assert arm["ok"] is True
    assert arm["skipped"] is True
    assert "no Android device" in arm["reason"]
    assert arm["captured_paths"] == []


@pytest.mark.asyncio
async def test_adb_arm_captures_when_device_attached(tmp_path):
    """adb present with a device attached → screencap + pull writes a frame."""
    from mr_roboto.capture_screenshots import _capture_device_adb_arm

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    async def _adb_with_device(mission_id, cmd, **kwargs):
        if cmd[:2] == ["adb", "devices"]:
            return {
                "exit": 0,
                "stdout_tail": "List of devices attached\nemulator-5554\tdevice\n",
                "stderr_tail": "", "duration_s": 0.1,
                "timed_out": False, "ok": True,
            }
        if "pull" in cmd:
            # `adb pull` destination is the last arg — create the file.
            dest = cmd[-1]
            _fake_png(dest)
            return {"ok": True, "exit": 0, "stdout_tail": "", "stderr_tail": ""}
        # screencap / rm
        return {"ok": True, "exit": 0, "stdout_tail": "", "stderr_tail": ""}

    with patch.object(_run_cmd_mod, "run_cmd", AsyncMock(side_effect=_adb_with_device)):
        arm = await _capture_device_adb_arm(
            mission_id=1,
            workspace_path=str(tmp_path),
            out_dir=str(out_dir),
        )

    assert arm["arm"] == "adb"
    assert arm["ok"] is True
    assert arm["skipped"] is False
    assert len(arm["captured_paths"]) == 1
    name = os.path.basename(arm["captured_paths"][0])
    assert name.startswith("device_") and name.endswith("_android.png")


# ===========================================================================
# All arms skip → device run returns a soft-skip envelope (no gate on nothing)
# ===========================================================================

@pytest.mark.asyncio
async def test_device_mode_all_arms_skip_returns_soft_skip(tmp_path):
    """No preview URL + no adb + non-macOS → every arm skips → soft-skip."""
    ws = tmp_path / "mission_3"
    ws.mkdir()
    # No preview URL file at all.

    with patch.object(_cs_mod.importlib, "import_module", return_value=MagicMock()), \
         patch.object(_cs_mod.sys, "platform", "win32"), \
         patch.object(_cs_mod.platform, "system", return_value="Windows"), \
         patch.object(_run_cmd_mod, "run_cmd", AsyncMock(side_effect=_adb_absent_result)):
        result = await capture_screenshots(
            mission_id=3,
            step_id="step_y",
            routes=["/"],
            workspace_path=str(ws),
            capture_mode="device",
        )

    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["capture_mode"] == "device"
    assert result["captured_paths"] == []
    assert result["frame_count"] == 0
    # device_detail still reports each arm.
    assert set(result["device_detail"]) == {"playwright_device", "adb", "simctl"}


# ===========================================================================
# capture_mode="viewport" — byte-for-byte unchanged frame-name list
# ===========================================================================

# Frozen snapshot of the viewport-mode frame names for a single route ("/").
# 2 color modes × 4 breakpoints = 8 frames. If this list ever changes, Z4
# viewport behaviour has regressed.
_VIEWPORT_FRAME_SNAPSHOT = sorted([
    "root_light_375.png", "root_light_768.png",
    "root_light_1280.png", "root_light_1920.png",
    "root_dark_375.png", "root_dark_768.png",
    "root_dark_1280.png", "root_dark_1920.png",
])


@pytest.mark.asyncio
async def test_viewport_mode_frame_names_unchanged(tmp_path):
    """capture_mode='viewport' produces exactly the Z4 frame-name set."""
    ws = tmp_path / "mission_v"
    ws.mkdir()
    preview = ws / ".preview"
    preview.mkdir()
    (preview / "last_preview_url.txt").write_text(
        "https://preview.example.com\n", encoding="utf-8"
    )

    captured: list[str] = []
    factory = _build_mock_playwright(captured)

    with patch.object(_cs_mod, "_get_async_playwright", return_value=factory), \
         patch.object(_cs_mod.importlib, "import_module", return_value=MagicMock()):
        result = await capture_screenshots(
            mission_id=10,
            step_id="vp_step",
            routes=["/"],
            workspace_path=str(ws),
            capture_mode="viewport",
        )

    assert result["ok"] is True
    assert result["skipped"] is False
    assert result["frame_count"] == 8
    assert result["capture_mode"] == "viewport"

    names = sorted(os.path.basename(p) for p in result["captured_paths"])
    assert names == _VIEWPORT_FRAME_SNAPSHOT


@pytest.mark.asyncio
async def test_viewport_mode_default_when_omitted(tmp_path):
    """Omitting capture_mode defaults to viewport (no device dispatch)."""
    ws = tmp_path / "mission_d"
    ws.mkdir()
    # No preview URL → viewport soft-skip (NOT a device soft-skip).

    result = await capture_screenshots(
        mission_id=11,
        step_id="default_step",
        routes=["/"],
        workspace_path=str(ws),
    )

    assert result["skipped"] is True
    assert result["capture_mode"] == "viewport"
    assert "preview_url" in result["reason"]


# ===========================================================================
# visual_review — device frames route to device baselines / AUDIT mode
# ===========================================================================

@pytest.mark.asyncio
async def test_visual_review_device_frame_diff_against_device_baseline(tmp_path):
    """A device-namespaced captured frame DIFFs against a device-namespaced
    baseline of the same basename (resolution is by basename)."""
    from mr_roboto.visual_review import visual_review

    workspace = tmp_path / "ws"
    workspace.mkdir()
    mission_ws = workspace / "mission_50"
    mission_ws.mkdir()

    frame_name = "home_iphone_14_light.png"
    captured_dir = mission_ws / ".visual" / "captured" / "step_d"
    captured_path = str(captured_dir / frame_name)
    _fake_png(captured_path)

    # Per-mission baseline contains a device-namespaced frame of the same name.
    baseline_dir = mission_ws / ".visual" / "baseline"
    baseline_dir.mkdir(parents=True)
    _fake_png(str(baseline_dir / frame_name))

    image_counts: list[int] = []

    async def mock_analyze_image(paths, question=""):
        image_counts.append(len(paths))
        return json.dumps({"findings": []})

    with patch("src.tools.vision.analyze_image", side_effect=mock_analyze_image), \
         patch("mr_roboto.executors.social_preview_check._load_artifact",
               AsyncMock(return_value=None)):
        result = await visual_review(
            mission_id=50,
            step_id="step_d",
            captured_paths=[captured_path],
            baseline_dir=str(baseline_dir),
            workspace_path=str(workspace),
        )

    assert result["skipped"] is False
    assert result["verdict"] == "pass"
    # DIFF mode → 2 images (captured + device baseline).
    assert image_counts == [2], "device frame did not route to device baseline"


@pytest.mark.asyncio
async def test_visual_review_device_frame_audit_when_no_baseline(tmp_path):
    """A device-namespaced frame with no baseline falls through to AUDIT mode
    and the finding carries the device identity."""
    from mr_roboto.visual_review import visual_review

    workspace = tmp_path / "ws"
    workspace.mkdir()
    mission_ws = workspace / "mission_51"
    mission_ws.mkdir()

    frame_name = "checkout_pixel_7_light.png"
    captured_dir = mission_ws / ".visual" / "captured" / "step_e"
    captured_path = str(captured_dir / frame_name)
    _fake_png(captured_path)

    # Empty per-mission baseline dir — no matching file → AUDIT.
    baseline_dir = mission_ws / ".visual" / "baseline"
    baseline_dir.mkdir(parents=True)

    image_counts: list[int] = []

    async def mock_analyze_image(paths, question=""):
        image_counts.append(len(paths))
        return json.dumps({"findings": [
            {
                "kind": "layout", "component": "header",
                "description": "header offset by 10px",
                "expected": "flush", "observed": "shifted",
                "severity_hint": "blocker",
            }
        ]})

    with patch("src.tools.vision.analyze_image", side_effect=mock_analyze_image), \
         patch("mr_roboto.executors.social_preview_check._load_artifact",
               AsyncMock(return_value=None)):
        result = await visual_review(
            mission_id=51,
            step_id="step_e",
            captured_paths=[captured_path],
            baseline_dir=str(baseline_dir),
            workspace_path=str(workspace),
        )

    assert result["skipped"] is False
    # AUDIT mode → single image.
    assert image_counts == [1], "device frame with no baseline should AUDIT"
    assert len(result["findings"]) == 1
    finding = result["findings"][0]
    assert finding["device"] == "pixel_7"
    assert finding["route"] == "checkout"
    assert finding["breakpoint"] == ""
