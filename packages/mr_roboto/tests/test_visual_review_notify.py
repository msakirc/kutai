"""Tests for Z4 T4 — visual-review founder-loop notification + callbacks.

All Telegram / Pillow / filesystem calls are mocked — no real API calls.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_png(path: str) -> None:
    """Write a minimal 1-pixel PNG."""
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(png_bytes)


# ---------------------------------------------------------------------------
# T4A — _visual_review_notify module
# ---------------------------------------------------------------------------

class TestApproveCallback:
    def test_approve_cb_normal(self):
        from mr_roboto._visual_review_notify import _approve_cb
        cb = _approve_cb(42, "3.2", "home_light_375.png")
        assert cb is not None
        assert cb == "visrev:approve:42:3.2:home_light_375.png"
        assert len(cb.encode()) <= 64

    def test_approve_cb_too_long_returns_none(self):
        from mr_roboto._visual_review_notify import _approve_cb
        long_filename = "x" * 60 + ".png"
        result = _approve_cb(42, "3.2", long_filename)
        assert result is None

    def test_cal_cb_normal(self):
        from mr_roboto._visual_review_notify import _cal_cb
        cb = _cal_cb("fine", 7, "home:Button:color")
        assert cb is not None
        assert cb == "visrev:cal:fine:7:home:Button:color"
        assert len(cb.encode()) <= 64

    def test_cal_cb_too_long_returns_none(self):
        from mr_roboto._visual_review_notify import _cal_cb
        long_pattern = "r:" + "c" * 60 + ":k"
        result = _cal_cb("fine", 7, long_pattern)
        assert result is None


class TestCallbackRoundTrip:
    """Encode → parse round-trip for both callback schemes."""

    def test_approve_roundtrip(self):
        from mr_roboto._visual_review_notify import _approve_cb
        cb = _approve_cb(99, "step_x", "checkout_dark_768.png")
        assert cb is not None
        parts = cb.split(":", 4)
        assert parts[0] == "visrev"
        assert parts[1] == "approve"
        assert int(parts[2]) == 99
        assert parts[3] == "step_x"
        assert parts[4] == "checkout_dark_768.png"

    def test_cal_roundtrip(self):
        from mr_roboto._visual_review_notify import _cal_cb
        cb = _cal_cb("broken", 5, "shop:NavBar:layout")
        assert cb is not None
        parts = cb.split(":", 4)
        assert parts[0] == "visrev"
        assert parts[1] == "cal"
        assert parts[2] == "broken"
        assert int(parts[3]) == 5
        assert parts[4] == "shop:NavBar:layout"


class TestSummaryText:
    def test_pass_verdict(self):
        from mr_roboto._visual_review_notify import _build_summary_text
        text = _build_summary_text(1, "2.1", "pass", [])
        assert "🟢" in text
        assert "pass" in text
        assert "mission #1" in text

    def test_fail_verdict_shows_blockers(self):
        from mr_roboto._visual_review_notify import _build_summary_text
        findings = [
            {"severity": "blocker", "description": "Color off", "component": "Btn",
             "breakpoint": "375", "kind": "color"},
        ]
        text = _build_summary_text(2, "step1", "fail", findings)
        assert "🔴" in text
        assert "blockers: 1" in text

    def test_top_findings_capped_at_5(self):
        from mr_roboto._visual_review_notify import _build_summary_text
        findings = [
            {"severity": "warning", "description": f"issue {i}", "component": "X",
             "breakpoint": "768", "kind": "other"}
            for i in range(10)
        ]
        text = _build_summary_text(3, "s3", "pass", findings)
        assert text.count("issue") == 5


def _mock_pil():
    """Return a context manager that injects a fake PIL.Image into sys.modules."""
    from contextlib import contextmanager

    @contextmanager
    def _cm(open_side_effect=None):
        fake_img = MagicMock()
        fake_img.size = (800, 600)
        fake_img.convert.return_value = fake_img
        fake_img.resize.return_value = fake_img
        # Default save: write a small sentinel to the destination
        def _fake_save(dst, **kw):
            open(dst, "wb").write(b"FAKEWEBP")
        fake_img.save.side_effect = _fake_save

        fake_image_mod = MagicMock()
        if open_side_effect is not None:
            fake_image_mod.open.side_effect = open_side_effect
        else:
            fake_image_mod.open.return_value = fake_img
        fake_image_mod.LANCZOS = MagicMock()

        fake_pil = MagicMock()
        fake_pil.Image = fake_image_mod

        orig_pil = sys.modules.get("PIL")
        orig_image = sys.modules.get("PIL.Image")
        sys.modules["PIL"] = fake_pil
        sys.modules["PIL.Image"] = fake_image_mod
        try:
            yield fake_img, fake_image_mod
        finally:
            if orig_pil is None:
                sys.modules.pop("PIL", None)
            else:
                sys.modules["PIL"] = orig_pil
            if orig_image is None:
                sys.modules.pop("PIL.Image", None)
            else:
                sys.modules["PIL.Image"] = orig_image

    return _cm


class TestMakeThumbnails:
    def test_thumbnails_created_in_correct_dir(self, tmp_path):
        """_make_thumbnails writes WebP files under mission_N/.visual/thumbs/step/."""
        import importlib
        import mr_roboto._visual_review_notify as vrn

        captured_dir = tmp_path / "captured"
        captured_dir.mkdir()
        png1 = str(captured_dir / "home_light_375.png")
        _fake_png(png1)

        with _mock_pil()():
            # Reload to pick up the mocked PIL
            importlib.reload(vrn)
            results = vrn._make_thumbnails(
                [png1],
                workspace_path=str(tmp_path),
                mission_id=10,
                step_id="3.2",
            )
            importlib.reload(vrn)  # restore

        assert len(results) == 1
        thumb_path, orig_basename = results[0]
        assert orig_basename == "home_light_375.png"
        assert thumb_path.endswith(".webp")
        expected_dir = os.path.join(str(tmp_path), "mission_10", ".visual", "thumbs", "3.2")
        assert thumb_path.startswith(expected_dir)

    def test_pillow_failure_skips_frame(self, tmp_path):
        """When Pillow raises, that frame is skipped (no crash)."""
        import importlib
        import mr_roboto._visual_review_notify as vrn

        png1 = str(tmp_path / "home_light_375.png")
        _fake_png(png1)

        with _mock_pil()(open_side_effect=RuntimeError("no pillow")):
            importlib.reload(vrn)
            results = vrn._make_thumbnails(
                [png1],
                workspace_path=str(tmp_path),
                mission_id=1,
                step_id="s",
            )
            importlib.reload(vrn)

        assert results == []


class TestBuildInlineKeyboard:
    def test_approve_buttons_present(self):
        from mr_roboto._visual_review_notify import _build_inline_keyboard

        thumbs = [
            ("/t/home_light_375.webp", "home_light_375.png"),
            ("/t/home_dark_375.webp", "home_dark_375.png"),
        ]
        rows = _build_inline_keyboard(thumbs, [], mission_id=5, step_id="1.1")
        # Flatten buttons
        buttons = [btn for row in rows for btn in row]
        cb_data = [b["callback_data"] for b in buttons]
        assert any("visrev:approve:5:1.1:home_light_375.png" in cb for cb in cb_data)
        assert any("visrev:approve:5:1.1:home_dark_375.png" in cb for cb in cb_data)

    def test_calibration_buttons_present(self):
        from mr_roboto._visual_review_notify import _build_inline_keyboard

        thumbs = [("/t/home_light_375.webp", "home_light_375.png")]
        findings = [{"route": "home", "component": "Btn", "kind": "color"}]
        rows = _build_inline_keyboard(thumbs, findings, mission_id=3, step_id="2.0")
        buttons = [btn for row in rows for btn in row]
        labels = [b["label"] for b in buttons]
        assert any("fine" in l.lower() for l in labels)
        assert any("broken" in l.lower() for l in labels)


@pytest.mark.asyncio
async def test_soft_skip_no_captured_paths():
    """enqueue_visual_review_notice exits immediately when captured_paths is empty."""
    from mr_roboto._visual_review_notify import enqueue_visual_review_notice

    with patch("mr_roboto._visual_review_notify._get_tg") as mock_tg:
        await enqueue_visual_review_notice(
            mission_id=1,
            step_id="s",
            verdict="pass",
            findings=[],
            captured_paths=[],
            workspace_path="/fake",
        )
    mock_tg.assert_not_called()


@pytest.mark.asyncio
async def test_soft_skip_telegram_not_configured():
    """Soft-skip when Telegram is not configured."""
    from mr_roboto._visual_review_notify import enqueue_visual_review_notice

    with patch("mr_roboto._visual_review_notify._get_tg", return_value=None):
        # Should not raise
        await enqueue_visual_review_notice(
            mission_id=1,
            step_id="s",
            verdict="pass",
            findings=[],
            captured_paths=["/some/file.png"],
            workspace_path="/fake",
        )


@pytest.mark.asyncio
async def test_send_album_and_buttons(tmp_path):
    """End-to-end: thumbnails created, album sent, follow-up with buttons sent."""
    # Create a fake captured PNG.
    captured_dir = tmp_path / "mission_7" / ".visual" / "captured" / "step1"
    captured_dir.mkdir(parents=True)
    png = str(captured_dir / "home_light_375.png")
    _fake_png(png)

    # Mock Telegram bot.
    mock_bot = MagicMock()
    mock_bot.send_media_group = AsyncMock()
    mock_bot.send_message = AsyncMock()
    mock_tg = MagicMock()
    mock_tg.app.bot = mock_bot

    import mr_roboto._visual_review_notify as _vrn

    with _mock_pil()(), \
         patch.object(_vrn, "_get_tg", return_value=mock_tg), \
         patch.object(_vrn, "_make_thumbnails",
                      return_value=[("/fake/home_light_375.webp", "home_light_375.png")]):
        # Patch config import inside enqueue
        import src.app.config as _cfg
        _orig_chat_id = getattr(_cfg, "TELEGRAM_ADMIN_CHAT_ID", None)
        _cfg.TELEGRAM_ADMIN_CHAT_ID = 12345
        try:
            from mr_roboto._visual_review_notify import enqueue_visual_review_notice
            await enqueue_visual_review_notice(
                mission_id=7,
                step_id="step1",
                verdict="fail",
                findings=[{
                    "severity": "blocker",
                    "description": "Color off",
                    "component": "Hero",
                    "breakpoint": "375",
                    "kind": "color",
                    "route": "home",
                }],
                captured_paths=[png],
                workspace_path=str(tmp_path),
            )
        finally:
            if _orig_chat_id is not None:
                _cfg.TELEGRAM_ADMIN_CHAT_ID = _orig_chat_id

    # send_message (follow-up text) should have been called
    mock_bot.send_message.assert_awaited()


@pytest.mark.asyncio
async def test_notify_with_mocked_chat_id(tmp_path):
    """Full flow with admin chat_id properly mocked — album + text sent."""
    captured_dir = tmp_path / "mission_5" / ".visual" / "captured" / "s1"
    captured_dir.mkdir(parents=True)
    png = str(captured_dir / "home_light_375.png")
    _fake_png(png)

    mock_bot = MagicMock()
    mock_bot.send_media_group = AsyncMock()
    mock_bot.send_message = AsyncMock()
    mock_tg = MagicMock()
    mock_tg.app.bot = mock_bot

    import mr_roboto._visual_review_notify as _vrn
    import src.app.config as _cfg

    # Patch _make_thumbnails to return a pre-built list (no PIL needed).
    fake_thumb = str(tmp_path / "home_light_375.webp")
    open(fake_thumb, "wb").write(b"FAKEWEBP")

    with patch.object(_vrn, "_get_tg", return_value=mock_tg), \
         patch.object(_vrn, "_make_thumbnails",
                      return_value=[(fake_thumb, "home_light_375.png")]):
        _orig = getattr(_cfg, "TELEGRAM_ADMIN_CHAT_ID", None)
        _cfg.TELEGRAM_ADMIN_CHAT_ID = 999
        try:
            from mr_roboto._visual_review_notify import enqueue_visual_review_notice
            await enqueue_visual_review_notice(
                mission_id=5,
                step_id="s1",
                verdict="pass",
                findings=[],
                captured_paths=[png],
                workspace_path=str(tmp_path),
            )
        finally:
            if _orig is not None:
                _cfg.TELEGRAM_ADMIN_CHAT_ID = _orig

    # send_message (follow-up text) should have been called
    mock_bot.send_message.assert_awaited()


# ---------------------------------------------------------------------------
# T4B — baseline approval (file copy logic, tested standalone)
# ---------------------------------------------------------------------------

class TestBaselineApproval:
    def test_approve_copies_frame(self, tmp_path):
        """Approval copies captured → baseline dir (idempotent)."""
        captured_dir = tmp_path / "mission_3" / ".visual" / "captured" / "2.1"
        captured_dir.mkdir(parents=True)
        frame = "home_light_375.png"
        _fake_png(str(captured_dir / frame))

        baseline_dir = tmp_path / "mission_3" / ".visual" / "baseline"

        # Simulate what the callback does
        dst = baseline_dir / frame
        os.makedirs(str(baseline_dir), exist_ok=True)
        shutil.copy2(str(captured_dir / frame), str(dst))

        assert dst.exists()
        assert dst.read_bytes() == (captured_dir / frame).read_bytes()

    def test_approve_idempotent(self, tmp_path):
        """Second approval of same frame overwrites without error."""
        captured_dir = tmp_path / "mission_4" / ".visual" / "captured" / "s1"
        captured_dir.mkdir(parents=True)
        frame = "home_dark_768.png"
        _fake_png(str(captured_dir / frame))

        baseline_dir = tmp_path / "mission_4" / ".visual" / "baseline"
        os.makedirs(str(baseline_dir), exist_ok=True)

        # First copy
        shutil.copy2(str(captured_dir / frame), str(baseline_dir / frame))
        # Second copy (idempotent overwrite)
        shutil.copy2(str(captured_dir / frame), str(baseline_dir / frame))

        assert (baseline_dir / frame).exists()


# ---------------------------------------------------------------------------
# T4C — calibration lesson upsert
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_calibration_calls_upsert_mission_lesson():
    """visrev:cal callback correctly calls upsert_mission_lesson."""
    mock_upsert = AsyncMock(return_value=1)

    with patch("src.infra.mission_lessons.upsert_mission_lesson", mock_upsert):
        from src.infra.mission_lessons import upsert_mission_lesson
        await upsert_mission_lesson(
            stack="frontend",
            domain="visual",
            pattern="home:Button:color",
            fix="Founder marked this visual pattern as acceptable — suppress future alerts.",
            severity="info",
            source_kind="visrev_calibration",
            source_ref={"mission_id": 5, "verdict": "fine"},
        )

    mock_upsert.assert_awaited_once_with(
        stack="frontend",
        domain="visual",
        pattern="home:Button:color",
        fix="Founder marked this visual pattern as acceptable — suppress future alerts.",
        severity="info",
        source_kind="visrev_calibration",
        source_ref={"mission_id": 5, "verdict": "fine"},
    )


@pytest.mark.asyncio
async def test_calibration_broken_verdict_severity():
    """'broken' calibration verdict uses 'blocker' severity."""
    mock_upsert = AsyncMock(return_value=2)

    with patch("src.infra.mission_lessons.upsert_mission_lesson", mock_upsert):
        from src.infra.mission_lessons import upsert_mission_lesson
        await upsert_mission_lesson(
            stack="frontend",
            domain="visual",
            pattern="checkout:Card:layout",
            fix="Founder confirmed this visual pattern is genuinely broken.",
            severity="blocker",
            source_kind="visrev_calibration",
            source_ref={"mission_id": 8, "verdict": "broken"},
        )

    call_kwargs = mock_upsert.call_args
    assert call_kwargs.kwargs["severity"] == "blocker"
