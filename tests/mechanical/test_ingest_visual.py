"""Tests for the B7+C16 visual-ingest mechanical action.

Covers:
- happy path (2 images → visual_brief.md with both sections)
- failure: no_images_at_paths
- failure: vision_capability_unavailable
- schema: emitted artifact has _schema_version: "1" + valid YAML frontmatter
- Telegram handler: photo upload triggers clarify-shape question with REPLY_KEYBOARD

The vision LLM call is mocked in every test — no real model is loaded.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

import mr_roboto
from mr_roboto import ingest_visual as iv_mod


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────


def _make_image(path: Path) -> None:
    """Write a tiny dummy image file (content is irrelevant; we mock vision)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\nfake-png-bytes")


def _vision_response(intent: str = "test screenshot") -> str:
    return json.dumps({
        "inferred_intent": intent,
        "structural_elements": ["header", "hero", "cta"],
        "color_palette_inferred": ["#ffffff", "#1a73e8", "#202124"],
        "style_keywords": ["clean", "minimal"],
        "text_excerpts": ["Sign up free"],
        "confidence": 0.82,
    })


# ──────────────────────────────────────────────────────────────────────────────
# happy path
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_happy_path_two_images_produces_visual_brief(tmp_path, monkeypatch):
    img1 = tmp_path / "src" / "a.png"
    img2 = tmp_path / "src" / "b.png"
    _make_image(img1)
    _make_image(img2)
    workspace = tmp_path / "ws_mission_42"

    fake_vision = AsyncMock(side_effect=[
        _vision_response("first image"),
        _vision_response("second image"),
    ])
    with patch("src.tools.vision.analyze_image", new=fake_vision):
        action = await mr_roboto.run({
            "id": 1,
            "mission_id": 42,
            "payload": {
                "action": "ingest_visual",
                "mission_id": 42,
                "file_paths": [str(img1), str(img2)],
                "purpose": "competitor_screenshot",
                "workspace_path": str(workspace),
            },
        })

    assert action.status == "completed", action.error
    res = action.result
    assert res["ok"] is True
    assert res["image_count"] == 2
    assert res["purpose"] == "competitor_screenshot"
    assert res["_schema_version"] == "1"

    artifact_path = Path(res["artifact_path"])
    assert artifact_path.exists()
    assert artifact_path.name == "visual_brief.md"
    assert artifact_path.parent.name == ".intake"

    md = artifact_path.read_text(encoding="utf-8")
    # both image sections rendered
    assert "## Image 1 — a.png" in md
    assert "## Image 2 — b.png" in md
    assert "first image" in md
    assert "second image" in md
    assert fake_vision.await_count == 2


# ──────────────────────────────────────────────────────────────────────────────
# failure: no_images_at_paths
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_images_at_paths_empty_list(tmp_path):
    action = await mr_roboto.run({
        "id": 1,
        "mission_id": 42,
        "payload": {
            "action": "ingest_visual",
            "mission_id": 42,
            "file_paths": [],
            "purpose": "competitor_screenshot",
            "workspace_path": str(tmp_path),
        },
    })
    assert action.status == "failed"
    assert "no_images_at_paths" in (action.error or "")


@pytest.mark.asyncio
async def test_no_images_at_paths_all_missing(tmp_path):
    action = await mr_roboto.run({
        "id": 1,
        "mission_id": 42,
        "payload": {
            "action": "ingest_visual",
            "mission_id": 42,
            "file_paths": [str(tmp_path / "missing1.png"),
                           str(tmp_path / "missing2.png")],
            "purpose": "moodboard",
            "workspace_path": str(tmp_path),
        },
    })
    assert action.status == "failed"
    assert "no_images_at_paths" in (action.error or "")


# ──────────────────────────────────────────────────────────────────────────────
# failure: vision_capability_unavailable
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_vision_capability_unavailable(tmp_path):
    img = tmp_path / "src" / "x.png"
    _make_image(img)
    workspace = tmp_path / "ws"

    # The real tool prefixes capability failures with "Error:" — mimic that.
    fake_vision = AsyncMock(
        return_value="Error: vision call failed (no eligible model for vision)"
    )
    with patch("src.tools.vision.analyze_image", new=fake_vision):
        action = await mr_roboto.run({
            "id": 1,
            "mission_id": 7,
            "payload": {
                "action": "ingest_visual",
                "mission_id": 7,
                "file_paths": [str(img)],
                "purpose": "wireframe_sketch",
                "workspace_path": str(workspace),
            },
        })

    assert action.status == "failed"
    assert "vision_capability_unavailable" in (action.error or "")


# ──────────────────────────────────────────────────────────────────────────────
# schema verification
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_artifact_schema_version_and_valid_yaml(tmp_path):
    img = tmp_path / "src" / "only.png"
    _make_image(img)
    workspace = tmp_path / "ws"

    fake_vision = AsyncMock(return_value=_vision_response("schema check"))
    with patch("src.tools.vision.analyze_image", new=fake_vision):
        action = await mr_roboto.run({
            "id": 1,
            "mission_id": 99,
            "payload": {
                "action": "ingest_visual",
                "mission_id": 99,
                "file_paths": [str(img)],
                "purpose": "inspiration",
                "workspace_path": str(workspace),
            },
        })

    assert action.status == "completed"
    md = Path(action.result["artifact_path"]).read_text(encoding="utf-8")
    assert md.startswith("---\n")
    fm_end = md.index("\n---\n", 4)
    fm_text = md[4:fm_end]
    fm = yaml.safe_load(fm_text)
    assert fm["_schema_version"] == "1"
    assert fm["mission_id"] == 99
    assert fm["purpose"] == "inspiration"
    assert isinstance(fm["images"], list) and len(fm["images"]) == 1
    img_fm = fm["images"][0]
    assert img_fm["inferred_intent"] == "schema check"
    assert "header" in img_fm["structural_elements"]
    assert "#1a73e8" in img_fm["color_palette_inferred"]
    assert 0.0 <= img_fm["confidence"] <= 1.0
    assert fm["evidence_refs"] == [str(img)]


@pytest.mark.asyncio
async def test_per_image_failure_skips_and_continues(tmp_path):
    img1 = tmp_path / "src" / "good.png"
    img2 = tmp_path / "src" / "bad.png"
    _make_image(img1)
    _make_image(img2)
    workspace = tmp_path / "ws"

    # First image OK; second hits a per-image error (corrupt etc.).
    fake_vision = AsyncMock(side_effect=[
        _vision_response("good one"),
        "Error analyzing image: corrupted bytes",
    ])
    with patch("src.tools.vision.analyze_image", new=fake_vision):
        action = await mr_roboto.run({
            "id": 1,
            "mission_id": 5,
            "payload": {
                "action": "ingest_visual",
                "mission_id": 5,
                "file_paths": [str(img1), str(img2)],
                "purpose": "moodboard",
                "workspace_path": str(workspace),
            },
        })

    assert action.status == "completed"
    res = action.result
    assert res["image_count"] == 1
    assert res["skipped"] and res["skipped"][0]["path"] == str(img2)
    md = Path(res["artifact_path"]).read_text(encoding="utf-8")
    assert "Skipped:" in md  # bad image gets a skipped section


# ──────────────────────────────────────────────────────────────────────────────
# Telegram handler: photo upload → clarify-shape with keyboard
# ──────────────────────────────────────────────────────────────────────────────


def _make_handler_instance():
    """Build a minimal TelegramInterface stub.

    We instantiate without going through __init__ because the real bot
    construction requires env vars + a polling loop. We attach only the
    attributes handle_photo touches.
    """
    from src.app.telegram_bot import TelegramInterface
    inst = TelegramInterface.__new__(TelegramInterface)
    inst._pending_action = {}
    inst._kb_state = {}
    return inst


class _FakePhoto:
    def __init__(self, file_id: str, file_unique_id: str):
        self.file_id = file_id
        self.file_unique_id = file_unique_id


@pytest.mark.asyncio
async def test_telegram_photo_upload_prompts_purpose_with_keyboard(tmp_path, monkeypatch):
    from src.app import telegram_bot as tg_mod

    inst = _make_handler_instance()

    # Fake mission
    async def fake_get_active_missions():
        return [{"id": 11, "title": "T", "workflow": "i2p"}]
    monkeypatch.setattr(tg_mod, "get_active_missions", fake_get_active_missions)

    # Fake workspace dir resolution
    workspace = tmp_path / "ws_11"
    workspace.mkdir()
    monkeypatch.setattr(
        "src.tools.workspace.get_mission_workspace",
        lambda mid: str(workspace),
    )

    # Fake telegram file download
    async def fake_download(target: str):
        Path(target).write_bytes(b"PNG-bytes")
    fake_file = MagicMock()
    fake_file.download_to_drive = AsyncMock(side_effect=fake_download)
    bot = MagicMock()
    bot.get_file = AsyncMock(return_value=fake_file)

    captured: dict = {}

    async def fake_reply(self_, update_or_msg, text, **kwargs):
        captured["text"] = text
        captured["reply_markup"] = kwargs.get("reply_markup")

    monkeypatch.setattr(
        tg_mod.TelegramInterface, "_reply", fake_reply, raising=True
    )

    # Build a fake update
    update = MagicMock()
    update.message.chat_id = 1234
    update.message.photo = [_FakePhoto("fid_lo", "uniq"),
                             _FakePhoto("fid_hi", "uniq")]
    context = MagicMock()
    context.bot = bot

    await inst.handle_photo(update, context)

    # File was saved into the mission's intake/visuals directory
    saved = list((workspace / ".intake" / "visuals").glob("photo_uniq.jpg"))
    assert saved and saved[0].exists()

    # _pending_action stashed for next message
    pending = inst._pending_action.get(1234)
    assert pending is not None
    assert pending["command"] == "_visual_purpose"
    assert pending["mission_id"] == 11
    assert pending["file_path"] == str(saved[0])

    # User got prompted with a keyboard (REPLY_KEYBOARD-style ReplyKeyboardMarkup)
    assert captured["reply_markup"] is not None
    assert "Bu ne için" in captured["text"]
    from telegram import ReplyKeyboardMarkup
    assert isinstance(captured["reply_markup"], ReplyKeyboardMarkup)
    flat = [b.text for row in captured["reply_markup"].keyboard for b in row]
    # All five purpose labels + cancel must appear
    assert "🖼 Rakip Ekran" in flat
    assert "🎨 Moodboard" in flat
    assert "✏️ Wireframe" in flat
    assert "💡 İlham" in flat
    assert "📱 Mevcut Ürün" in flat
    assert "❌ İptal" in flat


@pytest.mark.asyncio
async def test_telegram_photo_upload_no_active_mission(tmp_path, monkeypatch):
    from src.app import telegram_bot as tg_mod

    inst = _make_handler_instance()

    async def fake_get_active_missions():
        return []
    monkeypatch.setattr(tg_mod, "get_active_missions", fake_get_active_missions)

    captured: dict = {}

    async def fake_reply(self_, update_or_msg, text, **kwargs):
        captured["text"] = text

    monkeypatch.setattr(
        tg_mod.TelegramInterface, "_reply", fake_reply, raising=True
    )

    update = MagicMock()
    update.message.chat_id = 1234
    update.message.photo = [_FakePhoto("fid", "uniq")]
    context = MagicMock()

    await inst.handle_photo(update, context)
    assert "aktif bir görev yok" in captured["text"]
    assert 1234 not in inst._pending_action
