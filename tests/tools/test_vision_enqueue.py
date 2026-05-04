"""
Tests for Site 7 migration: analyze_image() in vision.py calls beckman.enqueue
directly instead of dispatcher.request() alias.
"""
from __future__ import annotations

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def fake_image(tmp_path):
    """Create a minimal valid PNG-like file for testing."""
    img_path = tmp_path / "test.png"
    # Minimal 1x1 PNG bytes
    png_bytes = (
        b'\x89PNG\r\n\x1a\n'  # PNG signature
        b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
        b'\x08\x02\x00\x00\x00\x90wS\xde'
        b'\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N'
        b'\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    img_path.write_bytes(png_bytes)
    return str(img_path)


@pytest.mark.asyncio
async def test_analyze_image_enqueues_with_tool_call_kind(tmp_path, monkeypatch, fake_image):
    """analyze_image must enqueue with kind='tool_call' and await_inline=True."""
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    captured = {}

    async def fake_enqueue(spec, **kwargs):
        captured["spec"] = spec
        captured["kwargs"] = kwargs
        from general_beckman import TaskResult
        return TaskResult(
            status="completed",
            result={"content": "A minimal 1x1 PNG test image."},
            error=None,
        )

    with patch("general_beckman.enqueue", fake_enqueue), \
         patch("dogru_mu_samet.assess") as mock_assess:
        mock_assess.return_value = MagicMock(is_degenerate=False, summary="ok")
        from src.tools.vision import analyze_image
        result = await analyze_image(fake_image, "What do you see?")

    assert captured["kwargs"].get("await_inline") is True
    assert captured["spec"]["kind"] == "tool_call"
    assert captured["spec"]["context"]["llm_call"]["raw_dispatch"] is True
    assert "1x1" in result or len(result) > 0


@pytest.mark.asyncio
async def test_analyze_image_enqueue_carries_needs_vision_flag(tmp_path, monkeypatch, fake_image):
    """The llm_call payload must carry needs_vision=True."""
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    captured = {}

    async def fake_enqueue(spec, **kwargs):
        captured["spec"] = spec
        from general_beckman import TaskResult
        return TaskResult(
            status="completed",
            result={"content": "Image shows a small test png."},
            error=None,
        )

    with patch("general_beckman.enqueue", fake_enqueue), \
         patch("dogru_mu_samet.assess") as mock_assess:
        mock_assess.return_value = MagicMock(is_degenerate=False)
        from src.tools.vision import analyze_image
        await analyze_image(fake_image)

    llm_call = captured["spec"]["context"]["llm_call"]
    assert llm_call.get("needs_vision") is True
    assert llm_call["call_category"] == "main_work"


@pytest.mark.asyncio
async def test_analyze_image_parent_id_from_current_task_id(tmp_path, monkeypatch, fake_image):
    """parent_id must be taken from current_task_id ContextVar."""
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    captured = {}

    async def fake_enqueue(spec, **kwargs):
        captured["kwargs"] = kwargs
        from general_beckman import TaskResult
        return TaskResult(
            status="completed",
            result={"content": "Vision result for parent task."},
            error=None,
        )

    from src.core.heartbeat import current_task_id as _ctid
    token = _ctid.set(77)
    try:
        with patch("general_beckman.enqueue", fake_enqueue), \
             patch("dogru_mu_samet.assess") as mock_assess:
            mock_assess.return_value = MagicMock(is_degenerate=False)
            from src.tools.vision import analyze_image
            await analyze_image(fake_image)
    finally:
        _ctid.reset(token)

    assert captured["kwargs"].get("parent_id") == 77


@pytest.mark.asyncio
async def test_analyze_image_returns_error_string_on_failed_result(tmp_path, monkeypatch, fake_image):
    """When Beckman returns status='failed', analyze_image should return error string."""
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    async def fake_enqueue(spec, **kwargs):
        from general_beckman import TaskResult
        return TaskResult(status="failed", result=None, error="vision model unavailable")

    with patch("general_beckman.enqueue", fake_enqueue):
        from src.tools.vision import analyze_image
        result = await analyze_image(fake_image)

    assert result.startswith("Error")
