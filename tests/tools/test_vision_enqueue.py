"""
Tests for CPS SP4a migration: analyze_image() in vision.py uses husam.run
(synchronous single-call worker) instead of the blocking await_inline=True primitive.
"""
from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
async def _reset_db_singleton():
    """Drop module-level cached aiosqlite connection between tests so each
    monkeypatch.setattr on DB_PATH binds a fresh connection."""
    import src.infra.db as _dbmod
    if _dbmod._db_connection is not None:
        try:
            await _dbmod._db_connection.close()
        except Exception:
            pass
    _dbmod._db_connection = None
    yield
    if _dbmod._db_connection is not None:
        try:
            await _dbmod._db_connection.close()
        except Exception:
            pass
    _dbmod._db_connection = None


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
async def test_analyze_image_calls_husam_with_tool_call_spec(tmp_path, monkeypatch, fake_image):
    """analyze_image must call husam.run with a raw_dispatch tool_call spec carrying needs_vision."""
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    captured = {}

    async def fake_run(spec):
        captured["spec"] = spec
        return {"content": "A minimal 1x1 PNG test image."}

    with patch("husam.run", fake_run), \
         patch("dogru_mu_samet.assess") as mock_assess:
        mock_assess.return_value = MagicMock(is_degenerate=False, summary="ok")
        from src.tools.vision import analyze_image
        result = await analyze_image(fake_image, "What do you see?")

    llm_call = captured["spec"]["context"]["llm_call"]
    assert captured["spec"]["kind"] == "tool_call"
    assert llm_call["raw_dispatch"] is True
    assert llm_call.get("needs_vision") is True
    assert llm_call["call_category"] == "main_work"
    assert "1x1" in result or len(result) > 0


@pytest.mark.asyncio
async def test_analyze_image_no_await_inline_in_module():
    """Guard: vision.py must not use the blocking await_inline primitive anymore."""
    import pathlib
    _root = pathlib.Path(__file__).resolve().parents[2]
    src = (_root / "src" / "tools" / "vision.py").read_text(encoding="utf-8")
    offenders = [ln for ln in src.splitlines()
                 if "await_inline=True" in ln and not ln.lstrip().startswith("#")]
    assert not offenders, f"vision.py still uses await_inline: {offenders}"


@pytest.mark.asyncio
async def test_analyze_image_degenerate_output_returns_error(tmp_path, monkeypatch, fake_image):
    """Degenerate vision output is still caught and reported."""
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    async def fake_run(spec):
        return {"content": "aaaa aaaa aaaa"}

    with patch("husam.run", fake_run), \
         patch("dogru_mu_samet.assess") as mock_assess:
        mock_assess.return_value = MagicMock(is_degenerate=True, summary="repetitive")
        from src.tools.vision import analyze_image
        result = await analyze_image(fake_image)

    assert result.startswith("Error")
    assert "degenerate" in result


@pytest.mark.asyncio
async def test_analyze_image_returns_error_string_when_husam_raises(tmp_path, monkeypatch, fake_image):
    """When husam.run raises, analyze_image returns an error string (outer except)."""
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    async def fake_run(spec):
        raise RuntimeError("vision model unavailable")

    with patch("husam.run", fake_run):
        from src.tools.vision import analyze_image
        result = await analyze_image(fake_image)

    assert result.startswith("Error")
