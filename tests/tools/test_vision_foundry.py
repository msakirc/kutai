"""Test: vision rubric builds the same text block content as the old inline question."""
import pytest
from unittest.mock import MagicMock, patch


def test_vision_rubric_text_char_exact():
    """The text block content from build_messages must be char-exact vs the question string."""
    from prompt_foundry.build import build_messages

    question = "Describe what you see in this image."
    msgs = build_messages("vision", {"question": question})
    # The user content from build_messages is the question text verbatim
    assert msgs[1]["content"] == question


def test_vision_rubric_custom_question():
    """Custom questions pass through verbatim."""
    from prompt_foundry.build import build_messages

    question = "Is this a cat or a dog? What colour is it?"
    msgs = build_messages("vision", {"question": question})
    assert msgs[1]["content"] == question


@pytest.mark.asyncio
async def test_analyze_image_message_structure_with_foundry(tmp_path, monkeypatch):
    """analyze_image messages list: [{"role":"user","content":[text_block, img_block]}].
    The text block text must equal the question string (via Foundry rubric)."""
    import src.infra.db as _db_mod
    monkeypatch.setattr(_db_mod, "DB_PATH", str(tmp_path / "test.db"))
    from src.infra.db import init_db
    await init_db()

    # Create a minimal PNG
    img_path = tmp_path / "test.png"
    png_bytes = (
        b'\x89PNG\r\n\x1a\n'
        b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
        b'\x08\x02\x00\x00\x00\x90wS\xde'
        b'\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N'
        b'\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    img_path.write_bytes(png_bytes)

    captured = {}

    async def fake_run(spec):
        captured["spec"] = spec
        return {"content": "A test image."}

    question = "What do you see?"
    with patch("husam.run", fake_run), \
         patch("dogru_mu_samet.assess") as mock_assess:
        mock_assess.return_value = MagicMock(is_degenerate=False, summary="ok")
        from src.tools.vision import analyze_image
        await analyze_image(str(img_path), question)

    messages = captured["spec"]["context"]["llm_call"]["messages"]
    assert len(messages) == 1
    user_msg = messages[0]
    assert user_msg["role"] == "user"
    content_list = user_msg["content"]
    assert isinstance(content_list, list)
    text_block = content_list[0]
    assert text_block["type"] == "text"
    # Text from Foundry rubric must equal the question string
    assert text_block["text"] == question
