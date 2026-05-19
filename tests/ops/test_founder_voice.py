import pytest
from src.ops.brand_voice import load_founder_voice


def test_unfilled_template_returns_empty(tmp_path):
    vf = tmp_path / "founder.md"
    vf.write_text(
        "---\nslug: founder\n---\n"
        "<!-- FOUNDER_VOICE_UNFILLED: delete this -->\n\n## Who I am\n",
        encoding="utf-8",
    )
    assert load_founder_voice(voices_dir=str(tmp_path)) == ""


def test_missing_file_returns_empty(tmp_path):
    assert load_founder_voice(voices_dir=str(tmp_path)) == ""


def test_filled_template_returns_body(tmp_path):
    vf = tmp_path / "founder.md"
    vf.write_text(
        "---\nslug: founder\n---\n"
        "## Who I am\n\nI build KutAI, a personal AI agent.\n",
        encoding="utf-8",
    )
    out = load_founder_voice(voices_dir=str(tmp_path))
    assert "I build KutAI" in out
    assert out.strip() != ""


def test_real_repo_template_is_unfilled():
    # The shipped template must read as unfilled until the founder edits it.
    assert load_founder_voice() == ""
