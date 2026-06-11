import pathlib
from src.agents.prompt_writer import load_diffusion_prompt_template, _DEFAULT_TEMPLATE_PATH


def test_template_loads():
    body = load_diffusion_prompt_template()
    assert body is not None and len(body) > 200


def test_has_few_shot_block():
    body = load_diffusion_prompt_template()
    assert "EXAMPLE 1" in body and "EXAMPLE 2" in body
    assert any(w in body.lower() for w in ("coral", "slate", "color"))


def test_has_slot_placeholders():
    body = load_diffusion_prompt_template()
    for slot in ("{design_tokens}", "{brand_voice}", "{section_intent}",
                 "{placeholders}"):
        assert slot in body, f"missing slot: {slot}"


def test_file_under_docs_templates():
    # Use the __file__-anchored constant so this assertion is cwd-independent.
    assert _DEFAULT_TEMPLATE_PATH.is_file(), (
        f"Template not found at {_DEFAULT_TEMPLATE_PATH}"
    )
