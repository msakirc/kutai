"""Test: yalayut_synth rubric builds the same user message as the old inline prompt."""


def test_yalayut_synth_uses_foundry_build():
    from finch.build import build_messages

    msgs = build_messages("yalayut_synth", {
        "name_original": "my-cool-skill",
        "raw_text": "Does something useful with files.",
    })
    assert len(msgs) == 2
    assert msgs[1]["role"] == "user"
    assert "JSON manifest" in msgs[1]["content"]
    assert "my-cool-skill" in msgs[1]["content"]
    assert "Does something useful with files." in msgs[1]["content"]
    assert "intent_keywords" in msgs[1]["content"]


def test_yalayut_synth_messages_char_exact():
    """User content must be character-exact vs old inline prompt."""
    from finch.build import build_messages

    name_original = "awesome-lib"
    raw_text_slice = "A library for doing awesome things in Python."

    # Old inline prompt from source
    old_prompt = (
        "You normalize a software-artifact description into a JSON manifest. "
        "Return ONLY a JSON object with keys: intent_keywords (array of "
        "lowercase strings), mechanizable (boolean), kind (one of "
        "prompt_skill|shell_recipe|procedure|agent_config), install_cmd "
        "(string or null), auth_env (string env-var name or null).\n\n"
        f"Artifact name: {name_original}\n"
        f"Description / README:\n{raw_text_slice}\n"
    )

    msgs = build_messages("yalayut_synth", {
        "name_original": name_original,
        "raw_text": raw_text_slice,
    })

    assert msgs[1]["content"] == old_prompt, "user content mismatch vs old prompt"
