"""Test: brand_voice rubric builds the same messages as the old inline constants."""
import pytest


def test_brand_voice_uses_foundry_build():
    from prompt_foundry.build import build_messages

    msgs = build_messages("brand_voice", {
        "profile_name": "TestBrand",
        "voice_body": "Be friendly and concise.",
        "text": "Hello there! We save you 10 hours a week.",
    })
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert "brand-voice tone reviewer" in msgs[0]["content"]
    assert "0" in msgs[0]["content"] and "10" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    assert "TestBrand" in msgs[1]["content"]
    assert "Be friendly and concise." in msgs[1]["content"]
    assert "Hello there!" in msgs[1]["content"]


def test_brand_voice_messages_char_exact():
    """Message output must be character-exact vs old inline constants."""
    from prompt_foundry.build import build_messages

    # Old constants preserved here for equivalence check
    _OLD_SYSTEM = (
        "You are a brand-voice tone reviewer. Score the provided text's tone match "
        "against the stated voice profile on a scale of 0–10 (10 = perfect match). "
        "Identify the 1–2 sections most misaligned with the voice. "
        "Reply ONLY in JSON with exactly: "
        '{"score": <int 0-10>, "flagged_sections": [{"excerpt": "...", "reason": "..."}]}'
    )
    _OLD_PROMPT = (
        "Brand voice profile: {profile_name}\n"
        "Voice body guidance:\n"
        "{voice_body}\n"
        "\n"
        "Text to score:\n"
        "{text}\n"
        "\n"
        "Return JSON with score (0-10) and up to 2 flagged sections."
    )

    fields = {
        "profile_name": "Acme Corp",
        "voice_body": "Warm, professional, concise.",
        "text": "We help teams work faster.",
    }

    old_messages = [
        {"role": "system", "content": _OLD_SYSTEM},
        {"role": "user", "content": _OLD_PROMPT.format(**fields)},
    ]
    new_messages = build_messages("brand_voice", fields)

    assert new_messages[0]["content"] == old_messages[0]["content"], (
        "system mismatch"
    )
    assert new_messages[1]["content"] == old_messages[1]["content"], (
        "user mismatch"
    )
