"""Test: copy_compliance rubric builds the same user message as the old inline prompt."""
import pytest


def test_copy_compliance_uses_foundry_build():
    from finch.build import build_messages

    msgs = build_messages("copy_compliance", {
        "copy_excerpt": "We never sell your data.",
        "policy_excerpt": "We sell anonymized data to partners.",
    })
    assert len(msgs) == 2
    assert msgs[1]["role"] == "user"
    assert "legal-copy compliance reviewer" in msgs[1]["content"]
    assert "We never sell your data." in msgs[1]["content"]
    assert "We sell anonymized data to partners." in msgs[1]["content"]
    assert "contradicts" in msgs[1]["content"]


def test_copy_compliance_messages_char_exact():
    """User content must be character-exact vs old inline prompt."""
    from finch.build import build_messages

    copy_excerpt = "We help teams save 10 hours/week guaranteed."
    policy_excerpt = "We collect usage data to improve the service."

    # Old inline prompt from source
    old_prompt = (
        "You are a legal-copy compliance reviewer. "
        "Compare the MARKETING COPY below against the PRIVACY POLICY excerpt. "
        "Identify any claim in the marketing copy that directly contradicts or "
        "is materially inconsistent with what the privacy policy says about data "
        "collection, usage, sharing, or retention.\n\n"
        "Respond with ONLY a JSON object — no prose, no markdown fences:\n"
        '{"contradicts": "yes"|"no"|"unclear", "citation": "<offending sentence or empty string>"}\n\n'
        "MARKETING COPY:\n"
        f"{copy_excerpt}\n\n"
        "PRIVACY POLICY (excerpt):\n"
        f"{policy_excerpt}"
    )

    msgs = build_messages("copy_compliance", {
        "copy_excerpt": copy_excerpt,
        "policy_excerpt": policy_excerpt,
    })

    assert msgs[1]["content"] == old_prompt, "user content mismatch vs old prompt"
