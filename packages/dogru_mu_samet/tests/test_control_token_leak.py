"""Detect leaked model function-calling control tokens (2026-06-20).

A cloaked OpenRouter model (owl-alpha = LongCat-Flash) was selected for analyst
work and emitted its native function-calling special tokens — <longcat_tool_call>,
<longcat_arg_key>, <longcat_arg_value> — as literal text in its output (its
tool-call attempts were never parsed on the json/text path). The deterministic
shape gate flagged the <...> placeholders and DLQ'd the task after retries on the
SAME model.

Treating leaked control tokens as degenerate output lets the call-path quality
gate (hallederiz_kadir _check_quality → retryable quality_failure) reject the
response mid-loop and re-select a clean model — model-agnostic, no hardcoded
aliases.
"""
from dogru_mu_samet.detectors import check_control_token_leak
from dogru_mu_samet import assess


def test_detects_namespaced_control_tokens():
    text = ("Competitive positioning analysis. <longcat_tool_call> "
            "<longcat_arg_key> <longcat_arg_value>")
    score, breached, reason = check_control_token_leak(text)
    assert breached is True
    assert reason == "control_token_leak"
    assert score >= 3


def test_clean_prose_not_flagged():
    text = ("Our product beats competitors on price, speed, and Turkish-market "
            "depth. The home screen surfaces habits and errands. ") * 4
    score, breached, reason = check_control_token_leak(text)
    assert breached is False
    assert reason is None


def test_assess_flags_leak_as_degenerate():
    text = "## Positioning\nWe win because <longcat_arg_key> and <longcat_arg_value>."
    r = assess(text)
    assert r.is_degenerate is True
    assert "control_token_leak" in r.reasons
