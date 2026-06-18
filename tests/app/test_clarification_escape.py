"""Escape-phrase vs substantive-answer disambiguation for clarification replies.

Regression: single-word escape tokens (skip/stop/cancel/...) were matched as
substrings anywhere in the message, so a substantive clarification answer that
merely *contained* one of those words ("skip the onboarding, keep signup") was
swallowed — dismissed as an escape or classified as not-a-response. Escape
intent must require the token to be (essentially) the whole short message.
"""
from __future__ import annotations

import asyncio

from src.app.telegram_bot import TelegramInterface


def _bot():
    # Bypass __init__ (which builds the PTB Application).
    return object.__new__(TelegramInterface)


def test_bare_escape_words_are_escape():
    b = _bot()
    for w in ["skip", "stop", "cancel", "dismiss", "ignore", "stop it",
              "cancel that please", "/restart", "forget it", "never mind"]:
        assert b._message_is_escape(w) is True, w


def test_substantive_answer_with_escape_substring_is_not_escape():
    b = _bot()
    assert b._message_is_escape(
        "skip the onboarding but keep the signup flow"
    ) is False
    assert b._message_is_escape(
        "stop charging users monthly in the new plan"
    ) is False


def test_clarification_keeps_substantive_answer_containing_escape_word():
    b = _bot()
    r1 = asyncio.run(b._is_likely_clarification_response(
        "skip the onboarding but keep signup", {"title": "x"}))
    assert r1 is True
    r2 = asyncio.run(b._is_likely_clarification_response(
        "stop charging users monthly please", {"title": "x"}))
    assert r2 is True


def test_clarification_bare_escape_is_not_a_response():
    b = _bot()
    assert asyncio.run(
        b._is_likely_clarification_response("skip", {"title": "x"})) is False
    assert asyncio.run(
        b._is_likely_clarification_response("cancel", {"title": "x"})) is False


def test_clarification_short_answer_still_true():
    b = _bot()
    assert asyncio.run(
        b._is_likely_clarification_response("the blue one", {"title": "x"})) is True


def test_clarification_new_task_still_false():
    b = _bot()
    assert asyncio.run(b._is_likely_clarification_response(
        "can you build a new dashboard", {"title": "x"})) is False
