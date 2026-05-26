"""A dispatch TimeoutError must be classified as 'timeout' with a real message.

Bug (2026-05-26): when a call hung the full 600s wall-clock cap, a bare
asyncio.TimeoutError (str()=='') reached the orchestrator's generic
`except Exception`, which built error="TimeoutError: " with NO error_category.
Beckman then defaulted the category to 'worker' and the DB row's reason was
blank — losing both the right retry curve and any forensic 'where'. The
classifier helper tags timeouts explicitly and names the held model.
"""
from __future__ import annotations

import asyncio

from src.core.orchestrator import _dispatch_exc_to_result


def test_asyncio_timeout_classified_as_timeout():
    r = _dispatch_exc_to_result(asyncio.TimeoutError(), {})
    assert r["status"] == "failed"
    assert r["error_category"] == "timeout"
    assert r["error"], "timeout error message must not be empty"


def test_builtin_timeout_classified_as_timeout():
    r = _dispatch_exc_to_result(TimeoutError(), {})
    assert r["error_category"] == "timeout"


def test_timeout_message_includes_held_model():
    class _Pick:
        class model:
            name = "cerebras/zai-glm-4.7"

    r = _dispatch_exc_to_result(asyncio.TimeoutError(), {"_held_pick": _Pick()})
    assert "cerebras/zai-glm-4.7" in r["error"]
    assert r["error_category"] == "timeout"


def test_generic_exception_preserves_name_and_no_category():
    r = _dispatch_exc_to_result(ValueError("boom"), {})
    assert r["error"].startswith("ValueError: boom")
    assert "error_category" not in r
