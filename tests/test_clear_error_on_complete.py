"""Regression for handoff item B: ``tasks.error`` must clear on
successful completion.

When attempt N fails with error X and attempt N+1 succeeds, the row's
error column kept X — misled post-mortems and /queue UI. The fix
clears error + error_category + next_retry_at + retry_reason +
failed_in_phase on every completed-transition path.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from general_beckman.apply import _apply_complete
from general_beckman.result_router import Complete


def _capture_kwargs():
    """Return AsyncMock + accessor for the last call's kwargs."""
    m = AsyncMock(return_value=None)
    return m


@pytest.mark.asyncio
async def test_apply_complete_clears_failure_metadata():
    fake_update = _capture_kwargs()
    with patch("src.infra.db.update_task", fake_update):
        action = Complete(task_id=42, result="ok")
        await _apply_complete({"id": 42}, action)

    assert fake_update.call_count == 1
    kwargs = fake_update.call_args.kwargs
    assert kwargs["status"] == "completed"
    # The whole point of (B):
    assert kwargs["error"] is None
    assert kwargs["error_category"] is None
    assert kwargs["next_retry_at"] is None
    assert kwargs["retry_reason"] is None
    assert kwargs["failed_in_phase"] is None


@pytest.mark.asyncio
async def test_apply_complete_serializes_dict_result():
    """Pre-existing behaviour preserved alongside (B)."""
    fake_update = _capture_kwargs()
    with patch("src.infra.db.update_task", fake_update):
        action = Complete(task_id=42, result={"a": 1, "b": [2, 3]})
        await _apply_complete({"id": 42}, action)

    kwargs = fake_update.call_args.kwargs
    import json as _json
    assert _json.loads(kwargs["result"]) == {"a": 1, "b": [2, 3]}
    # Failure metadata still cleared.
    assert kwargs["error"] is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
