"""The board context layer must not byte-slice the blackboard block.

Regression: the caller wrapped the (structurally-truncated) blackboard
block in ``truncate_to_tokens`` — a raw ``text[:max_chars]`` cut — which
re-severed it mid-content. The layer budget must be enforced *inside*
``format_blackboard_for_prompt`` (structural) and the caller must not
slice the result again.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.base import BaseAgent


class _Agent(BaseAgent):
    name = "fake"


def _big_board() -> dict:
    return {
        "architecture": {},
        "files": {},
        "decisions": [],
        "open_issues": [],
        # Constraints are rendered in full (no item cap) — the layer
        # budget is the only limit, so this overflows it.
        "constraints": [f"constraint {i}: " + "x" * 60 for i in range(120)],
        "dependency_map": {},
    }


@pytest.mark.asyncio
async def test_board_block_is_not_byte_sliced(monkeypatch):
    from src.memory import context_policy as cp
    # Realistic analyst-shape policy: board shares the pool with deps+rag,
    # so its slice is small (weight 2/10). With model_ctx=4096 that is
    # ~327 tok = ~1308 chars — below the blackboard block size, which is
    # exactly when the caller's raw truncate_to_tokens used to byte-slice.
    monkeypatch.setattr(
        cp, "get_context_policy",
        lambda *_a, **_k: {"deps": True, "rag": True, "board": True},
    )
    monkeypatch.setattr(cp, "apply_heuristics", lambda task, p: p)

    monkeypatch.setattr(
        "src.collaboration.blackboard.get_or_create_blackboard",
        AsyncMock(return_value=_big_board()),
    )

    task = {
        "id": 1,
        "title": "T",
        "description": "do thing",
        "mission_id": 42,
        "context": json.dumps({"mission_id": 42}),
    }

    # Call build_user_context directly with a fixed small model_ctx so the
    # board budget is deterministic (independent of any model loaded on the
    # host running the suite).
    from src.runtime.context import build_user_context

    fake_store = AsyncMock()
    fake_store.retrieve = AsyncMock(return_value=None)
    with patch(
        "src.workflows.engine.hooks.get_artifact_store",
        return_value=fake_store,
    ):
        ctx, _ = await build_user_context(_Agent(), task, model_ctx=4096)

    # The raw-slice marker from truncate_to_tokens must never touch the board.
    assert "[truncated to budget]" not in ctx
    # Structural omission note instead (budget did force a drop here).
    assert "read_blackboard" in ctx
    # Every surviving constraint bullet is whole (byte-identical to source).
    for ln in ctx.split("\n"):
        if ln.startswith("  - constraint "):
            assert ln.endswith("x" * 60), f"sliced constraint line: {ln!r}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
