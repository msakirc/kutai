"""Test that ``ArtifactStore.store`` rejects empty / whitespace / None
writes (architectural fix discussed 2026-04-26).

Mission 46 had 9 design artifacts (screen_specifications, prd_final,
business_rules, etc.) stored as empty strings because the upstream
``post_execute_workflow_step`` fell through to ``store.store(name,
output_value)`` with ``output_value=""`` (workspace recovery missed
files due to bind-mount mismatch). Every implementer agent in phase 8
then saw "specs missing" and DLQ'd. The architectural fix gates at
the boundary: ArtifactStore refuses to persist empty content.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.workflows.engine.artifacts import ArtifactStore


@pytest.mark.asyncio
async def test_empty_string_rejected():
    store = ArtifactStore(use_db=False)
    ok = await store.store(1, "x", "")
    assert ok is False
    assert await store.retrieve(1, "x") is None


@pytest.mark.asyncio
async def test_whitespace_only_rejected():
    store = ArtifactStore(use_db=False)
    ok = await store.store(1, "x", "   \n\t  ")
    assert ok is False
    assert await store.retrieve(1, "x") is None


@pytest.mark.asyncio
async def test_none_rejected():
    store = ArtifactStore(use_db=False)
    ok = await store.store(1, "x", None)  # type: ignore[arg-type]
    assert ok is False


@pytest.mark.asyncio
async def test_non_string_rejected():
    store = ArtifactStore(use_db=False)
    ok = await store.store(1, "x", 123)  # type: ignore[arg-type]
    assert ok is False


@pytest.mark.asyncio
async def test_real_content_accepted():
    store = ArtifactStore(use_db=False)
    ok = await store.store(1, "x", "real content here")
    assert ok is True
    assert await store.retrieve(1, "x") == "real content here"


@pytest.mark.asyncio
async def test_existing_artifact_not_overwritten_by_empty():
    """Empty rejection means a later empty-store doesn't clobber prior good value."""
    store = ArtifactStore(use_db=False)
    await store.store(1, "x", "good value")
    ok = await store.store(1, "x", "")
    assert ok is False
    assert await store.retrieve(1, "x") == "good value"


@pytest.mark.asyncio
async def test_db_path_skipped_for_rejected_value():
    """Rejection happens before any DB write — no leakage to blackboard."""
    store = ArtifactStore(use_db=True)
    fake_update = AsyncMock()
    with patch(
        "src.collaboration.blackboard.update_blackboard_entry", fake_update,
    ):
        ok = await store.store(1, "x", "")
    assert ok is False
    fake_update.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
