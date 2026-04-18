import pytest


@pytest.mark.asyncio
async def test_public_api_importable():
    import general_beckman
    assert hasattr(general_beckman, "next_task")
    assert hasattr(general_beckman, "on_task_finished")
    assert hasattr(general_beckman, "tick")


@pytest.mark.asyncio
async def test_next_task_stub_returns_none():
    import general_beckman
    assert await general_beckman.next_task() is None
