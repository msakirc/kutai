import asyncio
import pytest
from src.security.risk_assessor import assess_risk


@pytest.mark.asyncio
async def test_assess_risk_is_awaitable():
    result = await assess_risk(
        task_title="delete all files",
        task_description="rm -rf /",
    )
    assert isinstance(result, dict)
    assert "score" in result
    assert "needs_approval" in result


@pytest.mark.asyncio
async def test_assess_risk_low_risk_task():
    result = await assess_risk(
        task_title="list files in current directory",
        task_description="ls",
    )
    assert isinstance(result, dict)
    assert result["needs_approval"] is False
