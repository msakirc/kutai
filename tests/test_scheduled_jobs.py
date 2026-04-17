import pytest
from unittest.mock import AsyncMock, MagicMock
from src.app.scheduled_jobs import ScheduledJobs


@pytest.mark.asyncio
async def test_scheduled_jobs_construction():
    telegram = MagicMock()
    jobs = ScheduledJobs(telegram=telegram)
    assert jobs is not None
    assert callable(jobs.tick_todos)
    assert callable(jobs.tick_api_discovery)
    assert callable(jobs.tick_digest)
    assert callable(jobs.tick_price_watches)
