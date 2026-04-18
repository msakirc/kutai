"""Tests for ScheduledJobs.tick_benchmark_refresh()."""
import asyncio
import time
from pathlib import Path
import pytest

from src.app.scheduled_jobs import ScheduledJobs


@pytest.fixture
def jobs(tmp_path):
    return ScheduledJobs(telegram=None)


@pytest.mark.asyncio
async def test_skips_when_cache_fresh(tmp_path, monkeypatch, jobs):
    cache_dir = tmp_path / ".benchmark_cache"
    cache_dir.mkdir()
    fresh = cache_dir / "_bulk_artificialanalysis.json"
    fresh.write_text('{"timestamp": %d, "models": {}}' % int(time.time()))

    call_count = {"n": 0}

    def _fake_refresh():
        call_count["n"] += 1
        return (0, 0)

    monkeypatch.setattr(
        "src.app.scheduled_jobs._benchmark_refresh_impl",
        _fake_refresh,
    )
    monkeypatch.setattr(
        "src.app.scheduled_jobs._benchmark_cache_dir",
        lambda: cache_dir,
    )

    await jobs.tick_benchmark_refresh()
    assert call_count["n"] == 0


@pytest.mark.asyncio
async def test_refreshes_when_cache_stale(tmp_path, monkeypatch, jobs):
    cache_dir = tmp_path / ".benchmark_cache"
    cache_dir.mkdir()
    stale = cache_dir / "_bulk_artificialanalysis.json"
    stale.write_text(
        '{"timestamp": %d, "models": {}}' % int(time.time() - 48 * 3600)
    )
    import os
    old = time.time() - 48 * 3600
    os.utime(stale, (old, old))

    call_count = {"n": 0}

    def _fake_refresh():
        call_count["n"] += 1
        return (5, 7)

    monkeypatch.setattr(
        "src.app.scheduled_jobs._benchmark_refresh_impl",
        _fake_refresh,
    )
    monkeypatch.setattr(
        "src.app.scheduled_jobs._benchmark_cache_dir",
        lambda: cache_dir,
    )

    await jobs.tick_benchmark_refresh()
    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_noop_when_refresh_in_flight(tmp_path, monkeypatch, jobs):
    from src.app import scheduled_jobs as sj_mod
    import os

    cache_dir = tmp_path / ".benchmark_cache"
    cache_dir.mkdir()
    stale = cache_dir / "_bulk_artificialanalysis.json"
    stale.write_text('{"timestamp": 0, "models": {}}')
    old = time.time() - 48 * 3600
    os.utime(stale, (old, old))

    monkeypatch.setattr(sj_mod, "_benchmark_cache_dir", lambda: cache_dir)

    release = asyncio.Event()
    call_count = {"n": 0}

    def _slow_refresh():
        call_count["n"] += 1
        import time as _t
        while not release.is_set():
            _t.sleep(0.01)
        return (0, 0)

    monkeypatch.setattr(sj_mod, "_benchmark_refresh_impl", _slow_refresh)

    first = asyncio.create_task(jobs.tick_benchmark_refresh())
    await asyncio.sleep(0)
    await jobs.tick_benchmark_refresh()
    release.set()
    await first
    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_exception_is_swallowed(tmp_path, monkeypatch, jobs, caplog):
    import logging
    import os
    from src.app import scheduled_jobs as sj_mod

    cache_dir = tmp_path / ".benchmark_cache"
    cache_dir.mkdir()
    stale = cache_dir / "_bulk_artificialanalysis.json"
    stale.write_text('{"timestamp": 0, "models": {}}')
    old = time.time() - 48 * 3600
    os.utime(stale, (old, old))
    monkeypatch.setattr(sj_mod, "_benchmark_cache_dir", lambda: cache_dir)

    def _boom():
        raise RuntimeError("network dead")

    monkeypatch.setattr(sj_mod, "_benchmark_refresh_impl", _boom)
    caplog.set_level(logging.WARNING, logger="app.scheduled_jobs")

    await jobs.tick_benchmark_refresh()
    assert any("benchmark refresh failed" in r.getMessage().lower()
               for r in caplog.records)
