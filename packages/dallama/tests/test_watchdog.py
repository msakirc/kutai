"""Tests for HealthWatchdog and IdleUnloader."""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch
import pytest
from dallama.config import DaLLaMaConfig, ServerConfig
from dallama.watchdog import HealthWatchdog, IdleUnloader

# -- HealthWatchdog --

@pytest.fixture
def watchdog_cfg():
    return DaLLaMaConfig(health_check_interval_seconds=0.1, health_fail_threshold=2)

@pytest.fixture
def mock_server():
    s = MagicMock()
    s.is_alive.return_value = True
    s.health_check = AsyncMock(return_value=True)
    s._health_check_status = AsyncMock(return_value=200)
    s.stop = AsyncMock()
    return s

@pytest.fixture
def mock_swap():
    s = MagicMock()
    s.swap_in_progress = False
    s.intentional_unload = False
    s.swap = AsyncMock(return_value=True)
    return s

@pytest.mark.asyncio
async def test_watchdog_detects_crash(watchdog_cfg, mock_server, mock_swap):
    current_config = ServerConfig(model_path="/m/test.gguf", model_name="test", context_length=4096)
    wd = HealthWatchdog(watchdog_cfg, mock_server, mock_swap)
    call_count = 0
    def is_alive_side_effect():
        nonlocal call_count
        call_count += 1
        return call_count <= 2
    mock_server.is_alive.side_effect = is_alive_side_effect
    task = asyncio.create_task(wd.run(lambda: current_config))
    await asyncio.sleep(0.5)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    mock_swap.swap.assert_called()

@pytest.mark.asyncio
async def test_watchdog_detects_hang(watchdog_cfg, mock_server, mock_swap):
    current_config = ServerConfig(model_path="/m/test.gguf", model_name="test", context_length=4096)
    wd = HealthWatchdog(watchdog_cfg, mock_server, mock_swap)
    mock_server.is_alive.return_value = True
    mock_server._health_check_status = AsyncMock(return_value=0)
    task = asyncio.create_task(wd.run(lambda: current_config))
    await asyncio.sleep(0.5)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert mock_server.stop.called or mock_swap.swap.called

@pytest.mark.asyncio
async def test_watchdog_skips_during_swap(watchdog_cfg, mock_server, mock_swap):
    current_config = ServerConfig(model_path="/m/test.gguf", model_name="test", context_length=4096)
    mock_swap.swap_in_progress = True
    mock_server._health_check_status = AsyncMock(return_value=0)
    wd = HealthWatchdog(watchdog_cfg, mock_server, mock_swap)
    task = asyncio.create_task(wd.run(lambda: current_config))
    await asyncio.sleep(0.3)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    mock_swap.swap.assert_not_called()

# -- IdleUnloader --

@pytest.mark.asyncio
async def test_idle_unloader_unloads_after_timeout():
    cfg = DaLLaMaConfig(idle_timeout_seconds=0.3)
    server = AsyncMock()
    server.is_alive.return_value = True
    swap = MagicMock()
    swap.swap_in_progress = False
    swap.has_inflight = False
    calls = []
    cfg.on_ready = lambda m, r: calls.append((m, r))
    unloader = IdleUnloader(cfg, server, swap)
    unloader.reset_timer()
    task = asyncio.create_task(unloader.run())
    await asyncio.sleep(0.6)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    server.stop.assert_called()
    assert any(r == "idle_unload" for _, r in calls)

@pytest.mark.asyncio
async def test_idle_unloader_reset_prevents_unload():
    cfg = DaLLaMaConfig(idle_timeout_seconds=0.5)
    server = AsyncMock()
    server.is_alive.return_value = True
    swap = MagicMock()
    swap.swap_in_progress = False
    swap.has_inflight = False
    unloader = IdleUnloader(cfg, server, swap)
    unloader.reset_timer()
    task = asyncio.create_task(unloader.run())
    await asyncio.sleep(0.3)
    unloader.reset_timer()
    await asyncio.sleep(0.3)
    unloader.reset_timer()
    await asyncio.sleep(0.3)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    server.stop.assert_not_called()

@pytest.mark.asyncio
async def test_watchdog_skips_during_intentional_unload(watchdog_cfg, mock_server, mock_swap):
    """HealthWatchdog must not trigger recovery when IdleUnloader is stopping the server."""
    current_config = ServerConfig(model_path="/m/test.gguf", model_name="test", context_length=4096)
    mock_swap.intentional_unload = True
    mock_server.is_alive.return_value = False  # server is dead (being unloaded)
    wd = HealthWatchdog(watchdog_cfg, mock_server, mock_swap)
    task = asyncio.create_task(wd.run(lambda: current_config))
    await asyncio.sleep(0.3)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    mock_swap.swap.assert_not_called()


@pytest.mark.asyncio
async def test_idle_unloader_sets_intentional_unload_flag():
    """IdleUnloader must set intentional_unload while stopping the server."""
    cfg = DaLLaMaConfig(idle_timeout_seconds=0.2)
    server = AsyncMock()
    server.is_alive.return_value = True
    swap = MagicMock()
    swap.swap_in_progress = False
    swap.has_inflight = False
    swap.intentional_unload = False

    flag_during_stop = []
    original_stop = server.stop

    async def capture_flag():
        flag_during_stop.append(swap.intentional_unload)
        return await original_stop()

    server.stop = capture_flag

    unloader = IdleUnloader(cfg, server, swap)
    unloader.reset_timer()
    task = asyncio.create_task(unloader.run())
    await asyncio.sleep(0.5)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert len(flag_during_stop) >= 1, "stop() must have been called at least once"
    assert all(flag_during_stop), "intentional_unload must be True while stop() runs"
    assert swap.intentional_unload is False, "intentional_unload must be cleared after stop()"


@pytest.mark.asyncio
async def test_idle_unloader_skips_when_inflight():
    cfg = DaLLaMaConfig(idle_timeout_seconds=0.2)
    server = AsyncMock()
    server.is_alive.return_value = True
    swap = MagicMock()
    swap.swap_in_progress = False
    swap.has_inflight = True
    unloader = IdleUnloader(cfg, server, swap)
    unloader.reset_timer()
    task = asyncio.create_task(unloader.run())
    await asyncio.sleep(0.5)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    server.stop.assert_not_called()
