"""Tests for SwapManager — drain, circuit breaker, swap orchestration."""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock
import pytest
from dallama.config import DaLLaMaConfig, ServerConfig
from dallama.swap import SwapManager

@pytest.fixture
def cfg():
    return DaLLaMaConfig(circuit_breaker_threshold=2, circuit_breaker_cooldown_seconds=1.0,
                         inference_drain_timeout_seconds=2.0)

@pytest.fixture
def swap(cfg):
    return SwapManager(cfg)

@pytest.fixture
def server():
    s = AsyncMock()
    s.is_alive.return_value = True
    s.health_check = AsyncMock(return_value=True)
    s.stop = AsyncMock()
    s.start = AsyncMock(return_value=True)
    s.process = MagicMock()
    return s

@pytest.fixture
def server_config():
    return ServerConfig(model_path="/m/test.gguf", model_name="test-model", context_length=4096)

# -- Inference tracking --
def test_inflight_initially_false(swap):
    assert swap.has_inflight is False

def test_mark_inference_start_end(swap):
    gen = swap.mark_inference_start()
    assert swap.has_inflight is True
    swap.mark_inference_end(gen)
    assert swap.has_inflight is False

def test_mark_inference_end_wrong_generation(swap):
    gen = swap.mark_inference_start()
    swap.force_reset_inflight()
    swap.mark_inference_end(gen)
    assert swap.has_inflight is False

def test_multiple_inflight(swap):
    g1 = swap.mark_inference_start()
    g2 = swap.mark_inference_start()
    assert swap.has_inflight is True
    swap.mark_inference_end(g1)
    assert swap.has_inflight is True
    swap.mark_inference_end(g2)
    assert swap.has_inflight is False

# -- Circuit breaker --
@pytest.mark.asyncio
async def test_circuit_breaker_blocks_after_threshold(swap, server, server_config):
    server.start = AsyncMock(return_value=False)
    await swap.swap(server, server_config)  # fail 1
    await swap.swap(server, server_config)  # fail 2
    result3 = await swap.swap(server, server_config)  # blocked
    assert result3 is False
    assert server.start.call_count == 2  # third was blocked, not called

@pytest.mark.asyncio
async def test_circuit_breaker_resets_on_success(swap, server, server_config):
    server.start = AsyncMock(return_value=False)
    await swap.swap(server, server_config)  # fail 1
    server.start = AsyncMock(return_value=True)
    result = await swap.swap(server, server_config)
    assert result is True
    assert swap._fail_count == 0

@pytest.mark.asyncio
async def test_circuit_breaker_cooldown_expires(swap, server, server_config):
    server.start = AsyncMock(return_value=False)
    await swap.swap(server, server_config)  # fail 1
    await swap.swap(server, server_config)  # fail 2 -> breaker trips
    await asyncio.sleep(1.1)  # wait for 1s cooldown
    server.start = AsyncMock(return_value=True)
    result = await swap.swap(server, server_config)
    assert result is True

# -- Drain --
@pytest.mark.asyncio
async def test_swap_drains_inflight(swap, server, server_config):
    gen = swap.mark_inference_start()
    async def finish_inference():
        await asyncio.sleep(0.5)
        swap.mark_inference_end(gen)
    asyncio.create_task(finish_inference())
    result = await swap.swap(server, server_config)
    assert result is True
    assert swap.has_inflight is False

@pytest.mark.asyncio
async def test_swap_force_drains_on_timeout(swap, server, server_config):
    swap._config = DaLLaMaConfig(inference_drain_timeout_seconds=0.5)
    swap.mark_inference_start()  # never ended
    result = await swap.swap(server, server_config)
    assert result is True
    assert swap.has_inflight is False

# -- VRAM check --
@pytest.mark.asyncio
async def test_swap_refuses_insufficient_vram(server, server_config):
    cfg = DaLLaMaConfig(min_free_vram_mb=4096, get_vram_free_mb=lambda: 2000)
    swap = SwapManager(cfg)
    result = await swap.swap(server, server_config)
    assert result is False
    server.start.assert_not_called()

@pytest.mark.asyncio
async def test_swap_proceeds_without_vram_callback(swap, server, server_config):
    result = await swap.swap(server, server_config)
    assert result is True

# -- on_ready callback --
@pytest.mark.asyncio
async def test_swap_calls_on_ready(server, server_config):
    calls = []
    cfg = DaLLaMaConfig(on_ready=lambda m, r: calls.append((m, r)))
    swap = SwapManager(cfg)
    await swap.swap(server, server_config)
    assert len(calls) == 1
    assert calls[0] == ("test-model", "model_loaded")

@pytest.mark.asyncio
async def test_swap_calls_on_ready_failure(server, server_config):
    calls = []
    cfg = DaLLaMaConfig(on_ready=lambda m, r: calls.append((m, r)))
    swap = SwapManager(cfg)
    server.start = AsyncMock(return_value=False)
    await swap.swap(server, server_config)
    assert calls[0] == (None, "load_failed")
