"""Tests for NerdHerdClient — HTTP proxy for NerdHerd."""
from __future__ import annotations

import pytest
import pytest_asyncio
from nerd_herd import NerdHerd
from nerd_herd.client import NerdHerdClient, GPUStateProxy

TEST_PORT = 19882
UNREACHABLE_PORT = 19899


@pytest_asyncio.fixture
async def server():
    """Start a real NerdHerd server on TEST_PORT, yield, then stop."""
    nh = NerdHerd(metrics_port=TEST_PORT, llama_server_url=None)
    await nh.start()
    yield nh
    await nh.stop()


@pytest_asyncio.fixture
async def client(server):
    """NerdHerdClient pointing at the live server."""
    c = NerdHerdClient(port=TEST_PORT)
    yield c
    await c.close()


@pytest_asyncio.fixture
async def dead_client():
    """NerdHerdClient pointing at an unused port — all calls should degrade safely."""
    c = NerdHerdClient(port=UNREACHABLE_PORT, timeout=0.5)
    yield c
    await c.close()


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_load_mode(client, server):
    mode = await client.get_load_mode()
    assert mode == "full"


@pytest.mark.asyncio
async def test_set_load_mode(client, server):
    result = await client.set_load_mode("shared", source="test")
    assert isinstance(result, str)
    # Server state should have changed
    mode = await client.get_load_mode()
    assert mode == "shared"
    # Restore
    await client.set_load_mode("full", source="test")


@pytest.mark.asyncio
async def test_enable_auto_management(client, server):
    # First set a manual mode to disable auto
    await client.set_load_mode("shared", source="user")
    # Re-enable auto
    await client.enable_auto_management()
    is_auto = await client.is_auto_managed()
    assert is_auto is True
    # Restore
    await client.set_load_mode("full", source="test")


@pytest.mark.asyncio
async def test_is_auto_managed(client, server):
    result = await client.is_auto_managed()
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_is_local_inference_allowed(client, server):
    result = await client.is_local_inference_allowed()
    assert result is True  # default "full" mode allows inference


@pytest.mark.asyncio
async def test_get_vram_budget_fraction(client, server):
    fraction = await client.get_vram_budget_fraction()
    assert isinstance(fraction, float)
    assert 0.0 <= fraction <= 1.0


@pytest.mark.asyncio
async def test_get_vram_budget_mb(client, server):
    mb = await client.get_vram_budget_mb()
    assert isinstance(mb, int)
    assert mb >= 0


@pytest.mark.asyncio
async def test_gpu_state(client, server):
    state = await client.gpu_state()
    assert isinstance(state, GPUStateProxy)
    assert isinstance(state.vram_total_mb, int)
    assert isinstance(state.vram_free_mb, int)
    assert isinstance(state.vram_used_mb, int)
    assert isinstance(state.gpu_name, str)
    assert isinstance(state.gpu_util_pct, (int, float))


@pytest.mark.asyncio
async def test_mark_degraded(client, server):
    # Should not raise
    await client.mark_degraded("test_capability")


@pytest.mark.asyncio
async def test_prometheus_lines(client, server):
    text = await client.prometheus_lines()
    assert isinstance(text, str)
    assert len(text) > 0


# ---------------------------------------------------------------------------
# Graceful degradation tests (unreachable server)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dead_get_load_mode(dead_client):
    mode = await dead_client.get_load_mode()
    assert mode == "full"


@pytest.mark.asyncio
async def test_dead_set_load_mode(dead_client):
    result = await dead_client.set_load_mode("minimal")
    assert isinstance(result, str)  # safe default, no exception


@pytest.mark.asyncio
async def test_dead_enable_auto_management(dead_client):
    # Should not raise
    await dead_client.enable_auto_management()


@pytest.mark.asyncio
async def test_dead_is_auto_managed(dead_client):
    result = await dead_client.is_auto_managed()
    assert result is True  # safe default


@pytest.mark.asyncio
async def test_dead_is_local_inference_allowed(dead_client):
    result = await dead_client.is_local_inference_allowed()
    assert result is True  # safe default


@pytest.mark.asyncio
async def test_dead_get_vram_budget_fraction(dead_client):
    result = await dead_client.get_vram_budget_fraction()
    assert result == 1.0  # safe default


@pytest.mark.asyncio
async def test_dead_get_vram_budget_mb(dead_client):
    result = await dead_client.get_vram_budget_mb()
    assert result == 0  # safe default


@pytest.mark.asyncio
async def test_dead_gpu_state(dead_client):
    state = await dead_client.gpu_state()
    assert isinstance(state, GPUStateProxy)
    assert state.vram_total_mb == 0
    assert state.gpu_name == ""


@pytest.mark.asyncio
async def test_dead_mark_degraded(dead_client):
    # Should not raise
    await dead_client.mark_degraded("some_cap")


@pytest.mark.asyncio
async def test_dead_prometheus_lines(dead_client):
    text = await dead_client.prometheus_lines()
    assert isinstance(text, str)  # empty string is acceptable
