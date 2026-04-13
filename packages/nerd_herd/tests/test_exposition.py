# tests/test_exposition.py
"""Tests for MetricsServer including the HTTP API endpoints."""
from __future__ import annotations

import pytest
import aiohttp
from nerd_herd.exposition import build_metrics_text
from nerd_herd.registry import CollectorRegistry
from nerd_herd import NerdHerd


# ---------------------------------------------------------------------------
# Unit tests (no server)
# ---------------------------------------------------------------------------

def test_build_metrics_text_empty():
    reg = CollectorRegistry()
    text = build_metrics_text(reg)
    assert isinstance(text, str)


def test_build_metrics_text_with_collector():
    from prometheus_client import Gauge

    class StubCollector:
        name = "stub"
        def collect(self):
            return {"val": 42}
        def prometheus_metrics(self):
            return [_g]

    # Use a unique gauge name to avoid conflicts with other tests
    _g = Gauge("stub_exposition_test_val", "test value for exposition")
    _g.set(42)

    reg = CollectorRegistry()
    reg.register("stub", StubCollector())
    text = build_metrics_text(reg)
    assert isinstance(text, str)
    assert "stub_exposition_test_val" in text


# ---------------------------------------------------------------------------
# Integration tests — real NerdHerd on port 19882 (avoid conflict with 19881)
# ---------------------------------------------------------------------------

TEST_PORT = 19882
BASE_URL = f"http://127.0.0.1:{TEST_PORT}"


@pytest.fixture
async def running_nh():
    nh = NerdHerd(metrics_port=TEST_PORT, llama_server_url=None)
    await nh.start()
    yield nh
    await nh.stop()


@pytest.mark.asyncio
async def test_get_state(running_nh):
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/api/state") as resp:
            assert resp.status == 200
            data = await resp.json()
    assert "load_mode" in data
    assert "vram_budget_fraction" in data
    assert "vram_budget_mb" in data
    assert "local_inference_allowed" in data
    assert "auto_managed" in data
    assert "degraded" in data
    assert isinstance(data["degraded"], list)


@pytest.mark.asyncio
async def test_get_state_degraded_list(running_nh):
    running_nh.mark_degraded("inference")
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/api/state") as resp:
            data = await resp.json()
    assert "inference" in data["degraded"]


@pytest.mark.asyncio
async def test_post_mode_valid(running_nh):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BASE_URL}/api/mode",
            json={"mode": "shared", "source": "user"},
        ) as resp:
            assert resp.status == 200
            data = await resp.json()
    assert data["mode"] == "shared"
    assert "result" in data


@pytest.mark.asyncio
async def test_post_mode_invalid(running_nh):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BASE_URL}/api/mode",
            json={"mode": "nonexistent"},
        ) as resp:
            assert resp.status == 200
            data = await resp.json()
    # set_load_mode returns an error message string for unknown modes
    assert "result" in data


@pytest.mark.asyncio
async def test_post_mode_missing_field(running_nh):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BASE_URL}/api/mode",
            json={},
        ) as resp:
            assert resp.status == 400
            data = await resp.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_post_auto(running_nh):
    # First disable auto-management by setting mode via user
    running_nh.set_load_mode("shared", source="user")
    assert not running_nh._load.is_auto_managed()

    async with aiohttp.ClientSession() as session:
        async with session.post(f"{BASE_URL}/api/auto") as resp:
            assert resp.status == 200
            data = await resp.json()
    assert data["auto_managed"] is True
    assert running_nh._load.is_auto_managed()


@pytest.mark.asyncio
async def test_get_gpu(running_nh):
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/api/gpu") as resp:
            assert resp.status == 200
            data = await resp.json()
    assert "vram_total_mb" in data
    assert "vram_free_mb" in data
    assert "vram_used_mb" in data
    assert "gpu_util_pct" in data


@pytest.mark.asyncio
async def test_post_degraded_valid(running_nh):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BASE_URL}/api/degraded",
            json={"capability": "test_cap"},
        ) as resp:
            assert resp.status == 200
            data = await resp.json()
    assert data["capability"] == "test_cap"
    assert data["degraded"] is True
    assert not running_nh.is_healthy("test_cap")


@pytest.mark.asyncio
async def test_post_degraded_missing_field(running_nh):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{BASE_URL}/api/degraded",
            json={},
        ) as resp:
            assert resp.status == 400
            data = await resp.json()
    assert "error" in data


@pytest.mark.asyncio
async def test_existing_endpoints_still_work(running_nh):
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/health") as resp:
            assert resp.status == 200
            data = await resp.json()
        assert data["status"] == "ok"

        async with session.get(f"{BASE_URL}/metrics") as resp:
            assert resp.status == 200
            text = await resp.text()
        assert "nerd_herd" in text
