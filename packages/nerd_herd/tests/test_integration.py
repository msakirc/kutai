# tests/test_integration.py
"""Integration test — NerdHerd starts, serves /metrics, stops cleanly."""
import asyncio
import pytest
import aiohttp
from nerd_herd import NerdHerd


@pytest.mark.asyncio
async def test_full_lifecycle():
    nh = NerdHerd(
        metrics_port=19881,  # non-standard port to avoid conflicts
        llama_server_url=None,  # no llama-server in test
    )

    nh.mark_degraded("test_service")
    nh.mark_healthy("other_service")

    await nh.start()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://127.0.0.1:19881/metrics") as resp:
                assert resp.status == 200
                text = await resp.text()
                assert "nerd_herd_gpu_vram" in text or "nerd_herd_load_mode" in text
                assert "nerd_herd_capability_healthy" in text

            async with session.get("http://127.0.0.1:19881/health") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "ok"
    finally:
        await nh.stop()


@pytest.mark.asyncio
async def test_prometheus_lines_without_server():
    """prometheus_lines() works without starting the HTTP server."""
    nh = NerdHerd(llama_server_url=None)
    nh.mark_healthy("telegram")
    text = nh.prometheus_lines()
    assert "nerd_herd_capability_healthy" in text
