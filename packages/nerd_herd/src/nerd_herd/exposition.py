"""Prometheus exposition — HTTP server serving /metrics."""
from __future__ import annotations

import asyncio

from aiohttp import web
from prometheus_client import generate_latest, REGISTRY

from yazbunu import get_logger

from nerd_herd.registry import CollectorRegistry

logger = get_logger("nerd_herd.exposition")


def build_metrics_text(registry: CollectorRegistry) -> str:
    """Trigger collection on all collectors and return Prometheus text."""
    registry.all_prometheus_metrics()
    return generate_latest(REGISTRY).decode("utf-8")


class MetricsServer:
    """Lightweight aiohttp server serving /metrics for Grafana."""

    def __init__(self, registry: CollectorRegistry, port: int = 9881) -> None:
        self._registry = registry
        self._port = port
        self._runner: web.AppRunner | None = None

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/metrics", self._handle_metrics)
        app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self._port, reuse_address=True)
        await site.start()
        logger.info("Metrics server started", port=self._port)

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            logger.info("Metrics server stopped")

    async def _handle_metrics(self, request: web.Request) -> web.Response:
        text = build_metrics_text(self._registry)
        return web.Response(
            text=text,
            content_type="text/plain; version=0.0.4",
            charset="utf-8",
        )

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})
