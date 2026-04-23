"""Prometheus exposition — HTTP server serving /metrics."""
from __future__ import annotations

import asyncio
import dataclasses
from typing import TYPE_CHECKING

from aiohttp import web
from prometheus_client import generate_latest, REGISTRY

from yazbunu import get_logger

from nerd_herd.registry import CollectorRegistry

if TYPE_CHECKING:
    from nerd_herd.nerd_herd import NerdHerd

# Bump this whenever routes or response schemas change.
# The client checks this on connect and triggers a sidecar restart on mismatch.
API_VERSION = 2

logger = get_logger("nerd_herd.exposition")


def build_metrics_text(registry: CollectorRegistry) -> str:
    """Trigger collection on all collectors and return Prometheus text."""
    registry.all_prometheus_metrics()
    return generate_latest(REGISTRY).decode("utf-8")


class MetricsServer:
    """Lightweight aiohttp server serving /metrics for Grafana."""

    def __init__(
        self,
        registry: CollectorRegistry,
        port: int = 9881,
        nerd_herd: "NerdHerd | None" = None,
    ) -> None:
        self._registry = registry
        self._port = port
        self._nh = nerd_herd
        self._runner: web.AppRunner | None = None

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/metrics", self._handle_metrics)
        app.router.add_get("/health", self._handle_health)

        if self._nh is not None:
            app.router.add_get("/api/state", self._handle_state)
            app.router.add_post("/api/mode", self._handle_set_mode)
            app.router.add_post("/api/auto", self._handle_enable_auto)
            app.router.add_get("/api/gpu", self._handle_gpu)
            app.router.add_post("/api/degraded", self._handle_degraded)
            app.router.add_post("/api/local_state", self._handle_local_state)
            app.router.add_post("/api/in_flight", self._handle_in_flight)
            app.router.add_get("/api/snapshot", self._handle_snapshot)

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
        return web.json_response({"status": "ok", "api_version": API_VERSION})

    async def _handle_state(self, request: web.Request) -> web.Response:
        nh = self._nh
        return web.json_response({
            "load_mode": nh.get_load_mode(),
            "vram_budget_fraction": nh.get_vram_budget_fraction(),
            "vram_budget_mb": nh.get_vram_budget_mb(),
            "local_inference_allowed": nh.is_local_inference_allowed(),
            "auto_managed": nh._load.is_auto_managed(),
            "degraded": [k for k, v in nh._health._capabilities.items() if not v],
        })

    async def _handle_set_mode(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON body"}, status=400)

        mode = body.get("mode")
        source = body.get("source", "user")
        if not mode:
            return web.json_response({"error": "missing 'mode' field"}, status=400)

        result = self._nh.set_load_mode(mode, source=source)
        return web.json_response({
            "result": result,
            "mode": self._nh.get_load_mode(),
        })

    async def _handle_enable_auto(self, request: web.Request) -> web.Response:
        self._nh.enable_auto_management()
        return web.json_response({"auto_managed": True})

    async def _handle_gpu(self, request: web.Request) -> web.Response:
        state = self._nh.gpu_state()
        data = dataclasses.asdict(state)
        # Rename gpu_utilization_pct to gpu_util_pct for the API contract
        data["gpu_util_pct"] = data.pop("gpu_utilization_pct", 0)
        # Add gpu_name (not in dataclass, default to empty string)
        data.setdefault("gpu_name", "")
        return web.json_response(data)

    async def _handle_degraded(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON body"}, status=400)

        capability = body.get("capability")
        if not capability:
            return web.json_response({"error": "missing 'capability' field"}, status=400)

        self._nh.mark_degraded(capability)
        return web.json_response({"capability": capability, "degraded": True})

    async def _handle_local_state(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON body"}, status=400)

        from nerd_herd.types import LocalModelState
        state = LocalModelState(
            model_name=body.get("model_name"),
            thinking_enabled=bool(body.get("thinking_enabled", False)),
            vision_enabled=bool(body.get("vision_enabled", False)),
            measured_tps=float(body.get("measured_tps", 0.0)),
            context_length=int(body.get("context_length", 0)),
            is_swapping=bool(body.get("is_swapping", False)),
            kv_cache_ratio=float(body.get("kv_cache_ratio", 0.0)),
            idle_seconds=float(body.get("idle_seconds", 0.0)),
            requests_processing=int(body.get("requests_processing", 0)),
        )
        self._nh.push_local_state(state)
        return web.json_response({"ok": True})

    async def _handle_in_flight(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON body"}, status=400)

        calls_data = body.get("calls") or []
        if not isinstance(calls_data, list):
            return web.json_response({"error": "'calls' must be a list"}, status=400)

        from nerd_herd.types import InFlightCall
        calls = []
        for c in calls_data:
            if not isinstance(c, dict):
                continue
            try:
                calls.append(InFlightCall(
                    call_id=str(c.get("call_id", "")),
                    task_id=c.get("task_id"),
                    category=str(c.get("category", "")),
                    model=str(c.get("model", "")),
                    provider=str(c.get("provider", "")),
                    is_local=bool(c.get("is_local", False)),
                    started_at=float(c.get("started_at", 0.0)),
                ))
            except Exception:
                continue
        self._nh.push_in_flight(calls)
        return web.json_response({"ok": True, "count": len(calls)})

    async def _handle_snapshot(self, request: web.Request) -> web.Response:
        snap = self._nh.snapshot()
        return web.json_response(dataclasses.asdict(snap))
