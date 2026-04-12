"""DaLLaMa — main class composing all modules."""
from __future__ import annotations
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from .config import DaLLaMaConfig, DaLLaMaLoadError, InferenceSession, ServerConfig, ServerStatus
from .metrics import MetricsParser
from .platform import PlatformHelper
from .server import ServerProcess
from .swap import SwapManager
from .watchdog import HealthWatchdog, IdleUnloader

logger = logging.getLogger(__name__)


class DaLLaMa:
    def __init__(self, config: DaLLaMaConfig):
        self._config = config
        self._platform = PlatformHelper()
        self._server = ServerProcess(config, self._platform)
        self._swap = SwapManager(config)
        self._metrics = MetricsParser()
        self._idle_unloader = IdleUnloader(config, self._server, self._swap)
        self._watchdog = HealthWatchdog(config, self._server, self._swap)
        self._current_config: ServerConfig | None = None
        self._last_tps: float = 0.0
        self._watchdog_task: asyncio.Task | None = None
        self._idle_task: asyncio.Task | None = None

    async def start(self):
        self._platform.kill_orphans()
        self._watchdog_task = asyncio.create_task(self._watchdog.run(lambda: self._current_config))
        self._idle_task = asyncio.create_task(self._idle_unloader.run())
        logger.info("DaLLaMa started (port %d)", self._config.port)

    async def stop(self):
        for task in (self._watchdog_task, self._idle_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._watchdog_task = None
        self._idle_task = None
        await self._server.stop()
        self._current_config = None
        logger.info("DaLLaMa stopped")

    @asynccontextmanager
    async def infer(self, config: ServerConfig) -> AsyncIterator[InferenceSession]:
        needs_swap = (
            self._current_config is None
            or config.model_name != self._current_config.model_name
            or config.thinking != self._current_config.thinking
            or config.vision_projector != self._current_config.vision_projector
        )
        if needs_swap:
            success = await self._swap.swap(self._server, config)
            if not success:
                raise DaLLaMaLoadError(config.model_name)
            self._current_config = config
        gen = self._swap.mark_inference_start()
        try:
            yield InferenceSession(
                url=f"http://{self._config.host}:{self._config.port}",
                model_name=config.model_name,
            )
        finally:
            self._swap.mark_inference_end(gen)
            self._idle_unloader.reset_timer()
            asyncio.ensure_future(self._refresh_tps())

    def keep_alive(self):
        self._idle_unloader.reset_timer()

    @property
    def status(self) -> ServerStatus:
        return ServerStatus(
            model_name=self._current_config.model_name if self._current_config else None,
            healthy=self._server.is_alive(),
            busy=self._swap.has_inflight,
            measured_tps=self._last_tps,
            context_length=self._current_config.context_length if self._current_config else 0,
        )

    async def _refresh_tps(self):
        try:
            snap = await self._metrics.fetch(self._server.api_base)
            if snap.generation_tokens_per_second > 0:
                self._last_tps = snap.generation_tokens_per_second
        except Exception:
            pass
