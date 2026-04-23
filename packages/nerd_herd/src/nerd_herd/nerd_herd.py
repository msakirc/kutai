"""NerdHerd — main facade for the observability package."""
from __future__ import annotations

from typing import Callable

from nerd_herd.registry import CollectorRegistry, Collector
from nerd_herd.gpu import GPUCollector
from nerd_herd.load import LoadManager
from nerd_herd.health import HealthRegistry
from nerd_herd.inference import InferenceCollector
from nerd_herd.exposition import MetricsServer, build_metrics_text
from nerd_herd.swap_budget import SwapBudget
from nerd_herd.types import GPUState, HealthStatus, InFlightCall, LocalModelState, CloudProviderState, QueueProfile, SystemSnapshot


class NerdHerd:
    """Main entry point. Creates registry, registers built-in collectors."""

    def __init__(
        self,
        metrics_port: int = 9881,
        llama_server_url: str | None = None,
        detect_interval: int = 30,
        upgrade_delay: int = 300,
        initial_load_mode: str = "full",
        inference_poll_interval: int = 5,
    ) -> None:
        self.registry = CollectorRegistry()

        self._gpu = GPUCollector()
        self.registry.register("gpu", self._gpu)

        self._load = LoadManager(
            gpu_collector=self._gpu,
            initial_mode=initial_load_mode,
            detect_interval=detect_interval,
            upgrade_delay=upgrade_delay,
        )
        self.registry.register("load", self._load)

        self._health = HealthRegistry()
        self.registry.register("health", self._health)

        self._inference: InferenceCollector | None = None
        if llama_server_url:
            self._inference = InferenceCollector(
                llama_server_url=llama_server_url,
                poll_interval=inference_poll_interval,
            )
            self.registry.register("inference", self._inference)

        self._local_state: LocalModelState = LocalModelState()
        self._cloud_state: dict[str, CloudProviderState] = {}
        self._queue_profile: QueueProfile | None = None
        self._in_flight_calls: list[InFlightCall] = []

        self._swap_budget = SwapBudget(window_seconds=300)

        self._server = MetricsServer(self.registry, port=metrics_port, nerd_herd=self)

    async def start(self) -> None:
        await self._server.start()
        if self._inference:
            await self._inference.start()

    async def start_auto_detect(self, notify_fn: Callable | None = None) -> None:
        await self._load.start_auto_detect(notify_fn)

    async def stop(self) -> None:
        await self._load.stop_auto_detect()
        if self._inference:
            await self._inference.stop()
        await self._server.stop()

    def gpu_state(self) -> GPUState:
        return self._gpu.gpu_state()

    def get_vram_budget_mb(self) -> int:
        return self._load.get_vram_budget_mb()

    def get_vram_budget_fraction(self) -> float:
        return self._load.get_vram_budget_fraction()

    def get_load_mode(self) -> str:
        return self._load.get_load_mode()

    def set_load_mode(self, mode: str, source: str = "user") -> str:
        return self._load.set_load_mode(mode, source)

    def enable_auto_management(self) -> None:
        self._load.enable_auto_management()

    def is_local_inference_allowed(self) -> bool:
        return self._load.is_local_inference_allowed()

    def on_mode_change(self, callback: Callable[[str, str, str], None]) -> None:
        self._load.on_mode_change(callback)

    def mark_degraded(self, capability: str) -> None:
        self._health.mark_degraded(capability)

    def mark_healthy(self, capability: str) -> None:
        self._health.mark_healthy(capability)

    def is_healthy(self, capability: str) -> bool:
        return self._health.is_healthy(capability)

    def get_health_status(self) -> HealthStatus:
        return self._health.get_status()

    def recent_swap_count(self) -> int:
        return self._swap_budget.recent_count()

    def record_swap(self, model_name: str = "") -> None:
        self._swap_budget.record_swap()

    def register_collector(self, name: str, collector: Collector) -> None:
        self.registry.register(name, collector)

    def push_local_state(self, state: LocalModelState) -> None:
        """Replace the current local model state (called by DaLLaMa on each swap)."""
        self._local_state = state

    def push_cloud_state(self, state: CloudProviderState) -> None:
        """Upsert a cloud provider state entry (called by KDV on each API response)."""
        self._cloud_state[state.provider] = state

    def push_queue_profile(self, profile: QueueProfile) -> None:
        """Store latest queue profile (pushed by Beckman on queue-change events)."""
        self._queue_profile = profile

    def push_in_flight(self, calls: list[InFlightCall]) -> None:
        """Replace in-flight call list (pushed by dispatcher on begin/end).

        Full-list replacement is intentional: dispatcher is sole producer,
        atomic swap keeps readers consistent.
        """
        self._in_flight_calls = list(calls)

    def snapshot(self) -> SystemSnapshot:
        """Return a point-in-time snapshot of all system state.

        Overlays live inference metrics (requests_processing, idle_seconds,
        kv_cache_ratio) onto the pushed _local_state so callers always see
        the freshest values without DaLLaMa having to push every tick.
        """
        gpu = self._gpu.gpu_state()
        local = self._local_state
        if self._inference is not None and local.model_name:
            live = self._inference.collect()
            # Shallow copy with live overlay — don't mutate _local_state in place.
            from dataclasses import replace
            local = replace(
                local,
                requests_processing=int(live.get("requests_processing", local.requests_processing)),
                idle_seconds=float(live.get("idle_seconds", local.idle_seconds)),
                kv_cache_ratio=float(live.get("kv_cache_ratio", local.kv_cache_ratio)),
            )
        return SystemSnapshot(
            vram_available_mb=self.get_vram_budget_mb() if gpu.available else 0,
            local=local,
            cloud=dict(self._cloud_state),
            queue_profile=self._queue_profile,
            in_flight_calls=list(self._in_flight_calls),
        )

    def prometheus_lines(self) -> str:
        return build_metrics_text(self.registry)
