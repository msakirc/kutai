"""Shared dataclasses for Nerd Herd collectors."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GPUState:
    available: bool
    vram_total_mb: int = 0
    vram_used_mb: int = 0
    vram_free_mb: int = 0
    gpu_utilization_pct: int = 0
    temperature_c: int = 0
    power_draw_w: float = 0.0
    timestamp: float = 0.0

    @property
    def vram_usage_pct(self) -> float:
        if self.vram_total_mb == 0:
            return 0.0
        return (self.vram_used_mb / self.vram_total_mb) * 100

    @property
    def is_throttling(self) -> bool:
        return self.temperature_c > 85

    @property
    def is_busy(self) -> bool:
        return self.gpu_utilization_pct > 80


@dataclass
class SystemState:
    ram_total_mb: int = 0
    ram_available_mb: int = 0
    cpu_percent: float = 0.0
    gpu: GPUState = field(default_factory=lambda: GPUState(available=False))

    @property
    def can_load_model(self) -> bool:
        return self.ram_available_mb > 4096


@dataclass
class ExternalGPUUsage:
    detected: bool = False
    external_vram_mb: int = 0
    external_process_count: int = 0
    our_vram_mb: int = 0
    total_vram_mb: int = 0

    @property
    def external_vram_fraction(self) -> float:
        if self.total_vram_mb == 0:
            return 0.0
        return self.external_vram_mb / self.total_vram_mb


@dataclass
class HealthStatus:
    boot_time: str
    capabilities: dict[str, bool] = field(default_factory=dict)

    @property
    def degraded(self) -> list[str]:
        return [k for k, v in self.capabilities.items() if not v]
