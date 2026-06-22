"""GPU load mode manager. In-memory state, callback-driven persistence."""
from __future__ import annotations

import asyncio
import time
from typing import Callable

from prometheus_client import Gauge

from yazbunu import get_logger

from nerd_herd.types import GPUState

logger = get_logger("nerd_herd.load")

LOAD_MODES = ("full", "balanced", "minimal")
MODE_ORDER = ("minimal", "balanced", "full")

# Legacy modes (pre-2026-06-22 4-mode set) normalize to the closest survivor.
_LEGACY_ALIASES = {"heavy": "balanced", "shared": "balanced"}


def _normalize_mode(mode: str) -> str:
    """Map any input to a canonical mode. Legacy heavy/shared -> balanced;
    unknown -> full. Idempotent on canonical values."""
    if mode in LOAD_MODES:
        return mode
    return _LEGACY_ALIASES.get(mode, "full")


# Observability only (feeds Prometheus nerd_herd_vram_budget_fraction -> Grafana).
# NOT a VRAM cap — placement is owned by S13/S14 + --fit since 2026-06-09.
VRAM_BUDGETS: dict[str, float] = {
    "full": 1.0,
    "balanced": 0.5,
    "minimal": 0.0,
}

# User-facing mode descriptions. Load mode is placement, not a VRAM cap: it
# weights the desktop presence/contention signals (S13/S14) that bias work
# cloud↔local. VRAM_BUDGETS fractions above are advisory/observability only.
DESCRIPTIONS: dict[str, str] = {
    "full": "Yerel Serbest — masaüstü sinyallerini yoksay; yerele serbest gönder",
    "balanced": "Dengeli — sen aktifken güçlü bulut yönelimi (2×)",
    "minimal": "Sadece Bulut — yerel çıkarım kapalı, yalnızca bulut",
}

_g_mode = Gauge("nerd_herd_load_mode", "Current GPU load mode (0=minimal,1=balanced,2=full)")
_g_mode_info = Gauge("nerd_herd_load_mode_info", "GPU load mode as label", ["mode"])
_g_budget = Gauge("nerd_herd_vram_budget_fraction", "VRAM budget fraction 0.0-1.0")
_g_auto = Gauge("nerd_herd_auto_managed", "Whether GPU mode is auto-managed")

_ALL_GAUGES = [_g_mode, _g_mode_info, _g_budget, _g_auto]


def _mode_index(mode: str) -> int:
    try:
        return MODE_ORDER.index(mode)
    except ValueError:
        return len(MODE_ORDER) - 1   # treat unknown as least-restrictive (full)


class LoadManager:
    """VRAM budget policy with auto-detect loop. No DB — in-memory only."""

    name = "load"

    def __init__(
        self,
        gpu_collector,
        initial_mode: str = "full",
        detect_interval: int = 30,
        upgrade_delay: int = 300,
    ) -> None:
        self._gpu = gpu_collector
        self._mode = _normalize_mode(initial_mode)
        self._auto_managed = True
        self._detect_interval = detect_interval
        self._upgrade_delay = upgrade_delay
        self._callbacks: list[Callable[[str, str, str], None]] = []
        self._detect_task: asyncio.Task | None = None
        self._last_external_fraction: float = 0.0

    def get_load_mode(self) -> str:
        return self._mode

    def set_load_mode(self, mode: str, source: str = "user") -> str:
        if mode not in LOAD_MODES and mode not in _LEGACY_ALIASES:
            return f"Unknown mode '{mode}'. Choose: {', '.join(LOAD_MODES)}"
        mode = _normalize_mode(mode)
        prev = self._mode
        if prev == mode:
            return f"Already in *{mode}* mode"

        self._mode = mode
        if source == "user":
            self._auto_managed = False

        logger.info("load mode changed", prev=prev, new=mode, source=source)

        for cb in self._callbacks:
            try:
                cb(prev, mode, source)
            except Exception:
                pass

        return f"Load mode set to *{mode}*: {DESCRIPTIONS[mode]}"

    def enable_auto_management(self) -> None:
        self._auto_managed = True
        logger.info("auto-management enabled")

    def is_auto_managed(self) -> bool:
        return self._auto_managed

    def get_vram_budget_fraction(self) -> float:
        return VRAM_BUDGETS.get(self._mode, 1.0)

    def get_vram_budget_mb(self) -> int:
        # Placement, not capping (resource-signals 2026-06-09): the
        # desktop signals + --fit own contention now. snapshot.vram_available_mb
        # reflects the true free VRAM, never a mode-scaled fraction.
        gpu = self._gpu.gpu_state()
        return int(gpu.vram_free_mb)

    def _record_external(self, ext) -> None:
        """Cache the latest external-GPU fraction from the auto-detect loop."""
        try:
            self._last_external_fraction = float(ext.external_vram_fraction)
        except Exception:
            self._last_external_fraction = 0.0

    def get_external_gpu_fraction(self) -> float:
        """Last external-GPU fraction seen by the 30s auto-detect loop.

        Read by snapshot() / S14 — cheap, no pynvml probe on the hot path.
        """
        return self._last_external_fraction

    def is_local_inference_allowed(self) -> bool:
        return self._mode != "minimal"

    def on_mode_change(self, callback: Callable[[str, str, str], None]) -> None:
        self._callbacks.append(callback)

    @staticmethod
    def suggest_mode_for_external_usage(external_vram_fraction: float) -> str:
        # External-GPU-only fallback (no presence). 3-mode set.
        if external_vram_fraction < 0.10:
            return "full"
        elif external_vram_fraction < 0.60:
            return "balanced"
        else:
            return "minimal"

    async def start_auto_detect(self, notify_fn: Callable | None = None) -> None:
        self._detect_task = asyncio.create_task(self._auto_detect_loop(notify_fn))

    async def stop_auto_detect(self) -> None:
        if self._detect_task:
            self._detect_task.cancel()
            try:
                await self._detect_task
            except asyncio.CancelledError:
                pass

    async def _auto_detect_loop(self, notify_fn: Callable | None = None) -> None:
        upgrade_candidate: str | None = None
        upgrade_stable_since: float = 0.0

        logger.info("GPU auto-detect loop started",
                     interval=self._detect_interval, upgrade_delay=self._upgrade_delay)

        while True:
            try:
                await asyncio.sleep(self._detect_interval)

                if not self._auto_managed:
                    upgrade_candidate = None
                    continue

                ext = self._gpu.detect_external_gpu_usage()
                self._record_external(ext)
                suggested = self.suggest_mode_for_external_usage(ext.external_vram_fraction)
                current = self._mode
                now = time.time()

                current_idx = _mode_index(current)
                suggested_idx = _mode_index(suggested)

                if suggested_idx < current_idx:
                    self.set_load_mode(suggested, source="auto")
                    upgrade_candidate = None
                    upgrade_stable_since = 0.0
                    msg = (
                        f"GPU auto-detect: external usage at {ext.external_vram_fraction:.0%} "
                        f"({ext.external_vram_mb}MB, {ext.external_process_count} procs). "
                        f"Switched {current} -> {suggested}"
                    )
                    if notify_fn:
                        try:
                            await notify_fn(msg)
                        except Exception:
                            pass

                elif suggested_idx > current_idx:
                    if upgrade_candidate != suggested:
                        upgrade_candidate = suggested
                        upgrade_stable_since = now
                    elif now - upgrade_stable_since >= self._upgrade_delay:
                        self.set_load_mode(suggested, source="auto")
                        upgrade_candidate = None
                        upgrade_stable_since = 0.0
                        msg = (
                            f"GPU auto-detect: external usage dropped to {ext.external_vram_fraction:.0%}. "
                            f"Upgraded {current} -> {suggested}"
                        )
                        if notify_fn:
                            try:
                                await notify_fn(msg)
                            except Exception:
                                pass
                else:
                    upgrade_candidate = None
                    upgrade_stable_since = 0.0

            except asyncio.CancelledError:
                logger.info("GPU auto-detect loop cancelled")
                break
            except Exception as e:
                logger.debug("GPU auto-detect error", error=str(e))

    def collect(self) -> dict[str, float | int | str]:
        return {
            "load_mode": self._mode,
            "vram_budget_fraction": self.get_vram_budget_fraction(),
            "auto_managed": int(self._auto_managed),
            "local_inference_allowed": int(self.is_local_inference_allowed()),
        }

    def prometheus_metrics(self) -> list:
        mode_val = {"minimal": 0, "balanced": 1, "full": 2}.get(self._mode, 2)
        _g_mode.set(mode_val)
        for m in LOAD_MODES:
            _g_mode_info.labels(mode=m).set(1 if m == self._mode else 0)
        _g_budget.set(self.get_vram_budget_fraction())
        _g_auto.set(1 if self._auto_managed else 0)
        return list(_ALL_GAUGES)
