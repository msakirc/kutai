"""GPU load mode manager. In-memory state, callback-driven persistence."""
from __future__ import annotations

import asyncio
import time
from typing import Callable

from prometheus_client import Gauge

from yazbunu import get_logger

from nerd_herd.types import GPUState

logger = get_logger("nerd_herd.load")

LOAD_MODES = ("full", "heavy", "shared", "minimal")
MODE_ORDER = ("minimal", "shared", "heavy", "full")

VRAM_BUDGETS: dict[str, float] = {
    "full": 1.0,
    "heavy": 0.9,
    "shared": 0.5,
    "minimal": 0.0,
}

DESCRIPTIONS: dict[str, str] = {
    "full": "Full GPU — all local capacity available",
    "heavy": "Heavy GPU — 90% VRAM cap, slight headroom for OS/desktop",
    "shared": "Shared GPU — 50% VRAM cap, prefer cloud for heavy tasks",
    "minimal": "Minimal GPU — local inference disabled, cloud only",
}

_g_mode = Gauge("nerd_herd_load_mode", "Current GPU load mode (0=minimal,1=shared,2=heavy,3=full)")
_g_mode_info = Gauge("nerd_herd_load_mode_info", "GPU load mode as label", ["mode"])
_g_budget = Gauge("nerd_herd_vram_budget_fraction", "VRAM budget fraction 0.0-1.0")
_g_auto = Gauge("nerd_herd_auto_managed", "Whether GPU mode is auto-managed")

_ALL_GAUGES = [_g_mode, _g_mode_info, _g_budget, _g_auto]


def _mode_index(mode: str) -> int:
    try:
        return MODE_ORDER.index(mode)
    except ValueError:
        return 3


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
        self._mode = initial_mode if initial_mode in LOAD_MODES else "full"
        self._auto_managed = True
        self._detect_interval = detect_interval
        self._upgrade_delay = upgrade_delay
        self._callbacks: list[Callable[[str, str, str], None]] = []
        self._detect_task: asyncio.Task | None = None

    def get_load_mode(self) -> str:
        return self._mode

    def set_load_mode(self, mode: str, source: str = "user") -> str:
        if mode not in LOAD_MODES:
            return f"Unknown mode '{mode}'. Choose: {', '.join(LOAD_MODES)}"

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
        gpu = self._gpu.gpu_state()
        return int(gpu.vram_free_mb * self.get_vram_budget_fraction())

    def is_local_inference_allowed(self) -> bool:
        return self._mode != "minimal"

    def on_mode_change(self, callback: Callable[[str, str, str], None]) -> None:
        self._callbacks.append(callback)

    @staticmethod
    def suggest_mode_for_external_usage(external_vram_fraction: float) -> str:
        if external_vram_fraction < 0.10:
            return "full"
        elif external_vram_fraction < 0.30:
            return "heavy"
        elif external_vram_fraction < 0.60:
            return "shared"
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
        mode_val = {"minimal": 0, "shared": 1, "heavy": 2, "full": 3}.get(self._mode, 3)
        _g_mode.set(mode_val)
        for m in LOAD_MODES:
            _g_mode_info.labels(mode=m).set(1 if m == self._mode else 0)
        _g_budget.set(self.get_vram_budget_fraction())
        _g_auto.set(1 if self._auto_managed else 0)
        return list(_ALL_GAUGES)
