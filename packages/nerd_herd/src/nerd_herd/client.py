"""NerdHerdClient — lightweight HTTP proxy with the same public API as NerdHerd.

Drop-in replacement for callers that talk to a remote NerdHerd sidecar.
All methods return safe defaults when the server is unreachable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import aiohttp

from nerd_herd.exposition import API_VERSION
from nerd_herd.types import (
    CloudProviderState,
    LocalModelState,
    RateLimits,
    SystemSnapshot,
)
from yazbunu import get_logger

logger = get_logger("nerd_herd.client")


@dataclass
class GPUStateProxy:
    """Minimal GPU state returned by the HTTP API."""
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    vram_used_mb: int = 0
    gpu_name: str = ""
    gpu_util_pct: float = 0.0


class NerdHerdClient:
    """HTTP client that mirrors the NerdHerd public API.

    Parameters
    ----------
    port:
        TCP port where the NerdHerd metrics/API server is listening.
    host:
        Hostname or IP. Defaults to loopback.
    timeout:
        Per-request timeout in seconds. Defaults to 3.0.
    """

    def __init__(
        self,
        port: int = 9881,
        host: str = "127.0.0.1",
        timeout: float = 3.0,
    ) -> None:
        self._base = f"http://{host}:{port}"
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None
        self._cached_snapshot = SystemSnapshot()
        # Local swap budget — the orchestrator process is the one
        # triggering swaps, so tracking in-process matches the real
        # cadence without an extra RPC hop.
        from nerd_herd.swap_budget import SwapBudget
        self._swap_budget = SwapBudget()

    # ------------------------------------------------------------------
    # Swap event stream (sync passthroughs — data only, policy in hoca)
    # ------------------------------------------------------------------
    def recent_swap_count(self) -> int:
        return self._swap_budget.recent_count()

    def record_swap(self, model_name: str = "") -> None:
        self._swap_budget.record_swap()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _get_session(self) -> aiohttp.ClientSession:
        """Lazily create and reuse a single aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Version handshake
    # ------------------------------------------------------------------

    async def check_version(self) -> bool:
        """Return True if sidecar API version matches this client.

        Returns False (stale) when the sidecar is unreachable or reports
        a different API_VERSION.  The caller should restart the sidecar
        in that case.
        """
        data = await self._get_json("/health", default=None)
        if not isinstance(data, dict):
            return False
        remote = data.get("api_version")
        if remote == API_VERSION:
            return True
        logger.warning(
            "Sidecar API version mismatch",
            remote=remote,
            expected=API_VERSION,
        )
        return False

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    async def _get_json(self, path: str, default: Any = None) -> Any:
        """GET *path* and return parsed JSON, or *default* on any error."""
        try:
            session = self._get_session()
            async with session.get(f"{self._base}{path}") as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.debug("HTTP GET non-200", path=path, status=resp.status)
                return default
        except Exception as exc:
            logger.debug("HTTP GET failed", path=path, error=str(exc))
            return default

    async def _post_json(self, path: str, data: dict, default: Any = None) -> Any:
        """POST JSON *data* to *path* and return parsed response, or *default*."""
        try:
            session = self._get_session()
            async with session.post(f"{self._base}{path}", json=data) as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.debug("HTTP POST non-200", path=path, status=resp.status)
                return default
        except Exception as exc:
            logger.debug("HTTP POST failed", path=path, error=str(exc))
            return default

    async def _get_state(self) -> dict:
        """Fetch /api/state, returning {} on failure."""
        result = await self._get_json("/api/state", default={})
        return result if isinstance(result, dict) else {}

    # ------------------------------------------------------------------
    # Public API (mirrors NerdHerd)
    # ------------------------------------------------------------------

    async def get_load_mode(self) -> str:
        """Return current load mode. Default: "full"."""
        state = await self._get_state()
        return state.get("load_mode", "full")

    async def set_load_mode(self, mode: str, source: str = "user") -> str:
        """Set load mode. Returns server result string, or empty string on failure."""
        resp = await self._post_json("/api/mode", {"mode": mode, "source": source}, default={})
        if isinstance(resp, dict):
            return resp.get("result", "")
        return ""

    async def enable_auto_management(self) -> None:
        """Enable auto GPU management on the sidecar."""
        await self._post_json("/api/auto", {})

    async def is_auto_managed(self) -> bool:
        """Return whether GPU mode is auto-managed. Default: True."""
        state = await self._get_state()
        return bool(state.get("auto_managed", True))

    async def is_local_inference_allowed(self) -> bool:
        """Return whether local inference is currently allowed. Default: True."""
        state = await self._get_state()
        return bool(state.get("local_inference_allowed", True))

    async def get_vram_budget_fraction(self) -> float:
        """Return VRAM budget as a fraction [0.0, 1.0]. Default: 1.0."""
        state = await self._get_state()
        try:
            return float(state.get("vram_budget_fraction", 1.0))
        except (TypeError, ValueError):
            return 1.0

    async def get_vram_budget_mb(self) -> int:
        """Return VRAM budget in MB. Default: 0."""
        state = await self._get_state()
        try:
            return int(state.get("vram_budget_mb", 0))
        except (TypeError, ValueError):
            return 0

    async def gpu_state(self) -> GPUStateProxy:
        """Return GPU state from the sidecar. Returns all-zero proxy on failure."""
        data = await self._get_json("/api/gpu", default={})
        if not isinstance(data, dict):
            return GPUStateProxy()
        return GPUStateProxy(
            vram_total_mb=int(data.get("vram_total_mb", 0)),
            vram_free_mb=int(data.get("vram_free_mb", 0)),
            vram_used_mb=int(data.get("vram_used_mb", 0)),
            gpu_name=str(data.get("gpu_name", "")),
            gpu_util_pct=float(data.get("gpu_util_pct", 0.0)),
        )

    async def push_local_state(
        self,
        model_name: str | None,
        thinking_enabled: bool = False,
        vision_enabled: bool = False,
        measured_tps: float = 0.0,
        context_length: int = 0,
        is_swapping: bool = False,
        kv_cache_ratio: float = 0.0,
        idle_seconds: float = 0.0,
        requests_processing: int = 0,
    ) -> None:
        """Push current local model state to the NerdHerd sidecar."""
        await self._post_json("/api/local_state", {
            "model_name": model_name,
            "thinking_enabled": thinking_enabled,
            "vision_enabled": vision_enabled,
            "measured_tps": measured_tps,
            "context_length": context_length,
            "is_swapping": is_swapping,
            "kv_cache_ratio": kv_cache_ratio,
            "idle_seconds": idle_seconds,
            "requests_processing": requests_processing,
        })

    # ------------------------------------------------------------------
    # Snapshot (sync cached + async refresh)
    # ------------------------------------------------------------------

    def snapshot(self) -> SystemSnapshot:
        """Return the last cached SystemSnapshot (sync, for Fatih Hoca).

        Call refresh_snapshot() periodically to keep this fresh.
        """
        return self._cached_snapshot

    async def refresh_snapshot(self) -> SystemSnapshot:
        """Fetch a fresh SystemSnapshot from the sidecar and cache it.

        Tries /api/snapshot first (new sidecar). Falls back to building
        a snapshot from /api/state + /api/gpu (old sidecar without the
        snapshot endpoint).
        """
        data = await self._get_json("/api/snapshot", default=None)
        if isinstance(data, dict):
            self._cached_snapshot = self._parse_snapshot(data)
            return self._cached_snapshot

        # Fallback: build snapshot from existing endpoints
        state = await self._get_state()
        gpu_data = await self._get_json("/api/gpu", default={})
        if not isinstance(gpu_data, dict):
            gpu_data = {}

        vram_mb = int(state.get("vram_budget_mb", 0))
        # If vram_budget_mb is 0 but GPU reports free VRAM, use that
        if vram_mb == 0:
            vram_mb = int(gpu_data.get("vram_free_mb", 0))

        self._cached_snapshot = SystemSnapshot(
            vram_available_mb=vram_mb,
            local=self._cached_snapshot.local,   # preserve last known local state
            cloud=self._cached_snapshot.cloud,    # preserve last known cloud state
        )
        return self._cached_snapshot

    def _parse_snapshot(self, data: dict) -> SystemSnapshot:
        """Parse a SystemSnapshot from the /api/snapshot JSON response."""
        local_data = data.get("local") or {}
        local = LocalModelState(
            model_name=local_data.get("model_name"),
            thinking_enabled=bool(local_data.get("thinking_enabled", False)),
            vision_enabled=bool(local_data.get("vision_enabled", False)),
            measured_tps=float(local_data.get("measured_tps", 0.0)),
            context_length=int(local_data.get("context_length", 0)),
            is_swapping=bool(local_data.get("is_swapping", False)),
            kv_cache_ratio=float(local_data.get("kv_cache_ratio", 0.0)),
            idle_seconds=float(local_data.get("idle_seconds", 0.0)),
        )

        cloud: dict[str, CloudProviderState] = {}
        for prov, prov_data in (data.get("cloud") or {}).items():
            cloud[prov] = CloudProviderState(
                provider=prov_data.get("provider", prov),
                utilization_pct=float(prov_data.get("utilization_pct", 0.0)),
                consecutive_failures=int(prov_data.get("consecutive_failures", 0)),
                last_failure_at=prov_data.get("last_failure_at"),
                limits=RateLimits(),
            )

        return SystemSnapshot(
            vram_available_mb=int(data.get("vram_available_mb", 0)),
            local=local,
            cloud=cloud,
        )

    async def mark_degraded(self, capability: str) -> None:
        """Mark a capability as degraded on the sidecar."""
        await self._post_json("/api/degraded", {"capability": capability})

    async def prometheus_lines(self) -> str:
        """Fetch raw Prometheus text from /metrics. Returns "" on failure."""
        try:
            session = self._get_session()
            async with session.get(f"{self._base}/metrics") as resp:
                if resp.status == 200:
                    return await resp.text()
                return ""
        except Exception as exc:
            logger.debug("prometheus_lines failed", error=str(exc))
            return ""


# Process-wide singleton. Owned by nerd_herd so it resolves to the same module
# regardless of how the entry-point script is launched (avoids __main__ vs
# src.app.run module-identity splits).
_default: NerdHerdClient | None = None


def get_default() -> NerdHerdClient | None:
    return _default


def set_default(client: NerdHerdClient | None) -> None:
    global _default
    _default = client
