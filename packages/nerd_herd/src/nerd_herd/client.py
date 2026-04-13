"""NerdHerdClient — lightweight HTTP proxy with the same public API as NerdHerd.

Drop-in replacement for callers that talk to a remote NerdHerd sidecar.
All methods return safe defaults when the server is unreachable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import aiohttp

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
