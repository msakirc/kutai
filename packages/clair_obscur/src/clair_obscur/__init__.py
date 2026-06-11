"""clair_obscur — local image-server wrapper. Parallel to dallama (LLM-side).

Lifecycle: start() / stop() / status() / available() / base_url() /
record_release_hint(). Holds a PID-lock at logs/image_server.lock and
reconciles its own orphan (ComfyUI / A1111 process — NEVER llama-server)."""
from __future__ import annotations

from .config import ClairObscurConfig, load_config
from .server import ImageServer

__all__ = [
    "ClairObscurConfig", "load_config", "ImageServer",
    "start", "stop", "status", "available", "base_url",
    "record_release_hint", "get_singleton",
]

_singleton: ImageServer | None = None


def get_singleton() -> ImageServer:
    global _singleton
    if _singleton is None:
        _singleton = ImageServer(load_config())
    return _singleton


async def start() -> str:
    return await get_singleton().start()


async def stop() -> None:
    await get_singleton().stop()


def status() -> dict:
    return get_singleton().status()


def available() -> bool:
    return get_singleton().available()


def base_url() -> str:
    return get_singleton().base_url()


def record_release_hint() -> None:
    """Beckman tells clair_obscur it MAY release (lane switch). The idle
    backstop in ImageServer.start() then times the actual stop() after
    config.idle_release_seconds. Direct .stop() is for forced/emergency
    shutdown only — normal lane switches go through this function."""
    get_singleton().record_release_hint()
