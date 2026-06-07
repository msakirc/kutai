"""ImageServer — async lifecycle wrapper around ComfyUI/A1111.

PID-locked at logs/image_server.lock. Boot orphan-reconcile validates the
stale PID's cmdline (via psutil) matches the configured backend launcher
BEFORE killing — never touches llama-server or any unrelated tenant.

Backstop discipline (v2): the watcher fires when record_release_hint() has
been called AND idle_release_seconds have elapsed since the hint. Direct
stop() is for forced/emergency shutdown only — normal lane switches drive
release through record_release_hint() so a back-to-back image batch can
reuse the warm server without restart."""
from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Optional

import httpx

from .config import ClairObscurConfig

_LOCK_PATH = os.path.join("logs", "image_server.lock")
_IS_WINDOWS = sys.platform == "win32"


class ImageServer:
    def __init__(self, config: ClairObscurConfig):
        self._config = config
        self._pid: Optional[int] = None
        self._resident: bool = False
        self._release_hint_at: Optional[float] = None
        self._idle_task: Optional[asyncio.Task] = None
        self._health_timeout_seconds: float = 120.0
        self._health_poll_interval: float = 1.0

    # ───────── Public API ─────────
    def available(self) -> bool:
        return bool(self._config.exe_path) and os.path.exists(self._config.exe_path)

    def base_url(self) -> str:
        return self._config.base_url

    def status(self) -> dict:
        return {
            "resident": self._resident, "pid": self._pid,
            "backend": self._config.backend, "base_url": self._config.base_url,
            "release_hint_at": self._release_hint_at,
        }

    async def start(self) -> str:
        if self._resident and self._pid is not None:
            # Idempotent + clears any pending release hint so the backstop
            # window resets for the new image task.
            self._release_hint_at = None
            return self._config.base_url
        self._reconcile_orphan()
        await self._launch_process()
        self._acquire_lock()
        deadline = time.time() + self._health_timeout_seconds
        while time.time() < deadline:
            if await self._health_probe():
                self._resident = True
                self._release_hint_at = None
                self._notify_nerd_herd_resident(
                    vram_mb=self._estimated_resident_vram_mb()
                )
                self._arm_idle_backstop()
                return self._config.base_url
            await asyncio.sleep(self._health_poll_interval)
        # Health never came up → tear down + surface TimeoutError.
        if self._pid is not None:
            try:
                self._kill_own_pid(self._pid)
            finally:
                self._pid = None
        self._release_lock()
        raise TimeoutError(
            f"clair_obscur {self._config.backend} did not become healthy "
            f"within {self._health_timeout_seconds}s"
        )

    async def stop(self) -> None:
        """Forced/emergency stop. Normal lane switches go through
        record_release_hint() so the backstop handles release after idle."""
        if self._idle_task is not None:
            self._idle_task.cancel()
            self._idle_task = None
        pid = self._pid
        self._pid = None
        self._resident = False
        self._release_hint_at = None
        if pid is not None:
            self._kill_own_pid(pid)
        self._release_lock()
        self._notify_nerd_herd_resident(vram_mb=0)

    def record_release_hint(self) -> None:
        """Beckman tells us we may release. The watcher fires the actual
        stop() after idle_release_seconds — gives a back-to-back image task
        a window to reuse the warm server without restart."""
        self._release_hint_at = time.time()

    # ───────── Internals (test-mockable) ─────────
    async def _launch_process(self) -> None:
        """Spawn the backend. ComfyUI: `python main.py --listen <host> --port
        <port>`. A1111: `webui-user.bat` / `launch.py --api --listen --port
        <port>`. PID captured for our lock."""
        import subprocess
        cmd = self._build_launch_cmd()
        creationflags = 0
        if _IS_WINDOWS:
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        proc = subprocess.Popen(  # noqa: S603 — caller-supplied exe
            cmd, creationflags=creationflags,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._pid = proc.pid

    def _build_launch_cmd(self) -> list[str]:
        cfg = self._config
        if cfg.backend == "comfyui":
            return [cfg.exe_path, "-u", "main.py",
                    "--listen", cfg.host, "--port", str(cfg.port)]
        # a1111
        return [cfg.exe_path, "--api", "--listen",
                "--port", str(cfg.port), "--nowebui"]

    async def _health_probe(self) -> bool:
        url = self._health_url()
        try:
            async with httpx.AsyncClient(timeout=2.0) as c:
                resp = await c.get(url)
                return 200 <= resp.status_code < 500
        except Exception:
            return False

    def _health_url(self) -> str:
        if self._config.backend == "comfyui":
            return f"{self._config.base_url}/system_stats"
        return f"{self._config.base_url}/sdapi/v1/sd-models"

    def _acquire_lock(self) -> None:
        if self._pid is None:
            return
        try:
            os.makedirs(os.path.dirname(_LOCK_PATH), exist_ok=True)
            with open(_LOCK_PATH, "w", encoding="utf-8") as f:
                f.write(f"{self._pid}\n{self._config.backend}\n")
        except Exception:
            pass

    def _release_lock(self) -> None:
        try:
            if os.path.exists(_LOCK_PATH):
                os.remove(_LOCK_PATH)
        except Exception:
            pass

    def _reconcile_orphan(self) -> None:
        """Boot orphan-reconcile. Kills ONLY the PID written into our lock if
        AND ONLY IF its cmdline matches our configured backend launcher.
        Mirrors packages/nerd_herd/src/nerd_herd/platform.py:1-21 safety
        discipline. Never touches a PID we didn't write."""
        if not os.path.exists(_LOCK_PATH):
            return
        try:
            with open(_LOCK_PATH, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
            stale_pid = int(lines[0]) if lines else 0
            stale_backend = lines[1] if len(lines) > 1 else ""
        except Exception:
            try:
                os.remove(_LOCK_PATH)
            except Exception:
                pass
            return

        if stale_pid <= 0 or stale_backend != self._config.backend:
            try:
                os.remove(_LOCK_PATH)
            except Exception:
                pass
            return

        if self._is_own_backend_pid(stale_pid):
            self._kill_own_pid(stale_pid)
        try:
            os.remove(_LOCK_PATH)
        except Exception:
            pass

    def _is_own_backend_pid(self, pid: int) -> bool:
        """True iff pid is alive AND cmdline matches our backend launcher."""
        if pid <= 0:
            return False
        try:
            import psutil
            if not psutil.pid_exists(pid):
                return False
            p = psutil.Process(pid)
            cmdline = " ".join(p.cmdline()).lower()
        except Exception:
            return False
        if self._config.backend == "comfyui":
            return "main.py" in cmdline or "comfyui" in cmdline
        return "webui" in cmdline or "launch.py" in cmdline

    def _kill_own_pid(self, pid: int) -> None:
        """Kill ONLY the given PID. NEVER taskkill-by-name."""
        try:
            import psutil
            if psutil.pid_exists(pid):
                proc = psutil.Process(pid)
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
        except Exception:
            pass

    def _estimated_resident_vram_mb(self) -> int:
        """SDXL-Turbo fp16 + activations @ 1024×1024 ≈ 4.5 GB."""
        return 4500

    def _notify_nerd_herd_resident(self, vram_mb: int) -> None:
        try:
            import nerd_herd
            nerd_herd.record_image_server_state(
                resident=(vram_mb > 0), vram_mb=vram_mb,
            )
        except Exception:
            pass

    def _arm_idle_backstop(self) -> None:
        """Schedule the safety/normal release. Watcher fires the actual
        stop() when (now - release_hint_at) >= idle_release_seconds AND
        resident is still True. Cancelled on stop() / restart."""
        if self._idle_task is not None and not self._idle_task.done():
            return

        async def _watcher():
            cfg = self._config
            tick = min(5.0, max(1.0, cfg.idle_release_seconds / 4))
            while self._resident:
                await asyncio.sleep(tick)
                if not self._resident:
                    return
                hint = self._release_hint_at
                if hint is not None and (time.time() - hint) >= cfg.idle_release_seconds:
                    await self.stop()
                    return
        try:
            loop = asyncio.get_running_loop()
            self._idle_task = loop.create_task(_watcher())
        except RuntimeError:
            self._idle_task = None
