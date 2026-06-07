"""ImageServer — async lifecycle wrapper around ComfyUI/A1111.

PID-locked at logs/image_server.lock. Boot orphan-reconcile validates the
stale PID's cmdline (via psutil) matches the configured backend launcher
BEFORE killing — never touches llama-server or any unrelated tenant.

On Windows the child is assigned to a Job Object with KILL_ON_JOB_CLOSE so
it dies if KutAI dies (mirrors packages/dallama PlatformHelper). The boot
orphan-reconcile remains the secondary net for ungraceful crashes.

Backstop discipline (v2): the watcher fires when record_release_hint() has
been called AND idle_release_seconds have elapsed since the hint. Direct
stop() is for forced/emergency shutdown only — normal lane switches drive
release through record_release_hint() so a back-to-back image batch can
reuse the warm server without restart."""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time

import httpx

from .config import ClairObscurConfig

try:
    from src.infra.logging_config import get_logger

    logger = get_logger("clair_obscur.server")
except Exception:  # pragma: no cover - standalone fallback
    import logging

    logger = logging.getLogger("clair_obscur.server")

_IS_WINDOWS = sys.platform == "win32"


def _log_dir() -> str:
    """Resolve the log directory, anchored to CLAIR_OBSCUR_LOG_DIR.

    Default is the bare relative ``logs`` so existing tests/tools that expect
    ``logs/image_server.lock`` keep working.
    """
    return os.environ.get("CLAIR_OBSCUR_LOG_DIR", "logs")


# Lock path is resolved at import time against the default log dir so module
# constants (imported by tests) stay stable; runtime paths use _log_dir().
_LOCK_PATH = os.path.join(_log_dir(), "image_server.lock")


def _create_job_object():
    """Return a Windows Job Object handle (KILL_ON_JOB_CLOSE) or None.

    Reuses dallama's helper when importable, otherwise None (no Job Object →
    boot orphan-reconcile is the only net). Non-Windows always returns None.
    """
    if not _IS_WINDOWS:
        return None
    try:
        from dallama.platform import _create_job_object as _dallama_job

        return _dallama_job()
    except Exception as exc:
        logger.debug("Job Object setup unavailable: %s", exc)
        return None


def _assign_to_job(job_handle, pid: int) -> None:
    """Assign *pid* to *job_handle* (no-op when handle/pid missing)."""
    if job_handle is None or not _IS_WINDOWS:
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        h_process = kernel32.OpenProcess(0x1F0FFF, False, pid)  # PROCESS_ALL_ACCESS
        if h_process:
            ok = kernel32.AssignProcessToJobObject(job_handle, h_process)
            kernel32.CloseHandle(h_process)
            if ok:
                logger.debug("PID %s assigned to Job Object", pid)
            else:
                logger.warning("Failed to assign PID %s to Job Object", pid)
    except Exception as exc:
        logger.debug("Job assignment failed: %s", exc)


class ImageServer:
    def __init__(self, config: ClairObscurConfig):
        self._config = config
        self._pid: int | None = None
        self._resident: bool = False
        self._release_hint_at: float | None = None
        self._idle_task: asyncio.Task | None = None
        self._health_timeout_seconds: float = 120.0
        self._health_poll_interval: float = 1.0
        self._stderr_path: str = ""
        self._stderr_fh = None
        self._job_object = _create_job_object()

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
        logger.info("Launching %s backend", self._config.backend)
        await self._launch_process()
        logger.info("%s backend launched (pid=%s)", self._config.backend, self._pid)
        self._acquire_lock()
        deadline = time.time() + self._health_timeout_seconds
        while time.time() < deadline:
            if await self._health_probe():
                self._resident = True
                self._release_hint_at = None
                logger.info(
                    "%s backend healthy at %s (pid=%s)",
                    self._config.backend, self._config.base_url, self._pid,
                )
                self._notify_nerd_herd_resident(
                    vram_mb=self._estimated_resident_vram_mb()
                )
                self._arm_idle_backstop()
                return self._config.base_url
            await asyncio.sleep(self._health_poll_interval)
        # Health never came up → tear down + surface TimeoutError.
        tail = self._read_stderr_tail(30)
        logger.error(
            "%s backend (pid=%s) did not become healthy within %.0fs. "
            "Last stderr:\n%s",
            self._config.backend, self._pid, self._health_timeout_seconds, tail,
        )
        if self._pid is not None:
            try:
                self._kill_own_pid(self._pid)
            finally:
                self._pid = None
        self._close_stderr()
        self._release_lock()
        raise TimeoutError(
            f"clair_obscur {self._config.backend} did not become healthy "
            f"within {self._health_timeout_seconds}s"
        )

    async def stop(self) -> None:
        """Forced/emergency stop. Normal lane switches go through
        record_release_hint() so the backstop handles release after idle."""
        task = self._idle_task
        self._idle_task = None
        if task is not None and not task.done():
            # Guard against awaiting ourselves: the watcher calls stop() and
            # `return`s immediately after, so cancelling+awaiting the current
            # task would deadlock. Skip the await in that case.
            task.cancel()
            if task is not asyncio.current_task():
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        pid = self._pid
        self._pid = None
        self._resident = False
        self._release_hint_at = None
        if pid is not None:
            logger.info("Stopping %s backend (pid=%s)", self._config.backend, pid)
            self._kill_own_pid(pid)
        self._close_stderr()
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
        <port>`. PID captured for our lock.

        On Windows the process is created with CREATE_NO_WINDOW and assigned to
        the Job Object so it dies when KutAI dies. stderr is redirected to
        logs/image_server.stderr.log for crash diagnostics."""
        cmd = self._build_launch_cmd()
        creationflags = 0
        if _IS_WINDOWS:
            # CREATE_NO_WINDOW (matches dallama) — no console pop-up; the Job
            # Object handles parent-death cleanup, not the process group.
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        log_dir = _log_dir()
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            pass
        self._stderr_path = os.path.join(log_dir, "image_server.stderr.log")
        try:
            self._stderr_fh = open(self._stderr_path, "w", encoding="utf-8")
            stderr_target = self._stderr_fh
        except Exception as exc:
            logger.warning("Could not open stderr log %s: %s", self._stderr_path, exc)
            self._stderr_fh = None
            stderr_target = subprocess.DEVNULL

        proc = subprocess.Popen(  # noqa: S603 — caller-supplied exe
            cmd, creationflags=creationflags,
            stdout=subprocess.DEVNULL, stderr=stderr_target,
        )
        self._pid = proc.pid
        _assign_to_job(self._job_object, proc.pid)

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

    def _read_stderr_tail(self, lines: int = 20) -> str:
        """Read the last *lines* lines of the stderr log. Empty on any failure."""
        if not self._stderr_path:
            return ""
        try:
            with open(self._stderr_path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
            return "\n".join(content.splitlines()[-lines:])
        except Exception:
            return ""

    def _close_stderr(self) -> None:
        fh = self._stderr_fh
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
            self._stderr_fh = None

    def _acquire_lock(self) -> None:
        if self._pid is None:
            return
        try:
            os.makedirs(os.path.dirname(_LOCK_PATH) or ".", exist_ok=True)
            with open(_LOCK_PATH, "w", encoding="utf-8") as f:
                f.write(f"{self._pid}\n{self._config.backend}\n")
        except Exception as exc:
            logger.warning("Failed to write lock %s: %s", _LOCK_PATH, exc)

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
            logger.warning(
                "Orphan-reconcile: killing stale %s backend (pid=%s)",
                self._config.backend, stale_pid,
            )
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
            # Direct construction (no pid_exists pre-check) closes the TOCTOU
            # window: psutil raises if the PID vanished between the two calls.
            p = psutil.Process(pid)
            cmdline = " ".join(p.cmdline()).lower()
        except Exception:
            # NoSuchProcess / AccessDenied / ZombieProcess / import failure →
            # treat as "not our backend" (fail-safe: never kill on doubt).
            return False
        if self._config.backend == "comfyui":
            return "main.py" in cmdline or "comfyui" in cmdline
        return "webui" in cmdline or "launch.py" in cmdline

    def _kill_own_pid(self, pid: int) -> None:
        """Kill ONLY the given PID. NEVER taskkill-by-name. Synchronous —
        called from sync (_reconcile_orphan) and async paths; uses psutil's
        own short wait (non-event-loop) so it never blocks for more than ~2s.
        Backend match is re-verified before kill (inviolable safety)."""
        try:
            import psutil
            # Direct construction; psutil raises if the PID is already gone.
            try:
                proc = psutil.Process(pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                return
            proc.terminate()
            try:
                # gone, alive = wait_procs is non-blocking-friendly; cap at 2s
                # so the sync caller never stalls the boot path noticeably.
                psutil.wait_procs([proc], timeout=2)
                if proc.is_running():
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        except Exception as exc:
            logger.error("Failed to kill backend pid %s: %s", pid, exc)

    def _estimated_resident_vram_mb(self) -> int:
        """SDXL-Turbo fp16 + activations @ 1024×1024 ≈ 4.5 GB."""
        return 4500

    def _notify_nerd_herd_resident(self, vram_mb: int) -> None:
        try:
            import nerd_herd
            nerd_herd.record_image_server_state(
                resident=(vram_mb > 0), vram_mb=vram_mb,
            )
        except Exception as exc:
            logger.warning("Failed to notify nerd_herd of image-server state: %s", exc)

    def _arm_idle_backstop(self) -> None:
        """Schedule the safety/normal release. Watcher fires the actual
        stop() when (now - release_hint_at) >= idle_release_seconds AND
        resident is still True. Cancelled on stop() / restart."""
        if self._idle_task is not None and not self._idle_task.done():
            return

        async def _watcher():
            cfg = self._config
            # Poll ~4x within the idle window so the backstop honors short
            # windows too (capped at 5s, tiny floor to avoid a busy loop).
            # The old `max(1.0, ...)` floor could not fire any idle window
            # shorter than ~4s and added up to 1s of latency at the default.
            tick = min(5.0, max(0.01, cfg.idle_release_seconds / 4))
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
