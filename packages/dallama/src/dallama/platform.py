"""OS-specific process management for DaLLaMa.

Abstracts Windows Job Objects, process creation flags, orphan cleanup,
and graceful shutdown so the rest of DaLLaMa never needs platform checks.
"""
from __future__ import annotations

import asyncio
import logging
import platform
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

_IS_WINDOWS = platform.system() == "Windows"


def _create_job_object() -> Optional[int]:
    """Create a Windows Job Object with KILL_ON_JOB_CLOSE.

    When the parent process dies (even ungracefully), Windows closes all
    handles — including this job — which auto-kills every child assigned to it.
    Returns the handle on success, None on non-Windows or failure.
    """
    if not _IS_WINDOWS:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32

        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            logger.debug("Failed to create Job Object")
            return None

        # ── JOBOBJECT structs defined inline ───────────────────────────────
        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JobObjectExtendedLimitInformation = 9

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

        ok = kernel32.SetInformationJobObject(
            handle,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            logger.debug("Failed to set KILL_ON_JOB_CLOSE on Job Object")
            kernel32.CloseHandle(handle)
            return None

        logger.info("Windows Job Object created (KILL_ON_JOB_CLOSE)")
        return handle
    except Exception as exc:
        logger.debug(f"Job Object setup failed: {exc}")
        return None


class PlatformHelper:
    """OS-specific helpers for managing llama-server child processes."""

    def __init__(self) -> None:
        self._job_object: Optional[int] = _create_job_object()

    # ── Public API ──────────────────────────────────────────────────────────

    def create_process(self, cmd: list[str], stderr_path: str) -> subprocess.Popen:
        """Launch *cmd* as a subprocess, redirecting stderr to *stderr_path*.

        On Windows the process is created with ``CREATE_NO_WINDOW`` and
        assigned to the Job Object so it dies when the parent dies.

        The open stderr file handle is stored on the returned ``Popen``
        object as ``_dallama_stderr`` so callers can close it later.
        """
        creation_flags = subprocess.CREATE_NO_WINDOW if _IS_WINDOWS else 0
        stderr_fh = open(stderr_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=stderr_fh,
            creationflags=creation_flags,
        )
        # Attach handle so callers (or graceful_stop) can close it
        proc._dallama_stderr = stderr_fh  # type: ignore[attr-defined]
        self._assign_to_job(proc)
        return proc

    async def graceful_stop(self, process: subprocess.Popen, timeout: float = 10) -> None:
        """Terminate *process* gracefully, force-killing if it ignores the signal.

        Strategy:
        1. Send terminate signal (CTRL_BREAK_EVENT on Windows, SIGTERM elsewhere).
        2. Poll every 0.25 s up to *timeout* seconds.
        3. If still alive, ``kill()`` and wait via ``run_in_executor`` (never
           blocks the event loop).
        4. Close the ``_dallama_stderr`` handle if present.
        """
        if process.poll() is not None:
            self._close_stderr(process)
            return

        loop = asyncio.get_running_loop()

        try:
            if _IS_WINDOWS:
                # CTRL_BREAK_EVENT requires the child to have its own process
                # group (CREATE_NEW_PROCESS_GROUP — see create_process). With
                # the flag, llama-server gets a chance to flush before dying;
                # without it, the signal is silently dropped and we fall
                # through to force-kill every time.
                import signal as _signal
                process.send_signal(_signal.CTRL_BREAK_EVENT)
            else:
                process.terminate()

            # Poll until dead or timeout
            elapsed = 0.0
            interval = 0.25
            while elapsed < timeout:
                if process.poll() is not None:
                    break
                await asyncio.sleep(interval)
                elapsed += interval
            else:
                # Still alive — force kill
                logger.warning("Process %s didn't stop gracefully, killing...", process.pid)
                process.kill()
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(None, process.wait),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning("Process %s did not exit within 5s after kill", process.pid)
        except Exception as exc:
            logger.warning("Error stopping process %s: %s", process.pid, exc)
            try:
                process.kill()
            except Exception:
                pass

        self._close_stderr(process)

    def kill_orphans(self, executable_name: str = "llama-server") -> None:
        """Kill leftover processes matching *executable_name*.

        Windows: ``taskkill /F /IM <name>.exe``
        Linux/macOS: ``pkill -f <name>``

        Does not raise — failures are logged at DEBUG level.
        """
        try:
            if _IS_WINDOWS:
                # Ensure .exe suffix
                exe = executable_name if executable_name.endswith(".exe") else f"{executable_name}.exe"
                result = subprocess.run(
                    ["taskkill", "/F", "/IM", exe],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    logger.warning("Killed orphaned %s process(es): %s", exe, result.stdout.strip())
                else:
                    logger.debug("taskkill(%s) returned %d: %s", exe, result.returncode, result.stderr.strip())
            else:
                result = subprocess.run(
                    ["pkill", "-f", executable_name],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    logger.warning("Killed orphaned %s process(es)", executable_name)
                else:
                    logger.debug("pkill(%s) returned %d", executable_name, result.returncode)
        except Exception as exc:
            logger.debug("kill_orphans(%s) failed: %s", executable_name, exc)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _assign_to_job(self, process: subprocess.Popen) -> None:
        """Assign *process* to the Windows Job Object (no-op on non-Windows)."""
        if self._job_object is None:
            return
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            h_process = kernel32.OpenProcess(0x1F0FFF, False, process.pid)  # PROCESS_ALL_ACCESS
            if h_process:
                ok = kernel32.AssignProcessToJobObject(self._job_object, h_process)
                kernel32.CloseHandle(h_process)
                if ok:
                    logger.debug("PID %s assigned to Job Object", process.pid)
                else:
                    logger.warning("Failed to assign PID %s to Job Object", process.pid)
        except Exception as exc:
            logger.debug("Job assignment failed: %s", exc)

    @staticmethod
    def _close_stderr(process: subprocess.Popen) -> None:
        """Close the ``_dallama_stderr`` handle attached by ``create_process``."""
        fh = getattr(process, "_dallama_stderr", None)
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
            process._dallama_stderr = None  # type: ignore[attr-defined]
