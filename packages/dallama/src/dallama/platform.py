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


def read_stderr_tail(stderr_path: str, lines: int = 10) -> str:
    """Last *lines* lines of the stderr log at *stderr_path*.

    Returns ``""`` when the path is empty/missing/unreadable. Shared by every
    backend launcher (DaLLaMa's LlamaServer + clair_obscur's ImageServer) so
    crash-diagnostics tail logic lives in one place.
    """
    if not stderr_path:
        return ""
    try:
        with open(stderr_path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()
        return "\n".join(content.splitlines()[-lines:])
    except Exception:
        return ""


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

    def create_process(
        self,
        cmd: list[str],
        stderr_path: str,
        cwd: str | None = None,
    ) -> subprocess.Popen:
        """Launch *cmd* as a subprocess, redirecting stderr to *stderr_path*.

        On Windows the process is created with ``CREATE_NO_WINDOW`` and
        assigned to the Job Object so it dies when the parent dies. *cwd*,
        when given, is the directory the child launches FROM (Popen cwd) —
        needed by backends whose entrypoint resolves against cwd (e.g.
        ComfyUI's ``main.py``); ``None`` inherits the parent's cwd.

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
            cwd=cwd,
        )
        # Attach handle so callers (or graceful_stop) can close it
        proc._dallama_stderr = stderr_fh  # type: ignore[attr-defined]
        self._assign_to_job(proc)
        return proc

    async def graceful_stop(self, process: subprocess.Popen, timeout: float = 10) -> None:
        """Terminate *process*, falling back to force-kill if needed.

        Production triage 2026-05-01: 96% (2058/2133) of historical stop
        calls fell through to force-kill on Windows because llama-server
        does NOT install a CTRL_BREAK_EVENT handler. The 10s wait was
        pure dead time — the signal was never going to land. On Linux,
        SIGTERM is honored within ~1s.

        Windows: skip CTRL_BREAK_EVENT entirely. Try ``terminate()``
        (TerminateProcess on Windows — synchronous OS-level kill, much
        cleaner than SIGKILL via ``kill()``); poll briefly; ``kill()``
        if still alive.
        Non-Windows: SIGTERM with the original poll-loop.
        """
        if process.poll() is not None:
            self._close_stderr(process)
            return

        loop = asyncio.get_running_loop()

        try:
            if _IS_WINDOWS:
                # TerminateProcess: immediate OS-level kill. llama-server
                # cannot trap it, so no need to wait for graceful shutdown.
                # Poll briefly (1s) only to confirm the OS released the
                # PID before we close stderr.
                process.terminate()
                _win_timeout = min(2.0, timeout)
                elapsed = 0.0
                interval = 0.1
                while elapsed < _win_timeout:
                    if process.poll() is not None:
                        break
                    await asyncio.sleep(interval)
                    elapsed += interval
                if process.poll() is None:
                    # Extremely rare: TerminateProcess hung. Fall back.
                    logger.warning(
                        "Process %s didn't exit after terminate(), killing...",
                        process.pid,
                    )
                    process.kill()
                    try:
                        await asyncio.wait_for(
                            loop.run_in_executor(None, process.wait),
                            timeout=5.0,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Process %s did not exit within 5s after kill",
                            process.pid,
                        )
            else:
                process.terminate()
                elapsed = 0.0
                interval = 0.25
                while elapsed < timeout:
                    if process.poll() is not None:
                        break
                    await asyncio.sleep(interval)
                    elapsed += interval
                else:
                    logger.warning(
                        "Process %s didn't stop gracefully, killing...",
                        process.pid,
                    )
                    process.kill()
                    try:
                        await asyncio.wait_for(
                            loop.run_in_executor(None, process.wait),
                            timeout=5.0,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Process %s did not exit within 5s after kill",
                            process.pid,
                        )
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

    def kill_stray_servers(self, keep_port: int, executable_name: str = "llama-server") -> int:
        """Kill every llama-server EXCEPT the one (if any) listening on *keep_port*.

        Port-aware counterpart to :meth:`kill_orphans`. Preserves a healthy
        server already serving on the configured port (honoring the
        "never kill the good llama-server" rule) while clearing wrong-port
        orphans that occupy VRAM — the 2026-06-14 incident, where a stray on
        :8080 sat invisible to every port-specific check while the stack
        expected :8081.

        Returns the number of processes killed. Never raises.
        """
        try:
            pids = self._llama_server_pids(executable_name)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("kill_stray_servers: enumerate failed: %s", exc)
            return 0
        if not pids:
            return 0

        # A failed port probe must NOT be read as "nothing on keep_port" — that
        # would kill a healthy server on keep_port (never-kill rule). Abort.
        try:
            keeper: int | None = self._pid_on_port(keep_port)
        except Exception as exc:
            logger.warning(
                "kill_stray_servers: port probe failed (%s) — refusing to kill "
                "blindly", exc,
            )
            return 0

        killed = 0
        for pid in pids:
            if keeper is not None and pid == keeper:
                continue
            try:
                self._kill_pid(pid)
                killed += 1
                logger.warning(
                    "Killed stray llama-server pid=%s (not on port %d)", pid, keep_port
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("kill_stray_servers: kill pid=%s failed: %s", pid, exc)
        if keeper is not None:
            logger.info("Preserved llama-server pid=%s on port %d", keeper, keep_port)
        return killed

    # ── Internal helpers ────────────────────────────────────────────────────

    def _llama_server_pids(self, executable_name: str = "llama-server") -> set[int]:
        """Return PIDs of running llama-server processes."""
        if _IS_WINDOWS:
            exe = executable_name if executable_name.endswith(".exe") else f"{executable_name}.exe"
            out = subprocess.run(
                ["tasklist", "/FI", f"IMAGENAME eq {exe}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            pids: set[int] = set()
            for line in out.splitlines():
                line = line.strip()
                if not line.startswith('"'):
                    continue
                parts = [p.strip('"') for p in line.split('","')]
                if len(parts) >= 2 and parts[0].lower() == exe.lower():
                    try:
                        pids.add(int(parts[1]))
                    except ValueError:
                        pass
            return pids
        out = subprocess.run(
            ["pgrep", "-f", executable_name],
            capture_output=True, text=True, timeout=5,
        ).stdout
        return {int(x) for x in out.split() if x.strip().isdigit()}

    def _pid_on_port(self, port: int) -> int | None:
        """Return the PID listening on *port*, or None."""
        needle = f":{port}"
        if _IS_WINDOWS:
            out = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[3].upper() == "LISTENING" and parts[1].endswith(needle):
                    try:
                        return int(parts[4])
                    except ValueError:
                        pass
            return None
        out = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        for x in out.split():
            if x.strip().isdigit():
                return int(x)
        return None

    def _kill_pid(self, pid: int) -> None:
        """Force-kill a single PID."""
        if _IS_WINDOWS:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=10)
        else:
            subprocess.run(["kill", "-9", str(pid)], capture_output=True, timeout=10)

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
