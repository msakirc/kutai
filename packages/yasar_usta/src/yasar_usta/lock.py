"""Cross-platform single-instance lock with stale recovery."""

from __future__ import annotations

import atexit
import os
import sys
from pathlib import Path

_lock_handle = None
_lock_file: Path | None = None
_sentinel_file: Path | None = None
_PID_WIDTH = 10


def is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
        except Exception:
            pass
        return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except Exception:
            return False


def acquire_lock(lock_dir: str | Path, name: str = "guard") -> None:
    """Acquire an exclusive lock. Exits if another instance is running.

    Creates two files in lock_dir:
      {name}.lock  — PID (plain text, never locked, always readable)
      {name}.lk    — sentinel (OS-locked, content irrelevant)

    Args:
        lock_dir: Directory for lock files.
        name: Base name for lock files.

    Raises:
        SystemExit: If another instance holds the lock.
    """
    global _lock_handle, _lock_file, _sentinel_file

    lock_dir = Path(lock_dir)
    lock_dir.mkdir(parents=True, exist_ok=True)
    _lock_file = lock_dir / f"{name}.lock"
    _sentinel_file = lock_dir / f"{name}.lk"

    try:
        import msvcrt
    except ImportError:
        msvcrt = None

    if msvcrt is not None:
        _acquire_lock_msvcrt(msvcrt)
    else:
        _acquire_lock_unix()

    atexit.register(release_lock)


def _acquire_lock_msvcrt(msvcrt) -> None:
    """Windows lock using a separate sentinel file."""
    global _lock_handle

    def _try_lock(fh):
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except (OSError, IOError):
            return False

    def _write_pid():
        with open(_lock_file, "w") as f:
            f.write(str(os.getpid()).zfill(_PID_WIDTH))

    def _read_pid():
        try:
            raw = _lock_file.read_text().strip()
            return int(raw) if raw else None
        except (ValueError, OSError):
            return None

    if not _sentinel_file.exists():
        _sentinel_file.write_text("L")

    _lock_handle = open(_sentinel_file, "r+")

    if _try_lock(_lock_handle):
        _write_pid()
        return

    existing_pid = _read_pid()

    if existing_pid is not None and is_pid_alive(existing_pid):
        _lock_handle.close()
        _lock_handle = None
        print(f"ERROR: Already running (PID {existing_pid}).")
        sys.exit(1)

    print(f"Stale lock detected (PID {existing_pid or '?'} is dead). Cleaning up.")
    _lock_handle.close()
    _lock_handle = None
    try:
        _sentinel_file.unlink(missing_ok=True)
    except Exception:
        pass

    _sentinel_file.write_text("L")
    _lock_handle = open(_sentinel_file, "r+")
    if _try_lock(_lock_handle):
        _write_pid()
        return

    _lock_handle.close()
    _lock_handle = None
    print("ERROR: Could not acquire lock even after stale-lock cleanup.")
    sys.exit(1)


def _acquire_lock_unix() -> None:
    """Unix lock using fcntl, with PID-based fallback."""
    global _lock_handle
    try:
        import fcntl
        _lock_handle = open(_lock_file, "w")
        fcntl.flock(_lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_handle.write(str(os.getpid()).zfill(_PID_WIDTH))
        _lock_handle.flush()
    except (OSError, IOError):
        print("ERROR: Already running.")
        sys.exit(1)
    except ImportError:
        if _lock_file.exists():
            try:
                old_pid = int(_lock_file.read_text().strip())
                if is_pid_alive(old_pid):
                    print(f"ERROR: Already running (PID {old_pid}).")
                    sys.exit(1)
                else:
                    print(f"Stale lock (PID {old_pid} is dead). Cleaning up.")
            except Exception:
                pass
        _lock_file.write_text(str(os.getpid()).zfill(_PID_WIDTH))


def release_lock() -> None:
    """Release lock and remove lock files."""
    global _lock_handle
    if _lock_handle:
        try:
            import msvcrt
            _lock_handle.seek(0)
            msvcrt.locking(_lock_handle.fileno(), msvcrt.LK_UNLOCK, _PID_WIDTH)
        except Exception:
            pass
        try:
            _lock_handle.close()
        except Exception:
            pass
        _lock_handle = None
    if _lock_file:
        try:
            _lock_file.unlink(missing_ok=True)
        except Exception:
            pass
