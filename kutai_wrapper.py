#!/usr/bin/env python3
"""
KutAI Wrapper — always-running process manager for the orchestrator.

Launches src/app/run.py as a subprocess and monitors it.
Provides Telegram commands when KutAI is down (/kutai_start, /kutai_status).
When KutAI is running, it handles /kutai_restart and /kutai_stop via exit codes.

Exit code protocol:
  0  = clean shutdown (wrapper waits for /kutai_start)
  42 = restart requested (wrapper restarts immediately)
  *  = crash (wrapper auto-restarts with backoff)

Usage:
  python kutai_wrapper.py [--no-auto-restart]
"""
import asyncio
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ─── Single-instance guard ────────────────────────────────────────────────
# Prevent duplicate wrappers using an OS-level exclusive file lock.
# PID-based checks have a race condition at boot when both start simultaneously.
_LOCK_FILE = Path("logs/wrapper.lock")
_lock_handle = None


def _check_single_instance():
    """Exit if another wrapper is already running. Uses OS-level file lock."""
    global _lock_handle
    _LOCK_FILE.parent.mkdir(exist_ok=True)

    try:
        import msvcrt
        # Open file for writing — this itself doesn't block
        _lock_handle = open(_LOCK_FILE, "w")
        # Try exclusive lock (non-blocking) — fails if another process holds it
        msvcrt.locking(_lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
        # Write our PID for informational purposes
        _lock_handle.write(str(os.getpid()))
        _lock_handle.flush()
    except (OSError, IOError):
        # Another wrapper holds the lock
        try:
            existing_pid = _LOCK_FILE.read_text().strip()
        except Exception:
            existing_pid = "unknown"
        print(f"ERROR: Another wrapper is already running (PID {existing_pid}).")
        print("Kill it first or delete logs/wrapper.lock")
        sys.exit(1)
    except ImportError:
        # Non-Windows fallback: try fcntl
        try:
            import fcntl
            _lock_handle = open(_LOCK_FILE, "w")
            fcntl.flock(_lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            _lock_handle.write(str(os.getpid()))
            _lock_handle.flush()
        except (OSError, IOError):
            print("ERROR: Another wrapper is already running.")
            sys.exit(1)
        except ImportError:
            # No locking available, fall back to PID check
            if _LOCK_FILE.exists():
                try:
                    old_pid = int(_LOCK_FILE.read_text().strip())
                    import ctypes
                    kernel32 = ctypes.windll.kernel32
                    handle = kernel32.OpenProcess(0x1000, False, old_pid)
                    if handle:
                        kernel32.CloseHandle(handle)
                        print(f"ERROR: Another wrapper is already running (PID {old_pid}).")
                        sys.exit(1)
                except Exception:
                    pass
            _LOCK_FILE.write_text(str(os.getpid()))


def _cleanup_lock():
    """Release lock and remove lock file on exit."""
    global _lock_handle
    if _lock_handle:
        try:
            _lock_handle.close()
        except Exception:
            pass
    try:
        _LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


import atexit

_check_single_instance()
atexit.register(_cleanup_lock)

# ─── Venv guard ───────────────────────────────────────────────────────────
# Hard stop if running with system Python when venv exists
_EXPECTED_VENV = Path(__file__).parent / ".venv"
_in_venv = hasattr(sys, "real_prefix") or (
    hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
)
if not _in_venv and _EXPECTED_VENV.exists():
    print(f"ERROR: Running with system Python ({sys.executable})")
    print(f"Use: .venv\\Scripts\\python.exe kutai_wrapper.py")
    sys.exit(1)


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ADMIN_CHAT_ID = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "")
RESTART_EXIT_CODE = 42
BACKOFF_STEPS = [5, 15, 60, 300]  # seconds
BACKOFF_RESET_AFTER = 600  # reset backoff after 10 min stable
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
WRAPPER_LOG = LOG_DIR / "wrapper_meta.log"


def _wlog(msg: str):
    """Append a timestamped line to the wrapper's own meta-log (not piped output)."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(WRAPPER_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


class KutAIWrapper:
    def __init__(self, auto_restart: bool = True):
        self.auto_restart = auto_restart
        self.process: asyncio.subprocess.Process | None = None
        self.running = False
        self.crash_count = 0
        self.total_crashes = 0
        self.start_time: float | None = None
        self.last_crash_time: float = 0
        self.last_exit_code: int | None = None
        self.stderr_tail: deque[str] = deque(maxlen=50)
        self._stop_requested = False
        self._telegram_poller = None
        self._shutdown = False
        self._wrapper_start_time = time.time()

    # ── Subprocess Management ─────────────────────────────────────────────

    async def start_kutai(self):
        """Launch the KutAI orchestrator as a subprocess."""
        if self.process and self.process.returncode is None:
            return

        # Clear stale process reference so wait_for_exit won't spin
        self.process = None
        self.stderr_tail.clear()
        self._stop_requested = False

        # Stop wrapper's Telegram poller before starting KutAI
        await self._stop_telegram_poller()

        venv_python = self._find_python()
        _wlog(f"Starting orchestrator (python={venv_python})")

        run_script = str(Path(__file__).parent / "src" / "app" / "run.py")
        try:
            self.process = await asyncio.create_subprocess_exec(
                venv_python, run_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(Path(__file__).parent),
            )
        except Exception as e:
            _wlog(f"ERROR: Failed to spawn subprocess: {e}")
            self.process = None
            self.running = False
            return

        self.running = True
        self.start_time = time.time()

        # Pipe output to console and log file
        asyncio.create_task(self._pipe_output(self.process.stdout, "stdout"))
        asyncio.create_task(self._pipe_output(self.process.stderr, "stderr"))

    async def stop_kutai(self, timeout: int = 30):
        """Send SIGINT and wait for graceful shutdown."""
        if not self.process or self.process.returncode is not None:
            return
        self._stop_requested = True
        print("\n[Wrapper] Sending shutdown signal to KutAI...")
        try:
            self.process.send_signal(signal.SIGINT)
            await asyncio.wait_for(self.process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            print("[Wrapper] Graceful shutdown timed out, killing...")
            self.process.kill()
            await self.process.wait()
        self.running = False

    async def wait_for_exit(self) -> int:
        """Wait for the subprocess to exit and return the exit code.

        Returns -1 if no process is running. When the process reference is
        stale (already exited), clears it to prevent spin-loops.
        """
        if not self.process:
            # No subprocess — caller should enter poll mode, not spin
            self.running = False
            return -1
        if self.process.returncode is not None:
            # Already exited (stale reference) — clear and return
            code = self.process.returncode
            self.process = None
            self.running = False
            self.last_exit_code = code
            return code
        code = await self.process.wait()
        self.process = None
        self.running = False
        self.last_exit_code = code
        return code

    # ── Output Piping ─────────────────────────────────────────────────────

    async def _pipe_output(self, stream, name: str):
        """Read subprocess output line by line, print to console and save."""
        log_file = LOG_DIR / "wrapper.log"
        try:
            async for line_bytes in stream:
                line = line_bytes.decode("utf-8", errors="replace").rstrip()
                print(line)
                if name == "stderr":
                    self.stderr_tail.append(line)
                try:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(f"[{name}] {line}\n")
                except Exception:
                    pass
        except Exception:
            pass

    # ── Backoff Logic ─────────────────────────────────────────────────────

    def _get_backoff(self) -> int:
        idx = min(self.crash_count, len(BACKOFF_STEPS) - 1)
        return BACKOFF_STEPS[idx]

    def _maybe_reset_backoff(self):
        if self.start_time and (time.time() - self.start_time) > BACKOFF_RESET_AFTER:
            self.crash_count = 0

    # ── Telegram Notifications ────────────────────────────────────────────

    async def _send_telegram(self, text: str):
        """Send a message to the admin via Telegram API (no library needed)."""
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_ADMIN_CHAT_ID:
            return
        import aiohttp
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_ADMIN_CHAT_ID,
            "text": text,
            "parse_mode": "Markdown",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)):
                    pass
        except Exception as e:
            print(f"[Wrapper] Telegram send failed: {e}")

    async def _notify_crash(self, exit_code: int):
        last_lines = "\n".join(self.stderr_tail) if self.stderr_tail else "(no output)"
        # Truncate for Telegram message limit
        if len(last_lines) > 1500:
            last_lines = last_lines[-1500:]
        backoff = self._get_backoff()
        msg = (
            f"🔴 *KutAI Crashed*\n"
            f"Exit code: `{exit_code}`\n"
            f"Crash #{self.total_crashes}\n"
            f"Restarting in {backoff}s\n\n"
            f"```\n{last_lines}\n```"
        )
        await self._send_telegram(msg)

    async def _notify_stopped(self):
        await self._send_telegram(
            "⏹ *KutAI Stopped*\n"
            "Send /kutai\\_start to restart."
        )

    async def _notify_started(self):
        await self._send_telegram("✅ *KutAI Started*")

    # ── Mini Telegram Bot (active only when KutAI is down) ────────────────

    async def _start_telegram_poller(self):
        """Start a minimal Telegram polling loop for /kutai_start."""
        if self._telegram_poller or not TELEGRAM_BOT_TOKEN:
            return
        self._telegram_poller = asyncio.create_task(self._telegram_poll_loop())

    async def _stop_telegram_poller(self):
        if self._telegram_poller:
            self._telegram_poller.cancel()
            try:
                await self._telegram_poller
            except asyncio.CancelledError:
                pass
            self._telegram_poller = None

    async def _telegram_poll_loop(self):
        """Poll Telegram for commands while KutAI is down."""
        import aiohttp
        base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
        offset = 0

        # Flush pending updates first
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{base_url}/getUpdates",
                    params={"offset": -1, "timeout": 0},
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    data = await resp.json()
                    updates = data.get("result", [])
                    if updates:
                        offset = updates[-1]["update_id"] + 1
        except Exception:
            pass

        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{base_url}/getUpdates",
                        params={"offset": offset, "timeout": 30},
                        timeout=aiohttp.ClientTimeout(total=40),
                    ) as resp:
                        data = await resp.json()

                for update in data.get("result", []):
                    offset = update["update_id"] + 1
                    msg = update.get("message", {})
                    text = msg.get("text", "")
                    chat_id = str(msg.get("chat", {}).get("id", ""))

                    if chat_id != str(TELEGRAM_ADMIN_CHAT_ID):
                        continue

                    if text.startswith("/kutai_start") or text.startswith("/start"):
                        await self._send_telegram("🚀 Starting KutAI...")
                        await self.start_kutai()
                        return  # Exit poll loop — KutAI takes over

                    elif text.startswith("/kutai_status") or text.startswith("/status"):
                        await self._send_status()

            except asyncio.CancelledError:
                return
            except Exception as e:
                print(f"[Wrapper] Telegram poll error: {e}")
                await asyncio.sleep(5)

    async def _send_status(self):
        uptime_w = int(time.time() - self._wrapper_start_time)
        uptime_k = int(time.time() - self.start_time) if self.start_time and self.running else 0
        state = "🟢 Running" if self.running else "🔴 Stopped"
        msg = (
            f"*KutAI Wrapper Status*\n"
            f"State: {state}\n"
            f"Wrapper uptime: {uptime_w // 3600}h {(uptime_w % 3600) // 60}m\n"
            f"KutAI uptime: {uptime_k // 3600}h {(uptime_k % 3600) // 60}m\n"
            f"Total crashes: {self.total_crashes}\n"
            f"Last exit code: `{self.last_exit_code}`"
        )
        await self._send_telegram(msg)

    # ── Python Discovery ──────────────────────────────────────────────────

    @staticmethod
    def _find_python() -> str:
        """Find the venv Python or fall back to sys.executable."""
        venv = Path(__file__).parent / ".venv"
        if sys.platform == "win32":
            candidates = [venv / "Scripts" / "python.exe"]
        else:
            candidates = [venv / "bin" / "python"]
        for p in candidates:
            if p.exists():
                return str(p)
        return sys.executable

    # ── Main Loop ─────────────────────────────────────────────────────────

    async def run(self):
        """Main wrapper loop."""
        _wlog(f"KutAI Wrapper started (auto_restart={self.auto_restart})")

        # Start KutAI immediately
        await self.start_kutai()
        if self.running:
            await self._notify_started()
        else:
            _wlog("Initial start failed — entering Telegram poll mode")
            await self._start_telegram_poller()

        while not self._shutdown:
            try:
                exit_code = await self.wait_for_exit()
                self._maybe_reset_backoff()

                if exit_code == -1:
                    # No process was running (spawn failed) — wait for user
                    _wlog("No process to wait on — entering Telegram poll mode")
                    await self._start_telegram_poller()
                    while not self._shutdown and not self.running:
                        await asyncio.sleep(1)
                    if self.running:
                        await self._notify_started()
                    continue

                _wlog(f"KutAI exited with code {exit_code}")

                if exit_code == RESTART_EXIT_CODE:
                    # Restart requested via /kutai_restart
                    await self._send_telegram("♻️ *KutAI Restarting...*")
                    await asyncio.sleep(2)
                    await self.start_kutai()
                    if self.running:
                        await self._notify_started()
                    continue

                elif exit_code == 0:
                    # Clean stop via /kutai_stop or Ctrl+C
                    _wlog("KutAI stopped cleanly (exit 0)")
                    await self._notify_stopped()
                    # Wait for /kutai_start via Telegram
                    await self._start_telegram_poller()
                    while not self._shutdown and not self.running:
                        await asyncio.sleep(1)
                    if self.running:
                        await self._notify_started()
                    continue

                else:
                    # Crash (any non-zero, non-42 exit code)
                    self.crash_count += 1
                    self.total_crashes += 1
                    self.last_crash_time = time.time()
                    backoff = self._get_backoff()
                    _wlog(f"KutAI crashed (exit {exit_code}), "
                          f"crash #{self.total_crashes}, backoff {backoff}s")
                    await self._notify_crash(exit_code)

                    if not self.auto_restart:
                        await self._start_telegram_poller()
                        while not self._shutdown and not self.running:
                            await asyncio.sleep(1)
                        continue

                    # Start Telegram poller during backoff (user can /kutai_start immediately)
                    await self._start_telegram_poller()
                    for i in range(backoff):
                        if self._shutdown or self.running:
                            break
                        await asyncio.sleep(1)

                    if not self.running and not self._shutdown:
                        await self.start_kutai()
                        if self.running:
                            await self._notify_started()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                _wlog(f"UNHANDLED ERROR in main loop: {exc!r}")
                # Always fall back to Telegram poller so we stay reachable
                try:
                    await self._send_telegram(
                        f"⚠️ *Wrapper Error*\n`{exc!r}`\n\n"
                        "Wrapper is still alive. Send /start to retry."
                    )
                except Exception:
                    pass
                await self._start_telegram_poller()
                while not self._shutdown and not self.running:
                    await asyncio.sleep(5)

        # Shutdown
        if self.running:
            await self.stop_kutai()
        await self._stop_telegram_poller()
        _wlog("Wrapper exiting.")


async def async_main():
    auto_restart = "--no-auto-restart" not in sys.argv
    wrapper = KutAIWrapper(auto_restart=auto_restart)

    def _sig(sig, frame):
        _wlog(f"Signal {sig} received, shutting down...")
        wrapper._shutdown = True

    signal.signal(signal.SIGINT, _sig)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _sig)

    try:
        await wrapper.run()
    except Exception as exc:
        _wlog(f"FATAL: async_main crashed: {exc!r}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        _wlog("KeyboardInterrupt — exiting")
    except Exception as exc:
        _wlog(f"FATAL top-level: {exc!r}")
        raise
