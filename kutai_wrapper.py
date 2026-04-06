#!/usr/bin/env python3
"""
Yaşar Usta — KutAI'nin süreç yöneticisi (process manager).

Launches src/app/run.py as a subprocess and monitors it.
Provides Telegram commands when KutAI is down (/kutai_start, /kutai_status).
When KutAI is running, it handles /kutai_restart and /kutai_stop via exit codes.

Exit code protocol:
  0  = clean shutdown (Yaşar Usta waits for /kutai_start)
  42 = restart requested (Yaşar Usta restarts immediately)
  *  = crash (Yaşar Usta auto-restarts with backoff)

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


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
        except Exception:
            pass
        return False
    else:
        try:
            os.kill(pid, 0)  # signal 0 = existence check, no actual signal
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # process exists but we lack permission
        except Exception:
            return False


def _check_single_instance():
    """Exit if another wrapper is already running. Uses OS-level file lock.

    On Windows, uses msvcrt.locking for a byte-range lock on the lock file.
    The file is kept open for the lifetime of the process so the lock persists.

    Robustness against power failures / hard kills:
    - Windows auto-releases msvcrt byte-range locks when a process dies
    - If the lock file exists but the lock is stale (PID dead), we clean up
      and re-acquire
    - PID is zero-padded to a fixed width to ensure the locked byte range
      is always valid
    """
    global _lock_handle
    _LOCK_FILE.parent.mkdir(exist_ok=True)

    # Lock strategy: PID in wrapper.lock (plain text, no OS lock on it).
    # Exclusive lock on a SEPARATE file (wrapper.lk) — a zero-byte sentinel.
    # This avoids the msvcrt.locking + buffered I/O interaction where
    # LockFile's mandatory byte-range lock blocks Python's buffered reads
    # of the PID even when locking a byte past the read range.
    _PID_WIDTH = 10

    try:
        import msvcrt
    except ImportError:
        msvcrt = None

    if msvcrt is not None:
        _acquire_lock_msvcrt(msvcrt, _PID_WIDTH)
    else:
        _acquire_lock_unix(_PID_WIDTH)


_SENTINEL_FILE = Path("logs/wrapper.lk")


def _acquire_lock_msvcrt(msvcrt, pid_width: int):
    """Windows lock using a separate sentinel file.

    wrapper.lock  — stores PID as plain text (never locked, always readable)
    wrapper.lk    — locked with msvcrt; content irrelevant
    """
    global _lock_handle

    def _try_lock(fh):
        """Attempt non-blocking exclusive lock on sentinel file."""
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except (OSError, IOError):
            return False

    def _write_pid():
        """Write our PID to the (unlocked) PID file."""
        with open(_LOCK_FILE, "w") as f:
            f.write(str(os.getpid()).zfill(pid_width))

    def _read_pid():
        """Read PID from the (unlocked) PID file."""
        try:
            raw = _LOCK_FILE.read_text().strip()
            return int(raw) if raw else None
        except (ValueError, OSError):
            return None

    # Ensure sentinel file exists with at least 1 byte
    _SENTINEL_FILE.parent.mkdir(exist_ok=True)
    if not _SENTINEL_FILE.exists():
        _SENTINEL_FILE.write_text("L")

    _lock_handle = open(_SENTINEL_FILE, "r+")

    if _try_lock(_lock_handle):
        _write_pid()
        return

    # Lock failed — read PID from the separate (unlocked) file
    existing_pid = _read_pid()

    if existing_pid is not None and _is_pid_alive(existing_pid):
        _lock_handle.close()
        _lock_handle = None
        print(f"ERROR: Yasar Usta already running (PID {existing_pid}).")
        print("Use /kutai_stop in Telegram or delete logs/wrapper.lock")
        sys.exit(1)

    # PID is dead or unreadable — stale lock (e.g. after power failure).
    print(f"[Yasar Usta] Stale lock detected (PID {existing_pid or '?'} is dead). Cleaning up.")
    _lock_handle.close()
    _lock_handle = None
    try:
        _SENTINEL_FILE.unlink(missing_ok=True)
    except Exception:
        pass

    _SENTINEL_FILE.write_text("L")
    _lock_handle = open(_SENTINEL_FILE, "r+")
    if _try_lock(_lock_handle):
        _write_pid()
        return

    _lock_handle.close()
    _lock_handle = None
    print("ERROR: Could not acquire wrapper lock even after stale-lock cleanup.")
    sys.exit(1)


def _acquire_lock_unix(pid_width: int):
    """Unix lock acquisition using fcntl, with PID-based fallback."""
    global _lock_handle
    try:
        import fcntl
        _lock_handle = open(_LOCK_FILE, "w")
        fcntl.flock(_lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_handle.write(str(os.getpid()).zfill(pid_width))
        _lock_handle.flush()
    except (OSError, IOError):
        print("ERROR: Yasar Usta already running.")
        sys.exit(1)
    except ImportError:
        # No OS-level locking available — fall back to PID check
        if _LOCK_FILE.exists():
            try:
                old_pid = int(_LOCK_FILE.read_text().strip())
                if _is_pid_alive(old_pid):
                    print(f"ERROR: Yasar Usta already running (PID {old_pid}).")
                    sys.exit(1)
                else:
                    print(f"[Wrapper] Stale lock (PID {old_pid} is dead). Cleaning up.")
            except Exception:
                pass
        _LOCK_FILE.write_text(str(os.getpid()).zfill(pid_width))


def _cleanup_lock():
    """Release lock and remove lock file on exit."""
    global _lock_handle
    _LOCK_BYTES = 10  # must match _PID_WIDTH in _check_single_instance
    if _lock_handle:
        try:
            import msvcrt
            _lock_handle.seek(0)
            msvcrt.locking(_lock_handle.fileno(), msvcrt.LK_UNLOCK, _LOCK_BYTES)
        except Exception:
            pass
        try:
            _lock_handle.close()
        except Exception:
            pass
        _lock_handle = None
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
CLAUDE_REMOTE_SIGNAL = LOG_DIR / "claude_remote.signal"
PROJECT_ROOT = Path(__file__).resolve().parent
CLAUDE_CMD = Path(os.environ.get("APPDATA", "")) / "npm" / "claude.cmd"


WRAPPER_JSONL = LOG_DIR / "wrapper.jsonl"


def _wlog(msg: str, level: str = "INFO"):
    """Append a timestamped line to the wrapper's own meta-log (not piped output).

    Also writes JSONL to wrapper.jsonl so the yazbunu viewer can display it.
    """
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    try:
        print(line)
    except UnicodeEncodeError:
        # cp1252 console can't handle Turkish chars — print ASCII-safe version
        print(line.encode("ascii", errors="replace").decode())
    try:
        with open(WRAPPER_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    # JSONL for yazbunu viewer — inline, no yazbunu import
    try:
        import json as _json
        from datetime import datetime, timezone
        doc = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "src": "wrapper.yasar_usta",
            "msg": msg,
        }
        with open(WRAPPER_JSONL, "a", encoding="utf-8") as f:
            f.write(_json.dumps(doc, ensure_ascii=False) + "\n")
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
        self._signal_watcher: asyncio.Task | None = None
        self._claude_process: asyncio.subprocess.Process | None = None
        # yazbunu is fully independent (detached process with PID file)

    # ── Subprocess Management ─────────────────────────────────────────────

    async def start_kutai(self, _from_poller: bool = False):
        """Launch the KutAI orchestrator as a subprocess."""
        if self.process and self.process.returncode is None:
            return

        # Clear stale process reference so wait_for_exit won't spin
        self.process = None
        self.stderr_tail.clear()
        self._stop_requested = False

        # Stop wrapper's Telegram poller before starting KutAI.
        # Skip if called from within the poller itself (it will exit after return).
        if not _from_poller:
            await self._stop_telegram_poller()
        else:
            # Clear the poller reference — the poller task is about to return
            self._telegram_poller = None

        # Brief pause to ensure any in-flight getUpdates long-poll request
        # from the wrapper has fully completed before the orchestrator starts
        # its own polling.  Without this, both pollers can race.
        # 3s gives the wrapper's short-timeout poll (5s) time to finish and
        # release the Telegram connection before the orchestrator claims it.
        await asyncio.sleep(3)

        venv_python = self._find_python()
        _wlog(f"Starting orchestrator (python={venv_python})")

        run_script = str(Path(__file__).parent / "src" / "app" / "run.py")
        try:
            import subprocess as _sp
            _kwargs = {}
            if sys.platform == "win32":
                _kwargs["creationflags"] = _sp.CREATE_NEW_PROCESS_GROUP

            self.process = await asyncio.create_subprocess_exec(
                venv_python, run_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024 * 1024,  # 1 MB line buffer (default 64 KB too small for long log lines)
                cwd=str(Path(__file__).parent),
                **_kwargs,
            )
        except Exception as e:
            _wlog(f"ERROR: Failed to spawn subprocess: {e}", level="ERROR")
            self.process = None
            self.running = False
            return

        self.running = True
        self.start_time = time.time()

        # Reset heartbeat so the hung-detector doesn't see a stale timestamp
        # from the previous run and immediately kill the new orchestrator.
        hb_path = os.path.join("logs", "orchestrator.heartbeat")
        try:
            with open(hb_path, "w") as f:
                f.write(str(time.time()))
        except Exception:
            pass

        # Pipe output to console and log file
        asyncio.create_task(self._pipe_output(self.process.stdout, "stdout"))
        asyncio.create_task(self._pipe_output(self.process.stderr, "stderr"))

    def _is_orchestrator_hung(self, max_stale_seconds=120):
        hb_path = os.path.join("logs", "orchestrator.heartbeat")
        try:
            with open(hb_path, "r") as f:
                last_beat = float(f.read().strip())
            return (time.time() - last_beat) > max_stale_seconds
        except (FileNotFoundError, ValueError):
            return False  # No file yet = still starting up

    def _is_orchestrator_healthy(self) -> bool:
        """Check if orchestrator is alive by reading heartbeat file.

        Returns True if the heartbeat timestamp is fresher than 90 seconds.
        Returns False if the file is missing, unreadable, or stale.
        """
        heartbeat_file = Path("logs/heartbeat")
        if not heartbeat_file.exists():
            return False
        try:
            ts = float(heartbeat_file.read_text().strip())
            return (time.time() - ts) < 90
        except (ValueError, OSError):
            return False

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
        Uses a timeout loop with heartbeat checking to detect hung processes.
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

        # Timeout loop with heartbeat check
        while True:
            try:
                code = await asyncio.wait_for(self.process.wait(), timeout=30)
                self.process = None
                self.running = False
                self.last_exit_code = code
                return code
            except asyncio.TimeoutError:
                if self._is_orchestrator_hung():
                    _wlog("Orchestrator heartbeat stale >120s — killing hung process", level="ERROR")
                    try:
                        self.process.kill()
                    except Exception:
                        pass
                    self.process = None
                    self.running = False
                    self.last_exit_code = -1
                    return -1

    # ── Output Piping ─────────────────────────────────────────────────────

    async def _pipe_output(self, stream, name: str):
        """Read subprocess output line by line, print to console and save.

        CRITICAL: This loop must NEVER die.  If it stops reading, the pipe
        buffer fills up (64 KB on Windows) and the child process blocks on
        stdout, freezing the event loop and killing the heartbeat.
        """
        log_file = LOG_DIR / "wrapper.log"
        while True:
            try:
                line_bytes = await stream.readline()
                if not line_bytes:
                    break  # EOF — subprocess exited
                line = line_bytes.decode("utf-8", errors="replace").rstrip()
            except ValueError:
                # "Separator is found, but chunk is longer than limit"
                # A single log line exceeded the 64KB StreamReader buffer.
                # Drain the oversized chunk and continue.
                try:
                    await stream.readuntil(b"\n")
                except Exception:
                    try:
                        # Last resort: read whatever is available
                        stream._buffer.clear()
                    except Exception:
                        pass
                continue
            except Exception:
                # Stream broken (process died, pipe closed, etc.)
                break

            try:
                print(line)
            except UnicodeEncodeError:
                print(line.encode("ascii", errors="replace").decode())
            except Exception:
                pass

            if name == "stderr":
                self.stderr_tail.append(line)
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"[{name}] {line}\n")
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

    async def _send_telegram(self, text: str, reply_markup: dict | None = None):
        """Send a message to the admin via Telegram API (no library needed).

        Optionally include a reply_markup dict (e.g. ReplyKeyboardMarkup JSON).
        """
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_ADMIN_CHAT_ID:
            return
        import aiohttp
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_ADMIN_CHAT_ID,
            "text": text,
            "parse_mode": "Markdown",
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)):
                    pass
        except Exception as e:
            print(f"[Wrapper] Telegram send failed: {e}")

    # Reply keyboard shown when KutAI is down
    _KB_BASLAT = {
        "keyboard": [
            [{"text": "▶️ Başlat"}, {"text": "🔧 Yaşar Usta"}],
        ],
        "resize_keyboard": True,
        "one_time_keyboard": False,
        "is_persistent": True,
    }

    async def _send_baslat_prompt(self, reason: str = ""):
        """Send the [▶️ Başlat] keyboard with an optional reason prefix."""
        msg = "⚠️ Kutay durdu. Başlatmak için butona bas."
        if reason:
            msg = f"{reason}\n{msg}"
        await self._send_telegram(msg, reply_markup=self._KB_BASLAT)

    async def _notify_crash(self, exit_code: int):
        last_lines = "\n".join(self.stderr_tail) if self.stderr_tail else "(no output)"
        # Truncate for Telegram message limit
        if len(last_lines) > 1500:
            last_lines = last_lines[-1500:]
        backoff = self._get_backoff()
        msg = (
            f"🔴 *Kutay Crashed*\n"
            f"Exit code: `{exit_code}`\n"
            f"Crash #{self.total_crashes}\n"
            f"Restarting in {backoff}s\n\n"
            f"```\n{last_lines}\n```"
        )
        await self._send_telegram(msg, reply_markup=self._KB_BASLAT)

    async def _notify_stopped(self):
        await self._send_telegram(
            "⏹ *Kutay Stopped*\n"
            "Send /kutai\\_start to restart.",
            reply_markup=self._KB_BASLAT,
        )

    async def _notify_started(self):
        await self._send_telegram("✅ *Kutay Started*")

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
        """Poll Telegram for commands while KutAI is down.

        Consumes ALL updates while polling.  The orchestrator is not running,
        so there is nobody else to process them.  Non-command messages from
        the admin get a "Kutay is down" reply with the Başlat keyboard.

        Wrapper commands handled here:
        - "▶️ Başlat" / /kutai_start: start the orchestrator
        - /kutai_status: show status
        - "⚙️ Sistem": kill hung orchestrator + show Başlat
        - "🔧 Yaşar Usta": show process list
        - /logs: show recent logs
        """
        import aiohttp
        base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

        offset = 0
        # Rate-limit "Kutay is down" replies so we don't spam on every tap
        last_down_reply: float = 0
        DOWN_REPLY_COOLDOWN = 30  # seconds
        _wlog("Telegram poller started")

        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{base_url}/getUpdates",
                        params={"offset": offset, "timeout": 5},
                        timeout=aiohttp.ClientTimeout(total=15),
                    ) as resp:
                        data = await resp.json()

                updates = data.get("result", [])
                if not updates:
                    continue

                max_uid = 0
                for update in updates:
                    uid = update["update_id"]
                    if uid > max_uid:
                        max_uid = uid

                    # ── Handle callback queries (inline buttons) ──
                    cb = update.get("callback_query")
                    if cb:
                        cb_chat_id = str(cb.get("message", {}).get("chat", {}).get("id", ""))
                        cb_data = cb.get("data", "")
                        cb_msg_id = cb.get("message", {}).get("message_id")
                        if cb_chat_id == str(TELEGRAM_ADMIN_CHAT_ID):
                            # Answer callback to remove loading spinner
                            try:
                                async with aiohttp.ClientSession() as s_cb:
                                    await s_cb.post(
                                        f"{base_url}/answerCallbackQuery",
                                        json={"callback_query_id": cb["id"]},
                                        timeout=aiohttp.ClientTimeout(total=5),
                                    )
                            except Exception:
                                pass
                            if cb_data == "restart_usta":
                                offset = max_uid + 1
                                await self._restart_self()
                                return
                            elif cb_data == "usta_refresh":
                                await self._send_processes(edit_message_id=cb_msg_id)
                            elif cb_data == "restart_yazbunu":
                                await self._stop_yazbunu()
                                await self._start_yazbunu()
                                await self._send_processes(edit_message_id=cb_msg_id)
                        continue

                    msg = update.get("message", {})
                    text = msg.get("text", "")
                    chat_id = str(msg.get("chat", {}).get("id", ""))

                    # Only handle messages from admin
                    if chat_id != str(TELEGRAM_ADMIN_CHAT_ID):
                        continue

                    # ── Wrapper commands ──
                    if text.startswith("/kutai_start") or text == "▶️ Başlat":
                        await self._send_telegram("🚀 Kutay başlatılıyor...")
                        await self.start_kutai(_from_poller=True)
                        return

                    elif text.startswith("/kutai_status") or text == "🔧 Yaşar Usta":
                        await self._send_processes()

                    elif text.startswith("/restart_usta"):
                        await self._restart_self()
                        return  # won't reach here, but for clarity

                    elif text.startswith("/logs"):
                        await self._send_logs(text)

                    elif text == "⚙️ Sistem":
                        if self.process and self.process.returncode is None:
                            _wlog("⚙️ Sistem tapped — killing hung orchestrator")
                            try:
                                self.process.kill()
                                await self.process.wait()
                            except Exception as e:
                                _wlog(f"Failed to kill hung orchestrator: {e}", level="ERROR")
                            self.process = None
                            self.running = False
                        await self._send_baslat_prompt("🔴 Kutay yanıt vermiyor.")

                    elif text:
                        # Any other admin message — let them know KutAI is down
                        now = time.time()
                        if now - last_down_reply > DOWN_REPLY_COOLDOWN:
                            last_down_reply = now
                            await self._send_baslat_prompt("⏸ Kutay şu an kapalı.")

                # Consume all processed updates
                offset = max_uid + 1

            except asyncio.CancelledError:
                return
            except Exception as e:
                _wlog(f"Telegram poll error: {e}", level="ERROR")
                await asyncio.sleep(5)

    # ── Signal File Watcher ──────────────────────────────────────────────

    async def _start_signal_watcher(self):
        """Start background task that watches for signal files."""
        if self._signal_watcher:
            return
        self._signal_watcher = asyncio.create_task(self._signal_watch_loop())

    async def _stop_signal_watcher(self):
        if self._signal_watcher:
            self._signal_watcher.cancel()
            try:
                await self._signal_watcher
            except asyncio.CancelledError:
                pass
            self._signal_watcher = None

    async def _signal_watch_loop(self):
        """Poll for signal files while the orchestrator is running."""
        while True:
            try:
                await asyncio.sleep(3)
                if CLAUDE_REMOTE_SIGNAL.exists():
                    CLAUDE_REMOTE_SIGNAL.unlink()
                    await self._start_claude_remote()
            except asyncio.CancelledError:
                return
            except Exception as e:
                _wlog(f"Signal watcher error: {e!r}", level="ERROR")
                await asyncio.sleep(10)

    async def _start_claude_remote(self):
        """Spawn a Claude Code remote-control server session.

        Starts `claude remote-control` which creates a persistent server that
        appears in the user's claude.ai/code session list. The session URL
        is extracted from stdout and sent via Telegram.
        """
        if self._claude_process and self._claude_process.returncode is None:
            _wlog(f"Existing Claude session still running (PID {self._claude_process.pid}), starting another")
        await self._send_telegram("🖥️ Claude Code oturumu başlatılıyor...")

        _wlog("Starting Claude Code remote-control server")
        try:
            claude_bin = str(CLAUDE_CMD) if CLAUDE_CMD.exists() else "claude"
            self._claude_process = await asyncio.create_subprocess_exec(
                claude_bin, "remote-control",
                "--name", "Kutay",
                "--permission-mode", "bypassPermissions",
                cwd=str(PROJECT_ROOT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _wlog(f"Claude remote-control server started (PID {self._claude_process.pid})")

            # Read stdout to capture session URL (first few lines)
            session_url = None
            try:
                for _ in range(20):  # read up to 20 lines looking for URL
                    line_bytes = await asyncio.wait_for(
                        self._claude_process.stdout.readline(), timeout=10,
                    )
                    if not line_bytes:
                        break
                    line = line_bytes.decode("utf-8", errors="replace").strip()
                    _wlog(f"[claude-rc] {line}")
                    if "claude.ai" in line or "http" in line.lower():
                        # Extract URL from line
                        import re
                        url_match = re.search(r"https?://\S+", line)
                        if url_match:
                            session_url = url_match.group(0)
                            break
            except asyncio.TimeoutError:
                pass

            if session_url:
                await self._send_telegram(
                    "🖥️ *Claude Code Remote Control*\n\n"
                    f"🔗 [Session'a bağlan]({session_url})\n\n"
                    f"PID: `{self._claude_process.pid}`",
                )
            else:
                await self._send_telegram(
                    "🖥️ *Claude Code Remote Control started*\n"
                    f"PID: `{self._claude_process.pid}`\n\n"
                    "claude.ai/code adresinden bağlanabilirsin."
                )

            # Continue reading output in background
            asyncio.create_task(self._pipe_output(
                self._claude_process.stdout, "claude-rc"))

        except FileNotFoundError:
            _wlog("'claude' command not found — is Claude Code installed?", level="WARNING")
            await self._send_telegram("❌ `claude` command not found. Claude Code kurulu mu?")
        except Exception as e:
            _wlog(f"Failed to start Claude remote-control: {e!r}", level="ERROR")
            await self._send_telegram(f"❌ Claude Code başlatılamadı: `{e!r}`")

    # ── Yazbunu Log Viewer ─────────────────────────────────────────────────

    _YAZBUNU_PID_FILE = LOG_DIR / "yazbunu.pid"

    @staticmethod
    def _yazbunu_pid_alive() -> int | None:
        """Read yazbunu PID file and return PID if the process is alive, else None."""
        pid_file = KutAIWrapper._YAZBUNU_PID_FILE
        try:
            pid = int(pid_file.read_text().strip())
        except (FileNotFoundError, ValueError):
            return None
        if _is_pid_alive(pid):
            return pid
        # Stale PID file
        try:
            pid_file.unlink(missing_ok=True)
        except Exception:
            pass
        return None

    @staticmethod
    async def _yazbunu_http_alive() -> bool:
        """Check if yazbunu is responding on its HTTP port."""
        import aiohttp
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    "http://127.0.0.1:9880/",
                    timeout=aiohttp.ClientTimeout(total=3),
                ) as r:
                    return r.status == 200
        except Exception:
            return False

    async def _start_yazbunu(self):
        """Ensure yazbunu is running. Spawns a detached process if none is alive."""
        if self._yazbunu_pid_alive() or await self._yazbunu_http_alive():
            _wlog("yazbunu already running")
            return
        venv_python = self._find_python()
        log_file = LOG_DIR / "yazbunu.log"
        _wlog("Starting yazbunu log viewer (port 9880)")
        try:
            import subprocess as _sp
            flags = _sp.CREATE_NEW_PROCESS_GROUP | _sp.DETACHED_PROCESS
            out_fh = open(log_file, "a", encoding="utf-8")
            proc = _sp.Popen(
                [venv_python, "-m", "yazbunu.server",
                 "--log-dir", "./logs", "--port", "9880", "--host", "0.0.0.0"],
                stdout=out_fh,
                stderr=out_fh,
                cwd=str(Path(__file__).parent),
                creationflags=flags,
                close_fds=True,
            )
            self._YAZBUNU_PID_FILE.write_text(str(proc.pid))
            _wlog(f"yazbunu started (PID {proc.pid}, detached)")
        except Exception as e:
            _wlog(f"Failed to start yazbunu: {e!r}", level="ERROR")

    async def _stop_yazbunu(self):
        """Kill the running yazbunu process (by PID file)."""
        pid = self._yazbunu_pid_alive()
        if not pid:
            return
        _wlog(f"Stopping yazbunu (PID {pid})")
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as e:
            _wlog(f"yazbunu stop error: {e!r}", level="ERROR")
        try:
            self._YAZBUNU_PID_FILE.unlink(missing_ok=True)
        except Exception:
            pass

    async def _ensure_yazbunu(self):
        """Check if yazbunu is alive, restart if dead."""
        if self._yazbunu_pid_alive() or await self._yazbunu_http_alive():
            return
        _wlog("yazbunu not running, restarting", level="WARNING")
        await self._start_yazbunu()

    # ── Status & Logs ────────────────────────────────────────────────────────

    async def _send_status(self):
        uptime_w = int(time.time() - self._wrapper_start_time)
        uptime_k = int(time.time() - self.start_time) if self.start_time and self.running else 0
        state = "🟢 Running" if self.running else "🔴 Stopped"
        yz_state = "🟢 Running" if self._yazbunu_pid_alive() else "🔴 Stopped"
        msg = (
            f"*Yaşar Usta - Durum*\n"
            f"Durum: {state}\n"
            f"Yaşar Usta çalışma: {uptime_w // 3600}s {(uptime_w % 3600) // 60}dk\n"
            f"Kutay çalışma: {uptime_k // 3600}s {(uptime_k % 3600) // 60}dk\n"
            f"Yazbunu: {yz_state}\n"
            f"Toplam çökme: {self.total_crashes}\n"
            f"Son çıkış kodu: `{self.last_exit_code}`"
        )
        await self._send_telegram(msg)

    async def _build_status_text(self) -> str:
        """Build the Yaşar Usta status panel text."""
        import subprocess as _sp
        lines = []
        wrappers = []
        orchestrators = []
        llama_pids = []

        # Python processes
        try:
            raw = _sp.check_output(
                ['wmic', 'process', 'where', "name='python.exe'",
                 'get', 'ProcessId,CommandLine'],
                text=True, timeout=5,
            )
            for line in raw.strip().splitlines():
                line = line.strip()
                if not line or line.startswith("CommandLine"):
                    continue
                pid = line.split()[-1] if line.split() else ""
                if "wrapper" in line.lower():
                    wrappers.append(pid)
                elif "run.py" in line:
                    orchestrators.append(pid)
        except Exception as e:
            lines.append(f"⚠️ Süreç listesi alınamadı: {e}")

        # llama-server
        try:
            llama_raw = _sp.check_output(
                ['tasklist', '/FI', 'IMAGENAME eq llama-server.exe'],
                text=True, timeout=5,
            )
            for ll in llama_raw.splitlines():
                if 'llama-server' in ll.lower():
                    parts = ll.split()
                    if len(parts) >= 2:
                        llama_pids.append(parts[1])
        except Exception:
            pass

        n_wrappers = (len(wrappers) + 1) // 2
        n_orchestrators = (len(orchestrators) + 1) // 2

        # ── Health: Yaşar Usta ──
        uptime_w = int(time.time() - self._wrapper_start_time)
        usta_line = f"🔵 Yaşar Usta: çalışıyor ({uptime_w // 3600}s {(uptime_w % 3600) // 60}dk)"

        # ── Health: Kutay (orchestrator) ──
        kutay_healthy = False
        heartbeat_age = None
        for hb_path in ["logs/orchestrator.heartbeat", "logs/heartbeat"]:
            try:
                with open(hb_path, "r") as f:
                    last_beat = float(f.read().strip())
                heartbeat_age = time.time() - last_beat
                kutay_healthy = heartbeat_age < 60
                break
            except (FileNotFoundError, ValueError):
                continue

        if n_orchestrators == 0:
            kutay_line = "💀 Kutay: çalışmıyor"
        elif kutay_healthy:
            age_str = f"{int(heartbeat_age)}sn önce" if heartbeat_age else ""
            kutay_line = f"💚 Kutay: sağlıklı (heartbeat {age_str})"
        elif heartbeat_age is not None:
            kutay_line = f"🔴 Kutay: YANIT VERMİYOR ({int(heartbeat_age)}sn sessiz)"
        else:
            kutay_line = "⚪ Kutay: heartbeat dosyası yok"

        # ── Health: llama-server ──
        if llama_pids:
            llama_line = f"🟡 llama-server: çalışıyor (PID {', '.join(llama_pids)})"
        else:
            llama_line = "⚫ llama-server: çalışmıyor"

        # ── Health: yazbunu ──
        # Check HTTP first (authoritative), PID file is secondary
        yz_responding = await self._yazbunu_http_alive()
        yz_pid = self._yazbunu_pid_alive()
        if yz_responding:
            pid_str = f", PID {yz_pid}" if yz_pid else ""
            yz_line = f"📊 yazbunu: çalışıyor (port 9880{pid_str})"
        elif yz_pid:
            yz_line = f"🟠 yazbunu: süreç var ama yanıt yok (PID {yz_pid})"
        else:
            yz_line = "⚫ yazbunu: çalışmıyor"

        # ── Summary ──
        warnings = []
        if n_wrappers > 1:
            warnings.append(f"⚠️ {n_wrappers} Yaşar Usta süreci var!")
        if n_orchestrators > 1:
            warnings.append(f"⚠️ {n_orchestrators} Kutay süreci var!")

        ts = time.strftime("%H:%M:%S")
        text = (
            f"🔧 *Yaşar Usta*\n\n"
            f"{usta_line}\n"
            f"{kutay_line}\n"
            f"{llama_line}\n"
            f"{yz_line}\n"
            f"\nÇökmeler: {self.total_crashes}"
        )
        if warnings:
            text += "\n" + "\n".join(warnings)
        text += f"\n\n_Son güncelleme: {ts}_"
        return text

    def _status_inline_keyboard(self) -> dict:
        """Build inline keyboard for the status panel."""
        buttons = [
            [
                {"text": "🔄 Yenile", "callback_data": "usta_refresh"},
            ],
            [
                {"text": "♻️ Usta'yı Yeniden Başlat", "callback_data": "restart_usta"},
                {"text": "📊 Yazbunu Yeniden Başlat", "callback_data": "restart_yazbunu"},
            ],
        ]
        return {"inline_keyboard": buttons}

    async def _send_processes(self, edit_message_id: int | None = None):
        """Show status panel. If edit_message_id is given, edit that message instead of sending new."""
        import aiohttp
        text = await self._build_status_text()
        inline_kb = self._status_inline_keyboard()

        if edit_message_id:
            # Edit existing message in-place (keeps inline keyboard)
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/editMessageText"
            payload = {
                "chat_id": TELEGRAM_ADMIN_CHAT_ID,
                "message_id": edit_message_id,
                "text": text,
                "parse_mode": "Markdown",
                "reply_markup": inline_kb,
            }
        else:
            # New message — first set the reply keyboard (replaces orchestrator's),
            # then send the status panel with inline buttons.
            await self._send_telegram("🔧 Yaşar Usta paneli:", reply_markup=self._KB_BASLAT)
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": TELEGRAM_ADMIN_CHAT_ID,
                "text": text,
                "parse_mode": "Markdown",
                "reply_markup": inline_kb,
            }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)):
                    pass
        except Exception as e:
            _wlog(f"Failed to send status: {e}", level="ERROR")

    async def _restart_self(self):
        """Restart the wrapper process itself.

        Spawns a new detached wrapper process, then exits cleanly.
        The new process will acquire the lock after we release it.
        """
        _wlog("Self-restart requested via Telegram")
        await self._send_telegram("🔄 *Yaşar Usta yeniden başlatılıyor...*")

        # Stop managed processes (yazbunu is independent — leave it running)
        if self.running:
            await self.stop_kutai()
        await self._stop_telegram_poller()

        # Flush Telegram update queue so the new instance starts clean
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                # offset=-1 confirms all pending updates
                await session.get(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
                    params={"offset": -1, "timeout": 0},
                    timeout=aiohttp.ClientTimeout(total=5),
                )
        except Exception:
            pass

        # Spawn new wrapper (detached)
        import subprocess as _sp
        venv_python = self._find_python()
        script = str(Path(__file__).resolve())
        argv = [a for a in sys.argv[1:] if a != script]
        _wlog(f"Spawning new wrapper: {venv_python} {script}")
        _sp.Popen(
            [venv_python, script] + argv,
            creationflags=(_sp.CREATE_NEW_PROCESS_GROUP
                           | _sp.DETACHED_PROCESS
                           | _sp.CREATE_NO_WINDOW),
            close_fds=True,
            cwd=str(Path(__file__).parent),
        )

        # Release lock and exit
        _cleanup_lock()
        _wlog("Old wrapper exiting for restart")
        os._exit(0)

    async def _send_logs(self, text: str):
        """Read and send last N lines of orchestrator.jsonl."""
        import json as _json

        parts = text.strip().split()
        n = 20
        if len(parts) > 1:
            try:
                n = min(int(parts[1]), 50)
            except ValueError:
                pass

        log_path = Path("logs/orchestrator.jsonl")
        if not log_path.exists():
            await self._send_telegram("📋 No log file found.")
            return

        try:
            with open(log_path, "r", encoding="utf-8") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 100_000))
                chunk = f.read()
                lines = chunk.strip().split("\n")

            last_n = lines[-n:]
            formatted = []
            for line in last_n:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = _json.loads(line)
                    ts = entry.get("ts", "?")
                    if "T" in ts:
                        ts = ts.split("T")[1][:8]
                    elif " " in ts:
                        ts = ts.split(" ")[1][:8]
                    level = entry.get("level", "?")[:4]
                    comp = entry.get("src", "?").split(".")[-1]
                    msg = entry.get("msg", "")[:120]
                    icon = {"ERRO": "🔴", "CRIT": "🔴", "WARN": "🟡", "INFO": "⚪", "DEBU": "⚫"}.get(level, "⚪")
                    formatted.append(f"{icon} `{ts}` *{comp}*: {msg}")
                except (ValueError, KeyError):
                    formatted.append(f"⚫ {line[:120]}")

            if not formatted:
                await self._send_telegram("📋 No log entries found.")
                return

            msg = "\n".join(formatted)
            if len(msg) > 4000:
                msg = msg[-4000:]
                msg = "...(truncated)\n" + msg[msg.index("\n") + 1:]

            if self._yazbunu_pid_alive():
                msg += "\n\n📊 Full viewer: http://localhost:9880"
            await self._send_telegram(msg)
        except Exception as e:
            await self._send_telegram(f"❌ Error reading logs: {e}")

    # ── Post-exit Cleanup ──────────────────────────────────────────────────

    @staticmethod
    def _kill_orphan_processes():
        """Kill orphaned llama-server (and optionally Ollama) after orchestrator exits.

        This is the ultimate safety net — the wrapper always survives the
        orchestrator, so this covers crashes, taskkill, os._exit(), etc.
        """
        import subprocess as sp

        # Kill llama-server
        try:
            result = sp.run(
                ["taskkill", "/F", "/IM", "llama-server.exe"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                _wlog(f"Killed orphaned llama-server: {result.stdout.strip()}")
            # returncode != 0 means no matching process — that's fine
        except Exception as e:
            _wlog(f"llama-server cleanup error: {e}", level="WARNING")

        # Stop Ollama if it's running (it auto-restarts via Windows startup,
        # but we don't want it hogging VRAM between orchestrator restarts).
        try:
            result = sp.run(
                ["taskkill", "/F", "/IM", "ollama.exe"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                _wlog(f"Stopped Ollama: {result.stdout.strip()}")
        except Exception as e:
            _wlog(f"Ollama cleanup error: {e}", level="WARNING")

        # Also kill ollama runner processes (ollama_llama_server.exe) that
        # Ollama spawns for model inference — these hold VRAM.
        try:
            result = sp.run(
                ["taskkill", "/F", "/IM", "ollama_llama_server.exe"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                _wlog(f"Stopped Ollama runner: {result.stdout.strip()}")
        except Exception:
            pass

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
        _wlog(f"Yasar Usta started (auto_restart={self.auto_restart})")

        # Clean up stale signal files from previous runs
        if CLAUDE_REMOTE_SIGNAL.exists():
            CLAUDE_REMOTE_SIGNAL.unlink()

        # Start yazbunu log viewer (runs independently of orchestrator)
        await self._start_yazbunu()

        # Announce ourselves
        await self._send_telegram(
            "🔧 *Bennn... Yaşar Usta!*\n\n"
            "Kutay'ı başlatıyorum...",
            reply_markup=self._KB_BASLAT,
        )

        # Start KutAI immediately
        await self.start_kutai()
        if self.running:
            await self._notify_started()
            await self._start_signal_watcher()
        else:
            _wlog("Initial start failed — entering Telegram poll mode")
            await self._start_telegram_poller()

        while not self._shutdown:
            try:
                exit_code = await self.wait_for_exit()
                await self._ensure_yazbunu()
                await self._stop_signal_watcher()
                self._kill_orphan_processes()
                self._maybe_reset_backoff()

                if exit_code == -1:
                    if self.last_crash_time and (time.time() - self.last_crash_time) < 10:
                        # Hung kill right after a crash — treat as spawn failure,
                        # wait for user to avoid rapid restart loops
                        _wlog("No process to wait on — entering Telegram poll mode")
                        await self._start_telegram_poller()
                        while not self._shutdown and not self.running:
                            await asyncio.sleep(1)
                        if self.running:
                            await self._notify_started()
                            await self._start_signal_watcher()
                        continue
                    # Orchestrator was killed for being hung — auto-restart
                    _wlog("KutAI hung — Yasar Usta auto-restarting after 5s", level="ERROR")
                    self.crash_count += 1
                    self.total_crashes += 1
                    self.last_crash_time = time.time()
                    await self._send_telegram(
                        "🔴 Kutay dondu — Yaşar Usta 5sn içinde yeniden başlatıyor"
                    )
                    await self._start_telegram_poller()
                    for i in range(5):
                        if self._shutdown or self.running:
                            break
                        await asyncio.sleep(1)
                    if not self.running and not self._shutdown:
                        await self.start_kutai()
                        if self.running:
                            await self._notify_started()
                            await self._start_signal_watcher()
                    continue

                _wlog(f"KutAI exited with code {exit_code}")

                if exit_code == RESTART_EXIT_CODE:
                    # Restart requested via /kutai_restart
                    # Do NOT start Telegram poller during restart — it steals updates
                    await self._send_telegram("♻️ *Kutay yeniden başlatılıyor...*")
                    await asyncio.sleep(3)
                    await self.start_kutai()
                    if self.running:
                        await self._notify_started()
                        await self._start_signal_watcher()
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
                        await self._start_signal_watcher()
                    continue

                else:
                    # Crash (any non-zero, non-42 exit code)
                    self.crash_count += 1
                    self.total_crashes += 1
                    self.last_crash_time = time.time()
                    backoff = self._get_backoff()
                    _wlog(f"KutAI crashed (exit {exit_code}), "
                          f"crash #{self.total_crashes}, backoff {backoff}s",
                          level="ERROR")
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
                            await self._start_signal_watcher()

            except asyncio.CancelledError:
                _wlog("Main loop cancelled (CancelledError)")
                break
            except Exception as exc:
                _wlog(f"UNHANDLED ERROR in main loop: {exc!r}", level="CRITICAL")
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

        # Shutdown (yazbunu is independent — leave it running)
        if self.running:
            await self.stop_kutai()
        await self._stop_signal_watcher()
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
        _wlog(f"FATAL: async_main crashed: {exc!r}", level="CRITICAL")
        raise
    finally:
        _wlog("Wrapper process ending (finally block)")


if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        _wlog("KeyboardInterrupt — exiting")
    except Exception as exc:
        _wlog(f"FATAL top-level: {exc!r}", level="CRITICAL")
        raise
    finally:
        _wlog("Wrapper process terminated.")
