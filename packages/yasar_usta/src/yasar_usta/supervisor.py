"""TargetSupervisor — supervises exactly one process target. No telegram
poller, no lock, no signal handlers (all owned by Hub)."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Awaitable, Callable

from .backoff import BackoffTracker
from .config import GuardConfig
from .remote import find_claude_cmd, list_sessions, start_claude_remote
from .sidecar import SidecarManager
from .subprocess_mgr import SubprocessManager

logger = logging.getLogger("yasar_usta.supervisor")


def _extract_traceback(lines) -> str:
    """Extract Python traceback from stderr lines, skipping debug noise.

    Scans backwards for the last 'Traceback' block. Falls back to
    the last few non-empty lines if no traceback is found.
    """
    items = list(lines)
    if not items:
        return "(no output)"

    # Find the last traceback start
    tb_start = None
    for i in range(len(items) - 1, -1, -1):
        if items[i].startswith("Traceback"):
            tb_start = i
            break

    if tb_start is not None:
        return "\n".join(items[tb_start:])

    # No traceback — return last non-empty lines
    tail = [l for l in items if l.strip()][-20:]
    return "\n".join(tail) or "(no output)"


class TargetSupervisor:
    def __init__(self, project_id: str, config: GuardConfig,
                 notify: Callable[..., Awaitable[None]],
                 reply_keyboard: dict | None = None):
        self.project_id = project_id
        self.cfg = config
        self.msgs = config.messages
        self.notify = notify
        self.reply_keyboard = reply_keyboard

        self.subprocess = SubprocessManager(
            command=config.command, log_dir=config.log_dir, cwd=config.cwd,
            stop_timeout=config.stop_timeout, heartbeat_file=config.heartbeat_file,
            heartbeat_stale_seconds=config.heartbeat_stale_seconds,
            env=config.env,
        )
        self.backoff = BackoffTracker(steps=config.backoff_steps,
                                      reset_after=config.backoff_reset_after)
        self.sidecars: dict[str, SidecarManager] = {}
        for sc in config.sidecars:
            if sc.command:
                self.sidecars[sc.name] = SidecarManager(
                    name=sc.name, command=sc.command, pid_file=sc.pid_file,
                    health_url=sc.health_url, health_timeout=sc.health_timeout,
                    log_file=str(Path(config.log_dir) / f"{sc.name}.log"),
                    cwd=config.cwd, detached=sc.detached, env=config.env,
                )
        self._claude_cmd = find_claude_cmd(config.claude_cmd) if config.claude_enabled else None
        self._claude_session_dir = str(Path(config.log_dir) / "claude_sessions")

        self._signal_watcher: asyncio.Task | None = None
        self._shutdown = False
        self._restart_requested = False
        self._stop_requested = False

    # ── Intent API (called by Hub; Hub never touches self.subprocess directly)
    def request_restart(self) -> None:
        self._restart_requested = True
        self._write_shutdown_signal("restart")

    def request_stop(self) -> None:
        self._stop_requested = True
        self._write_shutdown_signal("stop")

    async def request_start(self) -> None:
        if not self.subprocess.running:
            await self._start_app()
            if self.subprocess.running:
                self.backoff.mark_started()
                await self._notify_started()
                await self._start_signal_watcher()

    async def do_restart_now(self) -> None:
        if self.subprocess.running:
            await self.subprocess.stop()

    async def do_stop_now(self) -> None:
        if self.subprocess.running:
            await self.subprocess.stop()

    async def kill_now(self) -> None:
        proc = self.subprocess.process
        if proc and proc.returncode is None:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
        self.subprocess.process = None
        self.subprocess.running = False

    def request_shutdown(self) -> None:
        self._shutdown = True

    @property
    def is_running(self) -> bool:
        return self.subprocess.running

    def status(self) -> dict:
        return {
            "project_id": self.project_id,
            "name": self.cfg.name,
            "app_name": self.cfg.app_name,
            "running": self.subprocess.running,
            "heartbeat_age": self.subprocess.heartbeat_age(),
            "heartbeat_healthy_seconds": self.cfg.heartbeat_healthy_seconds,
            "total_crashes": self.backoff.total_crashes,
            "sidecars": {name: {"name": name, "pid": sc.pid_alive()}
                         for name, sc in self.sidecars.items()},
            "extra_processes": self.cfg.extra_processes,
        }

    # ── Shutdown signal ────────────────────────────────────────────────

    def _write_shutdown_signal(self, intent: str) -> None:
        """Write a shutdown signal file for the managed app to read.

        The app checks for this file in its main loop and shuts down
        gracefully — much more reliable than OS signals on Windows.
        """
        try:
            signal_path = Path(self.cfg.log_dir) / "shutdown.signal"
            signal_path.parent.mkdir(parents=True, exist_ok=True)
            signal_path.write_text(intent, encoding="utf-8")
            logger.info("Wrote shutdown signal: %s", intent)
        except Exception as e:
            logger.error("Failed to write shutdown signal: %s", e)

    # ── Notifications ─────────────────────────────────────────────────

    async def _send_start_prompt(self, reason: str = "") -> None:
        if reason:
            msg = self.msgs.down_with_reason.format(
                reason=reason, app_name=self.cfg.app_name)
        else:
            msg = self.msgs.down_prompt.format(app_name=self.cfg.app_name)
        await self.notify(msg, reply_markup=self.reply_keyboard)

    async def _notify_crash(self, exit_code: int) -> None:
        stderr = _extract_traceback(self.subprocess.stderr_tail)
        if len(stderr) > 1500:
            stderr = stderr[-1500:]
        msg = self.msgs.crash.format(
            app_name=self.cfg.app_name,
            exit_code=exit_code,
            crash_count=self.backoff.total_crashes,
            backoff=self.backoff.get_delay(),
            stderr=stderr,
        )
        await self.notify(msg, reply_markup=self.reply_keyboard)

    async def _notify_stopped(self) -> None:
        await self.notify(
            self.msgs.stopped.format(app_name=self.cfg.app_name),
            reply_markup=self.reply_keyboard,
        )

    async def _notify_started(self) -> None:
        await self.notify(self.msgs.started.format(app_name=self.cfg.app_name))

    # ── Claude remote ─────────────────────────────────────────────────

    async def _handle_remote(self) -> None:
        if not self._claude_cmd:
            await self.notify(self.msgs.remote_not_found)
            return

        # Report any existing live sessions
        alive = list_sessions(self._claude_session_dir)
        if alive:
            lines = ["🖥️ *Active Claude sessions:*"]
            for pid, url in alive:
                if url:
                    lines.append(f"  • PID `{pid}` — [Connect]({url})")
                else:
                    lines.append(f"  • PID `{pid}` (no URL)")
            lines.append("\nStarting a new session...")
            await self.notify("\n".join(lines))
        else:
            await self.notify(self.msgs.remote_starting)

        pid, url = await start_claude_remote(
            self._claude_cmd,
            name=self.cfg.claude_name or self.cfg.app_name,
            cwd=self.cfg.cwd,
            session_dir=self._claude_session_dir,
        )
        if pid is None:
            await self.notify(self.msgs.remote_failed.format(error=url or "process failed to start"))
        elif url:
            await self.notify(self.msgs.remote_started.format(url=url, pid=pid))
        else:
            await self.notify(self.msgs.remote_started_no_url.format(pid=pid))

    # ── Signal file watcher ───────────────────────────────────────────

    async def _start_signal_watcher(self) -> None:
        if self._signal_watcher:
            return
        # Watcher runs if we have a signal file to watch OR a sidecar to monitor
        if not self.cfg.claude_signal_file and not self.sidecars:
            return
        self._signal_watcher = asyncio.create_task(self._signal_watch_loop())

    async def _stop_signal_watcher(self) -> None:
        if self._signal_watcher:
            self._signal_watcher.cancel()
            try:
                await self._signal_watcher
            except asyncio.CancelledError:
                pass
            self._signal_watcher = None

    async def _signal_watch_loop(self) -> None:
        signal_file = Path(self.cfg.claude_signal_file) if self.cfg.claude_signal_file else None
        sidecar_check_counter = 0
        while True:
            try:
                await asyncio.sleep(3)
                # Check for Claude remote signal
                if signal_file and signal_file.exists():
                    signal_file.unlink()
                    await self._handle_remote()
                # Check sidecar health every ~30s (10 iterations * 3s)
                sidecar_check_counter += 1
                if self.sidecars and sidecar_check_counter >= 10:
                    sidecar_check_counter = 0
                    for sc in self.sidecars.values():
                        await sc.ensure()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("Signal watcher error: %s", e)
                await asyncio.sleep(10)

    async def _start_app(self) -> None:
        """Start the managed app subprocess."""
        await self.subprocess.start()

    # ── Main loop ─────────────────────────────────────────────────────

    async def run(self) -> None:
        # Clear a stale claude signal file so the watcher doesn't fire on boot
        # (ported from guard.py:621-625).
        if self.cfg.claude_signal_file:
            _sf = Path(self.cfg.claude_signal_file)
            if _sf.exists():
                _sf.unlink()
        # Start this target's sidecars BEFORE the app (review finding #6):
        for sc in self.sidecars.values():
            if sc.command:
                await sc.start()
        # Initial app start (mirrors guard.py:645-652):
        await self._start_app()
        if self.subprocess.running:
            self.backoff.mark_started()
            await self._notify_started()
            await self._start_signal_watcher()
        else:
            logger.info("%s: initial start failed — waiting for start command",
                        self.project_id)

        while not self._shutdown:
            try:
                exit_code = await self.subprocess.wait_for_exit()

                if self.sidecars:
                    for sc in self.sidecars.values():
                        try:
                            await sc.ensure()
                        except Exception as e:
                            logger.warning("Sidecar %s ensure failed: %s", sc.name, e)
                await self._stop_signal_watcher()
                if self.cfg.on_exit:
                    self.cfg.on_exit(exit_code)
                self.backoff.maybe_reset()

                if exit_code == -1:
                    if (self.backoff.last_crash_time
                            and (time.time() - self.backoff.last_crash_time) < 10):
                        logger.info("No process — waiting for /start command via Telegram")
                        while not self._shutdown and not self.subprocess.running:
                            await asyncio.sleep(1)
                        if self.subprocess.running:
                            self.backoff.mark_started()
                            await self._notify_started()
                            await self._start_signal_watcher()
                        continue

                    self.backoff.record_crash()
                    logger.error("App hung — restarting after 5s")
                    await self.notify(self.msgs.hung.format(
                        app_name=self.cfg.app_name, delay=5))
                    for _ in range(5):
                        if self._shutdown or self.subprocess.running:
                            break
                        await asyncio.sleep(1)
                    if not self.subprocess.running and not self._shutdown:
                        await self._start_app()
                        if self.subprocess.running:
                            self.backoff.mark_started()
                            await self._notify_started()
                            await self._start_signal_watcher()
                    continue

                logger.info("App exited with code %d", exit_code)

                # Check intent flags set by the Telegram poller
                if self._restart_requested:
                    self._restart_requested = False
                    logger.info("Restart requested via Telegram — restarting app")
                    await asyncio.sleep(1)  # brief pause before restart
                    await self._start_app()
                    if self.subprocess.running:
                        self.backoff.mark_started()
                        await self._notify_started()
                        await self._start_signal_watcher()
                    continue

                if self._stop_requested:
                    self._stop_requested = False
                    logger.info("Stop requested via Telegram — app stopped cleanly")
                    # Already notified in poller; just wait for /start
                    while not self._shutdown and not self.subprocess.running:
                        await asyncio.sleep(1)
                    if self.subprocess.running:
                        self.backoff.mark_started()
                        await self._notify_started()
                        await self._start_signal_watcher()
                    continue

                if exit_code == self.cfg.restart_exit_code:
                    await self.notify(self.msgs.restarting.format(app_name=self.cfg.app_name))
                    await asyncio.sleep(1)  # brief pause before restart
                    await self._start_app()
                    if self.subprocess.running:
                        self.backoff.mark_started()
                        await self._notify_started()
                        await self._start_signal_watcher()
                    continue

                elif exit_code == 0:
                    logger.info("App stopped cleanly")
                    await self._notify_stopped()
                    while not self._shutdown and not self.subprocess.running:
                        await asyncio.sleep(1)
                    if self.subprocess.running:
                        self.backoff.mark_started()
                        await self._notify_started()
                        await self._start_signal_watcher()
                    continue

                else:
                    self.backoff.record_crash()
                    backoff_delay = self.backoff.get_delay()
                    logger.error(
                        "App crashed (exit %d), crash #%d, backoff %ds",
                        exit_code, self.backoff.total_crashes, backoff_delay,
                    )
                    await self._notify_crash(exit_code)

                    if not self.cfg.auto_restart:
                        while not self._shutdown and not self.subprocess.running:
                            await asyncio.sleep(1)
                        continue

                    for _ in range(backoff_delay):
                        if self._shutdown or self.subprocess.running:
                            break
                        await asyncio.sleep(1)

                    if not self.subprocess.running and not self._shutdown:
                        await self._start_app()
                        if self.subprocess.running:
                            self.backoff.mark_started()
                            await self._notify_started()
                            await self._start_signal_watcher()

            except asyncio.CancelledError:
                logger.info("Main loop cancelled")
                break
            except Exception as exc:
                logger.critical("UNHANDLED ERROR: %s", exc, exc_info=True)
                try:
                    await self.notify(self.msgs.wrapper_error.format(error=repr(exc)))
                except Exception:
                    pass
                while not self._shutdown and not self.subprocess.running:
                    await asyncio.sleep(5)

        # Shutdown: stop subprocess + watcher (poller/lock are Hub-owned).
        if self.subprocess.running:
            await self.subprocess.stop()
        await self._stop_signal_watcher()
        logger.info("%s: supervisor exiting.", self.project_id)
