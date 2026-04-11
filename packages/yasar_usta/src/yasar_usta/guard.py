"""ProcessGuard — the main orchestrator class."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path

from .backoff import BackoffTracker
from .commands import build_start_keyboard, build_status_inline_keyboard, format_log_entries
from .config import GuardConfig, SidecarConfig
from .lock import acquire_lock, release_lock
from .remote import find_claude_cmd, start_claude_remote
from .sidecar import SidecarManager
from .status import build_status_text
from .subprocess_mgr import SubprocessManager
from .telegram import TelegramAPI

logger = logging.getLogger("yasar_usta")


class ProcessGuard:
    """Telegram-controlled process manager.

    Args:
        config: GuardConfig with all settings.
    """

    def __init__(self, config: GuardConfig):
        self.cfg = config
        self.msgs = config.messages

        self.telegram = TelegramAPI(config.telegram_token, config.telegram_chat_id)
        self.subprocess = SubprocessManager(
            command=config.command,
            log_dir=config.log_dir,
            cwd=config.cwd,
            stop_timeout=config.stop_timeout,
            heartbeat_file=config.heartbeat_file,
            heartbeat_stale_seconds=config.heartbeat_stale_seconds,
        )
        self.backoff = BackoffTracker(
            steps=config.backoff_steps,
            reset_after=config.backoff_reset_after,
        )

        # Sidecar
        self.sidecar: SidecarManager | None = None
        if config.sidecar and config.sidecar.command:
            sc = config.sidecar
            self.sidecar = SidecarManager(
                name=sc.name,
                command=sc.command,
                pid_file=sc.pid_file,
                health_url=sc.health_url,
                health_timeout=sc.health_timeout,
                log_file=str(Path(config.log_dir) / f"{sc.name}.log"),
                cwd=config.cwd,
                detached=sc.detached,
            )

        # Claude remote
        self._claude_cmd = find_claude_cmd(config.claude_cmd) if config.claude_enabled else None
        self._claude_process: asyncio.subprocess.Process | None = None

        # State
        self._telegram_poller: asyncio.Task | None = None
        self._signal_watcher: asyncio.Task | None = None
        self._shutdown = False
        self._guard_start_time = time.time()

    # ── Telegram helpers ──────────────────────────────────────────────

    def _kb(self) -> dict:
        return build_start_keyboard(self.msgs)

    async def _send(self, text: str, reply_markup: dict | None = None) -> None:
        await self.telegram.send(text, reply_markup=reply_markup)

    async def _send_start_prompt(self, reason: str = "") -> None:
        if reason:
            msg = self.msgs.down_with_reason.format(
                reason=reason, app_name=self.cfg.app_name)
        else:
            msg = self.msgs.down_prompt.format(app_name=self.cfg.app_name)
        await self._send(msg, reply_markup=self._kb())

    async def _notify_crash(self, exit_code: int) -> None:
        stderr = "\n".join(self.subprocess.stderr_tail) or "(no output)"
        if len(stderr) > 1500:
            stderr = stderr[-1500:]
        msg = self.msgs.crash.format(
            app_name=self.cfg.app_name,
            exit_code=exit_code,
            crash_count=self.backoff.total_crashes,
            backoff=self.backoff.get_delay(),
            stderr=stderr,
        )
        await self._send(msg, reply_markup=self._kb())

    async def _notify_stopped(self) -> None:
        await self._send(
            self.msgs.stopped.format(app_name=self.cfg.app_name),
            reply_markup=self._kb(),
        )

    async def _notify_started(self) -> None:
        await self._send(self.msgs.started.format(app_name=self.cfg.app_name))

    # ── Status panel ──────────────────────────────────────────────────

    async def _send_status(self, edit_message_id: int | None = None) -> None:
        sidecar_name = self.sidecar.name if self.sidecar else None
        sidecar_alive = await self.sidecar.is_alive() if self.sidecar else False
        sidecar_pid = self.sidecar.pid_alive() if self.sidecar else None
        sidecar_http = await self.sidecar.http_alive() if self.sidecar else False

        text = build_status_text(
            name=self.cfg.name,
            app_name=self.cfg.app_name,
            guard_start_time=self._guard_start_time,
            app_running=self.subprocess.running,
            heartbeat_age=self.subprocess.heartbeat_age(),
            heartbeat_healthy_seconds=self.cfg.heartbeat_healthy_seconds,
            total_crashes=self.backoff.total_crashes,
            sidecar_name=sidecar_name,
            sidecar_alive=sidecar_alive,
            sidecar_pid=sidecar_pid,
            sidecar_health_url=self.sidecar.health_url if self.sidecar else None,
            sidecar_http_alive=sidecar_http,
            extra_processes=self.cfg.extra_processes,
        )
        inline_kb = build_status_inline_keyboard(
            self.msgs, self.cfg.name, sidecar_name)

        if edit_message_id:
            await self.telegram.edit(edit_message_id, text, reply_markup=inline_kb)
        else:
            await self._send(
                self.msgs.status_title.format(name=self.cfg.name) + "panel:",
                reply_markup=self._kb(),
            )
            await self.telegram.send(text, reply_markup=inline_kb)

    # ── Logs ──────────────────────────────────────────────────────────

    async def _send_logs(self, text: str) -> None:
        parts = text.strip().split()
        n = 20
        if len(parts) > 1:
            try:
                n = min(int(parts[1]), 50)
            except ValueError:
                pass

        log_path = self.cfg.log_file or str(Path(self.cfg.log_dir) / "orchestrator.jsonl")
        formatted = format_log_entries(log_path, n)
        if formatted is None:
            await self._send(self.msgs.no_log_file)
            return

        if self.sidecar and self.sidecar.health_url:
            pid = self.sidecar.pid_alive()
            if pid:
                formatted += f"\n\n📊 Full viewer: {self.sidecar.health_url}"
        await self._send(formatted)

    # ── Claude remote ─────────────────────────────────────────────────

    async def _handle_remote(self) -> None:
        if not self._claude_cmd:
            await self._send(self.msgs.remote_not_found)
            return
        await self._send(self.msgs.remote_starting)
        proc, url = await start_claude_remote(
            self._claude_cmd,
            name=self.cfg.claude_name or self.cfg.app_name,
            cwd=self.cfg.cwd,
        )
        self._claude_process = proc
        if proc is None:
            await self._send(self.msgs.remote_failed.format(error="process failed to start"))
        elif url:
            await self._send(self.msgs.remote_started.format(url=url, pid=proc.pid))
        else:
            await self._send(self.msgs.remote_started_no_url.format(pid=proc.pid))

    # ── Signal file watcher ───────────────────────────────────────────

    async def _start_signal_watcher(self) -> None:
        if self._signal_watcher or not self.cfg.claude_signal_file:
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
        signal_file = Path(self.cfg.claude_signal_file)
        while True:
            try:
                await asyncio.sleep(3)
                if signal_file.exists():
                    signal_file.unlink()
                    await self._handle_remote()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("Signal watcher error: %s", e)
                await asyncio.sleep(10)

    # ── Telegram poller ───────────────────────────────────────────────

    async def _start_telegram_poller(self) -> None:
        if self._telegram_poller or not self.telegram.enabled:
            return
        self._telegram_poller = asyncio.create_task(self._telegram_poll_loop())

    async def _stop_telegram_poller(self) -> None:
        if self._telegram_poller:
            self._telegram_poller.cancel()
            try:
                await self._telegram_poller
            except asyncio.CancelledError:
                pass
            self._telegram_poller = None

    async def _telegram_poll_loop(self) -> None:
        offset = 0
        last_down_reply: float = 0
        DOWN_REPLY_COOLDOWN = 30
        logger.info("Telegram poller started")

        while True:
            try:
                updates = await self.telegram.get_updates(offset=offset)
                if not updates:
                    continue

                max_uid = 0
                for update in updates:
                    uid = update["update_id"]
                    if uid > max_uid:
                        max_uid = uid

                    # Callback queries
                    cb = update.get("callback_query")
                    if cb:
                        cb_chat = str(cb.get("message", {}).get("chat", {}).get("id", ""))
                        if cb_chat == str(self.cfg.telegram_chat_id):
                            await self.telegram.answer_callback(cb["id"])
                            cb_data = cb.get("data", "")
                            cb_msg_id = cb.get("message", {}).get("message_id")
                            if cb_data == "restart_guard":
                                offset = max_uid + 1
                                await self._restart_self()
                                return
                            elif cb_data == "guard_refresh":
                                await self._send_status(edit_message_id=cb_msg_id)
                            elif cb_data == "restart_sidecar" and self.sidecar:
                                await self.sidecar.stop()
                                await self.sidecar.start()
                                await self._send_status(edit_message_id=cb_msg_id)
                        continue

                    msg = update.get("message", {})
                    text = msg.get("text", "")
                    chat_id = str(msg.get("chat", {}).get("id", ""))

                    if chat_id != str(self.cfg.telegram_chat_id):
                        continue

                    # Built-in commands
                    if text == self.msgs.btn_start or text.startswith("/start"):
                        await self._send(self.msgs.starting.format(app_name=self.cfg.app_name))
                        await self._start_app_from_poller()
                        return

                    elif text == self.msgs.btn_status or text.startswith("/status"):
                        await self._send_status()

                    elif text.startswith("/restart_guard"):
                        await self._restart_self()
                        return

                    elif text.startswith("/logs"):
                        await self._send_logs(text)

                    elif text.startswith("/remote"):
                        await self._handle_remote()

                    elif text == self.msgs.btn_system:
                        if self.subprocess.process and self.subprocess.process.returncode is None:
                            logger.info("System tap — killing hung app")
                            try:
                                self.subprocess.process.kill()
                                await self.subprocess.process.wait()
                            except Exception:
                                pass
                            self.subprocess.process = None
                            self.subprocess.running = False
                        await self._send_start_prompt(
                            f"🔴 {self.cfg.app_name} not responding.")

                    # Extra commands
                    elif text in self.cfg.extra_commands:
                        handler = self.cfg.extra_commands[text]
                        result = handler(self)
                        if asyncio.iscoroutine(result):
                            await result

                    elif text:
                        now = time.time()
                        if now - last_down_reply > DOWN_REPLY_COOLDOWN:
                            last_down_reply = now
                            await self._send_start_prompt(
                                self.msgs.down_reply.format(app_name=self.cfg.app_name))

                offset = max_uid + 1

            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("Telegram poll error: %s", e)
                await asyncio.sleep(5)

    async def _start_app_from_poller(self) -> None:
        """Start the managed app from the Telegram poller context."""
        self._telegram_poller = None
        await asyncio.sleep(2)
        await self.subprocess.start()

    async def _restart_self(self) -> None:
        """Restart the guard process itself."""
        logger.info("Self-restart requested")
        await self._send(self.msgs.self_restarting.format(name=self.cfg.name))

        if self.subprocess.running:
            await self.subprocess.stop()
        await self._stop_telegram_poller()
        await self.telegram.flush_updates()

        import subprocess as _sp
        python = self.subprocess.command[0] if self.subprocess.command else sys.executable
        script = str(Path(sys.argv[0]).resolve())
        argv = [a for a in sys.argv[1:] if a != script]
        logger.info("Spawning new guard: %s %s", python, script)

        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = (
                _sp.CREATE_NEW_PROCESS_GROUP | _sp.DETACHED_PROCESS | _sp.CREATE_NO_WINDOW
            )
        _sp.Popen(
            [sys.executable, script] + argv,
            close_fds=True,
            cwd=self.cfg.cwd or str(Path(script).parent),
            **kwargs,
        )

        release_lock()
        logger.info("Old guard exiting for restart")
        os._exit(0)

    # ── Main loop ─────────────────────────────────────────────────────

    async def run(self) -> None:
        """Main guard loop."""
        log_dir = Path(self.cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        acquire_lock(self.cfg.log_dir, name="guard")
        logger.info("%s started (auto_restart=%s)", self.cfg.name, self.cfg.auto_restart)

        # Clean up stale signal files
        if self.cfg.claude_signal_file:
            sf = Path(self.cfg.claude_signal_file)
            if sf.exists():
                sf.unlink()

        # Start sidecar
        if self.sidecar and self.sidecar.command:
            await self.sidecar.start()

        # Announce
        await self._send(
            self.msgs.announce.format(name=self.cfg.name, app_name=self.cfg.app_name),
            reply_markup=self._kb(),
        )

        # Start app
        await self.subprocess.start()
        if self.subprocess.running:
            self.backoff.mark_started()
            await self._notify_started()
            await self._start_signal_watcher()
        else:
            logger.info("Initial start failed — entering Telegram poll mode")
            await self._start_telegram_poller()

        while not self._shutdown:
            try:
                exit_code = await self.subprocess.wait_for_exit()

                if self.sidecar:
                    await self.sidecar.ensure()
                await self._stop_signal_watcher()
                if self.cfg.on_exit:
                    self.cfg.on_exit(exit_code)
                self.backoff.maybe_reset()

                if exit_code == -1:
                    if (self.backoff.last_crash_time
                            and (time.time() - self.backoff.last_crash_time) < 10):
                        logger.info("No process — entering Telegram poll mode")
                        await self._start_telegram_poller()
                        while not self._shutdown and not self.subprocess.running:
                            await asyncio.sleep(1)
                        if self.subprocess.running:
                            self.backoff.mark_started()
                            await self._notify_started()
                            await self._start_signal_watcher()
                        continue

                    self.backoff.record_crash()
                    logger.error("App hung — restarting after 5s")
                    await self._send(self.msgs.hung.format(
                        app_name=self.cfg.app_name, delay=5))
                    await self._start_telegram_poller()
                    for _ in range(5):
                        if self._shutdown or self.subprocess.running:
                            break
                        await asyncio.sleep(1)
                    if not self.subprocess.running and not self._shutdown:
                        await self.subprocess.start()
                        if self.subprocess.running:
                            self.backoff.mark_started()
                            await self._notify_started()
                            await self._start_signal_watcher()
                    continue

                logger.info("App exited with code %d", exit_code)

                if exit_code == self.cfg.restart_exit_code:
                    await self._send(self.msgs.restarting.format(app_name=self.cfg.app_name))
                    await asyncio.sleep(1)
                    await self.subprocess.start()
                    if self.subprocess.running:
                        self.backoff.mark_started()
                        await self._notify_started()
                        await self._start_signal_watcher()
                    continue

                elif exit_code == 0:
                    logger.info("App stopped cleanly")
                    await self._notify_stopped()
                    await self._start_telegram_poller()
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
                        await self._start_telegram_poller()
                        while not self._shutdown and not self.subprocess.running:
                            await asyncio.sleep(1)
                        continue

                    await self._start_telegram_poller()
                    for _ in range(backoff_delay):
                        if self._shutdown or self.subprocess.running:
                            break
                        await asyncio.sleep(1)

                    if not self.subprocess.running and not self._shutdown:
                        await self.subprocess.start()
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
                    await self._send(self.msgs.wrapper_error.format(error=repr(exc)))
                except Exception:
                    pass
                await self._start_telegram_poller()
                while not self._shutdown and not self.subprocess.running:
                    await asyncio.sleep(5)

        # Shutdown
        if self.subprocess.running:
            await self.subprocess.stop()
        await self._stop_signal_watcher()
        await self._stop_telegram_poller()
        logger.info("Guard exiting.")

    def request_shutdown(self) -> None:
        """Request the guard to shut down."""
        self._shutdown = True
