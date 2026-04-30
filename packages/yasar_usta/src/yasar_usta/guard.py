"""ProcessGuard — the main orchestrator class."""

from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import os
import shutil
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .backoff import BackoffTracker
from .commands import build_start_keyboard, build_status_inline_keyboard, format_log_entries
from .config import GuardConfig
from .lock import acquire_lock, release_lock
from .remote import find_claude_cmd, list_sessions, start_claude_remote
from .sidecar import SidecarManager
from .status import build_status_text
from .subprocess_mgr import SubprocessManager
from .telegram import TelegramAPI

logger = logging.getLogger("yasar_usta")


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

        # Sidecars
        self.sidecars: dict[str, SidecarManager] = {}
        for sc in config.sidecars:
            if sc.command:
                self.sidecars[sc.name] = SidecarManager(
                    name=sc.name,
                    command=sc.command,
                    pid_file=sc.pid_file,
                    health_url=sc.health_url,
                    health_timeout=sc.health_timeout,
                    log_file=str(Path(config.log_dir) / f"{sc.name}.log"),
                    cwd=config.cwd,
                    detached=sc.detached,
                )

        # Claude remote — sessions tracked in a directory, survive restarts
        self._claude_cmd = find_claude_cmd(config.claude_cmd) if config.claude_enabled else None
        self._claude_session_dir = str(Path(config.log_dir) / "claude_sessions")

        # State
        self._telegram_poller: asyncio.Task | None = None
        self._signal_watcher: asyncio.Task | None = None
        self._shutdown = False
        self._restart_requested = False
        self._stop_requested = False
        self._guard_start_time = time.time()

    # ── Shutdown signal ────────────────────────────────────────────────

    def _write_shutdown_signal(self, intent: str) -> None:
        """Write a shutdown signal file for the managed app to read.

        The app checks for this file in its main loop and shuts down
        gracefully — much more reliable than OS signals on Windows.
        """
        try:
            signal_path = Path(self.cfg.log_dir) / "shutdown.signal"
            signal_path.write_text(intent, encoding="utf-8")
            logger.info("Wrote shutdown signal: %s", intent)
        except Exception as e:
            logger.error("Failed to write shutdown signal: %s", e)

    # ── Telegram helpers ──────────────────────────────────────────────

    def _kb(self) -> dict:
        return build_start_keyboard(self.msgs, app_name=self.cfg.app_name)

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
        try:
            sidecar_infos = []
            for name, sc in self.sidecars.items():
                sidecar_infos.append({
                    "name": name,
                    "alive": await sc.is_alive(),
                    "pid": sc.pid_alive(),
                    "health_url": sc.health_url,
                    "http_alive": await sc.http_alive(),
                })

            # Derive script basenames for duplicate detection
            _guard_script = Path(sys.argv[0]).name if sys.argv else None
            _app_script = (
                Path(self.cfg.command[-1]).name
                if self.cfg.command else None
            )

            text = build_status_text(
                name=self.cfg.name,
                app_name=self.cfg.app_name,
                guard_start_time=self._guard_start_time,
                app_running=self.subprocess.running,
                heartbeat_age=self.subprocess.heartbeat_age(),
                heartbeat_healthy_seconds=self.cfg.heartbeat_healthy_seconds,
                total_crashes=self.backoff.total_crashes,
                sidecar_infos=sidecar_infos,
                extra_processes=self.cfg.extra_processes,
                guard_script=_guard_script,
                app_script=_app_script,
            )
            sidecar_names = list(self.sidecars.keys()) if self.sidecars else []
            inline_kb = build_status_inline_keyboard(
                self.msgs, self.cfg.name, sidecar_names=sidecar_names)

            if edit_message_id:
                result = await self.telegram.edit(
                    edit_message_id, text, reply_markup=inline_kb)
                # Markdown fail → retry as plain text
                if result and not result.get("ok"):
                    await self.telegram.edit(
                        edit_message_id, text, reply_markup=inline_kb,
                        parse_mode=None)
            else:
                result = await self.telegram.send(text, reply_markup=inline_kb)
                # Markdown fail → retry as plain text
                if result and not result.get("ok"):
                    await self.telegram.send(
                        text, reply_markup=inline_kb, parse_mode=None)
        except Exception as e:
            logger.error("Status panel failed: %s", e)
            await self._send(f"⚠️ Status panel error: {e}")

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

        yazbunu = self.sidecars.get("yazbunu")
        if yazbunu and yazbunu.health_url:
            if await yazbunu.http_alive():
                url = yazbunu.health_url.replace("/health", "/")
                formatted += f"\n\n📊 [Yazbunu Log Viewer]({url})"
        await self._send(formatted)

    # ── Claude remote ─────────────────────────────────────────────────

    async def _handle_remote(self) -> None:
        if not self._claude_cmd:
            await self._send(self.msgs.remote_not_found)
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
            await self._send("\n".join(lines))
        else:
            await self._send(self.msgs.remote_starting)

        pid, url = await start_claude_remote(
            self._claude_cmd,
            name=self.cfg.claude_name or self.cfg.app_name,
            cwd=self.cfg.cwd,
            session_dir=self._claude_session_dir,
        )
        if pid is None:
            await self._send(self.msgs.remote_failed.format(error=url or "process failed to start"))
        elif url:
            await self._send(self.msgs.remote_started.format(url=url, pid=pid))
        else:
            await self._send(self.msgs.remote_started_no_url.format(pid=pid))

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

    # ── Telegram poller ───────────────────────────────────────────────

    async def _start_telegram_poller(self, initial_offset: int = 0) -> None:
        if self._telegram_poller or not self.telegram.enabled:
            return
        self._telegram_poller = asyncio.create_task(
            self._telegram_poll_loop(initial_offset))

    async def _stop_telegram_poller(self) -> None:
        if self._telegram_poller:
            self._telegram_poller.cancel()
            try:
                await self._telegram_poller
            except asyncio.CancelledError:
                pass
            self._telegram_poller = None
        await self.telegram.close()

    async def _telegram_poll_loop(self, initial_offset: int = 0) -> None:
        offset = initial_offset
        last_down_reply: float = 0
        DOWN_REPLY_COOLDOWN = 30
        _poll_fail_count = 0
        logger.info("Telegram poller started")

        while True:
            try:
                updates = await self.telegram.get_updates(offset=offset)
                if _poll_fail_count > 0:
                    logger.info("Telegram poll recovered after %d failures", _poll_fail_count)
                    _poll_fail_count = 0
                if not updates:
                    continue

                max_uid = max(u["update_id"] for u in updates)
                offset = max_uid + 1  # advance BEFORE processing

                for update in updates:
                    uid = update["update_id"]

                    # Callback queries
                    cb = update.get("callback_query")
                    if cb:
                        cb_chat = str(cb.get("message", {}).get("chat", {}).get("id", ""))
                        if cb_chat == str(self.cfg.telegram_chat_id):
                            cb_data = cb.get("data", "")
                            cb_msg_id = cb.get("message", {}).get("message_id")
                            if cb_data in ("restart_guard", "restart_usta"):
                                logger.warning("restart_guard triggered by callback uid=%s", uid)
                                await self.telegram.answer_callback(cb["id"])
                                await self.telegram.delete(cb_msg_id)
                                await self._send("♻️ *Yaşar Usta yeniden başlatılıyor...*")
                                await self._restart_self()
                                return
                            elif cb_data in ("guard_refresh", "usta_refresh"):
                                await self.telegram.answer_callback(cb["id"])
                                await self._send_status(edit_message_id=cb_msg_id)
                            elif cb_data.startswith("restart_sidecar:"):
                                sc_name = cb_data.split(":", 1)[1]
                                sc = self.sidecars.get(sc_name)
                                if sc:
                                    await self.telegram.answer_callback(cb["id"])
                                    await self.telegram.delete(cb_msg_id)
                                    await self._send(f"📊 *{sc_name}* yeniden başlatılıyor...")
                                    await sc.stop()
                                    await sc.start()
                                    await self._send(f"✅ *{sc_name}* yeniden başlatıldı")
                                else:
                                    await self.telegram.answer_callback(cb["id"])
                            elif cb_data in ("restart_sidecar", "restart_yazbunu"):
                                # Legacy compat for old button payloads
                                sc = self.sidecars.get("yazbunu") or (
                                    next(iter(self.sidecars.values()), None) if self.sidecars else None)
                                if sc:
                                    await self.telegram.answer_callback(cb["id"])
                                    await self.telegram.delete(cb_msg_id)
                                    await self._send(f"📊 *{sc.name}* yeniden başlatılıyor...")
                                    await sc.stop()
                                    await sc.start()
                                    await self._send(f"✅ *{sc.name}* yeniden başlatıldı")
                                else:
                                    await self.telegram.answer_callback(cb["id"])
                            elif cb_data == "confirm_restart":
                                _app = self.cfg.app_name
                                await self.telegram.answer_callback(cb["id"])
                                await self.telegram.delete(cb_msg_id)
                                await self._send(f"♻️ *{_app} yeniden başlatılıyor...*")
                                self._restart_requested = True
                                self._write_shutdown_signal("restart")
                                await self.subprocess.stop()
                            elif cb_data == "confirm_stop":
                                _app = self.cfg.app_name
                                await self.telegram.answer_callback(cb["id"])
                                await self.telegram.delete(cb_msg_id)
                                await self._send(f"⏹ *{_app} durduruluyor...*")
                                self._stop_requested = True
                                self._write_shutdown_signal("stop")
                                await self.subprocess.stop()
                            elif cb_data == "confirm_cancel":
                                await self.telegram.answer_callback(cb["id"])
                                await self.telegram.delete(cb_msg_id)
                                await self._send("👍 İptal edildi.")
                            else:
                                await self.telegram.answer_callback(cb["id"])
                        continue

                    msg = update.get("message", {})
                    text = msg.get("text", "")
                    chat_id = str(msg.get("chat", {}).get("id", ""))

                    if chat_id != str(self.cfg.telegram_chat_id):
                        continue

                    # Button labels (formatted with app_name)
                    _app = self.cfg.app_name
                    _btn_start = self.msgs.btn_start.format(app_name=_app)
                    _btn_restart = self.msgs.btn_restart.format(app_name=_app)
                    _btn_stop = self.msgs.btn_stop.format(app_name=_app)

                    # Built-in commands
                    if (text == _btn_start
                            or text.startswith("/start")
                            or text.startswith("/kutai_start")):
                        if not self.subprocess.running:
                            await self._send(self.msgs.starting.format(app_name=_app))
                            await self._start_app()
                        else:
                            await self._send_status()

                    elif (text == self.msgs.btn_status
                          or text.startswith("/status")
                          or text.startswith("/kutai_status")):
                        await self._send_status()

                    elif (text == _btn_restart
                          or (text.startswith("/restart")
                              and not text.startswith("/restart_guard")
                              and not text.startswith("/restart_usta"))):
                        if self.subprocess.running:
                            await self.telegram.send(
                                f"🔄 *{_app} yeniden başlatılsın mı?*",
                                reply_markup={"inline_keyboard": [[
                                    {"text": "🔄 Evet", "callback_data": "confirm_restart"},
                                    {"text": "❌ Vazgeç", "callback_data": "confirm_cancel"},
                                ]]},
                            )
                        else:
                            await self._send(self.msgs.starting.format(app_name=_app))
                            await self._start_app()

                    elif (text == _btn_stop
                          or text.startswith("/stop")):
                        if self.subprocess.running:
                            await self.telegram.send(
                                f"⚠️ *{_app} durdurulsun mu?*\n"
                                "Manuel olarak yeniden başlatmanız gerekecek.",
                                reply_markup={"inline_keyboard": [[
                                    {"text": "⏹ Evet, durdur", "callback_data": "confirm_stop"},
                                    {"text": "❌ Vazgeç", "callback_data": "confirm_cancel"},
                                ]]},
                            )
                        else:
                            await self._send_status()

                    elif (text.startswith("/restart_guard")
                          or text.startswith("/restart_usta")):
                        logger.warning("restart_guard triggered by text=%r uid=%s", text, uid)
                        await self._restart_self()
                        return

                    elif (text == self.msgs.btn_logs
                          or text.startswith("/logs")):
                        await self._send_logs(text)

                    elif (text == self.msgs.btn_remote
                          or text.startswith("/remote")):
                        # _handle_remote awaits start_claude_remote which
                        # polls the session log for up to 30s waiting for
                        # the URL. Awaiting here would block the Telegram
                        # message-poll loop for that window — observed
                        # 2026-04-30: user pressed "🔧 Durum" mid-launch
                        # and got no reply because getUpdates wasn't
                        # firing. Spawn as background task instead.
                        asyncio.create_task(self._handle_remote())

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

                    elif text and not self.subprocess.running:
                        now = time.time()
                        if now - last_down_reply > DOWN_REPLY_COOLDOWN:
                            last_down_reply = now
                            await self._send_start_prompt(
                                self.msgs.down_reply.format(app_name=self.cfg.app_name))

            except asyncio.CancelledError:
                return
            except Exception as e:
                _poll_fail_count += 1
                if _poll_fail_count <= 3 or _poll_fail_count % 60 == 0:
                    logger.error("Telegram poll error (fail #%d): %s",
                                 _poll_fail_count, e)
                await asyncio.sleep(5)

    async def _start_app(self) -> None:
        """Start the managed app subprocess."""
        await self.subprocess.start()

    async def _restart_self(self) -> None:
        """Restart the guard process itself."""
        logger.info("Self-restart requested")
        await self._send(self.msgs.self_restarting.format(name=self.cfg.name))

        # Set shutdown BEFORE stopping subprocess so the main loop
        # breaks out instead of sending a spurious crash notification.
        self._shutdown = True
        if self.subprocess.running:
            await self.subprocess.stop()
        await self._stop_telegram_poller()
        # Flush pending updates so the new guard doesn't reprocess
        # the same callback that triggered this restart.
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

        # Set up file logging for the guard itself (rotating JSONL)
        class _JsonlFmt(logging.Formatter):
            def format(self, record):
                return json.dumps({
                    "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "level": record.levelname,
                    "src": record.name,
                    "msg": record.getMessage(),
                }, ensure_ascii=False)

        def _safe_rotator(source: str, dest: str) -> None:
            for attempt in range(5):
                try:
                    if os.path.exists(dest):
                        os.remove(dest)
                    os.rename(source, dest)
                    return
                except PermissionError:
                    time.sleep(0.1 * (attempt + 1))
            try:
                shutil.copy2(source, dest)
                with open(source, "w"):
                    pass
            except Exception:
                pass

        fh = logging.handlers.RotatingFileHandler(
            str(log_dir / "wrapper_meta.jsonl"),
            maxBytes=10_000_000, backupCount=3, encoding="utf-8")
        fh.rotator = _safe_rotator
        fh.setFormatter(_JsonlFmt())
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

        acquire_lock(self.cfg.log_dir, name="guard")
        logger.info("%s started (auto_restart=%s)", self.cfg.name, self.cfg.auto_restart)

        # Clean up stale signal files
        if self.cfg.claude_signal_file:
            sf = Path(self.cfg.claude_signal_file)
            if sf.exists():
                sf.unlink()

        # Flush stale Telegram updates from previous runs so old
        # callbacks (like restart_guard) don't trigger on startup.
        _flush_offset = await self.telegram.flush_updates()

        # Start sidecars
        for sc in self.sidecars.values():
            if sc.command:
                await sc.start()

        # Announce
        await self._send(
            self.msgs.announce.format(name=self.cfg.name, app_name=self.cfg.app_name),
            reply_markup=self._kb(),
        )

        # Always-on Telegram poller — runs for the entire lifetime of the guard
        await self._start_telegram_poller(initial_offset=_flush_offset)

        # Start app
        await self._start_app()
        if self.subprocess.running:
            self.backoff.mark_started()
            await self._notify_started()
            await self._start_signal_watcher()
        else:
            logger.info("Initial start failed — waiting for /start command via Telegram")

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
                    await self._send(self.msgs.hung.format(
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
                    await self._send(self.msgs.restarting.format(app_name=self.cfg.app_name))
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
                    await self._send(self.msgs.wrapper_error.format(error=repr(exc)))
                except Exception:
                    pass
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
