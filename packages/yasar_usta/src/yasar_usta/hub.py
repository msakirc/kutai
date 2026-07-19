"""Hub — owns the shared Telegram poller, the single lock, N supervisors,
the dashboard, coordinated shutdown, and hub self-restart."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

from .commands import (build_dashboard_keyboard, build_hub_reply_keyboard,
                       format_log_entries)
from .config import HubConfig, ProjectConfig
from .hooks import load_hook, run_pre_boot
from .lock import acquire_lock, release_lock
from .singleton import (_win32_create_mutex, enforce_singleton,
                        release_singleton)
from .status import build_dashboard_text
from .supervisor import TargetSupervisor
from .telegram import TelegramAPI

logger = logging.getLogger("yasar_usta.hub")


class Hub:
    def __init__(self, hub_cfg: HubConfig, projects: list[ProjectConfig]):
        self.cfg = hub_cfg
        self.msgs = hub_cfg.messages
        self.projects = projects
        self.telegram = TelegramAPI(hub_cfg.telegram_token, hub_cfg.telegram_chat_id)
        self._guard_start_time = time.time()
        self._shutdown = False
        self._telegram_poller: asyncio.Task | None = None
        self._bg_tasks: set = set()  # strong refs to fire-and-forget tasks
        # Singleton seam (injectable for tests); real Win32 mutex by default.
        self._create_mutex = _win32_create_mutex
        self._singleton_exit = sys.exit

        # Persistent reply keyboard, built once from hub Messages (spec R4).
        self._reply_kb = build_hub_reply_keyboard(self.msgs)

        # One supervisor per target, keyed by a unique routing id. Single-target
        # project → routing id is the project id; multi-target → `${pid}:${tgt}`.
        self.supervisors: dict[str, TargetSupervisor] = {}
        self._hooks: dict[str, object] = {}  # loaded once, reused in run()
        for proj in projects:
            hook = load_hook(proj.hook_module)
            self._hooks[proj.id] = hook
            for tgt in proj.targets:
                if hook is not None and hasattr(hook, "on_exit"):
                    tgt.on_exit = hook.on_exit
                rid = proj.id if len(proj.targets) == 1 else f"{proj.id}:{tgt.name}"
                self.supervisors[rid] = TargetSupervisor(
                    rid, tgt, notify=self._notify, reply_keyboard=self._reply_kb)

    async def _notify(self, text: str, reply_markup: dict | None = None) -> None:
        result = await self.telegram.send(text, reply_markup=reply_markup)
        if result and not result.get("ok"):
            await self.telegram.send(text, reply_markup=reply_markup, parse_mode=None)

    # ── Single-instance gate (never-duplicates authority) ────────────────
    def _acquire_singleton(self) -> None:
        """Become the one hub, or exit. Runs before the file lock so no other
        cleanup/pre_boot executes on a mutex-loser (§4.1). Fail-closed."""
        enforce_singleton(
            "YasarUstaHub",
            state_dir=self.cfg.log_dir,
            create_mutex=self._create_mutex,
            alert=self._sync_alert,
            exit_fn=self._singleton_exit,
        )

    def _sync_alert(self, msg: str) -> None:
        """Best-effort BLOCKING Telegram post — usable before the async poller
        exists (the mutex gate runs pre-loop). Never raises."""
        token = getattr(self.cfg, "telegram_token", "")
        chat = getattr(self.cfg, "telegram_chat_id", "")
        if not token or not chat:
            logger.error("SINGLETON: %s", msg)
            return
        try:
            import urllib.parse
            import urllib.request
            data = urllib.parse.urlencode({"chat_id": chat, "text": msg}).encode()
            urllib.request.urlopen(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data=data, timeout=5)
        except Exception:
            logger.error("SINGLETON (alert send failed): %s", msg)

    def _kb(self) -> dict:
        return self._reply_kb

    def _resolve_bare_target(self) -> "TargetSupervisor | None":
        if len(self.supervisors) == 1:
            return next(iter(self.supervisors.values()))
        return None

    async def _sidecar_health(self, sc) -> dict:
        http = await sc.http_alive()
        pid = sc.pid_alive()
        return {"name": sc.name, "http_alive": http, "pid": pid,
                "alive": bool(http or pid)}

    # ── Dashboard ────────────────────────────────────────────────────────
    async def _send_dashboard(self, edit_message_id: int | None = None) -> None:
        # Gather sidecar health on the loop (aiohttp http_alive can't run in the
        # worker thread), THEN offload the blocking text render via to_thread.
        try:
            states = []
            for s in self.supervisors.values():
                st = s.status()
                st["sidecar_health"] = list(await asyncio.gather(
                    *[self._sidecar_health(sc) for sc in s.sidecars.values()]
                )) if s.sidecars else []
                states.append(st)
            text = await asyncio.to_thread(
                build_dashboard_text, self.cfg.name, states, self._guard_start_time)
            kb = build_dashboard_keyboard(states)
            if edit_message_id:
                result = await self.telegram.edit(edit_message_id, text, reply_markup=kb)
                if result and not result.get("ok"):
                    await self.telegram.edit(edit_message_id, text, reply_markup=kb, parse_mode=None)
            else:
                result = await self.telegram.send(text, reply_markup=kb)
                if result and not result.get("ok"):
                    await self.telegram.send(text, reply_markup=kb, parse_mode=None)
        except Exception as e:
            logger.error("dashboard failed: %s", e)
            try:
                await self.telegram.send(f"⚠️ Dashboard error: {e}")
            except Exception:
                pass

    # ── Callback routing ─────────────────────────────────────────────────
    async def _route_callback(self, cb_data: str, cb_msg_id) -> None:
        if cb_data == "dashboard_refresh":
            await self._send_dashboard(edit_message_id=cb_msg_id)
            return
        if cb_data == "restart_hub":
            await self._notify("♻️ *Hub yeniden başlatılıyor...*")
            await self._do_restart_hub()
            return
        if cb_data == "confirm_cancel":
            if cb_msg_id:
                await self.telegram.delete(cb_msg_id)
            return
        if cb_data.startswith("restart_sidecar:"):
            rest = cb_data[len("restart_sidecar:"):]
            pid, _, name = rest.rpartition(":")
            sup = self.supervisors.get(pid)
            sc = sup.sidecars.get(name) if sup else None
            if sc:
                await self._notify(f"📊 *{name}* yeniden başlatılıyor...")
                await sc.stop()
                await sc.start()
                await self._notify(f"✅ *{name}* yeniden başlatıldı")
            return
        if ":" not in cb_data:
            return
        verb, pid = cb_data.split(":", 1)
        sup = self.supervisors.get(pid)
        if not sup:
            return
        # restart/stop are semi-destructive → confirm first (review finding #5).
        if verb == "restart":
            await self._confirm(pid, sup.cfg.app_name, "restart",
                                f"🔄 *{sup.cfg.app_name} yeniden başlatılsın mı?*")
        elif verb == "stop":
            await self._confirm(pid, sup.cfg.app_name, "stop",
                                f"⚠️ *{sup.cfg.app_name} durdurulsun mu?*\n"
                                "Manuel olarak yeniden başlatmanız gerekecek.")
        elif verb == "confirm_restart":
            if cb_msg_id:
                await self.telegram.delete(cb_msg_id)
            await self._notify(f"♻️ *{sup.cfg.app_name} yeniden başlatılıyor...*")
            sup.request_restart()
            await sup.do_restart_now()
        elif verb == "confirm_stop":
            if cb_msg_id:
                await self.telegram.delete(cb_msg_id)
            await self._notify(f"⏹ *{sup.cfg.app_name} durduruluyor...*")
            sup.request_stop()
            await sup.do_stop_now()
        elif verb == "start":
            await self._notify(f"🚀 {sup.cfg.app_name} başlatılıyor...")
            await sup.request_start()
        elif verb == "kill":
            await sup.kill_now()
            await sup._send_start_prompt(f"🔴 {sup.cfg.app_name} not responding.")
        elif verb == "remote":
            t = asyncio.create_task(sup._handle_remote())
            self._bg_tasks.add(t)
            t.add_done_callback(self._bg_tasks.discard)
        elif verb == "logs":
            await self._send_logs_for(sup)

    async def _confirm(self, pid: str, app_name: str, action: str, prompt: str) -> None:
        """Send a Yes/Cancel dialog whose Yes carries confirm_{action}:{pid}."""
        await self.telegram.send(prompt, reply_markup={"inline_keyboard": [[
            {"text": "✅ Evet", "callback_data": f"confirm_{action}:{pid}"},
            {"text": "❌ Vazgeç", "callback_data": "confirm_cancel"},
        ]]})

    async def _send_logs_for(self, sup: TargetSupervisor, n: int = 20) -> None:
        log_path = sup.cfg.log_file or str(Path(sup.cfg.log_dir) / "orchestrator.jsonl")
        formatted = format_log_entries(log_path, n)
        if formatted is None:
            await self._notify("📋 No log entries.")
            return
        yaz = sup.sidecars.get("yazbunu")
        if yaz and getattr(yaz, "health_url", None) and await yaz.http_alive():
            url = yaz.health_url.replace("/health", "/")
            formatted += f"\n\n📊 [Yazbunu Log Viewer]({url})"
        await self._notify(formatted)

    # ── Hub self-restart (spec finding #1) ───────────────────────────────
    async def _do_restart_hub(self) -> None:
        self._shutdown = True
        for sup in self.supervisors.values():
            sup.request_shutdown()
            await sup.do_stop_now()
        await self._stop_poller()
        await self.telegram.flush_updates()
        import subprocess as _sp
        script = str(Path(sys.argv[0]).resolve())
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = (
                _sp.CREATE_NEW_PROCESS_GROUP | _sp.DETACHED_PROCESS | _sp.CREATE_NO_WINDOW)
        # Release the singleton mutex + file lock BEFORE re-spawning, so the
        # replacement hub acquires a free mutex instead of deadlocking on the
        # one we still hold (zero-hub window). The old process is already
        # quiesced (supervisors + poller stopped) so there is no double-work.
        release_singleton()
        release_lock()
        _sp.Popen([sys.executable, script] + sys.argv[1:], close_fds=True,
                  cwd=str(Path(script).parent), **kwargs)
        os._exit(0)

    async def _stop_poller(self) -> None:
        if self._telegram_poller:
            self._telegram_poller.cancel()
            try:
                await self._telegram_poller
            except asyncio.CancelledError:
                pass
            self._telegram_poller = None
        await self.telegram.close()

    def request_shutdown(self) -> None:
        self._shutdown = True
        for sup in self.supervisors.values():
            sup.request_shutdown()

    async def _shutdown_watcher(self) -> None:
        """When shutdown is requested, fan out a graceful stop to every
        supervisor so their run() loops wake and exit promptly (spec: Hub
        fans out stop(timeout) on shutdown). Otherwise this task idles until
        cancelled by run()."""
        while not self._shutdown:
            await asyncio.sleep(0.5)
        for sup in self.supervisors.values():
            try:
                await sup.do_stop_now()
            except Exception as e:
                logger.error("shutdown stop for %s failed: %s", sup.project_id, e)

    # ── Poll loop ────────────────────────────────────────────────────────
    async def _poll_loop(self, initial_offset: int = 0) -> None:
        offset = initial_offset
        fail = 0
        logger.info("Hub poller started")
        while True:
            try:
                updates = await self.telegram.get_updates(offset=offset)
                fail = 0
                if not updates:
                    continue
                offset = max(u["update_id"] for u in updates) + 1
                for update in updates:
                    cb = update.get("callback_query")
                    if cb:
                        chat = str(cb.get("message", {}).get("chat", {}).get("id", ""))
                        if chat == str(self.cfg.telegram_chat_id):
                            await self.telegram.answer_callback(cb["id"])
                            await self._route_callback(
                                cb.get("data", ""),
                                cb.get("message", {}).get("message_id"))
                        continue
                    msg = update.get("message", {})
                    text = msg.get("text", "")
                    chat_id = str(msg.get("chat", {}).get("id", ""))
                    if chat_id != str(self.cfg.telegram_chat_id):
                        continue
                    await self._route_text(text)
            except asyncio.CancelledError:
                return
            except Exception as e:
                fail += 1
                if fail <= 3 or fail % 60 == 0:
                    logger.error("Hub poll error (#%d): %s", fail, e)
                await asyncio.sleep(5)

    async def _route_text(self, text: str) -> None:
        # Hub-global: dashboard (slash OR the persistent "Status" button label)
        if text.startswith("/status") or text.startswith("/kutai_status") \
                or text == self.msgs.btn_status:
            await self._send_dashboard()
            return
        if text.startswith("/restart_hub") or text.startswith("/restart_usta") \
                or text.startswith("/restart_guard"):
            await self._notify("♻️ *Hub yeniden başlatılıyor...*")
            await self._do_restart_hub()
            return
        # Persistent reply-keyboard labels for Logs / Remote (review finding #1/#2)
        if text.startswith("/logs") or text == self.msgs.btn_logs:
            n = 20
            parts = text.split()
            if len(parts) > 1:
                try:
                    n = max(1, min(int(parts[1]), 50))
                except ValueError:
                    pass
            sup = self._resolve_bare_target()
            if sup is None:
                await self._notify("⚠️ Multiple projects — open /status and use the buttons.")
            else:
                await self._send_logs_for(sup, n)
            return
        if text.startswith("/remote") or text == self.msgs.btn_remote:
            await self._for_bare_target("remote")
            return
        # Bare per-target action verbs (start/restart/stop) — slash aliases only.
        if text.startswith("/kutai_start"):
            await self._for_bare_target("start")
            return
        for verb in ("start", "restart", "stop"):
            if text.startswith("/" + verb):
                await self._for_bare_target(verb)
                return
        if text.startswith("/"):
            await self._send_dashboard()

    async def _for_bare_target(self, verb: str) -> None:
        """Apply a per-target verb to the sole supervisor, or reject if N>1
        (spec R4: never guess a target)."""
        sup = self._resolve_bare_target()
        if sup is None:
            await self._notify(
                "⚠️ Multiple projects — open /status and use the buttons.")
            return
        await self._route_callback(f"{verb}:{sup.project_id}", None)

    # ── Run ──────────────────────────────────────────────────────────────
    async def run(self) -> None:
        Path(self.cfg.log_dir).mkdir(parents=True, exist_ok=True)
        # Mutex is the singleton authority — gate BEFORE the file lock and any
        # pre_boot cleanup, so a second hub exits before killing anything (§4.1).
        self._acquire_singleton()
        acquire_lock(self.cfg.log_dir, name="hub")
        logger.info("Hub started with %d supervisors", len(self.supervisors))

        # pre_boot hooks (per project, once, after lock — spec finding #4).
        for proj in self.projects:
            run_pre_boot(self._hooks.get(proj.id), proj)

        offset = await self.telegram.flush_updates()
        try:
            _announce = self.msgs.announce.format(
                name=self.cfg.name,
                app_name=", ".join(p.name for p in self.projects))
        except Exception:
            _announce = f"🔧 *{self.cfg.name}* — {len(self.supervisors)} target(s) starting..."
        await self._notify(_announce, reply_markup=self._kb())
        if self.telegram.enabled:
            self._telegram_poller = asyncio.create_task(self._poll_loop(offset))

        sup_tasks = [asyncio.create_task(s.run()) for s in self.supervisors.values()]
        watcher = asyncio.create_task(self._shutdown_watcher())
        try:
            # return_exceptions=True: a crashing supervisor must NOT propagate
            # out of Hub.run() and kill the hub while orchestrators run headless.
            results = await asyncio.gather(*sup_tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, BaseException) and not isinstance(r, asyncio.CancelledError):
                    logger.error("supervisor task crashed: %r", r)
        finally:
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass
            for t in list(self._bg_tasks):
                t.cancel()
            await self._stop_poller()
        logger.info("Hub exiting.")
