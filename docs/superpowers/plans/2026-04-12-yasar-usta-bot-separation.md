# Yaşar Usta Bot Separation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Yaşar Usta its own Telegram bot so it never fights with KutAI for updates, and remove all wrapper commands from KutAI's bot.

**Architecture:** Yaşar Usta gets a dedicated bot token (always polling). KutAI keeps its existing token (polls only when running). The wrapper's poller runs continuously — not just when KutAI is down. Wrapper controls restart/stop by signaling the subprocess + setting internal flags so the guard main loop knows the intent. KutAI's system menu shrinks to debug/DLQ/logs/reset.

**Tech Stack:** Python, aiohttp (wrapper Telegram), python-telegram-bot (KutAI), asyncio

---

### Task 1: Add New Bot Token to .env

**Files:**
- Modify: `.env`

- [ ] **Step 1: Add the Yaşar Usta bot token**

Add to `.env`:
```
YASAR_USTA_BOT_TOKEN=8624073383:AAGumWfCsRlM06Wg03XeiD9hU35-FmYJWUU
```

- [ ] **Step 2: Commit**

```bash
git add .env
git commit -m "feat(wrapper): add dedicated Yaşar Usta bot token"
```

---

### Task 2: Update kutai_wrapper.py to Use New Token

**Files:**
- Modify: `kutai_wrapper.py:78`

- [ ] **Step 1: Change telegram_token to read from YASAR_USTA_BOT_TOKEN**

In `kutai_wrapper.py` line 78, change:
```python
    telegram_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
```
to:
```python
    telegram_token=os.getenv("YASAR_USTA_BOT_TOKEN", ""),
```

- [ ] **Step 2: Test import**

Run: `PYTHONIOENCODING=utf-8 .venv/Scripts/python -c "import kutai_wrapper; print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add kutai_wrapper.py
git commit -m "feat(wrapper): use dedicated Yaşar Usta bot token"
```

---

### Task 3: Refactor Guard — Always-On Poller + Restart/Stop Commands

**Files:**
- Modify: `packages/yasar_usta/src/yasar_usta/guard.py`

The current design: poller runs only when KutAI is down, stops when KutAI starts. With separate bots, the poller should run continuously so the wrapper bot always responds.

**Key changes:**
1. Add `_restart_requested` and `_stop_requested` flags
2. Make poller always-on (run alongside signal watcher)
3. Add `/restart` and `/stop` commands to the poller
4. Poller sets flag + calls `subprocess.stop()` for restart/stop
5. Main loop checks flags before interpreting exit codes
6. Remove the start/stop poller dance from `_start_app()` and the main loop — poller is always on
7. Remove `flush_updates()` calls (no longer sharing bot)

- [ ] **Step 1: Add intent flags to `__init__`**

In `guard.py` `__init__`, after `self._shutdown = False` (line 73), add:
```python
        self._restart_requested = False
        self._stop_requested = False
```

- [ ] **Step 2: Refactor `_telegram_poll_loop` to be always-on**

Replace the entire `_telegram_poll_loop` method (lines 248-350) with:

```python
    async def _telegram_poll_loop(self) -> None:
        offset = 0
        last_down_reply: float = 0
        DOWN_REPLY_COOLDOWN = 30
        logger.info("Telegram poller started (dedicated bot)")

        while not self._shutdown:
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
                            if cb_data in ("restart_guard", "restart_usta"):
                                await self._restart_self()
                                return
                            elif cb_data in ("guard_refresh", "usta_refresh"):
                                await self._send_status(edit_message_id=cb_msg_id)
                            elif cb_data in ("restart_sidecar", "restart_yazbunu") and self.sidecar:
                                await self.sidecar.stop()
                                await self.sidecar.start()
                                await self._send_status(edit_message_id=cb_msg_id)
                        continue

                    msg = update.get("message", {})
                    text = msg.get("text", "")
                    chat_id = str(msg.get("chat", {}).get("id", ""))

                    if chat_id != str(self.cfg.telegram_chat_id):
                        continue

                    app_running = self.subprocess.running

                    # /start — only when app is down
                    if (text == self.msgs.btn_start
                            or text.startswith("/start")
                            or text.startswith("/kutai_start")):
                        if not app_running:
                            await self._send(self.msgs.starting.format(
                                app_name=self.cfg.app_name))
                            await self.subprocess.start()
                            if self.subprocess.running:
                                self.backoff.mark_started()
                                await self._notify_started()
                        else:
                            await self._send(f"💚 {self.cfg.app_name} zaten çalışıyor.")

                    # /status
                    elif (text == self.msgs.btn_status
                          or text.startswith("/status")
                          or text.startswith("/kutai_status")):
                        await self._send_status()

                    # /restart — request graceful restart
                    elif text.startswith("/restart") and not text.startswith("/restart_"):
                        if app_running:
                            self._restart_requested = True
                            await self._send(self.msgs.restarting.format(
                                app_name=self.cfg.app_name))
                            await self.subprocess.stop()
                        else:
                            await self._send(f"⚠️ {self.cfg.app_name} zaten kapalı. /start ile başlat.")

                    # /stop — request graceful stop
                    elif text.startswith("/stop"):
                        if app_running:
                            self._stop_requested = True
                            await self._send(f"⏹ {self.cfg.app_name} durduruluyor...")
                            await self.subprocess.stop()
                        else:
                            await self._send(f"⚠️ {self.cfg.app_name} zaten kapalı.")

                    # /restart_guard or /restart_usta — restart the wrapper itself
                    elif (text.startswith("/restart_guard")
                          or text.startswith("/restart_usta")):
                        await self._restart_self()
                        return

                    # /logs
                    elif text.startswith("/logs"):
                        await self._send_logs(text)

                    # /remote
                    elif text.startswith("/remote"):
                        await self._handle_remote()

                    # System button (kill hung app)
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

                    # Unknown message when app is down
                    elif text and not app_running:
                        now = time.time()
                        if now - last_down_reply > DOWN_REPLY_COOLDOWN:
                            last_down_reply = now
                            await self._send_start_prompt(
                                self.msgs.down_reply.format(
                                    app_name=self.cfg.app_name))

                offset = max_uid + 1

            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("Telegram poll error: %s", e)
                await asyncio.sleep(5)
```

- [ ] **Step 3: Refactor `_start_app` — no more poller dance**

Replace `_start_app` (lines 352-365) with:

```python
    async def _start_app(self) -> None:
        """Start the managed app subprocess."""
        await self.subprocess.start()
```

- [ ] **Step 4: Update main loop to use intent flags and always-on poller**

Replace the `run()` method (lines 401-545) with:

```python
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

        # Start always-on Telegram poller
        await self._start_telegram_poller()

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
            logger.info("Initial start failed — waiting for /start command")

        while not self._shutdown:
            try:
                exit_code = await self.subprocess.wait_for_exit()

                if self.sidecar:
                    await self.sidecar.ensure()
                await self._stop_signal_watcher()
                if self.cfg.on_exit:
                    self.cfg.on_exit(exit_code)
                self.backoff.maybe_reset()

                # Check intent flags (set by poller commands)
                if self._restart_requested:
                    self._restart_requested = False
                    logger.info("Restart requested via Telegram")
                    await self._send(self.msgs.restarting.format(
                        app_name=self.cfg.app_name))
                    await self._start_app()
                    if self.subprocess.running:
                        self.backoff.mark_started()
                        await self._notify_started()
                        await self._start_signal_watcher()
                    continue

                if self._stop_requested:
                    self._stop_requested = False
                    logger.info("Stop requested via Telegram")
                    await self._notify_stopped()
                    # Wait for /start command (poller is always on)
                    while not self._shutdown and not self.subprocess.running:
                        await asyncio.sleep(1)
                    if self.subprocess.running:
                        self.backoff.mark_started()
                        await self._notify_started()
                        await self._start_signal_watcher()
                    continue

                if exit_code == -1:
                    if (self.backoff.last_crash_time
                            and (time.time() - self.backoff.last_crash_time) < 10):
                        logger.info("No process — waiting for /start command")
                        while not self._shutdown and not self.subprocess.running:
                            await asyncio.sleep(1)
                        if self.subprocess.running:
                            self.backoff.mark_started()
                            await self._notify_started()
                            await self._start_signal_watcher()
                        continue
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

                if exit_code == self.cfg.restart_exit_code:
                    await self._send(self.msgs.restarting.format(
                        app_name=self.cfg.app_name))
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
```

- [ ] **Step 5: Remove `flush_updates` from `_restart_self`**

In `_restart_self()` (line 367+), remove the line:
```python
        await self.telegram.flush_updates()
```

No longer needed — the wrapper owns its update stream.

- [ ] **Step 6: Test import**

Run: `PYTHONIOENCODING=utf-8 .venv/Scripts/python -c "from yasar_usta import ProcessGuard; print('OK')"`
Expected: OK

- [ ] **Step 7: Commit**

```bash
git add packages/yasar_usta/src/yasar_usta/guard.py
git commit -m "feat(wrapper): always-on dedicated bot + /restart /stop commands"
```

---

### Task 4: Remove Wrapper Commands from KutAI's Telegram Bot

**Files:**
- Modify: `src/app/telegram_bot.py`

Remove all wrapper-related functionality from KutAI. This is a pure deletion task.

- [ ] **Step 1: Remove `_check_yazbunu_health` static method**

Delete lines 1167-1209 (the `_check_yazbunu_health` method).

- [ ] **Step 2: Remove `_build_proc_panel` method**

Delete the `_build_proc_panel` method (lines 1211-1320).

- [ ] **Step 3: Remove `_show_processes` method**

Delete the `_show_processes` method (lines 1322-1331).

- [ ] **Step 4: Remove `cmd_usta` handler**

Delete `cmd_usta` (lines 2409-2411).

- [ ] **Step 5: Remove `cmd_kutai_restart` and `cmd_kutai_stop` and `_do_kutai_stop`**

Delete:
- `cmd_kutai_restart` (lines 2413-2427)
- `cmd_kutai_stop` (lines 2429-2439)
- `_do_kutai_stop` (lines 2441-2448)

- [ ] **Step 6: Remove `cmd_claude` handler**

Delete `cmd_claude` (lines 2452-2456).

- [ ] **Step 7: Remove command handler registrations**

In `_setup_handlers()`, remove these lines:
```python
self.app.add_handler(CommandHandler("kutai_restart", self.cmd_kutai_restart))
self.app.add_handler(CommandHandler("usta", self.cmd_usta))
self.app.add_handler(CommandHandler("restart", self.cmd_kutai_restart))
self.app.add_handler(CommandHandler("kutai_stop", self.cmd_kutai_stop))
self.app.add_handler(CommandHandler("stop", self.cmd_kutai_stop))
self.app.add_handler(CommandHandler("claude", self.cmd_claude))
```

- [ ] **Step 8: Remove `m:proc:*` callback handlers**

In `handle_callback()`, remove the entire `if data.startswith("m:proc:"):` block (lines ~5350-5432).

- [ ] **Step 9: Remove `m:confirm:proc_*` callback handlers**

Remove:
- The `m:confirm:proc_kill_kutai_only` handler
- The `m:confirm:proc_kill_wrapper` / `m:confirm:proc_kill_kutai` handler
- The `m:confirm:restart` handler  
- The `m:confirm:stop` handler
- The `stop_confirm` / `stop_cancel` handlers

- [ ] **Step 10: Clean up system menu keyboard**

In `KB_SISTEM` (lines 162-167), change from:
```python
KB_SISTEM = ReplyKeyboardMarkup([
    ["🖥 Yük Modu", "🐛 Debug", "📭 DLQ", "📋 Loglar"],
    ["🖥️ Claude Code", "🔧 Yaşar Usta", "🗑 Reset Tasks", "☢️ Reset All"],
    ["🔄 Yeniden Başlat", "⏹ Durdur"],
    ["🔙 Geri"],
], resize_keyboard=True)
```
to:
```python
KB_SISTEM = ReplyKeyboardMarkup([
    ["🖥 Yük Modu", "🐛 Debug", "📭 DLQ", "📋 Loglar"],
    ["🗑 Reset Tasks", "☢️ Reset All"],
    ["🔙 Geri"],
], resize_keyboard=True)
```

- [ ] **Step 11: Remove wrapper entries from `_BUTTON_ACTIONS`**

Remove these entries:
```python
"🖥️ Claude Code": ("special", "claude_code"),
"🔧 Yaşar Usta": ("special", "processes"),
"🔄 Yeniden Başlat": ("special", "restart"),
"⏹ Durdur": ("special", "stop"),
"▶️ Başlat": ("special", "start_kutai"),
```

- [ ] **Step 12: Remove `KB_BASLAT` keyboard definition**

Remove the `KB_BASLAT` keyboard (lines 175-177). This was shown when wrapper needed to start KutAI — now handled by the wrapper's own bot.

Also remove any references to `KB_BASLAT` in the codebase (keyboard state switching, etc.).

- [ ] **Step 13: Remove wrapper-related special button handlers**

In `_handle_special_button()`, remove handlers for:
- `"restart"` 
- `"stop"`
- `"claude_code"`
- `"processes"`
- `"start_kutai"`

- [ ] **Step 14: Remove `yasar_usta` import from telegram_bot.py**

Remove the `from yasar_usta import EXIT_RESTART, EXIT_STOP` import (or wherever it appears — check `cmd_kutai_restart` and `_do_kutai_stop`).

- [ ] **Step 15: Test import**

Run: `PYTHONIOENCODING=utf-8 .venv/Scripts/python -c "from src.app.telegram_bot import TelegramInterface; print('OK')"`
Expected: OK

- [ ] **Step 16: Run tests**

Run: `PYTHONIOENCODING=utf-8 .venv/Scripts/python -m pytest tests/ -x -q --ignore=tests/integration/test_e2e_llm_pipeline.py`
Expected: All tests pass (ignore the pre-existing LLM e2e failure)

- [ ] **Step 17: Commit**

```bash
git add src/app/telegram_bot.py
git commit -m "feat(bot): remove wrapper commands — now handled by dedicated Yaşar Usta bot"
```

---

### Task 5: Integration Test

- [ ] **Step 1: Verify wrapper starts with new bot token**

Run: `PYTHONIOENCODING=utf-8 .venv/Scripts/python -c "
import os
from dotenv import load_dotenv
load_dotenv()
token = os.getenv('YASAR_USTA_BOT_TOKEN', '')
print(f'Token loaded: {bool(token)} (len={len(token)})')
from yasar_usta import ProcessGuard, GuardConfig
g = ProcessGuard(GuardConfig(telegram_token=token, telegram_chat_id=os.getenv('TELEGRAM_ADMIN_CHAT_ID', '')))
print(f'Telegram enabled: {g.telegram.enabled}')
print('OK')
"`

Expected: Token loaded: True, Telegram enabled: True, OK

- [ ] **Step 2: Verify KutAI bot still starts cleanly**

Run: `PYTHONIOENCODING=utf-8 .venv/Scripts/python -c "from src.app.telegram_bot import TelegramInterface; t = TelegramInterface(); print('OK')"`
Expected: OK

- [ ] **Step 3: Run full test suite**

Run: `PYTHONIOENCODING=utf-8 .venv/Scripts/python -m pytest tests/ -x -q --ignore=tests/integration/test_e2e_llm_pipeline.py`
Expected: All pass

- [ ] **Step 4: Commit all remaining changes**

```bash
git add -A
git commit -m "feat: complete Yaşar Usta bot separation — dedicated bot token, always-on poller"
```
