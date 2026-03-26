# KutAI System Reliability Deep Analysis

**Date:** 2026-03-26
**Scope:** Every hang, crash, orphan-process, and deadlock scenario across the system
**Method:** Static analysis of all process management, async, and resource code paths

---

## 1. Wrapper (kutai_wrapper.py)

### 1.1 SIGINT delivery on Windows — CRITICAL

**Failure mode:** `stop_kutai()` calls `self.process.send_signal(signal.SIGINT)`. On Windows, `SIGINT` cannot be sent to a subprocess created with `asyncio.create_subprocess_exec`. Python's `send_signal(SIGINT)` on Windows actually calls `os.kill(pid, CTRL_C_EVENT)`, which sends a console control event. However, the child was spawned with piped stdout/stderr (no shared console), so the CTRL_C_EVENT may not reach it at all.

**Root cause:** Windows has no POSIX signals. `SIGINT` via `send_signal` is unreliable for non-console-attached subprocesses.

**Severity:** CRITICAL

**Fix:** Replace `send_signal(signal.SIGINT)` with `self.process.terminate()` (sends `TerminateProcess` on Windows, which is reliable). The child process (run.py) already has its own signal handler; for a graceful stop, the wrapper should first try writing to a coordination file or sending CTRL_BREAK_EVENT (which works cross-console), then fall back to terminate after timeout.

### 1.2 Venv Python shim spawning a child process — HIGH

**Failure mode:** On Windows, `.venv/Scripts/python.exe` is a launcher shim that spawns the real `python.exe` as a grandchild. When the wrapper calls `self.process.terminate()` or `self.process.kill()`, it only kills the shim, leaving the real Python (and its llama-server child) orphaned.

**Root cause:** Windows process model does not propagate termination to child processes. The wrapper tracks the shim PID, not the real Python PID.

**Severity:** HIGH

**Fix:** Use `CREATE_NEW_PROCESS_GROUP` creation flag when spawning, and use `taskkill /T /F /PID <pid>` to kill the entire process tree. Alternatively, create a Windows Job Object in the wrapper itself (similar to what `local_model_manager` does) and assign the child process to it with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`.

### 1.3 No watchdog for the wrapper itself — MEDIUM

**Failure mode:** If the wrapper process crashes or its event loop deadlocks, there is nothing to restart it. The system becomes completely unreachable until manual intervention.

**Root cause:** No external supervisor process or Windows Service registration.

**Severity:** MEDIUM

**Fix:** Register `kutai_wrapper.py` as a Windows Service (via `pywin32` or `nssm`) with automatic restart on failure. Alternatively, a Windows Scheduled Task that checks if the wrapper is running every 5 minutes and restarts it if not.

### 1.4 Wrapper _shutdown flag vs asyncio — LOW

**Failure mode:** The `_shutdown` flag is set from a signal handler (`_sig`), which runs on the main thread. The `wrapper.run()` coroutine checks this flag in `while not self._shutdown`, but since asyncio event loops are single-threaded, the flag change is visible immediately on the next iteration. This is correct. However, if the event loop is blocked in `asyncio.sleep(1)` or `wait_for_exit()`, the flag won't be checked until the sleep/wait completes.

**Root cause:** Signal handlers can only set flags; they can't interrupt a running `await`.

**Severity:** LOW (the sleep periods are short, max 1s)

**Fix:** Use `loop.add_signal_handler()` instead of `signal.signal()` on platforms that support it, or use `asyncio.Event` instead of a plain bool flag.

---

## 2. Orchestrator (src/core/orchestrator.py)

### 2.1 Concurrent task futures not cancelled on shutdown — HIGH

**Failure mode:** In the multi-task branch of `run_loop()`, `asyncio.wait()` returns when FIRST_COMPLETED. If `shutdown_fut` fires first, the loop breaks, but the remaining task futures are **still running**. The `finally` block in `start()` shields them with `asyncio.shield()` and waits 30s. But `asyncio.shield` prevents cancellation, so if any task is stuck in a blocking call (e.g., a hung LLM inference), it will run for the full 30s before being abandoned.

After the 30s timeout, `os._exit()` is called (line 368), which hard-kills the process. But any `_stop_server()` or DB commit that was still in progress is interrupted, potentially corrupting the SQLite database or leaving llama-server orphaned.

**Root cause:** `asyncio.shield()` prevents the system from actually cancelling stuck tasks. And `os._exit()` bypasses all cleanup.

**Severity:** HIGH

**Fix:** Instead of `shield()`, use plain `asyncio.gather(*futures, return_exceptions=True)` with the timeout. If timeout expires, explicitly `cancel()` each future, then wait a brief grace period, then run critical cleanup (stop llama-server, close DB) before calling `os._exit()`.

### 2.2 shutdown_event.wait() creates a new Future every loop cycle — MEDIUM

**Failure mode:** In `run_loop()`, line 2082 creates `shutdown_fut = asyncio.ensure_future(self.shutdown_event.wait())` on every cycle. While the future is cancelled if it doesn't fire (line 2130), there's a subtle issue: if the event is already set before `asyncio.wait()` is called, the `shutdown_fut` completes immediately but the task futures may also be ready. Both branches are handled, but creating unnecessary futures each cycle adds GC pressure.

**Root cause:** The shutdown_fut is re-created every iteration instead of being a persistent waiter.

**Severity:** MEDIUM (correctness is fine, just wasteful)

**Fix:** Create `shutdown_fut` once before the while loop, and re-create only after cancellation.

### 2.3 Watchdog resets stuck tasks without checking if they're actually running — MEDIUM

**Failure mode:** The watchdog (line 263) resets tasks stuck in "processing" for >5 minutes. But there's a race: a task may legitimately be running (long inference, complex pipeline). The watchdog resets it to "pending", so the next cycle picks it up again, creating a duplicate execution of the same task.

**Root cause:** The watchdog uses `started_at` age but doesn't check if the corresponding `asyncio.Task` future is still alive.

**Severity:** MEDIUM

**Fix:** Track running task IDs in a set (`self._running_task_ids`). The watchdog should only reset tasks that are NOT in the running set. Additionally, cross-reference with `AGENT_TIMEOUTS` — a task should only be considered stuck if it exceeds 2x its configured timeout.

### 2.4 process_task exception handling catches too broadly — LOW

**Failure mode:** The outer `except Exception` (line 1220) catches everything, including `asyncio.CancelledError` in Python < 3.9 (where it inherits from Exception). In Python 3.9+, CancelledError inherits from BaseException, so this is fine. But if running on 3.8, cancelling a task during shutdown would trigger the retry/fail logic instead of clean cancellation.

**Root cause:** Broad exception handler.

**Severity:** LOW (Python 3.9+ is likely in use)

**Fix:** Add `except asyncio.CancelledError: raise` before the broad except.

---

## 3. LLM Server (src/models/local_model_manager.py)

### 3.1 _stop_server uses synchronous process.poll() in async context — HIGH

**Failure mode:** `_stop_server()` (line 494) calls `self.process.terminate()` then loops with `self.process.poll()` and `asyncio.sleep(0.5)`. The `poll()` call itself is fine (non-blocking), but `self.process.kill()` followed by `self.process.wait(timeout=5)` on line 513 is a **synchronous blocking call**. This blocks the entire asyncio event loop for up to 5 seconds.

**Root cause:** Using `subprocess.Popen.wait()` (synchronous) instead of `asyncio.create_subprocess_exec` for the llama-server process.

**Severity:** HIGH

**Fix:** Use `asyncio.get_event_loop().run_in_executor(None, self.process.wait, 5)` for the blocking wait, or better yet, migrate llama-server management to `asyncio.create_subprocess_exec` so all operations are natively async.

### 3.2 Job Object handle leaked if wrapper uses os._exit() — HIGH

**Failure mode:** The Job Object mechanism (line 81) is excellent for preventing orphans: when the parent dies, Windows closes the handle and kills all assigned processes. However, `os._exit()` in `run.py` (line 368) terminates the process without running cleanup. The OS will close handles on process exit, so the Job Object DOES get closed, and `KILL_ON_JOB_CLOSE` DOES fire. This actually works correctly.

**BUT:** The Job Object is created in `LocalModelManager.__init__()`, which runs in the orchestrator process. If the **wrapper** kills the orchestrator process tree (via the venv shim issue from 1.2), the Job Object handle is closed by the OS, and llama-server is killed. This is actually correct behavior. The real issue is:

If `_kill_orphaned_servers()` runs while a llama-server is still shutting down (e.g., flushing KV cache), `taskkill /F` will corrupt its state.

**Root cause:** `_kill_orphaned_servers` uses `/F` (force kill) with no grace period.

**Severity:** HIGH (data corruption risk for llama.cpp KV cache)

**Fix:** First try `taskkill /IM llama-server.exe` (graceful), wait 5 seconds, then `/F` only if still running.

### 3.3 Model swap during active inference — HIGH

**Failure mode:** If task A is mid-inference on the local model, and task B requests a different model via `ensure_model()`, the `_swap_lock` prevents concurrent swaps. But `ensure_model()` will block on the lock until task A's inference completes and its `call_model` call finishes. However, `call_model` acquires the GPU slot via `acquire_inference_slot`, and the swap path doesn't acquire the GPU slot — it goes through `_swap_model` which takes `_swap_lock`. These are **different locks**.

Scenario: Task A holds the GPU slot (inference running). Task B calls `ensure_model("different_model")`, acquires `_swap_lock`, calls `_stop_server()` which kills llama-server **while task A is still reading from it**. Task A gets a connection error from litellm, retries, and may retry on a model that no longer exists.

**Root cause:** The swap lock and GPU scheduler are not coordinated. A swap should not proceed while any inference slot is held.

**Severity:** HIGH

**Fix:** `_swap_model` should first acquire the GPU scheduler's slot (or check if it's busy) before stopping the server. Alternatively, the scheduler should have a "drain" mode that prevents new acquisitions and waits for the current one to finish before allowing the swap.

### 3.4 Health watchdog only detects crashes, not hangs — MEDIUM

**Failure mode:** `run_health_watchdog()` (line 689) checks `self.process.poll() is not None` — this detects process exit. But if llama-server is hung (deadlock, infinite loop, GPU hang), the process is still alive and `poll()` returns None. The `/health` endpoint isn't checked by the watchdog — only by `_health_check()` which is called from `ensure_model()`.

**Root cause:** The watchdog only checks process liveness, not responsiveness.

**Severity:** MEDIUM

**Fix:** The health watchdog should also call `_health_check()` (HTTP ping to `/health`). If the process is alive but `/health` times out 3 times consecutively, declare it hung and restart.

### 3.5 _kill_orphaned_servers race with legitimate instances — MEDIUM

**Failure mode:** If two KutAI instances start simultaneously (e.g., manual start + wrapper auto-restart), both call `_kill_orphaned_servers`, and one may kill the other's freshly-started llama-server.

**Root cause:** `taskkill /F /IM llama-server.exe` kills ALL llama-server processes, not just orphans.

**Severity:** MEDIUM

**Fix:** Track the expected PID in a file (e.g., `logs/llama_server.pid`). On startup, check if the PID in the file is still running; only kill if it's from a previous session.

---

## 4. GPU Scheduler (src/models/gpu_scheduler.py)

### 4.1 release() uses call_soon_threadsafe + ensure_future — potential deadlock — HIGH

**Failure mode:** `release()` (line 143) calls `loop.call_soon_threadsafe(asyncio.ensure_future, self._do_release())`. This schedules `_do_release()` on the event loop. But `_do_release()` acquires `self._lock`. If `acquire()` is currently holding `self._lock` (waiting for the event), we have:

- `acquire()` holds `_lock`, waiting for `request.event.wait()` — but wait, it **releases** `_lock` before the `wait` (line 127). So this is actually fine.

The real issue: `release()` is called from `finally` blocks in `call_model`. If the event loop is busy (e.g., processing many callbacks), `_do_release()` may be delayed, causing the next waiter to wait longer than necessary.

**Root cause:** Indirect scheduling via `call_soon_threadsafe`.

**Severity:** LOW (not a deadlock, just latency)

**Fix:** Make `release()` an async method and call it directly: `await scheduler.release()`. The sync wrapper is only needed if called from non-async context, which doesn't happen in practice.

### 4.2 GPU slot release — VERIFIED CORRECT

**Analysis:** The GPU slot IS properly released in a `finally` block at router.py line 1137-1139:
```python
finally:
    if local_manager:
        local_manager.release_inference_slot()
```
This covers all paths: normal return, timeout, and exception.

**Severity:** N/A (not a bug)

### 4.3 No starvation prevention for low-priority tasks — MEDIUM

**Failure mode:** If high-priority tasks keep arriving, low-priority background tasks (priority 3) may wait indefinitely. The timeout (120s) prevents permanent starvation, but the task fails after timeout rather than being served.

**Root cause:** Pure priority queue with no aging or fairness mechanism.

**Severity:** MEDIUM

**Fix:** Implement priority aging: every 30s a request waits, its effective priority increases by 1. This ensures even low-priority tasks eventually get served.

---

## 5. Router (src/core/router.py)

### 5.1 asyncio.wait_for wrapping litellm.acompletion — potential zombie coroutine — HIGH

**Failure mode:** `asyncio.wait_for(litellm.acompletion(...), timeout=timeout_val)` at line 951 creates a timeout wrapper. When the timeout fires, `wait_for` cancels the inner task. But `litellm.acompletion` may be deep in an HTTP connection (via httpx or aiohttp). If the underlying HTTP library doesn't properly handle cancellation, the connection is leaked. Over time, this exhausts connection pools or file descriptors.

On local models specifically: the request to `http://127.0.0.1:8080/v1/chat/completions` may hang if llama-server is stuck in GPU computation. The `timeout_val` (120s for local) will fire, but the connection to localhost remains open.

**Root cause:** `asyncio.wait_for` cancels the task, but cleanup depends on the library handling `CancelledError` properly.

**Severity:** HIGH

**Fix:** After a timeout, explicitly check if the underlying connection is closed. Consider using `httpx` directly with a `timeout` parameter instead of relying on `asyncio.wait_for`, so the HTTP client handles its own timeout and cleanup.

### 5.2 GPU slot release on timeout — VERIFIED CORRECT

**Analysis:** The GPU slot IS properly released. The `try/finally` at router.py lines 934/1137-1139 covers all paths including timeouts. When `asyncio.wait_for` raises `TimeoutError`, the `continue` goes to the next retry attempt within the same `try` block. When the retry loop ends (via `break` or exhaustion), the `finally` fires and releases the slot.

**Severity:** N/A (not a bug)

### 5.3 Retry loop can cascade: 5 candidates x 3 retries = 15 attempts — MEDIUM

**Failure mode:** `call_model` tries up to 5 model candidates, each with 2-3 retries. If all providers are slow (not down, just slow), the total time could be 5 x 3 x 120s = 30 minutes. The orchestrator's `asyncio.wait_for(coro, timeout=timeout_seconds)` will cancel this, but only after the agent-level timeout (which can be up to 900s for workflows).

**Root cause:** No aggregate timeout across all candidates/retries.

**Severity:** MEDIUM

**Fix:** Add a `call_model`-level deadline (e.g., 5 minutes total across all candidates). Break the candidate loop when the deadline is exceeded.

---

## 6. Telegram Shutdown (/restart, /stop)

### 6.1 threading.Timer os._exit() is the ONLY reliable shutdown — acknowledged

**Failure mode:** The `cmd_kutai_restart` and `cmd_kutai_stop` handlers set `shutdown_event` and then arm a `threading.Timer(5.0, lambda: os._exit(code))`. If the graceful path succeeds within 5 seconds, `os._exit` runs anyway — but actually, `os._exit()` in the timer thread will terminate the process regardless. If the graceful path already called `os._exit()` first, the timer is moot.

The concern: if the event loop is completely blocked (e.g., stuck in synchronous `process.wait()`), the `shutdown_event.set()` never gets processed by the event loop. Only the timer fires and does `os._exit()`, which skips all cleanup (DB close, llama-server stop, metrics persist).

**Root cause:** `os._exit()` is a hard kill with no cleanup.

**Severity:** HIGH (but acceptable as a last resort — the real fix is ensuring the graceful path never blocks)

**Fix:** The timer should try a more graceful approach before `os._exit`:
1. At 5s: try `signal.raise_signal(SIGINT)` to interrupt the event loop
2. At 10s: `os._exit()` as absolute last resort

### 6.2 Telegram handler runs on the event loop thread — MEDIUM

**Failure mode:** python-telegram-bot's handlers run as asyncio tasks on the main event loop. If the event loop is blocked (e.g., by synchronous `process.wait()` in llama-server stop), Telegram commands cannot be processed at all. The user sends `/restart` but the handler never fires.

**Root cause:** All I/O and command processing shares a single event loop.

**Severity:** MEDIUM

**Fix:** The `threading.Timer` fallback already mitigates this. For extra safety, the wrapper's mini Telegram poller (which runs on a separate asyncio loop) provides an alternative control path when the orchestrator is unresponsive.

---

## 7. ntfy Blocking (src/infra/notifications.py)

### 7.1 NtfyAlertHandler.emit() does synchronous HTTP in logging handler — HIGH

**Failure mode:** `NtfyAlertHandler.emit()` (line 117) calls `send_ntfy()` which does `requests.post(timeout=3)`. This is called from the logging framework, which may be invoked from async code on the event loop thread. A synchronous HTTP call blocks the event loop for up to 3 seconds per ERROR log.

If ntfy is down or slow, every ERROR log blocks the event loop for 3 seconds. Multiple errors in quick succession could freeze the system for 10+ seconds.

**Root cause:** Synchronous `requests.post` in a logging handler that's called from async context.

**Severity:** HIGH

**Fix:** Use a background thread for the HTTP call, similar to how `NtfyBatchHandler._flush_locked()` does it (line 232). Wrap `send_ntfy()` in `threading.Thread(target=..., daemon=True).start()`.

### 7.2 NtfyBatchHandler._send_batch retry sleeps for 2 seconds — MEDIUM

**Failure mode:** `_send_batch` (line 234) has a `time.sleep(self.RETRY_DELAY)` (2 seconds) on failure. When called from `_flush_locked` (overflow path), it runs in a background thread, so this is fine. But `_flush()` (called from `_timer_flush`) also calls `_send_batch` directly. The timer callback runs in a separate thread, so this is also fine.

The only concern: if `_flush()` is somehow called from the main thread, it blocks for 2 seconds. Currently this only happens via `atexit`, which is fine.

**Root cause:** Synchronous sleep in retry logic.

**Severity:** LOW (all callers are on background threads)

**Fix:** No immediate fix needed. The current architecture is correct.

### 7.3 send_ntfy creates a new requests.Session per call — LOW

**Failure mode:** Each `send_ntfy` call creates an implicit new TCP connection (no session reuse). Under high error rates, this could exhaust ephemeral ports.

**Root cause:** No connection pooling.

**Severity:** LOW

**Fix:** Use a module-level `requests.Session()` for connection pooling.

---

## 8. Docker/Startup (src/app/run.py)

### 8.1 start_docker_services blocks the event loop for up to 120 seconds — HIGH

**Failure mode:** `start_docker_services()` (line 208) calls `subprocess.run(["docker", "compose", "up", "-d"], timeout=120)`. This is a synchronous call made **before** the asyncio event loop is fully running (called from `async def main()`). Since `main()` is the top-level coroutine, and no other tasks are running yet, this blocks the event loop for the entire duration.

If Docker is slow to start (pulling images, building), the system is completely unresponsive for up to 2 minutes. No Telegram commands, no health checks.

**Root cause:** Synchronous subprocess call in async context.

**Severity:** HIGH (blocks startup)

**Fix:** Move Docker startup to `asyncio.create_subprocess_exec` or `loop.run_in_executor`. Better yet, since no other tasks are running at this point, it's acceptable to block — but add a Telegram notification before starting: "Starting Docker services, please wait..."

### 8.2 startup_health_check can hang if Docker check hangs — MEDIUM

**Failure mode:** `_docker_check()` (line 175) calls `subprocess.run(timeout=5)` — synchronous, but with a 5s timeout. If Docker is not installed and the `FileNotFoundError` path is hit, it returns immediately. The concern is if Docker is installed but Docker Desktop is frozen — `docker inspect` may hang past the timeout (the `timeout=5` should prevent this, so this is actually handled).

The `_async_check` wrapper adds a `timeout=6` via `asyncio.wait_for`, providing double protection.

**Root cause:** N/A — actually handled correctly.

**Severity:** LOW

**Fix:** None needed.

### 8.3 os._exit() in run.py skips atexit handlers — MEDIUM

**Failure mode:** Line 368: `os._exit(orch.requested_exit_code)`. This bypasses Python's normal cleanup: `atexit` handlers, `__del__` methods, and buffered I/O. The `NtfyBatchHandler`'s `atexit.register(self._flush)` will not fire. Any buffered logs are lost.

**Root cause:** `os._exit()` is used intentionally to bypass uvicorn/asyncio cleanup issues on Windows. But it has side effects.

**Severity:** MEDIUM

**Fix:** Flush ntfy buffer explicitly in the orchestrator's shutdown sequence (before `os._exit`). Add a call to `logging.shutdown()` in the finally block.

---

## 9. Agent System (src/agents/base.py)

### 9.1 Agent iteration loop is bounded but tool execution is not — HIGH

**Failure mode:** The ReAct loop (line 1018) is bounded by `self.max_iterations`. But within each iteration, tool execution (`execute_tool`) has no timeout. If a tool (e.g., `shell` running a command, or `web_search` hitting a slow API) hangs, the iteration never completes.

The agent-level timeout in the orchestrator (`asyncio.wait_for(coro, timeout=timeout_seconds)`) is the only protection. But `timeout_seconds` can be up to 900s (15 min for workflows).

**Root cause:** No per-tool-call timeout.

**Severity:** HIGH

**Fix:** Wrap each `execute_tool` call with `asyncio.wait_for(execute_tool(...), timeout=60)`. For long-running tools like `shell`, use the tool's own timeout parameter.

### 9.2 Checkpoint save can block on DB write — LOW

**Failure mode:** `_save_checkpoint()` (referenced at line 1590) writes to SQLite. aiosqlite is used, which runs DB operations in a background thread. The `await` is non-blocking for the event loop. But if the DB is locked by another operation (WAL mode contention), the write may take a few seconds.

**Root cause:** SQLite write contention.

**Severity:** LOW (aiosqlite handles this gracefully with retries)

**Fix:** None needed unless observed in practice.

### 9.3 ask_agent tool can create unbounded recursion — MEDIUM

**Failure mode:** The agent can use `ask_agent` to delegate to another agent, which can itself use `ask_agent`. There's a `tool_depth` context key that should limit this, but it depends on correct propagation.

**Root cause:** Recursive agent invocation without hard depth limit.

**Severity:** MEDIUM

**Fix:** Enforce a hard depth limit (e.g., 3) in the `ask_agent` tool implementation. Check `tool_depth` in the context and reject if exceeded.

---

## 10. Cross-Cutting: The Deadliest Combo

### 10.1 The full death scenario — CRITICAL

**Scenario:** This is the most likely real-world hang:

1. User sends a complex task
2. Router selects local model, acquires GPU slot
3. litellm calls llama-server, which starts GPU inference
4. GPU hangs (driver issue, CUDA error, OOM) — llama-server stops responding
5. `asyncio.wait_for` fires after 120s, cancels the litellm coroutine
6. GPU slot IS released by the finally block, but the cancelled litellm coroutine may leak an HTTP connection
7. Subsequent tasks may find llama-server unresponsive (stuck in GPU)
8. User sends `/restart`
9. Telegram handler sets shutdown_event
10. Event loop processes shutdown
11. `_stop_server()` calls `self.process.terminate()` — but llama-server is stuck in GPU
12. After 10s, `self.process.kill()` + `self.process.wait(timeout=5)` **blocks the event loop for 5 seconds**
13. The 5s `threading.Timer` fires `os._exit(42)` — hard kills the process
14. Wrapper detects exit code 42, restarts
15. New orchestrator starts, `_kill_orphaned_servers` runs `taskkill /F /IM llama-server.exe`
16. If the old llama-server was still shutting down (flushing cache), its state is corrupted
17. New llama-server starts cleanly — but the issue repeats if the GPU is still in a bad state

**Root cause:** Multiple interacting issues: synchronous blocking in _stop_server, zombie HTTP connections, hard kill cascade.

**Severity:** CRITICAL

**Fix:** Requires fixing issues 3.1, 5.1, and 3.2 together:
1. Make `_stop_server` fully async
2. Ensure GPU slot is always released in a finally block
3. Add grace period to orphan cleanup
4. Add GPU health check (nvidia-smi) before restarting llama-server

---

## Summary: Priority-Ordered Fix List

| # | Issue | Severity | Effort | File |
|---|-------|----------|--------|------|
| 1.1 | SIGINT delivery fails on Windows | CRITICAL | Small | kutai_wrapper.py |
| 10.1 | Full death scenario (combo) | CRITICAL | Large | Multiple |
| 1.2 | Venv shim leaves orphan process tree | HIGH | Medium | kutai_wrapper.py |
| 2.1 | shield() prevents cancellation on shutdown | HIGH | Small | orchestrator.py |
| 3.1 | Synchronous process.wait() blocks event loop | HIGH | Medium | local_model_manager.py |
| 3.2 | taskkill /F corrupts llama-server state | HIGH | Small | local_model_manager.py |
| 3.3 | Model swap during active inference | HIGH | Medium | local_model_manager.py |
| 5.1 | Zombie coroutines from cancelled litellm calls | HIGH | Medium | router.py |
| 7.1 | NtfyAlertHandler blocks event loop | HIGH | Small | notifications.py |
| 8.1 | Docker startup blocks for 120s | HIGH | Small | run.py |
| 6.1 | os._exit skips cleanup | HIGH | Medium | telegram_bot.py |
| 9.1 | Tool execution has no per-call timeout | HIGH | Small | base.py |
| 2.3 | Watchdog resets legitimately running tasks | MEDIUM | Small | orchestrator.py |
| 3.4 | Health watchdog misses hangs | MEDIUM | Small | local_model_manager.py |
| 3.5 | _kill_orphaned_servers race condition | MEDIUM | Small | local_model_manager.py |
| 4.3 | Low-priority GPU starvation | MEDIUM | Small | gpu_scheduler.py |
| 5.3 | 15 retry cascade | MEDIUM | Small | router.py |
| 6.2 | Telegram handler blocked by event loop | MEDIUM | N/A | (mitigated by timer) |
| 8.3 | os._exit skips atexit handlers | MEDIUM | Small | run.py |
| 9.3 | ask_agent recursion | MEDIUM | Small | base.py |
| 1.3 | No wrapper watchdog | MEDIUM | Medium | (new) |
| 2.2 | shutdown_fut re-created each cycle | MEDIUM | Trivial | orchestrator.py |
| 2.4 | CancelledError caught as Exception | LOW | Trivial | orchestrator.py |
| 1.4 | _shutdown flag vs asyncio | LOW | Trivial | kutai_wrapper.py |
| 7.2 | ntfy retry sleep | LOW | N/A | (correct) |
| 7.3 | No HTTP session reuse | LOW | Trivial | notifications.py |
| 9.2 | Checkpoint DB contention | LOW | N/A | (acceptable) |
| 8.2 | Docker check hang | LOW | N/A | (handled) |

---

## Recommended Implementation Order

**Phase A — Stop the bleeding (1-2 days):**
1. Fix SIGINT on Windows (1.1) — use `terminate()` instead
2. Fix NtfyAlertHandler blocking (7.1) — move to background thread
3. Add per-tool timeout (9.1) — wrap `execute_tool` with `wait_for`
4. Make `_stop_server` fully async (3.1) — critical for shutdown reliability

**Phase B — Structural fixes (3-5 days):**
5. Add Job Object to wrapper for process tree management (1.2)
7. Coordinate model swap with GPU scheduler (3.3)
8. Fix shutdown to cancel tasks instead of shielding (2.1)
9. Add graceful orphan cleanup (3.2)

**Phase C — Hardening (ongoing):**
10. Health watchdog checks responsiveness not just liveness (3.4)
11. Watchdog respects running task set (2.3)
12. Register wrapper as Windows Service (1.3)
13. Add aggregate timeout to call_model (5.3)
14. Priority aging in GPU scheduler (4.3)
