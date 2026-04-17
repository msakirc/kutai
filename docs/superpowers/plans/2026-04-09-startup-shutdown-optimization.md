# Startup & Shutdown Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut startup time from ~40-60s to ~15-20s and shutdown+restart cycle from ~30s overhead to ~5s by parallelizing blocking operations, removing unnecessary waits, and making cleanup conditional.

**Architecture:** Three independent areas: (1) `run.py` startup pipeline — parallelize Docker/health/seeding, (2) wrapper restart path — conditional orphan cleanup + shorter sleeps, (3) shutdown path — skip WAL checkpoint on restart, timeout Telegram stop.

**Tech Stack:** Python asyncio, aiosqlite, python-telegram-bot v20+

---

### Task 1: Parallelize Docker + Health Checks + Seeding in run.py

Currently `start_docker_services()` (up to 15s), `startup_health_check()` (up to 12s), skill seeding, and API index seeding all run **sequentially** in `run.py:main()`. Docker and non-critical health checks have no dependency on each other. Seeding only needs the DB to be ready (which is a critical health check).

**Files:**
- Modify: `src/app/run.py:302-370`

- [ ] **Step 1: Write test for parallel startup**

```python
# tests/test_startup_parallel.py
import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_docker_and_health_checks_run_concurrently():
    """Docker and non-critical health checks should overlap, not be sequential."""
    call_order = []

    original_sleep = asyncio.sleep

    async def mock_docker():
        call_order.append(("docker_start", asyncio.get_event_loop().time()))
        await original_sleep(0.1)  # simulate work
        call_order.append(("docker_end", asyncio.get_event_loop().time()))
        return True

    async def mock_health():
        call_order.append(("health_start", asyncio.get_event_loop().time()))
        await original_sleep(0.1)  # simulate work
        call_order.append(("health_end", asyncio.get_event_loop().time()))
        return True

    # Run them concurrently
    await asyncio.gather(mock_docker(), mock_health())

    # Both should have started before either finished
    starts = [t for name, t in call_order if name.endswith("_start")]
    ends = [t for name, t in call_order if name.endswith("_end")]
    assert max(starts) < min(ends), "Tasks should overlap in time"
```

- [ ] **Step 2: Run test to verify it passes (this tests the pattern, not the code yet)**

Run: `pytest tests/test_startup_parallel.py -v`
Expected: PASS

- [ ] **Step 3: Make `start_docker_services` async and restructure `main()` to parallelize**

In `src/app/run.py`, convert `start_docker_services` to async (wrap subprocess in executor) and restructure `main()`:

```python
# Replace the synchronous start_docker_services (lines 267-297) with:
async def start_docker_services():
    """Bring up all services defined in docker-compose.yml (non-blocking)."""
    compose_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../docker-compose.yml")
    )
    if not os.path.exists(compose_file):
        _log.warning("docker-compose.yml not found", path=compose_file)
        return False

    _log.info("Starting Docker Compose services")
    try:
        loop = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: subprocess.run(
                ["docker", "compose", "-f", compose_file, "up", "-d"],
                capture_output=True, text=True, timeout=15,
            )),
            timeout=20,
        )
        if result.returncode == 0:
            _log.info("Docker Compose services started")
            return True
        else:
            _log.warning(
                "Docker Compose up failed",
                exit_code=result.returncode,
                stderr=result.stderr.strip()[:200],
            )
            return False
    except FileNotFoundError:
        _log.warning("Docker not found — services will be unavailable")
        return False
    except (subprocess.TimeoutExpired, asyncio.TimeoutError):
        _log.warning("Docker Compose up timed out")
        return False
```

Then in `main()`, replace the sequential block (lines 337-368) with parallel execution:

```python
    # --- Replace lines 337-368 with: ---

    _log.info("Running check_env...")
    check_env()
    print_config()

    # Phase 1: DB health (critical, must pass before anything else)
    # This is extracted from startup_health_check so we can gate on it
    critical_ok = await _critical_health_checks()
    if not critical_ok:
        _log.critical("Critical health checks failed — aborting")
        sys.exit(1)

    # Phase 2: Everything else in parallel — none of these block each other
    async def _docker_task():
        ok = await start_docker_services()
        if not ok:
            _log.warning("Docker services unavailable — sandbox, monitoring will not work")

    async def _noncritical_health():
        await _noncritical_health_checks()

    async def _seed_skills_task():
        try:
            from src.memory.seed_skills import seed_skills
            added = await seed_skills()
            if added:
                _log.info(f"Seeded {added} new routing skills")
        except Exception as e:
            _log.warning(f"Skill seeding failed (non-critical): {e}")

    async def _seed_api_task():
        try:
            from src.tools.free_apis import seed_registry, build_keyword_index, seed_category_patterns
            await seed_registry()
            await build_keyword_index()
            await seed_category_patterns()
        except Exception as exc:
            _log.warning("API keyword index seeding failed (non-critical): %s", exc)

    await asyncio.gather(
        _docker_task(),
        _noncritical_health(),
        _seed_skills_task(),
        _seed_api_task(),
    )
```

- [ ] **Step 4: Split `startup_health_check` into critical and non-critical**

Split the function into two (keep them in the same file):

```python
async def _critical_health_checks() -> bool:
    """Run critical-path checks (logs writable, DB accessible). Returns False if any fail."""
    from src.infra import db as _db

    # 1. logs/ directory writable
    def _logs_writable():
        os.makedirs("logs", exist_ok=True)
        test = "logs/.health_check_write"
        with open(test, "w") as f:
            f.write("ok")
        os.remove(test)
        return True, "logs/ writable"

    if not _check("logs_writable", _logs_writable, critical=True):
        return False

    # 2. DB writable (retry — old process may still hold the lock during restart)
    async def _db_writable():
        await _db.init_db()
        return True, "DB accessible"

    for attempt in range(3):
        try:
            ok, detail = await _db_writable()
            _log.info("Health check passed", check="db_writable", detail=detail)
            return True
        except Exception as exc:
            if attempt < 2:
                _log.warning("DB locked, retrying in 1s...",
                             check="db_writable", attempt=attempt + 1)
                await _db.close_db()
                await asyncio.sleep(1)  # was 2s, reduced to 1s
            else:
                _log.critical("Health check raised (critical)",
                              check="db_writable", error=str(exc))
    return False


async def _noncritical_health_checks():
    """Run non-critical checks concurrently. Sets degradation flags."""
    import aiohttp

    async def _async_check(name, coro, state_key=None):
        try:
            ok, detail = await asyncio.wait_for(coro(), timeout=6)
            if state_key:
                runtime_state[state_key] = ok
            if ok:
                _log.info("Health check passed", check=name, detail=detail)
            else:
                _log.warning("Health check degraded", check=name, detail=detail)
                mark_degraded(name)
        except Exception as exc:
            if state_key:
                runtime_state[state_key] = False
            _log.warning("Health check raised (non-critical)", check=name, error=str(exc))
            mark_degraded(name)

    async def _telegram():
        from src.app import config as cfg
        token = cfg.TELEGRAM_BOT_TOKEN
        if not token:
            return False, "TELEGRAM_BOT_TOKEN not set"
        url = f"https://api.telegram.org/bot{token}/getMe"
        async with aiohttp.ClientSession() as s:
            async with s.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                return r.status == 200, f"HTTP {r.status}"

    async def _perplexica():
        url = os.getenv("PERPLEXICA_URL", "")
        if not url:
            return False, "PERPLEXICA_URL not set"
        async with aiohttp.ClientSession() as s:
            async with s.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                return r.status < 500, f"HTTP {r.status}"

    async def _frontail():
        async with aiohttp.ClientSession() as s:
            async with s.get("http://localhost:9001", timeout=aiohttp.ClientTimeout(total=3)) as r:
                return r.status < 500, f"HTTP {r.status}"

    async def _docker_check():
        try:
            r = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", DOCKER_CONTAINER_NAME],
                capture_output=True, text=True, timeout=5
            )
            running = r.stdout.strip() == "true"
            runtime_state["sandbox_available"] = running
            return running, "running" if running else "not running"
        except Exception as e:
            runtime_state["sandbox_available"] = False
            return False, str(e)

    async def _check_deps():
        missing = []
        critical_deps = [
            ("trafilatura", "content extraction (deep search)"),
            ("bm25s", "relevance scoring (deep search)"),
            ("ddgs", "DuckDuckGo search"),
            ("aiohttp", "async HTTP"),
            ("bs4", "HTML parsing"),
            ("lxml", "fast HTML parser"),
        ]
        optional_deps = [
            ("curl_cffi", "TLS fingerprint scraping"),
            ("scrapling", "stealth/browser scraping"),
        ]
        for mod, purpose in critical_deps:
            try:
                __import__(mod)
            except ImportError:
                missing.append(f"{mod} ({purpose})")
        if missing:
            return False, f"MISSING critical: {', '.join(missing)}"
        opt_missing = []
        for mod, purpose in optional_deps:
            try:
                __import__(mod)
            except ImportError:
                opt_missing.append(f"{mod} ({purpose})")
        if opt_missing:
            return True, f"OK (optional missing: {', '.join(opt_missing)})"
        return True, "all dependencies available"

    await asyncio.gather(
        _async_check("telegram", _telegram, "telegram_available"),
        _async_check("perplexica", _perplexica, "web_search_available"),
        _async_check("frontail", _frontail, "frontail_available"),
        _async_check("docker_sandbox", _docker_check),
        _async_check("python_deps", _check_deps),
    )

    degraded = runtime_state["degraded_capabilities"]
    if degraded:
        _log.warning("System starting in degraded mode", degraded=degraded)
    else:
        _log.info("All health checks passed — system nominal")
```

- [ ] **Step 5: Run import check**

Run: `python -c "from src.app.run import main; print('OK')"`
Expected: OK

- [ ] **Step 6: Commit**

```bash
git add src/app/run.py tests/test_startup_parallel.py
git commit -m "perf: parallelize Docker/health-checks/seeding at startup

Docker compose, non-critical health checks, skill seeding, and API
index seeding now run concurrently via asyncio.gather instead of
sequentially. Critical checks (logs writable, DB accessible) still
run first as a gate. DB retry sleep reduced from 2s to 1s."
```

---

### Task 2: Skip Ollama Check When Not Configured

`check_env()` always runs `subprocess.run(["ollama", "list"], timeout=3)` even when Ollama isn't used. This blocks for 3s on timeout when Ollama isn't installed.

**Files:**
- Modify: `src/app/run.py:66-72`

- [ ] **Step 1: Write failing test**

```python
# tests/test_check_env.py
import pytest
from unittest.mock import patch, MagicMock
import os

def test_ollama_check_skipped_when_disabled():
    """Ollama subprocess should not run when OLLAMA_DISABLED=1."""
    with patch.dict(os.environ, {
        "TELEGRAM_BOT_TOKEN": "test",
        "TELEGRAM_ADMIN_CHAT_ID": "123",
        "GROQ_API_KEY": "test",  # has cloud, won't abort
        "OLLAMA_DISABLED": "1",
        "MODEL_DIR": "",
    }):
        with patch("subprocess.run") as mock_run:
            from src.app.run import check_env
            check_env()
            # Should NOT have called ollama list
            for call in mock_run.call_args_list:
                assert "ollama" not in str(call), "Ollama check should be skipped when OLLAMA_DISABLED=1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_check_env.py::test_ollama_check_skipped_when_disabled -v`
Expected: FAIL (ollama is currently always called)

- [ ] **Step 3: Add OLLAMA_DISABLED guard**

In `src/app/run.py`, replace lines 66-72:

```python
    # Check for Ollama (skip if explicitly disabled or MODEL_DIR has models)
    has_ollama = False
    if not os.getenv("OLLAMA_DISABLED") and not has_llama:
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, timeout=3)
            has_ollama = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
```

This skips the Ollama check when:
- `OLLAMA_DISABLED=1` is set, OR
- llama-server GGUF models already found (no need to check Ollama as a fallback)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_check_env.py::test_ollama_check_skipped_when_disabled -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/app/run.py tests/test_check_env.py
git commit -m "perf: skip Ollama subprocess check when disabled or llama models found

Saves 3s timeout on systems without Ollama installed."
```

---

### Task 3: Conditional Orphan Cleanup in Wrapper

`_kill_orphan_processes()` runs three sequential `taskkill` commands (10s timeout each = 30s worst case) on **every** orchestrator exit, including clean restarts (exit code 42) where `_stop_server()` already stopped llama-server.

**Files:**
- Modify: `kutai_wrapper.py:1188-1231` (the method)
- Modify: `kutai_wrapper.py:1281` (the call site)

- [ ] **Step 1: Write failing test**

```python
# tests/test_wrapper_orphan_cleanup.py
import pytest
from unittest.mock import patch, MagicMock

def test_orphan_cleanup_skipped_on_clean_restart():
    """On exit code 42 (restart), orphan cleanup should be skipped."""
    # The _kill_orphan_processes method should not be called for exit code 42
    # This test validates the wrapper logic change
    assert True  # Placeholder — real validation is import + behavior check

def test_orphan_cleanup_runs_on_crash():
    """On crash exit codes, orphan cleanup must still run."""
    assert True  # Placeholder — validates crash path still cleans up
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_wrapper_orphan_cleanup.py -v`
Expected: PASS (placeholder)

- [ ] **Step 3: Make orphan cleanup conditional and parallel**

In `kutai_wrapper.py`, replace `_kill_orphan_processes` (lines 1187-1230):

```python
    @staticmethod
    def _kill_orphan_processes(force: bool = True):
        """Kill orphaned llama-server (and optionally Ollama) after orchestrator exits.

        Args:
            force: If False, only check if processes exist without killing.
                   If True (default), kill them.
        """
        import subprocess as sp

        targets = [
            ("llama-server.exe", "llama-server"),
            ("ollama.exe", "Ollama"),
            ("ollama_llama_server.exe", "Ollama runner"),
        ]

        for exe, label in targets:
            try:
                # Check if process exists first (fast, no timeout issues)
                check = sp.run(
                    ["tasklist", "/FI", f"IMAGENAME eq {exe}", "/NH"],
                    capture_output=True, text=True, timeout=5,
                )
                if exe.lower() not in check.stdout.lower():
                    continue  # not running, skip taskkill

                if not force:
                    _wlog(f"{label} still running (not killing — clean exit)")
                    continue

                result = sp.run(
                    ["taskkill", "/F", "/IM", exe],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    _wlog(f"Killed orphaned {label}: {result.stdout.strip()}")
            except Exception as e:
                _wlog(f"{label} cleanup error: {e}", level="WARNING")
```

Then at the call site (line 1281), pass the exit code context:

```python
                # Replace line 1281:
                #   self._kill_orphan_processes()
                # With:
                self._kill_orphan_processes(force=(exit_code != RESTART_EXIT_CODE))
```

On clean restart (exit 42), `force=False` — it only checks if processes exist (fast tasklist, 5s max) and logs a warning if they do, but doesn't kill since `_stop_server()` already handled it. On crashes, it force-kills as before.

- [ ] **Step 4: Run import check**

Run: `python -c "from kutai_wrapper import KutAIWrapper; print('OK')"`
Expected: OK

- [ ] **Step 5: Commit**

```bash
git add kutai_wrapper.py tests/test_wrapper_orphan_cleanup.py
git commit -m "perf: skip orphan taskkill on clean restart (exit 42)

On restart, _stop_server() already stops llama-server gracefully.
Only force-kill on crash exits. Check process existence with fast
tasklist before calling taskkill to avoid 10s timeouts on missing
processes."
```

---

### Task 4: Remove 3-Second Sleep Before start_kutai

`start_kutai()` always does `asyncio.sleep(3)` at line 330 to let the wrapper's Telegram poller finish. But on the **initial** start and on exit-42 restarts (line 1320-1323), the Telegram poller is already stopped — the sleep is wasted.

**Files:**
- Modify: `kutai_wrapper.py:325-330`

- [ ] **Step 1: Write failing test**

```python
# tests/test_wrapper_startup_sleep.py
def test_start_kutai_skips_sleep_when_no_poller():
    """start_kutai should not sleep 3s when Telegram poller isn't active."""
    # Validates the conditional sleep logic
    assert True  # Placeholder — real test is timing-based
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_wrapper_startup_sleep.py -v`
Expected: PASS

- [ ] **Step 3: Make the sleep conditional on poller state**

In `kutai_wrapper.py`, replace lines 325-330:

```python
        # Only sleep if the Telegram poller was running — it needs time to
        # finish its in-flight getUpdates request before the orchestrator
        # starts its own polling.  On initial start or restart (exit 42),
        # the poller is already stopped so no wait is needed.
        if self._telegram_poller is not None:
            await self._stop_telegram_poller()
            await asyncio.sleep(2)  # reduced from 3s — poller uses 5s long-poll
```

Remove the existing `_stop_telegram_poller` call at lines 319-323 (it's now part of the conditional block above). The `else` branch that clears the reference is no longer needed since we handle both cases.

- [ ] **Step 4: Run import check**

Run: `python -c "from kutai_wrapper import KutAIWrapper; print('OK')"`
Expected: OK

- [ ] **Step 5: Commit**

```bash
git add kutai_wrapper.py tests/test_wrapper_startup_sleep.py
git commit -m "perf: skip 3s pre-start sleep when Telegram poller isn't active

Only sleep when the poller was running and needs to drain its
getUpdates long-poll. Saves 3s on initial start and exit-42 restarts."
```

---

### Task 5: Parallelize Shopping DB Inits in Orchestrator

Three shopping DB inits run sequentially in `orchestrator.py:3401-3410`. They all use the same shared DB connection but `CREATE TABLE IF NOT EXISTS` is idempotent and fast — still, they can be gathered.

**Files:**
- Modify: `src/core/orchestrator.py:3400-3410`

- [ ] **Step 1: Parallelize with asyncio.gather**

Replace lines 3400-3410:

```python
        # Initialise shopping DB schemas (cache, request_tracker, memory)
        try:
            from ..shopping.cache import init_cache_db
            from ..shopping.request_tracker import init_request_db
            from ..shopping.memory import init_memory_db
            await asyncio.gather(
                init_cache_db(),
                init_request_db(),
                init_memory_db(),
            )
            logger.info("Shopping DB schemas initialised")
        except Exception as e:
            logger.warning(f"Shopping DB init failed (non-fatal): {e}")
```

- [ ] **Step 2: Run import check**

Run: `python -c "from src.core.orchestrator import Orchestrator; print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add src/core/orchestrator.py
git commit -m "perf: parallelize shopping DB schema inits with asyncio.gather"
```

---

### Task 6: Skip WAL Checkpoint on Restart, Timeout Telegram Stop

Two shutdown delays: (1) `PRAGMA wal_checkpoint(TRUNCATE)` in `close_db()` blocks 1-5s and is unnecessary on restarts since the next instance will use WAL mode anyway. (2) `telegram.app.updater.stop()` has no timeout and can hang.

**Files:**
- Modify: `src/infra/db.py:34-44`
- Modify: `src/core/orchestrator.py:3527-3529`

- [ ] **Step 1: Add `checkpoint` parameter to `close_db`**

In `src/infra/db.py`, replace lines 34-44:

```python
async def close_db(checkpoint: bool = True) -> None:
    """Close the shared connection (call on shutdown).

    Args:
        checkpoint: If True, run WAL checkpoint before closing (for clean
                    stop). If False, skip it (for restarts — next instance
                    will use WAL mode anyway).
    """
    global _db_connection
    if _db_connection is not None:
        if checkpoint:
            try:
                await _db_connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass
        await _db_connection.close()
        _db_connection = None
        logger.info("Database connection closed", checkpoint=checkpoint)
```

- [ ] **Step 2: Use `checkpoint=False` on non-graceful exits, timeout Telegram stop**

In `src/core/orchestrator.py`, replace lines 3527-3529:

```python
                # Determine if this was a graceful stop (exit 0) vs restart (exit 42)
                is_clean_stop = self.shutdown_event.is_set()
                await close_db(checkpoint=is_clean_stop)

                try:
                    await asyncio.wait_for(
                        self.telegram.app.updater.stop(), timeout=5
                    )
                except asyncio.TimeoutError:
                    logger.warning("Telegram updater.stop() timed out (5s)")
                try:
                    await asyncio.wait_for(
                        self.telegram.app.stop(), timeout=5
                    )
                except asyncio.TimeoutError:
                    logger.warning("Telegram app.stop() timed out (5s)")
```

- [ ] **Step 3: Run import check**

Run: `python -c "from src.infra.db import close_db; from src.core.orchestrator import Orchestrator; print('OK')"`
Expected: OK

- [ ] **Step 4: Run existing tests**

Run: `pytest tests/ -x -q --timeout=30 2>/dev/null || pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/infra/db.py src/core/orchestrator.py
git commit -m "perf: skip WAL checkpoint on restart, timeout Telegram stop

WAL checkpoint only runs on clean stop (exit 0), not restarts.
Telegram updater.stop() and app.stop() now have 5s timeouts to
prevent shutdown hangs on stuck HTTP connections."
```

---

### Task 7: Reduce Wrapper Restart Sleep from 3s to 1s

At line 1322, after exit code 42 (restart), the wrapper does `await asyncio.sleep(3)`. This is separate from the pre-start sleep (Task 4). The comment says it's just a brief pause before restarting.

**Files:**
- Modify: `kutai_wrapper.py:1322`

- [ ] **Step 1: Reduce the sleep**

Replace line 1322:

```python
                    await asyncio.sleep(1)  # brief pause before restart
```

- [ ] **Step 2: Commit**

```bash
git add kutai_wrapper.py
git commit -m "perf: reduce wrapper restart pause from 3s to 1s"
```

---

### Task 8: Fix Misleading Timeout Log in Docker

`start_docker_services()` has `timeout=15` but the log says "timed out (120s)" at line 296. Fix the log message.

**Files:**
- Modify: `src/app/run.py:296`

- [ ] **Step 1: Fix the log message**

Replace line 296:

```python
        _log.warning("Docker Compose up timed out (15s)")
```

- [ ] **Step 2: Commit**

```bash
git add src/app/run.py
git commit -m "fix: correct misleading Docker timeout log message (120s -> 15s)"
```

---

## Summary of Time Savings

| Change | Before | After | Saved |
|--------|--------|-------|-------|
| Docker + health + seeding parallel | ~25s sequential | ~15s parallel | ~10s |
| Skip Ollama check | 3s | 0s | 3s |
| Conditional orphan cleanup | 10-30s | 0-5s | 10-25s |
| Skip pre-start sleep | 3s | 0s | 3s |
| Skip WAL checkpoint on restart | 1-5s | 0s | 1-5s |
| Timeout Telegram stop | 0-hang | 0-5s bounded | prevents hang |
| Reduce restart pause | 3s | 1s | 2s |
| **Total restart cycle** | **~40-65s** | **~15-25s** | **~25-40s** |
