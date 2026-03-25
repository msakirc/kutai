# run.py
import asyncio
import os
import subprocess
import sys
import time
from dotenv import load_dotenv

load_dotenv()

from src.app.config import DOCKER_CONTAINER_NAME, print_config

# ── Logging must be initialized before any other import that might log ────────
from src.infra.logging_config import init_logging, get_logger
init_logging()
_log = get_logger("app.run")

from src.core.orchestrator import Orchestrator
from src.infra.runtime_state import runtime_state, mark_degraded


# ─── Environment Check (hard abort on missing required vars) ─────────────────

def check_env():
    required = ["TELEGRAM_BOT_TOKEN", "TELEGRAM_ADMIN_CHAT_ID"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        _log.critical("Missing required env vars — create .env with TELEGRAM_BOT_TOKEN and TELEGRAM_ADMIN_CHAT_ID", vars=missing)
        sys.exit(1)

    providers = ["GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                 "GEMINI_API_KEY", "CEREBRAS_API_KEY", "SAMBANOVA_API_KEY"]
    has_cloud = any(os.getenv(p) for p in providers)

    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, timeout=3)
        has_local = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        has_local = False

    if not has_cloud and not has_local:
        _log.critical("No model providers available — set at least one API key or install Ollama")
        sys.exit(1)

    _log.info("Environment check passed", cloud_providers=has_cloud, local_models=has_local)


# ─── Startup Health Check ────────────────────────────────────────────────────

def _check(name: str, fn, critical: bool) -> bool:
    """Run one health check fn(); log result; return True on pass."""
    t0 = time.monotonic()
    try:
        ok, detail = fn()
        ms = int((time.monotonic() - t0) * 1000)
        if ok:
            _log.info("Health check passed", check=name, duration_ms=ms, detail=detail)
        elif critical:
            _log.critical("Health check FAILED (critical)", check=name, duration_ms=ms, detail=detail)
        else:
            _log.warning("Health check degraded (non-critical)", check=name, duration_ms=ms, detail=detail)
            mark_degraded(name)
        return ok
    except Exception as exc:
        ms = int((time.monotonic() - t0) * 1000)
        if critical:
            _log.critical("Health check raised (critical)", check=name, duration_ms=ms, error=str(exc))
        else:
            _log.warning("Health check raised (non-critical)", check=name, duration_ms=ms, error=str(exc))
            mark_degraded(name)
        return False


async def startup_health_check() -> bool:
    """
    Run all startup checks. Returns False if any critical check failed.
    Non-critical failures set degradation flags and continue.
    """
    import aiohttp
    from src.infra import db as _db

    critical_ok = True

    # 1. .env required vars already checked by check_env() — skip

    # 2. logs/ directory writable
    def _logs_writable():
        os.makedirs("logs", exist_ok=True)
        test = "logs/.health_check_write"
        with open(test, "w") as f:
            f.write("ok")
        os.remove(test)
        return True, "logs/ writable"

    if not _check("logs_writable", _logs_writable, critical=True):
        critical_ok = False

    # 3. DB writable
    def _db_writable():
        _db.init_db()
        return True, "DB accessible"

    if not _check("db_writable", _db_writable, critical=True):
        critical_ok = False

    # ── Non-critical checks: run concurrently (saves ~20s vs sequential) ──

    async def _async_check(name, coro, state_key=None):
        """Run one async health check with timeout, log result."""
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

    async def _ntfy():
        from src.app import config as cfg
        if not cfg.NTFY_URL:
            return False, "NTFY_URL not set"
        async with aiohttp.ClientSession() as s:
            async with s.get(cfg.NTFY_URL, timeout=aiohttp.ClientTimeout(total=5)) as r:
                return r.status < 500, f"HTTP {r.status}"

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

    await asyncio.gather(
        _async_check("ntfy", _ntfy, "ntfy_available"),
        _async_check("telegram", _telegram, "telegram_available"),
        _async_check("perplexica", _perplexica, "web_search_available"),
        _async_check("frontail", _frontail, "frontail_available"),
        _async_check("docker_sandbox", _docker_check),
    )

    # Report summary
    degraded = runtime_state["degraded_capabilities"]
    if degraded:
        _log.warning("System starting in degraded mode", degraded=degraded)
    else:
        _log.info("All health checks passed — system nominal")

    return critical_ok


# ─── Sandbox Build ────────────────────────────────────────────────────────────

def build_sandbox_if_needed():
    _log.info("Checking Docker sandbox image")
    try:
        result = subprocess.run(
            ["docker", "images", "-q", "orchestrator-sandbox"],
            capture_output=True, text=True, timeout=10
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        _log.warning("Docker check failed — shell tool will be unavailable", error=str(e))
        return False
    if not result.stdout.strip():
        _log.info("Building sandbox image")
        sandbox_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../sandbox")
        )
        try:
            build = subprocess.run(
                ["docker", "build", "-t", "orchestrator-sandbox", sandbox_dir],
                capture_output=False, timeout=120
            )
            if build.returncode != 0:
                _log.warning("Docker build failed — shell tool will be unavailable")
                return False
        except subprocess.TimeoutExpired:
            _log.warning("Docker build timed out — shell tool will be unavailable")
            return False
    return True


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    import platform
    _log.info(
        "Startup",
        python=sys.version.split()[0],
        platform=platform.system(),
        cwd=os.getcwd(),
    )

    check_env()
    print_config()
    build_sandbox_if_needed()

    critical_ok = await startup_health_check()
    if not critical_ok:
        _log.critical("Critical health checks failed — aborting")
        # Attempt ntfy alert before exit
        try:
            from src.infra.notifications import send_ntfy
            from src.app import config as cfg
            send_ntfy(cfg.NTFY_TOPIC_ERRORS, "Orchestrator failed to start",
                      "Critical health checks failed", priority=5, tags=["critical"])
        except Exception:
            pass
        sys.exit(1)

    _log.info("Starting orchestrator")

    import signal
    shutdown_event = asyncio.Event()

    def _signal_handler(sig, frame):
        _log.info("Shutdown signal received", signal=signal.Signals(sig).name)
        shutdown_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Phase 12.1: Start API server in background if uvicorn available
    api_port = int(os.getenv("API_PORT", "8000"))
    try:
        try:
            from .api import start_api_server
        except ImportError:
            # Running as script, not as package — use absolute import
            from src.app.api import start_api_server
        api_task = asyncio.create_task(
            start_api_server(host="0.0.0.0", port=api_port),
            name="api_server",
        )
        _log.info("API server task created", port=api_port)
    except Exception as exc:
        _log.warning("API server not started", reason=str(exc))
        api_task = None

    # Phase 14.3: Start monitoring loop in background
    monitor_task = None
    try:
        from ..infra.monitoring import run_monitoring_loop
        monitor_task = asyncio.create_task(
            run_monitoring_loop(),
            name="monitoring_loop",
        )
        _log.info("Monitoring loop started")
    except Exception as exc:
        _log.debug("Monitoring loop not started", reason=str(exc))

    # Phase 3: Start GPU auto-detect loop
    gpu_detect_task = None
    try:
        from ..infra.load_manager import run_gpu_autodetect_loop

        async def _notify_gpu_change(msg: str):
            try:
                from .config import TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_CHAT_ID
                if not TELEGRAM_BOT_TOKEN or not TELEGRAM_ADMIN_CHAT_ID:
                    return
                import aiohttp
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                async with aiohttp.ClientSession() as s:
                    await s.post(url, json={
                        "chat_id": TELEGRAM_ADMIN_CHAT_ID,
                        "text": msg,
                        "parse_mode": "Markdown",
                    }, timeout=aiohttp.ClientTimeout(total=5))
            except Exception:
                pass

        gpu_detect_task = asyncio.create_task(
            run_gpu_autodetect_loop(notify_fn=_notify_gpu_change),
            name="gpu_autodetect_loop",
        )
        _log.info("GPU auto-detect loop started")
    except Exception as exc:
        _log.debug("GPU auto-detect loop not started", reason=str(exc))

    orch = Orchestrator(shutdown_event=shutdown_event)
    await orch.start()

    if api_task and not api_task.done():
        api_task.cancel()
    if monitor_task and not monitor_task.done():
        monitor_task.cancel()
    if gpu_detect_task and not gpu_detect_task.done():
        gpu_detect_task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except SystemExit as e:
        # Propagate exit code (42 = restart, 0 = stop) to the wrapper
        sys.exit(e.code)
