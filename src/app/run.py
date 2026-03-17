# run.py
import asyncio
import os
import subprocess
import sys
import time
from dotenv import load_dotenv
load_dotenv()

# ── Logging must be initialized before any other import that might log ────────
from src.infra.logging_config import init_logging, get_logger
init_logging()
_log = get_logger("app.run")

from .config import print_config, DOCKER_CONTAINER_NAME
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

    # 4. ntfy reachable (non-critical)
    async def _ntfy():
        from src.app import config as cfg
        if not cfg.NTFY_URL:
            return False, "NTFY_URL not set"
        async with aiohttp.ClientSession() as s:
            async with s.get(cfg.NTFY_URL, timeout=aiohttp.ClientTimeout(total=5)) as r:
                ok = r.status < 500
                runtime_state["ntfy_available"] = ok
                return ok, f"HTTP {r.status}"

    try:
        ok, detail = await asyncio.wait_for(_ntfy(), timeout=6)
        ms = 0
        if ok:
            _log.info("Health check passed", check="ntfy", detail=detail)
        else:
            _log.warning("Health check degraded", check="ntfy", detail=detail)
            mark_degraded("ntfy")
    except Exception as exc:
        _log.warning("Health check raised (non-critical)", check="ntfy", error=str(exc))
        mark_degraded("ntfy")

    # 5. Docker sandbox alive (non-critical)
    def _docker():
        r = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", DOCKER_CONTAINER_NAME],
            capture_output=True, text=True, timeout=5
        )
        running = r.stdout.strip() == "true"
        runtime_state["sandbox_available"] = running
        return running, "running" if running else "not running"

    if not _check("docker_sandbox", _docker, critical=False):
        pass  # already marked degraded inside _check

    # 6. Telegram reachable (non-critical)
    async def _telegram():
        import aiohttp as _aio
        from src.app import config as cfg
        token = cfg.TELEGRAM_BOT_TOKEN
        if not token:
            return False, "TELEGRAM_BOT_TOKEN not set"
        url = f"https://api.telegram.org/bot{token}/getMe"
        async with _aio.ClientSession() as s:
            async with s.get(url, timeout=_aio.ClientTimeout(total=5)) as r:
                ok = r.status == 200
                runtime_state["telegram_available"] = ok
                return ok, f"HTTP {r.status}"

    try:
        ok, detail = await asyncio.wait_for(_telegram(), timeout=6)
        if ok:
            _log.info("Health check passed", check="telegram", detail=detail)
        else:
            _log.warning("Health check degraded", check="telegram", detail=detail)
            mark_degraded("telegram")
    except Exception as exc:
        _log.warning("Health check raised (non-critical)", check="telegram", error=str(exc))
        mark_degraded("telegram")

    # 7. Perplexica (non-critical)
    async def _perplexica():
        import aiohttp as _aio
        url = os.getenv("PERPLEXICA_URL", "")
        if not url:
            return False, "PERPLEXICA_URL not set"
        async with _aio.ClientSession() as s:
            async with s.get(url, timeout=_aio.ClientTimeout(total=5)) as r:
                ok = r.status < 500
                runtime_state["web_search_available"] = ok
                return ok, f"HTTP {r.status}"

    try:
        ok, detail = await asyncio.wait_for(_perplexica(), timeout=6)
        if ok:
            _log.info("Health check passed", check="perplexica", detail=detail)
        else:
            _log.warning("Health check degraded", check="perplexica", detail=detail)
            mark_degraded("web_search")
    except Exception as exc:
        _log.warning("Health check raised (non-critical)", check="perplexica", error=str(exc))
        mark_degraded("web_search")

    # 8. Frontail (non-critical)
    async def _frontail():
        import aiohttp as _aio
        async with _aio.ClientSession() as s:
            async with s.get("http://localhost:9001", timeout=_aio.ClientTimeout(total=3)) as r:
                ok = r.status < 500
                runtime_state["frontail_available"] = ok
                return ok, f"HTTP {r.status}"

    try:
        ok, detail = await asyncio.wait_for(_frontail(), timeout=4)
        if ok:
            _log.info("Health check passed", check="frontail", detail=detail)
        else:
            _log.warning("Health check degraded", check="frontail", detail=detail)
            mark_degraded("frontail")
    except Exception as exc:
        _log.warning("Health check raised (non-critical)", check="frontail", error=str(exc))
        mark_degraded("frontail")

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
    result = subprocess.run(
        ["docker", "images", "-q", "orchestrator-sandbox"],
        capture_output=True, text=True
    )
    if not result.stdout.strip():
        _log.info("Building sandbox image")
        sandbox_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../sandbox")
        )
        build = subprocess.run(
            ["docker", "build", "-t", "orchestrator-sandbox", sandbox_dir],
            capture_output=False
        )
        if build.returncode != 0:
            _log.warning("Docker build failed — shell tool will be unavailable")
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

    orch = Orchestrator(shutdown_event=shutdown_event)
    await orch.start()


if __name__ == "__main__":
    asyncio.run(main())
