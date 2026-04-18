# run.py
import asyncio
import io
import os
import signal as _signal
import subprocess
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path so `from src.…` imports work
# regardless of how this script is launched.
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Prevent UnicodeEncodeError on Windows consoles using cp1252 / legacy codepages
if sys.stdout and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

from src.app.config import DOCKER_CONTAINER_NAME, print_config

# ── Logging must be initialized before any other import that might log ────────
from src.infra.logging_config import init_logging, get_logger
init_logging(
    log_dir="logs",
    project="kutai",
    package_logs={
        # dallama uses logging.getLogger(__name__) → "dallama.*"
        "dallama": "dallama.jsonl",
        # hallederiz_kadir uses get_logger() → "kutai.hallederiz_kadir",
        # with fallback to logging.getLogger() → "hallederiz_kadir"
        "kutai.hallederiz_kadir": "hallederiz_kadir.jsonl",
        "hallederiz_kadir": "hallederiz_kadir.jsonl",
        # LiteLLM and openai log from within hallederiz_kadir calls
        "LiteLLM": "hallederiz_kadir.jsonl",
        "openai": "hallederiz_kadir.jsonl",
        # nerd_herd uses get_logger() → "kutai.nerd_herd.*"
        "kutai.nerd_herd": "nerd_herd.jsonl",
        # vecihi uses logging.getLogger("vecihi")
        "vecihi": "vecihi.jsonl",
        # kuleden_donen_var uses logging.getLogger(__name__) → "kuleden_donen_var.*"
        "kuleden_donen_var": "kuleden_donen_var.jsonl",
    },
)
_log = get_logger("app.run")

# Attach Telegram alert handler (ERROR+) to root logger
import logging as _logging
try:
    from src.infra.notifications import TelegramAlertHandler
    _logging.getLogger().addHandler(TelegramAlertHandler())
except Exception as e:
    _log = get_logger("app.run")
    _log.warning("Could not attach TelegramAlertHandler", error=str(e))

from nerd_herd.client import NerdHerdClient, get_default as get_nerd_herd, set_default as _set_nerd_herd_default

_nerd_herd: NerdHerdClient | None = None

_NERD_HERD_PID_FILE = os.path.join("logs", "nerd_herd.pid")

def _restart_stale_sidecar() -> None:
    """Kill a stale NerdHerd sidecar via its PID file.

    Yaşar Usta's ``ensure()`` cycle will notice it's gone and restart it
    with the current code.
    """
    _rlog = get_logger("app.run")
    try:
        pid = int(open(_NERD_HERD_PID_FILE).read().strip())
        os.kill(pid, _signal.SIGTERM)
        _rlog.info(
            "Killed stale NerdHerd sidecar (version mismatch, not a crash) "
            "— Yaşar Usta will restart it with current code",
            pid=pid,
        )
    except FileNotFoundError:
        _rlog.debug("No NerdHerd PID file found — sidecar may not be running")
    except (ValueError, OSError) as exc:
        _rlog.warning("Could not kill stale NerdHerd sidecar", error=str(exc))

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

    # Check for llama-server models (started lazily by local_model_manager)
    model_dir = os.getenv("MODEL_DIR", "")
    has_llama = False
    if model_dir and os.path.isdir(model_dir):
        import glob
        has_llama = bool(glob.glob(os.path.join(model_dir, "**", "*.gguf"), recursive=True))

    # Check for Ollama (skip if explicitly disabled or llama models found)
    has_ollama = False
    if not os.getenv("OLLAMA_DISABLED") and not has_llama:
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, timeout=3)
            has_ollama = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    has_local = has_llama or has_ollama

    if not has_cloud and not has_local:
        _log.critical("No model providers available — set at least one API key, "
                      "configure MODEL_DIR with GGUF files, or install Ollama")
        sys.exit(1)

    _log.info("Environment check passed", cloud_providers=has_cloud,
              local_models=has_local, llama_models=has_llama, ollama=has_ollama)


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


async def _critical_health_checks() -> bool:
    """
    Run critical startup checks (logs writable + DB accessible).
    Returns False if any critical check failed — caller should abort.
    """
    from src.infra import db as _db

    critical_ok = True

    # 1. logs/ directory writable
    def _logs_writable():
        os.makedirs("logs", exist_ok=True)
        test = "logs/.health_check_write"
        with open(test, "w") as f:
            f.write("ok")
        os.remove(test)
        return True, "logs/ writable"

    if not _check("logs_writable", _logs_writable, critical=True):
        critical_ok = False

    # 2. DB writable (retry — old process may still hold the lock during restart)
    async def _db_writable():
        await _db.init_db()
        return True, "DB accessible"

    db_ok = False
    for attempt in range(3):
        try:
            ok, detail = await _db_writable()
            _log.info("Health check passed", check="db_writable", detail=detail)
            db_ok = True
            try:
                from fatih_hoca.selector import enable_telemetry
                from src.app.config import DB_PATH as _DB_PATH
                enable_telemetry(_DB_PATH)
            except Exception:
                pass  # telemetry is optional
            break
        except Exception as exc:
            if attempt < 2:
                _log.warning("DB locked, retrying in 1s...",
                             check="db_writable", attempt=attempt + 1)
                await _db.close_db()  # release partial connection
                await asyncio.sleep(1)
            else:
                _log.critical("Health check raised (critical)",
                              check="db_writable", error=str(exc))
    if not db_ok:
        critical_ok = False

    return critical_ok


async def _noncritical_health_checks():
    """
    Run non-critical health checks concurrently.
    Failures set degradation flags but do not abort startup.
    """
    import aiohttp

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

    async def _telegram():
        from src.app import config as cfg
        token = cfg.TELEGRAM_BOT_TOKEN
        if not token:
            return False, "TELEGRAM_BOT_TOKEN not set"
        url = f"https://api.telegram.org/bot{token}/getMe"
        async with aiohttp.ClientSession() as s:
            async with s.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                return r.status == 200, f"HTTP {r.status}"

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
        """Verify critical Python dependencies are importable."""
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
        # Check optional deps — just log, don't degrade
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
        _async_check("docker_sandbox", _docker_check),
        _async_check("python_deps", _check_deps),
    )

    # Report summary
    degraded = runtime_state["degraded_capabilities"]
    if degraded:
        _log.warning("System starting in degraded mode", degraded=degraded)
    else:
        _log.info("All health checks passed — system nominal")


# ─── Docker Services ─────────────────────────────────────────────────────────

async def start_docker_services():
    """Bring up all services defined in docker-compose.yml."""
    compose_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../docker-compose.yml")
    )
    if not os.path.exists(compose_file):
        _log.warning("docker-compose.yml not found", path=compose_file)
        return False

    _log.info("Starting Docker Compose services")
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                ["docker", "compose", "-f", compose_file, "up", "-d"],
                capture_output=True, text=True, timeout=15,
            ),
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
    except subprocess.TimeoutExpired:
        _log.warning("Docker Compose up timed out (15s)")
        return False


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    import platform
    _log.info(
        "Startup",
        python=sys.version.split()[0],
        platform=platform.system(),
        cwd=os.getcwd(),
    )

    # Write heartbeat immediately and keep it alive during startup.
    # The old heartbeat file may be >120s stale from a previous hung
    # instance.  Startup (docker, health checks, model load) can take
    # 60-120s — without periodic heartbeats the wrapper kills us.
    from yasar_usta import HeartbeatWriter, write_heartbeat
    _hb_paths = ("logs/orchestrator.heartbeat", "logs/heartbeat")
    write_heartbeat(*_hb_paths)
    _hb_writer = HeartbeatWriter(*_hb_paths, interval=15.0)
    _hb_task = asyncio.create_task(_hb_writer.run())

    _log.info("Running check_env...")
    check_env()
    _log.info("check_env done, running print_config...")
    print_config()

    # Phase 1: Critical health checks (gate — abort if fails)
    critical_ok = await _critical_health_checks()
    if not critical_ok:
        _log.critical("Critical health checks failed — aborting")
        sys.exit(1)

    # Phase 2: Docker, non-critical checks, and seeding ALL in parallel
    async def _docker_phase():
        docker_ok = await start_docker_services()
        if not docker_ok:
            _log.warning("Docker services unavailable — sandbox, monitoring will not work")

    async def _seed_skills():
        try:
            from src.memory.seed_skills import seed_skills
            added = await seed_skills()
            if added:
                _log.info(f"Seeded {added} new routing skills")
        except Exception as e:
            _log.warning(f"Skill seeding failed (non-critical): {e}")

    async def _seed_api_index():
        try:
            from src.tools.free_apis import seed_registry, build_keyword_index, seed_category_patterns
            await seed_registry()
            await build_keyword_index()
            await seed_category_patterns()
        except Exception as exc:
            _log.warning("API keyword index seeding failed (non-critical): %s", exc)

    await asyncio.gather(
        _docker_phase(),
        _noncritical_health_checks(),
        _seed_skills(),
        _seed_api_index(),
    )

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
        from src.infra.monitoring import run_monitoring_loop
        monitor_task = asyncio.create_task(
            run_monitoring_loop(),
            name="monitoring_loop",
        )
        _log.info("Monitoring loop started")
    except Exception as exc:
        _log.debug("Monitoring loop not started", reason=str(exc))

    # Phase 3: Connect to NerdHerd sidecar (managed by Yaşar Usta)
    try:
        global _nerd_herd
        _nerd_herd = NerdHerdClient(port=9881)

        # Version handshake — restart stale sidecar so new endpoints work
        if not await _nerd_herd.check_version():
            _log.warning("NerdHerd sidecar is stale — killing so Yaşar Usta restarts it")
            _restart_stale_sidecar()
            # Give Yaşar Usta's ensure() cycle time to restart it
            await asyncio.sleep(5)
            if not await _nerd_herd.check_version():
                _log.warning("NerdHerd sidecar still stale after restart — running without it")
                _nerd_herd = None

        if _nerd_herd is not None:
            _mode = await _nerd_herd.get_load_mode()
            _log.info("Connected to NerdHerd sidecar", load_mode=_mode)
    except Exception as exc:
        _log.warning("NerdHerd client init failed — running without GPU monitoring",
                     error=str(exc))
        _nerd_herd = None

    _set_nerd_herd_default(_nerd_herd)

    # Wire NerdHerd client into LocalModelManager so _on_ready can push state
    try:
        from src.models.local_model_manager import get_local_manager
        get_local_manager().set_nerd_herd(_nerd_herd)
        _log.debug("LocalModelManager wired to NerdHerd client")
    except Exception as exc:
        _log.debug("Could not wire NerdHerd to LocalModelManager", error=str(exc))

    # Phase 3b: Initialize Fatih Hoca model selection with NerdHerd snapshot
    try:
        import fatih_hoca
        from pathlib import Path as _Path
        from src.app.config import AVAILABLE_KEYS

        _catalog = str(_Path(__file__).resolve().parent.parent / "models" / "models.yaml")
        _models_dir = os.getenv("MODEL_DIR", "") or None
        _providers = {p for p, ok in AVAILABLE_KEYS.items() if ok}

        # Seed the snapshot cache before init so first select() has VRAM data
        if _nerd_herd:
            try:
                await _nerd_herd.refresh_snapshot()
            except Exception:
                pass

        _fh_models = fatih_hoca.init(
            catalog_path=_catalog,
            models_dir=_models_dir,
            nerd_herd=_nerd_herd,  # has sync snapshot() returning cached value
            available_providers=_providers,
        )
        _log.info("Fatih Hoca initialized", model_count=len(_fh_models),
                   cloud_providers=sorted(_providers) if _providers else "none")
    except Exception as exc:
        _log.error("Fatih Hoca init failed — model selection degraded", error=str(exc))

    # Phase 3c: Background snapshot refresh (keeps Fatih Hoca's cached snapshot fresh)
    _snapshot_task = None
    if _nerd_herd:
        async def _snapshot_refresh_loop():
            while True:
                try:
                    await asyncio.sleep(2)
                    await _nerd_herd.refresh_snapshot()
                except asyncio.CancelledError:
                    break
                except Exception:
                    pass  # sidecar temporarily unreachable — use stale cache

        _snapshot_task = asyncio.create_task(
            _snapshot_refresh_loop(), name="snapshot_refresh"
        )

    # Cancel startup heartbeat — the orchestrator's own _heartbeat_loop
    # takes over once start() creates its background tasks.
    _hb_task.cancel()

    orch = Orchestrator(shutdown_event=shutdown_event)
    await orch.start()

    if api_task and not api_task.done():
        api_task.cancel()
    if monitor_task and not monitor_task.done():
        monitor_task.cancel()
    if _snapshot_task and not _snapshot_task.done():
        _snapshot_task.cancel()
    if _nerd_herd:
        await _nerd_herd.close()

    # Propagate exit code to wrapper (EXIT_RESTART=42, EXIT_STOP=0).
    # Use sys.exit() so that atexit handlers (llama-server cleanup) still run.
    # The orchestrator's finally block has already stopped llama-server by this
    # point, but atexit provides a second safety net.
    if orch.requested_exit_code is not None:
        sys.exit(orch.requested_exit_code)


if __name__ == "__main__":
    asyncio.run(main())
