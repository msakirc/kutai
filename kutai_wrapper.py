#!/usr/bin/env python3
"""
Yaşar Usta — KutAI'nin süreç yöneticisi (process manager).

Thin consumer of the yasar-usta package, configured for KutAI.
"""
import asyncio
import os
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Venv guard ──
_EXPECTED_VENV = Path(__file__).parent / ".venv"
_in_venv = hasattr(sys, "real_prefix") or (
    hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
)
if not _in_venv and _EXPECTED_VENV.exists():
    print(f"ERROR: Running with system Python ({sys.executable})")
    print(f"Use: .venv\\Scripts\\python.exe kutai_wrapper.py")
    sys.exit(1)

from yasar_usta import Hub, load_registry

PROJECT_ROOT = Path(__file__).resolve().parent


def _apply_runtime_values(hub_cfg, projects) -> None:
    """Inject process-specific runtime values the declarative registry can't hold,
    plus the KutAI Turkish Messages (review finding #11)."""
    from yasar_usta.hooks import load_hook
    appdata = os.environ.get("APPDATA", "")
    claude_cmd = str(Path(appdata) / "npm" / "claude.cmd") if appdata else None
    auto_restart = "--no-auto-restart" not in sys.argv
    db_path = os.getenv("DB_PATH", str(PROJECT_ROOT / "data" / "kutai.db"))
    for proj in projects:
        hook = load_hook(proj.hook_module)
        msgs = getattr(hook, "MESSAGES", None)
        if msgs is not None:
            hub_cfg.messages = msgs  # hub keyboard + announce use these
        for tgt in proj.targets:
            tgt.auto_restart = auto_restart
            if msgs is not None:
                tgt.messages = msgs  # crash/stop/down notifications stay Turkish
            if claude_cmd:
                tgt.claude_cmd = claude_cmd
            # nerd_herd sidecar needs --db-path appended (kept out of YAML)
            for sc in tgt.sidecars:
                if sc.name == "nerd_herd" and "--db-path" not in sc.command:
                    sc.command += ["--db-path", db_path]


async def main():
    hub_cfg, projects = load_registry(PROJECT_ROOT / "registry.yaml",
                                      project_root=str(PROJECT_ROOT))
    _apply_runtime_values(hub_cfg, projects)
    hub = Hub(hub_cfg, projects)

    def _sig(sig, frame):
        print(f"\n[Yasar Usta] Signal {sig} received, shutting down...")
        hub.request_shutdown()

    signal.signal(signal.SIGINT, _sig)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _sig)

    if sys.platform == "win32":
        try:
            import ctypes

            @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_ulong)
            def _console_handler(event):
                if event in (0, 2):
                    hub.request_shutdown()
                    return True
                return False

            ctypes.windll.kernel32.SetConsoleCtrlHandler(_console_handler, True)
            hub._console_handler = _console_handler  # GC anchor
        except Exception:
            pass

    await hub.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[Yasar Usta] KeyboardInterrupt — exiting")
    except Exception as exc:
        print(f"[Yasar Usta] FATAL: {exc!r}")
        raise
