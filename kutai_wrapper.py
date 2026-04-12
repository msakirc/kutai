#!/usr/bin/env python3
"""
Yaşar Usta — KutAI'nin süreç yöneticisi (process manager).

Thin consumer of the yasar-usta package, configured for KutAI.
"""
import asyncio
import os
import signal
import subprocess as _sp
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

from yasar_usta import ProcessGuard, GuardConfig, Messages, SidecarConfig

PROJECT_ROOT = Path(__file__).resolve().parent


def _find_python() -> str:
    venv = PROJECT_ROOT / ".venv"
    if sys.platform == "win32":
        p = venv / "Scripts" / "python.exe"
    else:
        p = venv / "bin" / "python"
    return str(p) if p.exists() else sys.executable


def _kill_orphan_processes(exit_code: int) -> None:
    """Kill orphaned llama-server after orchestrator exits (KutAI-specific)."""
    if exit_code == 42:
        return  # Clean restart — don't kill llama-server

    targets = [
        ("llama-server.exe", "llama-server"),
        ("ollama.exe", "Ollama"),
        ("ollama_llama_server.exe", "Ollama runner"),
    ]
    for exe, label in targets:
        try:
            check = _sp.run(
                ["tasklist", "/FI", f"IMAGENAME eq {exe}", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            if exe.lower() not in check.stdout.lower():
                continue
            result = _sp.run(
                ["taskkill", "/F", "/IM", exe],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                print(f"[Yasar Usta] Killed orphaned {label}: {result.stdout.strip()}")
        except Exception as e:
            print(f"[Yasar Usta] {label} cleanup error: {e}")


venv_python = _find_python()

config = GuardConfig(
    name="Yaşar Usta",
    app_name="Kutay",
    command=[venv_python, str(PROJECT_ROOT / "src" / "app" / "run.py")],
    cwd=str(PROJECT_ROOT),

    telegram_token=os.getenv("YASAR_USTA_BOT_TOKEN", ""),
    telegram_chat_id=os.getenv("TELEGRAM_ADMIN_CHAT_ID", ""),

    backoff_steps=[5, 15, 60, 300],
    backoff_reset_after=600,

    heartbeat_file=str(PROJECT_ROOT / "logs" / "orchestrator.heartbeat"),
    heartbeat_stale_seconds=120,
    heartbeat_healthy_seconds=90,

    restart_exit_code=42,
    log_dir=str(PROJECT_ROOT / "logs"),
    log_file=str(PROJECT_ROOT / "logs" / "orchestrator.jsonl"),
    stop_timeout=30,
    auto_restart="--no-auto-restart" not in sys.argv,

    claude_enabled=True,
    claude_name="Kutay",
    claude_signal_file=str(PROJECT_ROOT / "logs" / "claude_remote.signal"),

    sidecar=SidecarConfig(
        name="yazbunu",
        command=[venv_python, "-m", "yazbunu.server",
                 "--log-dir", "./logs", "--port", "9880", "--host", "0.0.0.0"],
        health_url="http://127.0.0.1:9880/health",
        pid_file=str(PROJECT_ROOT / "logs" / "yazbunu.pid"),
        detached=True,
        auto_start=True,
    ),

    on_exit=_kill_orphan_processes,

    extra_processes=[
        {"exe": "llama-server.exe", "label": "llama-server"},
    ],

    messages=Messages(
        announce="🔧 *Bennn... Yaşar Usta!*\n\nKutay'ı başlatıyorum...",
        started="✅ *Kutay Started*",
        stopped="⏹ *Kutay Stopped*\nSend /kutai\\_start to restart.",
        crash=(
            "🔴 *Kutay Crashed*\n"
            "Exit code: `{exit_code}`\n"
            "Crash #{crash_count}\n"
            "Restarting in {backoff}s\n\n"
            "```\n{stderr}\n```"
        ),
        hung="🔴 Kutay dondu — Yaşar Usta {delay}sn içinde yeniden başlatıyor",
        restarting="♻️ *Kutay yeniden başlatılıyor...*",
        self_restarting="🔄 *Yaşar Usta yeniden başlatılıyor...*",
        down_prompt="⚠️ Kutay durdu. Başlatmak için butona bas.",
        down_reply="⏸ Kutay şu an kapalı.",
        starting="🚀 Kutay başlatılıyor...",
        btn_start="▶️ {app_name}'ı Başlat",
        btn_status="🔧 Durum",
        btn_system="⚙️ Sistem",
        btn_restart="🔄 {app_name}'ı Yeniden Başlat",
        btn_stop="⏹ {app_name}'ı Durdur",
        btn_logs="📋 Loglar",
        btn_remote="🖥️ Claude Code",
        btn_refresh="🔄 Yenile",
        btn_restart_guard="♻️ Usta'yı Yeniden Başlat",
        btn_restart_sidecar="📊 Yazbunu Yeniden Başlat",
        remote_starting="🖥️ Claude Code oturumu başlatılıyor...",
        remote_not_found="❌ `claude` command not found. Claude Code kurulu mu?",
    ),
)


async def main():
    guard = ProcessGuard(config)

    def _sig(sig, frame):
        print(f"\n[Yasar Usta] Signal {sig} received, shutting down...")
        guard.request_shutdown()

    signal.signal(signal.SIGINT, _sig)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _sig)

    if sys.platform == "win32":
        try:
            import ctypes

            @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_ulong)
            def _console_handler(event):
                if event in (0, 2):
                    guard.request_shutdown()
                    return True
                return False

            ctypes.windll.kernel32.SetConsoleCtrlHandler(_console_handler, True)
            guard._console_handler = _console_handler
        except Exception:
            pass

    await guard.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[Yasar Usta] KeyboardInterrupt — exiting")
    except Exception as exc:
        print(f"[Yasar Usta] FATAL: {exc!r}")
        raise
