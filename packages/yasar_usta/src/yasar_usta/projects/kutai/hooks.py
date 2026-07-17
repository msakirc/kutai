"""KutAI-specific lifecycle hooks + Turkish Messages (moved from kutai_wrapper.py)."""

from __future__ import annotations

import os
import subprocess as _sp

from yasar_usta.config import Messages

# Turkish strings that were the Messages(...) block in kutai_wrapper.py:194-223.
# The entry point applies MESSAGES to HubConfig + each target GuardConfig so the
# keyboard + crash/stop/down notifications stay Turkish (review finding #11).
MESSAGES = Messages(
    announce="🔧 *Bennn... Yaşar Usta!*\n\nKutay'ı başlatıyorum...",
    started="✅ *Kutay Started*",
    stopped="⏹ *Kutay Stopped*\nSend /start to restart.",
    hung="🔴 Kutay dondu — Yaşar Usta {delay}sn içinde yeniden başlatıyor",
    restarting="♻️ *Kutay yeniden başlatılıyor...*",
    self_restarting="🔄 *Yaşar Usta yeniden başlatılıyor...*",
    down_prompt="⚠️ Kutay durdu. Başlatmak için butona bas.",
    down_reply="⏸ Kutay şu an kapalı.",
    starting="🚀 Kutay başlatılıyor...",
    btn_status="🔧 Durum",
    btn_logs="📋 Loglar",
    btn_remote="🖥️ Claude Code",
    remote_starting="🖥️ Claude Code oturumu başlatılıyor...",
    remote_not_found="❌ `claude` command not found. Claude Code kurulu mu?",
)


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


def _kill_stale_orchestrators() -> None:
    """Kill any stale orchestrator (run.py) processes left from a previous crash."""
    my_pid = os.getpid()
    try:
        raw = _sp.check_output(
            ['wmic', 'process', 'where', "name='python.exe'",
             'get', 'ProcessId,CommandLine'],
            text=True, timeout=5,
        )
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("CommandLine"):
                continue
            if "run.py" not in line:
                continue
            pid_str = line.split()[-1]
            try:
                pid = int(pid_str)
            except ValueError:
                continue
            if pid == my_pid:
                continue
            print(f"[Yasar Usta] Killing stale orchestrator PID {pid}")
            _sp.run(['taskkill', '/F', '/PID', str(pid)],
                    capture_output=True, timeout=5)
    except Exception as e:
        print(f"[Yasar Usta] Stale orchestrator cleanup error: {e}")


def _reconcile_stray_llama() -> None:
    """Kill any llama-server NOT on the configured port (frees VRAM).

    Singleton hardening for the 2026-06-14 wrong-port orphan: a llama-server
    spawned by a dead/stale process on the wrong port (e.g. :8080 when the
    stack expects :8081) sits invisible to every port-specific check, blocking
    VRAM. This clears such strays at supervisor boot while preserving a
    healthy server already on the configured port. Fail-soft: the supervisor
    must never crash on cleanup (unlike the orchestrator, which fails loud).
    """
    raw = os.environ.get("LLAMA_SERVER_PORT")
    if raw is None:
        print("[Yasar Usta] LLAMA_SERVER_PORT unset — skipping stray-llama reconcile")
        return
    try:
        port = int(raw)
    except ValueError:
        print(f"[Yasar Usta] LLAMA_SERVER_PORT={raw!r} invalid — skipping reconcile")
        return
    try:
        from dallama.platform import PlatformHelper

        n = PlatformHelper().kill_stray_servers(port)
        if n:
            print(f"[Yasar Usta] Reconciled {n} stray llama-server(s) not on port {port}")
    except Exception as e:
        print(f"[Yasar Usta] Stray-llama reconcile error: {e}")


def pre_boot(project) -> None:
    """Runs once before KutAI's supervisor starts (was module-import cleanup)."""
    _kill_stale_orchestrators()
    _reconcile_stray_llama()


def on_exit(exit_code: int) -> None:
    _kill_orphan_processes(exit_code)
