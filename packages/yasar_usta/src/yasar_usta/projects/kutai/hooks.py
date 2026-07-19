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


def _norm(s: str) -> str:
    """Normalize a path/cmdline for slash-direction- and case-insensitive match."""
    return s.replace("\\", "/").lower()


def _orchestrator_script_paths(project) -> set:
    """The absolute run-script path(s) that identify THIS project's orchestrator,
    taken from each target's command (the ``.py`` arg). Project-specific + path-
    specific, so it never matches library run.py or another project."""
    paths = set()
    for tgt in getattr(project, "targets", []) or []:
        for arg in getattr(tgt, "command", []) or []:
            if isinstance(arg, str) and arg.lower().endswith(".py"):
                paths.add(_norm(arg))
    return paths


def _stale_orchestrator_pids(script_paths, processes, my_pid) -> list:
    """PIDs whose cmdline contains one of ``script_paths`` (excluding my_pid).
    ``processes`` is an iterable of ``(pid, cmdline_str)``."""
    out = []
    for pid, cmdline in processes:
        if pid == my_pid:
            continue
        cl = _norm(cmdline)
        if any(sp in cl for sp in script_paths):
            out.append(pid)
    return out


def _iter_python_processes():
    """Yield (pid, cmdline) for every python process (psutil; no deprecated wmic)."""
    import psutil
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if "python" not in (p.info.get("name") or "").lower():
                continue
            yield (p.info["pid"], " ".join(p.info.get("cmdline") or []))
        except Exception:
            continue


def _kill_pid(pid) -> None:
    try:
        import psutil
        psutil.Process(pid).kill()
    except Exception as e:
        print(f"[Yasar Usta] kill {pid} failed: {e}")


def _kill_stale_orchestrators(project, *, list_processes=None, kill=None) -> None:
    """Kill orphaned orchestrator processes left by a previous hub crash, matched
    by the target's ABSOLUTE run-script path — not a bare ``"run.py"`` substring
    (which force-killed torch/pexpect/openai/watchfiles run.py and, once
    multi-project, sibling projects' orchestrators). Fail-soft."""
    list_processes = list_processes or _iter_python_processes
    kill = kill or _kill_pid
    script_paths = _orchestrator_script_paths(project)
    if not script_paths:
        return
    try:
        procs = list(list_processes())
    except Exception as e:
        print(f"[Yasar Usta] Stale orchestrator scan error: {e}")
        return
    for pid in _stale_orchestrator_pids(script_paths, procs, os.getpid()):
        print(f"[Yasar Usta] Killing stale orchestrator PID {pid}")
        kill(pid)


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
    _kill_stale_orchestrators(project)
    _reconcile_stray_llama()


def on_exit(exit_code: int) -> None:
    _kill_orphan_processes(exit_code)
