#!/usr/bin/env python3
"""KutAI lifecycle hooks, invoked by the Yaşar Usta hub as a subprocess in
kutay's OWN venv: python yasar_hooks.py <pre_boot|on_exit> --context <json>.
Free to import dallama/psutil (kutay venv)."""
import argparse
import json
import os
import subprocess as _sp


def _norm(s: str) -> str:
    return s.replace("\\", "/").lower()


def _iter_python_processes():
    """Yield (pid, cmdline) for python processes. Resilient to psutil races:
    process_iter() is called WITHOUT an attrs list (so it never triggers the
    batch as_dict that raises when a process vanishes mid-scan), and every
    per-process attribute read is guarded — a dying process must never crash
    pre_boot (which fails loud and would otherwise abort hub startup)."""
    import psutil
    try:
        procs = psutil.process_iter()
    except Exception:
        return
    for p in procs:
        try:
            if "python" not in (p.name() or "").lower():
                continue
            yield (p.pid, " ".join(p.cmdline() or []))
        except Exception:
            continue


def _kill_pid(pid) -> None:
    try:
        import psutil
        psutil.Process(pid).kill()
    except Exception as e:
        print(f"[yasar_hooks] kill {pid} failed: {e}")


def _kill_stale_orchestrators(script_paths) -> None:
    paths = {_norm(p) for p in script_paths if p.lower().endswith(".py")}
    if not paths:
        return
    my_pid = os.getpid()
    # Also exclude the parent (the hub process that spawned us — never a stale orch)
    try:
        parent_pid = os.getppid()
    except Exception:
        parent_pid = None
    safe_pids = {my_pid, parent_pid} if parent_pid else {my_pid}
    for pid, cmdline in _iter_python_processes():
        if pid in safe_pids:
            continue
        if any(sp in _norm(cmdline) for sp in paths):
            print(f"[yasar_hooks] killing stale orchestrator PID {pid}")
            _kill_pid(pid)


def _reconcile_stray_llama() -> None:
    raw = os.environ.get("LLAMA_SERVER_PORT")
    if raw is None:
        print("[yasar_hooks] LLAMA_SERVER_PORT unset — skipping stray-llama reconcile")
        return
    try:
        port = int(raw)
    except ValueError:
        print(f"[yasar_hooks] LLAMA_SERVER_PORT={raw!r} invalid — skipping")
        return
    try:
        from dallama.platform import PlatformHelper
        n = PlatformHelper().kill_stray_servers(port)
        if n:
            print(f"[yasar_hooks] reconciled {n} stray llama-server(s) not on {port}")
    except Exception as e:
        print(f"[yasar_hooks] stray-llama reconcile error: {e}")


def _kill_orphan_processes(exit_code: int) -> None:
    if exit_code == 42:
        return
    for exe, label in (("llama-server.exe", "llama-server"),
                       ("ollama.exe", "Ollama"),
                       ("ollama_llama_server.exe", "Ollama runner")):
        try:
            check = _sp.run(["tasklist", "/FI", f"IMAGENAME eq {exe}", "/NH"],
                            capture_output=True, text=True, timeout=5)
            if exe.lower() not in check.stdout.lower():
                continue
            r = _sp.run(["taskkill", "/F", "/IM", exe], capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                print(f"[yasar_hooks] killed orphaned {label}")
        except Exception as e:
            print(f"[yasar_hooks] {label} cleanup error: {e}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["pre_boot", "on_exit"])
    ap.add_argument("--context", default="{}")
    args = ap.parse_args(argv)
    ctx = json.loads(args.context)
    if args.phase == "pre_boot":
        _kill_stale_orchestrators(ctx.get("script_paths", []))
        _reconcile_stray_llama()
    elif args.phase == "on_exit":
        _kill_orphan_processes(int(ctx.get("exit_code") or 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
