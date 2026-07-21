"""Outer hub-liveness watchdog — the code a Task-Scheduler task runs to catch a
HUNG-but-alive hub (which the scheduler's restart-on-failure can't see, since
the process is still "running").

The hub writes ``hub.alive`` (a timestamp) on a fixed cadence, decoupled from
its crash/backoff loop. This watchdog, run every few minutes:
  read hub.alive → if stale AND a hub process is still alive → kill it
so the main task's restart-on-failure relaunches a fresh hub. A DEAD hub is left
alone (nothing to kill; the scheduler's exit-code handling owns that).

Design: docs/superpowers/specs/2026-07-17-yasar-usta-always-live-singleton-design.md §7.
Usage (Task Scheduler action, every 3 min):
    python -m yasar_usta.watchdog --alive <path-to-hub.alive>
"""

from __future__ import annotations

from pathlib import Path

# Threshold > the 300s max backoff step, so a hub legitimately sleeping between
# orchestrator respawns is never mistaken for a hang.
DEFAULT_STALE_SECONDS = 360
DEFAULT_INTERVAL_SECONDS = 180


def read_alive_ts(path) -> float | None:
    """The last hub.alive timestamp, or None if missing/corrupt."""
    try:
        return float(Path(path).read_text().strip())
    except Exception:
        return None


def is_stale(ts, now: float, threshold: float = DEFAULT_STALE_SECONDS) -> bool:
    return ts is not None and (now - ts) > threshold


def decide_kill(ts, now: float, hub_pids, threshold: float = DEFAULT_STALE_SECONDS) -> list:
    """PIDs to kill: the live hub processes, but ONLY when hub.alive is stale.
    Fresh / missing / no-live-process → kill nothing."""
    if not is_stale(ts, now, threshold):
        return []
    return list(hub_pids)


def find_hub_pids() -> list:
    """Live hub processes (kutai_wrapper.py). psutil; skips inaccessible procs."""
    import psutil
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if "python" not in (p.info.get("name") or "").lower():
                continue
            if "kutai_wrapper.py" in " ".join(p.info.get("cmdline") or []):
                out.append(p.info["pid"])
        except Exception:
            continue
    return out


def kill_pid(pid) -> None:
    try:
        import psutil
        psutil.Process(pid).kill()
    except Exception:
        pass


def run_once(alive_path, now: float, *, threshold: float = DEFAULT_STALE_SECONDS,
             find_pids=find_hub_pids, kill=kill_pid) -> list:
    """One watchdog tick. Returns the PIDs it killed (empty if the hub is fresh
    or dead)."""
    ts = read_alive_ts(alive_path)
    to_kill = decide_kill(ts, now, list(find_pids()), threshold)
    for pid in to_kill:
        print(f"[Yasar Watchdog] hub hung (alive stale) — killing PID {pid}")
        kill(pid)
    return to_kill


def main(argv=None) -> int:
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Yaşar Usta hub-liveness watchdog")
    ap.add_argument("--alive", required=True, help="path to hub.alive")
    ap.add_argument("--threshold", type=float, default=DEFAULT_STALE_SECONDS)
    args = ap.parse_args(argv)
    killed = run_once(args.alive, now=time.time(), threshold=args.threshold)
    return 0 if not killed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
