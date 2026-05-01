"""Standalone entry point for NerdHerd sidecar.

Usage:
    python -m nerd_herd --port 9881 --llama-url http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import asyncio
import os
import platform
import signal
import sys

# Holds strong references to fire-and-forget mode-persistence tasks so GC
# can't reap them mid-flight. Tasks remove themselves via done-callback.
_pending_mode_tasks: set[asyncio.Task] = set()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="nerd_herd",
        description="NerdHerd GPU observability sidecar",
    )
    parser.add_argument("--port", type=int, default=9881, help="Metrics HTTP port (default: 9881)")
    parser.add_argument(
        "--llama-url",
        default="http://127.0.0.1:8080",
        help="llama-server base URL (default: http://127.0.0.1:8080)",
    )
    parser.add_argument("--pid-file", default=None, help="Write PID to this file after startup")
    parser.add_argument(
        "--db-path",
        default=None,
        help="SQLite DB for load mode persistence",
    )
    parser.add_argument(
        "--detect-interval",
        type=int,
        default=30,
        help="GPU auto-detect poll interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--upgrade-delay",
        type=int,
        default=300,
        help="Seconds before upgrading load mode after conditions improve (default: 300)",
    )
    return parser.parse_args()


async def _apply_pragmas(db) -> None:
    """Apply WAL + 60s busy_timeout to coexist with other writers on kutai.db.

    Inlined here (not imported from src.infra.db) so this subprocess stays
    package-self-contained.
    """
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA synchronous=NORMAL")
    await db.execute("PRAGMA busy_timeout=60000")


async def _load_mode_from_db(db_path: str) -> str:
    """Read current load mode from the load_mode table. Returns 'full' on any error."""
    try:
        import aiosqlite

        async with aiosqlite.connect(db_path) as db:
            await _apply_pragmas(db)
            async with db.execute(
                "SELECT mode FROM load_mode WHERE id = 1"
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return row[0]
    except Exception:
        pass
    return "full"


async def _persist_mode(db_path: str, mode: str, auto_managed: bool) -> None:
    """INSERT OR REPLACE load mode into the load_mode table."""
    try:
        import aiosqlite

        async with aiosqlite.connect(db_path) as db:
            await _apply_pragmas(db)
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS load_mode (
                    id INTEGER PRIMARY KEY,
                    mode TEXT NOT NULL,
                    auto_managed INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            await db.execute(
                """
                INSERT OR REPLACE INTO load_mode (id, mode, auto_managed, updated_at)
                VALUES (1, ?, ?, datetime('now'))
                """,
                (mode, int(auto_managed)),
            )
            await db.commit()
    except Exception:
        pass


async def _main() -> None:
    args = _parse_args()

    # Allow the project root to be injected so imports resolve when running
    # outside the installed package (e.g. during development).
    project_root = os.environ.get("NERD_HERD_PROJECT_ROOT")
    if project_root and project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Import here so sys.path manipulation above takes effect first.
    from nerd_herd.nerd_herd import NerdHerd  # noqa: PLC0415

    # --- load initial mode ---
    initial_mode = "full"
    if args.db_path:
        initial_mode = await _load_mode_from_db(args.db_path)

    # --- build NerdHerd instance ---
    nh = NerdHerd(
        metrics_port=args.port,
        llama_server_url=args.llama_url,
        detect_interval=args.detect_interval,
        upgrade_delay=args.upgrade_delay,
        initial_load_mode=initial_mode,
    )

    # --- wire mode-change persistence ---
    if args.db_path:
        db_path = args.db_path

        def _on_mode_change(prev: str, mode: str, source: str) -> None:  # noqa: ARG001
            auto_managed = source != "user"
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            task = loop.create_task(_persist_mode(db_path, mode, auto_managed))
            _pending_mode_tasks.add(task)
            task.add_done_callback(_pending_mode_tasks.discard)

        nh.on_mode_change(_on_mode_change)

    # --- start services ---
    await nh.start()
    await nh.start_auto_detect()

    # --- write PID file ---
    if args.pid_file:
        try:
            with open(args.pid_file, "w") as f:
                f.write(str(os.getpid()))
        except OSError as exc:
            print(f"[nerd_herd] WARNING: could not write PID file: {exc}", file=sys.stderr)

    print(
        f"[nerd_herd] started — metrics on :{args.port}  llama-url={args.llama_url}  "
        f"mode={nh.get_load_mode()}  detect-interval={args.detect_interval}s",
        flush=True,
    )

    # --- signal handling ---
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    if platform.system() == "Windows":
        # Windows: signal.signal (cannot use loop.add_signal_handler on Windows)
        def _win_handler(signum, frame):  # noqa: ARG001
            loop.call_soon_threadsafe(stop_event.set)

        signal.signal(signal.SIGINT, _win_handler)
        signal.signal(signal.SIGTERM, _win_handler)
    else:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)

    # --- wait for shutdown ---
    await stop_event.wait()

    print("[nerd_herd] shutting down…", flush=True)
    await nh.stop()

    # --- clean up PID file ---
    if args.pid_file:
        try:
            os.unlink(args.pid_file)
        except OSError:
            pass

    print("[nerd_herd] stopped.", flush=True)


if __name__ == "__main__":
    asyncio.run(_main())
