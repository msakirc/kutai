"""Smart-RAG Phase 2 — ChromaDB on-disk loadability guardrail (P4).

The ChromaDB store can grow unbounded on disk in two ways that a static
per-collection row cap does not cover:

1. **SQLite bloat** from deleted rows — reclaimed by ``chroma vacuum``.
2. **Orphaned HNSW segment directories** — ``delete_collection`` removes the
   ``segments`` table rows but leaves the segment's on-disk directory behind,
   and ``chroma vacuum`` does NOT touch it. Those dirs leak *permanently*
   (proven 2026-07-27 on a copy of the prod store: deleting a 22.4k-row
   collection + vacuum reclaimed 37.6 MB of SQLite but left the 71.8 MB HNSW
   dir byte-identical; a fresh client open did not GC it either).

This module is the missing half: a deterministic, fail-safe **orphan segment
GC** plus a bytes-budget boot gate. It touches the store only via read-only
SQLite reads and ``shutil.rmtree`` of directories that have **no row in the
``segments`` table**. It is meant to run at BOOT, before the ChromaDB client
opens, when the store is quiescent.

Safety invariants:
  * If the ``segments`` table cannot be read → treat as "unknown" → reclaim
    NOTHING. Never infer that every dir is an orphan.
  * Only UUID-shaped subdirectories are ever considered (chroma segment dirs).
  * A ``min_age_seconds`` guard skips freshly-created dirs so an in-flight
    segment whose ``segments`` row is not yet committed is never deleted.
"""
from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import sqlite3
import sys
import time

logger = logging.getLogger("memory.chroma_maintenance")

# ChromaDB segment directories are named with a v4 UUID.
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

# Default on-disk budget for the ChromaDB store. The whale that starved cold
# boot in 2026-07 was 9.6 GB; a healthy store is ~250 MB. 2 GB leaves ample
# headroom while catching runaway growth long before it starves boot.
DEFAULT_BUDGET_BYTES = 2_000_000_000

# Orphan dirs younger than this are skipped (defends against deleting an
# in-flight segment whose ``segments`` row is not yet committed).
DEFAULT_MIN_AGE_SECONDS = 3600


def store_size_bytes(path: str) -> int:
    """Total bytes of every file under ``path`` (recursive). Missing → 0."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                # File vanished mid-walk (concurrent compaction) — skip.
                continue
    return total


def list_on_disk_segment_dirs(db_dir: str) -> set[str]:
    """UUID-named subdirectories of ``db_dir`` (candidate segment dirs)."""
    out: set[str] = set()
    try:
        entries = os.listdir(db_dir)
    except OSError:
        return out
    for name in entries:
        if _UUID_RE.match(name) and os.path.isdir(os.path.join(db_dir, name)):
            out.add(name)
    return out


def list_segment_ids(db_dir: str) -> set[str]:
    """Segment ids from ``chroma.sqlite3``'s ``segments`` table.

    Raises if the DB / table cannot be read — callers MUST treat a raise as
    "cannot determine orphans" and reclaim nothing.
    """
    path = os.path.join(db_dir, "chroma.sqlite3")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # Read-only URI open: this module's contract is a read-only touch of the
    # store. ro mode never mutates rows and won't contend a live opener with a
    # write lock. If the schema differs across chroma versions (renamed table/
    # column) the SELECT raises here → the caller's fail-safe reclaims nothing.
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10.0)
    try:
        rows = conn.execute("SELECT id FROM segments").fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows}


def find_orphan_segment_dirs(
    db_dir: str,
    min_age_seconds: float = DEFAULT_MIN_AGE_SECONDS,
    now: float | None = None,
) -> list[str]:
    """On-disk segment dirs with no row in the ``segments`` table.

    Fail-safe: any failure to read the ``segments`` table returns ``[]``.
    """
    on_disk = list_on_disk_segment_dirs(db_dir)
    if not on_disk:
        return []
    try:
        seg_ids = list_segment_ids(db_dir)
    except Exception as e:  # noqa: BLE001 — fail-safe: never guess orphans
        logger.warning(
            "orphan GC: cannot read segments table (%s) — reclaiming nothing",
            e,
        )
        return []

    now = time.time() if now is None else now
    orphans: list[str] = []
    for name in on_disk:
        if name in seg_ids:
            continue
        p = os.path.join(db_dir, name)
        if min_age_seconds > 0:
            try:
                age = now - os.path.getmtime(p)
            except OSError:
                continue
            if age < min_age_seconds:
                logger.info(
                    "orphan GC: %s unregistered but young (%.0fs<%.0fs) — skip",
                    name, age, min_age_seconds,
                )
                continue
        orphans.append(name)
    return sorted(orphans)


def reclaim_orphan_segment_dirs(
    db_dir: str,
    min_age_seconds: float = DEFAULT_MIN_AGE_SECONDS,
    dry_run: bool = False,
    now: float | None = None,
) -> dict:
    """Delete orphaned segment dirs. Returns ``{removed, bytes_freed}``."""
    orphans = find_orphan_segment_dirs(db_dir, min_age_seconds, now)
    removed: list[str] = []
    freed = 0
    for name in orphans:
        p = os.path.join(db_dir, name)
        size = store_size_bytes(p)
        if dry_run:
            removed.append(name)
            freed += size
            continue
        try:
            shutil.rmtree(p)
            removed.append(name)
            freed += size
            logger.info("orphan GC: removed %s (%d bytes)", name, size)
        except OSError as e:
            logger.warning("orphan GC: rmtree(%s) failed: %s", name, e)
    return {"removed": removed, "bytes_freed": freed}


@contextlib.contextmanager
def _destructive_lock(db_dir: str):
    """Non-blocking cross-process exclusive lock for destructive maintenance.

    Yields ``True`` if this process acquired sole ownership, else ``False``.
    The store's own ``_init_lock`` is an in-process ``asyncio.Lock`` and does
    NOT serialise the two Python processes that can load the vector store
    concurrently (orchestrator + a parallel session). This file lock ensures
    ``chroma vacuum`` (which rewrites chroma.sqlite3) and orphan ``rmtree``
    only run when no sibling process is touching the store; otherwise the
    caller skips (fail-open) and retries on a later boot.
    """
    lock_path = os.path.join(db_dir, ".maintenance.lock")
    fd = None
    acquired = False
    try:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
            os.write(fd, b"0")  # ensure ≥1 byte so byte-range locking works
            os.lseek(fd, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError:
            acquired = False
        yield acquired
    finally:
        if fd is not None:
            if acquired:
                try:
                    if os.name == "nt":
                        import msvcrt
                        os.lseek(fd, 0, os.SEEK_SET)
                        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(fd)


def _chroma_exe() -> str:
    """Locate the ``chroma`` CLI shipped alongside the running interpreter."""
    scripts = os.path.dirname(sys.executable)
    for cand in ("chroma.exe", "chroma"):
        p = os.path.join(scripts, cand)
        if os.path.exists(p):
            return p
    return "chroma"  # fall back to PATH


def vacuum_store(db_dir: str, runner=None, timeout: int = 300) -> bool:
    """Run ``chroma vacuum`` on ``db_dir``. Reclaims SQLite bytes of deleted
    rows (NOT orphaned HNSW dirs — use ``reclaim_orphan_segment_dirs`` for
    those). ``runner`` is injectable for tests: ``callable(args) -> int``.

    Must run with the store CLOSED (no live client). Returns True on rc 0.
    """
    args = [
        _chroma_exe(), "vacuum",
        "--path", db_dir,
        "--force",
        "--timeout", str(timeout),
    ]
    if runner is None:
        import subprocess

        def runner(a):
            return subprocess.run(
                a, capture_output=True, timeout=timeout + 30
            ).returncode

    try:
        rc = runner(args)
    except Exception as e:  # noqa: BLE001 — vacuum is best-effort
        logger.warning("vacuum failed: %s", e)
        return False
    if rc != 0:
        logger.warning("vacuum returned non-zero exit %s", rc)
        return False
    return True


def enforce_size_budget(
    db_dir: str,
    budget_bytes: int = DEFAULT_BUDGET_BYTES,
    *,
    size_fn=None,
    reclaim_fn=None,
    vacuum_fn=None,
    do_vacuum: bool = True,
    min_age_seconds: float = DEFAULT_MIN_AGE_SECONDS,
) -> dict:
    """Boot-gate decision: measure the store; if over budget, reclaim orphaned
    segment dirs then vacuum. All I/O deps are injectable for testing.

    Returns a report dict. Runs with the store closed (call before the client
    opens). Cheap when under budget (a single ``os.walk``).
    """
    size_fn = size_fn or store_size_bytes
    reclaim_fn = reclaim_fn or reclaim_orphan_segment_dirs
    vacuum_fn = vacuum_fn or vacuum_store

    size = size_fn(db_dir)
    report = {
        "size_bytes": size,
        "budget_bytes": budget_bytes,
        "over_budget": size > budget_bytes,
        "action": "none",
    }
    if size <= budget_bytes:
        return report

    # Over budget → destructive reclaim. Serialise across processes: only the
    # sole owner may vacuum/rmtree. If a sibling holds the lock, skip and let a
    # later boot reclaim (fail-open — never block, never race).
    with _destructive_lock(db_dir) as locked:
        if not locked:
            report["action"] = "skipped_locked"
            logger.warning(
                "Chroma over budget (%d > %d) but maintenance lock held by "
                "another process — skipping reclaim this boot", size, budget_bytes,
            )
            return report
        report["action"] = "reclaim"
        report["reclaim"] = reclaim_fn(db_dir, min_age_seconds=min_age_seconds)
        if do_vacuum:
            report["vacuumed"] = vacuum_fn(db_dir)
        # Re-walk (not size-minus-freed): captures vacuum's SQLite reclaim too,
        # which the orphan-GC's freed byte-count does not.
        report["size_after"] = size_fn(db_dir)
    return report
