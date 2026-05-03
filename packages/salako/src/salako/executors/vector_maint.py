"""Vector store maintenance executors.

Two cron-seeded mechanical tasks:
- ``vector_maint_wal``      — ChromaDB WAL checkpoint (every 30 min)
- ``vector_maint_snapshot`` — ChromaDB directory snapshot (every 24h)

Both operations are synchronous I/O and are wrapped in
``asyncio.get_event_loop().run_in_executor(None, ...)`` so the orchestrator
pump's event loop stays responsive. This was the root cause of the 120s wedge
in mission 46 — the old _vector_maint_loop called these on the event loop.
"""
from __future__ import annotations

from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("salako.vector_maint")


async def run_wal(task: dict[str, Any]) -> dict[str, Any]:
    """Run ChromaDB WAL checkpoint.

    Delegates to src.memory.vector_store.wal_checkpoint which already wraps
    the sync sqlite3 call in asyncio.to_thread. Catches all errors — a
    ChromaDB hiccup must not DLQ-cascade and freeze maintenance.
    """
    try:
        from src.memory.vector_store import wal_checkpoint
        ok = await wal_checkpoint()
        logger.info("vector_maint_wal: WAL checkpoint complete", ok=ok)
        return {"ok": ok}
    except Exception as exc:
        logger.warning("vector_maint_wal: WAL checkpoint failed", error=str(exc))
        return {"ok": False, "error": str(exc)}


async def run_snapshot(task: dict[str, Any]) -> dict[str, Any]:
    """Run ChromaDB directory snapshot.

    Delegates to src.memory.vector_store.snapshot_chroma which already wraps
    the sync shutil.copytree in asyncio.to_thread. Catches all errors.
    """
    try:
        from src.memory.vector_store import snapshot_chroma
        dst = await snapshot_chroma(keep=3)
        if dst:
            logger.info("vector_maint_snapshot: snapshot taken", path=dst)
        else:
            logger.debug("vector_maint_snapshot: snapshot returned None (chroma dir absent?)")
        return {"dst": dst}
    except Exception as exc:
        logger.warning("vector_maint_snapshot: snapshot failed", error=str(exc))
        return {"dst": None, "error": str(exc)}
