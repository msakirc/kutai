"""Z10 T3C — ``rollback_mission`` mechanical verb.

Restores a mission to a previously-recorded green checkpoint:

  1. ``git checkout`` the recorded annotated tag in the workspace.
  2. Restore mission-scoped DB rows (DELETE current + INSERT from snapshot).
  3. Restore Chroma collections: drop current ``mission_{id}__*`` collections
     and rebuild from the gz snapshot.
  4. Best-effort schema-rewind: if the green tag was captured before some
     migration applied since, walk ``schema_migrations`` rows newer than the
     green-version IN REVERSE and apply their ``reversal_sql``. Rows with
     NULL ``reversal_sql`` are skipped + logged.

Reversibility: ``irreversible`` — rolling forward isn't a primitive. T1B
registers this in ``VERB_REVERSIBILITY``; T1C confirmation flow auto-gates.

Idempotency: calling twice with the same target is safe — the second run
restores to the same green and is effectively a no-op.
"""
from __future__ import annotations

import asyncio
import gzip
import json
import logging
import os

logger = logging.getLogger("mr_roboto.rollback_mission")


async def _git_checkout_tag(repo_path: str, tag: str) -> tuple[bool, str]:
    """Best-effort git checkout of ``tag``. Returns (ok, err)."""
    try:
        from src.tools.git_ops import _resolve_repo, _run_git
    except Exception as e:
        return False, f"git_ops import failed: {e}"
    target = _resolve_repo(repo_path) or repo_path or os.getcwd()
    try:
        code, _out, err = await _run_git(["checkout", tag], cwd=target)
        return (code == 0), (err or "")
    except Exception as e:
        return False, str(e)


async def _restore_chroma_from_snapshot(
    mission_id: int, chroma_snapshot_path: str
) -> dict:
    """Drop mission_NN__* collections and rebuild from the snapshot gz."""
    out = {"restored": [], "failed": []}
    try:
        from src.memory.vector_store import (
            purge_mission_chroma_collections,
            _get_or_create_namespaced_collection,
        )
    except Exception as e:
        logger.warning(f"rollback: vector_store import failed: {e}")
        return out

    # 1. Wipe current per-mission collections.
    try:
        await purge_mission_chroma_collections(int(mission_id))
    except Exception as e:
        logger.warning(f"rollback: purge failed: {e}")

    # 2. Reload from snapshot.
    if not os.path.exists(chroma_snapshot_path):
        # Snapshot file was deleted off-disk; we already purged so the
        # mission's collections are simply empty. Not a failure.
        return out

    def _read():
        with gzip.open(chroma_snapshot_path, "rt", encoding="utf-8") as f:
            return json.load(f)

    try:
        payload = await asyncio.to_thread(_read)
    except Exception as e:
        logger.warning(f"rollback: snapshot read failed: {e}")
        return out

    collections = (payload or {}).get("collections") or {}
    for name, blob in collections.items():
        col = await _get_or_create_namespaced_collection(name)
        if col is None:
            out["failed"].append(name)
            continue
        ids = blob.get("ids") or []
        if not ids:
            out["restored"].append(name)
            continue
        try:
            await asyncio.to_thread(
                col.upsert,
                ids=ids,
                documents=blob.get("documents") or [None] * len(ids),
                metadatas=blob.get("metadatas") or [{}] * len(ids),
                embeddings=blob.get("embeddings") or [None] * len(ids),
            )
            out["restored"].append(name)
        except Exception as e:
            logger.warning(f"rollback: chroma upsert {name} failed: {e}")
            out["failed"].append(name)
    return out


async def run(
    mission_id: int,
    target_task_id: int | None = None,
    *,
    repo_path: str | None = None,
) -> dict:
    """Roll mission ``mission_id`` back to a green-tag snapshot.

    Returns a dict with the resolved tag/paths and per-step counts.
    """
    from dabidabi import (
        get_latest_green_tag,
        restore_mission_db_rows,
        rewind_migrations_to,
    )

    mission_id = int(mission_id)
    ledger = await get_latest_green_tag(
        mission_id, int(target_task_id) if target_task_id is not None else None
    )
    if ledger is None:
        return {
            "ok": False,
            "error": "no green tag recorded for mission",
            "mission_id": mission_id,
        }

    # Resolve workspace path lazily.
    if repo_path is None:
        try:
            from src.tools.workspace import get_mission_workspace_relative
            repo_path = get_mission_workspace_relative(mission_id) or ""
        except Exception:
            repo_path = ""

    # 1. git checkout the tag.
    git_ok, git_err = await _git_checkout_tag(repo_path or "", ledger["git_tag"])

    # 2. Schema rewind FIRST so DELETE/INSERT runs against the matching shape.
    rewind_result = await rewind_migrations_to(
        ledger.get("schema_migrations_at")
    )

    # 3. Restore mission-scoped DB rows.
    db_path = ledger["db_snapshot_path"]
    db_counts: dict = {}
    db_err: str | None = None
    if not os.path.exists(db_path):
        db_err = f"db snapshot missing: {db_path}"
    else:
        def _read():
            with gzip.open(db_path, "rt", encoding="utf-8") as f:
                return json.load(f)
        try:
            snap = await asyncio.to_thread(_read)
            db_counts = await restore_mission_db_rows(mission_id, snap)
        except Exception as e:
            db_err = str(e)

    # 4. Restore Chroma.
    chroma_result = await _restore_chroma_from_snapshot(
        mission_id, ledger["chroma_snapshot_path"]
    )

    ok = git_ok and (db_err is None) and not chroma_result["failed"]
    return {
        "ok": ok,
        "mission_id": mission_id,
        "ledger": ledger,
        "git": {"ok": git_ok, "error": git_err},
        "db": {"counts": db_counts, "error": db_err},
        "chroma": chroma_result,
        "schema_rewind": rewind_result,
    }


__all__ = ["run"]
