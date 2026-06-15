"""Z10 T3C — ``mark_green`` mechanical verb.

Captures a paired green checkpoint for a mission:

  1. Annotated git tag ``green-{mission_id}-{task_id}`` in the workspace.
  2. JSON-gzipped snapshot of mission-scoped DB rows at
     ``data/snapshots/mission-{mission_id}-{task_id}/db.json.gz``.
  3. Chroma collection snapshot (each ``mission_{mission_id}__*`` collection
     dumped via ``col.get(include=[...])``) at
     ``data/snapshots/mission-{mission_id}-{task_id}/chroma.json.gz``.
  4. Ledger row in ``mission_green_tags`` linking all three.

Trigger contract
----------------
Triggered explicitly via ``payload.mark_green=True`` on milestone gate
steps; founder + reviewer cosign + tests-pass moments only. Auto-firing
every commit is too costly — each green tag is hundreds of KB to MB on
disk (Chroma + DB JSON-gz snapshot per fire).

Currently wired in src/workflows/i2p/i2p_v3.json (z10-wire-fixes F2, cap
≤6) on: 7.5.git_commit (frontend scaffold sealed), 8.spike.git_commit
(spike validated), 4.16.git_commit_green (architecture_review approved),
13.14.git_commit_green (launch_go_no_go approved). Each is a paired green
checkpoint (git tag + DB rows + Chroma collections + ledger row).

Reversibility: ``full`` — pure snapshot, no destructive op.
"""
from __future__ import annotations

import gzip
import json
import os
import time
from typing import Any

from src.infra.logging_config import get_logger

logger = get_logger("mr_roboto.mark_green")


def _project_root() -> str:
    here = os.path.abspath(__file__)
    # packages/mr_roboto/src/mr_roboto/mark_green.py → up 4 to repo root
    return os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))


def _snapshot_dir(mission_id: int, task_id: int) -> str:
    return os.path.join(
        _project_root(),
        "data",
        "snapshots",
        f"mission-{int(mission_id)}-{int(task_id)}",
    )


async def _capture_git_tag(
    mission_id: int, task_id: int, repo_path: str, summary: str
) -> str | None:
    """Create annotated tag ``green-{mid}-{tid}`` at HEAD. Return tag name."""
    from src.tools.git_ops import _resolve_repo, _run_git, ensure_git_repo
    try:
        await ensure_git_repo(repo_path)
    except Exception as e:
        logger.warning(f"mark_green: ensure_git_repo failed: {e}")
    target = _resolve_repo(repo_path) or repo_path or _project_root()
    tag = f"green-{int(mission_id)}-{int(task_id)}"
    try:
        # Force overwrite a pre-existing tag at the same coords (idempotent
        # re-runs land on the latest HEAD).
        await _run_git(
            ["tag", "-fa", tag, "-m", (summary or "green checkpoint")[:200]],
            cwd=target,
        )
        return tag
    except Exception as e:
        logger.warning(f"mark_green: git tag {tag} failed: {e}")
        return None


async def _snapshot_db(mission_id: int, dst: str) -> str:
    """Dump mission-scoped rows to ``dst/db.json.gz``. Returns path."""
    from dabidabi import snapshot_mission_db_rows
    rows = await snapshot_mission_db_rows(mission_id)
    os.makedirs(dst, exist_ok=True)
    path = os.path.join(dst, "db.json.gz")

    def _write():
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(rows, f, default=str)

    import asyncio
    await asyncio.to_thread(_write)
    return path


async def _snapshot_chroma(mission_id: int, dst: str) -> str:
    """Dump every ``mission_{id}__*`` Chroma collection to ``dst/chroma.json.gz``."""
    os.makedirs(dst, exist_ok=True)
    path = os.path.join(dst, "chroma.json.gz")
    payload: dict[str, Any] = {"collections": {}}

    try:
        from src.memory.vector_store import (
            list_mission_chroma_collections,
            _get_or_create_namespaced_collection,
        )
        names = await list_mission_chroma_collections(int(mission_id))
        for name in names:
            col = await _get_or_create_namespaced_collection(name)
            if col is None:
                continue
            try:
                import asyncio
                got = await asyncio.to_thread(
                    col.get,
                    include=["documents", "embeddings", "metadatas"],
                )
                payload["collections"][name] = {
                    "ids": got.get("ids") or [],
                    "documents": got.get("documents") or [],
                    "metadatas": got.get("metadatas") or [],
                    "embeddings": got.get("embeddings") or [],
                    "metadata": dict(getattr(col, "metadata", {}) or {}),
                }
            except Exception as e:
                logger.warning(
                    f"mark_green: chroma snapshot of '{name}' failed: {e}"
                )
    except Exception as e:
        logger.warning(f"mark_green: chroma snapshot failed: {e}")

    def _write():
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(payload, f, default=str)

    import asyncio
    await asyncio.to_thread(_write)
    return path


async def run(
    mission_id: int,
    task_id: int,
    *,
    summary: str = "",
    repo_path: str | None = None,
) -> dict:
    """Capture a green checkpoint. Returns dict with ledger row id + paths."""
    t0 = time.time()
    mission_id = int(mission_id)
    task_id = int(task_id)

    if repo_path is None:
        try:
            from src.tools.workspace import get_mission_workspace_relative
            repo_path = get_mission_workspace_relative(mission_id) or ""
        except Exception:
            repo_path = ""

    dst = _snapshot_dir(mission_id, task_id)

    # 1. git tag
    tag = await _capture_git_tag(mission_id, task_id, repo_path or "", summary)

    # 2. DB snapshot
    db_path = await _snapshot_db(mission_id, dst)

    # 3. Chroma snapshot
    chroma_path = await _snapshot_chroma(mission_id, dst)

    # 4. Schema migration high-water mark — already inside the DB snapshot
    # under `_meta`, but record it on the ledger row too so rollback can
    # branch on it without re-reading the gz.
    schema_v: str | None = None
    try:
        def _read_meta():
            with gzip.open(db_path, "rt", encoding="utf-8") as f:
                return json.load(f)
        import asyncio
        snap = await asyncio.to_thread(_read_meta)
        meta = (snap or {}).get("_meta") or {}
        schema_v = meta.get("schema_migrations_at")
    except Exception:
        pass

    # 5. Ledger row
    from dabidabi import record_green_tag
    rowid = await record_green_tag(
        mission_id=mission_id,
        task_id=task_id,
        git_tag=tag or f"green-{mission_id}-{task_id}",
        db_snapshot_path=db_path,
        chroma_snapshot_path=chroma_path,
        schema_migrations_at=schema_v,
    )

    return {
        "id": rowid,
        "mission_id": mission_id,
        "task_id": task_id,
        "git_tag": tag,
        "db_snapshot_path": db_path,
        "chroma_snapshot_path": chroma_path,
        "schema_migrations_at": schema_v,
        "elapsed_s": time.time() - t0,
    }


__all__ = ["run"]
