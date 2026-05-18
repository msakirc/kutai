"""Z10 T3C — mark_green + rollback_mission."""
from __future__ import annotations

import gzip
import json
import os
import subprocess
from pathlib import Path

import aiosqlite
import pytest


async def _init_db(tmp_path, monkeypatch):
    db_path = tmp_path / "green.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    from src.infra import db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(db_path), raising=False)
    await db_mod.init_db()
    return db_mod


def _init_git_repo(repo_path: Path) -> str:
    repo_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=repo_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo_path, check=True)
    f = repo_path / "a.txt"
    f.write_text("first\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo_path, check=True)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return sha


@pytest.mark.asyncio
async def test_mark_green_creates_tag_and_snapshots(tmp_path, monkeypatch):
    db_mod = await _init_db(tmp_path, monkeypatch)

    repo = tmp_path / "workspace"
    _init_git_repo(repo)
    monkeypatch.chdir(repo)

    # Insert a mission row and a task row so snapshot has content.
    db = await db_mod.get_db()
    await db.execute(
        "INSERT INTO missions (id, title, status) VALUES (?, ?, ?)",
        (777, "test mission", "active"),
    )
    await db.commit()
    await db_mod.add_task(
        title="green task",
        description="x",
        mission_id=777,
    )

    # Patch project_root + workspace resolver to the tmp repo.
    from mr_roboto import mark_green as mg
    monkeypatch.setattr(mg, "_project_root", lambda: str(tmp_path))

    # Stub chroma snapshot (vector store not initialized in tests).
    async def _no_chroma(*a, **kw):
        return []
    monkeypatch.setattr(
        "src.memory.vector_store.list_mission_chroma_collections", _no_chroma
    )

    res = await mg.run(
        mission_id=777,
        task_id=1,
        summary="test green",
        repo_path=str(repo),
    )
    assert res["mission_id"] == 777
    assert os.path.exists(res["db_snapshot_path"])
    assert os.path.exists(res["chroma_snapshot_path"])
    # Tag exists in the repo.
    code = subprocess.run(
        ["git", "rev-parse", res["git_tag"]], cwd=repo,
        capture_output=True, text=True,
    ).returncode
    assert code == 0

    # Ledger row was inserted.
    async with aiosqlite.connect(db_mod.DB_PATH) as db2:
        cur = await db2.execute(
            "SELECT mission_id, task_id, git_tag FROM mission_green_tags "
            "WHERE mission_id = 777"
        )
        rows = await cur.fetchall()
    assert len(rows) == 1
    assert rows[0][2] == "green-777-1"


@pytest.mark.asyncio
async def test_mark_green_idempotent(tmp_path, monkeypatch):
    db_mod = await _init_db(tmp_path, monkeypatch)
    repo = tmp_path / "ws"
    _init_git_repo(repo)
    monkeypatch.chdir(repo)
    db = await db_mod.get_db()
    await db.execute(
        "INSERT INTO missions (id, title, status) VALUES (?, ?, ?)",
        (888, "m", "active"),
    )
    await db.commit()

    from mr_roboto import mark_green as mg
    monkeypatch.setattr(mg, "_project_root", lambda: str(tmp_path))

    async def _no_chroma(*a, **kw):
        return []
    monkeypatch.setattr(
        "src.memory.vector_store.list_mission_chroma_collections", _no_chroma
    )

    r1 = await mg.run(mission_id=888, task_id=2, repo_path=str(repo))
    r2 = await mg.run(mission_id=888, task_id=2, repo_path=str(repo))
    assert r1["id"] == r2["id"]


@pytest.mark.asyncio
async def test_rollback_restores_db_rows(tmp_path, monkeypatch):
    db_mod = await _init_db(tmp_path, monkeypatch)
    repo = tmp_path / "ws"
    _init_git_repo(repo)
    monkeypatch.chdir(repo)

    db = await db_mod.get_db()
    await db.execute(
        "INSERT INTO missions (id, title, status) VALUES (?, ?, ?)",
        (555, "rb", "active"),
    )
    await db.commit()
    await db_mod.add_task(title="t1", description="x", mission_id=555)
    await db_mod.add_task(title="t2", description="y", mission_id=555)

    from mr_roboto import mark_green as mg
    from mr_roboto import rollback_mission as rb_mod
    monkeypatch.setattr(mg, "_project_root", lambda: str(tmp_path))

    async def _no_chroma_list(*a, **kw):
        return []
    monkeypatch.setattr(
        "src.memory.vector_store.list_mission_chroma_collections",
        _no_chroma_list,
    )
    async def _no_purge(*a, **kw):
        return 0
    monkeypatch.setattr(
        "src.memory.vector_store.purge_mission_chroma_collections",
        _no_purge,
    )

    green = await mg.run(mission_id=555, task_id=10, repo_path=str(repo))

    # Add a third task after the green tag.
    await db_mod.add_task(title="t3", description="z", mission_id=555)

    async with aiosqlite.connect(db_mod.DB_PATH) as d:
        cur = await d.execute(
            "SELECT COUNT(*) FROM tasks WHERE mission_id = 555"
        )
        n_before = (await cur.fetchone())[0]
    assert n_before == 3

    res = await rb_mod.run(
        mission_id=555,
        target_task_id=10,
        repo_path=str(repo),
    )
    assert res["ok"] is True or res["db"]["error"] is None

    async with aiosqlite.connect(db_mod.DB_PATH) as d:
        cur = await d.execute(
            "SELECT COUNT(*) FROM tasks WHERE mission_id = 555"
        )
        n_after = (await cur.fetchone())[0]
    # Restored to green: 2 tasks.
    assert n_after == 2


@pytest.mark.asyncio
async def test_rollback_schema_rewind(tmp_path, monkeypatch):
    """Synthesize a migration after the green tag → rollback rewinds it."""
    db_mod = await _init_db(tmp_path, monkeypatch)
    repo = tmp_path / "ws"
    _init_git_repo(repo)
    monkeypatch.chdir(repo)

    db = await db_mod.get_db()
    await db.execute(
        "INSERT INTO missions (id, title, status) VALUES (?, ?, ?)",
        (333, "rw", "active"),
    )
    await db.commit()

    from mr_roboto import mark_green as mg
    from mr_roboto import rollback_mission as rb_mod
    monkeypatch.setattr(mg, "_project_root", lambda: str(tmp_path))

    async def _no_chroma_list(*a, **kw):
        return []
    monkeypatch.setattr(
        "src.memory.vector_store.list_mission_chroma_collections",
        _no_chroma_list,
    )
    async def _no_purge(*a, **kw):
        return 0
    monkeypatch.setattr(
        "src.memory.vector_store.purge_mission_chroma_collections",
        _no_purge,
    )

    await mg.run(mission_id=333, task_id=20, repo_path=str(repo))

    # Apply a NEW migration after the green tag.
    await db_mod.apply_migration(
        version="t3c-rewind-test",
        sql="ALTER TABLE missions ADD COLUMN test_rewind_col TEXT",
        reversal_sql="ALTER TABLE missions DROP COLUMN test_rewind_col",
        description="should be rewound",
    )

    async with aiosqlite.connect(db_mod.DB_PATH) as d:
        cur = await d.execute("PRAGMA table_info(missions)")
        cols = [r[1] for r in await cur.fetchall()]
    assert "test_rewind_col" in cols

    res = await rb_mod.run(
        mission_id=333, target_task_id=20, repo_path=str(repo)
    )
    assert "t3c-rewind-test" in (res["schema_rewind"]["rewound"] or [])

    async with aiosqlite.connect(db_mod.DB_PATH) as d:
        cur = await d.execute("PRAGMA table_info(missions)")
        cols = [r[1] for r in await cur.fetchall()]
    assert "test_rewind_col" not in cols
